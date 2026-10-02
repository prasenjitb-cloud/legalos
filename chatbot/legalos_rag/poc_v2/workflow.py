"""A bounded controller that fills an evidence table before answering."""

import copy
from pathlib import Path
import re
import time

from .corpus import ACT_ID, EMBEDDING_MODEL, EMBEDDING_REVISION, QUERY_INSTRUCTION, RERANKER_MODEL, RERANKER_REVISION, digest, section_references
from .model import ActPlan, Plan, Selection, EvidenceMap, PLAN_PROMPT, EVIDENCE_PROMPT, validate_map
from .facts import worker_count, required_user_facts, worker_condition_units


def question_parts(question):
    boundary = r"(?:,\s*(?:and\s+)?|\s+and\s+)(?=(?:what|where|who|how|when|can|does|is|are)\b)"
    parts, start = [], 0
    for match in re.finditer(boundary, question, re.I):
        prefix = question[start:match.start()]
        if re.search(r"\b(?:what|where|who|how|when|can|does|is|are)\b", prefix, re.I):
            parts.append(prefix.strip())
            start = match.end()
    parts.append(question[start:].strip())
    return parts if len(parts) <= 4 else [question]


def small_workplace_check(question, unit_ids):
    legal_unit = next((u for u in ["6(1)", "2(p)"] if u in unit_ids), None)
    if not legal_unit:
        return None
    fact = worker_count(question)
    if fact["status"] != "KNOWN":
        return None
    value = fact["value"]
    return {"condition": "worker count less than ten", "status": "MET" if value < 10 else "NOT_MET",
            "value": value, "source_quote": fact["source_quotes"][0], "legal_unit": legal_unit,
            "limit": "This compares the numeric condition only; the complete legal qualification is quoted."}


def context_dependencies(unit, section_units, *, complete_section=False):
    section = unit["section_id"]
    if complete_section or any(re.search(r"\bunder this section\b", u["text"], re.I) for u in section_units):
        return {u["id"] for u in section_units}
    related = {f"{section}({number.upper()})" for number in re.findall(
        r"under sub\s*-\s*section\s*\(\s*(\d+[A-Z]?)\s*\)\s+(?:shall|may)\b", unit["text"], re.I)}
    for candidate in section_units:
        text = candidate["text"]
        if re.search(r"for the purposes of this section|provisions of this section shall be applicable", text, re.I):
            related.add(candidate["id"])
        if re.search(r"\bnotwithstanding\b", text, re.I):
            references = {f"{section}({number.upper()})" for number in re.findall(
                r"sub\s*-\s*sections?\s*\(\s*(\d+[A-Z]?)\s*\)", text, re.I)}
            if unit["id"] in references:
                related.add(candidate["id"])
    return related & {u["id"] for u in section_units}


class Controller:
    def __init__(self, store, model, trace, *, first_omit=(), unavailable=()):
        self.store, self.model, self.trace = store, model, trace
        self.first_omit, self.unavailable = set(first_omit), set(unavailable)
        self.tool_calls = 0
        self.trace.event("configuration", {
            "model": model.name, "model_settings": model.settings,
            "embedding": {"model": EMBEDDING_MODEL, "revision": EMBEDDING_REVISION, "device": "cpu", "normalize": True,
                          "query_instruction": QUERY_INSTRUCTION},
            "lexical": {"stemmer": "snowball_english", "stop_words": "sklearn English except against/not/no/without"},
            "reranker": {"model": RERANKER_MODEL, "revision": RERANKER_REVISION,
                         "backend": "onnxruntime_cpu", "candidate_sections": 12,
                         "queries": ["question_part", "evidence_target"], "initial_disagreements": "read both leaders within budget"},
            "corpus_sha256": digest(store.path), "source_sha256": store.data["source_sha256"],
            "source_updated": store.data["source_updated"], "act_id": self.store.act_id,
            "code_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
            "controlled_first_omit": sorted(self.first_omit), "controlled_unavailable": sorted(self.unavailable),
            "limits": {"model_calls": 10, "tool_calls": 12, "recovery_operations": 2},
        })

    def tool(self, name, arguments, operation):
        if self.tool_calls >= 12:
            raise RuntimeError("Tool-call budget exhausted")
        self.tool_calls += 1
        event = self.trace.event("tool_start", {"call": self.tool_calls, "name": name, "arguments": arguments})
        started = time.monotonic()
        try:
            result = operation()
            self.trace.event("tool_result", {"start_event": event, "result": result,
                                            "seconds": time.monotonic() - started})
            return result
        except Exception as exc:
            self.trace.event("tool_error", {"start_event": event, "error_type": type(exc).__name__, "error": str(exc)})
            raise

    def fetch(self, section):
        if section in self.unavailable:
            self.trace.event("controlled_unavailable", {"section_id": section})
            return None
        return self.tool("fetch", {"act_id": self.store.act_id, "section_id": section},
                         lambda: self.store.fetch(self.store.act_id, section))

    def search(self, part, target, *, mode, exclude=(), initial=False):
        rows = self.tool("search", {"act_id": self.store.act_id, "question_part": part,
                         "evidence_target": target, "mode": mode, "exclude": list(exclude), "k": 3},
                         lambda: self.store.search(part, target, mode=mode, exclude=exclude, k=3))
        self.trace.event("retrieval_ranking", self.store.last_search)
        omitted = self.unavailable | (self.first_omit if initial else set())
        kept = [row for row in rows if row["section_id"] not in omitted]
        if kept != rows:
            self.trace.event("controlled_filter", {"removed": [r for r in rows if r not in kept]})
        return kept

    def reuse_candidates(self, state, need_ids=None):
        for need in state["needs"]:
            if need["references"] or (need_ids is not None and need["id"] not in need_ids):
                continue
            supplied = state["need_sections"][need["id"]]
            for rank, section in enumerate(state["need_candidates"][need["id"]], 1):
                if section in state["provisions"] and section not in supplied:
                    supplied.append(section)
                    self.trace.event("cached_candidate_reused", {
                        "need_id": need["id"], "section": section, "initial_candidate_rank": rank})

    def evaluate(self, state, pending=None):
        needs = [n for n in state["needs"] if pending is None or n["id"] in pending]
        merged = {row["need_id"]: row for row in state["evidence"]}
        for need in needs:
            requested = set(need["references"])
            missing_refs = requested - set(state["provisions"])
            if missing_refs:
                row = Selection(need_id=need["id"], unit_ids=[], missing_user_fact="",
                                missing_evidence="Requested section unavailable: " + ", ".join(sorted(missing_refs)))
                merged[need["id"]] = row.model_dump()
                self.trace.event("requested_reference_missing", row.model_dump())
                continue
            section_ids = state["need_sections"].get(need["id"], [])
            if requested:
                allowed = requested | {s for ref in requested for s in state["provisions"][ref]["references"]}
                section_ids = [s for s in section_ids if s in allowed]
            candidates = state["need_candidates"].get(need["id"], [])
            section_ids = sorted(section_ids, key=lambda s: candidates.index(s) if s in candidates else len(candidates))
            provisions = [state["provisions"][s] for s in section_ids]
            units = {u["id"]: {**u, "act_id": p["act_id"]} for p in provisions for u in p["units"]}
            source_data = [{"act_id": p["act_id"], "section": p["id"],
                            "title": p["title"], "units": p["units"]} for p in provisions]
            payload = {"act_id": self.store.act_id, "provisions": source_data,
                       "stated_facts": state["stated_facts"], "question": need["question"],
                       "required_facts": state.get("required_facts", {}).get(need["id"], []),
                       "needs": [{"id": need["id"], "question": need["question"]}]}
            if self.store.act_id != ACT_ID:
                payload["act_name"] = self.store.act_name
            self.trace.event("need_evidence_scope", {"need_id": need["id"], "sections": section_ids})
            try:
                if any(f["name"] == "worker_count" for f in payload["required_facts"]):
                    matched = worker_condition_units(units)[:1]
                    result = EvidenceMap(selections=[Selection(need_id=need["id"], unit_ids=matched,
                        answer_quotes=[units[uid]["text"] for uid in matched], missing_user_fact="",
                        missing_evidence="" if matched else "The worker-count condition is absent from the supplied provisions.")])
                    self.trace.event("numeric_condition_evidence", {"need_id": need["id"], "parsed": result.model_dump(),
                        "rule": "Read the reviewed less-than-ten-workers predicate; apply user facts separately."})
                else:
                    result = self.model.call(EvidenceMap, EVIDENCE_PROMPT, payload, self.trace,
                                             lambda value: validate_map(value, [need], units, self.store.act_id))
            except ValueError as exc:
                self.trace.event("selection_rejected", {"need_id": need["id"], "error": str(exc)})
                result = EvidenceMap(selections=[Selection(need_id=need["id"], unit_ids=[],
                    missing_evidence="Supporting evidence could not be verified for: " + need["question"],
                    missing_user_fact="")])
            for row in result.selections:
                facts = state.get("required_facts", {}).get(row.need_id, [])
                if any(f["status"] == "UNKNOWN" for f in facts):
                    row.missing_user_fact = " ".join(f["question"] for f in facts if f["status"] == "UNKNOWN")
                    self.trace.event("required_fact_gap", {"need_id": row.need_id, "facts": facts,
                                                           "missing_user_fact": row.missing_user_fact})
                merged[row.need_id] = row.model_dump()
                self.trace.event("evidence_row", row.model_dump())
        state["evidence"] = [merged[n["id"]] for n in state["needs"] if n["id"] in merged]
        full_sections = set(re.findall(r"\bexplain\s+section\s+(\d+[A-Z]?)\b(?!\s*\()", state["question"], re.I))
        for row in state["evidence"]:
            pending_units, selected = list(row["unit_ids"]), set()
            while pending_units:
                unit_id = pending_units.pop(0)
                if unit_id in selected:
                    continue
                selected.add(unit_id)
                unit = self.store.units[unit_id]
                section = unit["section_id"]
                section_units = state["provisions"][section]["units"]
                pending_units.extend(context_dependencies(unit, section_units, complete_section=section in full_sections))
            sections = dict.fromkeys(self.store.units[u]["section_id"] for u in row["unit_ids"])
            expanded = [u["id"] for s in sections for u in state["provisions"][s]["units"] if u["id"] in selected]
            if expanded != row["unit_ids"]:
                self.trace.event("provision_context_expansion", {
                    "need_id": row["need_id"], "selected": row["unit_ids"], "expanded": expanded,
                    "reason": "Retain the complete requested section or an explicit dependency within the same provision.",
                })
                row["unit_ids"] = expanded
        self.trace.event("evidence_table", {"entries": state["evidence"]})
        return state

    def first_pass(self, question):
        state = {"question": question, "needs": [], "stated_facts": [], "provisions": {},
                 "required_facts": {}, "need_sections": {}, "need_candidates": {}, "evidence": [],
                 "need_alternatives": {}, "recovery_actions": [], "terminal_reason": None}
        self.trace.event("question", {"text": question})
        explicit = section_references(question)
        aliases = self.store.data.get("aliases", ["POSH", "sexual harassment of women at workplace"])
        named_act = any(re.search(r"\b" + re.escape(alias) + r"\b", question, re.I) for alias in aliases)
        selected_scope = "POSH" if self.store.act_id == ACT_ID else "SELECTED"
        if explicit and not named_act and not re.search(r"\b(?:Act|Code)\b", question, re.I):
            state["terminal_reason"] = "CLARIFY"
            state["clarification"] = "Which Act does the section reference refer to?"
            self.trace.event("scope_decision", state)
            return state

        parts = question_parts(question)
        self.trace.event("question_parts", {"parts": parts, "method": "Explicit question-clause boundaries; original words retained"})

        def validate_plan(plan):
            if plan.scope == selected_scope and not plan.needs:
                raise ValueError("A selected-Act question needs at least one evidence requirement")
            rejected_facts = [fact for fact in plan.stated_facts if fact not in question]
            if rejected_facts:
                self.trace.event("planner_facts_rejected", {"facts": rejected_facts,
                    "reason": "Not exact user text; omitted from stated facts"})
                plan.stated_facts = [fact for fact in plan.stated_facts if fact in question]
            if plan.scope == selected_scope and [n.question for n in plan.needs] != parts:
                raise ValueError("Use exactly the supplied question_parts, in order, without rewriting")
            original = " ".join(question.split()).casefold()
            normalized_parts = [" ".join(n.question.split()).casefold().strip(" ?.,") for n in plan.needs]
            for part in normalized_parts:
                if not part or part not in original:
                    raise ValueError("Copy the question part from the user's original words; do not add conditions. Invalid part: " + part)
            if len(set(normalized_parts)) != len(normalized_parts):
                raise ValueError("Each need must address a distinct part of the question")
            references = {s for n in plan.needs for s in n.references}
            if references - set(explicit):
                raise ValueError("references must contain only section numbers explicitly named by the user")
            retained = references | {s for n in plan.needs for s in
                                    section_references(n.question + " " + n.target)}
            if plan.scope == selected_scope and set(explicit) - retained:
                raise ValueError("Retain every user-named section on its relevant need: " + ", ".join(explicit))

        schema, instruction = Plan, PLAN_PROMPT
        if self.store.act_id != ACT_ID:
            schema = ActPlan
            instruction = PLAN_PROMPT.replace("The selected corpus is the POSH Act, 2013, snapshot 2021-08-31.",
                f"The selected corpus is {self.store.act_name}, snapshot {self.store.data['source_updated']}.")
            instruction = instruction.replace("For POSH-domain questions, use POSH.", "For questions about this selected Act, use SELECTED.")
        plan = self.model.call(schema, instruction, {"question": question, "question_parts": parts}, self.trace, validate_plan)
        if plan.scope != selected_scope:
            state["terminal_reason"] = "CLARIFY" if plan.scope == "AMBIGUOUS" else "OUT_OF_SCOPE"
            state["clarification"] = plan.clarification or "This request covers only the selected Act snapshot: " + self.store.act_name
            self.trace.event("scope_decision", state)
            return state
        state["needs"] = [{"id": f"n{i + 1}", **need.model_dump()} for i, need in enumerate(plan.needs)]
        for need in state["needs"]:
            question_refs = set(section_references(need["question"]))
            target_refs = set(section_references(need["target"]))
            invented = (question_refs | target_refs) - set(explicit)
            if invented:
                proposed = copy.deepcopy(need)
                if question_refs - set(explicit):
                    need["question"] = question
                need["target"] = need["question"]
                self.trace.event("planner_reference_rejected", {
                    "proposed": proposed, "used": need, "unverified_sections": sorted(invented),
                    "reason": "Search targets cannot introduce section numbers absent from the user question.",
                })
            need["references"] = list(dict.fromkeys(need["references"] + [s for s in
                section_references(need["question"] + " " + need["target"]) if s in explicit]))
        state["stated_facts"] = plan.stated_facts
        for need in state["needs"]:
            facts = required_user_facts(question, need["question"] + " " + need["target"]) if self.store.act_id == ACT_ID else []
            if facts:
                state["required_facts"][need["id"]] = facts
        self.trace.event("required_user_facts", state["required_facts"])
        for section in explicit:
            if section not in self.store.provisions:
                state["terminal_reason"] = "INSUFFICIENT"
                state["clarification"] = f"Section {section} is not in the selected Act snapshot."
                return state
        for need in state["needs"]:
            state["need_sections"][need["id"]] = []
            if need["references"]:
                state["need_candidates"][need["id"]] = need["references"]
                sections = [s for s in need["references"] if s not in self.first_omit]
            else:
                rows = self.search(need["question"], need["target"], mode="hybrid", initial=True)
                state["need_candidates"][need["id"]] = [row["section_id"] for row in rows]
                state["need_alternatives"][need["id"]] = [row["section_id"] for row in rows[1:]
                    if 1 in row.get("rerank_ranks", {}).values()]
                sections = [row["section_id"] for row in rows[:1]]
            for section in sections:
                provision = state["provisions"].get(section)
                if provision:
                    self.trace.event("cached_provision_reused", {"need_id": need["id"], "section": section})
                else:
                    provision = self.fetch(section)
                if provision:
                    state["provisions"][provision["id"]] = provision
                    state["need_sections"][need["id"]].append(provision["id"])
        self.reuse_candidates(state)
        for need in state["needs"]:
            for section in state["need_alternatives"].get(need["id"], []):
                if section in state["need_sections"][need["id"]]:
                    continue
                if self.tool_calls >= 10:
                    self.trace.event("initial_alternative_deferred", {"need_id": need["id"], "section": section})
                    continue
                provision = self.fetch(section)
                if provision:
                    state["provisions"][section] = provision
                    state["need_sections"][need["id"]].append(section)
                    self.trace.event("ranking_disagreement_read", {"need_id": need["id"], "section": section})
        self.reuse_candidates(state)
        return self.evaluate(state)

    def recover(self, first):
        state = copy.deepcopy(first)
        self.trace.event("recovery_checkpoint", state)
        if state["terminal_reason"]:
            return state
        gaps = [r for r in state["evidence"] if r["status"] in {"MISSING", "PARTIAL"}]
        if not gaps:
            self.trace.event("controller_stop", {"reason": "No missing legal evidence; user-fact gaps do not trigger search."})
            return state
        before = {r["need_id"]: r for r in state["evidence"]}
        needs = {n["id"]: n for n in state["needs"]}
        operations = 0
        for gap in gaps:
            need = needs[gap["need_id"]]
            self.reuse_candidates(state, {need["id"]})
            if state["need_sections"][need["id"]] != first["need_sections"][need["id"]]:
                continue
            if operations >= 2:
                continue
            need, query = needs[gap["need_id"]], gap["missing"]
            if not query.strip():
                continue
            requested_missing = [s for s in need["references"] if s not in state["provisions"]]
            if requested_missing:
                section = requested_missing[0]
                operations += 1
                decision = {"need_id": need["id"], "action": "fetch_requested_reference", "section": section, "gap": query}
                self.trace.event("controller_decision", decision)
                state["recovery_actions"].append(decision)
                provision = self.fetch(section)
                if provision:
                    state["provisions"][section] = provision
                    state["need_sections"][need["id"]].append(section)
                continue
            mentioned = section_references(query)
            linked = {s for ref in state["need_sections"][need["id"]] for s in state["provisions"][ref]["references"]}
            references = [s for s in mentioned if s in linked and s in self.store.provisions and s not in state["provisions"]]
            if references:
                section = references[0]
                operations += 1
                decision = {"need_id": gap["need_id"], "action": "fetch_cross_reference", "section": section, "gap": query}
                self.trace.event("controller_decision", decision)
                state["recovery_actions"].append(decision)
                p = self.fetch(section)
                if p:
                    state["provisions"][p["id"]] = p
                    state["need_sections"][need["id"]].append(p["id"])
                continue
            operations += 1
            decision = {"need_id": gap["need_id"], "action": "lexical_search", "target": query,
                        "reason": "Search specifically for an unfilled requirement; exclude provisions already read."}
            self.trace.event("controller_decision", decision)
            state["recovery_actions"].append(decision)
            rows = self.search(need["question"], query, mode="lexical", exclude=state["need_sections"][need["id"]])
            if rows and operations < 2:
                section = rows[0]["section_id"]
                p = state["provisions"].get(section)
                if p:
                    self.trace.event("cached_provision_reused", {"need_id": need["id"], "section": section})
                else:
                    p = self.fetch(section)
                operations += 1
                if p:
                    state["provisions"][p["id"]] = p
                    state["need_sections"][need["id"]].append(p["id"])
        self.reuse_candidates(state, {gap["need_id"] for gap in gaps})
        if state["need_sections"] == first["need_sections"]:
            self.trace.event("controller_stop", {"reason": "Recovery added no provisions"})
            return state
        self.evaluate(state, pending={gap["need_id"] for gap in gaps})
        for index, row in enumerate(state["evidence"]):
            previous = before[row["need_id"]]
            added = set(state["need_sections"][row["need_id"]]) - set(first["need_sections"][row["need_id"]])
            fresh_units = {u["id"] for section in added for u in state["provisions"][section]["units"]}
            if previous["status"] in {"MISSING", "PARTIAL"} and row["status"] != "MISSING":
                if not fresh_units.intersection(row["unit_ids"]):
                    self.trace.event("recovery_update_rejected", {
                        "proposed": row, "kept": previous,
                        "reason": "No additional source supplied for this requirement supports the update.",
                    })
                    state["evidence"][index] = previous
        self.trace.event("coverage_change", {"before": before, "after": state["evidence"],
            "new_sections": sorted(set(state["provisions"]) - set(first["provisions"])),
            "need_sections": state["need_sections"]})
        return state

    def render(self, state):
        evidence = state.get("evidence", [])
        covered = [e for e in evidence if e["unit_ids"] and e["status"] in {"COVERED", "PARTIAL", "NEEDS_FACT"}]
        unit_ids = list(dict.fromkeys(u for e in covered for u in e["unit_ids"]))
        quotes = []
        for unit_id in unit_ids:
            unit = self.store.units[unit_id]
            p = state["provisions"][unit["section_id"]]
            quotes.append({"unit_id": unit_id, "section": p["id"], "title": p["title"],
                           "quote": unit["text"], "act_id": self.store.act_id,
                           "source_file": self.store.data["source_file"], "pdf_pages": p["pages"]})
        if state.get("terminal_reason"):
            status = state["terminal_reason"]
        elif any(e["status"] == "NEEDS_FACT" for e in evidence):
            status = "NEEDS_FACT"
        elif evidence and all(e["status"] == "COVERED" for e in evidence):
            status = "COVERED"
        else:
            status = "PARTIAL" if quotes else "INSUFFICIENT"
        result = {
            "status": status, "question": state["question"], "act_name": self.store.act_name,
            "source_updated": self.store.data["source_updated"], "answer_format": "source excerpts",
            "evidence": evidence, "quotes": quotes, "stated_facts": state.get("stated_facts", []),
            "required_facts": state.get("required_facts", {}),
            "gaps": [e["missing"] for e in evidence if e["missing"]],
            "clarification": state.get("clarification"),
            "condition_check": small_workplace_check(state["question"], unit_ids) if self.store.act_id == ACT_ID else None,
            "model_calls": self.model.calls, "tool_calls": self.tool_calls,
        }
        self.trace.event("answer_rendered", result)
        return result
