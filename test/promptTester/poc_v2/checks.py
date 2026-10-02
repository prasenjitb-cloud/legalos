"""Deterministic contracts; these do not measure model judgment."""

import copy
import json
from pathlib import Path
import tempfile

from chatbot.legalos_rag.poc_v2.corpus import EvidenceStore, ACT_ID
from chatbot.legalos_rag.poc_v2.model import Model, DEFAULT_MODEL, CONFIG_PATH, EVIDENCE_PROMPT, validate_map, EvidenceMap
from chatbot.legalos_rag.poc_v2.facts import worker_count, required_user_facts, worker_condition_units
from chatbot.legalos_rag.poc_v2.trace import Trace
from chatbot.legalos_rag.poc_v2.workflow import Controller, small_workplace_check, context_dependencies, question_parts
from chatbot.legalos_rag.poc_v2.answer import AnswerDraft, validate_draft
from chatbot.legalos_rag.poc_v2.catalog import ActCatalog


class ScriptedModel:
    name, settings, seed = "scripted-check", {}, 0

    def __init__(self, results):
        self.results, self.calls = iter(results), 0
        self.payloads = []

    def call(self, schema, instruction, payload, trace, validate=None):
        self.calls += 1
        self.payloads.append(payload)
        result = schema.model_validate(next(self.results))
        if validate:
            validate(result)
        trace.event("scripted_model", {"schema": schema.__name__, "output": result.model_dump()})
        return result


def selection(status, units, missing="", need_id="n1"):
    return {"selections": [{"need_id": need_id, "unit_ids": units,
                            "missing_evidence": missing if status != "NEEDS_FACT" else "",
                            "missing_user_fact": missing if status == "NEEDS_FACT" else ""}]}


def main():
    assert DEFAULT_MODEL == json.loads(CONFIG_PATH.read_text())["model"]["model_name"]
    catalog = ActCatalog("chatbot/legalos_rag/poc_v2/data/catalog.json")
    for entry in catalog.entries:
        source = Path("chatbot/legalos_rag/poc_v2/data") / entry["corpus"]
        assert EvidenceStore(source, ".", dense=False).act_id == entry["act_id"]
    corpus = Path("chatbot/legalos_rag/poc_v2/data/posh.json")
    with tempfile.TemporaryDirectory() as directory:
        units = [
            {"id": "69(1)", "section_id": "69", "text": "A complaint must be filed within two years."},
            {"id": "69(2)", "section_id": "69", "text": "Notwithstanding anything contained in sub-section (1), a late complaint may be entertained for sufficient cause."},
        ]
        assert context_dependencies(units[0], units) == {"69(2)"}
        draft = AnswerDraft.model_validate({"parts": [{"need_id": "n1", "text": "Two years.", "citations": ["69(1)"]}]})
        validate_draft(draft, [{"need_id": "n1", "unit_ids": ["69(1)"]}])
        try:
            validate_draft(draft, [{"need_id": "n1", "unit_ids": ["9(1)"]}])
            raise AssertionError("Unsupported answer citation accepted")
        except ValueError:
            pass
        catalog_path = Path(directory) / "catalog.json"
        catalog_path.write_text(json.dumps([
            {"act_id": "a", "act_name": "First Act", "aliases": ["First Act"], "corpus": "a.json"},
            {"act_id": "b", "act_name": "Second Act", "aliases": ["Second Act"], "corpus": "b.json"}]))
        catalog = ActCatalog(catalog_path)
        assert catalog.resolve("First Act deadline?")[0] == Path(directory) / "a.json"
        assert catalog.resolve("First Act and Second Act?")[0] is None
        assert catalog.resolve("What is the deadline?")[0] is None
        assert question_parts("Under POSH, what relief is available during inquiry, and what factors govern compensation?") == [
            "Under POSH, what relief is available during inquiry", "what factors govern compensation?"]
        assert question_parts("What are the rights of workers and employers?") == ["What are the rights of workers and employers?"]
        store = EvidenceStore(corpus, directory, dense=False)
        original_search = store.search
        store.search = lambda part, target, **kwargs: original_search(part, target, **{**kwargs, "mode": "lexical"})
        assert len(store.provisions) == 30 and len(store.units) == 83
        assert set(worker_condition_units(store.units)) == {"6(1)", "2(p)"}
        assert "three months" in store.units["9(1)"]["text"] and "Provided further" in store.units["9(1)"]["text"]
        assert "complaints of sexual harassment" in store.units["6(1)"]["text"]
        assert store.provisions["6"]["pages"] == [6, 7] and store.provisions["15"]["pages"] == [10]
        assert small_workplace_check("We have eight workers.", ["6(1)"])["status"] == "MET"
        assert small_workplace_check("We have ten workers.", ["6(1)"])["status"] == "NOT_MET"
        assert small_workplace_check("We have workers.", ["6(1)"]) is None
        assert small_workplace_check("We have eight workers.", ["9(1)"]) is None
        assert small_workplace_check("What if there are fewer than ten workers?", ["6(1)"]) is None
        assert small_workplace_check("We do not have eight workers.", ["6(1)"]) is None
        assert small_workplace_check("My POSH workplace has seven workers.", ["6(1)"])["value"] == 7
        assert small_workplace_check("My workplace has ten workers.", ["2(p)"])["status"] == "NOT_MET"
        assert not required_user_facts("What does the fewer-than-ten-workers condition mean for my workplace?", "fewer-than-ten-workers condition")
        for question in ["We have eight workers?", "We have eight workers or maybe twelve.",
                         "We do not have eight workers.", "My workplace has eight workers. My workplace has twelve workers."]:
            assert worker_count(question)["status"] == "UNKNOWN", question
        personal = EvidenceMap.model_validate(selection("NEEDS_FACT", ["9(1)"], "date of incident"))
        assert personal.selections[0].status == "NEEDS_FACT"
        scope_model = Model("scope-contract")
        scope_model.recorded_identity = True
        scope_model.client = object()
        wrong_scope = {"act_id": ACT_ID, "needs": [{"id": "n1", "question": "Complaint deadline?"}],
                       "provisions": [{"act_id": "other_act", "section": "9", "units": [store.units["9(1)"]]}]}
        scoped = scope_model.call(EvidenceMap, EVIDENCE_PROMPT, wrong_scope, Trace(directory, label="scope-boundary"))
        assert scoped.selections[0].status == "MISSING" and not scoped.selections[0].unit_ids
        assert scope_model.calls == 0
        topic = {"act_id": ACT_ID, "question": "Appeal deadline?", "needs": [{"id": "n1", "question": "Appeal deadline?"}],
                 "provisions": [{"act_id": ACT_ID, "section": "9", "title": store.provisions["9"]["title"],
                                  "units": store.provisions["9"]["units"]}]}
        scoped = scope_model.call(EvidenceMap, EVIDENCE_PROMPT, topic, Trace(directory, label="topic-boundary"))
        assert scoped.selections[0].status == "MISSING" and scope_model.calls == 0
        for units in [{"9(1)": {**store.units["9(1)"], "act_id": "synthetic_other_act"}}, {}]:
            try:
                validate_map(EvidenceMap.model_validate(selection("COVERED", ["9(1)"])), [{"id": "n1"}], units)
                raise AssertionError("Invalid evidence accepted")
            except ValueError:
                pass
        plan = {"scope": "POSH", "needs": [{"question": "What factors determine POSH compensation?", "target": "determination of compensation factors"}],
                "stated_facts": [], "clarification": ""}
        fact_plan = copy.deepcopy(plan)
        fact_plan["stated_facts"] = ["A fact absent from the question"]
        fact_model = ScriptedModel([fact_plan, selection("COVERED", ["15"])])
        fact_trace = Trace(directory, label="discard-unverified-fact")
        fact_controller = Controller(store, fact_model, fact_trace)
        assert fact_controller.first_pass("What factors determine POSH compensation?")["stated_facts"] == []
        assert any(json.loads(line)["kind"] == "planner_facts_rejected" for line in (fact_trace.directory / "events.jsonl").read_text().splitlines())
        drift_plan = copy.deepcopy(plan)
        drift_plan["needs"][0]["question"] = "What factors determine POSH compensation during an inquiry?"
        drift_controller = Controller(store, ScriptedModel([drift_plan]), Trace(directory, label="reject-question-drift"))
        try:
            drift_controller.first_pass("What factors determine POSH compensation?")
            raise AssertionError("Question drift accepted")
        except ValueError:
            pass
        model = ScriptedModel([plan, selection("MISSING", [], "determination of compensation factors"),
                               selection("COVERED", ["15"])])
        trace = Trace(directory, label="recovery")
        controller = Controller(store, model, trace, first_omit=["15"])
        first = controller.first_pass("What factors determine POSH compensation?")
        snapshot = copy.deepcopy(first)
        assert controller.render(first)["status"] == "INSUFFICIENT"
        recovered = controller.recover(first)
        assert first == snapshot
        assert "15" in recovered["provisions"] and controller.render(recovered)["status"] == "COVERED"
        assert len(recovered["recovery_actions"]) == 1 and controller.tool_calls <= 12
        assert any(json.loads(line)["kind"] == "coverage_change" for line in (trace.directory / "events.jsonl").read_text().splitlines())
        model = ScriptedModel([plan, selection("MISSING", [], "determination of compensation factors"),
                               selection("MISSING", [], "determination of compensation factors")])
        controller = Controller(store, model, Trace(directory, label="unavailable"), unavailable=["15"])
        recovered = controller.recover(controller.first_pass("What factors determine POSH compensation?"))
        assert "15" not in recovered["provisions"] and not controller.render(recovered)["quotes"]
        named_plan = copy.deepcopy(plan)
        named_plan["needs"][0]["references"] = ["9"]
        named_plan["needs"][0]["question"] = "What does Section 9 of POSH say about compensation factors?"
        model = ScriptedModel([named_plan, selection("MISSING", [], "compensation factors"),
                               selection("COVERED", ["9(1)"])])
        controller = Controller(store, model, Trace(directory, label="no-new-support"))
        first = controller.first_pass("What does Section 9 of POSH say about compensation factors?")
        recovered = controller.recover(first)
        assert controller.render(recovered)["status"] == "INSUFFICIENT"
        bad_plan = copy.deepcopy(plan)
        bad_plan["needs"][0]["target"] = "POSH Section 9"
        model = ScriptedModel([bad_plan, selection("COVERED", ["15"])])
        controller = Controller(store, model, Trace(directory, label="unverified-reference"))
        state = controller.first_pass("What factors determine POSH compensation?")
        assert state["needs"][0]["target"] == "What factors determine POSH compensation?"
        assert "Section 9" not in store.last_search["query"]
        section_plan = {"scope": "POSH", "needs": [{"question": "Explain Section 18 of POSH.", "target": "Section 18"}],
                        "stated_facts": [], "clarification": ""}
        model = ScriptedModel([section_plan, selection("COVERED", ["18(2)"])])
        controller = Controller(store, model, Trace(directory, label="complete-section"))
        answer = controller.render(controller.first_pass("Explain Section 18 of POSH."))
        assert [q["unit_id"] for q in answer["quotes"]] == ["18(1)", "18(2)"]
        leave_plan = {"scope": "POSH", "needs": [{"question": "How much additional leave does Section 12 of POSH allow?", "target": "leave", "references": ["12"]}],
                      "stated_facts": [], "clarification": ""}
        model = ScriptedModel([leave_plan, selection("COVERED", ["12(2)"])])
        controller = Controller(store, model, Trace(directory, label="local-dependency"))
        answer = controller.render(controller.first_pass("How much additional leave does Section 12 of POSH allow?"))
        assert [q["unit_id"] for q in answer["quotes"]] == ["12(1)", "12(2)", "12(3)"]
        leave_plan["needs"][0]["question"] = "What relief does Section 12 of POSH allow?"
        model = ScriptedModel([leave_plan, selection("COVERED", ["12(1)"])])
        controller = Controller(store, model, Trace(directory, label="attached-qualification"))
        answer = controller.render(controller.first_pass("What relief does Section 12 of POSH allow?"))
        assert [q["unit_id"] for q in answer["quotes"]] == ["12(1)", "12(2)", "12(3)"]
        pair_plan = {"scope": "POSH", "needs": [
            {"question": "Under POSH, what is the deadline in Section 9", "target": "complaint deadline", "references": ["9"]},
            {"question": "what is the deadline in Section 18?", "target": "appeal deadline", "references": ["18"]}],
            "stated_facts": [], "clarification": ""}
        question = "Under POSH, what is the deadline in Section 9 and what is the deadline in Section 18?"
        model = ScriptedModel([pair_plan, selection("COVERED", ["9(1)"]),
                               selection("COVERED", ["18(2)"], need_id="n2")])
        controller = Controller(store, model, Trace(directory, label="named-recovery"), first_omit=["18"])
        first = controller.first_pass(question)
        assert controller.render(first)["status"] == "PARTIAL"
        assert controller.render(controller.recover(first))["status"] == "COVERED"
        assert [p["section"] for p in model.payloads[-1]["provisions"]] == ["18"]
        model = ScriptedModel([pair_plan, selection("COVERED", ["9(1)"])])
        controller = Controller(store, model, Trace(directory, label="named-unavailable"), unavailable=["18"])
        first = controller.first_pass(question)
        assert controller.render(controller.recover(first))["status"] == "PARTIAL"
        assert model.calls == 2
        scoped_plan = copy.deepcopy(pair_plan)
        scoped_plan["needs"][1]["references"] = ["15"]
        scoped_plan["needs"][1]["question"] = "what is the deadline in Section 15?"
        model = ScriptedModel([scoped_plan, selection("COVERED", ["15"]),
                               selection("COVERED", ["15"], need_id="n2")])
        controller = Controller(store, model, Trace(directory, label="separate-evidence"))
        first = controller.first_pass("Under POSH, what is the deadline in Section 9 and what is the deadline in Section 15?")
        assert first["evidence"][0]["status"] == "MISSING"
        assert first["evidence"][1]["status"] == "COVERED"
        assert first["need_sections"] == {"n1": ["9"], "n2": ["15"]}
        previous_search = store.search

        def ranked(part, target, **kwargs):
            store.last_search = {}
            return [{"section_id": s} for s in ["11", "9"] if s not in kwargs.get("exclude", ())]

        store.search = ranked
        deadline_plan = {"scope": "POSH", "needs": [
            {"question": "What is the POSH deadline", "target": "filing deadline"},
            {"question": "what is the extension?", "target": "filing extension"}], "stated_facts": [], "clarification": ""}
        model = ScriptedModel([deadline_plan, selection("MISSING", [], "filing deadline"),
            selection("MISSING", [], "filing extension", need_id="n2"), selection("COVERED", ["9(1)"]),
            selection("COVERED", ["9(1)"], need_id="n2")])
        trace = Trace(directory, label="recovered-candidate-reuse")
        controller = Controller(store, model, trace)
        first = controller.first_pass("What is the POSH deadline and what is the extension?")
        recovered = controller.recover(first)
        assert controller.render(recovered)["status"] == "COVERED"
        assert recovered["need_sections"] == {"n1": ["11", "9"], "n2": ["11", "9"]}
        assert controller.tool_calls == 5
        assert any(json.loads(line)["kind"] == "cached_candidate_reused" for line in (trace.directory / "events.jsonl").read_text().splitlines())
        store.search = lambda part, target, **kwargs: [{"section_id": "9"}] if part == "What is the POSH deadline" else ranked(part, target, **kwargs)
        model = ScriptedModel([deadline_plan, selection("COVERED", ["9(1)"]),
                               selection("COVERED", ["9(1)"], need_id="n2")])
        controller = Controller(store, model, Trace(directory, label="initial-candidate-reuse"))
        first = controller.first_pass("What is the POSH deadline and what is the extension?")
        assert first["need_sections"]["n2"] == ["11", "9"]
        assert controller.render(first)["status"] == "COVERED"
        store.search = previous_search
        count_plan = {"scope": "POSH", "needs": [{"question": "Under POSH Section 6, does my workplace meet the fewer-than-ten-workers condition?",
            "target": "worker-count condition", "references": ["6"]}], "stated_facts": [], "clarification": ""}
        model = ScriptedModel([count_plan, selection("COVERED", ["6(1)"])])
        controller = Controller(store, model, Trace(directory, label="unknown-count-guard"))
        first = controller.first_pass("Under POSH Section 6, does my workplace meet the fewer-than-ten-workers condition?")
        assert controller.render(first)["status"] == "NEEDS_FACT"
        assert controller.render(first)["condition_check"] is None
        assert model.calls == 1
        calls = controller.tool_calls
        controller.recover(first)
        assert controller.tool_calls == calls
        store.search = lambda *args, **kwargs: [
            {"section_id": "6", "rerank_ranks": {"question": 1, "target": 8}},
            {"section_id": "9", "rerank_ranks": {"question": 4, "target": 1}}]
        route_plan = {"scope": "POSH", "needs": [{"question": "Under POSH, who receives complaints against the employer?",
            "target": "reception of complaints"}], "stated_facts": [], "clarification": ""}
        model = ScriptedModel([route_plan, selection("COVERED", ["6(1)"])])
        controller = Controller(store, model, Trace(directory, label="ranking-disagreement"))
        first = controller.first_pass("Under POSH, who receives complaints against the employer?")
        assert first["need_sections"]["n1"] == ["6", "9"] and controller.tool_calls == 3
        store.search = previous_search
        other_plan = {"scope": "OTHER", "needs": [], "stated_facts": [], "clarification": "This POC covers POSH only."}
        model = ScriptedModel([other_plan])
        controller = Controller(store, model, Trace(directory, label="named-other-act"))
        assert controller.first_pass("Explain Section 73 of the CGST Act.")["terminal_reason"] == "OUT_OF_SCOPE"
        model = ScriptedModel([])
        controller = Controller(store, model, Trace(directory, label="ambiguous"))
        assert controller.render(controller.first_pass("What does Section 6 say?"))["status"] == "CLARIFY"
        assert model.calls == 0
        try:
            store.fetch("different_act", "9")
            raise AssertionError("Wrong Act accepted")
        except ValueError:
            pass
        controller.tool_calls = 12
        try:
            controller.fetch("9")
            raise AssertionError("Budget ignored")
        except RuntimeError:
            pass
    print("Source boundaries, qualifications, numeric condition, recovery, unavailable evidence, ambiguity, IDs, logging and budget checks passed.")


if __name__ == "__main__":
    main()
