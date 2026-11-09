"""Decide what can and cannot be said about a user's situation from the law and their stated facts."""

import argparse
import json
import pathlib
import re

import langchain_ollama

from chatbot.legalos_rag import conditionCheck, evidenceAgent, factExtractor, retriever


_MAX_CHECKED_SECTIONS = 2
_PERSONAL = re.compile(r"\b(?:i|i'm|i've|my|me|we|we're|our|us|mine)\b", re.IGNORECASE)
_NOTE = "Condition statuses are a model's assessment, not a finding; the evaluator must check them."


def _is_personal(question: str, facts: list[dict]) -> bool:
    """A question is about a situation when the user stated facts or speaks in the first person."""
    return bool(facts) or bool(_PERSONAL.search(question))


def _finding(personal: bool, checks: list[dict]) -> dict:
    """Choose what may be said, using only the condition statuses."""
    if not personal:
        return {"finding": "general", "ask": None, "conditional": None}
    items = [c for check in checks for c in check["conditions"]]
    conditions = [c for c in items if c["kind"] == "condition"]
    if not conditions:
        return {"finding": "cannot_assess", "ask": None, "conditional": None}
    if any(c["status"] == "NOT_MET" for c in conditions):
        return {"finding": "not_met_on_stated_facts", "ask": None, "conditional": None}
    if any(c["kind"] == "exception" and c["status"] == "MET" for c in items):
        return {"finding": "exception_applies", "ask": None, "conditional": None}
    open_conditions = [c for c in conditions if c["status"] == "UNKNOWN"]
    if open_conditions:
        decisive = max(open_conditions, key=lambda c: len(c["text"]))["text"]
        met = [c["text"] for c in conditions if c["status"] == "MET"]
        conditional = None
        if met:
            conditional = (f"On the stated facts {len(met)} condition(s) appear to be met. "
                           f"The result depends on {len(open_conditions)} still unknown.")
        return {"finding": "needs_fact", "ask": f"Is this true in your case: \"{decisive}\"?", "conditional": conditional}
    return {"finding": "conditions_met_on_stated_facts", "ask": None, "conditional": None}


def assess_applicability(config: dict, vectorstore, act_id: str, question: str, slm) -> dict:
    """Gather evidence for the question and say what can be said about the user's situation.

    Facts come from `factExtractor`, evidence from `evidenceAgent`, and each part's top
    sections go through `conditionCheck`. For a personal question the finding is one of
    `not_met_on_stated_facts`, `exception_applies`, `needs_fact` (with the longest open
    condition, quoted from the law, to ask the user about, and a conditional note), `conditions_met_on_stated_facts`
    or `cannot_assess`. A general question gets `general` and no request for facts. A part with no
    evidence is `no_evidence`. No finding is a personal conclusion: see the note on each part.
    """
    facts = factExtractor.extract_facts(question, slm)["facts"]
    evidence = evidenceAgent.gather_evidence(config, vectorstore, act_id, question, slm)
    parts = []
    for part in evidence["parts"]:
        personal = _is_personal(part["question"], facts) or _is_personal(question, facts)
        if part["status"] != "found":
            parts.append({"id": part["id"], "question": part["question"], "finding": "no_evidence",
                          "ask": None, "conditional": None, "missing": part["missing"], "checks": []})
            continue
        checks = [conditionCheck.check_conditions(config, item["provision"], facts, slm)
                  for item in part["evidence"][:_MAX_CHECKED_SECTIONS]]
        parts.append({"id": part["id"], "question": part["question"], "personal": personal,
                      **_finding(personal, checks), "checks": checks, "note": _NOTE})
    return {"act_id": act_id, "question": question, "facts": facts, "parts": parts}


def main() -> None:
    """Print what can be said about the user's situation for one question as JSON."""
    parser = argparse.ArgumentParser(description="Assess what the law says about a user's stated situation")
    parser.add_argument("--config", required=True, help="Path to the source-loading JSON config")
    parser.add_argument("--rag-config", required=True, help="Path to the RAG JSON config (vector store and model)")
    parser.add_argument("--act", required=True, help="Act ID, such as hma_1955")
    parser.add_argument("--question", required=True, help="The user's question, with the facts they state")
    args = parser.parse_args()
    try:
        config = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
        rag_config = json.loads(pathlib.Path(args.rag_config).read_text(encoding="utf-8"))
        slm = langchain_ollama.ChatOllama(model=rag_config["model"]["model_name"], temperature=0)
        vectorstore = retriever.open_vectorstore(rag_config["vectordbpath"])
        result = assess_applicability(config, vectorstore, args.act, args.question, slm)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    for part in result["parts"]:
        for check in part["checks"]:
            check["conditions"] = [{k: c[k] for k in ("id", "kind", "text", "status", "fact_ids")} for c in check["conditions"]]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
