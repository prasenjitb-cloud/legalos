"""LangGraph orchestrator: choose the next job from the evaluator's findings, within a call limit."""

import argparse
import json
import pathlib
from typing import Required, TypedDict

import langchain_ollama
from langgraph.graph import END, START, StateGraph

from chatbot.legalos_rag import answerAgent, applicabilityAgent, evaluator, evidenceAgent, evidenceFilter, retriever


_DEFAULTS = {"max_calls": 8, "max_repairs": 1, "allow_extra_search": False, "max_searches": 1}
_OMISSIONS = {"omitted_limit", "hedge_dropped", "omitted_proviso"}
_NOTE = "Condition statuses are a model's assessment, not a finding; the evaluator must check them."


class OrchestratorState(TypedDict, total=False):
    """State shared by the orchestrator nodes."""

    question: Required[str]
    act_id: Required[str]
    assessed: dict
    answer: dict
    evaluation: dict
    calls: int
    repairs: int
    searches: int
    trace: list[str]
    result: dict


def _settings(config: dict) -> dict:
    return {**_DEFAULTS, **config.get("orchestrator", {})}


def _kinds(evaluation: dict) -> set[str]:
    return {finding["kind"] for part in evaluation["parts"] for finding in part["findings"]}


def _repair(answer: dict, evaluation: dict) -> dict:
    """Add each omitted source clause, word for word, to its part's explanation."""
    findings = {part["id"]: [f["source_text"] for f in part["findings"] if f["kind"] in _OMISSIONS and f["source_text"]]
                for part in evaluation["parts"]}
    parts = []
    for part in answer["parts"]:
        clauses = list(dict.fromkeys(findings.get(part["id"], [])))
        if clauses:
            addition = " ".join(f"The section also states: “{clause}”" for clause in clauses)
            part = {**part, "explanation": f"{part['explanation']} {addition}".strip(), "source": "model+quotes"}
        parts.append(part)
    return {**answer, "parts": parts}


def _search_more(config: dict, vectorstore, assessed: dict, evaluation: dict, slm) -> dict:
    """Search again with the query rewriter for each part that found nothing, keeping only checked evidence."""
    evidence_parts = {part["id"]: part for part in assessed["evidence"]["parts"]}
    parts = {part["id"]: part for part in assessed["parts"]}
    for part in evaluation["parts"]:
        evidence_part = evidence_parts[part["id"]]
        if evidence_part["status"] == "found":
            continue
        target = evidence_part["target"]
        found = retriever.retrieve(config, vectorstore, assessed["act_id"], evidence_part["question"], target, slm)
        checked = evidenceFilter.filter_evidence(evidence_part["question"], target, found["results"], slm)
        if not checked["kept"]:
            continue
        evidence_part.update(status="found", completeness="not_checked", evidence=[{
            "source_id": item["section"]["source_id"], "section": item["section"]["section"],
            "title": item["section"]["title"], "support_quote": item["support_quote"],
            "found_by": item["found_by"], "provision": item["section"]} for item in checked["kept"]])
        evidence_part.pop("missing", None)
        parts[part["id"]].update(finding="general", ask=None, conditional=None, checks=[], note=_NOTE)
    return assessed


def build_orchestrator(config: dict, vectorstore, slm):
    """Compile the assess -> answer -> evaluate graph, with repair and optional search jobs.

    Settings come from the `orchestrator` object of the source config: `max_calls` (jobs per
    question, default 8), `max_repairs` (default 1), `allow_extra_search` (default false) and
    `max_searches` (default 1). A request already completed for the same Act and question is reused.
    """
    settings = _settings(config)
    completed = {}

    def assess(state: OrchestratorState) -> dict:
        key = (state["act_id"], state["question"])
        reused = key in completed
        if not reused:
            completed[key] = applicabilityAgent.assess_applicability(
                config, vectorstore, state["act_id"], state["question"], slm)
        return {"assessed": completed[key], "calls": state.get("calls", 0) + (0 if reused else 1),
                "trace": state.get("trace", []) + ["assess (reused)" if reused else "assess"]}

    def answer(state: OrchestratorState) -> dict:
        return {"answer": answerAgent.write_answer(state["assessed"], slm), "calls": state["calls"] + 1,
                "trace": state["trace"] + ["answer"]}

    def evaluate(state: OrchestratorState) -> dict:
        evaluation = evaluator.evaluate_answer(config, state["answer"], state["assessed"]["evidence"])
        return {"evaluation": evaluation, "trace": state["trace"] + [f"evaluate: {evaluation['verdict']}"]}

    def repair(state: OrchestratorState) -> dict:
        return {"answer": _repair(state["answer"], state["evaluation"]), "calls": state["calls"] + 1,
                "repairs": state.get("repairs", 0) + 1, "trace": state["trace"] + ["repair"]}

    def search_more(state: OrchestratorState) -> dict:
        assessed = _search_more(config, vectorstore, state["assessed"], state["evaluation"], slm)
        return {"assessed": assessed, "calls": state["calls"] + 1, "searches": state.get("searches", 0) + 1,
                "trace": state["trace"] + ["search more"]}

    def next_job(state: OrchestratorState) -> str:
        evaluation, kinds = state["evaluation"], _kinds(state["evaluation"])
        if state["calls"] >= settings["max_calls"] or evaluation["verdict"] == "fail":
            return "finish"
        if kinds & _OMISSIONS and state.get("repairs", 0) < settings["max_repairs"]:
            return "repair"
        if ("not_found" in kinds and settings["allow_extra_search"]
                and state.get("searches", 0) < settings["max_searches"]):
            return "search_more"
        return "finish"

    def finish(state: OrchestratorState) -> dict:
        evaluation, assessed = state["evaluation"], state["assessed"]
        asks = [part["ask"] for part in assessed["parts"] if part.get("ask")]
        not_found = [part["missing"] for part in assessed["evidence"]["parts"] if part["status"] != "found"]
        if evaluation["verdict"] == "fail":
            status = "failed_checks"
        elif state["calls"] >= settings["max_calls"] and evaluation["verdict"] != "pass":
            status = "call_limit"
        elif len(not_found) == len(assessed["evidence"]["parts"]):
            status = "no_evidence"
        elif asks:
            status = "needs_fact"
        elif not_found or evaluation["verdict"] != "pass":
            status = "partial"
        else:
            status = "answered"
        return {"result": {
            "status": status, "question": state["question"], "act_id": state["act_id"],
            "answer": state["answer"]["text"] if state["answer"]["parts"] else "", "asks": asks,
            "missing": list(dict.fromkeys(not_found + evaluation["missing_details"])) if status != "answered" else [],
            "evaluation": evaluation, "calls": state["calls"], "trace": state["trace"]}}

    builder = StateGraph(OrchestratorState)
    for name, node in (("assess", assess), ("answer", answer), ("evaluate", evaluate), ("repair", repair),
                       ("search_more", search_more), ("finish", finish)):
        builder.add_node(name, node)
    builder.add_edge(START, "assess")
    builder.add_edge("assess", "answer")
    builder.add_edge("answer", "evaluate")
    builder.add_conditional_edges("evaluate", next_job, {"repair": "repair", "search_more": "search_more", "finish": "finish"})
    builder.add_edge("repair", "evaluate")
    builder.add_edge("search_more", "answer")
    builder.add_edge("finish", END)
    return builder.compile()


def run_question(graph, act_id: str, question: str) -> dict:
    """Run one question through the compiled orchestrator and return its result."""
    return graph.invoke({"question": question, "act_id": act_id, "calls": 0, "trace": []})["result"]


def main() -> None:
    """Print the orchestrator's result for one question."""
    parser = argparse.ArgumentParser(description="Answer a question from one verified POC Act with the orchestrator")
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
        result = run_question(build_orchestrator(config, vectorstore, slm), args.act, args.question)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"Status: {result['status']}  (jobs: {result['calls']}; path: {' > '.join(result['trace'])})\n")
    print(result["answer"])
    for ask in result["asks"]:
        print(f"\nPlease tell us: {ask}")
    for detail in result["missing"]:
        print(f"\nStill missing: {detail}")


if __name__ == "__main__":
    main()
