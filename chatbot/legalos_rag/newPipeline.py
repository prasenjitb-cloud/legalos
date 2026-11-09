"""Run the new graph for the CLI and the batch runner, chosen by the `pipeline` object of the config."""

import json
import pathlib
import time

from chatbot.legalos_rag import orchestrator, retriever


_MODES = ("legacy", "new", "both")


def load_pipeline(config: dict, slm) -> dict | None:
    """Read the `pipeline` object of a RAG config; return None when the current path alone is used.

    `mode` is `legacy` (the default), `new` or `both`. The new path also needs `act_id` and
    `sources_config`, the path of the source-loading config.
    """
    settings = config.get("pipeline", {})
    mode = settings.get("mode", "legacy")
    if mode not in _MODES:
        raise ValueError(f"pipeline.mode must be one of {', '.join(_MODES)}")
    if mode == "legacy":
        return None
    for key in ("act_id", "sources_config"):
        if not isinstance(settings.get(key), str) or not settings[key].strip():
            raise ValueError(f"pipeline.{key} must be a non-empty string")
    sources_config = json.loads(pathlib.Path(settings["sources_config"]).read_text(encoding="utf-8"))
    return {"mode": mode, "act_id": settings["act_id"], "sources_config": sources_config,
            "db_path": config["vectordbpath"], "slm": slm}


def run_new(pipeline: dict, question: str, act_id: str | None = None) -> dict:
    """Answer one question with the orchestrator and return a record of what happened.

    The record has the status, answer, asks, missing details, sources, decisions (the path
    taken), jobs (`calls`) and `seconds`. The vector store is opened for this question and
    closed afterwards, because the local store allows one open client at a time and the
    current path opens its own. A failure is returned as `status: error`.
    """
    started = time.monotonic()
    vectorstore = None
    try:
        vectorstore = retriever.open_vectorstore(str(pipeline["db_path"]))
        graph = orchestrator.build_orchestrator(pipeline["sources_config"], vectorstore, pipeline["slm"])
        result = orchestrator.run_question(graph, act_id or pipeline["act_id"], question)
        record = {"status": result["status"], "act_id": result["act_id"], "answer": result["answer"],
                  "asks": result["asks"], "missing": result["missing"], "sources": result["sources"],
                  "decisions": result["trace"], "calls": result["calls"], "verdict": result["evaluation"]["verdict"]}
    except Exception as error:
        record = {"status": "error", "error": str(error)}
    finally:
        if vectorstore is not None:
            vectorstore.client.close()
    record["seconds"] = round(time.monotonic() - started, 1)
    return record


def format_record(record: dict) -> str:
    """Text shown to a CLI user for one new-path record."""
    if record["status"] == "error":
        return f"New path error: {record['error']}"
    lines = [f"New path ({record['status']}; {record['calls']} jobs, {record['seconds']} s)", record["answer"]]
    lines += [f"Please tell us: {ask}" for ask in record["asks"]]
    lines += [f"Still missing: {detail}" for detail in record["missing"]]
    return "\n".join(line for line in lines if line)
