"""Run the isolated evidence POC with mandatory append-only traces."""

import argparse
import json
from pathlib import Path
import time

from .corpus import EvidenceStore
from .catalog import ActCatalog
from .answer import compose_answer
from .model import DEFAULT_MODEL, Model
from .trace import Trace
from .workflow import Controller


def write_answer(trace, result):
    lines = [result["question"], "", f"Status: {result['status']}",
             f"Source: {result.get('act_name', 'No Act selected')} — snapshot {result.get('source_updated', 'not selected')}", ""]
    for part in result.get("answer", {}).get("parts", []):
        lines += [part["text"] + " [" + ", ".join(part["citations"]) + "]", ""]
    lines += ["Source excerpts:", ""]
    for quote in result.get("quotes", []):
        lines += [f"Section {quote['section']} / {quote['unit_id']} — PDF pages {quote['pdf_pages']}",
                  quote["quote"], ""]
    if result.get("condition_check"):
        check = result["condition_check"]
        lines += [f"Numeric check: {check['source_quote']} → {check['condition']}: {check['status']}.",
                  check["limit"], ""]
    lines += ["Unresolved: " + gap for gap in result.get("gaps", [])]
    if result.get("clarification"):
        lines.append(result["clarification"])
    if result.get("error"):
        lines.append("Run error: " + result["error"])
    (trace.directory / "answer.md").write_text("\n".join(lines) + "\n")


def run(question, store, output, *, model_name=DEFAULT_MODEL, seed=11, recovery=True, draft=False):
    trace = Trace(output, metadata={"question": question, "recovery": recovery, "seed": seed})
    model = Model(model_name, seed)
    controller = Controller(store, model, trace)
    started = time.monotonic()
    try:
        state = controller.first_pass(question)
        trace.event("first_pass_checkpoint", state)
        if recovery:
            state = controller.recover(state)
        result = controller.render(state)
        result["state"] = state
        if draft:
            compose_answer(model, result, trace)
    except Exception as exc:
        trace.event("request_error", {"type": type(exc).__name__, "message": str(exc)})
        result = {"status": "ERROR", "question": question, "error": str(exc), "quotes": [],
                  "act_name": store.act_name, "source_updated": store.data["source_updated"],
                  "model_calls": model.calls, "tool_calls": controller.tool_calls}
    result.update(seconds=time.monotonic() - started, trace_directory=str(trace.directory),
                  recovery=recovery, seed=seed)
    trace.result(result)
    write_answer(trace, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--output", required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--corpus", help="Use one pinned corpus directly")
    source.add_argument("--catalog", default=str(Path(__file__).parent / "data/catalog.json"), help="Resolve an explicitly named Act using this source catalog")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--no-recovery", action="store_true")
    parser.add_argument("--draft", action="store_true", help="Write a concise answer with checked citation IDs")
    args = parser.parse_args()
    if not args.question.strip():
        parser.error("Question cannot be empty")
    setup = Trace(args.output, label="setup", metadata={"corpus": args.corpus, "model": args.model})
    if not args.corpus:
        corpus, resolution = ActCatalog(args.catalog).resolve(args.question)
        setup.event("act_resolution", resolution)
        if corpus is None:
            result = {**resolution, "question": args.question, "quotes": [], "model_calls": 0, "tool_calls": 0}
            setup.result(result)
            write_answer(setup, result)
            print(json.dumps(result, indent=2))
            return
        args.corpus = str(corpus)
    try:
        store = EvidenceStore(args.corpus, Path(args.output) / "index")
        setup.result({"status": "READY", "sections": len(store.provisions), "search_chunks": len(store.chunks)})
    except Exception as exc:
        setup.result({"status": "ERROR", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    result = run(args.question, store, args.output, model_name=args.model,
                 seed=args.seed, recovery=not args.no_recovery, draft=args.draft)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
