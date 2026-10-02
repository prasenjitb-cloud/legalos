"""Compare the existing pipeline and POC with the configured model and full local traces."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import ollama
from langchain_ollama import ChatOllama

from chatbot.legalos_rag import queryRewriter, runRag
from chatbot.legalos_rag.poc_v2.catalog import ActCatalog
from chatbot.legalos_rag.poc_v2.corpus import EvidenceStore, digest
from chatbot.legalos_rag.poc_v2.main import run
from chatbot.legalos_rag.poc_v2.model import CONFIG_PATH, DEFAULT_MODEL, Model
from chatbot.legalos_rag.poc_v2.trace import Trace

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "chatbot/legalos_rag/poc_v2/data"


def baseline(cases, database, output, settings):
    # Match the POC's CPU embedding device without changing the original retrieval functions.
    factory = runRag.langchain_huggingface.HuggingFaceEmbeddings
    setup_store = runRag._setup_vectorstore
    invoke = ChatOllama.invoke
    trace = None
    store = None

    def cpu_embeddings(**kwargs):
        return factory(**kwargs, model_kwargs={"device": "cpu"})

    def logged_invoke(self, prompt, *args, **kwargs):
        event = trace.event("model_start", {"prompt": str(prompt), "model": self.model, "settings": settings})
        try:
            response = invoke(self, prompt, *args, **kwargs)
            trace.event("model_response", {"start_event": event, "raw": response.content,
                "usage": response.usage_metadata, "metadata": response.response_metadata})
            return response
        except Exception as exc:
            trace.event("model_error", {"start_event": event, "error": str(exc)})
            raise

    try:
        runRag.langchain_huggingface.HuggingFaceEmbeddings = cpu_embeddings
        store = setup_store(str(database), runRag.COLLECTION_NAME)
        runRag._setup_vectorstore = lambda *args, **kwargs: store
        ChatOllama.invoke = logged_invoke
        search = store.similarity_search_with_score

        def logged_search(query, *args, **kwargs):
            trace.event("search_request", {"query": query, "args": args, "kwargs": kwargs})
            hits = search(query, *args, **kwargs)
            trace.event("search_result", [{"text": doc.page_content, "metadata": doc.metadata,
                                           "score": float(score)} for doc, score in hits])
            return hits

        store.similarity_search_with_score = logged_search
        model = ChatOllama(model=DEFAULT_MODEL, **settings, client_kwargs={"timeout": 120})
        template = json.loads(CONFIG_PATH.read_text())["promptTemplate"]["text"]
        for case in cases:
            print("START baseline", case["id"], flush=True)
            trace = Trace(output, label="baseline-" + case["id"], metadata=case)
            started = time.monotonic()
            result = {"question_id": case["id"], "question": case["question"]}
            try:
                queries = queryRewriter.rewrite_and_expand(case["question"], model)
                trace.event("rewrite_output", queries)
                context = runRag.getFactsMulti(queries=queries, db_path=str(database))
                trace.event("retrieve_output", context)
                answer, prompt = runRag.invoker(model, context, case["question"], template) if context else (None, None)
                result.update(answer=answer.model_dump() if answer else None, rewritten_queries=queries)
                trace.event("generate_output", {"prompt": prompt, "answer": result["answer"]})
            except Exception as exc:
                result["error"] = str(exc)
            result["seconds"] = time.monotonic() - started
            trace.result(result)
            append(output / "baseline.jsonl", result)
            print("DONE baseline", case["id"], round(result["seconds"], 1), flush=True)
    finally:
        ChatOllama.invoke = invoke
        runRag._setup_vectorstore = setup_store
        runRag.langchain_huggingface.HuggingFaceEmbeddings = factory
        if store is not None:
            store.client.close()


def append(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fixture = Path(__file__).with_name("questions.json")
    cases = json.loads(fixture.read_text())["development"]
    settings = Model().settings
    identity = next(model for model in ollama.Client(timeout=5).list().models if model.model == DEFAULT_MODEL)
    sources = [CONFIG_PATH, fixture, Path(__file__), *sorted((ROOT / "chatbot/legalos_rag").rglob("*.py")),
               *sorted(DATA.glob("*.json"))]
    manifest = {"model": DEFAULT_MODEL, "model_digest": identity.digest, "settings": settings,
        "source_hashes": {str(path.relative_to(ROOT)): digest(path) for path in sources},
        "packages": {name: importlib.metadata.version(name) for name in
            ["ollama", "langchain-ollama", "sentence-transformers", "torch", "onnxruntime"]},
        "protocol": "Known development questions, one attempt each, serial baseline then POC. Same model and sampling settings; both use CPU embeddings. Original rewrite/retrieve/generate functions and prompt unchanged. POC uses prepared Act-specific sources. Timing excludes store initialization. Automated source coverage is not semantic quality."}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    baseline(cases, args.database, args.output, settings)
    catalog = ActCatalog(DATA / "catalog.json")
    stores = {}
    for case in cases:
        print("START poc", case["id"], flush=True)
        corpus, resolution = catalog.resolve(case["question"])
        setup = Trace(args.output, label="scope-" + case["id"], metadata=resolution)
        if corpus is None:
            raise ValueError("Benchmark question must name one supported Act")
        if corpus not in stores:
            stores[corpus] = EvidenceStore(corpus, args.output.parent / "index")
        setup.result({"status": "READY", "act_id": stores[corpus].act_id})
        result = run(case["question"], stores[corpus], args.output, draft=True)
        quoted = {q["unit_id"] for q in result["quotes"]}
        record = {"question_id": case["id"], "question": case["question"], "result": result,
                  "required_units": case["required_units"],
                  "source_check": result["status"] == case["expected"] and set(case["required_units"]) <= quoted}
        append(args.output / "poc.jsonl", record)
        print("DONE poc", case["id"], result["status"], record["source_check"], flush=True)


if __name__ == "__main__":
    main()
