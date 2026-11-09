import concurrent.futures
import json
import os

import flask
import flask_cors

import chatbot.legalos_rag
import chatbot.legalos_rag.newPipeline
import chatbot.legalos_rag.workflow
import chatbot.main


# -------------------- SETTINGS --------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "rag_v1.json")
DEFAULT_TIMEOUT_SECONDS = 300
ANSWERED_STATUSES = ("answered", "partial", "needs_fact")


# -------------------- RESPONSES --------------------

def _reply(status: str, **fields) -> dict:
    """One response shape for every path; `answer_found`, `reply` and `citations` keep the old keys."""
    body = {"status": status, "answer_found": False, "reply": "", "citations": [], "asks": [],
            "missing": [], "sources": [], "calls": 0, "seconds": 0}
    body.update(fields)
    return body


def _legacy_reply(result) -> dict:
    if result is None or not result.answer_found:
        return _reply("not_found", reply="Not found in the documents")
    return _reply(
        "answered",
        answer_found=True,
        reply=result.explanation,
        citations=[{"page": c.page, "quote": c.quote} for c in result.citations],
    )


def _new_reply(record: dict) -> dict:
    if record["status"] == "error":
        return _reply("error", error=record["error"], seconds=record["seconds"])
    return _reply(
        record["status"],
        answer_found=record["status"] in ANSWERED_STATUSES,
        reply=record["answer"],
        asks=record["asks"],
        missing=record["missing"],
        sources=record["sources"],
        calls=record["calls"],
        seconds=record["seconds"],
    )


# -------------------- APP --------------------

def create_app(config: dict) -> flask.Flask:
    """Build the Flask app from a RAG config.

    The optional `pipeline` object chooses the path as in the CLI: `legacy` (default), `new` or
    `both`. The optional `api.timeout_seconds` (default 300) limits one request.
    """
    app = flask.Flask(__name__)
    flask_cors.CORS(app)

    # The log file is for the interactive CLI; the API does not write one.
    db_path, template, slm, _, _ = chatbot.legalos_rag.ensure_requirements(
        {key: value for key, value in config.items() if key != "logging"}
    )
    pipeline = chatbot.legalos_rag.newPipeline.load_pipeline(config, slm)
    if pipeline:
        pipeline["db_path"] = db_path
    graph = None
    if pipeline is None or pipeline["mode"] == "both":
        graph = chatbot.legalos_rag.workflow.build_rag_graph(
            db_path=str(db_path), prompt_template=template["text"], slm=slm)
    timeout = config.get("api", {}).get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    acts = {source["act_id"] for source in pipeline["sources_config"]["sources"]} if pipeline else set()

    # One worker: the local vector store allows one open client at a time.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def answer(query: str, act_id: str | None) -> dict:
        body = None
        if pipeline:
            body = _new_reply(chatbot.legalos_rag.newPipeline.run_new(pipeline, query, act_id))
        if graph is not None:
            result, _, _, _ = chatbot.main.run_rag(query, db_path, template["text"], slm, rag_graph=graph)
            legacy = _legacy_reply(result)
            body = legacy if body is None else {**body, "legacy": legacy}
        body["mode"] = pipeline["mode"] if pipeline else "legacy"
        return body

    @app.route("/chat", methods=["GET", "POST"])
    def chat():
        if flask.request.method == "GET":
            data = flask.request.args
        else:
            data = flask.request.get_json(silent=True) or {}
        query = (data.get("message") or "").strip()
        act_id = data.get("act_id")

        if not query:
            return flask.jsonify(_reply("error", error="Empty query")), 400
        if pipeline and act_id and act_id not in acts:
            return flask.jsonify(_reply("error", error=f"Unknown Act: {act_id}")), 400

        future = executor.submit(answer, query, act_id)
        try:
            body = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return flask.jsonify(_reply("timeout", error=f"The request took longer than {timeout} seconds")), 504
        except Exception as error:
            return flask.jsonify(_reply("error", error=str(error))), 500
        return flask.jsonify(body), 500 if body["status"] == "error" else 200

    return app


# -------------------- ENTRY POINT --------------------

if __name__ == "__main__":
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        create_app(json.load(f)).run(host="0.0.0.0", port=5000)
