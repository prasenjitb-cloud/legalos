import json
import os

import flask_cors
import flask
import langchain_ollama

import chatbot.legalos_rag.factsRetriever
import chatbot.legalos_rag.ragInvoker


# -------------------- APP SETUP --------------------

app = flask.Flask(__name__)
flask_cors.CORS(app)


# -------------------- LOAD CONFIG --------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(BASE_DIR, "config", "rag_v1.json")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

VECTOR_DB_PATH = os.path.abspath(
    os.path.join(BASE_DIR, config["vectordbpath"])
)
template = config["template"]


# -------------------- MODEL SETUP --------------------

SLM_MODEL_NAME = "qwen2.5:3b-instruct"

slm = langchain_ollama.ChatOllama(
    model=SLM_MODEL_NAME,
    temperature=1,
)


# -------------------- ROUTES --------------------


@app.route("/chat", methods=["GET", "POST"])
def chat():
    if flask.request.method == "GET":
        query = flask.request.args.get("message", "").strip()
    else:
        data = flask.request.get_json(silent=True) or {}
        query = data.get("message", "").strip()

    if not query:
        return flask.jsonify({"error": "Empty query"}), 400

    retrieved_docs = chatbot.legalos_rag.factsRetriever.getFacts(
        q=query,
        db_path=VECTOR_DB_PATH,
    )
    if not retrieved_docs:
        return flask.jsonify({
            "answer_found": False,
            "reply": "Not found in the documents",
            "citations": [],
        })

    # Run RAG
    [result, final_prompt]= chatbot.legalos_rag.ragInvoker.invoker(
        slm,
        retrieved_docs,
        query,
        template,
    )

    if not result.answer_found:
        return flask.jsonify({
            "answer_found": False,
            "reply": "Not found in the documents",
            "citations": [],
        })

    return flask.jsonify({
        "answer_found": True,
        "reply": result.explanation,
        "citations": [
            {
                "page": c.page,
                "quote": c.quote,
            }
            for c in result.citations
        ],
    })


# -------------------- ENTRY POINT --------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)