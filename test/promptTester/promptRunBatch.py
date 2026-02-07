import argparse
import os
import json
import datetime
import langchain_ollama

import chatbot.legalos_rag.factsRetriever
import chatbot.legalos_rag.ragInvoker
import chatbot.legalos_rag.prompt.promptSchema

# -------------------- GLOBAL VARIABLES --------------------

SLM_MODEL_NAME = "qwen2.5:3b-instruct"
OUTPUTS_DIR = "/test/promptTester/outputs"

# -------------------- LLM SETUP --------------------

def setup_llm():
    """Create and return the ChatOllama LLM instance used for RAG (model and temperature from module constants).
    Returns:
        langchain_ollama.ChatOllama: The ChatOllama LLM instance.
    """
    return langchain_ollama.ChatOllama(
        model= SLM_MODEL_NAME,
        temperature=0.2,
    )

# -------------------- QUESTIONS LOADING --------------------
def load_questions(path):
    """Load the question set from a JSON file. Expects a list of objects with at least 'id' and 'question' keys.
    Args:
        path: The path to the JSON file containing the question set.
    Returns:
        list: A list of dictionaries, each containing 'id' and 'question' keys.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def _write_run_outputs_dir():
    """Ensure the outputs directory exists and return its absolute path.
    Returns:
        str: The absolute path to the outputs directory.
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    return os.path.abspath(OUTPUTS_DIR)


# -------------------- PROMPT RUN BATCH --------------------
def prompt_run_batch(db_path: str, template: str, questionsetfile: str):
    """
    Run the RAG pipeline for every question in the question set and write one run file.

    For each question: retrieve docs from the vector DB at db_path, then either call the
    RAG invoker (template + docs + question -> LegalAnswer) or set answer_found=False
    when no docs are retrieved. All results are collected into a single run log and
    written to outputs/run_<run_id>.json.
    Args:
        db_path: The path to the vector DB.
        template: The template to use for the RAG pipeline.
        questionsetfile: The path to the JSON file containing the question set.
    """
    slm = setup_llm()
    outputs_path = _write_run_outputs_dir()

    questions = load_questions(questionsetfile)

    run_id = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")

    run_log = {
        "run_id": run_id,
        "model": SLM_MODEL_NAME,
        "vectordbpath": db_path,
        "questionsetfile": questionsetfile,
        "template": template,
        "results": []
    }

    for item in questions:
        qid = item["id"]
        question_text = item["question"]
        if not question_text:
            print("Empty question. Skipping.")
            continue

        retrieved_docs = chatbot.legalos_rag.factsRetriever.getFacts(
            q=question_text,
            db_path=db_path
        )

        if not retrieved_docs:
            result = chatbot.legalos_rag.prompt.promptSchema.LegalAnswer(
                answer_found=False,
                explanation=None,
                citations=[]
            )
        else:
            [result, final_prompt]= chatbot.legalos_rag.ragInvoker.invoker(
            slm,
            retrieved_docs,
            question_text,
            template,
            )

        # Log this run to a new JSON file in outputs/
        run_log["results"].append({
            "question_id": qid,
            "question": question_text,
            "retrieved_docs": retrieved_docs if retrieved_docs else [],
            "output": result.model_dump()
        })


        print(f"RAG run complete for {qid}")

    run_filename = f"run_{run_id}.json"
    run_path = os.path.join(outputs_path, run_filename)

    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, ensure_ascii=False, indent=2)

    print(f"RAG batch run complete. Logged to {run_path}")



def main():
    """Parse CLI for --config, load and validate config (vectordbpath, template, questionsetfile), then run the batch.
    """
    parser= argparse.ArgumentParser(
        description="Run a batch of prompts"
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to the config file"
    )
    
    args= parser.parse_args()

    config_path= os.path.abspath(args.config)

    if not os.path.isfile(config_path):
        raise ValueError(f"Config file does not exist: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config: dict = json.load(f)

    vectordbpath = config.get("vectordbpath")
    template = config.get("template")
    questionsetfile = config.get("questionsetfile")

    if not questionsetfile:
        raise ValueError("Config must provide 'questionsetfile'")
    if not vectordbpath:
        raise ValueError("Config must provide 'vectordbpath'")
    if not isinstance(template, str) or not template.strip():
        raise ValueError("'template' must be a non-empty string")

    db_path = os.path.abspath(vectordbpath)
    questionsetfile = os.path.abspath(questionsetfile)

    if not os.path.isdir(db_path):
        raise ValueError(f"Vector DB path does not exist: {db_path}")
    if not os.path.isfile(questionsetfile):
        raise ValueError(f"Questionset file does not exist: {questionsetfile}")

    prompt_run_batch(db_path=db_path, template=template, questionsetfile=questionsetfile)
           




if __name__ == "__main__":
    main()
