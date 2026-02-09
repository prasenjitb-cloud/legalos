import argparse
import os
import json
import datetime
import pathlib
import chatbot.legalos_rag.runRag
import chatbot.legalos_rag.prompt.promptSchema
import chatbot.legalos_rag

# -------------------- LOAD QUESTIONS --------------------
def _load_questions(path):
    """Load the question set from a JSON file. Expects a list of objects with at least 'id' and 'question' keys.
    Args:
        path: The path to the JSON file containing the question set.
    Returns:
        list: A list of dictionaries, each containing 'id' and 'question' keys.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------- WRITE RUN OUTPUTS DIRECTORY --------------------
def _write_run_outputs_dir(outputpath):
    """Ensure the outputs directory exists and return its absolute path.
    Returns:
        str: The absolute path to the outputs directory.
    """
    os.makedirs(outputpath, exist_ok=True)
    return pathlib.Path(outputpath).resolve()


# -------------------- PROMPT RUN BATCH --------------------
def prompt_run_batch(db_path: str, promptTemplate: str, questionsetfile: str, slm, model_name, outputpath):
    """
    Run the RAG pipeline for every question in the question set and write one run file.

    For each question: retrieve docs from the vector DB at db_path, then either call the
    RAG invoker (promptTemplate + docs + question -> LegalAnswer) or set answer_found=False
    when no docs are retrieved. All results are collected into a single run log and
    written to outputs/run_<run_id>.json.
    Args:
        db_path: The path to the vector DB.
        promptTemplate: The prompt template to use for the RAG pipeline.
        questionsetfile: The path to the JSON file containing the question set.
        slm: The Small Language Model instance.
    """
    outputs_path = _write_run_outputs_dir(outputpath)

    questions = _load_questions(questionsetfile)

    run_id = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")

    run_log = {
        "run_id": run_id,
        "model": model_name,
        "vectordbpath": db_path,
        "questionsetfile": questionsetfile,
        "promptTemplate": promptTemplate,
        "results": []
    }

    for item in questions:
        qid = item["id"]
        question_text = item["question"]
        if not question_text:
            print("Empty question. Skipping.")
            continue

        retrieved_docs = chatbot.legalos_rag.runRag.getFacts(
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
            [result, final_prompt]= chatbot.legalos_rag.runRag.invoker(
            slm,
            retrieved_docs,
            question_text,
            promptTemplate,
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
    """Parse CLI for --config, load and validate config (vectordbpath, promptTemplate, questionsetfile), then run the batch.
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

    config_path = pathlib.Path(args.config).resolve()

    if not config_path.is_file():
        raise ValueError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config: dict = json.load(f)

    db_path, promptTemplate, slm , model_name= chatbot.legalos_rag.ensure_requirements(config)

    questionsetfile = config.get("questionsetfile")
    outputpath = config.get("outputpath")
    if not outputpath:
        raise ValueError("Config must provide 'outputpath'")

    if not questionsetfile:
        raise ValueError("Config must provide 'questionsetfile'")

    outputpath= pathlib.Path(outputpath).resolve()
    questionsetfile = pathlib.Path(questionsetfile).resolve()

    prompt_run_batch(db_path=db_path, promptTemplate=promptTemplate["text"], questionsetfile=questionsetfile, slm=slm, model_name=model_name, outputpath=outputpath)
           




if __name__ == "__main__":
    main()
