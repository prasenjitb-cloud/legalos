import argparse
import os
import json
import datetime
import pathlib
import chatbot.legalos_rag
import chatbot.legalos_rag.newPipeline
import chatbot.legalos_rag.workflow
import chatbot.main
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
        Path: Absolute path to the outputs directory.
    """
    os.makedirs(outputpath, exist_ok=True)
    return pathlib.Path(outputpath).resolve()


# -------------------- PROMPT RUN BATCH --------------------
def prompt_run_batch(db_path: str, promptTemplate: str, questionsetfile: str, slm, model_name, outputpath, new_pipeline=None):
    """
    Run the RAG pipeline for every question in the question set and append each result to a JSONL run file.

    Compile the graph once, then call run_rag() for each question and append the result
    immediately. When no chunks are retrieved, run_rag returns
    (None, [], None, queries); output is recorded as null.
    Args:
        db_path: Path to the vector DB.
        promptTemplate: Prompt template string for the RAG pipeline.
        questionsetfile: Path to the JSON file containing the question set.
        slm: Small Language Model instance.
        model_name: Model name (for metadata).
        outputpath: Directory for the run output file.
        new_pipeline: Settings from newPipeline.load_pipeline, or None to run only the current path.
            In "both" mode each record also has a "new" object; in "new" mode it has only that.
            A question may carry its own "act_id".
    """
    outputs_path = _write_run_outputs_dir(outputpath)
    questions = _load_questions(questionsetfile)

    run_id = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    run_filename = f"run_{run_id}.jsonl"
    run_path = os.path.join(outputs_path, run_filename)
    # Reuse one graph for the full batch.
    rag_graph = chatbot.legalos_rag.workflow.build_rag_graph(
        db_path=str(db_path),
        prompt_template=promptTemplate,
        slm=slm,
    )

    # write run metadata
    metadata = {
        "run_id": run_id,
        "model": model_name,
        "vectordbpath": str(db_path),
        "questionsetfile": str(questionsetfile),
        "promptTemplate": promptTemplate,
        "type": "metadata",
    }
    if new_pipeline:
        metadata.update({"pipeline": new_pipeline["mode"], "act_id": new_pipeline["act_id"]})

    with open(run_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

        # Write one JSON object per question
        for item in questions:
            try:
                qid = item["id"]
                question_text = item["question"]
                if not question_text.strip():
                    print("Empty question. Skipping.")
                    continue

                new_record = None
                if new_pipeline:
                    new_record = chatbot.legalos_rag.newPipeline.run_new(
                        new_pipeline, question_text, item.get("act_id"))
                    if new_pipeline["mode"] == "new":
                        f.write(json.dumps({"question_id": qid, "question": question_text, "new": new_record},
                                           ensure_ascii=False) + "\n")
                        f.flush()
                        print(f"New path run complete for {qid}")
                        continue

                result, retrieved_chunks, _, rewritten_queries = chatbot.main.run_rag(
                    question_text,
                    db_path,
                    promptTemplate,
                    slm,
                    rag_graph=rag_graph,
                )

                # Append result line (flush so partial run is persisted if interrupted)
                record = {
                    "question_id": qid,
                    "question": question_text,
                    "rewritten_queries": rewritten_queries,
                    "retrieved_chunks": retrieved_chunks if retrieved_chunks else "",
                    "output": result.model_dump() if result else None,
                }
                if new_record:
                    record["new"] = new_record
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                print(f"RAG run complete for {qid}")

            except Exception as e:
                qid_safe = item.get("id", "")
                print(f"Error running RAG for question {qid_safe}: {e}")
                # Log error record so run file stays complete
                record = {
                    "question_id": item.get("id", ""),
                    "question": item.get("question", ""),
                    "error": str(e),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

    print(f"RAG batch run complete. Logged to {run_path}")



def main() -> None:
    """Parse CLI for --config, validate config (vectordbpath, promptTemplate, questionsetfile, outputpath), run batch.
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

    db_path, promptTemplate, slm, model_name, _ = chatbot.legalos_rag.ensure_requirements(config)

    # Batch-specific config (not in ensure_requirements)
    questionsetfile = config.get("questionsetfile")
    outputpath = config.get("outputpath")
    if not outputpath:
        raise ValueError("Config must provide 'outputpath'")

    if not questionsetfile:
        raise ValueError("Config must provide 'questionsetfile'")

    outputpath= pathlib.Path(outputpath).resolve()
    questionsetfile = pathlib.Path(questionsetfile).resolve()

    prompt_run_batch(db_path=db_path, promptTemplate=promptTemplate["text"], questionsetfile=questionsetfile, slm=slm, model_name=model_name, outputpath=outputpath, new_pipeline=chatbot.legalos_rag.newPipeline.load_pipeline(config, slm))
           




if __name__ == "__main__":
    main()
