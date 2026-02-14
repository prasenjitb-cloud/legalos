import os
import json
import argparse

import chatbot.legalos_rag.runRag
import pathlib
import chatbot.legalos_rag

# -------------------- GLOBAL VARIABLES --------------------



# -------------------- MAIN RAG LOOP --------------------

def run_rag(db_path: str, promptTemplate: str, model_name: str, logfile: str, exclude_model_name: bool, exclude_prompt: bool, slm):
    """
    Run the interactive RAG system for legal question answering.
    
    Enters an interactive loop that retrieves relevant
    documents from the vector database and generates answers with citations.
    
    Args:
        db_path: Path to the Qdrant vector database
        promptTemplate: Full prompt template string to pass to the RAG invoker
        model_name: Ollama model name (from config model.model_name)
        logfile: Path to the RAG run log file
        exclude_model_name: Whether to omit model name in log entries
        exclude_prompt: Whether to omit prompt in log entries
        slm: Small Language Model instance
    """

    while True:
        query = input("\nAsk a legal question (type 'exit' to quit): ").strip()

        # Check for exit commands
        if query.lower() in {"exit", "quit", "q"}:
            print("Exiting.")
            break

        if not query:
            print("Empty question. Try again.")
            continue

        # Retrieve top-k relevant document chunks from vector database
        retrieved_docs = chatbot.legalos_rag.runRag.getFacts(
            q=query,
            db_path=db_path
        )

        if not retrieved_docs:
            print("\nAnswer:\n Not found in the documents")
            continue

        # Generate structured answer using RAG pipeline (retrieve + generate + parse)
        [result, final_prompt]= chatbot.legalos_rag.runRag.invoker(
            slm,
            retrieved_docs,
            query,
            promptTemplate,
        )

        # Log the query, prompt, and response to JSONL file
        chatbot.legalos_rag.runRag.log_rag_run(
            query=query,
            final_prompt=final_prompt,
            output=result.model_dump(),
            model=model_name,
            log_file=logfile,
            exclude_model_name=exclude_model_name,
            exclude_prompt=exclude_prompt,
        )

        # Display results to user
        if not result.answer_found:
            print("Not found in the documents.")
        else:
            print("SLM Response:")
            print(result.explanation)
            for c in result.citations:
                print(c.page, c.quote)



# -------------------- ENTRY POINT --------------------
def main():
    """
    Parse configuration from a JSON config file and start the RAG loop.
    """

    parser = argparse.ArgumentParser(
        description="Run Legalos RAG system (config-only)"
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to JSON config file with vectordbpath and template",
    )

    args = parser.parse_args()

    # Resolve config file path to absolute path
    config_path = pathlib.Path(args.config).resolve()

    if not config_path.is_file():
        raise ValueError(f"Config file does not exist: {config_path}")

    # Load configuration from JSON file
    with config_path.open("r", encoding="utf-8") as f:
        config: dict = json.load(f)

    # Validate config and initialize SLM (returns 4 values)
    db_path, promptTemplate, slm , model_name= chatbot.legalos_rag.ensure_requirements(config)

    # Extract logging configuration
    logging_cfg = config.get("logging") or {}
    logfile_val = logging_cfg.get("logfile")
    exclude_model_name = logging_cfg.get("exclude_model_name")
    exclude_prompt = logging_cfg.get("exclude_prompt")

    # Validate logging configuration
    if not logfile_val:
        raise ValueError("Config must provide 'logging.logfile'")

    logfile = pathlib.Path(logfile_val).resolve()

    if exclude_model_name is None:
        raise ValueError("Config must provide 'logging.exclude_model_name'")

    if exclude_prompt is None:
        raise ValueError("Config must provide 'logging.exclude_prompt'")

    # Ensure log file exists
    if not os.path.isfile(logfile):
        raise ValueError(f"Log file does not exist: {logfile}")


    # -------------------- RUN --------------------
    # Kick off the interactive RAG loop with resolved configuration.
    run_rag(
        db_path=db_path,
        promptTemplate=promptTemplate["text"],
        slm=slm,
        model_name=model_name,
        logfile=logfile,
        exclude_model_name=exclude_model_name,
        exclude_prompt=exclude_prompt,
    )


if __name__ == "__main__":
    main()
