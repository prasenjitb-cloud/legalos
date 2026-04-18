import os
import json
import argparse

import chatbot.legalos_rag.runRag
import chatbot.legalos_rag.queryRewriter
import pathlib
import chatbot.legalos_rag

# -------------------- SINGLE RAG INVOCATION --------------------

def run_rag(query: str, db_path: str, prompt_template: str, slm) -> tuple:
    """
    Perform one RAG run: retrieve relevant chunks, generate answer, and return result.

    Retrieves top-k document chunks from the vector database for the given query,
    runs the RAG invoker (generate + parse), and returns the structured result,
    the retrieved chunks, the final prompt used, and the rewritten query variants.

    Args:
        query: The user's legal question.
        db_path: Path to the Qdrant vector database.
        prompt_template: Full prompt template string for the RAG invoker.
        slm: Small Language Model instance.

    Returns:
        (result, retrieved_chunks, final_prompt, queries) when chunks were found.
        (None, [], None, queries) when no relevant chunks were retrieved.
    """

    # Step 1: rewrite query into legal terms and generate alternative phrasing
    queries = chatbot.legalos_rag.queryRewriter.rewrite_and_expand(query, slm)

    # Step 2: retrieve chunks for all query variants (deduplicated by pdf_number + page)
    retrieved_chunks = chatbot.legalos_rag.runRag.getFactsMulti(
        queries=queries,
        db_path=db_path,
    )
    if not retrieved_chunks:
        return (None, [], None, queries)

    # Step 3: generate answer using the original query (more natural for the SLM)
    result, final_prompt = chatbot.legalos_rag.runRag.invoker(
        slm,
        retrieved_chunks,
        query,
        prompt_template,
    )
    return (result, retrieved_chunks, final_prompt, queries)


# -------------------- INTERACTIVE QUESTIONING --------------------

def run_rag_loop(
    db_path: str,
    prompt_template: str,
    model_name: str,
    logfile: str,
    exclude_model_name: bool,
    exclude_prompt: bool,
    slm,
) -> None:
    """
    Run the interactive legal Q&A session: prompt for questions, invoke RAG, log, and print.

    Enters a loop that repeatedly asks for a legal question, runs a single RAG
    invocation via run_rag(), logs the run to the configured JSONL file, and
    displays the answer and citations. Handles exit commands and empty input.
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

        result, _, final_prompt, _ = run_rag(query, db_path, prompt_template, slm)

        if result is None:
            print("\nAnswer:\n Not found in the documents")
            continue

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

    # Validate config and initialize SLM (returns 5 values)
    db_path, prompt_template, slm, model_name, logging = chatbot.legalos_rag.ensure_requirements(config)

    # -------------------- RUN --------------------
    # Kick off the interactive RAG loop with resolved configuration.
    run_rag_loop(
        db_path=db_path,
        prompt_template=prompt_template["text"],
        slm=slm,
        model_name=model_name,
        logfile=logging.logfile,
        exclude_model_name=logging.exclude_model_name,
        exclude_prompt=logging.exclude_prompt,
    )


if __name__ == "__main__":
    main()
