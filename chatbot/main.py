import dotenv 
dotenv.load_dotenv()

import os
import json
import argparse

import langchain_ollama
import pathlib
import chatbot.legalos_rag.factsRetriever 
import chatbot.legalos_rag.ragInvoker 
import chatbot.legalos_rag.logger

# -------------------- GLOBAL VARIABLES --------------------

FAILED_LOG_FILE= "failed_pdf_embeddings.txt"

# -------------------- SLM SETUP --------------------

def setup_slm(model_name: str):
    """
    Set up and return a SLM instance using Ollama.
    
    This function initializes a ChatOllama instance configured with the specified
    SLM model and temperature setting. The SLM is used for generating responses
    in the RAG system.
    
    Args:
        model_name: Ollama model name (e.g. from config model.model_name).
    
    Returns:
        langchain_ollama.ChatOllama: Configured SLM instance ready for use
    """
    return langchain_ollama.ChatOllama(
        model=model_name,
        temperature=1,
    )

# -------------------- MAIN RAG LOOP --------------------

def run_rag(db_path: str, template: str, model_name: str, logfile: str, exclude_model_name: bool, exclude_prompt: bool):
    """
    Run the interactive RAG system for legal question answering.
    
    Sets up the SLM and enters an interactive loop that retrieves relevant
    documents from the vector database and generates answers with citations.
    
    Args:
        db_path: Path to the Qdrant vector database
        template: Full prompt template string to pass to the RAG invoker
        model_name: Ollama model name (from config model.model_name)
        logfile: Path to the RAG run log file
        exclude_model_name: Whether to omit model name in log entries
        exclude_prompt: Whether to omit prompt in log entries
    """
    slm = setup_slm(model_name)

    while True:
        query = input("\nAsk a legal question (type 'exit' to quit): ").strip()

        if query.lower() in {"exit", "quit", "q"}:
            print("Exiting.")
            break

        if not query:
            print("Empty question. Try again.")
            continue

        # Retrieve relevant documents from vector database using local module legalos_rag.factsRetriever
        retrieved_docs = chatbot.legalos_rag.factsRetriever.getFacts(
            q=query,
            db_path=db_path
        )

        if not retrieved_docs:
            print("\nAnswer:\n Not found in the documents")
            continue

        # Generate RAG answer using local module legalos_rag.ragInvoker
        [result, final_prompt]= chatbot.legalos_rag.ragInvoker.invoker(
            slm,
            retrieved_docs,
            query,
            template,
        )

        chatbot.legalos_rag.logger.log_rag_run(
            query=query,
            final_prompt=final_prompt,
            output=result.model_dump(),
            model=model_name,
            log_file=logfile,
            exclude_model_name=exclude_model_name,
            exclude_prompt=exclude_prompt,
        )




        if not result.answer_found:
            print("Not found in the documents.")
        else:
            print("SLM")
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

    config_path = os.path.abspath(args.config)

    if not os.path.isfile(config_path):
        raise ValueError(f"Config file does not exist: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config: dict = json.load(f)

    # Required keys in the config:
    #   - vectordbpath: path to the Qdrant vector database
    #   - template: full prompt template string
    #   - model.model_name: Ollama model name for the SLM
    #   - logging: { logfile, exclude_model_name, exclude_prompt }
    vectordbpath = config.get("vectordbpath")
    template = config.get("template")
    model_name = (config.get("model") or {}).get("model_name")
    logging_cfg = config.get("logging") or {}
    logfile_val = logging_cfg.get("logfile")
    exclude_model_name = logging_cfg.get("exclude_model_name")
    exclude_prompt = logging_cfg.get("exclude_prompt")
    if not vectordbpath:
        raise ValueError("Config must provide 'vectordbpath'")

    if not isinstance(template, str) or not template.strip():
        raise ValueError("'template' must be a non-empty string")

    if not model_name:
        raise ValueError("Config must provide 'model.model_name'")

    if not logfile_val:
        raise ValueError("Config must provide 'logging.logfile'")
    logfile = pathlib.Path(logfile_val).resolve()

    if exclude_model_name is None:
        raise ValueError("Config must provide 'logging.exclude_model_name'")

    if exclude_prompt is None:
        raise ValueError("Config must provide 'logging.exclude_prompt'")

    # Normalize vector DB path to absolute
    db_path = os.path.abspath(vectordbpath)

    # Check that the vector DB path is a directory
    if not os.path.isdir(db_path):
        raise ValueError(f"Vector DB path does not exist: {db_path}")
    if not os.path.isfile(logfile):
        raise ValueError(f"Log file does not exist: {logfile}")

    # -------------------- RUN --------------------
    # Kick off the interactive RAG loop with resolved configuration.
    run_rag(
        db_path=db_path,
        template=template,
        model_name=model_name,
        logfile=logfile,
        exclude_model_name=exclude_model_name,
        exclude_prompt=exclude_prompt,
    )


if __name__ == "__main__":
    main()
