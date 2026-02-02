import dotenv 
dotenv.load_dotenv()

import argparse
import os
import json

import langchain_ollama

import chatbot.legalos_rag.factsRetriever 
import chatbot.legalos_rag.ragInvoker 

# -------------------- GLOBAL VARIABLES --------------------

SLM_MODEL_NAME = "qwen2.5:3b-instruct"
FAILED_LOG_FILE= "failed_pdf_embeddings.txt"


# -------------------- SLM SETUP --------------------

def setup_slm():
    """
    Set up and return a SLM instance using Ollama.
    
    This function initializes a ChatOllama instance configured with the specified
    SLM model (qwen2.5:3b-instruct) and temperature setting. The SLM is used
    for generating responses in the RAG system.
    
    Returns:
        langchain_ollama.ChatOllama: Configured SLM instance ready for use
    """
    return langchain_ollama.ChatOllama(
        model= SLM_MODEL_NAME,
        temperature=1,
    )

# -------------------- MAIN RAG LOOP --------------------

def run_rag(db_path: str, prompt: str, templates_path: str | None):
    """
    Run the interactive RAG system for legal question answering.
    
    Sets up the SLM and enters an interactive loop that retrieves relevant
    documents from the vector database and generates answers with citations.
    
    Args:
        db_path: Path to the Qdrant vector database
        prompt: Prompt identifier/version to pass to the RAG invoker
        templates_path: Path to prompt templates to pass to the RAG invoker
    """
    slm = setup_slm()

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
        result= chatbot.legalos_rag.ragInvoker.invoker(
            slm,
            retrieved_docs,
            query,
            SLM_MODEL_NAME,
            prompt,
            templates_path,
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
    Parse configuration (CLI + optional JSON config file) and start the RAG loop.

    Precedence for configuration is:
        1. CLI flags
        2. JSON config file (if provided via --config)
    """

    # Argument parser for CLI usage
    parser = argparse.ArgumentParser(
        description="Run Legalos RAG system"
    )

    # Optional JSON config file with keys:
    #   - vectordbpath
    #   - prompt
    #   - templatespath
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON config file with run parameters",
    )

    # Vector DB path (can also come from config)
    parser.add_argument(
        "--vectordbpath",
        type=str,
        help="Path to the Qdrant vector database",
    )

    # Prompt version / identifier (e.g. v1, v2)
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt version to use (e.g. v1, v2)",
    )

    # Path to prompt templates JSON (can also come from config)
    parser.add_argument(
        "--templatespath",
        type=str,
        help="Path to prompt templates JSON",
    )

    args = parser.parse_args()

    # -------------------- LOAD CONFIG --------------------
    # If a config file is provided, load it as a simple JSON dict.
    config: dict = {}
    if args.config:
        config_path = os.path.abspath(args.config)
        if not os.path.isfile(config_path):
            raise ValueError(f"Config file does not exist: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    # -------------------- MERGE (CLI > CONFIG ) --------------------
    # Resolve each setting in order of precedence:
    #   CLI flag > config file value
    vectordbpath = (
        args.vectordbpath
        or config.get("vectordbpath")
    )

    prompt = (
        args.prompt
        or config.get("prompt")
    )

    templatespath = (
        args.templatespath
        or config.get("templatespath")
    )

    # Basic validation of required values
    if not vectordbpath:
        raise ValueError(
            "vectordbpath must be provided via CLI or config"
        )

    if not prompt:
        raise ValueError(
            "prompt must be provided via CLI or config"
        )

    if not templatespath:
        raise ValueError(
            "templatespath must be provided via CLI or config"
        )

    # Normalize paths to absolute
    db_path = os.path.abspath(vectordbpath)
    templates_path = os.path.abspath(templatespath)

    # Check that the vector DB path is a directory
    if not os.path.isdir(db_path):
        raise ValueError(f"Vector DB path does not exist: {db_path}")

    # Check that the templates path points to a file
    if not os.path.isfile(templates_path):
        raise ValueError(f"Templates file does not exist: {templates_path}")

    # -------------------- RUN --------------------
    # Kick off the interactive RAG loop with resolved configuration.
    run_rag(
        db_path=db_path,
        prompt=prompt,
        templates_path=templates_path
    )


if __name__ == "__main__":
    main()
