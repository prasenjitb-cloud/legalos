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

SLM_MODEL_NAME = "qwen2.5:3b-instruct"
FAILED_LOG_FILE= "failed_pdf_embeddings.txt"
LOG_FILE = pathlib.Path("chatbot/rag_runs.jsonl")

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

def run_rag(db_path: str, template: str):
    """
    Run the interactive RAG system for legal question answering.
    
    Sets up the SLM and enters an interactive loop that retrieves relevant
    documents from the vector database and generates answers with citations.
    
    Args:
        db_path: Path to the Qdrant vector database
        template: Full prompt template string to pass to the RAG invoker
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
            model=SLM_MODEL_NAME,
            log_file=LOG_FILE,
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
    vectordbpath = config.get("vectordbpath")
    template = config.get("template")

    if not vectordbpath:
        raise ValueError("Config must provide 'vectordbpath'")

    if not template:
        raise ValueError("Config must provide 'template'")

    # Normalize vector DB path to absolute
    db_path = os.path.abspath(vectordbpath)

    # Check that the vector DB path is a directory
    if not os.path.isdir(db_path):
        raise ValueError(f"Vector DB path does not exist: {db_path}")

    # -------------------- RUN --------------------
    # Kick off the interactive RAG loop with resolved configuration.
    run_rag(
        db_path=db_path,
        template=template,
    )


if __name__ == "__main__":
    main()
