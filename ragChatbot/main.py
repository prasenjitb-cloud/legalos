import dotenv 
dotenv.load_dotenv()

import argparse
import os

import langchain_ollama

import legalos_rag.factsRetriever 
import legalos_rag.ragInvoker 

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

def run_rag(db_path: str):
    """
    Run the interactive RAG system for legal question answering.
    
    Sets up the SLM and enters an interactive loop that retrieves relevant
    documents from the vector database and generates answers with citations.
    
    Args:
        db_path: Path to the Qdrant vector database
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
        retrieved_docs = legalos_rag.factsRetriever.getFacts(
            q=query,
            db_path=db_path
        )

        if not retrieved_docs:
            print("\nAnswer:\n Not found in the documents")
            continue

        # Generate RAG answer using local module legalos_rag.ragInvoker
        result= legalos_rag.ragInvoker.invoker(slm,retrieved_docs,query,SLM_MODEL_NAME)



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
    Parse CLI args and run the RAG system.
    
    Args:
        vectordbpath: Path to the Qdrant vector database
    
    Raises:
        ValueError: If the vector database path does not exist
    """

    parser = argparse.ArgumentParser(
        description="Run Legalos RAG system"
    )

    parser.add_argument(
        "--vectordbpath",
        required=True,
        type=str,
        help="Path to the Qdrant vector database"
    )

    args = parser.parse_args()

    db_path = os.path.abspath(args.vectordbpath)

    if not os.path.isdir(db_path):
        raise ValueError(f"Vector database path does not exist: {db_path}")
    
    # Run the RAG system
    run_rag(db_path=db_path)


if __name__ == "__main__":
    main()
