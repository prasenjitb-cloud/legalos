import dotenv as _dotenv
_dotenv.load_dotenv()

import argparse
import os

import langchain_ollama as _langchain_ollama
import langchain_groq as _langchain_groq

import utils.contextRetriever as _contextRetriever
import utils.chainInvoker as _chainInvoker

# alias back to original names
ChatOllama = _langchain_ollama.ChatOllama
ChatGroq = _langchain_groq.ChatGroq
getContext = _contextRetriever.getContext
chainInvoker = _chainInvoker.chainInvoker

# -------------------- GLOBAL VARIABLES --------------------

SLM_MODEL_NAME = "qwen2.5:3b-instruct"
COMPARITOR_MODEL_NAME = "gemini-2.0-flash-lite"
FAILED_LOG_FILE= "failed_pdf_embeddings.txt"


# -------------------- LLM SETUP --------------------

def setup_llm():
    return ChatOllama(
        model= SLM_MODEL_NAME,
        temperature=1,
    )
# -------------------- COMPARITOR LLM SETUP --------------------

def setup_comparitor():
    return  ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)


# -------------------- MAIN RAG LOOP --------------------

def run_rag(db_path: str):

    llm = setup_llm()
    comp = setup_comparitor()

    while True:
        q = input("\nAsk a legal question (type 'exit' to quit): ").strip()

        if q.lower() in {"exit", "quit", "q"}:
            print("Exiting.")
            break

        if not q:
            print("Empty question. Try again.")
            continue

        retrieved_docs = getContext(
            q=q,
            db_path=db_path
        )

        if not retrieved_docs:
            print("\nAnswer:\n Not found in the documents")
            continue

        result1= chainInvoker(llm,retrieved_docs,q,SLM_MODEL_NAME)
        result2=chainInvoker(comp,retrieved_docs,q,COMPARITOR_MODEL_NAME)



        if not result1.answer_found:
            print("Not found in the documents.")
        else:
            print("SLM")
            print(result1.explanation)
            for c in result1.citations:
                print(c.page, c.quote)

        print("")

        if not result2.answer_found:
            print("Not found in the documents.")
        else:
            print("LLM")
            print(result2.explanation)
            for c in result2.citations:
                print(c.page, c.quote)




# -------------------- ENTRY POINT --------------------
def main():
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

    run_rag(db_path=db_path)


if __name__ == "__main__":
    main()
