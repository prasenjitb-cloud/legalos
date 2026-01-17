from dotenv import load_dotenv
load_dotenv()

import argparse
import os
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import PromptTemplate


# -------------------- GLOBAL VARIABLES --------------------

COLLECTION_NAME = "state_acts"
EMBEDDINGS_MODEL_NAME = "BAAI/bge-small-en"
SLM_MODEL_NAME = "qwen2.5:3b-instruct"
FAILED_LOG_FILE= "failed_pdf_embeddings.txt"


# -------------------- VECTOR STORE SETUP --------------------

def setup_vectorstore(
    db_path: str ,
    collection_name: str ,
):
    embeddings = HuggingFaceEmbeddings(
        model_name= EMBEDDINGS_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )

    client = QdrantClient(path=db_path)

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    return vectorstore


# -------------------- LLM SETUP --------------------

def setup_llm():
    return ChatOllama(
        model= SLM_MODEL_NAME,
        temperature=0,
    )


# -------------------- PROMPT --------------------

def setup_prompt():
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a legal assistant.

Answer ONLY using the provided legal documents.
If the answer is not present, say "Not found in the documents".
Mention the Act name and Section number if available.

Context:
{context}

Question:
{question}

Answer:
"""
    )


# -------------------- MAIN RAG LOOP --------------------

def run_rag(db_path: str):
    vectorstore = setup_vectorstore(
        db_path=db_path,
        collection_name=COLLECTION_NAME,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = setup_llm()
    prompt = setup_prompt()

    while True:
        q = input("\nAsk a legal question (type 'exit' to quit): ").strip()

        if q.lower() in {"exit", "quit", "q"}:
            print("Exiting.")
            break

        if not q:
            print("Empty question. Try again.")
            continue

        retrieved_docs = retriever.invoke(q)

        if not retrieved_docs:
            print("\nAnswer:\n Not found in the documents")
            continue

        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        final_prompt = prompt.invoke(
            {"context": context_text, "question": q}
        )

        answer = llm.invoke(final_prompt)
        print("\nAnswer:\n", answer.content)




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
