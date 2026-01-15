import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


# -------------------- DB SETUP --------------------

def setup_vector_db(
    db_path: str = "./DB",
    collection_name: str = "state_acts",
):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en",
        encode_kwargs={"normalize_embeddings": True},
    )

    client = QdrantClient(path=db_path)

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    return vectorstore


# -------------------- PDF INGESTION --------------------

def ingest_pdfs_from_dir(
    pdf_dir: str,
    vectorstore: QdrantVectorStore,
    failed_log: str = "failed_pdfs.txt",
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    pdf_counter = 0
    open(failed_log, "w").close()

    for filename in sorted(os.listdir(pdf_dir)):
        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(pdf_dir, filename)
        pdf_counter += 1
        print(f"Processing PDF #{pdf_counter}: {filename}")

        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            docs = [
                d for d in docs
                if "ARRANGEMENT OF SECTIONS" not in d.page_content.upper()
            ]

            splits = splitter.split_documents(docs)

            enriched_docs = [
                Document(
                    page_content=d.page_content,
                    metadata={
                        **d.metadata,
                        "doc_type": "state_act",
                        "source": "indiacode",
                        "file_name": filename,
                        "pdf_number": pdf_counter,
                    },
                )
                for d in splits
            ]

            vectorstore.add_documents(enriched_docs)

        except Exception as e:
            with open(failed_log, "a") as f:
                f.write(
                    f"PDF #{pdf_counter} | FILE: {filename} | ERROR: {str(e)}\n"
                )
            print(f"❌ Failed PDF #{pdf_counter}")
            continue


# -------------------- MAIN --------------------

def main():
    pdf_directory = "./stateActPdfs/data"

    vectorstore = setup_vector_db(
        db_path="./DB",
        collection_name="state_acts",
    )

    ingest_pdfs_from_dir(
        pdf_dir=pdf_directory,
        vectorstore=vectorstore,
        failed_log="failed_pdf_embeddings.txt",
    )


if __name__ == "__main__":
    main()
