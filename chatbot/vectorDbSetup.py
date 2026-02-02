import os

import dotenv
dotenv.load_dotenv()

import argparse

import langchain_community.document_loaders
import langchain_text_splitters 
import langchain_core.documents 
import langchain_huggingface 
import langchain_qdrant 
import qdrant_client 
import qdrant_client.http.models 

# -------------------- GLOBAL VARIABLES --------------------

COLLECTION_NAME = "central_acts"
MODEL_NAME = "BAAI/bge-small-en"
FAILED_LOG_FILE= "failed_pdf_embeddings.txt"

# -------------------- DB SETUP --------------------

def setup_vector_db(
    db_path: str ,
    collection_name: str ,
):
    """Create a Qdrant collection and return a vectorstore with HuggingFace embeddings.
    
    Args:
        db_path: Path to the Qdrant vector database
        collection_name: Name of the Qdrant collection
    """


    embeddings = langchain_huggingface.HuggingFaceEmbeddings(
        model_name= MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )

    client = qdrant_client.QdrantClient(path=db_path)

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config= qdrant_client.http.models.VectorParams(
            size=384,
            distance=qdrant_client.http.models.Distance.COSINE,
        ),
    )

    vectorstore = langchain_qdrant.QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    return vectorstore


# -------------------- PDF INGESTION --------------------

def ingest_pdfs_from_dir(
    pdf_dir: str,
    vectorstore: langchain_qdrant.QdrantVectorStore,
    failed_log: str ,
):
    """Load PDFs from a directory, chunk them, and add embeddings to the vectorstore. 
    Log failures to failed_log.
    
    Args:
        pdf_dir: Directory containing Act PDFs to ingest
        vectorstore: Qdrant vector store to add embeddings to
        failed_log: File to log failed PDF ingestion
    """

    
    splitter = langchain_text_splitters.RecursiveCharacterTextSplitter(
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
            loader = langchain_community.document_loaders.PyPDFLoader(pdf_path)
            docs = loader.load()

            docs = [
                d for d in docs
                if "ARRANGEMENT OF SECTIONS" not in d.page_content.upper()
            ]

            splits = splitter.split_documents(docs)

            enriched_docs = [
                langchain_core.documents.Document(
                    page_content=d.page_content,
                    metadata={
                        **d.metadata,
                        "doc_type": COLLECTION_NAME,
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
    """Parse CLI args, set up the vector DB, and ingest PDFs from the given directory.
    
    Args:
        pdf_dir: Directory containing Act PDFs to ingest
        db_dir: Directory where Qdrant vector DB will be created
        
    Raises:
        ValueError: If the PDF directory does not exist
    """
    parser = argparse.ArgumentParser(
        description="Ingest PDFs into Qdrant vector database"
    )

    parser.add_argument(
        "--pdfActsDirectory",
        required=True,
        type=str,
        help="Directory containing Act PDFs to ingest"
    )

    parser.add_argument(
        "--vectordbDirectory",
        required=True,
        type=str,
        help="Directory where Qdrant vector DB will be created"
    )

    args = parser.parse_args()

    pdf_dir = os.path.abspath(args.pdfActsDirectory)
    db_dir = os.path.abspath(args.vectordbDirectory)

    if not os.path.isdir(pdf_dir):
        raise ValueError(f"PDF directory does not exist: {pdf_dir}")

    vectorstore = setup_vector_db(
        db_path=db_dir,
        collection_name= COLLECTION_NAME,
    )

    ingest_pdfs_from_dir(
        pdf_dir=pdf_dir,
        vectorstore=vectorstore,
        failed_log=os.path.join(os.path.dirname(pdf_dir), FAILED_LOG_FILE),
    )
    
if __name__ == "__main__":
    main()
