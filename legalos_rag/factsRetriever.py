import langchain_qdrant 
import qdrant_client 
import langchain_huggingface 
import langchain_core.documents 


# -------------------- GLOBAL VARIABLES --------------------

COLLECTION_NAME = "central_acts"
EMBEDDINGS_MODEL_NAME = "BAAI/bge-small-en"

# -------------------- VECTOR STORE SETUP --------------------

def setup_vectorstore(
    db_path: str ,
    collection_name: str ,
):
    """Build a Qdrant vectorstore with HuggingFace embeddings for the given DB path and collection.
    
    Args:
        db_path: Path to the Qdrant vector database
        collection_name: Name of the Qdrant collection
    """
    embeddings = langchain_huggingface.HuggingFaceEmbeddings(
        model_name= EMBEDDINGS_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )

    client = qdrant_client.QdrantClient(path=db_path)

    vectorstore = langchain_qdrant.QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    return vectorstore


def format_docs(docs: list[langchain_core.documents.Document]) -> str:
    """Turn a list of documents into a single string with [DOC n], text, and source metadata.
    
    Args:
        docs: List of documents to format

    Returns:
        str: Formatted string of documents
    """
    blocks = []
    for i, doc in enumerate[langchain_core.documents.Document](docs, 1):
        meta = doc.metadata
        blocks.append(
            f"""[DOC {i}]
TEXT:
{doc.page_content}

SOURCE:
pdf_number: {meta.get("pdf_number")}
page: {meta.get("page")}
file_name: {meta.get("file_name")}
"""
        )
    return "\n\n".join(blocks)


def getFacts(
        q:str,
        db_path: str
):
    """Return top-k retrieved chunks for query q from the vector DB, formatted as a single string.
    
    Args:
        q: Query string
        db_path: Path to the Qdrant vector database

    Returns:
        str: Formatted string of documents
    """
    vectorstore = setup_vectorstore(
        db_path=db_path,
        collection_name=COLLECTION_NAME,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return format_docs(retriever.invoke(q))