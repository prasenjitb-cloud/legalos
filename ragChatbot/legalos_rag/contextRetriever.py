import langchain_qdrant 
import qdrant_client 
import langchain_huggingface 
import langchain_core.documents 


# -------------------- GLOBAL VARIABLES --------------------

COLLECTION_NAME = "state_acts"
EMBEDDINGS_MODEL_NAME = "BAAI/bge-small-en"

# -------------------- VECTOR STORE SETUP --------------------

def setup_vectorstore(
    db_path: str ,
    collection_name: str ,
):
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
    blocks = []
    for i, doc in enumerate(docs, 1):
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


def getContext(
        q:str,
        db_path: str
):
    vectorstore = setup_vectorstore(
        db_path=db_path,
        collection_name=COLLECTION_NAME,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return format_docs(retriever.invoke(q))