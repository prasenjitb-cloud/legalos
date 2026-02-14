"""
RAG pipeline: vector store retrieval, prompt building, SLM invocation, and run logging.

Used by chatbot.main (interactive loop) and test.promptTester.promptRunBatch (batch runs).
Exposes getFacts(), invoker(), and log_rag_run() for retrieval, generation, and logging.
"""
import json
import datetime
import pathlib

import langchain_core.output_parsers
import langchain_core.documents

import chatbot.legalos_rag.prompt.promptSchema
import chatbot.legalos_rag.prompt.prompts

import langchain_qdrant
import qdrant_client
import langchain_huggingface


# -------------------- GLOBAL VARIABLES --------------------

COLLECTION_NAME = "central_acts"           # Qdrant collection name for central acts embeddings
EMBEDDINGS_MODEL_NAME = "BAAI/bge-small-en"  # HuggingFace model for embedding queries and chunks

# -------------------- VECTOR STORE SETUP --------------------

def _setup_vectorstore(
    db_path: str,
    collection_name: str,
):
    """Build a Qdrant vectorstore with HuggingFace embeddings for the given DB path and collection.

    Args:
        db_path: Path to the Qdrant vector database (on-disk path).
        collection_name: Name of the Qdrant collection.

    Returns:
        langchain_qdrant.QdrantVectorStore: Vector store used for similarity search.
    """
    embeddings = langchain_huggingface.HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )
    client = qdrant_client.QdrantClient(path=db_path)
    vectorstore = langchain_qdrant.QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    return vectorstore


def _safe_json(obj):
    """Serialize a Python object to a JSON string with Unicode preserved (no ASCII escaping)."""
    return json.dumps(obj, ensure_ascii=False)


def _format_docs(docs: list[langchain_core.documents.Document]) -> str:
    """Turn a list of documents into a single string with [DOC n], text, and source metadata.

    Args:
        docs: List of LangChain Document objects (page_content + metadata).

    Returns:
        str: Formatted string suitable for the {facts} placeholder in the RAG prompt.
    """
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


def getFacts(q: str, db_path: str):
    """Retrieve top-k chunks for a query from the vector DB and return them as a formatted string.

    Args:
        q: User query string.
        db_path: Path to the Qdrant vector database directory.

    Returns:
        str: Formatted string of [DOC n] blocks (text + source metadata) for use in the RAG prompt.
    """
    vectorstore = _setup_vectorstore(
        db_path=db_path,
        collection_name=COLLECTION_NAME,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return _format_docs(retriever.invoke(q))


def invoker(
    slm,
    retrieved_docs,  # Formatted string from getFacts, or list[Document] (prompt.format accepts both)
    query: str,
    template: str,
):
    """Run the RAG pipeline: build prompt, invoke the SLM, and parse the response to LegalAnswer.

    Does not perform logging; callers (e.g. chatbot.main) call log_rag_run() after invoker().

    Args:
        slm: Small Language Model instance (e.g. ChatOllama).
        retrieved_docs: Formatted facts string (from getFacts) or list of Documents for {facts}.
        query: User question string.
        template: Full prompt template with {format_instructions}, {facts}, {question}.

    Returns:
        tuple: (parsed_result: LegalAnswer, final_prompt_text: str).
    """
    parser = langchain_core.output_parsers.PydanticOutputParser(
        pydantic_object=chatbot.legalos_rag.prompt.promptSchema.LegalAnswer
    )
    prompt = chatbot.legalos_rag.prompt.prompts.setup_rag_prompt_skeleton(
        parser,
        template,
    )

    # Build the final prompt string sent to the SLM (and optionally logged by the caller)
    final_prompt_text = prompt.format(
        facts=retrieved_docs,
        question=query,
    )
    raw_response = slm.invoke(final_prompt_text)

    # ChatOllama returns a message with .content; raw LLM returns a string
    raw_text = (
        raw_response.content
        if hasattr(raw_response, "content")
        else raw_response
    )
    parsed_result = parser.parse(raw_text)
    return parsed_result, final_prompt_text





# -------------------- LOG FILE --------------------

def log_rag_run(
    query: str,
    final_prompt: str,
    output: dict,
    model: str,
    log_file: pathlib.Path,
    exclude_model_name: bool,
    exclude_prompt: bool,
):
    """Append one RAG run as a single JSON line to the given log file (JSONL format).

    Args:
        query: User question that was answered.
        final_prompt: Full prompt text that was sent to the SLM.
        output: Parsed output (e.g. LegalAnswer.model_dump()) to store.
        model: Model name (e.g. SLM_MODEL_NAME) for reproducibility.
        log_file: Path to the JSONL log file; created or appended to.
        exclude_model_name: Whether to omit model name in log entries
        exclude_prompt: Whether to omit prompt in log entries
    """
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "model": model if not exclude_model_name else None,
        "query": query,
        "final_prompt": final_prompt if not exclude_prompt else None,
        "output": output,
    }
    with log_file.open("a", encoding="utf-8") as f:
        f.write(_safe_json(log_entry) + "\n")

