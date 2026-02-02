import langchain_core.output_parsers 
import langchain_core.documents 

import chatbot.legalos_rag.prompt.promptSchema 
import chatbot.legalos_rag.prompt.prompts

import json
import datetime
import pathlib

LOG_FILE = pathlib.Path("rag_runs.jsonl")

# -------------------- LOG FILE --------------------

def log_rag_run(
    query: str,
    final_prompt: str,
    output: str,
    model: str,
):
    """Append one RAG run (query, prompt, output, model) as a JSON line to the log file.
    
    Args:
        query: Query string
        final_prompt: Final prompt text
        output: Output JSON string
        model: Model name
    """

    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "model": model,
        "query": query,
        "final_prompt": final_prompt,
        "output": output,
    }

    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def safe_json(obj):
    """Return a JSON-serializable copy of obj, or its string form if serialization fails.
    
    Args:
        obj: Object to convert to JSON
    
    Returns:
        str: JSON string of obj
    """
    try:
        return json.loads(json.dumps(obj))
    except Exception:
        return str(obj)



def invoker(
        slm,
        retrieved_docs: list[langchain_core.documents.Document],
        query: str,
        model: str,
        prompt: str,
        templates_path: str | None,
):
    """Run the RAG pipeline: format prompt with docs and query, invoke the SLM, parse to LegalAnswer, and log the run.
    
    Args:
        slm: Small Language Model instance
        retrieved_docs: List of documents retrieved from the vector database
        query: Query string
        model: Model name
        prompt: Prompt identifier/version 
        templates_path: Path to prompt templates

    
    Returns:
        chatbot.legalos_rag.prompt.promptSchema.LegalAnswer: Parsed result
    """
    
    parser = langchain_core.output_parsers.PydanticOutputParser(pydantic_object=chatbot.legalos_rag.prompt.promptSchema.LegalAnswer)


    prompt = chatbot.legalos_rag.prompt.prompts.setup_rag_prompt_skeleton(parser, prompt, templates_path)

    # Render final prompt text (for logging)
    final_prompt_text = prompt.format(
        facts=retrieved_docs,
        question=query
    )

    raw_response = slm.invoke(final_prompt_text)

    # Handle ChatModel vs LLM output
    raw_text = (
        raw_response.content
        if hasattr(raw_response, "content")
        else raw_response
    )

    # Parse into schema
    parsed_result = parser.parse(raw_text)

    # Log everything you want

    log_rag_run(
        query=query,
        final_prompt=final_prompt_text,
        output=safe_json(parsed_result.model_dump()),
        model= model
    )

    return parsed_result



