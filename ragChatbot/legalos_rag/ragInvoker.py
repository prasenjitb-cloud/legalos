import langchain_core.output_parsers 
import langchain_core.documents 

import legalos_rag.promptSchema 
import legalos_rag.prompts

import json
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("rag_runs.jsonl")

# -------------------- LOG FILE --------------------

def log_rag_run(
    query: str,
    final_prompt: str,
    output: str,
    model: str,
):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": model,
        "query": query,
        "final_prompt": final_prompt,
        "output": output,
    }

    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def safe_json(obj):
    try:
        return json.loads(json.dumps(obj))
    except Exception:
        return str(obj)



def invoker(
        llm,
        retrieved_docs: list[langchain_core.documents.Document],
        q: str,
        model: str
):
    parser = langchain_core.output_parsers.PydanticOutputParser(pydantic_object=legalos_rag.promptSchema.LegalAnswer)

    context= retrieved_docs

    prompt = legalos_rag.prompts.setup_prompt(parser)

    # Render final prompt text (for logging)
    final_prompt_text = prompt.format(
        context=context,
        question=q
    )

    raw_response = llm.invoke(final_prompt_text)

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
        query=q,
        final_prompt=final_prompt_text,
        output=safe_json(parsed_result.model_dump()),
        model= model
    )

    return parsed_result



