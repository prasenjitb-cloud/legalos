from langchain_core.output_parsers import PydanticOutputParser
from .promptSchema import LegalAnswer
from langchain_core.documents import Document
from .contextRetriever import format_docs
from .prompts import setup_prompt
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



def chainInvoker(
        llm,
        retrieved_docs: list[Document],
        q: str,
        model: str
):
    parser = PydanticOutputParser(pydantic_object=LegalAnswer)

    context= format_docs(retrieved_docs)

    prompt = setup_prompt(parser)

    # 1️⃣ Render final prompt text (for logging)
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

    # 3️⃣ Parse into schema
    parsed_result = parser.parse(raw_text)

    # 4️⃣ Log everything you want
    log_rag_run(
        query=q,
        final_prompt=final_prompt_text,
        output=safe_json(parsed_result.model_dump()),
        model= model
    )

    return parsed_result



