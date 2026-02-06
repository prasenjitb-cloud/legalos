import json
import datetime
import pathlib


def safe_json(obj):
    return json.dumps(obj, ensure_ascii=False)

# -------------------- LOG FILE --------------------

def log_rag_run(
    query: str,
    final_prompt: str,
    output: dict,
    model: str,
    log_file: pathlib.Path,
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

    with log_file.open("a", encoding="utf-8") as f:
        f.write(safe_json(log_entry) + "\n")

