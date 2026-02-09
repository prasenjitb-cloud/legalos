import langchain_ollama
import os
import pathlib

SLM_MODEL_NAME = "qwen2.5:3b-instruct"


# -------------------- SLM SETUP --------------------

def _setup_slm(model_name: str):
    """
    Set up and return a SLM instance using Ollama.
    
    This function initializes a ChatOllama instance configured with the specified
    SLM model (qwen2.5:3b-instruct) and temperature setting. The SLM is used
    for generating responses in the RAG system.
    
    Returns:
        langchain_ollama.ChatOllama: Configured SLM instance ready for use
    """
    return langchain_ollama.ChatOllama(
        model= model_name,
        temperature=1,
    )


def ensure_requirements(config: dict):

    vectordbpath = config.get("vectordbpath")
    promptTemplate = config.get("promptTemplate")
    model_name = (config.get("model") or {}).get("model_name")

    if not model_name:
        raise ValueError(
            "Config missing required key 'model_name' (Ollama model name, e.g. 'qwen2.5:3b-instruct'). "
            "Add it to your JSON config file."
        )

    if not vectordbpath:
        raise ValueError(
            "Config missing required key 'vectordbpath' (path to the Qdrant vector DB, e.g. \"./vectorDB\"). "
            "Add it to your JSON config file."
        )

    if not promptTemplate:
        raise ValueError(
            "Config missing required key 'promptTemplate' (object with a 'text' key containing the prompt template). "
            "Add it to your JSON config file."
        )

    # Normalize vector DB path to absolute
    db_path = os.path.abspath(vectordbpath)


    if not os.path.isdir(db_path):
        raise ValueError(
            f"Vector DB path is missing or not a directory: {db_path!r}. "
            "Create the vector DB first (e.g. run vectorDbSetup) or fix the path in config."
        )

    slm = _setup_slm(model_name)


    return db_path, promptTemplate, slm, model_name