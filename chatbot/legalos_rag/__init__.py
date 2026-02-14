import langchain_ollama
import pathlib


# -------------------- SLM SETUP --------------------

def _setup_slm(model_name: str):
    """
    Set up and return a SLM instance using Ollama.
    
    This function initializes a ChatOllama instance configured with the specified
    SLM model (qwen2.5:3b-instruct) and temperature setting. The SLM is used
    for generating responses in the RAG system.

    Args:
        model_name (str): Ollama model name (e.g. "qwen2.5:3b-instruct")
    
    Returns:
        langchain_ollama.ChatOllama: Configured SLM instance ready for use
    """
    return langchain_ollama.ChatOllama(
        model= model_name,
        temperature=1,
    )


def ensure_requirements(config: dict):
    """
    Validate RAG configuration and initialize the Small Language Model.
    
    This function validates that all required configuration keys are present and valid,
    verifies that the vector database path exists, and initializes the SLM with the
    specified model. Used by both the interactive CLI (chatbot.main) and the batch
    runner (test.promptTester.promptRunBatch).
    
    Args:
        config (dict): Configuration dictionary containing:
            - vectordbpath (str): Path to the Qdrant vector database directory
            - promptTemplate (dict): Object with a "text" key containing the prompt template string
            - model.model_name (str): Ollama model name (e.g. "qwen2.5:3b-instruct")
    
    Returns:
        tuple: A 4-tuple containing:
            - db_path (pathlib.Path): Absolute path to the vector database
            - promptTemplate (dict): Validated prompt template object
            - slm (langchain_ollama.ChatOllama): Initialized SLM instance
            - model_name (str): Model name used for the SLM

    """
    vectordbpath = config.get("vectordbpath")
    promptTemplate = config.get("promptTemplate")
    model_name = (config.get("model") or {}).get("model_name")

    if not model_name:
        raise ValueError(
            "Config missing required key 'model.model_name' (Ollama model name, e.g. 'qwen2.5:3b-instruct'). "
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

    # Normalize vector DB path to absolute path
    db_path = pathlib.Path(vectordbpath).resolve()

    if not db_path.is_dir():
        raise ValueError(
            f"Vector DB path is missing or not a directory: {db_path}. "
            "Create the vector DB first (e.g. run vectorDbSetup) or fix the path in config."
        )

    slm = _setup_slm(model_name)

    return db_path, promptTemplate, slm, model_name