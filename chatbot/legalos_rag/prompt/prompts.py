import json
import pathlib
import langchain_core.output_parsers
import langchain_core.prompts


def setup_rag_prompt_skeleton(
    parser: langchain_core.output_parsers.BaseOutputParser,
    prompt_version: str,
    templates_path: str
) -> langchain_core.prompts.PromptTemplate:
    """
    Builds the RAG prompt template by loading a versioned template
    from JSON and injecting format instructions.

    Args:
        parser: Output parser to generate format instructions
        prompt_version: Prompt version key (e.g. "v1", "v2")
        templates_path: Path to templates.json

    Returns:
        langchain_core.prompts.PromptTemplate
    """
    data = json.loads(pathlib.Path(templates_path).read_text())

    if prompt_version not in data:
        raise ValueError(f"Prompt version '{prompt_version}' not found")

    template = data[prompt_version]["template"]

    return langchain_core.prompts.PromptTemplate(
        input_variables=["facts", "question"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
        template=template
    )



