import langchain_core.output_parsers
import langchain_core.prompts


def setup_rag_prompt_skeleton(
    parser: langchain_core.output_parsers.BaseOutputParser,
    template: str,
) -> langchain_core.prompts.PromptTemplate:
    """
    Builds the RAG prompt template from a raw template string and
    injects format instructions.

    Args:
        parser: Output parser to generate format instructions
        template: Full prompt template string containing placeholders
                  for {format_instructions}, {facts}, and {question}.

    Returns:
        langchain_core.prompts.PromptTemplate
    """
    return langchain_core.prompts.PromptTemplate(
        input_variables=["facts", "question"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
        template=template
    )



