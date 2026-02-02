import langchain_core.prompts

def setup_rag_prompt_v1(parser):
    """Build the RAG prompt template with format instructions from the given output parser.
    
    Args:
        parser: Output parser to use for formatting instructions

    Returns:
        langchain_core.prompts.PromptTemplate: RAG prompt template
    """
    return langchain_core.prompts.PromptTemplate(
        input_variables=["facts", "question"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
        template="""
You are a legal document reader.

Use ONLY the Facts provided.
Context entries contain:
- content: statutory text
- metadata: source details

Task:
- Find text that directly answers the query.
- If a situation is described, map it ONLY to the text.
- Do NOT add advice or interpretation.

If no text clearly answers the query:
- answer_found = false

If answer_found is true, always provide a short explanation summarizing the quoted text.

If multiple documents are provided:
- Identify which document most directly answers the question.
- Ignore documents that are less relevant, even if they discuss similar topics.


Output:
{format_instructions}

Facts:
{facts}

Query:
{question}
"""
    )

