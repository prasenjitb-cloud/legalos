from langchain_core.prompts import PromptTemplate


def setup_prompt(parser):
    return PromptTemplate(
        input_variables=["context", "question"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
        template="""
You are a legal document reader.

Use ONLY the Context provided.
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

Context:
{context}

Query:
{question}
"""
    )

