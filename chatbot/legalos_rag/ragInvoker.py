import langchain_core.output_parsers 
import langchain_core.documents 

import chatbot.legalos_rag.prompt.promptSchema 
import chatbot.legalos_rag.prompt.prompts




def invoker(
        slm,
        retrieved_docs: list[langchain_core.documents.Document],
        query: str,
        template: str,
):
    """Run the RAG pipeline: format prompt with docs and query, invoke the SLM, parse to LegalAnswer, and log the run.
    
    Args:
        slm: Small Language Model instance
        retrieved_docs: List of documents retrieved from the vector database
        query: Query string
        model: Model name
        template: Full prompt template string containing placeholders
                  for {format_instructions}, {facts}, and {question}.

    
    Returns:
        chatbot.legalos_rag.prompt.promptSchema.LegalAnswer: Parsed result
    """
    
    parser = langchain_core.output_parsers.PydanticOutputParser(
        pydantic_object=chatbot.legalos_rag.prompt.promptSchema.LegalAnswer
    )

    prompt = chatbot.legalos_rag.prompt.prompts.setup_rag_prompt_skeleton(
        parser,
        template,
    )

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


    return parsed_result, final_prompt_text



