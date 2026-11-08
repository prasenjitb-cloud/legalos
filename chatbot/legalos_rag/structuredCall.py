"""Ask the local model for one structured answer, as the query rewriter does."""

import langchain_core.output_parsers
import langchain_core.prompts
import langchain_ollama


def ask_model(slm, template: str, schema, **inputs):
    """Fill the template, call the model at temperature 0, and parse the reply into `schema`.

    The template may use `{format_instructions}` and one placeholder per input.
    Parse and model errors are raised so each caller can choose its own fallback.
    """
    parser = langchain_core.output_parsers.PydanticOutputParser(pydantic_object=schema)
    prompt = langchain_core.prompts.PromptTemplate(
        input_variables=list(inputs),
        partial_variables={"format_instructions": parser.get_format_instructions()},
        template=template,
    )
    model = langchain_ollama.ChatOllama(model=slm.model, temperature=0)
    reply = model.invoke(prompt.format(**inputs))
    return parser.parse(reply.content)
