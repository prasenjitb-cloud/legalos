"""Ask the local model for one structured answer, as the query rewriter does."""

import json

import langchain_core.exceptions
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
    try:
        return parser.parse(reply.content)
    except langchain_core.exceptions.OutputParserException:
        # Small models sometimes add text after the JSON or answer with the schema's own
        # wrapper, {"properties": {...}}. Read the first JSON object and unwrap that shape.
        data, _ = json.JSONDecoder().raw_decode(reply.content[reply.content.index("{"):])
        if isinstance(data, dict) and isinstance(data.get("properties"), dict):
            data = data["properties"]
        return schema.model_validate(data)
