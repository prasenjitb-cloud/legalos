"""
Query preprocessing: legal query rewriting and multi-query generation.

Exposes rewrite_and_expand() which takes a user query and returns a list of
query strings (original + rewritten + variant) to improve retrieval coverage.
"""
import pydantic
import langchain_core.output_parsers
import langchain_core.prompts
import langchain_ollama


# -------------------- SCHEMA --------------------

class RewrittenQueries(pydantic.BaseModel):
    """Output schema for the query rewriter: two short search queries using legal terminology.

    Args:
        rewritten: Short search query rephrasing the main legal issue in statutory terms.
        variant: Short search query targeting a related legal concept (remedy, procedure, penalty).
    """
    rewritten: str = pydantic.Field(
        ..., description="Short search query (under 15 words) rephrasing the core issue in legal terms"
    )
    variant: str = pydantic.Field(
        ..., description="Short search query (under 15 words) for a related legal concept like remedy or procedure"
    )


# -------------------- PROMPT --------------------

_REWRITER_PROMPT_TEMPLATE = (
    "You are a legal terminology translator.\n\n"
    "Task: Rephrase an informal legal question into 2 SHORT search queries "
    "using proper legal and statutory language.\n\n"
    "RULES:\n"
    "- Do NOT guess or name any specific Act or law.\n"
    "- Replace informal words with legal terms "
    '(e.g. "left me" → "desertion", "fake product" → "defective goods", '
    '"kicked out" → "eviction", "personal messages" → "unwelcome conduct of sexual nature").\n'
    "- Keep ALL key concepts from the original question.\n"
    "- Keep each query under 15 words.\n"
    "- rewritten = the main legal issue in statutory language.\n"
    "- variant = a related concept (remedy, procedure, penalty, or a different angle).\n\n"
    "EXAMPLES:\n\n"
    "Query: My boss sends me personal messages at night. Is this harassment?\n"
    "{{\n"
    '  "rewritten": "definition of sexual harassment unwelcome conduct at workplace",\n'
    '  "variant": "complaint procedure for sexual harassment by employer at workplace"\n'
    "}}\n\n"
    "Query: My husband left me 5 years ago. Can I remarry?\n"
    "{{\n"
    '  "rewritten": "divorce on ground of desertion by spouse",\n'
    '  "variant": "dissolution of marriage when spouse not heard of for seven years"\n'
    "}}\n\n"
    "Query: My wife and I both want a divorce. I want custody of my child.\n"
    "{{\n"
    '  "rewritten": "mutual consent divorce and custody of minor child",\n'
    '  "variant": "welfare of child and guardianship rights in matrimonial proceedings"\n'
    "}}\n\n"
    "Query: I sold my car but buyer hasn't transferred the RC\n"
    "{{\n"
    '  "rewritten": "transfer of ownership and registration certificate of motor vehicle",\n'
    '  "variant": "liability of transferor when vehicle registration not transferred"\n'
    "}}\n\n"
    "Query: I got a fake product and seller won't refund\n"
    "{{\n"
    '  "rewritten": "complaint for defective goods and refund to consumer",\n'
    '  "variant": "unfair trade practice and consumer right to seek redressal"\n'
    "}}\n\n"
    "Query: I want to file a complaint but my company has no Internal Committee\n"
    "{{\n"
    '  "rewritten": "complaint of sexual harassment when no internal complaints committee exists",\n'
    '  "variant": "constitution of local committee for workplace harassment complaints"\n'
    "}}\n\n"
    "Query: A student told me about abuse at home. Am I required to report it?\n"
    "{{\n"
    '  "rewritten": "mandatory reporting of child abuse and offence of non-reporting",\n'
    '  "variant": "penalty for failure to report child in need of care and protection"\n'
    "}}\n\n"
    "Output ONLY the JSON object below — no explanation, no extra text.\n\n"
    "{format_instructions}\n\n"
    "Query:\n"
    "{question}"
)


# -------------------- REWRITE AND EXPAND --------------------

def rewrite_and_expand(query: str, slm) -> list:
    """Rephrase an informal legal question into statutory terminology for better retrieval.

    Uses the SLM to translate casual/layman language into legal terms
    (e.g. "left me" → "desertion", "fake product" → "defective goods") without
    guessing specific Act names. Produces two short search queries: one for the
    main issue and one for a related concept (remedy, procedure, penalty).

    On any failure (parse error, SLM error), falls back to returning only the original query
    so the pipeline continues without interruption.

    Args:
        query: Original user query string (may be informal or layman language).
        slm: Small Language Model instance (e.g. ChatOllama).

    Returns:
        list[str]: [original_query, rewritten_query, variant].
                   Falls back to [original_query] on error.
    """
    parser = langchain_core.output_parsers.PydanticOutputParser(
        pydantic_object=RewrittenQueries
    )
    prompt = langchain_core.prompts.PromptTemplate(
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        template=_REWRITER_PROMPT_TEMPLATE,
    )

    rewriter_slm = langchain_ollama.ChatOllama(
        model=slm.model,
        temperature=0.3,
    )

    try:
        final_prompt_text = prompt.format(question=query)
        raw_response = rewriter_slm.invoke(final_prompt_text)
        raw_text = (
            raw_response.content
            if hasattr(raw_response, "content")
            else raw_response
        )
        parsed = parser.parse(raw_text)
        return [query, parsed.rewritten, parsed.variant]

    except Exception as e:
        print(f"[queryRewriter] Rewrite failed, using original query only. Reason: {e}")
        return [query]
