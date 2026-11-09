"""Record the facts a user states, as exact quotes from the question."""

import argparse
import json
import pathlib
import re

import langchain_ollama
import pydantic

from chatbot.legalos_rag import structuredCall


_HYPOTHETICAL = re.compile(r"\b(?:if|suppose|supposing|assume|assuming|unless|whether|imagine)\b", re.IGNORECASE)


class _Facts(pydantic.BaseModel):
    quotes: list[str] = pydantic.Field(
        default_factory=list,
        description="Exact phrases copied from the user's text that state a fact about their own situation",
    )


_FACT_PROMPT = (
    "You find the facts a user states about their own situation in a legal question.\n\n"
    "RULES:\n"
    "- Copy each fact as an exact phrase from the user's text. Do not reword, "
    "combine, or add anything.\n"
    "- Facts are things like counts, dates, how long ago something happened, "
    "relationships, and what someone did or refused to do.\n"
    "- Do NOT include the question being asked, a general legal question, "
    "or anything the user only imagines with words like if, suppose or assume.\n"
    "- Do NOT guess a fact the user did not write. If there are no facts, return an empty list.\n\n"
    "EXAMPLES:\n\n"
    "Question: I run a company with 8 employees. Does the POSH Act apply to us?\n"
    '{{"quotes": ["I run a company with 8 employees"]}}\n\n'
    "Question: What is the time limit to file a complaint under the POSH Act?\n"
    '{{"quotes": []}}\n\n'
    "Question: If my company had 20 employees, would the POSH Act apply?\n"
    '{{"quotes": []}}\n\n'
    "Question: My husband left me 5 years ago. Can I remarry?\n"
    '{{"quotes": ["My husband left me 5 years ago"]}}\n\n'
    "Output ONLY the JSON object below, with no explanation.\n\n"
    "{format_instructions}\n\n"
    "Question:\n"
    "{question}"
)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _locate(quote: str, question: str) -> tuple[int, int] | None:
    """Find the quote in the question, ignoring case and spacing, and return its span."""
    words = re.findall(r"\S+", quote)
    if not words:
        return None
    pattern = r"\s+".join(re.escape(word) for word in words)
    match = re.search(pattern, question, re.IGNORECASE)
    return match.span() if match else None


def _is_hypothetical(question: str, start: int, end: int) -> bool:
    """True when an 'if'-style word appears in the quote or before it in the same sentence."""
    sentence_start = max(question.rfind(mark, 0, start) for mark in ".?!") + 1
    return bool(_HYPOTHETICAL.search(question[sentence_start:end]))


def extract_facts(question: str, slm) -> dict:
    """Return the facts the user stated, each as an exact quote of the question.

    A quote is kept only if it appears word for word in the question and is
    neither a question nor an imagined case. Anything the user did not state is simply absent, so later
    steps treat it as unknown. On a model or parse error no facts are recorded.
    """
    try:
        found = structuredCall.ask_model(slm, _FACT_PROMPT, _Facts, question=question)
    except Exception as error:
        print(f"[factExtractor] Extraction failed, recording no facts. Reason: {error}")
        return {"question": question, "facts": []}

    quotes = []
    for quote in found.quotes:
        span = _locate(quote, question)
        if span is None or "?" in quote or _is_hypothetical(question, *span):
            continue
        exact = question[span[0]:span[1]]
        if exact not in quotes:
            quotes.append(exact)
    return {"question": question, "facts": [{"id": f"f{n}", "quote": q} for n, q in enumerate(quotes, start=1)]}


def main() -> None:
    """Print the stated facts of one question as JSON."""
    parser = argparse.ArgumentParser(description="Record the facts stated in a legal question")
    parser.add_argument("--config", required=True, help="Path to the RAG JSON config (uses model.model_name)")
    parser.add_argument("--question", required=True, help="Question to read")
    args = parser.parse_args()
    try:
        config = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
        model_name = config["model"]["model_name"]
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"Cannot read model name from config: {error}")
    slm = langchain_ollama.ChatOllama(model=model_name, temperature=0)
    print(json.dumps(extract_facts(args.question, slm), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
