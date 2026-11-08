"""Split a question into independent legal parts and keep the facts each part relies on."""

import argparse
import json
import pathlib
import re

import langchain_ollama
import pydantic

from chatbot.legalos_rag import structuredCall


_MAX_PARTS = 4
_MAX_NEW_WORD_SHARE = 0.2


class _Part(pydantic.BaseModel):
    question: str = pydantic.Field(..., description="One legal question, using only words from the user's text")
    facts: list[str] = pydantic.Field(
        default_factory=list,
        description="Exact phrases copied from the user's text that this question depends on",
    )


class _Split(pydantic.BaseModel):
    parts: list[_Part]


_SPLIT_PROMPT = (
    "You split a user's legal question into separate questions only when needed.\n\n"
    "RULES:\n"
    "- Return ONE part unless the user asks about different legal matters, such as "
    "different Acts or unrelated situations.\n"
    "- Several details of one legal problem stay in ONE part. For example, where to file "
    "a complaint and the deadline for it are one problem.\n"
    "- Never return more than 4 parts.\n"
    "- Write each part as a complete question using only words from the user's text. "
    "Do not add facts, section numbers, or legal answers.\n"
    "- facts: exact phrases copied from the user's text that the part depends on, "
    "such as numbers, dates, and relationships. Use [] if there are none.\n"
    "- A fact needed by several parts must be listed in each of them.\n\n"
    "EXAMPLES:\n\n"
    "Question: What conduct counts as sexual harassment under the POSH Act?\n"
    '{{"parts": [{{"question": "What conduct counts as sexual harassment under the POSH Act?", "facts": []}}]}}\n\n'
    "Question: My company has 8 employees. Where do I file a harassment complaint and how long do I have?\n"
    '{{"parts": [{{"question": "My company has 8 employees. Where do I file a harassment complaint '
    'and how long do I have?", "facts": ["8 employees"]}}]}}\n\n'
    "Question: I bought a defective phone and the seller will not refund. I also sold my car "
    "and the buyer has not transferred the registration. What can I do in each case?\n"
    '{{"parts": [{{"question": "I bought a defective phone and the seller will not refund. What can I do?", '
    '"facts": ["bought a defective phone", "seller will not refund"]}}, '
    '{{"question": "I sold my car and the buyer has not transferred the registration. What can I do?", '
    '"facts": ["sold my car", "buyer has not transferred the registration"]}}]}}\n\n'
    "Output ONLY the JSON object below, with no explanation.\n\n"
    "{format_instructions}\n\n"
    "Question:\n"
    "{question}"
)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _normalise(text: str) -> str:
    return " ".join(_words(text))


def _is_supported(part: _Part, question_words: set[str]) -> bool:
    """A part may only rearrange the user's own words, apart from a few new ones."""
    words = _words(part.question)
    new_words = [word for word in words if word not in question_words]
    return bool(words) and len(new_words) <= len(words) * _MAX_NEW_WORD_SHARE


def _single_part(question: str) -> dict:
    return {"question": question, "parts": [{"id": "p1", "question": question, "facts": []}]}


def split_question(question: str, slm) -> dict:
    """Return the original question with its parts, each linked to the facts it relies on.

    Facts are kept only when they are exact phrases of the original question.
    If the model fails, returns too many parts, or rewrites a part with words
    the user did not use, the whole question stays as one part.
    """
    try:
        split = structuredCall.ask_model(slm, _SPLIT_PROMPT, _Split, question=question)
    except Exception as error:
        print(f"[questionSplitter] Split failed, keeping one part. Reason: {error}")
        return _single_part(question)

    question_words = set(_words(question))
    if not split.parts or len(split.parts) > _MAX_PARTS or not all(_is_supported(p, question_words) for p in split.parts):
        return _single_part(question)

    original = _normalise(question)
    parts = []
    for number, part in enumerate(split.parts, start=1):
        facts = [fact for fact in part.facts if _normalise(fact) and _normalise(fact) in original]
        parts.append({"id": f"p{number}", "question": part.question.strip(), "facts": facts})
    return {"question": question, "parts": parts}


def main() -> None:
    """Print the split of one question as JSON."""
    parser = argparse.ArgumentParser(description="Split a legal question into independent parts")
    parser.add_argument("--config", required=True, help="Path to the RAG JSON config (uses model.model_name)")
    parser.add_argument("--question", required=True, help="Question to split")
    args = parser.parse_args()
    try:
        config = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
        model_name = config["model"]["model_name"]
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"Cannot read model name from config: {error}")
    slm = langchain_ollama.ChatOllama(model=model_name, temperature=0)
    print(json.dumps(split_question(args.question, slm), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
