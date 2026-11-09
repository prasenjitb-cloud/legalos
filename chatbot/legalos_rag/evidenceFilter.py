"""Keep retrieved sections only when their passage states the evidence that was asked for."""

import argparse
import json
import pathlib
import re

import langchain_ollama
import pydantic

from chatbot.legalos_rag import structuredCall


_MAX_PASSAGES = 3
_WHOLE_SECTION_CHARS = 3000


class _Verdict(pydantic.BaseModel):
    sentence: int = pydantic.Field(
        ..., description="Number of the passage sentence that states the evidence wanted, or 0 if none does"
    )


_FILTER_PROMPT = (
    "You check whether a passage of law states the evidence that is wanted.\n"
    "The passage is split into numbered sentences.\n\n"
    "RULES:\n"
    "- Return the number of the sentence that states the wanted detail itself.\n"
    "- Return 0 if no sentence does.\n"
    "- A passage that only mentions the same people or topic, such as complaints or divorce, "
    "without stating the wanted detail does not count.\n"
    "- A passage about a different legal question does not count, even in the same Act.\n"
    "- Do not use general knowledge. Use only the passage.\n\n"
    "Output ONLY the JSON object below, with no explanation.\n\n"
    "{format_instructions}\n\n"
    "Question:\n{question}\n\n"
    "Evidence wanted:\n{target}\n\n"
    "Passage:\n{passage}"
)


def _passage(candidate: dict) -> str:
    """Return the whole section when it is short, else the matching excerpts the search found."""
    text = candidate["section"]["text"]
    if len(text) <= _WHOLE_SECTION_CHARS:
        return text
    pieces = []
    for source_id in candidate["hit_source_ids"][:_MAX_PASSAGES]:
        _, _, span = source_id.partition(":chars:")
        start, end = (int(n) for n in span.split("-"))
        pieces.append(text[start:end])
    return " ".join(pieces)


def _sentences(passage: str) -> list[str]:
    return [piece for piece in re.split(r"(?<=[.;])\s+", passage) if piece.strip()]


def filter_evidence(question: str, target: str, candidates: list[dict], slm) -> dict:
    """Split retrieved candidates into `kept`, `rejected` and `unchecked`.

    `candidates` are the `results` of `retriever.retrieve`. The model reads the
    passage as numbered sentences and names the one that states the evidence; a
    candidate is kept with that sentence as its `support_quote`, and rejected when
    the model names none or a number that does not exist. A model or parse error
    leaves the candidate `unchecked`, so the caller decides, instead of silently
    dropping it.
    """
    kept, rejected, unchecked = [], [], []
    for candidate in candidates:
        sentences = _sentences(_passage(candidate))
        numbered = "\n".join(f"{number}. {sentence}" for number, sentence in enumerate(sentences, start=1))
        try:
            verdict = structuredCall.ask_model(
                slm, _FILTER_PROMPT, _Verdict, question=question, target=target or question, passage=numbered
            )
        except Exception as error:
            unchecked.append({**candidate, "reason": f"Check failed: {error}"})
            continue
        if verdict.sentence == 0:
            rejected.append({**candidate, "reason": "No sentence states the evidence wanted"})
        elif not 1 <= verdict.sentence <= len(sentences):
            rejected.append({**candidate, "reason": "The named sentence is not in the passage"})
        else:
            kept.append({**candidate, "support_quote": sentences[verdict.sentence - 1].strip()})
    return {"kept": kept, "rejected": rejected, "unchecked": unchecked}


def main() -> None:
    """Retrieve for one request and print which sections are kept and rejected."""
    from chatbot.legalos_rag import retriever

    parser = argparse.ArgumentParser(description="Retrieve sections of a verified Act and filter them by the evidence wanted")
    parser.add_argument("--config", required=True, help="Path to the source-loading JSON config")
    parser.add_argument("--rag-config", required=True, help="Path to the RAG JSON config (vector store and model)")
    parser.add_argument("--act", required=True, help="Act ID, such as posh_2013")
    parser.add_argument("--question", required=True, help="Question to find evidence for")
    parser.add_argument("--target", required=True, help="Evidence wanted, such as 'filing process'")
    parser.add_argument("--max-sections", type=int, default=3, help="Number of sections to retrieve")
    args = parser.parse_args()
    try:
        config = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
        rag_config = json.loads(pathlib.Path(args.rag_config).read_text(encoding="utf-8"))
        slm = langchain_ollama.ChatOllama(model=rag_config["model"]["model_name"], temperature=0)
        vectorstore = retriever.open_vectorstore(rag_config["vectordbpath"])
        found = retriever.retrieve(config, vectorstore, args.act, args.question, args.target, None, args.max_sections)
        result = filter_evidence(args.question, args.target, found["results"], slm)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    summary = {name: [{"section": c["section"]["section"], "title": c["section"]["title"],
                       **{k: c[k] for k in ("support_quote", "reason") if k in c}} for c in result[name]]
               for name in result}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
