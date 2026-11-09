"""Shorten the evidence sent to the answer step without dropping conditions or qualifications."""

import argparse
import json
import pathlib
import re

import langchain_ollama

from chatbot.legalos_rag import evidenceAgent, evidenceFilter, retriever


_SHORT_SECTION_CHARS = 3000
_MIN_TOPIC_WORD = 5
_OMITTED = "[...]"
_ARTIFACTS = re.compile(r"\*(?:\s*\*)*")
QUALIFIER = re.compile(
    r"\b(?:provided|except|unless|subject to|notwithstanding|explanation|illustration|exception|save|"
    r"only if|shall not|not less|not more|not later|not earlier|not exceeding|within|before|after|"
    r"at least|at most|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    r"fifteen|sixteen|eighteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|"
    r"lakh|crore|million|half|twice|double|per cent)\b|\d",
    re.IGNORECASE,
)
_LIST_ITEM = re.compile(r"^(?:(?:or|and)\b)?[\s\[\]]*\((?:[a-z]{1,3}|\d+[a-z]?|[ivx]+)\)", re.IGNORECASE)
_PLAIN_WORDS = {"under", "which", "shall", "where", "whose", "other", "their", "there", "these", "those",
                "about", "after", "before", "would", "could", "being", "section", "provisions"}


def topic_words(*texts: str) -> set[str]:
    words = (word for text in texts for word in re.findall(r"[a-z]+", text.lower()))
    return {word for word in words if len(word) >= _MIN_TOPIC_WORD and word not in _PLAIN_WORDS}


def _clean(text: str) -> str:
    """Remove omission marks such as `* * *` and extra spaces; the words are unchanged."""
    return " ".join(_ARTIFACTS.sub(" ", text).split())


def _kept_sentences(section: dict, support: str, topic: set[str]) -> set[int]:
    """Choose the sentences to keep: the heading, the support, every qualifier and every on-topic one."""
    sentences = evidenceFilter.split_sentences([section["text"]])
    kept = set()
    for number, sentence in enumerate(sentences):
        on_topic = bool(topic & set(re.findall(r"[a-z]+", sentence.lower())))
        if number == 0 or sentence == support or on_topic or _LIST_ITEM.match(sentence) or QUALIFIER.search(sentence):
            kept.add(number)
    return kept


def compact_evidence(parts: list[dict]) -> dict:
    """Return one short, sourced text for the sections found for each question part.

    `parts` are the parts of `evidenceAgent.gather_evidence`. A section used by several parts
    appears once. Short sections are kept whole. In a longer section the heading, the support
    sentence, every enumerated item, every sentence with a qualifier (a proviso, exception,
    explanation, limit or number) and every sentence sharing a content word with the part's
    question or target are kept, in order, and each gap is marked `[...]`. Only whole sentences
    are removed.
    """
    sections = {}
    for part in parts:
        topic = topic_words(part["question"], part.get("target", ""))
        for item in part.get("evidence", []):
            provision = item["provision"]
            entry = sections.setdefault(item["source_id"], {
                "provision": provision, "kept": set(), "used_for": [], "topic": set()})
            entry["used_for"].append(part["id"])
            entry["topic"] |= topic
            if len(provision["text"]) > _SHORT_SECTION_CHARS:
                entry["kept"] |= _kept_sentences(provision, item["support_quote"], topic)

    output, original_chars, compact_chars = [], 0, 0
    for source_id, entry in sections.items():
        provision = entry["provision"]
        original_chars += len(provision["text"])
        if len(provision["text"]) <= _SHORT_SECTION_CHARS:
            text, total, kept = _clean(provision["text"]), 0, 0
        else:
            sentences = evidenceFilter.split_sentences([provision["text"]])
            pieces, gap = [], False
            for number, sentence in enumerate(sentences):
                if number in entry["kept"]:
                    pieces.append(_clean(sentence))
                    gap = False
                elif not gap:
                    pieces.append(_OMITTED)
                    gap = True
            text, total, kept = " ".join(pieces), len(sentences), len(entry["kept"])
        compact_chars += len(text)
        output.append({
            "source_id": source_id, "act_id": provision["act_id"], "section": provision["section"],
            "title": provision["title"], "pages": provision["pages"], "used_for": entry["used_for"],
            "text": text, "sentences_kept": kept, "sentences_total": total,
        })
    context = "\n\n".join(
        f"[{item['source_id']} | pages {item['pages']}]\n{item['text']}" for item in output)
    return {"sections": output, "context": context,
            "original_chars": original_chars, "compact_chars": compact_chars}


def main() -> None:
    """Gather evidence for one question and print the compacted context."""
    parser = argparse.ArgumentParser(description="Compact the evidence for a question without dropping conditions")
    parser.add_argument("--config", required=True, help="Path to the source-loading JSON config")
    parser.add_argument("--rag-config", required=True, help="Path to the RAG JSON config (vector store and model)")
    parser.add_argument("--act", required=True, help="Act ID, such as hma_1955")
    parser.add_argument("--question", required=True, help="Question to gather and compact evidence for")
    args = parser.parse_args()
    try:
        config = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
        rag_config = json.loads(pathlib.Path(args.rag_config).read_text(encoding="utf-8"))
        slm = langchain_ollama.ChatOllama(model=rag_config["model"]["model_name"], temperature=0)
        vectorstore = retriever.open_vectorstore(rag_config["vectordbpath"])
        evidence = evidenceAgent.gather_evidence(config, vectorstore, args.act, args.question, slm)
        result = compact_evidence(evidence["parts"])
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(result["context"])
    print(f"\n[{result['original_chars']} characters of source text compacted to {result['compact_chars']}]")


if __name__ == "__main__":
    main()
