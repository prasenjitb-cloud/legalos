"""BM25 keyword search over the excerpts of one verified Act."""

import argparse
import collections
import json
import math
import pathlib
import re

from chatbot.legalos_rag import sourceReview


_K1 = 1.5
_B = 0.75


def _tokens(text: str) -> list[str]:
    """Lower-case letters and digits, so identifiers such as `13b` stay whole."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _entries(source: dict) -> list[dict]:
    """One entry per excerpt. Its section number and title are searchable too."""
    entries = []
    for section in source["sections"]:
        for excerpt in section["excerpts"]:
            words = _tokens(f"section {section['section']} {section['title']} {excerpt['text']}")
            entries.append({"excerpt": excerpt, "section": section, "words": words,
                            "counts": collections.Counter(words)})
    return entries


def _score(query_words: list[str], entry: dict, idf: dict[str, float], average_length: float) -> float:
    length_weight = 1 - _B + _B * len(entry["words"]) / average_length
    score = 0.0
    for word in set(query_words):
        count = entry["counts"].get(word, 0)
        score += idf.get(word, 0.0) * count * (_K1 + 1) / (count + _K1 * length_weight)
    return score


def search_excerpts(config: dict, act_id: str, query: str, k: int = 5) -> list[dict]:
    """Return the `k` best excerpts of a verified Act for the query, best first.

    The Act must have passed its source check. Scores are raw BM25 values: they
    only order the results of one query and are not probabilities. Excerpts that
    share no word with the query are never returned. Each hit names its excerpt
    `source_id` and the `section_source_id`, so it can be expanded to the full section.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    source = sourceReview.load_verified_source(config, act_id)
    entries = _entries(source)
    query_words = _tokens(query)
    if not entries or not query_words:
        return []

    document_counts = collections.Counter(word for entry in entries for word in entry["counts"])
    idf = {word: math.log(1 + (len(entries) - n + 0.5) / (n + 0.5)) for word, n in document_counts.items()}
    average_length = sum(len(entry["words"]) for entry in entries) / len(entries)

    scored = [(_score(query_words, entry, idf, average_length), position, entry)
              for position, entry in enumerate(entries)]
    scored = [item for item in scored if item[0] > 0]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [{
        "rank": rank,
        "score": round(score, 4),
        "act_id": act_id,
        "source_id": entry["excerpt"]["source_id"],
        "section_source_id": entry["section"]["source_id"],
        "section": entry["section"]["section"],
        "title": entry["section"]["title"],
        "text": entry["excerpt"]["text"],
    } for rank, (score, _, entry) in enumerate(scored[:k], start=1)]


def main() -> None:
    """Print the best keyword matches in one verified Act as JSON."""
    parser = argparse.ArgumentParser(description="Keyword search over one verified POC Act")
    parser.add_argument("--config", required=True, help="Path to the source-loading JSON config")
    parser.add_argument("--act", required=True, help="Act ID, such as posh_2013")
    parser.add_argument("--query", required=True, help="Words or identifiers to search for")
    parser.add_argument("--k", type=int, default=5, help="Number of excerpts to return")
    args = parser.parse_args()
    try:
        config = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
        hits = search_excerpts(config, args.act, args.query, args.k)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(hits, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
