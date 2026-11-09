"""One shared retriever: keyword and vector search over a verified Act, merged by source ID."""

import argparse
import json
import pathlib
import re

import langchain_ollama
import qdrant_client.models

from chatbot.legalos_rag import keywordSearch, queryRewriter, runRag, sectionLookup, sourceReview


_SEARCH_DEPTH = 10
_RRF_CONSTANT = 60
_MIN_OVERLAP = 0.4


def open_vectorstore(db_path: str):
    """Open the existing Qdrant vector store used by the chatbot."""
    return runRag._setup_vectorstore(db_path, runRag.COLLECTION_NAME)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _excerpt_for_chunk(source: dict, chunk) -> str | None:
    """Return the ID of the verified excerpt that holds most of a vector chunk, or None.

    Only excerpts on the chunk's PDF page are considered, and at least
    `_MIN_OVERLAP` of the chunk's words must appear in the excerpt.
    """
    chunk_words = _words(chunk.page_content)
    page = int(chunk.metadata.get("page_label", -1))
    best, best_share = None, 0.0
    for section in source["sections"]:
        if page not in section["pages"]:
            continue
        for excerpt in section["excerpts"]:
            share = len(chunk_words & _words(excerpt["text"])) / max(len(chunk_words), 1)
            if share > best_share:
                best, best_share = excerpt["source_id"], share
    return best if best_share >= _MIN_OVERLAP else None


def _vector_ranking(source: dict, vectorstore, query: str) -> list[str]:
    """Search only this Act's PDF and return verified excerpt IDs, best first."""
    in_this_pdf = qdrant_client.models.Filter(must=[qdrant_client.models.FieldCondition(
        key="metadata.file_name", match=qdrant_client.models.MatchValue(value=source["source_file"]))])
    chunks = vectorstore.similarity_search_with_score(query, k=_SEARCH_DEPTH, filter=in_this_pdf)
    ranking = []
    for chunk, _ in chunks:
        excerpt_id = _excerpt_for_chunk(source, chunk)
        if excerpt_id and excerpt_id not in ranking:
            ranking.append(excerpt_id)
    return ranking


def retrieve(config: dict, vectorstore, act_id: str, question: str, target: str = "",
             slm=None, max_sections: int = 3) -> dict:
    """Find the sections of one verified Act that best match what is being looked for.

    `target` names the evidence wanted, such as a rule, a filing process or an
    outcome; it is added to the question for both searches. Keyword and vector
    results are merged by excerpt source ID (reciprocal rank fusion), grouped by
    section, and the best `max_sections` are returned as complete sections.
    Pass `slm` to also search with the query rewriter's phrasings; leave it out
    when the plain question is enough. The Act must have passed its source check.
    """
    source = sourceReview.load_verified_source(config, act_id)
    text = f"{target} {question}".strip()
    queries = queryRewriter.rewrite_and_expand(text, slm) if slm is not None else [text]

    scores, found_by = {}, {}
    for query in queries:
        keyword_ids = [hit["source_id"] for hit in keywordSearch.search_excerpts(config, act_id, query, _SEARCH_DEPTH)]
        for name, ranking in (("keyword", keyword_ids), ("vector", _vector_ranking(source, vectorstore, query))):
            for rank, excerpt_id in enumerate(ranking, start=1):
                scores[excerpt_id] = scores.get(excerpt_id, 0.0) + 1 / (_RRF_CONSTANT + rank)
                found_by.setdefault(excerpt_id, set()).add(name)

    sections = {}
    for excerpt_id, score in sorted(scores.items(), key=lambda item: -item[1]):
        section_id = excerpt_id.split(":chars:")[0]
        entry = sections.setdefault(section_id, {"score": 0.0, "found_by": set(), "hit_source_ids": []})
        entry["score"] = max(entry["score"], score)
        entry["found_by"] |= found_by[excerpt_id]
        entry["hit_source_ids"].append(excerpt_id)

    best = sorted(sections.items(), key=lambda item: -item[1]["score"])[:max_sections]
    return {
        "act_id": act_id,
        "question": question,
        "target": target,
        "queries": queries,
        "results": [{
            "rank": rank,
            "score": round(entry["score"], 5),
            "found_by": sorted(entry["found_by"]),
            "hit_source_ids": entry["hit_source_ids"],
            "section": sectionLookup.fetch_section(config, section_id),
        } for rank, (section_id, entry) in enumerate(best, start=1)],
    }


def main() -> None:
    """Print the retrieved sections for one request as JSON."""
    parser = argparse.ArgumentParser(description="Retrieve sections of one verified POC Act")
    parser.add_argument("--config", required=True, help="Path to the source-loading JSON config")
    parser.add_argument("--rag-config", required=True, help="Path to the RAG JSON config (vector store and model)")
    parser.add_argument("--act", required=True, help="Act ID, such as posh_2013")
    parser.add_argument("--question", required=True, help="Question to find evidence for")
    parser.add_argument("--target", default="", help="Evidence wanted, such as 'filing process'")
    parser.add_argument("--rewrite", action="store_true", help="Also search with the query rewriter's phrasings")
    parser.add_argument("--max-sections", type=int, default=3, help="Number of sections to return")
    args = parser.parse_args()
    try:
        config = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
        rag_config = json.loads(pathlib.Path(args.rag_config).read_text(encoding="utf-8"))
        slm = langchain_ollama.ChatOllama(model=rag_config["model"]["model_name"]) if args.rewrite else None
        vectorstore = open_vectorstore(rag_config["vectordbpath"])
        result = retrieve(config, vectorstore, args.act, args.question, args.target, slm, args.max_sections)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
