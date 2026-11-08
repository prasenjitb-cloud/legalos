"""Fetch a complete section of a verified Act from a section or excerpt ID."""

import argparse
import json
import pathlib

from chatbot.legalos_rag import sourceReview


_SECTION_ID_PARTS = 4
_EXCERPT_ID_PARTS = 6


def _parse_source_id(source_id: str) -> tuple[str, str, str]:
    """Split `<act>:<version>:section:<number>[:chars:<start>-<end>]` into its parts."""
    parts = source_id.split(":")
    is_section = len(parts) == _SECTION_ID_PARTS
    is_excerpt = len(parts) == _EXCERPT_ID_PARTS and parts[4] == "chars"
    if not (is_section or is_excerpt) or parts[2] != "section":
        raise ValueError(f"Invalid source ID: {source_id}")
    return parts[0], parts[1], parts[3]


def fetch_section(config: dict, source_id: str) -> dict:
    """Return the full section for a section ID or one of its excerpt IDs.

    The Act must have passed its source check, and the ID must name that Act's
    current source version.
    """
    act_id, source_version, number = _parse_source_id(source_id)
    source = sourceReview.load_verified_source(config, act_id)
    if source["source_version"] != source_version:
        raise ValueError(f"Source ID is for a different version of the Act: {source_id}")
    section = next((s for s in source["sections"] if s["section"] == number), None)
    if section is None:
        raise ValueError(f"Unknown section {number} in {act_id}")
    if source_id != section["source_id"] and source_id not in {e["source_id"] for e in section["excerpts"]}:
        raise ValueError(f"Unknown excerpt: {source_id}")
    return {
        "act_id": source["act_id"],
        "act_name": source["act_name"],
        "source_version": source["source_version"],
        "source_id": section["source_id"],
        "section": section["section"],
        "title": section["title"],
        "text": section["text"],
        "pages": section["pages"],
        "footnotes": section["footnotes"],
    }


def main() -> None:
    """Print the full section for a section or excerpt ID as JSON."""
    parser = argparse.ArgumentParser(description="Fetch a complete section of a verified POC Act")
    parser.add_argument("--config", required=True, help="Path to the source-loading JSON config")
    parser.add_argument("--id", required=True, help="Section or excerpt source ID")
    args = parser.parse_args()
    try:
        config = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
        section = fetch_section(config, args.id)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(section, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
