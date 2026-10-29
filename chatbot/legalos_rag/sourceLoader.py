"""Import pinned POC snapshots into a local section corpus for source review."""

import argparse
import hashlib
import json
import pathlib
import re


_EXCERPT_CHARACTERS = 1200


def _required_text(record: dict, key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _read_pinned(path: pathlib.Path, expected_sha256: str) -> bytes:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError(f"Invalid SHA-256 for {path}")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(f"Source differs from the pinned snapshot: {path}")
    return content


def _validate_footnotes(footnotes: dict, page_count: int) -> None:
    if not isinstance(footnotes, dict):
        raise ValueError("footnotes must map PDF page numbers to text")
    for page, text in footnotes.items():
        if (not page.isdecimal() or not 1 <= int(page) <= page_count
                or not isinstance(text, str) or not text.strip()):
            raise ValueError(f"Invalid footnote on PDF page {page}")


def _make_excerpts(text: str, source_id: str) -> list[dict]:
    excerpts = []
    start = 0
    while start < len(text):
        end = min(start + _EXCERPT_CHARACTERS, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary + 1
        excerpts.append({
            "source_id": f"{source_id}:chars:{start}-{end}",
            "start": start,
            "end": end,
            "text": text[start:end],
        })
        start = end
    return excerpts


def _load_sections(source: dict, section_count: int) -> list[dict]:
    sections = source.get("provisions")
    if not isinstance(sections, list) or len(sections) != section_count:
        raise ValueError(f"Expected {section_count} sections for {source['act_id']}")
    seen = set()
    records = []
    for section in sections:
        if not isinstance(section, dict):
            raise ValueError("Each section must be an object")
        number = _required_text(section, "id")
        if not re.fullmatch(r"[1-9][0-9]*[A-Z]*", number) or number in seen:
            raise ValueError(f"Invalid or duplicate section: {number}")
        seen.add(number)
        if section.get("act_id") != source["act_id"]:
            raise ValueError(f"Section {number} belongs to a different Act")
        pages = section.get("pages")
        if (not isinstance(pages, list) or not pages
                or any(type(page) is not int or not 1 <= page <= source["source_pages"] for page in pages)
                or pages != sorted(set(pages))):
            raise ValueError(f"Invalid PDF pages for section {number}")
        text = _required_text(section, "text")
        source_id = f"{source['act_id']}:{source['source_sha256']}:section:{number}"
        section_footnotes = section.get("footnotes", {})
        _validate_footnotes(section_footnotes, source["source_pages"])
        if any(int(page) not in pages for page in section_footnotes):
            raise ValueError(f"Footnote lies outside section {number} pages")
        footnotes = {
            page: note for page, note in source.get("footnotes", {}).items()
            if int(page) in pages
        }
        footnotes.update(section_footnotes)
        records.append({
            "source_id": source_id,
            "section": number,
            "title": _required_text(section, "title"),
            "text": text,
            "pages": pages,
            "footnotes": footnotes,
            "excerpts": _make_excerpts(text, source_id),
        })
    return records


def _load_source(entry: dict, source_directory: pathlib.Path, pdf_directory: pathlib.Path) -> dict:
    snapshot = source_directory / _required_text(entry, "snapshot_file")
    snapshot_sha256 = _required_text(entry, "snapshot_sha256")
    source = json.loads(_read_pinned(snapshot, snapshot_sha256))
    if not isinstance(source, dict):
        raise ValueError(f"Source snapshot must be an object: {snapshot}")
    for key in ("act_id", "act_name", "source_file", "source_sha256"):
        if _required_text(source, key) != _required_text(entry, key):
            raise ValueError(f"Source {key} differs from the config: {snapshot}")
    for key in ("source_pages", "section_count"):
        if type(entry.get(key)) is not int or entry[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(source.get("source_pages")) is not int or source["source_pages"] != entry["source_pages"]:
        raise ValueError(f"Source page count differs from the config: {snapshot}")
    _read_pinned(pdf_directory / source["source_file"], source["source_sha256"])
    footnotes = source.get("footnotes", {})
    _validate_footnotes(footnotes, source["source_pages"])
    return {
        "act_id": source["act_id"],
        "act_name": source["act_name"],
        "source_file": source["source_file"],
        "source_version": source["source_sha256"],
        "snapshot_sha256": snapshot_sha256,
        "source_updated": _required_text(source, "source_updated"),
        "source_pages": source["source_pages"],
        "review_status": "unreviewed",
        "coverage_note": source.get("coverage_note", ""),
        "extraction": source.get("extraction", ""),
        "footnotes": footnotes,
        "sections": _load_sections(source, entry["section_count"]),
    }


def load_sources(config: dict) -> dict:
    """Validate all configured snapshots and PDFs before returning source records.

    Paths are resolved from the repository root (the current working directory).
    Hash checks establish snapshot identity, not legal completeness or approval.
    """
    source_directory = pathlib.Path(_required_text(config, "source_directory"))
    pdf_directory = pathlib.Path(_required_text(config, "pdf_directory"))
    entries = config.get("sources")
    if not isinstance(entries, list) or not entries:
        raise ValueError("sources must be a non-empty list")
    seen = set()
    sources = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each source config must be an object")
        act_id = _required_text(entry, "act_id")
        if act_id in seen:
            raise ValueError(f"Duplicate Act: {act_id}")
        seen.add(act_id)
        sources.append(_load_source(entry, source_directory, pdf_directory))
    return {"sources": sources}


def main() -> None:
    """Load the pinned sources using a JSON config and write a new corpus file."""
    parser = argparse.ArgumentParser(description="Load POC law sources for review")
    parser.add_argument("--config", required=True, help="Path to the source-loading JSON config")
    args = parser.parse_args()
    try:
        config = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Source config must be an object")
        output = pathlib.Path(_required_text(config, "output_file"))
        if output.exists():
            raise ValueError(f"Output already exists; choose a new output_file: {output}")
        corpus = load_sources(config)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(corpus, ensure_ascii=False, indent=2) + "\n")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for source in corpus["sources"]:
        print(f"{source['act_id']}: {len(source['sections'])} sections (unreviewed)")
    print(f"Source corpus written to {output}")


if __name__ == "__main__":
    main()
