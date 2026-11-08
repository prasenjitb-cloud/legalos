"""Check imported POC sections against their source PDFs and gate unverified Acts."""

import argparse
import bisect
import collections
import json
import pathlib
import re

import pypdf


_HEADING_WORDS = 6
_MAX_MISSING_SHARE = 0.05
_MIN_MISSING_ALLOWED = 5
_QUALIFIERS = ("explanation", "exception", "illustration")
_CHAPTER_LINE = re.compile(r"^CHAPTER\s+[IVXLC]+\b", re.IGNORECASE)
_SCHEDULE_LINE = re.compile(r"^(?:THE\s+)?(?:[A-Z]+\s+)?SCHEDULE\b")
_NOTE_WORDS = ("subs", "ins", "omitted")
_SECTION_LINE = re.compile(r"^(?:\d+\[)?(\d+[A-Z]*)\.?\s*\d*\[?[A-Za-z]")


def _config_value(config: dict, key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _read_json(path: str) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def _body_lines(page: pypdf.PageObject) -> list[str]:
    return [line.strip() for line in (page.extract_text() or "").splitlines() if line.strip()]


def _page_words(lines: list[str], stop_at_schedule: bool) -> tuple[list[str], dict[str, int], bool]:
    """Return a page's words, where each numbered line starts, and whether a schedule begins.

    Chapter headings are skipped, and reading stops at a schedule heading once
    `stop_at_schedule` is set, because neither is section text.
    """
    words = []
    line_starts = {}
    in_chapter_heading = False
    for line in lines:
        if stop_at_schedule and _SCHEDULE_LINE.match(line):
            return words, line_starts, True
        section_line = _SECTION_LINE.match(line)
        if _CHAPTER_LINE.match(line):
            in_chapter_heading = True
        elif in_chapter_heading and section_line:
            in_chapter_heading = False
        if in_chapter_heading:
            continue
        if section_line:
            line_starts.setdefault(section_line.group(1), len(words))
        words += _words(line)
    return words, line_starts, False


def _read_pdf(
    pdf_path: pathlib.Path, last_section_page: int
) -> tuple[list[str], list[int], dict[tuple[int, str], int], int]:
    """Return all section words, each word's page, numbered line starts, and the PDF page count.

    Schedule headings before `last_section_page` are contents entries, not the end of the Act.
    """
    reader = pypdf.PdfReader(pdf_path)
    words, pages, line_starts = [], [], {}
    for number, page in enumerate(reader.pages, start=1):
        page_words, starts, schedule_begins = _page_words(_body_lines(page), number >= last_section_page)
        for section_number, offset in starts.items():
            line_starts[(number, section_number)] = len(words) + offset
        words += page_words
        pages += [number] * len(page_words)
        if schedule_begins:
            break
    return words, pages, line_starts, len(reader.pages)


def _find_headings(
    sections: list[dict], words: list[str], line_starts: dict[tuple[int, str], int]
) -> list[int | None]:
    """Find each heading by its number and title on its first recorded page, in document order."""
    positions = []
    previous = -1
    for section in sections:
        index = line_starts.get((section["pages"][0], section["section"]))
        heading = _words(section["title"])[:_HEADING_WORDS]
        if index is None or index <= previous or words[index:index + len(heading)] != heading:
            positions.append(None)
            continue
        positions.append(index)
        previous = index
    return positions


def _count_qualifiers(words: list[str]) -> collections.Counter:
    """Count clause openers such as 'provided that' and 'Explanation', not amendment notes."""
    counts = collections.Counter()
    for index, word in enumerate(words):
        if word == "provided" and words[index + 1:index + 2] == ["that"]:
            counts["provided that"] += 1
        elif (word in _QUALIFIERS and words[index - 1:index] != ["the"]
                and words[index + 1:index + 2] and words[index + 1] not in _NOTE_WORDS):
            counts[word] += 1
    return counts


def _review_section(section: dict, span: list[str]) -> list[str]:
    """Return the reasons this section differs from its PDF text."""
    section_words = _words(section["text"])
    for note in section["footnotes"].values():
        section_words += _words(note)
    missing = sum((collections.Counter(span) - collections.Counter(section_words)).values())
    allowed = max(_MIN_MISSING_ALLOWED, int(len(span) * _MAX_MISSING_SHARE))
    problems = []
    if missing > allowed:
        problems.append(f"{missing} of {len(span)} PDF words are missing from the section text")
    pdf_qualifiers = _count_qualifiers(span)
    section_qualifiers = _count_qualifiers(section_words)
    for word, count in pdf_qualifiers.items():
        if count > section_qualifiers[word]:
            problems.append(
                f"'{word}' appears {count} times in the PDF but {section_qualifiers[word]} times in the section"
            )
    return problems


def review_source(source: dict, pdf_directory: pathlib.Path) -> dict:
    """Compare one loaded Act with its PDF and return a passed or failed record."""
    last_section_page = source["sections"][-1]["pages"][0]
    words, pages, line_starts, page_count = _read_pdf(pdf_directory / source["source_file"], last_section_page)
    findings = []
    if page_count != source["source_pages"]:
        findings.append({"section": None, "problem": f"PDF has {page_count} pages, expected {source['source_pages']}"})
    act_title = " ".join(_words(source["act_name"].split(",")[0]))
    if act_title not in " ".join(words[:300]):
        findings.append({"section": None, "problem": "Act name not found at the start of the PDF"})

    sections = source["sections"]
    positions = _find_headings(sections, words, line_starts)
    for index, section in enumerate(sections):
        start = positions[index]
        if start is None:
            findings.append({"section": section["section"], "problem": "heading not found in order on its first page"})
            continue
        next_headings = [p for p in positions[index + 1:] if p is not None]
        end = min(
            next_headings[0] if next_headings else len(words),
            bisect.bisect_right(pages, section["pages"][-1]),
        )
        for problem in _review_section(section, words[start:end]):
            findings.append({"section": section["section"], "problem": problem})
    return {
        "act_id": source["act_id"],
        "source_version": source["source_version"],
        "sections_checked": len(sections),
        "status": "failed" if findings else "passed",
        "findings": findings,
    }


def review_sources(config: dict) -> dict:
    """Review every Act in the loaded section corpus against its PDF."""
    corpus = _read_json(_config_value(config, "output_file"))
    pdf_directory = pathlib.Path(_config_value(config, "pdf_directory"))
    return {"reviews": [review_source(source, pdf_directory) for source in corpus["sources"]]}


def load_verified_source(config: dict, act_id: str) -> dict:
    """Return the loaded Act only when its check passed for this exact source version."""
    corpus = _read_json(_config_value(config, "output_file"))
    reviews = _read_json(_config_value(config, "review_file"))["reviews"]
    source = next((s for s in corpus["sources"] if s["act_id"] == act_id), None)
    if source is None:
        raise ValueError(f"Unknown Act: {act_id}")
    review = next((r for r in reviews if r["act_id"] == act_id), None)
    if review is None or review["status"] != "passed" or review["source_version"] != source["source_version"]:
        raise ValueError(f"Act has not passed its source check: {act_id}")
    return source


def main() -> None:
    """Check the loaded sources against their PDFs and write a new review file."""
    parser = argparse.ArgumentParser(description="Check POC law sources against their PDFs")
    parser.add_argument("--config", required=True, help="Path to the source-loading JSON config")
    args = parser.parse_args()
    try:
        config = _read_json(args.config)
        review_file = pathlib.Path(_config_value(config, "review_file"))
        if review_file.exists():
            raise ValueError(f"Review file already exists; choose a new review_file: {review_file}")
        report = review_sources(config)
        review_file.parent.mkdir(parents=True, exist_ok=True)
        with review_file.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for review in report["reviews"]:
        print(f"{review['act_id']}: {review['status']} ({len(review['findings'])} findings)")
    print(f"Source review written to {review_file}")


if __name__ == "__main__":
    main()
