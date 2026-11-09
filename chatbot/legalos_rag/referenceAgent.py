"""Resolve a named Act and section in a question to a verified, complete provision."""

import argparse
import json
import pathlib
import re

from chatbot.legalos_rag import sectionLookup, sourceReview


_SECTION_REFERENCE = re.compile(
    r"\b(?:sections?|sec\.?|s\.)\s*(\d+[A-Za-z]?(?:\s*(?:,|and|&)\s*\d+[A-Za-z]?)*)((?:\s*\([0-9A-Za-z]+\))*)",
    re.IGNORECASE,
)
_CONNECTOR = re.compile(r"\s*(?:of|under|in|to|from)\s+(?:the\s+)?", re.IGNORECASE)
_ACT_LIST_JOINER = re.compile(r"\s*(?:,|and|or|&)\s+(?:the\s+)?", re.IGNORECASE)
# More title words before "Act" mean a different Act, such as an amending Act.
_LONGER_TITLE = r"(?!\s*(?:\([^)]*\)\s*Act\b|(?:(?-i:[A-Z])\w*\s+){1,3}Act\b))"
_OTHER_ACT = re.compile(r"\b(?:[A-Z][\w&-]*\s+){1,6}Act\b")
_YEAR = re.compile(r"\b(\d{4})\b")


def _act_patterns(config: dict) -> dict[str, re.Pattern]:
    """Build one pattern per Act from its name without parentheses, and its aliases."""
    patterns = {}
    for source in config["sources"]:
        name = re.sub(r"\([^)]*\)", " ", source["act_name"]).split(",")[0]
        core = re.sub(r"\s+Act$", "", " ".join(name.split()))
        names = sorted([core] + source.get("aliases", []), key=len, reverse=True)
        alternatives = "|".join(re.escape(n) for n in names)
        patterns[source["act_id"]] = re.compile(
            rf"\b(?:{alternatives}){_LONGER_TITLE}(?:\s+Act(?:\s*,?\s*(\d{{4}}))?)?\b", re.IGNORECASE
        )
    return patterns


def _find_acts(question: str, config: dict) -> list[dict]:
    """Return every Act named in the question, known or not, in order of appearance."""
    mentions = []
    for act_id, pattern in _act_patterns(config).items():
        for match in pattern.finditer(question):
            mentions.append({"act_id": act_id, "start": match.start(), "end": match.end(), "year": match.group(1)})
    for match in _OTHER_ACT.finditer(question):
        if not any(m["start"] < match.end() and match.start() < m["end"] for m in mentions):
            mentions.append({"act_id": None, "start": match.start(), "end": match.end(),
                             "year": None, "name": match.group(0)})
    return sorted(mentions, key=lambda m: m["start"])


def _find_references(question: str, acts: list[dict]) -> list[dict]:
    """Return each section named in the question with each Act written right after it, if any."""
    references = []
    for match in _SECTION_REFERENCE.finditer(question):
        connector = _CONNECTOR.match(question, match.end())
        named = []
        position = connector.end() if connector else None
        while position is not None:
            act = next((a for a in acts if a["start"] == position), None)
            if act is None:
                break
            named.append(act)
            joiner = _ACT_LIST_JOINER.match(question, act["end"])
            position = joiner.end() if joiner else None
        subsection = re.sub(r"\s+", "", match.group(2))
        for number in re.findall(r"\d+[A-Za-z]?", match.group(1)):
            for act in named or [None]:
                references.append({"text": question[match.start():act["end"] if act else match.end()],
                                   "section": number.upper(), "subsection": subsection, "act": act})
    return references


def _candidates(config: dict, number: str) -> list[dict]:
    """List the checked Acts that have a section with this number."""
    found = []
    for source in config["sources"]:
        try:
            verified = sourceReview.load_verified_source(config, source["act_id"])
        except ValueError:
            continue
        for section in verified["sections"]:
            if section["section"] == number:
                found.append({"act_id": verified["act_id"], "act_name": verified["act_name"],
                              "section": number, "title": section["title"], "source_id": section["source_id"]})
    return found


def _resolve(config: dict, reference: dict, acts: list[dict]) -> dict:
    result = {"reference": reference["text"], "section": reference["section"], "subsection": reference["subsection"]}
    act = reference["act"]
    if act is None:
        named = {a["act_id"] for a in acts}
        if len(named) == 1 and None not in named:
            act = next(a for a in acts if a["act_id"])
    if act is not None and act["act_id"] is None:
        return {**result, "status": "unsupported_act", "reason": f"{act['name']} is not one of the supported Acts"}
    if act is None:
        return {**result, "status": "ambiguous", "reason": "The Act is not named once for this section",
                "candidates": _candidates(config, reference["section"])}

    source = next(s for s in config["sources"] if s["act_id"] == act["act_id"])
    if act["year"] and act["year"] != _YEAR.findall(source["act_name"])[-1]:
        return {**result, "status": "not_found", "act_id": source["act_id"],
                "reason": f"{source['act_name']} is the supported version, not {act['year']}"}
    try:
        verified = sourceReview.load_verified_source(config, source["act_id"])
    except ValueError as error:
        return {**result, "status": "act_not_verified", "act_id": source["act_id"], "reason": str(error)}
    section = next((s for s in verified["sections"] if s["section"] == reference["section"]), None)
    if section is None:
        return {**result, "status": "not_found", "act_id": source["act_id"],
                "reason": f"Section {reference['section']} is not a numbered section of {verified['act_name']}"}
    return {**result, "status": "verified", "act_id": source["act_id"],
            "provision": sectionLookup.fetch_section(config, section["source_id"])}


def resolve_references(config: dict, question: str) -> list[dict]:
    """Return one result for each section number named in the question.

    A result is `verified` with the full provision, or says why not:
    `ambiguous` (the Act is missing or unclear, with candidate sections),
    `unsupported_act`, `act_not_verified`, or `not_found`. A question that
    names no section returns an empty list. No search score is used.
    """
    acts = _find_acts(question, config)
    return [_resolve(config, reference, acts) for reference in _find_references(question, acts)]


def main() -> None:
    """Print the resolved references of one question as JSON."""
    parser = argparse.ArgumentParser(description="Resolve a named Act and section to a verified provision")
    parser.add_argument("--config", required=True, help="Path to the source-loading JSON config")
    parser.add_argument("--question", required=True, help="Question that names an Act and section")
    args = parser.parse_args()
    try:
        config = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
        results = resolve_references(config, args.question)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
