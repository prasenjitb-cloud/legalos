"""Check an answer against its sources: citations, unsupported claims, and omitted conditions."""

import argparse
import json
import pathlib
import re

import langchain_ollama

from chatbot.legalos_rag import answerAgent, contextCompactor, evidenceFilter, retriever, sectionLookup


_HEDGES = ("as far as possible", "as far as practicable", "wherever possible", "where possible",
           "endeavour", "ordinarily")
_PROVISO = re.compile(r"\bProvid(?:ed|e)\b")
_SECTION_MENTION = re.compile(r"\bsections?\s+(\d+[A-Z]?)", re.IGNORECASE)
_QUOTED = re.compile(r"[\"“][^\"”]*[\"”]")
_SUBSECTION_START = re.compile(r"^\W*\(\d+\)")
_NUMBER = (r"(?:\d[\d,]*|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
           r"sixteen|eighteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)")
_LIMIT = re.compile(
    rf"\b({_NUMBER}(?:[\s-]+{_NUMBER})?)\s*(?:days?|weeks?|months?|years?|hours?|lakhs?|crores?|rupees|per\s+cent)",
    re.IGNORECASE)
_NUMBER_WORDS = {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
                 "8": "eight", "9": "nine", "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
                 "14": "fourteen", "15": "fifteen", "16": "sixteen", "18": "eighteen", "20": "twenty",
                 "21": "twenty one", "25": "twenty five", "30": "thirty", "40": "forty", "45": "forty five",
                 "50": "fifty", "60": "sixty", "90": "ninety"}
_MIN_WORD = 5
_PREFIX = 5
_MIN_COVERED_SHARE = 0.4


def _limit_words(text: str) -> set[str]:
    """Numbers that are limits (a number with a unit such as months or lakh), written out as words."""
    found = set()
    for match in _LIMIT.finditer(text):
        for token in re.findall(r"[a-z]+|\d+", match.group(1).lower().replace(",", "")):
            found.add(_NUMBER_WORDS.get(token, token))
    return found


def _prefixes(text: str) -> set[str]:
    """Word beginnings of five letters or more, so 'reasons' matches 'reason'."""
    return {word[:_PREFIX] for word in re.findall(r"[a-z]+", text.lower()) if len(word) >= _MIN_WORD}


def _scope(section_text: str, support: str, topic: set[str]) -> list[str]:
    """The sub-sections that matter for the part: the one with the support sentence and any
    that share a content word with the question or target. Provisos stay with their sub-section."""
    blocks = []
    for sentence in evidenceFilter.split_sentences([section_text]):
        if blocks and not _SUBSECTION_START.match(sentence):
            blocks[-1].append(sentence)
        else:
            blocks.append([sentence])
    return [" ".join(block) for block in blocks
            if support in block or topic & set(re.findall(r"[a-z]+", " ".join(block).lower()))]


def _clauses(scope: str) -> list[str]:
    """The sub-section's main clause and each proviso, as exact slices of its text."""
    starts = [match.start() for match in _PROVISO.finditer(scope)]
    bounds = [0] + starts + [len(scope)]
    return [scope[a:b].strip() for a, b in zip(bounds, bounds[1:]) if scope[a:b].strip()]


def _omissions(answer_text: str, scope: str) -> list[dict]:
    """Conditions in the scope that the answer leaves out, each with the exact source text."""
    answer_limits = _limit_words(answer_text) | {_NUMBER_WORDS.get(n, n) for n in answerAgent.numbers_in(answer_text)}
    answer_words = _prefixes(answer_text)
    lowered = answer_text.lower()
    findings = []
    for clause in _clauses(scope):
        missing_numbers = sorted(_limit_words(clause) - answer_limits)
        if missing_numbers:
            findings.append({"kind": "omitted_limit", "detail": f"The answer does not state {', '.join(missing_numbers)}",
                             "source_text": clause})
            continue
        hedges = [h for h in _HEDGES if h in clause.lower()]
        if hedges and not any(h in lowered for h in hedges) and _limit_words(clause) & answer_limits:
            findings.append({"kind": "hedge_dropped",
                             "detail": f"The answer states a limit without its qualification: {hedges[0]}",
                             "source_text": clause})
            continue
        if _PROVISO.match(clause):
            words = _prefixes(clause)
            if words and len(words & answer_words) / len(words) < _MIN_COVERED_SHARE:
                findings.append({"kind": "omitted_proviso", "detail": "The answer does not mention this proviso",
                                 "source_text": clause})
    return findings


def _part_findings(config: dict, answer_part: dict, evidence_part: dict) -> list[dict]:
    findings = []
    if evidence_part["status"] != "found":
        return [{"kind": "not_found", "detail": evidence_part["missing"], "source_text": ""}]
    explanation = answer_part["explanation"]
    cited_ids = {c["source_id"] for c in answer_part["citations"]}
    evidence_ids = {item["source_id"] for item in evidence_part["evidence"]}
    if not cited_ids:
        findings.append({"kind": "citation_missing", "detail": "A found part cites no section", "source_text": ""})
    for citation in answer_part["citations"]:
        try:
            fetched = sectionLookup.fetch_section(config, citation["source_id"])
        except ValueError as error:
            findings.append({"kind": "citation_invalid", "detail": str(error), "source_text": ""})
            continue
        if citation["source_id"] not in evidence_ids:
            findings.append({"kind": "citation_not_evidence",
                             "detail": f"Section {citation['section']} is not evidence for this part", "source_text": ""})
        if (fetched["section"], fetched["act_name"]) != (citation["section"], citation["act_name"]):
            findings.append({"kind": "citation_mismatch",
                             "detail": f"Citation says section {citation['section']} but the ID is section {fetched['section']}",
                             "source_text": ""})
    cited_numbers = {c["section"].upper() for c in answer_part["citations"]}
    for number in _SECTION_MENTION.findall(_QUOTED.sub(" ", explanation)):
        if number.upper() not in cited_numbers:
            findings.append({"kind": "section_not_cited",
                             "detail": f"The answer mentions section {number}, which it does not cite", "source_text": ""})
    source_text = " ".join(item["provision"]["text"] for item in evidence_part["evidence"])
    if not answerAgent.numbers_in(explanation) <= answerAgent.numbers_in(source_text):
        findings.append({"kind": "unsupported_number", "detail": "The answer has a number that is not in the cited sections",
                         "source_text": ""})
    if answerAgent.SECOND_PERSON.search(explanation):
        findings.append({"kind": "unsupported_conclusion", "detail": "The explanation speaks to the user about their own case",
                         "source_text": ""})
    template = answerAgent.SITUATION.get(answer_part["finding"], "")
    fixed_start = template.split("{ask}")[0].strip()
    if not (answer_part["situation"].startswith(fixed_start) if template else answer_part["situation"] == ""):
        findings.append({"kind": "situation_not_fixed", "detail": "The situation sentence is not one of the fixed sentences",
                         "source_text": ""})
    topic = contextCompactor.topic_words(evidence_part["question"], evidence_part.get("target", ""))
    for item in evidence_part["evidence"]:
        for scope in _scope(item["provision"]["text"], item["support_quote"], topic):
            findings += _omissions(explanation + " " + answer_part["situation"], scope)
    return findings


def evaluate_answer(config: dict, answer: dict, evidence: dict) -> dict:
    """Check the output of `answerAgent.write_answer` against the evidence it was written from.

    Citations are checked in code: each source ID must read back through `sectionLookup` for a
    verified Act, belong to the part's own evidence, and match the cited Act and section; a section
    mentioned in the prose must be cited. Numbers must come from the cited sections and the
    explanation must not speak to the user's own case. Omissions are found by comparing the answer
    with the sub-section that holds the support sentence: a number, a proviso or a hedge such as
    "as far as possible" that the answer leaves out is reported with the exact source text.
    No model is used. Verdicts: `fail` (a citation or unsupported-claim problem), `incomplete`
    (something is missing or not found) or `pass`; `missing_details` lists what to look for next.
    """
    answer_parts = {part["id"]: part for part in answer["parts"]}
    parts, missing = [], []
    for evidence_part in evidence["parts"]:
        answer_part = answer_parts.get(evidence_part["id"])
        if answer_part is None:
            findings = [{"kind": "part_missing", "detail": "The answer has no entry for this question part", "source_text": ""}]
        else:
            findings = _part_findings(config, answer_part, evidence_part)
        kinds = {finding["kind"] for finding in findings}
        errors = kinds - {"omitted_limit", "hedge_dropped", "omitted_proviso", "not_found", "part_missing"}
        verdict = "fail" if errors else "incomplete" if findings else "pass"
        parts.append({"id": evidence_part["id"], "question": evidence_part["question"], "verdict": verdict, "findings": findings})
        missing += [finding["source_text"] or finding["detail"] for finding in findings
                    if finding["kind"] in ("omitted_limit", "hedge_dropped", "omitted_proviso", "not_found")]
    verdicts = [part["verdict"] for part in parts]
    overall = "fail" if "fail" in verdicts else "incomplete" if "incomplete" in verdicts else "pass"
    return {"verdict": overall, "parts": parts, "missing_details": missing}


def main() -> None:
    """Answer one question, then print the evaluation of that answer as JSON."""
    from chatbot.legalos_rag import applicabilityAgent

    parser = argparse.ArgumentParser(description="Answer a question from one verified POC Act and evaluate the answer")
    parser.add_argument("--config", required=True, help="Path to the source-loading JSON config")
    parser.add_argument("--rag-config", required=True, help="Path to the RAG JSON config (vector store and model)")
    parser.add_argument("--act", required=True, help="Act ID, such as hma_1955")
    parser.add_argument("--question", required=True, help="The user's question, with the facts they state")
    args = parser.parse_args()
    try:
        config = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
        rag_config = json.loads(pathlib.Path(args.rag_config).read_text(encoding="utf-8"))
        slm = langchain_ollama.ChatOllama(model=rag_config["model"]["model_name"], temperature=0)
        vectorstore = retriever.open_vectorstore(rag_config["vectordbpath"])
        assessed = applicabilityAgent.assess_applicability(config, vectorstore, args.act, args.question, slm)
        answer = answerAgent.write_answer(assessed, slm)
        result = evaluate_answer(config, answer, assessed["evidence"])
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(answer["text"])
    print("\n--- Evaluation ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
