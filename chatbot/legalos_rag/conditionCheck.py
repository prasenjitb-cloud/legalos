"""Check each condition and exception of a full section against the facts a user stated."""

import argparse
import json
import pathlib
import re
from typing import Literal

import langchain_ollama
import pydantic

from chatbot.legalos_rag import sectionLookup, structuredCall


_MAX_CONDITIONS = 8
_MIN_CONDITION_WORDS = 8
_MAX_SECTION_CHARS = 6000
_MAX_REFERENCE_CHARS = 1500
_SECTION_REFERENCE = re.compile(r"\bsections?\s+(\d+[A-Z]?)", re.IGNORECASE)


class _Condition(pydantic.BaseModel):
    text: str = pydantic.Field(..., description="The condition, proviso or exception, copied exactly from the section")
    kind: Literal["condition", "exception"]


class _Conditions(pydantic.BaseModel):
    conditions: list[_Condition]


class _Verdict(pydantic.BaseModel):
    status: Literal["MET", "NOT_MET", "UNKNOWN"]
    fact_ids: list[str] = pydantic.Field(default_factory=list, description="IDs of the facts the verdict rests on")


_LIST_PROMPT = (
    "List the conditions and exceptions in a section of law that decide whether it applies.\n\n"
    "RULES:\n"
    "- Include each requirement, each 'Provided that' proviso and each exception.\n"
    "- Copy each one exactly from the section, under 250 characters. Do not reword.\n"
    "- kind is 'exception' for a proviso or exception that takes a case out of the rule, "
    "otherwise 'condition'.\n"
    "- Use at most 8 items. Do not add anything that is not in the section.\n\n"
    "Output ONLY the JSON object below, with no explanation.\n\n"
    "{format_instructions}\n\n"
    "Section:\n{section}"
)

_JUDGE_PROMPT = (
    "Decide whether a legal condition is met by the facts a user stated.\n\n"
    "RULES:\n"
    "- Use ONLY the stated facts. Do not assume or guess any fact.\n"
    "- MET: the facts show the condition is satisfied. For an exception, MET means the "
    "exception applies.\n"
    "- NOT_MET: the facts show the condition is not satisfied.\n"
    "- UNKNOWN: the facts do not say, or say only part of it.\n"
    "- fact_ids: the IDs of the facts the verdict rests on. Use [] for UNKNOWN.\n\n"
    "Output ONLY the JSON object below, with no explanation.\n\n"
    "{format_instructions}\n\n"
    "Condition ({kind}):\n{condition}\n\n"
    "{references}"
    "Stated facts:\n{facts}"
)


def _normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _referenced_sections(config: dict, section: dict, condition: str) -> list[dict]:
    """Fetch other sections of the same Act that the condition names."""
    found = []
    for number in dict.fromkeys(_SECTION_REFERENCE.findall(condition)):
        if number.upper() == section["section"].upper():
            continue
        source_id = f"{section['act_id']}:{section['source_version']}:section:{number.upper()}"
        try:
            found.append(sectionLookup.fetch_section(config, source_id))
        except ValueError:
            continue
    return found


def check_conditions(config: dict, section: dict, facts: list[dict], slm) -> dict:
    """Return each condition of a full section with MET, NOT_MET or UNKNOWN against the facts.

    `section` is the output of `sectionLookup.fetch_section`; `facts` are the
    `{"id", "quote"}` records of `factExtractor`. A condition is MET or NOT_MET only when
    the verdict cites facts that exist; otherwise it stays UNKNOWN, and with no facts every
    condition is UNKNOWN. An exception is never NOT_MET, because stated facts cannot show that
    an exception is absent. `overall` is `not_met` when a condition is not met, `excepted` when an
    exception applies, and `met` only when every condition is met and no exception is left
    open; otherwise it is `unknown`. Pass only sections that passed the evidence check.
    """
    result = {"act_id": section["act_id"], "source_id": section["source_id"], "section": section["section"],
              "conditions": [], "overall": "unknown"}
    if len(section["text"]) > _MAX_SECTION_CHARS:
        result["note"] = "The section is too long to check; no condition was assessed"
        return result
    try:
        listed = structuredCall.ask_model(slm, _LIST_PROMPT, _Conditions, section=section["text"]).conditions
    except Exception as error:
        result["note"] = f"Conditions could not be listed: {error}"
        return result

    text = _normalise(section["text"])
    known_facts = {fact["id"]: fact["quote"] for fact in facts}
    real = [c for c in listed if len(_normalise(c.text).split()) >= _MIN_CONDITION_WORDS and _normalise(c.text) in text]
    for number, item in enumerate(real, start=1):
        if number > _MAX_CONDITIONS:
            break
        entry = {"id": f"c{number}", "kind": item.kind, "text": item.text.strip(), "status": "UNKNOWN",
                 "fact_ids": [], "references": []}
        references = _referenced_sections(config, section, item.text)
        entry["references"] = [ref["source_id"] for ref in references]
        if known_facts:
            reference_text = "".join(
                f"Referenced section {ref['section']}:\n{ref['text'][:_MAX_REFERENCE_CHARS]}\n\n" for ref in references)
            fact_text = "\n".join(f"{fact_id}: {quote}" for fact_id, quote in known_facts.items())
            try:
                verdict = structuredCall.ask_model(slm, _JUDGE_PROMPT, _Verdict, kind=item.kind,
                                                   condition=item.text, references=reference_text, facts=fact_text)
                cited = [fact_id for fact_id in verdict.fact_ids if fact_id in known_facts]
                if item.kind == "exception" and verdict.status == "NOT_MET":
                    cited = []
                if verdict.status != "UNKNOWN" and cited:
                    entry["status"], entry["fact_ids"] = verdict.status, cited
            except Exception:
                pass
        entry["facts_used"] = [known_facts[fact_id] for fact_id in entry["fact_ids"]]
        result["conditions"].append(entry)

    conditions = [c["status"] for c in result["conditions"] if c["kind"] == "condition"]
    exceptions = [c["status"] for c in result["conditions"] if c["kind"] == "exception"]
    if "NOT_MET" in conditions:
        result["overall"] = "not_met"
    elif "MET" in exceptions:
        result["overall"] = "excepted"
    elif conditions and all(status == "MET" for status in conditions) and all(status == "NOT_MET" for status in exceptions):
        result["overall"] = "met"
    return result


def main() -> None:
    """Print the condition check of one section against stated facts as JSON."""
    parser = argparse.ArgumentParser(description="Check a verified section's conditions against stated facts")
    parser.add_argument("--config", required=True, help="Path to the source-loading JSON config")
    parser.add_argument("--rag-config", required=True, help="Path to the RAG JSON config (uses model.model_name)")
    parser.add_argument("--id", required=True, help="Section source ID")
    parser.add_argument("--fact", action="append", default=[], help="A fact quoted from the user; repeat for more")
    args = parser.parse_args()
    try:
        config = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
        rag_config = json.loads(pathlib.Path(args.rag_config).read_text(encoding="utf-8"))
        slm = langchain_ollama.ChatOllama(model=rag_config["model"]["model_name"], temperature=0)
        section = sectionLookup.fetch_section(config, args.id)
        facts = [{"id": f"f{n}", "quote": quote} for n, quote in enumerate(args.fact, start=1)]
        result = check_conditions(config, section, facts, slm)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
