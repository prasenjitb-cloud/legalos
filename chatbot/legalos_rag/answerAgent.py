"""Write the answer: the model explains the law for each part; code states what applies to the user."""

import argparse
import json
import pathlib
import re

import langchain_ollama
import pydantic

from chatbot.legalos_rag import applicabilityAgent, contextCompactor, retriever, structuredCall


SECOND_PERSON = re.compile(
    r"\byou(?:r)?\s+(?:are|can|may|will|must|do|have|cannot|can't|should|would|could|need|qualify|"
    r"complaint is|marriage is|petition is|case is)\b|\byou're\b",
    re.IGNORECASE,
)
_NUMBERS = re.compile(
    r"\d+|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|eighteen|twenty|"
    r"thirty|forty|fifty|sixty|ninety|hundred|thousand|lakh|crore)\b",
    re.IGNORECASE,
)
SITUATION = {
    "needs_fact": "Whether this applies to you depends on facts that have not been stated. {ask}",
    "not_met_on_stated_facts": "On the facts you stated, a condition of this rule appears not to be met. "
                               "This is an automated reading of the law and should be checked.",
    "exception_applies": "On the facts you stated, an exception to this rule appears to apply. "
                         "This is an automated reading of the law and should be checked.",
    "conditions_met_on_stated_facts": "On the facts you stated, the conditions listed appear to be met. "
                                      "This is an automated reading of the law, not a decision on your case.",
    "cannot_assess": "Your facts could not be checked against this rule.",
}


class _Explanation(pydantic.BaseModel):
    explanation: str = pydantic.Field(..., description="Plain-language explanation of the law, under 120 words")


_EXPLAIN_PROMPT = (
    "Explain what the law says in answer to a question, using ONLY the sections given.\n\n"
    "RULES:\n"
    "- Use plain language and under 120 words.\n"
    "- State the conditions, exceptions and time limits that the sections give. Do not add any number, "
    "period or amount that is not in the sections.\n"
    "- Explain the law in general terms. Do NOT tell the user what they can, may or must do in their own "
    "case, and do not say whether the law applies to them.\n"
    "- Do not use general knowledge.\n\n"
    "Output ONLY the JSON object below, with no explanation.\n\n"
    "{format_instructions}\n\n"
    "Question:\n{question}\n\n"
    "Sections:\n{sections}"
)


def numbers_in(text: str) -> set[str]:
    return {token.lower() for token in _NUMBERS.findall(text)}


def _quote_explanation(evidence: list[dict]) -> str:
    """A safe explanation made only of the support sentences."""
    return " ".join(f"Section {item['section']} says: \"{item['support_quote']}\"" for item in evidence)


def _explain(question: str, evidence: list[dict], sections: list[dict], slm) -> tuple[str, str]:
    """Return the explanation and where it came from: `model` or `quotes`."""
    text = "\n\n".join(f"[{s['source_id']}]\n{s['text']}" for s in sections)
    try:
        written = structuredCall.ask_model(slm, _EXPLAIN_PROMPT, _Explanation, question=question, sections=text)
    except Exception:
        return _quote_explanation(evidence), "quotes"
    explanation = written.explanation.strip()
    source_numbers = numbers_in(text)
    if not explanation or SECOND_PERSON.search(explanation) or not numbers_in(explanation) <= source_numbers:
        return _quote_explanation(evidence), "quotes"
    return explanation, "model"


def write_answer(result: dict, slm) -> dict:
    """Build the answer from the output of `applicabilityAgent.assess_applicability`.

    Every question part appears exactly once. For a found part the model explains the law from the
    part's sections; the explanation is replaced by the support sentences if it is missing, speaks to
    the user in the second person, or contains a number not in the sections. What applies to the user
    is a fixed sentence chosen by the part's finding, never model text. Citations are built in code from
    the part's own evidence. A part with no evidence says what was not found.
    """
    evidence_parts = {part["id"]: part for part in result["evidence"]["parts"]}
    compact = {section["source_id"]: section for section in
               contextCompactor.compact_evidence(result["evidence"]["parts"])["sections"]}
    parts = []
    for part in result["parts"]:
        found = evidence_parts[part["id"]]
        entry = {"id": part["id"], "question": part["question"], "finding": part["finding"]}
        if part["finding"] == "no_evidence":
            entry.update(explanation="", situation="", citations=[], source="none",
                         uncertainty=part["missing"] + ". No answer is given for this part.")
        else:
            items = found["evidence"]
            sections = [compact[item["source_id"]] for item in items]
            explanation, source = _explain(part["question"], items, sections, slm)
            situation = SITUATION.get(part["finding"], "").format(ask=part.get("ask") or "")
            entry.update(
                explanation=explanation, situation=situation, source=source,
                citations=[{"source_id": item["source_id"], "act_id": item["provision"]["act_id"],
                            "act_name": item["provision"]["act_name"], "section": item["section"],
                            "title": item["title"], "pages": item["provision"]["pages"]} for item in items],
                uncertainty="Based only on the sections cited; other provisions may also be relevant.")
        parts.append(entry)

    lines = []
    for entry in parts:
        lines.append(f"Question: {entry['question']}")
        for key in ("explanation", "situation"):
            if entry[key]:
                lines.append(entry[key])
        if entry["citations"]:
            lines.append("Sources: " + "; ".join(
                f"{c['act_name']}, section {c['section']} (pages {', '.join(map(str, c['pages']))})"
                for c in entry["citations"]))
        lines.append(f"Note: {entry['uncertainty']}")
        lines.append("")
    return {"act_id": result["act_id"], "question": result["question"], "parts": parts, "text": "\n".join(lines).strip()}


def answer_question(config: dict, vectorstore, act_id: str, question: str, slm) -> dict:
    """Gather evidence and facts for the question, assess them, and write the answer."""
    return write_answer(applicabilityAgent.assess_applicability(config, vectorstore, act_id, question, slm), slm)


def main() -> None:
    """Print the answer to one question."""
    parser = argparse.ArgumentParser(description="Answer a question from one verified POC Act, with sources")
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
        answer = answer_question(config, vectorstore, args.act, args.question, slm)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(answer["text"])


if __name__ == "__main__":
    main()
