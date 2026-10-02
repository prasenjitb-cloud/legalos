"""Write one short answer from selected source units, with checked citations."""

from pydantic import Field

from .model import StrictModel


class AnswerPart(StrictModel):
    need_id: str
    text: str = Field(min_length=1)
    citations: list[str] = Field(min_length=1)


class AnswerDraft(StrictModel):
    parts: list[AnswerPart]


ANSWER_PROMPT = '''Answer each supplied question part using only its source units.
Summarize in plain language, usually within 120 words per question part. Do not copy entire sections.
Preserve the rule's actors, conditions, exceptions and time limits.
Distinguish the main rule from special cases. Do not turn an optional power into a fixed requirement.
Use citations containing only the supplied unit IDs for that part. Do not invent legal references.
Do not add advice, personal facts, outside knowledge or an overall eligibility conclusion.
If a source says other criteria are prescribed elsewhere, retain that qualification.
Return exactly one answer part for each supplied need_id. Combine its sources into that one part; never create separate answer parts for separate source units.'''


def validate_draft(draft, evidence):
    allowed = {row["need_id"]: set(row["unit_ids"]) for row in evidence if row["unit_ids"]}
    if {part.need_id for part in draft.parts} != set(allowed) or len(draft.parts) != len(allowed):
        raise ValueError("Answer each supplied part exactly once")
    for part in draft.parts:
        if set(part.citations) - allowed[part.need_id]:
            raise ValueError("Cite only the source units supplied for this question part")


def compose_answer(model, result, trace):
    if result["status"] not in {"COVERED", "PARTIAL"} or not result["quotes"]:
        trace.event("answer_draft_skipped", {"reason": result["status"]})
        return
    if model.calls >= 10:
        trace.event("answer_draft_skipped", {"reason": "Model-call budget exhausted; retain source excerpts"})
        return
    units = {q["unit_id"]: q["quote"] for q in result["quotes"]}
    needs = {n["id"]: n["question"] for n in result["state"]["needs"]}
    payload = {"act_name": result["act_name"], "source_updated": result["source_updated"],
        "parts": [{"need_id": row["need_id"], "question": needs[row["need_id"]],
                   "sources": [{"id": uid, "text": units[uid]} for uid in row["unit_ids"]],
                   "unresolved": row["missing"]} for row in result["evidence"] if row["unit_ids"]]}
    try:
        draft = model.call(AnswerDraft, ANSWER_PROMPT, payload, trace,
                           lambda value: validate_draft(value, result["evidence"]))
        result["answer"] = draft.model_dump()
        trace.event("answer_draft", {"answer": result["answer"],
            "validation": "Part and citation boundaries checked; semantic faithfulness is evaluated separately."})
    except Exception as exc:
        trace.event("answer_draft_rejected", {"error": str(exc), "fallback": "source excerpts"})
    finally:
        result["model_calls"] = model.calls
