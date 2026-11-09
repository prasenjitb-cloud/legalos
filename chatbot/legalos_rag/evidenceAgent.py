"""Gather sourced evidence for each part of a question from one verified Act."""

import argparse
import json
import pathlib

import langchain_ollama
import pydantic

from chatbot.legalos_rag import evidenceFilter, questionSplitter, retriever, structuredCall


_MAX_SECTIONS = 3


class _Target(pydantic.BaseModel):
    target: str = pydantic.Field(
        ..., description="Short phrase (under 8 words) naming the legal detail the question asks for"
    )


_TARGET_PROMPT = (
    "Name the legal detail that a question asks for, as a short search phrase.\n\n"
    "RULES:\n"
    "- Use under 8 words, such as 'complaint filing process', 'definition of sexual harassment' "
    "or 'penalty for non-compliance'.\n"
    "- Describe what is wanted, not an answer. Do not name an Act or a section number.\n\n"
    "Output ONLY the JSON object below, with no explanation.\n\n"
    "{format_instructions}\n\n"
    "Question:\n{question}"
)

def _target_for(part_question: str, slm) -> str:
    try:
        return structuredCall.ask_model(slm, _TARGET_PROMPT, _Target, question=part_question).target.strip()
    except Exception:
        return ""


def _gather_part(config: dict, vectorstore, act_id: str, part: dict, slm) -> dict:
    target = _target_for(part["question"], slm)
    found = retriever.retrieve(config, vectorstore, act_id, part["question"], target, None, _MAX_SECTIONS)
    checked = evidenceFilter.filter_evidence(part["question"], target, found["results"], slm)
    kept = checked["kept"]
    result = {
        "id": part["id"],
        "question": part["question"],
        "target": target,
        "status": "found" if kept else "not_found",
        "evidence": [{
            "source_id": item["section"]["source_id"],
            "section": item["section"]["section"],
            "title": item["section"]["title"],
            "support_quote": item["support_quote"],
            "found_by": item["found_by"],
            "provision": item["section"],
        } for item in kept],
        "rejected": [{"section": item["section"]["section"], "reason": item["reason"]} for item in checked["rejected"]],
        "unchecked": [{"section": item["section"]["section"], "reason": item["reason"]} for item in checked["unchecked"]],
    }
    if kept:
        result["completeness"] = "not_checked"
    else:
        result["missing"] = f"No section of the Act was found that states: {target or part['question']}"
    return result


def gather_evidence(config: dict, vectorstore, act_id: str, question: str, slm) -> dict:
    """Split the question and return sourced evidence from one verified Act for each part.

    Each part has a `status` (`found` or `not_found`), the `evidence` sections that passed
    the evidence check with their support sentence and complete provision, and the sections
    `rejected` or left `unchecked`. A part with no passing section names what was not found
    in `missing`. For a found part `completeness` is `not_checked`: whether the sections
    answer everything the part asks is left to the evaluator.
    """
    split = questionSplitter.split_question(question, slm)
    return {
        "act_id": act_id,
        "question": question,
        "parts": [_gather_part(config, vectorstore, act_id, part, slm) for part in split["parts"]],
    }


def main() -> None:
    """Print the evidence gathered for each part of one question as JSON."""
    parser = argparse.ArgumentParser(description="Gather sourced evidence for a question from one verified POC Act")
    parser.add_argument("--config", required=True, help="Path to the source-loading JSON config")
    parser.add_argument("--rag-config", required=True, help="Path to the RAG JSON config (vector store and model)")
    parser.add_argument("--act", required=True, help="Act ID, such as posh_2013")
    parser.add_argument("--question", required=True, help="Question to gather evidence for")
    args = parser.parse_args()
    try:
        config = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
        rag_config = json.loads(pathlib.Path(args.rag_config).read_text(encoding="utf-8"))
        slm = langchain_ollama.ChatOllama(model=rag_config["model"]["model_name"], temperature=0)
        vectorstore = retriever.open_vectorstore(rag_config["vectordbpath"])
        result = gather_evidence(config, vectorstore, args.act, args.question, slm)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    for part in result["parts"]:
        for item in part["evidence"]:
            item["provision"] = {key: item["provision"][key] for key in ("source_version", "pages")}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
