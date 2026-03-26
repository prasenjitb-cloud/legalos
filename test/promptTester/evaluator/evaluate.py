"""
Evaluate RAG batch runs using an LLM evaluator.

Reads a JSONL file produced by promptRunBatch (line 0 = metadata, lines 1+ = per-question
results with question, retrieved_chunks, output). For each result, invokes the evaluator LLM
to score factual_existence, factual_faithfulness, query_relevance, legal_precision, clarity,
citation_quality, and explanation_from_citations, plus computed `total` and `percentage`.
Writes a single evaluation JSON (run_metadata, aggregate_scores, results) to the configured
output path.
"""
import dotenv
dotenv.load_dotenv()

import argparse
import datetime
import pathlib
import json

import langchain_groq
import re
import langchain_core

import test.promptTester.evaluator.evaluatorPrompt


def _format_model_answer(output):
    """Format RAG output (dict or None) into explanation and citations for the evaluator prompt.

    Args:
        output: RAG output from batch JSONL — a dict (LegalAnswer-like with answer_found,
            explanation, citations) or None when no chunks were retrieved.

    Returns:
        (explanation, citations): Tuple of strings. When no answer: ("Answer not found in the
        provided facts.", "(none)"). Otherwise (explanation text, formatted citations).
    """
    if output is None:
        return "Answer not found in the provided facts.", "(none)"
    if isinstance(output, dict) and not output.get("answer_found", False):
        return "Answer not found in the provided facts.", "(none)"

    citations = output.get("citations", [])
    citations_text = "\n".join(
        f"Page {c.get('page', '')}: {c.get('quote', '')}"
        for c in citations
    )
    explanation = output.get("explanation") or "(no explanation)"
    citations_str = citations_text if citations_text else "(none)"

    return explanation, citations_str

# -------------------- GLOBAL VARIABLES --------------------

EVALUATOR_MODEL_NAME = "llama-3.3-70b-versatile"

# -------------------- EVALUATOR LLM SETUP --------------------

def _setup_evaluator():
    """Build the Groq chat model used as the RAG evaluator (EVALUATOR_MODEL_NAME)."""
    return langchain_groq.ChatGroq(
        model_name=EVALUATOR_MODEL_NAME,
        temperature=0,
    )


def _load_batch_results(path: pathlib.Path):
    """Load a promptRunBatch JSONL file into metadata and per-question results.

    Args:
        path: Path to the JSONL file (first line = run metadata, rest = one JSON object per question).

    Returns:
        (metadata, data): metadata is the first line parsed as a dict, or None if file is empty;
        data is a list of result dicts (question_id, question, retrieved_chunks, output), or [].
    """
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return None, []

    metadata = json.loads(lines[0])
    data = [json.loads(line) for line in lines[1:]]
    return metadata, data




def evaluate_layer1(result: dict) -> dict:
    """Simple structural evaluation of RAG output before LLM evaluation."""

    output = result.get("output")
    retrieved_chunks = result.get("retrieved_chunks")

    # ---- explanation detection ----
    explanation_provided = False
    citations = []

    if isinstance(output, dict):
        citations = output.get("citations", []) or []
        explanation = output.get("explanation")
        explanation_provided = bool(explanation and str(explanation).strip())

    # ---- Retrieved chunks ----
    num_chunks = len(re.findall(r"\[DOC\s+\d+\]", retrieved_chunks))
    
    # ---- Citations ----
    num_citations = len(citations)

    return {
        "explanation_provided": explanation_provided,
        "num_retrieved_chunks": num_chunks,
        "num_citations": num_citations,
    }

# -------------------- MAIN EVALUATOR LOOP --------------------

def evaluate_rag(batch_result_file: pathlib.Path, outputpath: pathlib.Path) -> None:
    """Run the evaluator LLM on each batch result and write a single evaluation JSON.

    Loads the JSONL from batch_result_file, skips rows with an "error" key, and for each
    valid result invokes the evaluator to produce RAGEvaluation scores. Aggregates score
    sums and writes evaluation_<run_id>.json to outputpath (run_metadata, aggregate_scores,
    results). Creates outputpath if it does not exist.

    Args:
        batch_result_file: Path to the JSONL produced by promptRunBatch.
        outputpath: Directory to write evaluation_<run_id>.json into.
    """
    evaluator = _setup_evaluator()
    metadata, batch_results = _load_batch_results(batch_result_file)

    if metadata is None:
        raise ValueError("Batch result file is empty or invalid")

    all_results = []
    score_sums = {
        "factual_existence": 0,
        "factual_faithfulness": 0,
        "query_relevance": 0,
        "legal_precision": 0,
        "clarity": 0,
        "citation_quality": 0,
        "explanation_from_citations": 0,
        "total": 0,
        # Computed field on the RAGEvaluation schema (0-100).
        "percentage": 0.0,
    }
    evaluated_count = 0

    # Parser and prompt for evaluator LLM (RAGEvaluation schema)
    evaluator_parser = langchain_core.output_parsers.PydanticOutputParser(
        pydantic_object=test.promptTester.evaluator.evaluatorPrompt.RAGEvaluation
    )
    evaluator_prompt = test.promptTester.evaluator.evaluatorPrompt.setup_evaluator_prompt(
        evaluator_parser
    )

    for result in batch_results:
        if "error" in result:
            print(f"[SKIP] Q{result.get('question_id', '?')}: {result['error']}")
            continue

        # Normalize facts to string (batch JSONL may have string or list)
        facts = result.get("retrieved_chunks", "")
        if isinstance(facts, list):
            facts = "\n".join(str(x) for x in facts)

        layer1 = evaluate_layer1(result)

        explanation, citations = _format_model_answer(result.get("output"))
        final_evaluator_prompt = evaluator_prompt.format(
            question=result["question"],
            facts=facts,
            model_answer=explanation,
            citations=citations,
        )

        raw_response_evaluator = evaluator.invoke(final_evaluator_prompt)
        raw_eval_text = (
            raw_response_evaluator.content
            if hasattr(raw_response_evaluator, "content")
            else raw_response_evaluator
        )

        try:
            evaluator_result = evaluator_parser.parse(raw_eval_text)
        except Exception as e:
            print(f"[WARN] Evaluator failed for Q{result['question_id']}: {e}")
            continue

        output = result.get("output")
        raw = evaluator_result.model_dump()

        def metric(score, max_score):
            return {
                "score": score,
                "percentage": round((score / max_score) * 100, 2)
            }


        layer2 = {
            "factual_existence": metric(raw["factual_existence"], 1),
            "factual_faithfulness": metric(raw["factual_faithfulness"], 5),
            "query_relevance": metric(raw["query_relevance"], 5),
            "legal_precision": metric(raw["legal_precision"], 4),
            "clarity": metric(raw["clarity"], 3),
            "citation_quality": metric(raw["citation_quality"], 5),
            "explanation_from_citations": metric(raw["explanation_from_citations"], 5),
            "total": {
                "score": raw["total"],
                "percentage": raw["percentage"]
            }
        }
        

        result_entry = {
            "question_id": result["question_id"],
            "question": result["question"],
            "rag_output": output,
            "layer1": layer1,
            "layer2": layer2,
        }
        all_results.append(result_entry)

        answer_found = output.get("answer_found", False) if isinstance(output, dict) else False
        print(
            f"[{result['question_id']}] run complete | "
            f"answer_found={answer_found} | "
            f"score={evaluator_result.total} | "
            f"percentage={round(evaluator_result.percentage, 2)}"
        )

        for k in score_sums:
            score_sums[k] += getattr(evaluator_result, k)
        evaluated_count += 1

    # Averages per dimension (empty dict if none evaluated)
    aggregate_scores = (
        {f"{k}_avg": round(score_sums[k] / evaluated_count, 2) for k in score_sums}
        if evaluated_count > 0
        else {}
    )

    # Max scores per dimension (from RAGEvaluation schema)
    max_scores = {
        "factual_existence": 1,
        "factual_faithfulness": 5,
        "query_relevance": 5,
        "legal_precision": 4,
        "clarity": 3,
        "citation_quality": 5,
        "explanation_from_citations": 5,
        "total": 28,
        "percentage": 100.0,
    }

    # Single JSON: run_metadata, aggregate_scores, per-question results
    final_output = {
        "run_metadata": {
            "run_id": metadata["run_id"],
            "slm_model": metadata["model"],
            "evaluator_model": EVALUATOR_MODEL_NAME,
            "num_questions": len(batch_results),
            "num_evaluated": evaluated_count,
            "max_scores": max_scores,
        },
        "aggregate_scores": aggregate_scores,
        "results": all_results,
    }

    # Ensure output dir exists and write evaluation JSON
    outputpath.mkdir(parents=True, exist_ok=True)
    eval_id = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    out_file = outputpath / f"evaluation_{metadata['run_id']}_{eval_id}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"Evaluation written to {out_file}")





# -------------------- ENTRY POINT --------------------

def main() -> None:
    """Parse --config, load batchResultFile and outputpath, then run evaluate_rag."""
    parser = argparse.ArgumentParser(
        description="Run Legalos RAG evaluator"
    )

    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to the config file"
    )

    args = parser.parse_args()

    # Resolve config file path to absolute path
    config_path = pathlib.Path(args.config).resolve()

    if not config_path.is_file():
        raise ValueError(f"Config file does not exist: {config_path}")

    # Load configuration from JSON file
    with config_path.open("r", encoding="utf-8") as f:
        config: dict = json.load(f)

    batchResultFile = config.get("batchResultFile")
    outputpath = config.get("outputpath")
    if not batchResultFile:
        raise ValueError("Config must provide 'batchResultFile'")

    if not outputpath:
        raise ValueError("Config must provide 'outputpath'")

    batchResultFile = pathlib.Path(batchResultFile).resolve()
    outputpath = pathlib.Path(outputpath).resolve()

    if not batchResultFile.is_file():
        raise ValueError(f"Batch result file does not exist: {batchResultFile}")

    evaluate_rag(batch_result_file=batchResultFile, outputpath=outputpath)


if __name__ == "__main__":
    main()
