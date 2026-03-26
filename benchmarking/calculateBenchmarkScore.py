import argparse
import json
import pathlib



def calculate_benchmark_score(
    batch_path: pathlib.Path,
    output_path: pathlib.Path,
    question_set_path: pathlib.Path,
) -> dict:
    """Calculate benchmark score from evaluation results and question set.
    Args:
        batch_path: Path to the batch result file.
        output_path: Path to the output file.
        question_set_path: Path to the question set file.
    Returns:
        A dictionary containing the benchmark score and the section scores.
    """
    # Load the evaluation results and question set
    evaluation = json.loads(batch_path.read_text(encoding="utf-8"))
    questions = json.loads(question_set_path.read_text(encoding="utf-8"))

    # Group questions by id
    by_id = {r["question_id"]: r for r in evaluation["results"]}

    # Group questions by section
    section_ids = {}
    for q in questions:
        section_ids.setdefault(q["section"], []).append(q["id"])

    # Calculate the average score and percentage for each section,
    # and track global aggregates across all questions.
    sections = {}
    all_scores = []
    all_percentages = []

    for section, ids in section_ids.items():

        scores = []
        percentages = []
        explanation_provided = 0
        citations = 0
        retrieved = []

        for qid in ids:
            if qid not in by_id:
                continue

            r = by_id[qid]

            total = r["layer2"]["total"]
            scores.append(total["score"])
            percentages.append(total["percentage"])
            all_scores.append(total["score"])
            all_percentages.append(total["percentage"])

            l1 = r["layer1"]

            if l1["explanation_provided"]:
                explanation_provided += 1

            if l1["num_citations"] > 0:
                citations += 1

            retrieved.append(l1["num_retrieved_chunks"])

        n = len(scores)

        sections[section] = {
            # Average total score and percentage for this section
            "avg_score": round(sum(scores) / n, 2) if n else 0,
            "avg_percentage": round(sum(percentages) / n, 2) if n else 0,

            "support_metrics": {
                # Store rates as percentages (0–100)
                "explanation_provided_rate": round(explanation_provided / n * 100, 2) if n else 0,
                "citation_rate": round(citations / n * 100, 2) if n else 0,
                # This is a count, keep as-is
                "avg_retrieved_chunks": sum(retrieved) / n if n else 0
            }
        }

    # Overall benchmark: average score and percentage across all questions
    overall_avg_score = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0
    overall_avg_percentage = round(sum(all_percentages) / len(all_percentages), 2) if all_percentages else 0

    # Create the report
    report = {
        "evaluator_file": str(batch_path),
        "question_set_file": str(question_set_path),

        # Overall averages across all questions
        "benchmark_avg_score": overall_avg_score,
        "benchmark_avg_percentage": overall_avg_percentage,

        "sections": sections
    }

    # Write the report to the output file
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report

def main():
    """Parse CLI for batchResultFile and outputpath, then calculate the benchmark score."""
    parser = argparse.ArgumentParser(
        description="Calculate the benchmark score for a given configuration"
    )

    parser.add_argument("--config",
     type=str, 
     required=True,
     help="Path to the configuration file")

    args = parser.parse_args()

    config_path = pathlib.Path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    
    config = json.loads(config_path.read_text(encoding="utf-8"))

    batchResultFile = config.get("batchResultFile")
    outputpath = config.get("outputpath")
    questionSetFile = config.get("questionSetFile")
    if not questionSetFile:
        raise ValueError("Config must provide 'questionSetFile'")

    if not batchResultFile:
        raise ValueError("Config must provide 'batchResultFile'")

    if not outputpath:
        raise ValueError("Config must provide 'outputpath'")

    batch_path = pathlib.Path(batchResultFile)
    output_path = pathlib.Path(outputpath)
    question_set_path = pathlib.Path(questionSetFile)

    calculate_benchmark_score(batch_path, output_path, question_set_path)


if __name__ == "__main__":
    main()