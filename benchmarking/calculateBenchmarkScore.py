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

    # Calculate the average score for each section
    section_avgs = {}
    for s, ids in section_ids.items():
        scores = [
            by_id[qid]["evaluation"]["total"]
            for qid in ids
            if qid in by_id
        ]
        section_avgs[s] = sum(scores) / len(scores) if scores else 0

    # Calculate the overall benchmark score
    overall_score = sum(section_avgs.values()) / len(section_avgs) if section_avgs else 0
    # Create the report
    report = {
        "evaluator_file": str(batch_path),
        "question_set_file": str(question_set_path),
        "section_scores_avg": section_avgs,
        "benchmark_score": overall_score,
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

    batch_path = pathlib.Path(batchResultFile).resolve()
    output_path = pathlib.Path(outputpath).resolve()
    question_set_path = pathlib.Path(questionSetFile).resolve()

    calculate_benchmark_score(batch_path, output_path, question_set_path)


if __name__ == "__main__":
    main()