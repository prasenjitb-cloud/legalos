# Benchmarking

Compute benchmark scores from evaluation results, grouped by legal section.

## Overview

The benchmark score aggregates LLM evaluation results across a question set. It computes:

1. **Section averages** — mean `total` score per legal section (POSH, HMA, MVA, etc.)
2. **Overall benchmark score** — mean of all section averages

## Prerequisites

- Evaluation results from the [prompt tester evaluator](../test/promptTester/evaluator/)
- A question set with `id` and `section` for each question

## Usage

```bash
# From legalos root
python -m benchmarking.calculateBenchmarkScore --config benchmarking/benchmarks/config/b1.json
```

## Config Format

Create a JSON config in `benchmarking/benchmarks/config/`:

```json
{
  "batchResultFile": "test/promptTester/evaluator/evaluationResults/evaluation_20260307_143019_20260307_184410.json",
  "questionSetFile": "test/promptTester/questionSet.json",
  "outputpath": "benchmarking/benchmarks/b1"
}
```

| Key | Description |
|-----|-------------|
| `batchResultFile` | Path to evaluation results JSON |
| `questionSetFile` | Path to question set JSON (must include `id` and `section`) |
| `outputpath` | Output **file path** for the benchmark report JSON |

Paths are resolved relative to the current working directory.

## Output

The script writes a JSON report:

```json
{
  "evaluator_file": "/path/to/evaluation.json",
  "question_set_file": "/path/to/questionSet.json",
  "section_scores_avg": {
    "POSH": 4.5,
    "HMA": 9.0,
    "MVA": 13.33,
    "JJA": 9.5,
    "CGST": 9.0,
    "Consumer Protection Act": 9.0
  },
  "benchmark_score": 9.06
}
```

**Range of `benchmark_score`**: 0–28.

## Directory Structure

```
benchmarking/
├── README.md
├── calculateBenchmarkScore.py
└── benchmarks/
    ├── b1
    └── config/
        └── b1.json
```
