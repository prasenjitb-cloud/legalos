# Benchmarking

Compute benchmark scores from evaluation results, grouped by legal section.

## Overview

The benchmark aggregates LLM evaluation results across a question set and computes:

1. **Section-level averages** per legal section (POSH, HMA, MVA, etc.):
   - `avg_score` – mean `layer2.total.score` over questions in that section
   - `avg_percentage` – mean `layer2.total.percentage` (0–100) over questions in that section
   - `support_metrics` – explanation / citation / retrieval stats for that section
2. **Overall benchmark averages** across **all** evaluated questions:
   - `benchmark_avg_score` – mean `layer2.total.score` over all questions
   - `benchmark_avg_percentage` – mean `layer2.total.percentage` (0–100) over all questions

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

  "benchmark_avg_score": 9.5,
  "benchmark_avg_percentage": 33.93,

  "sections": {
    "POSH": {
      "avg_score": 3.21,
      "avg_percentage": 11.46,
      "support_metrics": {
        "explanation_provided_rate": 42.86,
        "citation_rate": 28.57,
        "avg_retrieved_chunks": 3.0
      }
    },
    "HMA": {
      "avg_score": 14.0,
      "avg_percentage": 50.0,
      "support_metrics": {
        "explanation_provided_rate": 60.0,
        "citation_rate": 40.0,
        "avg_retrieved_chunks": 2.8
      }
    }
  }
}
```

All averages and percentage fields are rounded to **2 decimal places**.

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
