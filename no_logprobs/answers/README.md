# Answers Directory

This directory stores normalized intermediate answer artifacts derived from benchmark outputs.

## Purpose

Files here sit between raw benchmark results and final metrics. They are designed to preserve the benchmark metadata needed for evaluation while exposing a clean answer field that can be scored consistently.

## Structure

- One subdirectory per benchmark: `aime/`, `gpqa/`, `mmlu/`, and `mmlu_pro/`.
- Files should keep the same run context as the source results whenever possible.
- The exact file naming convention should remain stable so that the evaluator can mirror the directory structure into `metrics/`.

## Consumers

These files are read by `BenchmarkEvaluator` and its helpers in `code/evaluator.py`, especially `AnswerExtractor`, `AccuracyComputer`, and `ConfidenceComputer`.

## Recommended Contents

An answer artifact should usually include:

- Benchmark name.
- Model identifier.
- Subject or category.
- Question text.
- Choices or options when present.
- The extracted answer.
- Optional metadata needed for confidence or auditing.

## Notes

- Keep this folder machine-readable.
- Avoid mixing raw logs with normalized answers.
- Use English for any new documentation or comments that describe this data layout.
