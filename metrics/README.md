# Metrics Directory

This directory stores computed evaluation metrics derived from normalized answer files.

## Purpose

The contents here are the final scoring layer of the pipeline. They summarize accuracy and confidence after the answer extraction step has been completed.

## Structure

- One subdirectory per benchmark: `aime/`, `gpqa/`, `mmlu/`, and `mmlu_pro/`.
- Files should mirror the benchmark and model layout used in `answers/`.
- The directory also contains `info.md`, which can be used to document the benchmark-specific metric format or collection notes.

## Producers

Metrics are written by the evaluation helpers in `code/evaluator.py`, especially:

- `compute_and_save_metrics_for_file()`
- `compute_and_save_all_metrics()`
- `run_all()`

These helpers rely on `BenchmarkEvaluator`, `AnswerExtractor`, `AccuracyComputer`, and `ConfidenceComputer`.

## Recommended Contents

A metrics file can include:

- Per-run accuracy.
- Per-subject accuracy.
- Aggregated mean accuracy.
- Confidence statistics from logprobs.
- Any optional diagnostic metadata used during analysis.

## Notes

- Keep metric files deterministic and easy to diff.
- Do not store raw model outputs here.
- Use English for new documentation or comments about metric generation.
