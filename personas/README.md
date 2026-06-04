# Personas Directory

This directory stores the benchmark personas that are injected into prompts during evaluation.

## Purpose

Each persona is a Markdown prompt that defines the expert voice, domain framing, and reasoning style expected for a benchmark subject or category.

## Structure

- One subdirectory per benchmark: `aime_2026/`, `gpqa/`, `mmlu/`, and `mmlu_pro/`.
- One Markdown file per subject or category.
- File names should match the subject or category names used by the benchmark runner.

## Consumers

Personas are loaded by `BenchmarkRunner._set_persona()` and `BenchmarkRunner._load_persona()` in `code/benchmark_runner.py`.

## Recommended Contents

A persona file should usually include:

- The expert role or identity.
- Domain-specific background and scope.
- Reasoning style or problem-solving expectations.
- Any constraints that help the model answer in the desired format.

## Notes

- Keep persona wording stable when possible so benchmark runs remain comparable.
- Match the benchmark naming convention exactly to avoid file lookup errors.
- Use English for new documentation and comments in this directory.
