# LongFact Project Instructions

This repository implements the LongFact assignment for long-document summarization factual consistency evaluation and correction.

## Project Goals

- Keep the project modular: `data/`, `summarize/`, `retrieval/`, `nli/`, `correction/`, and `eval/` each own one responsibility.
- Prefer small, reversible changes over broad refactors.
- Preserve the current experiment pipeline and keep fallback paths available for CPU-only development.

## Working Rules

- Use Python 3.8+ style and keep imports explicit.
- Keep runnable scripts executable from the repository root with `.venv\Scripts\python.exe`.
- When adding new functionality, update `README.md` and dependencies if needed.
- Do not overwrite user changes or unrelated files.
- Avoid hardcoding results; experiments should write outputs to files such as JSON or JSONL.

## Experiment Conventions

- Use GovReport samples for small-scale validation before larger runs.
- Keep ROUGE and factual consistency evaluation separate in the output.
- For HuggingFace model usage, provide a CPU fallback or a graceful failure path.

## Validation

- Prefer a focused runnable check after edits.
- If a change affects Python modules, verify they still import and a small sample pipeline still runs.
