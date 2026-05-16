---
name: longfact-experiment
description: "Use when running LongFact experiments on GovReport, building the summarization pipeline, evaluating ROUGE, checking factual consistency, or exporting correction cases."
user-invocable: true
---

# LongFact Experiment Workflow

## When to Use

- Load GovReport samples for quick validation
- Run the summarize -> retrieval -> NLI -> correction -> evaluation pipeline
- Export ROUGE and factual consistency results
- Collect success and failure correction cases for the report

## Procedure

1. Load a small sample first with `data/load_govreport.py`.
2. Run `run_experiment.py` on a tiny batch before scaling up.
3. Inspect the JSONL output for `prediction`, `corrected`, `support_rate`, and `rouge` fields.
4. If a model run fails, fall back to CPU-safe or placeholder execution paths.
5. Save representative examples for the final report.

## Output Expectations

- State whether the run used fallback logic or a real model.
- Summarize sample size, support rate, and any notable failure modes.
- Point to the output file paths when available.
