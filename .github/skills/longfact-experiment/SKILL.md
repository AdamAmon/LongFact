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

## Canonical Commands (verified)

```powershell
# 1) Tiny run for sanity check
python run_experiment.py --n 1 --device -1 --out results/exp_n1.jsonl

# 2) Small experiment run (task baseline)
python run_experiment.py --n 10 --use_model --device -1 --out results/exp_n10.jsonl

# 3) Sentence-level pipeline debug
python workflow/run_pipeline_simple.py --sample_size 1 --out results/pipeline_n1.jsonl --device -1
```

## Guardrails

- Do not use unsupported flags for `run_experiment.py` (e.g. `--sample-size`, `--input`).
- Always run a small validation (`n=1`) before larger runs.
- If NLI results are all `ERROR`, pause scaling and debug NLI first.

## Output Expectations

- State whether the run used fallback logic or a real model.
- Summarize sample size, support rate, and any notable failure modes.
- Point to the output file paths when available.
