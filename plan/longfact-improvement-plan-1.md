---
goal: LongFact current code improvement plan
version: 1.0
date_created: 2026-05-25
last_updated: 2026-05-25
owner: Copilot
status: 'In progress'
tags: [process, analysis, summarization, factual-consistency, longfact]
---

# Introduction

![Status: In progress](https://img.shields.io/badge/status-In%20progress-yellow)

This plan turns the current LongFact baseline into a submission-ready pipeline for tasks 3.1 and 3.2. The core code changes are now in place, including corrected-summary factual evaluation, length-based analysis, and reusable case export flow; the remaining work is large-scale validation and report writing.

## 1. Requirements & Constraints

- **REQ-001**: Keep the existing end-to-end pipeline working: sampling, summarization, retrieval, NLI, correction, and ROUGE evaluation must continue to run from `run_experiment.py`.
- **REQ-002**: Support task 3.1 with a runnable baseline on GovReport validation or test samples.
- **REQ-003**: Support task 3.2.1 by reporting summary length buckets and the corresponding ROUGE and factual consistency metrics.
- **REQ-004**: Support task 3.2.2 by exporting at least 10 correction cases, including both improved and failed examples, with reasons that can be written into the report.
- **REQ-005**: Preserve Windows compatibility and local/offline execution behavior.
- **REQ-006**: Avoid breaking the current JSONL output schema unless new fields are added in a backward-compatible way.
- **CON-001**: The current smoke run with `n=5` is only a diagnostic check and is not sufficient for report-level conclusions.
- **CON-002**: Large model loading should remain optional and configurable through existing CLI arguments.
- **CON-003**: Existing outputs in `results/` should remain usable for comparison.

## 2. Implementation Steps

### Implementation Phase 1

- GOAL-001: Stabilize the baseline experiment output so the pipeline records everything needed for later analysis.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | Inspect `run_experiment.py` output fields and ensure each record includes original summary, corrected summary, support rate, ROUGE before correction, ROUGE after correction, sentence-level NLI details, and summarization debug data. |  |  |
| TASK-002 | Add or verify a corrected-summary factual evaluation field so the pipeline can compute support rate for the corrected text, not only the original prediction. |  |  |
| TASK-003 | Keep JSONL output backward compatible by appending new fields instead of renaming existing ones. |  |  |

### Implementation Phase 2

- GOAL-002: Implement task 3.2.1 analysis for summary length versus factual error rate.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-004 | Extend `scripts/analyze_results.py` to bucket samples by summary length using a deterministic rule such as sentence count or token count. |  |  |
| TASK-005 | Compute per-bucket averages for ROUGE and factual consistency for each length interval. |  |  |
| TASK-006 | Export the bucketed analysis as a JSON summary and, if useful, a CSV table for report plotting. |  |  |

### Implementation Phase 3

- GOAL-003: Implement task 3.2.2 analysis for correction before/after comparison and case export.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-007 | Re-evaluate corrected summaries with the NLI pipeline and store corrected support rate per sample. |  |  |
| TASK-008 | Mark each sample as improved, worsened, or unchanged using both ROUGE change and factual consistency change. |  |  |
| TASK-009 | Add an export path for at least 10 cases that include successful corrections and failed corrections, with short machine-readable reason labels. |  |  |
| TASK-010 | Update `scripts/analyze_results.py` so it can surface representative cases for the report instead of only aggregate statistics. |  |  |

### Implementation Phase 4

- GOAL-004: Validate the revised pipeline at report scale and document the results.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-011 | Run a medium-scale experiment on a validation or test subset close to the assignment range, ideally 500 samples or more if the machine allows it. |  |  |
| TASK-012 | Verify that the new analysis outputs contain both overall and bucketed metrics, plus at least 10 correction cases. |  |  |
| TASK-013 | Update `README.md` with the final run commands, output locations, and interpretation notes needed for the report. |  |  |

## 3. Alternatives

- **ALT-001**: Keep only overall averages and skip length bucketing. Rejected because task 3.2.1 explicitly requires length-based analysis.
- **ALT-002**: Report only ROUGE improvement after correction. Rejected because the assignment also requires factual consistency analysis, not just surface overlap.
- **ALT-003**: Manually inspect a few examples without exporting structured cases. Rejected because the report needs at least 10 cases and reproducible evidence.

## 4. Dependencies

- **DEP-001**: `datasets` GovReport cache under `data/cache` must remain available for reproducible sampling.
- **DEP-002**: `transformers` model loading for summarization, NLI, and correction must continue to support CPU fallback or local model execution.
- **DEP-003**: `eval/evaluate.py` must remain the single source of truth for ROUGE and support-rate computation helpers.

## 5. Files

- **FILE-001**: `run_experiment.py` for experiment orchestration and per-sample result recording.
- **FILE-002**: `scripts/analyze_results.py` for aggregate analysis and report-ready summaries.
- **FILE-003**: `eval/evaluate.py` for support-rate and ROUGE helpers, especially if corrected-summary evaluation is added.
- **FILE-004**: `README.md` for updated usage instructions and analysis workflow.
- **FILE-005**: `results/` for generated JSONL, JSON, and optional CSV outputs.

## 6. Testing

- **TEST-001**: Run a small smoke experiment with `n=5` to confirm the pipeline still writes valid JSONL records.
- **TEST-002**: Run a focused analysis command to confirm the summary includes overall averages and length buckets.
- **TEST-003**: Verify that corrected-summary factual evaluation produces a non-empty metric for at least one sample.
- **TEST-004**: Verify that the case export contains at least 10 samples and includes both improved and failed correction examples.

## 7. Risks & Assumptions

- **RISK-001**: The correction model may not reliably improve every unsupported sentence, so the report must discuss failure cases instead of promising uniform gains.
- **RISK-002**: Length bucketing based on sentence count may differ from token-based bucketing, so the chosen rule must be stated clearly in the report.
- **RISK-003**: Large-scale runs may be slow on CPU-only machines and may require smaller batches or cached outputs.
- **ASSUMPTION-001**: The current JSONL schema can be extended without breaking downstream analysis scripts that read older fields.
- **ASSUMPTION-002**: The assignment report can cite the generated summary files in `results/` as evidence.

## 8. Related Specifications / Further Reading

- [长文摘要事实一致性评测与纠错.md](../长文摘要事实一致性评测与纠错.md)
- [README.md](../README.md)