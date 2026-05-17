---
goal: "LongFact simple-task pipeline for GovReport"
version: "1.0"
date_created: "2026-05-17"
last_updated: "2026-05-17"
owner: "GitHub Copilot"
status: "Planned"
tags: ["feature", "longfact", "govreport", "summarization", "nli", "correction"]
---

# Introduction

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

This plan defines the minimum executable implementation for LongFact task 3.1: GovReport long-document summarization, sentence-level factual consistency checking, and local correction using the default model combination:
Qwen/Qwen2.5-1.5B-Instruct for summarization and correction, and facebook/bart-large-mnli for NLI verification.

## 1. Requirements & Constraints

- **REQ-001**: Load GovReport validation or test samples and export them to JSONL for reproducible experiments.
- **REQ-002**: Generate summaries for long documents using Qwen/Qwen2.5-1.5B-Instruct as the primary summarization model.
- **REQ-003**: Split generated summaries into sentences and evaluate each sentence separately.
- **REQ-004**: Retrieve evidence passages from the source document before NLI classification.
- **REQ-005**: Use facebook/bart-large-mnli for sentence-level entailment / support checking.
- **REQ-006**: Apply local correction only to sentences that are not supported by evidence.
- **REQ-007**: Re-run support checking after correction and compare pre- and post-correction outputs.
- **REQ-008**: Keep a fallback execution path available when HuggingFace models, FAISS, or datasets are unavailable.
- **REQ-009**: Support GPU summarization and correction on RTX 4060 8GB, but allow NLI to run on CPU.
- **REQ-010**: Persist experiment outputs to JSON or JSONL files for later analysis and reporting.
- **CON-001**: Do not remove the existing fallback pipeline.
- **CON-002**: Keep CLI interfaces backward-compatible where possible.
- **CON-003**: Avoid large refactors that change unrelated modules.
- **GUD-001**: Prefer small, reversible changes that can be validated on a tiny sample before scaling.
- **PAT-001**: Keep long-running model work behind explicit CLI flags.

## 2. Implementation Steps

### Implementation Phase 1

- GOAL-001: Stabilize data loading, document chunking, and summary generation around the default Qwen model.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | Extend `data/load_govreport.py` to support reproducible sampling metadata such as split, sample size, and document length fields in exported records. | | |
| TASK-002 | Update `summarize/model_summarizer.py` so the primary HF path can load `Qwen/Qwen2.5-1.5B-Instruct` with a safe fallback path when the model cannot be initialized. | | |
| TASK-003 | Update `summarize/run_summarize.py` to support Qwen-style prompting, document chunking with a tunable chunk size, and JSON output for the fused summary and intermediate chunks. | | |
| TASK-004 | Verify that the summary CLI still works in fallback mode with no model access and that the output schema remains stable. | | |

### Implementation Phase 2

- GOAL-002: Implement sentence-level evidence retrieval and NLI-based factual consistency checking.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-005 | Refine `retrieval/retriever.py` to support document-passage indexing plus top-k evidence lookup for each summary sentence. | | |
| TASK-006 | Keep BM25 optional and preserve embedding retrieval as the default evidence retrieval path. | | |
| TASK-007 | Update `eval/evaluate.py` so sentence splitting, evidence lookup, and support-rate computation are deterministic and reusable by the experiment runner. | | |
| TASK-008 | Update `nli/nli_check.py` to expose a CPU-safe sentence-pair entailment API using `facebook/bart-large-mnli` as the default model. | | |
| TASK-009 | Validate the NLI path on a small document sample and confirm that entailment labels are mapped correctly to support decisions. | | |

### Implementation Phase 3

- GOAL-003: Implement local correction and end-to-end experiment logging for the simple task.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-010 | Update `correction/corrector.py` so unsupported sentences are rewritten with evidence-constrained prompts using Qwen/Qwen2.5-1.5B-Instruct. | | |
| TASK-011 | Ensure correction only rewrites unsupported sentences and preserves supported sentences unchanged in the final merged summary. | | |
| TASK-012 | Update `run_experiment.py` to produce a complete record per sample with prediction, support details, corrected summary, and ROUGE before/after correction. | | |
| TASK-013 | Add case-study export fields needed for later manual analysis of successful and failed corrections. | | |
| TASK-014 | Validate the end-to-end pipeline on a tiny GovReport batch and confirm the JSONL output can be aggregated for reporting. | | |

## 3. Alternatives

- **ALT-001**: Use `google/flan-t5-large` as the primary summarizer. Rejected because the required default combination already specifies Qwen/Qwen2.5-1.5B-Instruct.
- **ALT-002**: Keep NLI on GPU together with summarization. Rejected because the RTX 4060 8GB memory budget is tighter and CPU NLI is more stable for batch evaluation.
- **ALT-003**: Rewrite entire summaries during correction. Rejected because the assignment explicitly prefers local sentence-level correction.
- **ALT-004**: Replace embedding retrieval with a pure BM25 pipeline. Rejected because semantic retrieval is already available and BM25 should remain optional rather than exclusive.

## 4. Dependencies

- **DEP-001**: `transformers` for summarization, correction, and NLI model loading.
- **DEP-002**: `datasets` for GovReport loading.
- **DEP-003**: `sentence-transformers` and `faiss-cpu` for passage retrieval.
- **DEP-004**: `rank_bm25` for optional hybrid retrieval.
- **DEP-005**: `rouge-score` for ROUGE computation.
- **DEP-006**: Local GPU memory sufficient for Qwen/Qwen2.5-1.5B-Instruct inference with a lightweight loading strategy.

## 5. Files

- **FILE-001**: `data/load_govreport.py` for dataset sampling and export.
- **FILE-002**: `summarize/model_summarizer.py` for Qwen loading and fallback summarization.
- **FILE-003**: `summarize/run_summarize.py` for chunking, prompting, and summary fusion.
- **FILE-004**: `retrieval/retriever.py` for evidence passage indexing and retrieval.
- **FILE-005**: `nli/nli_check.py` for entailment-based support checking.
- **FILE-006**: `correction/corrector.py` for evidence-constrained local rewriting.
- **FILE-007**: `eval/evaluate.py` for sentence splitting, support rate, and ROUGE helpers.
- **FILE-008**: `run_experiment.py` for end-to-end orchestration and JSONL logging.
- **FILE-009**: `README.md` for usage documentation updates once the pipeline is finalized.

## 6. Testing

- **TEST-001**: Run `python data/load_govreport.py --split validation --sample_size 5 --out ...` and confirm the JSONL schema is valid.
- **TEST-002**: Run `python summarize/run_summarize.py --input ... --use_model --model_name Qwen/Qwen2.5-1.5B-Instruct --device 0` on a small local sample and confirm the fused summary is produced.
- **TEST-003**: Run a tiny end-to-end batch through `run_experiment.py --n 2` and confirm prediction, support details, corrected summary, and ROUGE fields are present.
- **TEST-004**: Verify fallback behavior by running the summarizer without model access and confirming the non-HF path still returns output.
- **TEST-005**: Confirm NLI support-rate calculations remain stable on a fixed toy example where the expected entailment result is known.

## 7. Risks & Assumptions

- **RISK-001**: Qwen/Qwen2.5-1.5B-Instruct may be slow or fail to initialize on the local GPU if memory pressure is too high.
- **RISK-002**: `facebook/bart-large-mnli` on CPU may become the bottleneck for 500+ samples if batch size is not controlled.
- **RISK-003**: Evidence retrieval quality may limit NLI accuracy, especially when chunk boundaries split relevant facts.
- **RISK-004**: Correction prompts may introduce new factual drift if they are too unconstrained.
- **ASSUMPTION-001**: The dataset fields returned by GovReport are stable enough to support document and summary extraction through the existing loader.
- **ASSUMPTION-002**: The project will continue to allow CPU fallback execution when GPU execution is not practical.
- **ASSUMPTION-003**: The initial implementation only needs to satisfy task 3.1, not the analysis and advanced tasks.

## 8. Related Specifications / Further Reading

- [长文摘要事实一致性评测与纠错.md](../长文摘要事实一致性评测与纠错.md)
- [README.md](../README.md)
- https://huggingface.co/datasets/ccdv/govreport-summarization
- https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- https://huggingface.co/facebook/bart-large-mnli
