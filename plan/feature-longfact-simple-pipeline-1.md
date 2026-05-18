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

此计划文档为任务 3.1（Simple Task）的可执行实施方案：在 GovReport 验证集上实现并验证“分块摘要 → 句子级证据检索 → NLI 判定 → 局部纠错”的端到端流水线，产出可复现的代码、n=10 实验结果与案例集，后续可扩展至更大规模实验。

## 1. 要求与约束（摘要）

- 载入 GovReport 样本并可导出为 JSONL（可重复抽样）。
- 使用默认模型组合：`Qwen/Qwen2.5-1.5B-Instruct`（摘要/纠错）与 `facebook/bart-large-mnli`（NLI）；检索默认 `sentence-transformers/all-MiniLM-L6-v2` + FAISS，BM25 为可选混合策略。
- 将摘要按句拆分，对每句检索 top-k 证据并用 NLI 判定是否“支持”。
- 仅对“不被支持”的句子执行证据约束的局部纠错，并在纠错后重新判定支持率。
- 保持回退实现以便在无模型或无网络环境下验证流水线逻辑。

## 2. 高级实施步骤（按顺序）

1) 环境与缓存确认
	- 确认 `./.env.local`、`config.py` 的路径指向 `data/cache` 和 `.hf-cache`。
	- 如需强制离线，启用 `HF_DATASETS_OFFLINE=1` 和 `HF_HUB_OFFLINE=1`。

2) 单样本运行与 `run_pipeline` 调试
	- 在单文档上运行 `run_pipeline` 并打印中间输出（`chunks`, `local_summaries`, `fused`），定位空输出或融合失败的根因。
	- 调试示例：

	  ```powershell
	  python -c "from summarize.run_summarize import run_pipeline; import json; doc='长文内容'; print(json.dumps(run_pipeline(doc, use_model=True, model_name=None, device=-1), ensure_ascii=False, indent=2))"
	  ```

3) 修复摘要融合或模型输出处理
	- 若 `fused` 为空，检查 `summarize/model_summarizer.py` 的生成调用、停止词/截断处理与去噪逻辑；为空输出添加重试或回退（per-chunk 拼接）策略。

4) 验证检索与 NLI
	- 在示例文档上确认 `retrieval/retriever.py` 返回的 top-k 证据与 `nli/nli_check.py.check_batch()` 的标签映射（entailment/neutral/contradiction → 支持/不支持）。

5) 验证纠错步骤
	- 确认 `correction/corrector.py` 在传入原文证据与需改写句子时能输出合理改写，且仅替换被判“不被支持”的句子。

6) 端到端小规模实验（n=10）与聚合分析
	- 执行：

	  ```powershell
	  python run_experiment.py --n 10 --use_model --device -1 --dataset_cache_dir data/cache --out results/experiment_n10.jsonl
	  python scripts/analyze_results.py --in results/experiment_n10.jsonl --out results/summary_n10.json
	  ```

	- 验收：每条记录包含 `prediction`、`corrected`、`support_rate`、`rouge` 字段；`results/summary_n10.json` 给出总体统计。

7) 抽取并保存案例（至少 10 个）
	- 从 `results/experiment_n10.jsonl` 挑选成功/失败纠错案例并保存为 `results/cases_n10.jsonl`，用于报告展示。

8) 撰写短报告与提交材料

## 已完成项（基于当前仓库状态）

- `retrieval/retriever.py` 的重要 bug 修复（已修复 BM25/numpy 导入相关问题） — 状态：Completed
- `nli/nli_check.py` 新增 `check_batch` 批量接口 — 状态：Completed
- 单元测试与集成测试已添加并通过（tests/*，6 passed） — 状态：Completed
- `README.md` 已更新以反映当前实现与离线优先策略 — 状态：Completed
- `.env.local` 已写入并包含 `HF_DATASETS_OFFLINE=1` 与 `HF_HUB_OFFLINE=1` — 状态：Completed
- 已实现并使用 `scripts/analyze_results.py` 对 `results/experiment_n10.jsonl` 进行聚合分析，生成 `results/summary_n10.json` — 状态：Completed
- 已运行 `run_experiment.py --n 10`，并写出 `results/experiment_n10.jsonl`（注意：部分 `prediction`/`corrected` 字段可能为空，需要单样本调试） — 状态：Completed (needs follow-up)

## 标注说明

- 标注为 `Completed` 的项表示仓库中已有相应实现或已在本地运行通过；对这些项不需要重复从零实现，但可能需要后续的 bug 修复或质量改进（在注记中以 `needs follow-up` 标明）。

	- 包含环境、参数、关键结果、10 个案例分析与结论。

## 3. 验收准则

- 单样本调试：`run_pipeline` 必须能在单文档上输出非空的 `fused`（或明确的回退输出）。
- n=10 运行：每条记录 `prediction` 字段非空或有回退文本，`results/summary_n10.json` 能给出 ROUGE 与 support_rate 聚合值。
- 提供 `results/cases_n10.jsonl`（≥10 条案例）。

## 4. 风险与缓解

- 风险：在 CPU 上运行大型模型慢。缓解：先用回退实现验证管道，然后在少量样本上用模型调试。
- 风险：本地缓存缺失或损坏。缓解：检查 `.hf-cache` 快照文件并使用 `bootstrap_local.py --download-models` 预热。

## 5. 下一步（我将执行）

- 如你同意，我现在开始执行步骤 2（单样本 `run_pipeline` 调试），将把调试输出贴回以便我们定位并修复 `fused`/`prediction` 为空的问题。

## 6. 参考链接

- `README.md`, `summarize/run_summarize.py`, `nli/nli_check.py`, `retrieval/retriever.py`, `run_experiment.py`

