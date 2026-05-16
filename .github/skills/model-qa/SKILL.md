---
name: Model QA Specialist
description: 本地化的模型质量审计与对抗测试技能，聚焦于 NLI、纠错与评估流水线的自动化 QA。
source: community
risk: low
date_added: '2026-05-16'
---

# Model QA Specialist (LongFact 适配)

## 目的
提供一套本地化的模型 QA 流程，用于评估 `nli/`, `retrieval/`, `summarize/`, `correction/` 模块的输出质量，执行可重复的对抗测试与错误注入检验。

## 适配说明（已移除远程/付费依赖）
- 移除 `WebFetch`/`Bash` 等外部工具依赖，所有示例调用本仓库脚本。
- 默认使用 CPU-safe 路径；若有 GPU 环境，示例中保留 `--device` 参数供用户替换。

## 本地技术交付
- 可运行的 QA 流程示例：数据抽样 → 生成摘要 → 检查支持率（NLI）→ 记录失败案例。
- 对抗测试脚本范例：在 `run_experiment.py` 增加 `--adversarial` 标志（用户可按需实现）。

## 快速上手命令示例

```powershell
# 从 GovReport 生成小样本
.venv\Scripts\python.exe data/load_govreport.py --out data/gov_sample.jsonl --n 50

# 运行一个小型实验（摘要 + 评估）
.venv\Scripts\python.exe run_experiment.py --input data/gov_sample.jsonl --out results/exp_small.jsonl --sample-size 10

# 仅运行 NLI 支持率检查
.venv\Scripts\python.exe eval/evaluate.py --pred results/exp_small.jsonl --mode support_rate
```

## 建议
- 将 QA 失败案例写入 `results/failures/` 以便后续人工标注与修复。
- 在 CI 中加入轻量化的 `model_qa_smoke` 步骤，运行最小样本的端到端检查。
