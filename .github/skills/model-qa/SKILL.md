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
# 最小回归：先跑 n=1，验证脚本和依赖
python run_experiment.py --n 1 --device -1 --out results/qa_n1.jsonl

# 小批量质量审计（可复现）
python run_experiment.py --n 10 --use_model --device -1 --out results/qa_n10.jsonl

# 句级排障（检索+NLI+纠错）
python workflow/run_pipeline_simple.py --sample_size 1 --out results/qa_pipeline_n1.jsonl --device -1
```

## 建议
- 将 QA 失败案例写入 `results/failures/` 以便后续人工标注与修复。
- 在 CI 中加入轻量化的 `model_qa_smoke` 步骤，运行最小样本的端到端检查。

## 今日经验固化
- 若出现句级 NLI 全部为 `ERROR`：优先检查 `nli/nli_check.py` 的方法绑定与异常栈输出。
- 若 CI 在 Windows 失败：优先检查平台依赖（如 `bitsandbytes`）和 shell 差异（bash/pwsh）。
