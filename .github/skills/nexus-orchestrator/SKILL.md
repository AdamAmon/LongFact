---
name: Nexus Orchestrator (LongFact 适配)
description: 简化版的多 agent 编排与交接模板，适用于本地实验流程（采样/检索/生成/评估/纠正）。source: community
risk: low
date_added: '2026-05-16'
---

# Nexus Orchestrator（项目适配）

## 目的
为 LongFact 项目定义清晰的阶段与交接模板，方便把 `run_experiment.py`、`summarize/run_summarize.py`、`eval/evaluate.py`、`correction/corrector.py` 组合成可复现的流水线。

## 核心交接模版（NEXUS 风格）
- Metadata（必填）：From / To / Phase / Task ID / Priority / Timestamp
- Context（必填）：Project / Files / Current State / Dependencies
- Deliverable Request（必填）：期望产出与验收标准

## 本地操作示例
```powershell
# 1) 先做最小验证（CPU）
python run_experiment.py --n 1 --device -1 --out results/nexus_n1.jsonl

# 2) 小规模实验（摘要 + 检索 + NLI + 纠错 + 评估）
python run_experiment.py --n 10 --use_model --device -1 --out results/nexus_n10.jsonl

# 3) 句级排障（当需要看每句证据与NLI时）
python workflow/run_pipeline_simple.py --sample_size 1 --out results/nexus_pipeline_n1.jsonl --device -1
```

## 建议
- 在项目根放置 `workflow/` 目录记录常用交接模板（Markdown），并在 PR 模板中要求注明 `NEXUS Handoff Document`。
- 把 `nexus-orchestrator` SKILL.md 作为项目级 runbook 的起点，逐步补足每一步的参数与接受条件。
