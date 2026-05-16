---
name: ZK Steward (LongFact 适配)
description: 证据与知识库管理技能，帮助维护检索到的证据片段、链接摘要与长期记忆条目。source: community
risk: low
date_added: '2026-05-16'
---

# ZK Steward（适配说明）

## 目的
为 LongFact 项目提供证据管理与链接策略：把检索到的段落/证据规范化、建立引用索引，并生成可复现的证据摘要条目（便于审计与回溯）。

## 本地化变更
- 将原文中与个人笔记 vault、外部工具的集成说明改为使用本仓库的 FAISS 索引与 JSONL 文件：`data/`、`results/`、`index/`。
- 不自动推送到外部服务；所有输出写到仓库内或 `results/` 目录。

## 工作流（示例）
1. 用 `retrieval/retriever.py` 检索支持段落并保存到 `results/evidence/`。
2. 用 `nli/nli_check.py` 对（证据, 断言）做支持性判断并记录分数。
3. 将验证通过的证据写入 `results/zk_notes.jsonl`，字段包括 `claim_id`, `evidence_id`, `text`, `source`, `nli_score`。

## 快速命令示例

```powershell
# 检索并保存 top-5 证据到 results/evidence/
.venv\Scripts\python.exe retrieval/retriever.py --query "示例断言文本" --top-k 5 --out results/evidence/claim_123.jsonl

# 对检索证据执行 NLI 检测
.venv\Scripts\python.exe nli/nli_check.py --evidence results/evidence/claim_123.jsonl --out results/evidence/claim_123.checked.jsonl
```

## 建议
- 约定 `results/zk_notes.jsonl` 的字段契约，便于后续自动化评估与人工复核。
- 定期运行小样本一致性检查并把失败案例提交到 `issues/`（或 PR）。
