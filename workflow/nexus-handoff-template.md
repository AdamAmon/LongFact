# NEXUS Handoff Template (LongFact)

## Metadata
- From: [Agent / Person]
- To: [Agent / Person]
- Phase: [采样 / 摘要 / 检索 / NLI / 纠错 / 评估]
- Task ID: [短 id]
- Priority: [Critical / High / Medium / Low]
- Timestamp: YYYY-MM-DDTHH:MM:SSZ

## Context
- Project: LongFact
- Current State: 一句话描述当前处理到哪一步
- Relevant Files: (列出路径，如 `data/gov_sample.jsonl`, `index/gov.index`, `results/exp_small.jsonl`)
- Dependencies: (外部或内部依赖)

## Deliverable Request
- What is needed: 具体可交付项（示例：为 sample set 计算 sentence-level support rate, 输出 results/support_rate.jsonl）
- Acceptance criteria: (明确定义验收标准)

## Notes / Constraints
- 使用 CPU-only safe 工具（若无 GPU，传 `--device -1`）
- 不要上传或提交大型模型文件到 repo

## Outputs
- Output files: 列出将被写入的文件路径
- Expected records per file: 简要 schema

## Next Steps
- 如果失败：谁来复核？如何标注？
- 如果成功：谁合并/谁部署？
