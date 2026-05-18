---
name: ZK Steward Companion (LongFact 适配)
description: 证据条目与知识库操作示例与模板，配合 `zk-steward` skill 使用，已本地化为使用本仓库的 `index/`、`data/` 与 `results/` 路径。
source: community
risk: low
date_added: '2026-05-16'
---

# ZK Steward Companion (本地化)

此技能为 `ZK Steward` 的配套模板，提供证据条目格式、日记条目模板、与证据索引/持久化说明。已移去所有外部服务调用，输出以 JSONL 存入 `results/` 目录。

## 数据契约示例（`results/zk_notes.jsonl`）
- `claim_id`: 字符串，断言/条目的唯一 id
- `evidence_id`: 字符串，本条证据的 id（例如 `doc123::sent45`）
- `text`: 证据文本片段
- `source`: 原始文档 id 或文件名
- `nli_label`: NLI 返回标签（`entailment`/`contradiction`/`neutral`）
- `nli_score`: 置信分数（0-1）
- `created_at`: ISO 时间戳

## 日志条目模板（每日/事件）
```markdown
### [YYYYMMDD] Short task title

- **Intent**: 说明本次证据收集/验证目的
- **Changes**: 写入或更新的 `claim_id` / `evidence_id`
- **Open loops**: 列出需要后续人工验证或追加检索的条目
```

## 本地工作流示例
1. 运行 `workflow/run_pipeline_simple.py` 生成句级证据与 NLI 结果。
2. 从 `results/*.jsonl` 中抽取每句的 `evidence`、`nli_label`、`nli_score`。
3. 将满足阈值的证据写入 `results/zk_notes.jsonl`。

## CLI 示例
```powershell
# 生成句级证据与NLI信息
python workflow/run_pipeline_simple.py --sample_size 1 --out results/claim_123.pipeline.jsonl --device -1

# 将通过的证据追加到 zk_notes
python - <<'PY'
import json,sys
inpath='results/claim_123.pipeline.jsonl'
out='results/zk_notes.jsonl'
with open(inpath,'r',encoding='utf8') as f, open(out,'a',encoding='utf8') as o:
    for line in f:
        rec=json.loads(line)
        for i,s in enumerate(rec.get('sentences',[])):
            if s.get('nli_score',0)>=0.6 and str(s.get('nli_label','')).lower()=='entailment':
                note={
                    'claim_id':f"{rec.get('id','unknown')}::{i}",
                    'evidence_id':f"{rec.get('id','unknown')}::{i}::top1",
                    'text':(s.get('evidence') or [''])[0],
                    'source':rec.get('id'),
                    'nli_label':s.get('nli_label'),
                    'nli_score':s.get('nli_score'),
                    'created_at':__import__('datetime').datetime.utcnow().isoformat()
                }
                o.write(json.dumps(note,ensure_ascii=False)+'\n')
PY
```

## 建议
- 约定 `results/zk_notes.jsonl` 的 schema，便于后续自动化分析。 
- 把证据写入后触发 `model-qa` 的复核任务（小脚本或 GitHub Action）。
