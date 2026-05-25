# LongFact — 长文摘要事实一致性评测与纠错

LongFact 是一个面向 GovReport 长文摘要的本地可复现实验流水线，目标是把“长文摘要是否事实一致”拆成可验证的阶段：数据采样、分块摘要、证据检索、句子级 NLI 判定、局部纠错、ROUGE 与支持率评估。

## 当前状态

- 核心流水线已可在本机运行：`run_experiment.py` 会串起采样 → 摘要 → 检索 → NLI → 纠错 → 评估。
- 已验证 8-bit 路径可用：摘要、NLI、纠错都支持 `--load_in_8bit`，当前机器上已成功完成 500 样本运行。
- 已修复批量 NLI 聚合：`NLIChecker.check_with_evidence()` 可用，并且 `eval/evaluate.py` 会优先使用它减少 forward 次数。
- 已增加模块级缓存：`summarize/model_summarizer.py` 与 `correction/corrector.py` 避免在同一轮实验中反复加载大模型。
- 已验证测试通过：`pytest -q tests/test_unit.py` 通过。

## 仓库结构

- `data/load_govreport.py`：加载 GovReport 数据集与本地缓存。
- `summarize/run_summarize.py`：分块摘要入口，输出 chunk、局部摘要与融合摘要。
- `summarize/model_summarizer.py`：摘要模型封装，支持回退实现与 8-bit 加载。
- `retrieval/retriever.py`：嵌入检索与 FAISS 索引，支持 BM25 混合查询。
- `nli/nli_check.py`：句子级 NLI 判定与批量接口。
- `correction/corrector.py`：基于证据的局部纠错。
- `eval/evaluate.py`：ROUGE 与支持率计算。
- `run_experiment.py`：端到端实验主入口。
- `scripts/analyze_results.py`：对 jsonl 实验结果做汇总统计。

## 环境要求

- 推荐 Python 3.8+；当前本机使用 Python 3.11 的虚拟环境已验证通过。
- 安装依赖：

```powershell
pip install -r requirements.txt
```

- 若使用 GPU，请确保已安装匹配版本的 `torch`、`accelerate`，并按需启用 `bitsandbytes`。

## 本地配置

`config.py` 会读取项目根目录的 `.env.local`。建议使用 `bootstrap_local.py --write-env` 生成本地配置，并确保缓存目录存在。

推荐变量示例：

```text
LONGFACT_DATA_DIR=./data/cache
LONGFACT_OUTPUT_DIR=./results
LONGFACT_SUMMARIZER_MODEL=Qwen/Qwen2.5-1.5B-Instruct
LONGFACT_NLI_MODEL=facebook/bart-large-mnli
LONGFACT_CORRECTOR_MODEL=Qwen/Qwen2.5-1.5B-Instruct
LONGFACT_RETRIEVER_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_HOME=./.hf-cache
HF_DATASETS_CACHE=./data/cache/datasets
TRANSFORMERS_CACHE=./.hf-cache/transformers
SENTENCE_TRANSFORMERS_HOME=./.hf-cache/sentence-transformers
HF_DATASETS_OFFLINE=1
HF_HUB_OFFLINE=1
```

离线运行时可在 PowerShell 中临时设置：

```powershell
$env:HF_DATASETS_OFFLINE='1'
$env:HF_HUB_OFFLINE='1'
$env:HF_DATASETS_CACHE='./data/cache'
$env:TRANSFORMERS_CACHE='./.hf-cache'
```

## 真实运行命令

先激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

单文档摘要调试：

```powershell
python summarize/run_summarize.py --input "这是一个用于测试的长文..." --out sample_result.json
```

单样本流水线调试：

```powershell
python -c "from summarize.run_summarize import run_pipeline; import json; doc='长文内容'; print(json.dumps(run_pipeline(doc, use_model=True, model_name=None, device=-1), ensure_ascii=False, indent=2))"
```

端到端小规模实验：

```powershell
python run_experiment.py --n 10 --use_model --model_name Qwen/Qwen2.5-1.5B-Instruct --device -1 --dataset_cache_dir data/cache --out results/experiment_n10.jsonl
```

8-bit 端到端实验（当前已验证可跑）：

```powershell
python run_experiment.py --n 500 --use_model --model_name Qwen/Qwen2.5-1.5B-Instruct --device 0 --load_in_8bit --dataset_cache_dir data/cache --out results/experiment_n500_qwen.jsonl
```

结果汇总：

```powershell
python scripts/analyze_results.py --in results/experiment_n500_qwen.jsonl --out results/summary_n500_qwen.json
```

## 已验证结果

- `results/experiment_n500_qwen.jsonl`：500 条样本全部写出。
- `results/summary_n500_qwen.json`：已生成。
- 统计摘要：
  - 平均 `support_rate` 约为 `0.6721`
  - 平均 `rouge1_fmeasure` 约为 `0.4811`
  - 500 条记录均包含 `prediction`、`corrected`、`support_rate`、`rouge`、`rouge_corrected`、`details`

## 输出字段说明

- `prediction`：摘要融合后的原始预测文本。
- `corrected`：纠错后的文本。
- `support_rate`：句子级支持率。
- `rouge` / `rouge_corrected`：纠错前后 ROUGE 分数。
- `details`：逐句 NLI 结果与证据。
- `summarization_debug`：分块、局部摘要、融合摘要与错误信息的调试快照。

## 已知限制

- 纠错结果的质量仍依赖所选模型和证据提示词；如果你切换到别的本地模型，仍建议先做单样本检查，确认 `corrected` 真的发生了变化。
- 某些摘要句在 NLI 上会被标记为非支持，但具体改写效果取决于模型输出，因此结果文件中仍可能保留与原句相近的文本。

## 测试与排错

运行单元测试：

```powershell
pytest -q
```

Windows smoke 检查：

```powershell
.\.github\ci\model_qa_smoke.ps1
```

Linux smoke 检查：

```bash
bash .github/ci/model_qa_smoke.sh
```

常见排错方向：

- 如果 `prediction` 为空，先单样本运行 `run_pipeline`，检查 `fused` 是否为空。
- 如果 NLI 全部异常，优先检查模型缓存、离线环境变量和 `nli/nli_check.py`。
- 如果检索不返回证据，检查 `data/cache` 下的 GovReport 缓存和 `retrieval/retriever.py` 索引构建逻辑。

## 提示

- `run_experiment.py` 使用的是真实参数：`--n`、`--use_model`、`--model_name`、`--device`、`--load_in_8bit`、`--dataset_cache_dir`、`--out`。
- `scripts/analyze_results.py` 使用 `--in` 和 `--out`，不是 `--input`。
- Windows 下 `bitsandbytes` 为可选依赖，避免平台安装失败。

如果你接下来要继续优化纠错效果，建议先检查 `correction/corrector.py` 的模型输出是否足够简洁，再决定是否需要换更适合的纠错模型或调整 prompt。
