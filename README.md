# LongFact — 长文摘要事实一致性评测与纠错（当前实现）

此仓库实现了一个可复现的实验流水线：数据采样 → 分块摘要 → 证据检索 → 句子级 NLI 判定 → 局部纠错 → 评估（ROUGE + 支持率）。README 已更新以反映当前实现、默认配置与离线优先流程。

## 要点概览
- 入口脚本：`run_experiment.py`（端到端 runner，采样→生成→检索→NLI→纠错→评估）
- 分块与摘要：`summarize/run_summarize.py`、`summarize/model_summarizer.py`
- 检索：`retrieval/retriever.py`（embedding + FAISS，支持 BM25 混合）
- 句子级 NLI：`nli/nli_check.py`（包含 `check_batch` 批量接口）
- 局部纠错：`correction/corrector.py`
- 评估：`eval/evaluate.py`（ROUGE、support_rate 计算）

## 环境与依赖
- 推荐 Python 3.8+（本仓库当前在本机使用 Python 3.11 测试通过）。
- 安装依赖：

```powershell
pip install -r requirements.txt
```

（若使用 GPU，请根据需要安装相应的 `torch` 版本与 `accelerate`。）

## 本地配置与离线优先
`config.py` 会在项目启动时读取根目录的 `.env.local`（若存在）。推荐在项目根创建或使用 `bootstrap_local.py --write-env` 生成 `.env.local`。常用变量示例：

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

本仓库已在 `.env.local` 中记录推荐离线设置，运行时 `config.ensure_local_dirs()` 会创建所需目录。

如果你需要完全禁止网络访问（强制仅用本地缓存），在运行前也可以在 PowerShell 中临时设置：

```powershell
$env:HF_DATASETS_OFFLINE='1'
$env:HF_HUB_OFFLINE='1'
$env:HF_DATASETS_CACHE='D:\WBC\NJUniversity\LongFact\data\cache'
$env:TRANSFORMERS_CACHE='D:\WBC\NJUniversity\LongFact\.hf-cache'
.\.venv\Scripts\python.exe run_experiment.py --n 1 --use_model --device -1
```

## 运行示例（常用场景）

- 生成单文档摘要（回退实现可离线运行）：

```powershell
python summarize/run_summarize.py --input "这是一个用于测试的长文..." --out sample_result.json
```

- 调试 `run_pipeline`（在单样本上打印中间输出，建议用于定位 `fused`/`prediction` 为空的问题）：

```powershell
python -c "from summarize.run_summarize import run_pipeline; import json; doc='长文内容'; print(json.dumps(run_pipeline(doc, use_model=True, model_name=None, device=-1), ensure_ascii=False, indent=2))"
```

- 端到端小规模实验（采样→生成→检索→NLI→纠错→评估）：

```powershell
python run_experiment.py --n 10 --use_model --model_name Qwen/Qwen2.5-1.5B-Instruct --device -1 --dataset_cache_dir data/cache --out results/experiment_n10.jsonl
```

提示：若要加速排查问题，先用 `--n 1 --device -1` 在单条样例上本地调试。

## 输出格式与分析脚本
- `summarize/run_summarize.py --out <path>` 会写入单条 JSON，字段包含 `chunks`, `local_summaries`, `fused` 等。
- `run_experiment.py` 输出为 jsonl（每行一个样本），字段示例： `prediction`（融合摘要）、`corrected`（纠错后文本）、`support_rate`（句子级支持率）、`rouge`（ROUGE 分数字典）。
- 已提供 `scripts/analyze_results.py`，用于汇总 jsonl 并生成 `results/summary_*.json`。

## 测试与质量保证
- 仓库包含单元测试（`tests/`），在本地虚拟环境中可用 `pytest -q` 运行。项目在离线缓存模式下也通过了基础测试（6 passed，具体请运行你的环境中的 `pytest`）。

## 开发提示
- `retrieval/retriever.py` 提供 FAISS 索引构建与 BM25 混合查询接口，建议当数据量增大时将索引持久化到磁盘以加速后续运行。
- `nli/nli_check.py` 提供 `check_batch` 批量 NLI 判定以提高吞吐量。
- 若 `prediction` 或 `corrected` 字段为空，请先在单样本上运行 `run_pipeline` 并检查 `fused` 是否被模型正确生成（见上方调试命令）。

## 常见问题与排错
- 若出现模型加载从网络拉取的情况，请检查 `.env.local` 或运行时环境变量，确保 `HF_HUB_OFFLINE` / `HF_DATASETS_OFFLINE` 已启用并且指向本地缓存路径。
- 模型缓存目录示例： `./.hf-cache/models--facebook--bart-large-mnli/snapshots/<id>/`（需包含 `pytorch_model.bin` 或 `model.safetensors` 与 `tokenizer.json`）。
- 当在 CPU 上运行大型模型速度很慢时，考虑使用更小的回退模型或仅在少量样本上运行完整流水线以验证逻辑。

## 需要帮助？
如果你希望我：
- 在本地对 `run_pipeline` 做一次 n=1 的调试并汇报中间输出，请回复 “调试 run_pipeline”。
- 或者我可以帮你把 `prediction` 为空的问题定位并修复为优先任务。

感谢使用 LongFact；如需我继续运行实验或修改代码文档，请告诉我下一步。
