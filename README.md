# LongFact — 长文摘要事实一致性评测与纠错（当前实现）

此仓库实现了一个可复现的实验流水线：数据采样 → 分块摘要 → 证据检索 → 句子级 NLI 判定 → 局部纠错 → 评估（ROUGE + 支持率）。README 已更新以反映当前实现、默认配置与离线优先流程。

## 当前项目状态（2026-05）
- 已完成 LongFact 简化流水线（摘要→检索→句子级 NLI→纠错→评估）并可在本地运行小规模样本（如 n=10）。
- 已修复 NLI 聚合调用链关键问题：`check_with_evidence` 已作为 `NLIChecker` 类方法稳定运行，输出不再全量落入 `ERROR`。
- 已提供结果汇总脚本：可从 jsonl 生成初步结果表（ROUGE 与句级 NLI统计）。
- 已修复 GitHub Actions 中 smoke 阶段的不兼容调用，改为轻量、离线友好的快速检查。

## CI 流水线说明
- 工作流文件：`.github/workflows/ci.yml`
- 当前 CI 主要步骤：依赖安装 → 编译检查 → 轻量 smoke 检查 → pytest。
- 为提升跨平台稳定性，Windows 与 Linux 的 smoke 步骤已拆分执行（分别调用 `.ps1` / `.sh`）。
- `bitsandbytes` 在 Windows 下改为可选（通过环境标记跳过安装），避免平台安装失败导致 CI 中断。

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
$env:HF_DATASETS_CACHE='./data/cache'
$env:TRANSFORMERS_CACHE='./.hf-cache'
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

## 补充说明（方便从 README 完全理解并复现实验）

以下说明旨在帮助你仅凭 README 即可复现实验、理解输出与常用运维步骤。

- **复现实验（Task 3.1 — n=10）步骤**：
	1. 确保模型与数据已缓存到本地（参见上文 HF 缓存变量）。
	2. 激活虚拟环境并安装依赖：

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

	3. 运行端到端实验（样例）：

```powershell
python run_experiment.py --n 10 --use_model --model_name Qwen/Qwen2.5-1.5B-Instruct --device -1 --dataset_cache_dir data/cache --out results/experiment_n10.jsonl
```

	4. 生成汇总（ROUGE 与句级 NLI 统计）：

```powershell
python scripts/analyze_results.py --input results/experiment_n10.jsonl --out results/summary_n10.json
```

- **重要输出（建议保留的“最终报告”文件）**：
	- `pipeline_n10_qwen.jsonl` — pipeline 原始逐样本输出（若存在）
	- `experiment_n10.jsonl` — run_experiment 输出（主 jsonl）
	- `summary_n10.json` — analyze_results 生成的汇总（ROUGE/NLI 支持率）
	- `per_sample_results.csv` — 每个样本的表格视图（方便导入 Excel）
	- `summary_results.json` / `summary_results_examples.jsonl` — 进一步的示例与汇总
	- `test_pipeline.jsonl` — 调试用的小样本输出

- **JSONL 每行字段（典型）**：
	- `id`：样本唯一标识
	- `reference`：参考摘要（gold）
	- `prediction` 或 `fused`：融合摘要（模型生成）
	- `sentences`：原文拆分句列表，每句包含 `text`, `nli_label`, `nli_per_evidence`（逐证据分数）
	- `corrected`：纠错后的最终文本（若已运行纠错模块）
	- `rouge`：ROUGE 各项分数字典
	- `support_rate`：句级支持率（基于 NLI 判定的比例）

- **如何产生 ROUGE 与句级 NLI 统计**：
	1. `run_experiment.py` 在每条记录中计算并写入 `rouge` 与 `sentences`。  
	2. `scripts/analyze_results.py` 汇总 jsonl 中的 `rouge`、`support_rate`，并写出 `summary_*.json` 与 `per_sample_results.csv`。

- **测试与 CI（本地复现）**：
	- 运行单元测试：

```powershell
pytest -q
```

	- 在本地模拟 CI smoke（Windows PowerShell）：

```powershell
.\ .github\ci\model_qa_smoke.ps1
```

	- 若在 Linux/macOS ：

```bash
bash .github/ci/model_qa_smoke.sh
```

	- 若 CI 报错请优先检查：`.env.local` 的离线缓存变量、`requirements.txt`（bitsandbytes 可选）、以及本地 Python 可执行路径（PowerShell 下优先使用 `.venv\Scripts\Activate.ps1`）。

- **结果清理与归档**：
	- 仓库包含简单的归档流程：将非最终报告文件移动到 `results/archive/<timestamp>/` 以便保留最小报告集合。示例（PowerShell）：

```powershell
$keep=@("pipeline_n10_qwen.jsonl","experiment_n10.jsonl","summary_n10.json","per_sample_results.csv","summary_results.json","summary_results_examples.jsonl","test_pipeline.jsonl");
$arch="results/archive/<timestamp>/phase2"; New-Item -ItemType Directory -Force -Path $arch | Out-Null; Get-ChildItem -File results | Where-Object { $keep -notcontains $_.Name } | Move-Item -Destination $arch -Force
```

	- 每次归档会在 `results/archive/.../CLEANUP_NOTES.md` 追加移动记录。

- **常见故障排查补充**：
	- NLI 全为 `ERROR`：检查 `nli/nli_check.py` 是否抛出异常；在离线模式下确认模型文件完整且 `HF_HUB_OFFLINE=1`。  
	- `prediction` 为空：在单样本模式下运行 `run_pipeline` 并检查 `fused` 字段；确认 summarizer 模型是否可用或回退到更小模型。  
	- 检索不返回证据：确认 `data/cache` 下是否有已索引的 embedding，或重新运行索引构建：`from retrieval.retriever import Retriever; r=Retriever(...); r.build_index()`。

- **提交与发布建议**：
	- 我建议在确认所有本地测试通过后，把代码更改分支提交并发起 PR，变更包含：`nli` 修复、CI smoke 调整、README 更新、归档脚本。  

如需，我可以直接把 README 中这些补充保存（我已添加），并可以同时生成一个更详尽的 `docs/` 页面或 `USAGE.md`，把命令、示例输出和 JSON 模式展开为可搜索的文档。告诉我是否需要我把 README 的这些更详细内容拆成单独文档并创建初始 files。 
