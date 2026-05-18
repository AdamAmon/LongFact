# LongFact — 项目脚手架

此仓库为“长文摘要事实一致性评测与纠错”作业的代码骨架，目标是提供可复现、模块化的流水线：
数据采样 → 分块摘要 → 证据检索 → NLI 判定 → 局部纠错 → 评估。

## 快速开始

- 环境：Python 3.8+
- 安装依赖：

```bash
pip install -r requirements.txt
```

（可选）如果使用 HuggingFace 模型并在 GPU 上运行，建议安装 `transformers`, `accelerate` 并配置 CUDA 驱动。

## 本地配置

这个仓库支持通过环境变量切换本地数据缓存目录和默认模型名。你可以先在 PowerShell 里设置一次：

```powershell
$env:LONGFACT_DATA_DIR = "D:\\WBC\\NJUniversity\\LongFact\\data\\cache"
$env:LONGFACT_OUTPUT_DIR = "D:\\WBC\\NJUniversity\\LongFact\\results"
$env:LONGFACT_GOVREPORT_DATASET = "ccdv/govreport-summarization"
$env:LONGFACT_SUMMARIZER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
$env:LONGFACT_NLI_MODEL = "facebook/bart-large-mnli"
$env:LONGFACT_CORRECTOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
```

这样做以后，下面的命令会自动优先使用这些默认值；如果模型已提前下载到 HuggingFace 缓存中，后续运行会直接复用本地权重。

如果你想一次性把本地环境准备好，可以直接运行：

```powershell
python bootstrap_local.py
```

它会默认完成这几件事：

- 写入项目根目录的 [.env.local](.env.local)，让后续脚本自动读取本地配置。
- 安装 [requirements.txt](requirements.txt) 里的依赖。
- 预热 GovReport 数据集缓存。
- 预下载默认模型：`Qwen/Qwen2.5-1.5B-Instruct`、`facebook/bart-large-mnli`、`all-MiniLM-L6-v2`。

### 本地引导与缓存（推荐）

项目包含 `bootstrap_local.py` 用于一次性准备本地环境：写入 `.env.local`、安装依赖（可选）、预热 GovReport 数据集缓存，并预下载默认模型到项目内缓存。这能避免后续运行时在线拉取大文件，提升试验稳定性。

示例（预热并写入 `.env.local`）：

```powershell
python bootstrap_local.py --write-env --download-dataset --download-models
```

默认位置：
- 模型缓存：`./.hf-cache/models--<owner>--<name>/snapshots/<id>/`（含 `model.safetensors`、`tokenizer.json` 等）
- 数据集缓存：`./data/cache/`

如果你已经装好依赖，只想做缓存预热，也可以分别执行：

```powershell
python bootstrap_local.py --write-env --download-dataset --download-models
```

如果你想跳过某一步，可以加对应的开关，例如 `--download-models` 或 `--install-deps` 单独使用。

## 运行示例

- 采样 GovReport（下载 validation 并保存为 jsonl）：

```bash
python data/load_govreport.py --split validation --sample_size 500 --out govreport_sample.jsonl --cache_dir data/cache
```

- 最小示例（回退实现，不需模型）：

```bash
python summarize/run_summarize.py --input "这是一个用于测试的长文..."
```

- 使用 HF 模型做局部摘要（CPU）：

```bash
python summarize/run_summarize.py --input path/to/document.txt --use_model --model_name Qwen/Qwen2.5-1.5B-Instruct --device 0 --out sample_result.json
```

- 运行小规模端到端实验（采样→生成→检索→NLI→纠错→评估）：

```bash
python run_experiment.py --n 10 --use_model --model_name Qwen/Qwen2.5-1.5B-Instruct --device 0 --dataset_cache_dir data/cache --out experiment_results.jsonl
```

### 强制仅用本地缓存（完全离线）

若需确保运行仅使用本地已下载的模型与数据（不会向 HF Hub 发起请求），可在 PowerShell 中设置离线环境变量并指向项目缓存：

```powershell
$env:HUGGINGFACE_HUB_OFFLINE = "1"
$env:TRANSFORMERS_CACHE = "D:\\WBC\\NJUniversity\\LongFact\\.hf-cache"
$env:HF_DATASETS_CACHE = "D:\\WBC\\NJUniversity\\LongFact\\data\\cache"
.\\.venv\\Scripts\\python.exe run_experiment.py --n 1 --use_model --device 0
```

另一种可靠做法是直接把 `--model_name` 指为本地 snapshot 路径：

```powershell
.\\.venv\\Scripts\\python.exe run_experiment.py --n 1 --use_model --device 0 --model_name "D:\\WBC\\NJUniversity\\LongFact\\.hf-cache\\models--Qwen--Qwen2.5-1.5B-Instruct\\snapshots\\<snapshot_id>"
```

## 设备说明

- `--device -1` 表示使用 CPU；`--device 0` 表示使用 GPU 0（若可用）。

## 主要文件/目录（快捷参考）

- `summarize/`：摘要相关代码，入口为 [summarize/run_summarize.py](summarize/run_summarize.py)
- `summarize/model_summarizer.py`：HF 封装与回退实现
- `retrieval/`：证据定位实现（embedding + FAISS，支持 BM25） — [retrieval/retriever.py](retrieval/retriever.py)
- `nli/`：NLI 判定封装 — [nli/nli_check.py](nli/nli_check.py)
- `correction/`：自动纠错模块 — [correction/corrector.py](correction/corrector.py)
- `eval/`：评估工具（ROUGE、句子级支持率） — [eval/evaluate.py](eval/evaluate.py)
- `data/`：数据下载/预处理脚本 — [data/load_govreport.py](data/load_govreport.py)
- `run_experiment.py`：小样本端到端实验运行器

## 运行测试与 CI（可复现）

推荐在本地使用虚拟环境为开发和测试创建可重复的环境：

1. 创建并激活虚拟环境（Windows PowerShell）：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. 安装依赖并运行测试：

```powershell
pip install --upgrade pip
pip install -r requirements.txt
# 安装本地检查工具
pip install ruff black pytest
pytest -q
```

3. 本仓库的 GitHub Actions CI 会在每次 push/PR 到 `main` 上执行：
- 缓存并安装 `requirements.txt` 中的依赖（基于 requirements 的 hash）
- 运行 `black --check` 与 `ruff check`（风格/静态检查）
- 运行 `pytest`（单元/集成测试）

CI 配置文件： [.github/workflows/ci.yml](.github/workflows/ci.yml)

本地运行 CI 风格检查的简易命令：

```powershell
black --check .
ruff check .
pytest -q
```

## 输出说明

- `summarize/run_summarize.py --out <path>` 会将单条运行结果（包含 `chunks`、`local_summaries`、`fused`）写入指定 JSON 文件。
- `run_experiment.py` 会输出每个样本的一行 JSON（jsonl），其中包含 `prediction`、`corrected`、`support_rate`、`rouge` 等字段，便于后续统计与可视化。

## 开发提示

- 若数据量增大，建议预计算并持久化 FAISS 索引；`retrieval/retriever.py` 提供基础索引构建 API，可扩展为磁盘持久化。
- NLI 模型（如 `facebook/bart-large-mnli`）可在 CPU 上运行以节省显存，但批量计算较慢。

## 常见问题与排错

- Git 在 Windows 上可能会在 `git add` 时显示 CRLF -> LF 的警告（行尾规范化）。仓库包含 `.gitattributes`，可按下列命令统一工作区格式：

```powershell
git add --renormalize .
git commit -m "Normalize line endings"
```

- 如果脚本报 `ModuleNotFoundError`（例如找不到 `torch`），请激活虚拟环境并安装依赖：

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- 若模型下载中断或加载失败，请检查本地缓存目录（`.hf-cache/models--.../snapshots/<id>/`）是否包含 `model.safetensors` 与 `tokenizer.json`，并使用上文的“完全离线”方法强制从本地加载。

- 如果 GPU 显存不足，尝试：使用 CPU（`--device -1`）、换小模型、或采用量化加载（bitsandbytes/4-bit）并调整 `transformers`/`accelerate` 配置。

## 准备推送

- 建议在首次提交前：
  - 运行 `python -m pyflakes .` 或 `flake8` 做快速语法检查
  - 在根目录添加 `.gitignore`（建议忽略 `*.pth`, `__pycache__`, `*.jsonl` 等）

如需我帮助生成 `.gitignore`、整理提交说明或运行一次本地小样本实验，请告诉我要执行的下一步。
