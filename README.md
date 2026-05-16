# LongFact — 项目脚手架

此仓库为“长文摘要事实一致性评测与纠错”作业的代码骨架，目标是提供可复现、模块化的流水线：
数据采样 → 分块摘要 → 证据检索 → NLI 判定 → 局部纠错 → 评估。

**快速开始**

- 环境：Python 3.8+
- 安装依赖：

```bash
pip install -r requirements.txt
```

（可选）如果使用 HuggingFace 模型并在 GPU 上运行，建议安装 `transformers`, `accelerate` 并配置 CUDA 驱动。

**运行示例**

- 采样 GovReport（下载 validation 并保存为 jsonl）：

```bash
python data/load_govreport.py --split validation --sample_size 500 --out govreport_sample.jsonl
```

- 最小示例（回退实现，不需模型）：

```bash
python summarize/run_summarize.py --input "这是一个用于测试的长文..."
```

- 使用 HF 模型做局部摘要（CPU）：

```bash
python summarize/run_summarize.py --input path/to/document.txt --use_model --model_name google/flan-t5-large --device -1 --out sample_result.json
```

- 运行小规模端到端实验（采样→生成→检索→NLI→纠错→评估）：

```bash
python run_experiment.py --n 10 --use_model --model_name google/flan-t5-large --device -1 --out experiment_results.jsonl
```

**设备说明**

- `--device -1` 表示使用 CPU；`--device 0` 表示使用 GPU 0（若可用）。

**主要文件/目录（快捷参考）**

- `summarize/`：摘要相关代码，入口为 [summarize/run_summarize.py](summarize/run_summarize.py)
- `summarize/model_summarizer.py`：HF 封装与回退实现
- `retrieval/`：证据定位实现（embedding + FAISS，支持 BM25） — [retrieval/retriever.py](retrieval/retriever.py)
- `nli/`：NLI 判定封装 — [nli/nli_check.py](nli/nli_check.py)
- `correction/`：自动纠错模块 — [correction/corrector.py](correction/corrector.py)
- `eval/`：评估工具（ROUGE、句子级支持率） — [eval/evaluate.py](eval/evaluate.py)
- `data/`：数据下载/预处理脚本 — [data/load_govreport.py](data/load_govreport.py)
- `run_experiment.py`：小样本端到端实验运行器

**输出说明**

- `summarize/run_summarize.py --out <path>` 会将单条运行结果（包含 `chunks`、`local_summaries`、`fused`）写入指定 JSON 文件。
- `run_experiment.py` 会输出每个样本的一行 JSON（jsonl），其中包含 `prediction`、`corrected`、`support_rate`、`rouge` 等字段，便于后续统计与可视化。

**开发提示**

- 若数据量增大，建议预计算并持久化 FAISS 索引；`retrieval/retriever.py` 提供基础索引构建 API，可扩展为磁盘持久化。
- NLI 模型（如 `facebook/bart-large-mnli`）可在 CPU 上运行以节省显存，但批量计算较慢。

**准备推送**

- 建议在首次提交前：
	- 运行 `python -m pyflakes .` 或 `flake8` 做快速语法检查
	- 在根目录添加 `.gitignore`（建议忽略 `*.pth`, `__pycache__`, `*.jsonl` 等）

如需我帮助生成 `.gitignore`、整理提交说明或运行一次本地小样本实验，请告诉我要执行的下一步。
