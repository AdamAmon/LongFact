Explanation: Comprehensive README update with commands, progress and performance guidance.
# LongFact — 长文摘要事实一致性评测与纠错（使用说明）

LongFact 是一套本地流水线：数据采样 → 分块摘要 → 证据检索 → 句子级 NLI 判定 → 局部纠错 → 评估（ROUGE / 支持率）。

本文档更新包含：GPU/8-bit/FP16 的运行建议、实用命令、实时进度显示说明，以及分析脚本的长度分桶与案例导出用法。

## 推荐硬件与软件（本仓库测试环境）

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU（8 GB 显存）
- 驱动：596.49，CUDA：13.2（由 `nvidia-smi` 报告）
- PyTorch：2.5.1+cu121
- `bitsandbytes`：可选（已在本机安装并验证）

注：8 GB 显存属于轻量级卡，对模型批次、精度与最大生成长度需做权衡。

## 快速开始（激活虚拟环境）

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 最佳实践与首选命令（针对 RTX 4060 / 8GB）

优先尝试 `fp16` + 适当增大摘要批次；若显存不足再降为 8-bit：

- 首选（通常最快、效果稳定）：

```powershell
python run_experiment.py --n 1 --start 4 --use_model --model_name Qwen/Qwen2.5-1.5B-Instruct --device 0 --precision fp16 --summary_batch_size 8 --summary_max_new_tokens 64 --out results/test_item5_fp16.jsonl
```

- 如果 fp16 因显存失败，再使用 8-bit（保留同样的批次设置）：

```powershell
python run_experiment.py --n 1 --start 4 --use_model --model_name Qwen/Qwen2.5-1.5B-Instruct --device 0 --load_in_8bit --summary_batch_size 8 --summary_max_new_tokens 64 --out results/test_item5_8bit.jsonl
```

- 吞吐优先（显存允许时可增批次）：

```powershell
python run_experiment.py --n 5 --use_model --device 0 --precision fp16 --summary_batch_size 16 --summary_max_new_tokens 64 --out results/test_n5_fp16_b16.jsonl
```

说明：`--summary_batch_size` 越大减少 GPU 上小批次切换开销，但显存占用上升；`summary_max_new_tokens` 控制每个 chunk 的生成长度，会显著影响耗时。

## 实时进度与可视化

本仓库现在在这些阶段都支持实时进度显示（基于 `tqdm`）：

- 模型加载（transformers 的权重加载进度）
- 分块摘要（chunk -> batch）
- 句子级 NLI（每个句子有 `tqdm`）
- 批量纠错（`correct_batch`）

如果 `tqdm` 未安装，进度将退化为无输出。要看到进度，请确保已安装 `tqdm`：

```powershell
pip install tqdm
```

进度栏位置说明（高级）：摘要、NLI、纠错的 `tqdm` 使用不同 position 参数，避免互相覆盖；在交互式 PowerShell 中可实时看到每个阶段的进度条。

## 为什么“一个样本”有时很慢

- 样本内部被切成多个 chunk（示例日志显示 37 个 chunk），摘要会对每个 chunk 发起生成请求（按批次处理）。
- 每条摘要生成后，会对摘要中的每个句子进行检索并执行 NLI（默认 `top_k=3`），这是大量前向计算。若存在不被支持的句子，还会触发纠错生成。
- 因此“1 个样本”= 多次生成 + 多次 NLI + 可能的批量纠错，多次模型调用累加造成整体耗时。

优化方向：增大摘要批次、降低每 chunk 的生成长度、使用批量 NLI/纠错（已有实现）或在可接受范围内减少 `top_k`。

## 分析脚本与长度分桶

`scripts/analyze_results.py` 支持按 token 或句子做长度分桶（参数 `--bucket-by token|sentence`），并可以导出案例与 CSV：

```powershell
python scripts.analyze_results.py --in results/experiment.jsonl --out results/summary.json --cases-out results/cases.json --csv-out results/length_buckets.csv --case-count 10 --bucket-by token

python scripts/analyze_results.py --in results/full_n500.jsonl --out results/analysis_summary.json --csv-out results/bucket_metrics.csv --cases-out results/correction_cases.json --case-count 10 --bucket-by token

python scripts/generate_results_table.py --in results/full_n500.jsonl --out-csv results/per_sample_results.csv --out-summary results/summary_results.json --examples 10
```

默认分桶策略为 token，也可用 `--bucket-by sentence` 切换。

## 输出字段（快速参考）

- `prediction`：融合后的摘要
- `corrected`：纠错后文本
- `support_rate` / `corrected_support_rate`：纠错前后句子级支持率
- `rouge` / `rouge_corrected`：ROUGE 分数
- `details`：逐句 NLI 结果与证据（每句包含 `evidences`、`best_label`、`best_score`）
- `prediction_length` / `corrected_length`：句子/字符/token 统计
- `summarization_debug`：分块信息，便于排错

## 进阶：分阶段耗时（可选）

如果你需要精确的分阶段耗时（摘要、检索、NLI、纠错），仓库中已有位置适合加入计时：可在 `run_experiment.py` 的 `run_sample` 中对每个样本记录阶段起止时间并写入结果 JSONL 的 `timing` 字段。我可以为你实现此功能并运行小样本验证，或把实现提交到分支。

## 常用调试与排错命令

- 运行单样本并观察 `run_pipeline` 输出（快速排查 fusion/空输出）：

```powershell
python -c "from summarize.run_summarize import run_pipeline; import json; doc='长文内容'; print(json.dumps(run_pipeline(doc, use_model=True, model_name=None, device=0), ensure_ascii=False, indent=2))"
```

- 端到端小样本（fp16 优先）：

```powershell
python run_experiment.py --n 5 --use_model --device 0 --precision fp16 --summary_batch_size 8 --summary_max_new_tokens 64 --out results/test_n5_fp16.jsonl
```

- 若要比较 fp16 与 8-bit：

```powershell
# fp16
python run_experiment.py --n 1 --start 4 --use_model --model_name Qwen/Qwen2.5-1.5B-Instruct --device 0 --precision fp16 --summary_batch_size 8 --summary_max_new_tokens 64 --out results/test_item5_fp16.jsonl

# 8-bit
python run_experiment.py --n 1 --start 4 --use_model --model_name Qwen/Qwen2.5-1.5B-Instruct --device 0 --load_in_8bit --summary_batch_size 8 --summary_max_new_tokens 64 --out results/test_item5_8bit.jsonl
```

## 已实现的重要改动（摘录）

- `eval/evaluate.py`：句子级 `compute_support_rate()` 增加 `tqdm` 可视进度（`show_progress`）。
- `correction/corrector.py`：`correct_batch()` 增加 `tqdm` 可视进度与 `show_progress` 参数。
- `run_experiment.py`：在原始 NLI / 批量纠错 / 纠错后 NLI 三处启用进度显示。

## 结语

如果你希望我把“分阶段耗时统计”加进 `run_experiment.py`（会打印每个样本的摘要时长、检索时长、NLI 时长、纠错时长与总时长），我可以现在实现并运行一次小样本验证，或直接把实现提交到分支。要我继续吗？

## 机房多机运行建议（5 台 RTX 3070，每台处理 100 条样本）

如果你将在机房用 5 台配置为 NVIDIA GeForce RTX 3070（8 GB 专用显存）的机器并行运行，每台处理 100 条样本，推荐如下：

- 首选精度：`--precision fp16`（在 8GB 卡上通常比 8-bit 更稳定）
- 每台建议 `--summary_batch_size 8`（若显存允许可尝试 12~16，但以 OOM 风险为界）
- 建议 `--summary_max_new_tokens 64`（保持质量与速度的折中）
- 每台输出文件建议命名为 `results/hostXX_results.jsonl`（便于后续合并）

示例命令（在 5 台机器上并行运行，覆盖样本区间 1–500，每台处理 100 条）：

在每台机器上把输出文件名按主机编号命名（`host01_results.jsonl` … `host05_results.jsonl`），并为每台指定 `--start` 起始样本索引与 `--n` 数量。示例：

```powershell
# host01: 样本 1 - 100
python run_experiment.py --start 1   --n 100 --use_model --model_name Qwen/Qwen2.5-1.5B-Instruct --device 0 --precision fp16 --summary_batch_size 8  --summary_max_new_tokens 64 --out results/host01_results.jsonl

# host02: 样本 101 - 200
python run_experiment.py --start 101 --n 100 --use_model --model_name Qwen/Qwen2.5-1.5B-Instruct --device 0 --precision fp16 --summary_batch_size 8  --summary_max_new_tokens 64 --out results/host02_results.jsonl

# host03: 样本 201 - 300
python run_experiment.py --start 201 --n 100 --use_model --model_name Qwen/Qwen2.5-1.5B-Instruct --device 0 --precision fp16 --summary_batch_size 8  --summary_max_new_tokens 64 --out results/host03_results.jsonl

# host04: 样本 301 - 400
python run_experiment.py --start 301 --n 100 --use_model --model_name Qwen/Qwen2.5-1.5B-Instruct --device 0 --precision fp16 --summary_batch_size 8  --summary_max_new_tokens 64 --out results/host04_results.jsonl

# host05: 样本 401 - 500
python run_experiment.py --start 401 --n 100 --use_model --model_name Qwen/Qwen2.5-1.5B-Instruct --device 0 --precision fp16 --summary_batch_size 8  --summary_max_new_tokens 64 --out results/host05_results.jsonl
```

说明：若遇显存不足（OOM），先把 `--summary_batch_size` 降到 `4` 或把 `--summary_max_new_tokens` 降为 `32`，或在该主机上改用 `--load_in_8bit` 作为后备方案。

## 合并多机结果

在所有机器完成后，你可以把所有 `results/host*_results.jsonl` 文件复制到一台汇总机器（或 NFS 共享目录），使用下面提供的脚本合并为一个 JSONL 文件并去重（若每条记录包含 `id` 字段则按 `id` 去重）：

用法示例：

```powershell
python scripts/merge_results.py --out results/merged_5hosts.jsonl results/host*_results.jsonl
```

脚本会：

- 顺序读取所有输入文件（支持通配符）
- 验证每行为合法 JSON 并写入输出文件
- 如果记录包含 `id` 字段，会按 `id` 去重，保留第一次出现的记录
- 返回合并后的总条数

脚本位于：`scripts/merge_results.py`。

## 项目结构（快速导航）

- **入口与实验**: `run_experiment.py` — 端到端实验运行器（采样→摘要→检索→NLI→纠错→评估）。
- **摘要**: `summarize/run_summarize.py`, `summarize/model_summarizer.py` — 分块、局部摘要与融合，支持 HF 模型与回退实现。
- **检索**: `retrieval/retriever.py` — sentence-transformers + FAISS（可选 BM25）封装与缓存。
- **NLI**: `nli/nli_check.py` — 句对 NLI 判定、批量/分桶处理与多证据聚合。
- **纠错**: `correction/corrector.py` — 基于证据的局部改写与批量纠错接口。
- **数据**: `data/load_govreport.py` — GovReport 数据集加载与采样工具；数据缓存位于 `data/cache/`。
- **评估**: `eval/evaluate.py` — ROUGE 与句子级支持率计算、结果详情生成。
- **工具**: `utils/hf_helpers.py` — 统一的 Hugging Face 模型/Tokenizer 加载与清理逻辑。
- **脚本**: `scripts/` — 结果合并、分析、benchmark 等实用脚本。
- **测试**: `tests/` — 单元与集成测试。

## 运行与验证（快捷命令）

- 端到端小样本（fp16 优先）:

```powershell
python run_experiment.py --n 5 --use_model --device 0 --precision fp16 --summary_batch_size 8 --summary_max_new_tokens 64 --out results/test_n5_fp16.jsonl
```

- 单样本快速调试（直接打印 pipeline 输出）:

```powershell
python -c "from summarize.run_summarize import run_pipeline; import json; doc='长文内容'; print(json.dumps(run_pipeline(doc, use_model=True, model_name=None, device=0), ensure_ascii=False, indent=2))"
```

- 运行测试（建议在虚拟环境中）:

```powershell
pip install -r requirements.txt
pytest -q
```

## 平台注意事项

- `bitsandbytes` 在 `requirements.txt` 中对 Windows 平台被条件排除（见文件头）。在 Windows 上若要使用 8-bit，请参考 bitsandbytes 官方安装说明并在虚拟环境中手动安装兼容版本。默认在 Windows 环境下优先使用 `fp16` 或 CPU 回退。 

## 我可以帮你做的事（可选）

- 实现并运行“每样本分阶段耗时统计”，把 `timing` 信息写入输出 JSONL；
- 在 `README.md` 中添加一个更详尽的运行示例与调优指南（基于你的硬件）；
- 运行一次端到端小样本并把结果发给你以便核验。 

如需我直接修改 README 并/或实现分阶段计时，请回复你希望优先的项。

