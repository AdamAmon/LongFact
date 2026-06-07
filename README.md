以下是更新后的完整 `README.md`，整合了单机多 GPU 并行启动、DCE 开关明确说明、性能基准参考、`timing` 字段说明以及多机合并增强命令。你可直接复制替换原有文件。

---

# LongFact — 长文摘要事实一致性评测与纠错（使用说明）

LongFact 是一套面向 NJUniversity 大作业的本地实验流水线，完整覆盖三个任务：

- **任务 3.1（基线）**：数据采样 → 分块摘要 → 证据检索 → 句子级 NLI 判定 → 局部纠错 → 评估（ROUGE / 支持率）
- **任务 3.2（分析）**：长度分桶分析 + 纠错案例导出
- **任务 3.3（进阶）**：DCE 双通道证据检索（检测方向原创改进）

本文档涵盖：GPU/8-bit/FP16 运行建议、实用命令、DCE 进阶检索使用、实时进度、分析脚本与案例导出、单机多 GPU 并行运行、多机分布式运行。

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

## 进阶检索策略：DCE（双通道证据检索）

针对任务 3.3（检测方向），实现了 **DCE（Dual-Channel Evidence Retrieval with Entailment-Guided Re-Ranking）** 原创进阶检索器，位于 `retrieval/advanced_retriever.py`。

### DCE 核心机制

| 机制 | 说明 |
|------|------|
| 双通道检索融合 | 语义 embedding（Sentence-BERT + FAISS）+ 关键词 BM25，合并去重 |
| 轻量级重排序 | n-gram 重叠 + 实体命中 + 位置连贯性，三项启发式打分（零额外模型开销） |
| 自适应 Top-K | 短句 k=3、中等句 k=5、长句 k=7，减少无效 NLI 调用 |
| 条件证据扩展 | NLI 低置信度时以 top-3 锚点扩展最近邻段落 |

### 如何开启/关闭 DCE？

通过 `--retrieval_strategy` 参数控制：

- **基线（关闭 DCE）**：`--retrieval_strategy baseline`
- **开启 DCE**：`--retrieval_strategy dce`（默认值，但建议显式指定以明确意图）

示例：

```powershell
# 基线（不使用 DCE）
python run_experiment.py --n 100 --retrieval_strategy baseline --out results/baseline_n100.jsonl

# DCE 进阶检索
python run_experiment.py --n 100 --retrieval_strategy dce --out results/dce_n100.jsonl
```

### 对比实验

```powershell
# 运行基线
python run_experiment.py --n 100 --use_model --device 0 --precision fp16 --summary_batch_size 8 --summary_max_new_tokens 64 --out results/baseline_n100.jsonl

# 运行 DCE
python run_experiment.py --n 100 --use_model --device 0 --precision fp16 --summary_batch_size 8 --summary_max_new_tokens 64 --retrieval_strategy dce --out results/dce_n100.jsonl

# 对比分析
python scripts/analyze_results.py --in results/baseline_n100.jsonl --out results/baseline_summary.json
python scripts/analyze_results.py --in results/dce_n100.jsonl --out results/dce_summary.json
```

## 单机多 GPU 并行启动（数据分片）

若你有一台多 GPU 机器（如 8 张 T4/3070），可使用提供的 `scripts/launch_multi_gpu.sh` 脚本自动将总样本数分片到各 GPU 并行运行，最后合并结果。这比手动指定 `--start/--n` 更省事。

### 基本用法

```bash
# 8 张 GPU，500 个样本（默认 batch_size=32, max_new_tokens=256, 检索策略=dce, 精度=fp16）
bash scripts/launch_multi_gpu.sh 8 500

# 完整参数：GPU数量 样本数 batch_size max_tokens 检索策略 精度
bash scripts/launch_multi_gpu.sh 8 500 32 256 dce fp16

bash scripts/launch_multi_gpu.sh 8 500 32 256 baseline fp16
```

### 脚本特点

- 自动计算每张 GPU 处理的样本区间（最后一张处理余数）。
- 每张 GPU 独立日志 `results/multi_gpu_8gpu/gpu{i}.log`，便于排错。
- 所有分片完成后自动调用 `scripts/merge_results.py` 合并，并生成分析摘要。
- 支持 `baseline` / `dce` 检索策略切换（通过第5个参数）。
- 各进程真实隔离（`CUDA_VISIBLE_DEVICES=${gpu}`），避免显存冲突。

### 合并与监控

合并后的文件位于 `results/multi_gpu_${GPU_COUNT}gpu/merged_n${TOTAL_N}_${TIMESTAMP}.jsonl`。  
运行过程中可用以下命令监控：

```bash
# 查看运行中的进程数
watch -n 2 'ps aux | grep run_experiment | wc -l'

# 实时显存占用
watch -n 1 nvidia-smi

# 查看某张 GPU 的实时日志
tail -f results/multi_gpu_8gpu/gpu0.log
```

## 实时进度与可视化

本仓库在这些阶段都支持实时进度显示（基于 `tqdm`）：

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
python scripts/analyze_results.py --in results/experiment.jsonl --out results/summary.json --cases-out results/cases.json --csv-out results/length_buckets.csv --case-count 10 --bucket-by token

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
- `timing`：字典，包含 `summarize`, `retrieve`, `nli`, `correct`, `correct_nli` 等阶段耗时（秒），便于定位瓶颈

## 机房多机运行建议（5 台 RTX 3070，每台处理 100 条样本）

如果你将在机房用 5 台配置为 NVIDIA GeForce RTX 3070（8 GB 专用显存）的机器并行运行，每台处理 100 条样本，推荐如下：

- 首选精度：`--precision fp16`（在 8GB 卡上通常比 8-bit 更稳定）
- 每台建议 `--summary_batch_size 8`（若显存允许可尝试 12~16，但以 OOM 风险为界）
- 建议 `--summary_max_new_tokens 64`（保持质量与速度的折中）
- 每台输出文件建议命名为 `results/hostXX_results.jsonl`（便于后续合并）

示例命令（在 5 台机器上并行运行，覆盖样本区间 1–500，每台处理 100 条）：

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

在所有机器完成后，你可以把所有 `results/host*_results.jsonl` 文件复制到一台汇总机器（或 NFS 共享目录），使用下面的命令合并为一个 JSONL 文件并去重：

```bash
# 合并所有 host 结果，自动按 id 去重
python scripts/merge_results.py --out merged_500.jsonl results/host*.jsonl
```

脚本会：

- 顺序读取所有输入文件（支持通配符）
- 验证每行为合法 JSON 并写入输出文件
- 如果记录包含 `id` 字段，会按 `id` 去重，保留第一次出现的记录
- 返回合并后的总条数

脚本位于：`scripts/merge_results.py`。

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

## 已实现的重要改动

- `retrieval/advanced_retriever.py`（新增）：DCE 双通道证据检索器，含自适应 Top-K、轻量级重排序、条件证据扩展。
- `run_experiment.py`：新增 `--retrieval_strategy baseline|dce` 参数，支持策略切换对比实验。
- `eval/evaluate.py`：句子级 `compute_support_rate()` 增加 `tqdm` 可视进度（`show_progress`）。
- `correction/corrector.py`：`correct_batch()` 增加 `tqdm` 可视进度与 `show_progress` 参数。
- `run_experiment.py`：在原始 NLI / 批量纠错 / 纠错后 NLI 三处启用进度显示，支持分阶段耗时记录（`timing` 字段）。

## 当前状态

- ✅ 任务 3.1（基线）：完整实现并通过测试
- ✅ 任务 3.2（分析）：长度分桶 + 案例导出（已有 n=500 实验结果）
- ✅ 任务 3.3（进阶-检测方向）：DCE 双通道证据检索已实现，16 项测试全部通过
- ✅ 分阶段耗时统计：已在 `timing` 字段中记录每个样本的各阶段耗时
- ✅ 单机多 GPU 并行启动脚本：`scripts/launch_multi_gpu.sh` 可用
- ✅ 多机结果合并脚本：`scripts/merge_results.py` 支持去重

## 项目结构（快速导航）

- **入口与实验**: `run_experiment.py` — 端到端实验运行器（采样→摘要→检索→NLI→纠错→评估），支持 `--retrieval_strategy baseline|dce`。
- **摘要**: `summarize/run_summarize.py`, `summarize/model_summarizer.py` — 分块、局部摘要与融合，支持 HF 模型与回退实现。
- **检索**: `retrieval/retriever.py` — sentence-transformers + FAISS（可选 BM25）封装与缓存；`retrieval/advanced_retriever.py` — DCE 双通道进阶检索器。
- **NLI**: `nli/nli_check.py` — 句对 NLI 判定、批量/分桶处理与多证据聚合。
- **纠错**: `correction/corrector.py` — 基于证据的局部改写与批量纠错接口。
- **数据**: `data/load_govreport.py` — GovReport 数据集加载与采样工具；数据缓存位于 `data/cache/`。
- **评估**: `eval/evaluate.py` — ROUGE 与句子级支持率计算、结果详情生成。
- **工具**: `utils/hf_helpers.py` — 统一的 Hugging Face 模型/Tokenizer 加载与清理逻辑。
- **脚本**: `scripts/` — 结果合并、分析、benchmark、案例图表等实用脚本。
- **测试**: `tests/` — 单元与集成测试（16 项全部通过）。

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

## CLI 参数速查

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--n` | int | 5 | 处理样本数 |
| `--start` | int | 0 | 数据集起始偏移 |
| `--use_model` | flag | False | 启用 HF 模型生成摘要 |
| `--model_name` | str | Qwen/Qwen2.5-1.5B-Instruct | 摘要/纠错模型 |
| `--device` | int | -1 | GPU 设备 ID（-1=CPU） |
| `--precision` | str | auto | 精度：auto/fp32/fp16/8bit |
| `--load_in_8bit` | flag | False | bitsandbytes 8-bit 加载 |
| `--summary_batch_size` | int | 4 | 摘要批次大小 |
| `--summary_max_new_tokens` | int | 256 | 每 chunk 最大生成 token |
| `--top_k` | int | 3 | 检索证据数量 |
| `--retrieval_strategy` | str | baseline | 检索策略：baseline/dce |
| `--step` | int | 0 | 分批大小（>0 时分批运行） |
| `--out` | str | experiment_results.jsonl | 输出文件路径 |

---