# LongFact — 长文摘要事实一致性评测与纠错

LongFact 是一个面向 GovReport 长文摘要的本地可复现实验流水线，目标是把“长文摘要是否事实一致”拆成可验证的阶段：数据采样、分块摘要、证据检索、句子级 NLI 判定、局部纠错、ROUGE 与支持率评估。

## 当前机器配置

本仓库当前在以下机器上完成了 GPU 8-bit 验证：

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU
- 显存：8 GB
- 驱动：596.49
- CUDA：13.2（`nvidia-smi` 显示）
- PyTorch：2.5.1+cu121
- `bitsandbytes`：已安装，可用

基于这台机器，当前最合适的运行方式是 `Qwen/Qwen2.5-1.5B-Instruct` + `--device 0 --load_in_8bit`，不需要回退到 CPU。

## 当前状态

- 核心流水线已可在本机运行：`run_experiment.py` 会串起采样 → 摘要 → 检索 → NLI → 纠错 → 评估。
- 已验证 8-bit 路径可用：摘要、NLI、纠错都支持 `--load_in_8bit`，当前机器上已成功完成 500 样本运行。
- 已验证当前机器适合直接跑 GPU 8-bit：RTX 4060 Laptop GPU，8GB 显存，`bitsandbytes` 可用。
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

- 进度条（可选）: 若希望在 `run_experiment.py` 看到进度条，建议安装 `tqdm`（已加入 `requirements.txt`）。

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

当前机器建议优先使用的 8-bit 小样本验证命令：

```powershell
python run_experiment.py --n 5 --use_model --model_name Qwen/Qwen2.5-1.5B-Instruct --device 0 --load_in_8bit --dataset_cache_dir data/cache --out results/test_n5_8bit.jsonl
```

结果汇总：

```powershell
python scripts/analyze_results.py --in results/experiment_n500_qwen.jsonl --out results/summary_n500_qwen.json
```

长度分桶与案例导出：

```powershell
python scripts/analyze_results.py --in results/experiment_n500_qwen.jsonl --out results/summary_n500_qwen.json --cases-out results/cases_n500_qwen.json --csv-out results/length_buckets_n500_qwen.csv --case-count 10 --bucket-by token
```

## 已验证结果

- `results/experiment_n500_qwen.jsonl`：500 条样本全部写出。
- `results/summary_n500_qwen.json`：已生成。
- 统计摘要：
  - 平均 `support_rate` 约为 `0.6721`
  - 平均 `rouge1_fmeasure` 约为 `0.4811`
  - 500 条记录均包含 `prediction`、`corrected`、`support_rate`、`rouge`、`rouge_corrected`、`details`

## 新增分析能力

- 每条样本现在还会记录 `corrected_support_rate`、`prediction_length`、`corrected_length`、`support_rate_delta` 和 `rouge1_fmeasure_delta`。
- `scripts/analyze_results.py` 现在会输出长度分桶统计，默认按摘要 token 数分为 `1-3`、`4-6`、`7-10`、`11-15`、`16+` 五档，也可用 `--bucket-by sentence` 切回句子数分桶。
- 分析脚本支持额外导出案例文件和 CSV 表格，且 `--case-count 10` 可直接导出足够写报告的成功/失败案例。

## 输出字段说明

- `prediction`：摘要融合后的原始预测文本。
- `corrected`：纠错后的文本。
- `support_rate`：句子级支持率。
- `corrected_support_rate`：纠错后摘要的支持率。
- `rouge` / `rouge_corrected`：纠错前后 ROUGE 分数。
- `details`：逐句 NLI 结果与证据。
- `prediction_length` / `corrected_length`：摘要长度统计，包含句子数与 token 数，便于做 3.2.1 的长度分桶分析。
- `support_rate_delta` / `rouge1_fmeasure_delta`：纠错前后变化量，便于抽取成功和失败案例。
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
- 如果要写 3.2.1 报告，优先看分析脚本生成的 `length_buckets` 和对应 CSV。
- 如果要写 3.2.2 报告，优先看 `cases-out` 导出的成功与失败案例集。

## 提示

- `run_experiment.py` 使用的是真实参数：`--n`、`--use_model`、`--model_name`、`--device`、`--load_in_8bit`、`--dataset_cache_dir`、`--out`。
- `scripts/analyze_results.py` 使用 `--in` 和 `--out`，不是 `--input`。
- Windows 下 `bitsandbytes` 为可选依赖，避免平台安装失败。

## 提速方向

- 保持 `--load_in_8bit`，这是当前机器上的主力加速方式。
- 优先做无损工程优化：
  - 复用 `Retriever` 实例，避免每个样本重复加载 embedding 模型。
  - 使用检索模型缓存（同模型同设备只加载一次）。
  - 摘要后处理仅做清洗，不做强制句数截断。
- 不做会影响实验口径的“提速”改动：
  - 不降模型规格；
  - 不降低 `summary_max_new_tokens` 作为默认行为；
  - 不跳过纠错或 NLI 阶段；
  - 不修改评测阈值与 top-k。

  ## 最佳速率（Best-throughput）配置与验证

  以下配置是在不明显降低质量（support_rate / ROUGE）前提下，经本地对比验证效果最稳健的“首选”运行方式。适合在本机（RTX 4060 / 8GB）或类似单卡环境中直接复现。

  - 首选参数
    - 精度：`--precision fp16`
    - 不启用 8-bit：不要使用 `--load_in_8bit`（本机上 bitsandbytes 在某些路径上并未带来稳定加速，先在服务器/更强硬件上再试）
    - 摘要 batch：`--summary_batch_size 32`（显存不足改为 16）
    - 生成长度限制：`--summary_max_new_tokens 32`
    - 可选试验：`--torch_compile`（短期对比测试后决定是否常驻）
    - 纠错 / NLI 批量：将 `correct_batch` / `check_batch` 的批次设置为 8~32，减少频繁的小批次切换开销

  - 推荐复现实验命令（PowerShell）

  ```powershell
  $env:PYTHONPATH='.'; $env:HF_DATASETS_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1';
  .\.venv\Scripts\python.exe run_experiment.py --n 5 --use_model --device 0 --precision fp16 --summary_batch_size 32 --summary_max_new_tokens 32 --out results/test_n5_accel.jsonl
  ```

  - 验证步骤与接受准则
    1. 小规模验证：先跑 `n=1` / `n=5`，比较 `support_rate`、`corrected_support_rate`、`rouge1_fmeasure` 与 baseline，质量阈值为相对 baseline 差异 ≤ ±0.03。
    2. 中等放大：`n=50`、`n=100`，监控显存、GPU 利用率、CPU 与 I/O；若通过再运行 `n=500`。
    3. 监控指标：`nvidia-smi`（GPU 使用率/显存）、系统负载、结果文件（results/*.jsonl）的 `support_rate` 与 `rouge1_fmeasure`。

  - 回滚与注意事项
    - 若质量超出可接受阈值：回滚 `--summary_max_new_tokens` 为上一个数值或将 `--summary_batch_size` 降低并重跑；或恢复 `correction/corrector.py` 中 `max_length` 的上一个值。
    - bitsandbytes / 8-bit：仅在目标服务器验证通过后再在实验脚本中启用。
    - 避免在主循环中重复构建模型与 pipeline：确保模型/`pipeline` 在进程内只加载一次并被复用。

  更多实现细节与关键文件：`run_experiment.py`、`summarize/model_summarizer.py`、`nli/nli_check.py`、`correction/corrector.py`、`retrieval/retriever.py`。

### 无损 A/B 对照命令

优化版小样本（n=5）：

```powershell
python run_experiment.py --n 5 --use_model --model_name Qwen/Qwen2.5-1.5B-Instruct --device 0 --load_in_8bit --summary_max_new_tokens 96 --summary_batch_size 4 --dataset_cache_dir data/cache --out results/test_n5_8bit_opt.jsonl
```

汇总优化版：

```powershell
python scripts/analyze_results.py --in results/test_n5_8bit_opt.jsonl --out results/test_n5_8bit_opt_summary.json --cases-out results/test_n5_8bit_opt_cases.json --csv-out results/test_n5_8bit_opt_buckets.csv
```

与历史基线（如 `results/test_n5_8bit.jsonl`）做并行对照时，建议同时检查：

- `prediction` / `corrected` 的可读性与长度分桶是否异常；
- `support_rate` 与 `corrected_support_rate` 是否下降；
- `rouge1_fmeasure` 与 `rouge1_fmeasure_delta` 是否整体退化。

如果你接下来要继续优化纠错效果，建议先检查 `correction/corrector.py` 的模型输出是否足够简洁，再决定是否需要换更适合的纠错模型或调整 prompt。
