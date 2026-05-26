# 在学校机房（Linux）运行 LongFact 的最终版说明

TL;DR
- 本文档面向学校机房的 Linux 环境（含 GPU），按顺序包含：环境检查、虚拟环境创建、依赖安装、烟雾运行（n=1/n=5）、正式运行（n=500）、验收标准、兼容性检查结果与常见故障排查。
- 我已经检查了核心运行文件与配置文件；当前的实验主流程可以直接在 Linux 上运行，默认目录也没有写死为绝对路径。
- 文档保留详细步骤，方便你在机房重新拉取项目后按图执行。

## 1 前提与检查
- 假定你在机房的机器上有登录权限并能连接内部网络或使用镜像。若无 GPU，请参照 CPU-only 步骤（速度会很慢）。
- 首先确认 GPU 与驱动：

```bash
nvidia-smi
```

- 检查 Python 与 CUDA 支持（在激活 venv 后）：

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'cuda_version:', torch.version.cuda)"
```

## 2 创建并激活虚拟环境（venv）

```bash
# 在仓库根目录执行
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
```

## 3 安装 PyTorch（与机房 CUDA 匹配）
- 请先根据 `nvidia-smi` 输出的 CUDA 版本选择对应的安装命令；以下为常见示例：

CUDA 12.1:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

CPU-only（仅作功能验证）:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

注意：如果机房有网络策略或需走镜像，请替换 pip 源。

## 4 安装项目依赖

```bash
pip install -r requirements.txt
```

若在安装中遇到二进制不兼容（例如 bitsandbytes 需要具体系统包），请联系机房管理员或跳过 bitsandbytes（本仓库可在 CPU 或 fp16 下运行，但速度受影响）。

## 5 环境变量与目录准备

设置输出与缓存目录为仓库的相对路径：

```bash
export LONGFACT_DATA_DIR="$PWD/data/cache"
export LONGFACT_OUTPUT_DIR="$PWD/results"
mkdir -p "$LONGFACT_DATA_DIR" "$LONGFACT_OUTPUT_DIR"
# 若希望离线 HF 数据：
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

把以上三行追加到 `~/.bashrc`（可选），或在每次运行前设置。

## 6 烟雾运行（必做 —— 验证流程）

1) 单样本验证（功能完整性）

```bash
source .venv/bin/activate
python run_experiment.py --n 1 --use_model --device 0 --precision fp16 --summary_batch_size 4 --summary_max_new_tokens 32 --out results/test_n1.jsonl
```

检查输出并打印关键字段：

```bash
ls -l results/test_n1.jsonl
python - <<'PY'
import json
r=json.loads(open('results/test_n1.jsonl','r',encoding='utf-8').read().splitlines()[0])
print('keys=', sorted(r.keys()))
print('support_rate=', r.get('support_rate'), 'corrected_support_rate=', r.get('corrected_support_rate'))
PY
```

2) 小批量（稳定性）

```bash
python run_experiment.py --n 5 --use_model --device 0 --precision fp16 --summary_batch_size 16 --summary_max_new_tokens 32 --out results/test_n5.jsonl
```

在运行时使用 `watch -n2 nvidia-smi` 观察显存与 GPU 占用。若报 OOM，请按故障排查调整。

## 7 正式运行（n=500）

在 `n=1` / `n=5` 均通过并确认显存可用后，运行推荐配置：

```bash
python run_experiment.py --n 500 --use_model --device 0 --precision fp16 --summary_batch_size 32 --summary_max_new_tokens 32 --out results/experiment_n500.jsonl
```

若在运行过程中出现显存不足或错误，按以下顺序尝试：
- 将 `--summary_batch_size` 从 32 降到 16 或 8
- 将 `--precision fp16` 改为 `fp32`（更稳但更慢）
- 若可用，先在小样本上试 `--load_in_8bit`，再决定是否在 500 样本上启用

## 8 验收标准（简单可行）

- 文件行数：
```bash
wc -l results/experiment_n500.jsonl
# 期望输出：500
```
- 运行后使用分析脚本生成 summary 与案例：

```bash
python scripts/analyze_results.py --in results/experiment_n500.jsonl --out results/summary_n500.json --cases-out results/cases_n500.json --csv-out results/length_buckets_n500.csv
```

- 验收阈值（可自定义）：`support_rate` 与 baseline 差异 ≤ 0.03（若没有 baseline，可用你本地小样本作为比较）。

## 9 常见故障与快速修复

- 找不到 GPU / torch.cuda.is_available() 为 False：确认 PyTorch 与 CUDA 版本匹配并已正确安装。
- OOM（显存不足）：降低 `--summary_batch_size`、降低 `--summary_max_new_tokens`、或切换到 `fp32`。
- pipeline 在循环中重复加载模型（导致慢或内存泄漏）：确保 `run_experiment.py` 在主循环外只创建一次 summarizer/retriever/corrector；若不确定，我可以帮你检查该文件。
- bitsandbytes 相关错误：若环境缺少系统依赖，暂时不要启用 `--load_in_8bit`，改用 `fp16` 或 `fp32`。

## 10 兼容性检查结果

以下是我实际检查过、且与你在 Linux 机房运行最相关的文件结论：

- [run_experiment.py](../run_experiment.py)
  - 没有发现硬编码绝对路径。
  - `Retriever`、`NLIChecker`、`Corrector` 是在 `run_sample()` 里统一实例化的，不是每个样本重复加载。
  - `--out`、`--dataset_cache_dir`、`--summary_batch_size`、`--summary_max_new_tokens` 都通过参数控制，适合 Linux 机房直接运行。

- [config.py](../config.py)
  - 默认值使用仓库相对路径，例如 `data/cache`、`results`、`data/emb_cache`。
  - `ensure_local_dirs()` 只会创建本地目录，不会写死系统盘路径。

- [.env.example](../.env.example)
  - 默认缓存与输出路径均为相对路径。
  - 没有发现 Windows 盘符路径。

- [summarize/run_summarize.py](../summarize/run_summarize.py)
  - 仅依赖 `open()`、`pathlib` 风格的输入处理，没有 Windows 专用路径逻辑。
  - `run_pipeline()` 与 `get_summarizer()` 组合适合 Linux 直接调用。

- [summarize/model_summarizer.py](../summarize/model_summarizer.py)
  - 使用模块级缓存避免重复加载模型。
  - 批量 `pipeline` 调用是标准 Python/HuggingFace 用法，没有系统路径依赖。

- [retrieval/retriever.py](../retrieval/retriever.py)
  - 嵌入缓存写入 `data/emb_cache`，使用 `pathlib` 组装路径。
  - 没有发现绝对路径或 Windows-only 分支。

- [nli/nli_check.py](../nli/nli_check.py)
  - 支持批量推理 `check_batch()`、`check_with_evidence()`。
  - 没有平台路径依赖。

- [correction/corrector.py](../correction/corrector.py)
  - 已将默认纠错长度设为较短的 `max_length=32`，适合作为你机房首轮验证参数。
  - 没有 Windows 路径硬编码。

- [eval/evaluate.py](../eval/evaluate.py)
  - 只做 ROUGE 与支持率统计，不涉及平台路径。

与 Linux 机房运行直接相关的残余平台项：

- [CONTRIBUTING.md](../CONTRIBUTING.md) 与 [README.md](../README.md) 里仍保留 Windows / PowerShell 示例。这些是文档示例，不会阻止你在 Linux 上运行主流程，但会让初次阅读的人看到 Windows 片段。
- `.github/workflows/ci.yml` 同时跑 Linux 和 Windows；Linux 分支调用的是 bash smoke 脚本，Windows 分支调用 PowerShell smoke 脚本。这只影响 CI，不影响你在机房手动运行实验。
- `.github/ci/model_qa_smoke.ps1` 与 `.github/hooks/longfact-guardrails.json` 是开发/CI 相关脚本，它们包含 PowerShell 命令。若机房没有 `pwsh`，这些脚本本身不会运行，但它们不影响你执行 `run_experiment.py`。

检查结论：

- 代码主流程没有发现必须修改的绝对路径问题。
- Linux 机房可以按本文档直接复现实验。
- 当前真正需要注意的是环境匹配、依赖安装和显存配置，而不是路径兼容。

## 11 你在机房最推荐的执行顺序

1. `nvidia-smi` 确认 GPU 与驱动。
2. `python3 -m venv .venv` 创建虚拟环境并 `source .venv/bin/activate`。
3. 先安装与 CUDA 匹配的 PyTorch，再 `pip install -r requirements.txt`。
4. 跑 `n=1`，确认能产出 `results/test_n1.jsonl`。
5. 跑 `n=5`，确认批处理稳定。
6. 再跑 `n=500`，使用 `--precision fp16 --summary_batch_size 32 --summary_max_new_tokens 32` 作为首选配置。
7. 用 `scripts/analyze_results.py` 生成 summary、案例和长度分桶结果。

## 12 机房执行时最常用的三条命令

```bash
source .venv/bin/activate
python run_experiment.py --n 1 --use_model --device 0 --precision fp16 --summary_batch_size 4 --summary_max_new_tokens 32 --out results/test_n1.jsonl
python run_experiment.py --n 500 --use_model --device 0 --precision fp16 --summary_batch_size 32 --summary_max_new_tokens 32 --out results/experiment_n500.jsonl
```

## 13 可选（后续我可以继续帮你）

- 如果你愿意，我可以继续把 [CONTRIBUTING.md](../CONTRIBUTING.md) 和 [README.md](../README.md) 里的 Windows 示例整理成 Linux 友好的补充说明，但这不属于你当前“只写机房文档”的要求。
- 如果你希望，我也可以再给你一份更短的“机房一页速查版”。

## 14 监控建议

- 在长期运行时，打开一个监控窗口：

```bash
watch -n 2 nvidia-smi
# 或者监控系统负载
watch -n 5 'free -h; uptime; ps aux --sort=-%mem | head -n 10'
```

---

如果你在机房按本文档执行时遇到报错，把完整报错贴给我，我会按你机器上的实际环境继续定位。祝运行顺利！
