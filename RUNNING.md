运行与调试指南

目标
- 在本地可重复地运行实验、观察进度条与调试 transformers / bitsandbytes 警告。
- 给出在 Windows (.venv) 环境下的推荐命令与环境变量，帮助收集运行日志并调整可见性。

前提
- 已安装依赖：建议使用 requirements.txt 中列的依赖创建虚拟环境并安装。
- GPU 环境下请确保 CUDA 驱动与 PyTorch 能访问 GPU。

快速运行（单样本）
- 在仓库根目录激活虚拟环境后运行：

```powershell
.\.venv\Scripts\Activate.ps1
python run_experiment.py --n 1 --start 4 --use_model --model_name Qwen/Qwen2.5-1.5B-Instruct --device 0 --load_in_8bit --out results/test_item5.jsonl
```

说明：
- `--n`：样本数量；`--start`：数据集偏移，用于分片运行；`--use_model`：启用模型生成而非回退；`--load_in_8bit`：尝试 bitsandbytes 8-bit 加载。

收集完整终端日志
- 将标准输出/错误写入文件：

```powershell
python run_experiment.py --n 1 --start 4 --use_model --model_name Qwen/Qwen2.5-1.5B-Instruct --device 0 --load_in_8bit --out results/test_item5.jsonl > run.log 2>&1
```

调试与可见性建议
1) 看到更清晰的进度条
- 我们已把外层样本进度与 chunk-level 进度放在不同的 tqdm `position`（position=1/2），以避免模型权重加载（transformers）显示的进度条覆盖它们。
- 如果仍然看不到进度条，尝试设置 `TRANSFORMERS_VERBOSITY` 环境变量来降低 transformers 内部 tqdm 的噪音（或在运行前把其关闭）：

```powershell
$env:TRANSFORMERS_VERBOSITY = "error"
python run_experiment.py ...
```

2) 处理 transformers 关于 generation 参数 / tokenizer 警告
- 我们在代码中已尽量：
  - 预加载 model/tokenizer 并清理 `model.generation_config.max_length`、`temperature`、`top_p`、`top_k` 等字段；
  - 尝试以 `clean_up_tokenization_spaces=False` 加载 tokenizer，并在可写时显式设置该属性。
- 如果仍有警告，可临时设置：

```powershell
$env:TRANSFORMERS_VERBOSITY = "error"
```

这会降低 transformers 的 info/warning 输出，但不是修复根本原因（代码已尽量修复）。

3) bitsandbytes 的 dtype 警告
- bitsandbytes 可能会警告："MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization"。这是正常提示，通常对推理无害。若你希望消除它，需要在模型/输入端上强制 dtype 转换（例如把输入转 float16），这可能影响数值行为，需谨慎。

其他有用命令
- 运行带有更大 batch 的小实验（可用于显存测试）：

```powershell
python run_experiment.py --n 5 --use_model --device 0 --load_in_8bit --summary_batch_size 4 --summary_max_new_tokens 96 --out results/test_n5.jsonl
```

- 以分片方式运行（比如每次处理 50 条并把结果追加到同一文件）：

```powershell
python run_experiment.py --n 500 --start 0 --step 50 --use_model --device 0 --load_in_8bit --out results/experiment_n500.jsonl
```

建议的排错步骤（若输出异常或模型为空）
1. 检查 `run.log` 中的第一个异常栈（通常会在模型加载处）
2. 确认 transformers、torch、bitsandbytes 版本兼容（见 requirements.txt）
3. 在报错上下文查找是否是网络/离线缓存问题（repo 支持离线 dataset 缓存）

最后的工程提示
- 我在代码中已把 HF 加载逻辑统一到 `utils/hf_helpers.py`，并把主加载点改为预加载（summarizer / corrector / nli / quick_quant_8bit）。如果你希望进一步把 helper 应用到其他自定义脚本或其他项目，请复用该 helper。

需要我为你执行一次完整的分片运行并收集日志吗？把你想运行的命令发给我，或允许我在本机执行，我会把日志和建议的后续修补点贴回给你。
