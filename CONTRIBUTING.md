# Contributing

感谢你参与 LongFact 项目。下面是本地开发、格式化与 CI 的使用说明。

## 代码风格与 pre-commit
我们使用 `pre-commit` 来保证代码风格与基础静态检查。建议在本地安装并启用：

```bash
python -m pip install --upgrade pip
pip install pre-commit
pre-commit install
# 可选：在提交前检查所有文件
pre-commit run --all-files
```

项目根含有 `.pre-commit-config.yaml`（Black / Ruff / isort / flake8 等）。如果你在 Windows 上工作，推荐使用 Git for Windows 的 MinGW Bash 或者 WSL 来获得一致行为。

## 运行 CI smoke 本地版本
CI 会在每次 push/PR 触发，包括一次轻量化的端到端 smoke 测试（编译 + 最小样本运行）。要在本地重现实验：

```bash
# 创建虚拟环境并激活
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# 安装依赖（如需要）
pip install -r requirements.txt

# 运行轻量 smoke（会在缺少样本时自动生成 tiny gov_sample）
# Linux / Mac
bash .github/ci/model_qa_smoke.sh
# Windows (PowerShell)
.\.github\ci\model_qa_smoke.ps1
```

## 提交与分支策略
- 使用 `main` 作为主分支。请基于 `main` 新建 feature 分支：`feature/<short-desc>`。
- 提交信息建议遵守简易约定：`<scope>: <short summary>`，例如 `run: add smoke test script`。

## PR 审查片段
PR 应包含：
- 变更概要（3-5 行）
- 影响文件清单
- 验证步骤（如何在本地复现）

如果你的变更涉及模型文件或大型数据，不要将这些大文件提交到仓库；使用 `results/`、`models/` 或 `index/` 目录保存并在 `.gitignore` 中忽略。

## CI 失败处理
CI 会在 `main` 的 push/PR 上运行。若 CI 失败：
- 检查 Actions 日志中具体报错（编译错误、依赖缺失或运行超时）
- 若是模型相关的资源导致失败（例如内存不足），请把 CI 的运行开关改为只运行 `compile_repo.py` 并在 PR 中注明原因

谢谢！如需我为你自动生成 PR 草稿或将这些更改提交到分支，请告诉我目标分支名。