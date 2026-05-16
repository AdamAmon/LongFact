---
description: "Use when editing Python code in LongFact. Covers CLI scripts, modular structure, safe imports, and lightweight validation."
applyTo: "**/*.py"
---
# Python Guidelines for LongFact

- Keep each module focused on a single task.
- Use `argparse` for runnable scripts and keep CLI flags backward compatible.
- Guard optional HuggingFace, FAISS, or dataset imports so the code can still load in minimal environments.
- Prefer structured return values such as dictionaries for pipeline and evaluation helpers.
- Keep long-running or model-heavy work behind CLI flags so the fallback path remains usable.
- Avoid side effects on import.
- Write outputs to files when the task produces experiment results, intermediate data, or case studies.
