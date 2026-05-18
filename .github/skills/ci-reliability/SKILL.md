---
name: ci-reliability
description: "Use when fixing GitHub Actions failures in LongFact, especially cross-platform shell issues, invalid CLI flags, and offline-safe smoke checks."
user-invocable: true
source: community
risk: low
date_added: '2026-05-18'
---

# CI Reliability (LongFact)

## When to Use

- GitHub Actions failed on Windows/Linux matrix.
- Smoke tests are flaky, too heavy, or require unavailable network/model assets.
- Commands in workflow/scripts mismatch actual Python CLI flags.

## Workflow

1. Reproduce locally with the same command shape as workflow.
2. Validate script flags against actual argparse definitions.
3. Split OS-specific commands (bash vs pwsh) when needed.
4. Keep smoke test lightweight and offline-safe.
5. Re-run tests and summarize exact file-level fixes.

## Proven Guardrails

- `run_experiment.py` uses `--n`, not `--sample-size` or `--input`.
- Prefer fast smoke command that does not require downloading large models.
- For Windows runners, avoid bash-embedded PowerShell branching.
- Use platform markers for dependencies that fail on Windows.

## Canonical Commands

```powershell
# Local quick regression
pytest -q

# Windows smoke script
./.github/ci/model_qa_smoke.ps1
```

## Expected Output

- Root cause summary (1-3 items)
- Minimal patch list with paths
- Validation results (tests/smoke)
- Residual risks and next checks
