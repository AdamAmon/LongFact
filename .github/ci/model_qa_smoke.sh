#!/usr/bin/env bash
set -euo pipefail
python -m compileall . >/dev/null
python -c "from summarize.run_summarize import run_pipeline; out=run_pipeline('Sentence one. Sentence two.', use_model=False); assert out.get('fused','').strip(); print('smoke_ok')" || { echo "lightweight smoke failed"; exit 1; }
echo "Smoke test complete."