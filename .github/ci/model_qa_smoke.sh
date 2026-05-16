#!/usr/bin/env bash
set -euo pipefail
python -m compileall . >/dev/null
if [ ! -f data/gov_sample.jsonl ]; then
  echo "No data/gov_sample.jsonl found; generating tiny sample..."
  python data/load_govreport.py --out data/gov_sample.jsonl --n 2 || { echo "failed to generate gov sample"; exit 1; }
fi

python run_experiment.py --input data/gov_sample.jsonl --sample-size 1 --out results/ci_smoke.jsonl || { echo "run_experiment failed"; exit 1; }
echo "Smoke test complete."