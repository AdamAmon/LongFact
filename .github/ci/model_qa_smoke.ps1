# Lightweight smoke test for CI (PowerShell)
Set-StrictMode -Version Latest
Write-Host "Running compileall..."
python -m compileall . | Out-Null
if ($LASTEXITCODE -ne 0) {
  Write-Error "compileall failed"
  exit 1
}

Write-Host "Compile OK. Running lightweight smoke command..."
# Ensure a small sample exists; if not, generate one using data/load_govreport.py
if (-not (Test-Path data/gov_sample.jsonl)) {
  Write-Host "No data/gov_sample.jsonl found; attempting to generate a tiny sample..."
  python data/load_govreport.py --out data/gov_sample.jsonl --n 2
  if ($LASTEXITCODE -ne 0) { Write-Error "failed to generate gov sample"; exit 1 }
}

python run_experiment.py --input data/gov_sample.jsonl --sample-size 1 --out results/ci_smoke.jsonl
if ($LASTEXITCODE -ne 0) { Write-Error "run_experiment failed"; exit 1 }
Write-Host "Smoke test complete."