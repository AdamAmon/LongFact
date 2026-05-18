# Lightweight smoke test for CI (PowerShell)
Set-StrictMode -Version Latest
Write-Host "Running compileall..."
python -m compileall . | Out-Null
if ($LASTEXITCODE -ne 0) {
  Write-Error "compileall failed"
  exit 1
}

Write-Host "Compile OK. Running lightweight smoke command..."
python -c "from summarize.run_summarize import run_pipeline; out=run_pipeline('Sentence one. Sentence two.', use_model=False); assert out.get('fused','').strip(); print('smoke_ok')"
if ($LASTEXITCODE -ne 0) { Write-Error "lightweight smoke failed"; exit 1 }
Write-Host "Smoke test complete."