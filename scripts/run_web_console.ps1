$ProjectDir = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectDir

$h = if ($env:HOST) { $env:HOST } else { "127.0.0.1" }
$p = if ($env:PORT) { $env:PORT } else { "8000" }

Write-Host "=============================================="
Write-Host "  StockAnalysis Web Console"
Write-Host "  Address: http://${h}:${p}"
Write-Host "  Do not close this terminal. Use Ctrl + Click to open the link above."
Write-Host "=============================================="
Write-Host ""

& python -m uvicorn webapp.server:app --host $h --port $p

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] uvicorn failed to start (exit code: $LASTEXITCODE)" -ForegroundColor Red
    Write-Host "Try running: pip install uvicorn fastapi" -ForegroundColor Yellow
}

Read-Host "Press Enter to exit"