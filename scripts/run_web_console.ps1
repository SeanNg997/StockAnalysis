$ProjectDir = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectDir

$h = if ($env:HOST) { $env:HOST } else { "127.0.0.1" }
$p = if ($env:PORT) { $env:PORT } else { "8000" }

Write-Host "=============================================="
Write-Host "  StockAnalysis Web Console"
Write-Host "  地址: http://${h}:${p}"
Write-Host "=============================================="
Write-Host ""

& python -m uvicorn webapp.server:app --host $h --port $p

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[错误] uvicorn 启动失败 (退出码: $LASTEXITCODE)" -ForegroundColor Red
    Write-Host "尝试运行: pip install uvicorn fastapi" -ForegroundColor Yellow
}

Read-Host "按回车键退出"
