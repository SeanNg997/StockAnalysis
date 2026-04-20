@echo off
chcp 65001 >nul 2>&1

cd /d "%~dp0\.."

set HOST=127.0.0.1
set PORT=8000

echo ==============================================
echo   StockAnalysis Web Console
echo   地址: http://%HOST%:%PORT%
echo ==============================================
echo.

python -m uvicorn webapp.server:app --host %HOST% --port %PORT%

echo.
echo 服务已停止。按任意键关闭窗口...
pause >nul
