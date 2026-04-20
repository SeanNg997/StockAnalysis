@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0\.."

echo ==============================================
echo   StockAnalysis Web Console
echo   地址: http://127.0.0.1:8000
echo ==============================================
echo.
echo 正在启动服务... 按 Ctrl+C 可停止
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "python -m uvicorn webapp.server:app --host 127.0.0.1 --port 8000"

echo.
echo 服务已停止。按任意键关闭窗口...
pause >nul
