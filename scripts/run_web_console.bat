@echo off
chcp 65001 >nul 2>&1

cd /d "%~dp0\.."

set HOST=127.0.0.1
set PORT=8000

where python >nul 2>&1
if %errorlevel% neq 0 (
    where python3 >nul 2>&1
    if %errorlevel% neq 0 (
        echo [错误] 未找到 Python，请确认已安装并添加到 PATH
        echo.
        pause
        exit /b 1
    )
    set PYTHON_BIN=python3
) else (
    set PYTHON_BIN=python
)

echo ==============================================
echo   StockAnalysis Web Console
echo   Python: %PYTHON_BIN%
echo   地址: http://%HOST%:%PORT%
echo ==============================================
echo.

%PYTHON_BIN% -c "import uvicorn" >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未安装 uvicorn，请运行: pip install uvicorn
    echo.
    pause
    exit /b 1
)

%PYTHON_BIN% -c "import webapp.server" >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 无法导入 webapp.server，请确认依赖已安装
    echo   尝试运行: pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo 正在启动服务...
echo 按 Ctrl+C 可停止服务
echo.

%PYTHON_BIN% -m uvicorn webapp.server:app --host %HOST% --port %PORT%

echo.
echo 服务已停止。按任意键关闭窗口...
pause >nul
