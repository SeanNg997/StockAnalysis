@echo off
setlocal

if /i not "%~1"=="__inner" (
  start "StockAnalysis Web Console" cmd /k "\"%~f0\" __inner"
  exit /b
)

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_DIR=%%~fI"

if "%HOST%"=="" set "HOST=127.0.0.1"
if "%PORT%"=="" set "PORT=8000"

set "USE_PY_LAUNCHER=0"
set "PYTHON_EXE="

if "%PYTHON_BIN%"=="" (
  if exist "%PROJECT_DIR%\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%PROJECT_DIR%\.venv\Scripts\python.exe"
  ) else (
    where py >nul 2>nul
    if not errorlevel 1 (
      set "USE_PY_LAUNCHER=1"
    ) else (
      set "PYTHON_EXE=python"
    )
  )
 ) else (
  set "PYTHON_EXE=%PYTHON_BIN%"
)

cd /d "%PROJECT_DIR%"

echo ==============================================
echo   StockAnalysis Web Console
echo   地址: http://%HOST%:%PORT%
echo ==============================================

if "%USE_PY_LAUNCHER%"=="1" (
  py -3 -m uvicorn webapp.server:app --host "%HOST%" --port "%PORT%"
) else (
  "%PYTHON_EXE%" -m uvicorn webapp.server:app --host "%HOST%" --port "%PORT%"
)
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo.
  echo 启动失败，退出码: %EXIT_CODE%
)

echo.
echo 按任意键关闭窗口...
pause >nul
exit /b %EXIT_CODE%
