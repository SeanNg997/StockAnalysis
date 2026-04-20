@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_DIR=%%~fI"

if "%HOST%"=="" set "HOST=127.0.0.1"
if "%PORT%"=="" set "PORT=8000"

if "%PYTHON_BIN%"=="" (
  if exist "%PROJECT_DIR%\.venv\Scripts\python.exe" (
    set "PYTHON_BIN=%PROJECT_DIR%\.venv\Scripts\python.exe"
  ) else (
    where py >nul 2>nul
    if %ERRORLEVEL%==0 (
      set "PYTHON_BIN=py -3"
    ) else (
      set "PYTHON_BIN=python"
    )
  )
)

cd /d "%PROJECT_DIR%"

echo ==============================================
echo   StockAnalysis Web Console
echo   地址: http://%HOST%:%PORT%
echo ==============================================

%PYTHON_BIN% -m uvicorn webapp.server:app --host "%HOST%" --port "%PORT%"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo.
  echo 启动失败，退出码: %EXIT_CODE%
)

echo.
pause
exit /b %EXIT_CODE%
