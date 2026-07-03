@echo off
chcp 65001 > nul
title Football AI Predictor
cd /d "%~dp0"

echo ==========================================
echo   Football AI Predictor
echo ==========================================
echo.

REM --- check Python ---
where python > nul 2>&1
if errorlevel 1 goto :no_python

REM --- check dependencies ---
python -c "import fastapi, uvicorn, botasaurus, sklearn, apscheduler, catboost" > nul 2>&1
if errorlevel 1 goto :install_deps
goto :check_db

:install_deps
echo [1/4] Installing dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 goto :pip_fail
echo.

:check_db
echo [2/4] Checking database...
if exist "data\football.db" (
    echo    Database found.
    goto :check_model
)

echo    Database empty — collecting data (this may take 5-10 minutes)...
python data_collector.py
if errorlevel 1 (
    echo    [WARN] Data collection had errors — continuing anyway.
)
echo.

:check_model
echo [3/4] Checking model...
if exist "model.pkl" (
    echo    Model found.
    goto :launch
)

echo    No model.pkl — training model...
python train.py
if errorlevel 1 (
    echo    [WARN] Training failed — you can train manually from the UI.
)
echo.

:launch
echo [4/4] Starting server...
echo.
REM --- open browser after 3 seconds in background ---
start "" /B python -c "import time, webbrowser; time.sleep(3); webbrowser.open('http://localhost:8000')"

echo Server starting at http://localhost:8000
echo Press Ctrl+C to stop.
echo.

python -m uvicorn app:app --port 8000 --log-level info

echo.
echo Server stopped.
goto :end

:no_python
echo [ERROR] Python is not installed or not in PATH.
echo Download Python 3.10 or newer from https://www.python.org/downloads/
echo During install, check the option: Add Python to PATH
goto :end

:pip_fail
echo [ERROR] Failed to install dependencies.
goto :end

:end
echo.
pause
