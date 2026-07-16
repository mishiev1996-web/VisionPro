@echo off
chcp 65001 >nul
title Football AI Predictor
cd /d "%~dp0"

echo ==========================================
echo   Football AI Predictor
echo ==========================================
echo.

REM --- check Python ---
where python >nul 2>&1
if errorlevel 1 goto :no_python

REM --- check dependencies ---
python -c "import fastapi, uvicorn, botasaurus, sklearn, apscheduler, catboost" >nul 2>&1
if errorlevel 1 goto :install_deps
goto :check_db

:install_deps
echo [1/6] Installing dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 goto :pip_fail
echo.

:check_db
echo [2/6] Checking database...
if exist "data\football.db" (
    echo    Database found.
    goto :sync_sstats
)

echo    Database empty - collecting data (this may take 5-10 minutes)...
python data_collector.py
if errorlevel 1 (
    echo    [WARN] Data collection had errors - continuing anyway.
)
echo.

:sync_sstats
echo [3/6] Syncing sstats data...
python -c "from scrapers.sstats_collector import sync_leagues, sync_today; sync_leagues(); sync_today()"
if errorlevel 1 (
    echo    [WARN] sstats sync had errors - continuing anyway.
)
echo.

:check_model
echo [4/6] Checking model...
if exist "model.pkl" (
    echo    Model found.
    goto :check_tunnel
)

echo    No model.pkl - training model...
python train.py
if errorlevel 1 (
    echo    [WARN] Training failed - you can train manually from the UI.
)
echo.

:check_tunnel
echo [5/6] Checking ngrok tunnel...
where ngrok >nul 2>&1
if errorlevel 1 (
    echo    ngrok not found - starting without tunnel.
    echo    To enable remote access: install ngrok from https://ngrok.com
    set "WEBAPP_URL=http://localhost:8000/mini-app"
    goto :launch
)

REM --- Start ngrok in background ---
echo    Starting ngrok tunnel...
start "ngrok" ngrok http 8000 >ngrok.log 2>&1
timeout /t 3 /nobreak >nul

REM --- Get ngrok URL ---
findstr /C:"https://" ngrok.log >nul 2>&1
if errorlevel 1 (
    echo    [WARN] Could not get ngrok URL - using localhost
    set "WEBAPP_URL=http://localhost:8000/mini-app"
) else (
    for /f "tokens=9" %%u in ('findstr /C:"https://" ngrok.log') do set "NGROK_URL=%%u"
    if "!NGROK_URL!"=="" (
        echo    [WARN] Could not get ngrok URL - using localhost
        set "WEBAPP_URL=http://localhost:8000/mini-app"
    ) else (
        echo    Tunnel: !NGROK_URL!
        set "WEBAPP_URL=!NGROK_URL!/mini-app"
    )
)
echo.

:launch
echo [6/6] Starting server and bot...
echo.
REM --- open browser after 3 seconds in background ---
start "" /B python -c "import time, webbrowser; time.sleep(3); webbrowser.open('http://localhost:8000')"

echo Server: http://localhost:8000
echo Tunnel: %WEBAPP_URL%
echo.
echo Press Ctrl+C to stop.
echo.

REM --- Start bot in separate window ---
start "Football AI Bot" python telegram_bot.py
echo Telegram bot started in separate window.

REM --- Start web server (blocks here) ---
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
