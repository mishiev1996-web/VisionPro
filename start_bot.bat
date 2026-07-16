@echo off
chcp 65001 > nul
title Football AI Telegram Bot
cd /d "%~dp0"

echo ==========================================
echo   Football AI Telegram Bot
echo ==========================================
echo.

REM --- Load .env if exists ---
if exist ".env" (
    echo Loading .env...
    for /F "usebackq tokens=1,* delims==" %%A in (".env") do (
        set "%%A=%%B"
    )
)

REM --- Check WEBAPP_URL ---
if "%WEBAPP_URL%"=="" (
    echo [WARNING] WEBAPP_URL not set.
    echo Mini App menu button requires HTTPS URL.
    echo Set WEBAPP_URL=https://your-tunnel.ngrok.io/mini-app
    echo.
)

python telegram_bot.py

echo.
pause
