@echo off
chcp 65001 > nul
title Football AI - stop

echo Stopping all services...

REM --- kill web server on port 8000 ---
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo Killing web server PID %%a
    taskkill /F /PID %%a > nul 2>&1
)

REM --- kill ngrok ---
taskkill /F /IM ngrok.exe > nul 2>&1

REM --- kill telegram bot ---
taskkill /F /FI "WINDOWTITLE eq Football AI Bot" > nul 2>&1

echo All services stopped.
timeout /t 2 > nul
