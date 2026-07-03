@echo off
chcp 65001 > nul
title Football AI - stop

echo Stopping server on port 8000...

REM --- find processes on :8000 and kill them ---
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo Killing PID %%a
    taskkill /F /PID %%a > nul 2>&1
)

echo Done.
timeout /t 2 > nul
