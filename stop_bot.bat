@echo off
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Football AI Telegram Bot*" >nul 2>&1
taskkill /F /IM python.exe /FI "IMAGENAME eq python.exe" >nul 2>&1
echo Bot stopped.
pause
