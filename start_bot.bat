@echo off
chcp 65001 > nul
title Football AI Telegram Bot
cd /d "%~dp0"

echo ==========================================
echo   Football AI Telegram Bot
echo ==========================================
echo.
echo Telegram: @FootballAI_predictor_bot
echo.

python telegram_bot.py

echo.
pause
