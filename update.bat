@echo off
chcp 65001 > nul
title Git Push → Render Auto Deploy
cd /d "%~dp0"

echo ==========================================
echo   Pushing to GitHub → Auto deploy on Render
echo ==========================================
echo.

git add .
git commit -m "update %date% %time%"
git push

echo.
echo Done! Render will auto-deploy in 2-3 minutes.
echo https://fase-tay6.onrender.com
echo.
pause
