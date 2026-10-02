@echo off
rem Update fyx sources and this benchmark harness from GitHub (Windows).
rem   set FYX_REPO=owner/name & set FYX_BRANCH=branch & update.cmd
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0suite\update.ps1"
exit /b %errorlevel%
