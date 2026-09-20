@echo off
REM Run daily glow prediction (training + predict + report)
cd /d "%~dp0"
python daily_run.py
pause
