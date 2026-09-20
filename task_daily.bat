@echo off
REM Called by scheduled task "ZhaoxiaDaily" every day (predict + WeChat push).
cd /d "%~dp0"
python daily_run.py --predict-only
