@echo off
REM Called by scheduled task "ZhaoxiaWeb" on system start (web server).
cd /d "%~dp0"
python -m http.server 8080 --directory output
