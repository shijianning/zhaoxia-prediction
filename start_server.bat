@echo off
title Zhaoxia Report - Web Server
cd /d "%~dp0"
echo ==============================================
echo   Serving the report on port 8080
echo ==============================================
echo   Entry page: http://127.0.0.1:8080/
echo   Report:     http://127.0.0.1:8080/report.html
echo   Map:        http://127.0.0.1:8080/national_map.html
echo   Public:     http://YOUR_SERVER_IP:8080/
echo.
echo   Keep this window open. Press Ctrl+C to stop.
echo ==============================================
python -m http.server 8080 --directory output
pause
