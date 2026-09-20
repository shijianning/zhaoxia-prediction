@echo off
setlocal
title Zhaoxia Prediction - Deploy
cd /d "%~dp0"

echo ==============================================
echo   Zhaoxia Prediction - One-click Deploy
echo ==============================================
echo.

REM 1) Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.9+ not found in PATH.
    echo   Install from: https://www.python.org/downloads/
    echo   IMPORTANT: check "Add Python to PATH" during install.
    echo   Then run this script again.
    pause
    exit /b 1
)
echo [1/5] Python detected:
python --version

REM 2) Install dependencies
echo.
echo [2/5] Installing dependencies ...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [WARN] Some packages failed. You can retry with:
    echo        python -m pip install -r requirements.txt
)

REM 3) Check local config (WeChat SendKey is NOT committed to git)
echo.
if not exist config.local.yaml (
    echo [INFO] config.local.yaml not found - WeChat push is DISABLED.
    echo        (Optional) Copy config.local.yaml.example to config.local.yaml
    echo        and fill your ServerChan SendKey to enable WeChat push.
)

REM 4) First run: single-city report + national map
echo.
echo [3/5] Generating single-city report ...
python daily_run.py --predict-only
echo.
echo [4/5] Generating national map ...
python national_map.py

REM 5) Copy entry page for the web server
if exist site_index.html (
    copy /y site_index.html output\index.html >nul
)

REM 6) Start web server to host output/
echo.
echo [5/5] Starting web server on port 8080 ...
start "zhaoxia-report" cmd /k "python -m http.server 8080 --directory output"

echo.
echo ==============================================
echo   DEPLOY DONE
echo ==============================================
echo   Entry page: http://127.0.0.1:8080/
echo   Report:     http://127.0.0.1:8080/report.html
echo   Map:        http://127.0.0.1:8080/national_map.html
echo.
echo   Public access: http://YOUR_SERVER_IP:8080/
echo   (Remember to open TCP port 8080 in Windows Firewall.)
echo.
echo   After a server reboot, run start_server.bat to restart the web server.
echo ==============================================
pause
