@echo off
setlocal
title Zhaoxia Prediction - Full Auto Setup
cd /d "%~dp0"

echo ==============================================
echo   Zhaoxia Prediction - FULL AUTO SETUP
echo   (Python + deps + first run + scheduled tasks)
echo ==============================================
echo.

REM ---- require administrator (needed for install + scheduled tasks) ----
net session >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Please run as Administrator:
    echo         Right-click setup.bat  -^>  "Run as administrator"
    echo         then run again.
    pause
    exit /b 1
)

REM ---- 1) Locate or auto-install Python ----
set "PY="
python --version >nul 2>&1 && set "PY=python"
if not defined PY if exist "%ProgramFiles%\Python312\python.exe" set "PY=%ProgramFiles%\Python312\python.exe"
if not defined PY if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PY=%LocalAppData%\Programs\Python\Python312\python.exe"

if not defined PY (
    echo [1/6] Python not found. Downloading Python 3.12.7 ...
    curl -L --retry 2 -o "%TEMP%\python-3.12.7-amd64.exe" "https://mirrors.huaweicloud.com/python/3.12.7/python-3.12.7-amd64.exe"
    if errorlevel 1 (
        echo         Huawei mirror failed, trying python.org ...
        curl -L --retry 2 -o "%TEMP%\python-3.12.7-amd64.exe" "https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe"
    )
    if errorlevel 1 (
        echo [ERROR] Cannot download Python. Check internet / firewall.
        echo         Install manually from https://www.python.org/downloads/
        pause
        exit /b 1
    )
    echo         Installing Python silently (1-2 min) ...
    "%TEMP%\python-3.12.7-amd64.exe" /quiet InstallAllUsers=1 PrependPath=1 Include_pip=1 Include_test=0
    timeout /t 5 >nul
    set "PY=%ProgramFiles%\Python312\python.exe"
    if not exist "%PY%" set "PY=%LocalAppData%\Programs\Python\Python312\python.exe"
    if not exist "%PY%" (
        echo [ERROR] Python install failed. Install manually.
        pause
        exit /b 1
    )
)
echo [1/6] Python OK:
"%PY%" --version

REM ---- 2) dependencies (Tsinghua mirror for China servers) ----
echo.
echo [2/6] Installing dependencies ...
"%PY%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo [WARN] Some packages failed. Retrying without mirror ...
    "%PY%" -m pip install -r requirements.txt
)

REM ---- 3) local config ----
echo.
if not exist config.local.yaml (
    echo [INFO] config.local.yaml not found - WeChat push DISABLED.
    echo        (Optional) copy config.local.yaml.example to config.local.yaml
    echo        and fill your ServerChan SendKey to enable WeChat push.
)

REM ---- 4) first run ----
echo.
echo [3/6] Generating single-city report ...
"%PY%" daily_run.py --predict-only
echo.
echo [4/6] Generating national map ...
"%PY%" national_map.py

REM ---- 5) entry page ----
if exist site_index.html copy /y site_index.html output\index.html >nul

REM ---- 6) scheduled tasks ----
echo.
echo [5/6] Registering scheduled tasks ...
schtasks /Create /F /TN "ZhaoxiaDaily" /TR "cmd /c \"%~dp0task_daily.bat\"" /SC DAILY /ST 06:30 >nul
schtasks /Create /F /TN "ZhaoxiaWeb" /TR "cmd /c \"%~dp0task_web.bat\"" /SC ONSTART /RL HIGHEST >nul
echo         - ZhaoxiaDaily : every day 06:30 (predict + WeChat push)
echo         - ZhaoxiaWeb   : on system start (web server)

REM ---- 7) start web now ----
echo.
echo [6/6] Starting web server now ...
start "zhaoxia-report" cmd /k "\"%PY%\" -m http.server 8080 --directory output"

echo.
echo ==============================================
echo   SETUP COMPLETE
echo ==============================================
echo   Entry page:  http://127.0.0.1:8080/
echo   Report:      http://127.0.0.1:8080/report.html
echo   Map:         http://127.0.0.1:8080/national_map.html
echo   Public:      http://YOUR_SERVER_IP:8080/
echo.
echo   NOTE:
echo   - Open TCP 8080 in Windows Firewall for public access.
echo   - Add your server IP to the Tencent map key whitelist.
echo ==============================================
pause
