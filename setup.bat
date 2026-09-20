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
set "PYCMD="

REM (a) python already on PATH
python --version >nul 2>&1 && set "PYCMD=python"

REM (b) py launcher (python.org installer always registers it) -> resolve real exe
if not defined PYCMD py -3 --version >nul 2>&1 && for /f "delims=" %%i in ('py -3 -c "import sys;print(sys.executable)"') do set "PYCMD=%%i"

REM (c) scan common per-user / per-machine install dirs (3.10 ~ 3.13)
if not defined PYCMD (
    for %%V in (313 312 311 310) do (
        if not defined PYCMD if exist "%LocalAppData%\Programs\Python\Python%%V\python.exe" set "PYCMD=%LocalAppData%\Programs\Python\Python%%V\python.exe"
        if not defined PYCMD if exist "%ProgramFiles%\Python%%V\python.exe" set "PYCMD=%ProgramFiles%\Python%%V\python.exe"
    )
)

if not defined PYCMD (
    echo [1/6] Python not found. Downloading Python 3.12 ...
    curl -L --retry 2 -o "%TEMP%\python-3.12.10-amd64.exe" "https://mirrors.huaweicloud.com/python/3.12.10/python-3.12.10-amd64.exe"
    if errorlevel 1 (
        echo         Huawei mirror failed, trying python.org ...
        curl -L --retry 2 -o "%TEMP%\python-3.12.10-amd64.exe" "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
    )
    if errorlevel 1 (
        echo [ERROR] Cannot download Python. Check internet / firewall.
        echo         Install manually from https://www.python.org/downloads/
        pause
        exit /b 1
    )
    echo         Installing Python silently (1-2 min) ...
    "%TEMP%\python-3.12.10-amd64.exe" /quiet InstallAllUsers=1 PrependPath=1 Include_pip=1 Include_test=0
    timeout /t 5 >nul
    set "PYCMD=%ProgramFiles%\Python312\python.exe"
    if not exist "%PYCMD%" set "PYCMD=%LocalAppData%\Programs\Python\Python312\python.exe"
    if not exist "%PYCMD%" (
        echo [ERROR] Python install failed. Install manually.
        pause
        exit /b 1
    )
)
echo [1/6] Python OK:
"%PYCMD%" --version

REM ---- 2) dependencies (Tsinghua mirror for China servers) ----
echo.
echo [2/6] Installing dependencies ...
"%PYCMD%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo [WARN] Some packages failed. Retrying without mirror ...
    "%PYCMD%" -m pip install -r requirements.txt
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
"%PYCMD%" daily_run.py --predict-only
echo.
echo [4/6] Generating national map ...
"%PYCMD%" national_map.py

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
start "zhaoxia-report" cmd /k "%PYCMD% -m http.server 8080 --directory output"

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
