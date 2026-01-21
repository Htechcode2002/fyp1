@echo off
echo ===========================================
echo   IA-Vision System One-Click Launcher
echo ===========================================

:: Check if venv exists
if not exist "venv" (
    echo [ERROR] Virtual environment 'venv' not found!
    echo Please follow the User Manual to set up the environment first.
    pause
    exit /b
)

:: Activate venv and run main.py
echo [INFO] Activating virtual environment...
call venv\Scripts\activate

echo [INFO] Starting IA-Vision System...
python main.py

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Application crashed. Check crash_log.txt for details.
    pause
)

pause
