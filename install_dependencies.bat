@echo off
echo ===========================================
echo   IA-Vision System Dependencies Installer
echo ===========================================

:: Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.10 and add it to PATH.
    pause
    exit /b
)

:: Create venv if it doesn't exist
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

:: Activate venv and install requirements
echo [INFO] Activating virtual environment...
call venv\Scripts\activate

echo [INFO] Updating pip...
python -m pip install --upgrade pip

echo [INFO] Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt

if %ERRORLEVEL% eq 0 (
    echo [SUCCESS] All dependencies installed successfully!
    echo You can now use start.bat to run the system.
) else (
    echo [ERROR] Installation failed. Please check the error messages above.
)

pause
