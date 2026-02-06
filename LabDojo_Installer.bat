@echo off
REM ============================================================================
REM Lab Dojo v10 - One-Click Installer for Windows
REM ============================================================================

title Lab Dojo v10 Installer

echo ========================================
echo        Lab Dojo v10 Installer
echo        Serverless Edition
echo ========================================
echo.

REM Get script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo [1/5] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   X Python not found!
    echo   Please install Python 3.11+ from https://www.python.org/downloads/
    echo   IMPORTANT: Check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo   [OK] Python %PYTHON_VERSION% found

echo.
echo [2/5] Installing Python dependencies...
pip install --quiet --upgrade pip
pip install --quiet fastapi uvicorn aiohttp aiosqlite pydantic
echo   [OK] Dependencies installed

echo.
echo [3/5] Checking Ollama...
where ollama >nul 2>&1
if %errorlevel% neq 0 (
    echo   Ollama not found!
    echo   Please install Ollama from https://ollama.com/download/windows
    echo   After installing, run: ollama pull qwen2.5:7b
    echo.
    echo   Press any key to continue without local AI...
    pause >nul
) else (
    echo   [OK] Ollama found
    
    REM Check if Ollama is running
    tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
    if %errorlevel% neq 0 (
        echo   Starting Ollama...
        start /B ollama serve
        timeout /t 3 /nobreak >nul
    )
    
    REM Check for model
    ollama list 2>nul | findstr /i "qwen2.5:7b" >nul
    if %errorlevel% neq 0 (
        echo   Pulling Qwen 2.5 7B model...
        ollama pull qwen2.5:7b
    )
    echo   [OK] Ollama ready with Qwen 2.5 7B
)

echo.
echo [4/5] Creating config directory...
if not exist "%USERPROFILE%\.labdojo\logs" mkdir "%USERPROFILE%\.labdojo\logs"
echo   [OK] Config directory: %USERPROFILE%\.labdojo

echo.
echo [5/5] Starting Lab Dojo...
echo.

REM Kill any existing instance
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Lab Dojo*" >nul 2>&1

REM Start Lab Dojo
python "%SCRIPT_DIR%labdojo.py"

pause
