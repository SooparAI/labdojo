@echo off
REM Build Lab Dojo Windows Installer (.exe)
REM Requires: Python 3.9+, PyInstaller

echo Building Lab Dojo Windows Installer...

REM Install PyInstaller if not present
pip install --quiet pyinstaller

REM Build the .exe
pyinstaller --onefile ^
    --windowed ^
    --name="LabDojo_v0.1.2_Setup" ^
    --icon=../logo.png ^
    --add-data="../labdojo.py;." ^
    --add-data="../science_catalog.json;." ^
    installer_gui.py

echo.
echo ✓ Installer built: dist\LabDojo_v0.1.2_Setup.exe
echo.
echo To sign the .exe (removes virus warnings):
echo 1. Get a code signing certificate from DigiCert, Sectigo, or similar
echo 2. Run: signtool sign /f cert.pfx /p password /t http://timestamp.digicert.com dist\LabDojo_v0.1.2_Setup.exe
echo.
pause
