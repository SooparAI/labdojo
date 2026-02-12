#!/bin/bash
# Build Lab Dojo macOS Installer (.dmg)
# Requires: Python 3.9+, PyInstaller, create-dmg

set -e

echo "Building Lab Dojo macOS Installer..."

# Install dependencies
pip3 install --quiet pyinstaller
brew install create-dmg 2>/dev/null || true

# Build the .app
pyinstaller --onefile \
    --windowed \
    --name="LabDojo Installer" \
    --icon=../logo.png \
    --add-data="../labdojo.py:." \
    --add-data="../science_catalog.json:." \
    installer_gui.py

# Create .dmg from .app
create-dmg \
    --volname "Lab Dojo v0.1.2" \
    --volicon "../logo.png" \
    --window-pos 200 120 \
    --window-size 600 400 \
    --icon-size 100 \
    --icon "LabDojo Installer.app" 175 120 \
    --hide-extension "LabDojo Installer.app" \
    --app-drop-link 425 120 \
    "LabDojo_v0.1.2_Installer.dmg" \
    "dist/LabDojo Installer.app"

echo ""
echo "✓ Installer built: LabDojo_v0.1.2_Installer.dmg"
echo ""
echo "To sign the .dmg (removes virus warnings):"
echo "1. Get an Apple Developer ID certificate"
echo "2. Run: codesign --force --deep --sign 'Developer ID Application: Your Name' dist/LabDojo\\ Installer.app"
echo "3. Run: codesign --verify --deep --strict --verbose=2 dist/LabDojo\\ Installer.app"
echo "4. Rebuild the .dmg with the signed .app"
echo ""
