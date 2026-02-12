# Lab Dojo Installer Build Scripts

This directory contains scripts to build signed `.exe` (Windows) and `.dmg` (macOS) installers that **eliminate virus warnings**.

## Why This Solves the Problem

The current `.bat` and `.command` files trigger security warnings because:
1. They're unsigned scripts
2. Windows/macOS flag any executable script from the internet  
3. They download and run code dynamically

**Solution:** Package the installer as a signed binary:
- Windows: `.exe` signed with a code signing certificate
- macOS: `.app` signed with an Apple Developer ID, packaged in a `.dmg`

## Building the Installers

### Windows (.exe)

1. Install Python 3.9+ and PyInstaller:
   ```cmd
   pip install pyinstaller
   ```

2. Run the build script:
   ```cmd
   cd build
   build_windows.bat
   ```

3. **Sign the .exe** (removes SmartScreen warnings):
   ```cmd
   signtool sign /f your_cert.pfx /p password /t http://timestamp.digicert.com dist\LabDojo_v0.1.2_Setup.exe
   ```

   **Where to get a certificate:**
   - [DigiCert](https://www.digicert.com/code-signing) (~$400/year)
   - [Sectigo](https://sectigo.com/ssl-certificates-tls/code-signing) (~$200/year)
   - [SSL.com](https://www.ssl.com/code-signing/) (~$200/year)

### macOS (.dmg)

1. Install Python 3.9+, PyInstaller, and create-dmg:
   ```bash
   pip3 install pyinstaller
   brew install create-dmg
   ```

2. Run the build script:
   ```bash
   cd build
   ./build_macos.sh
   ```

3. **Sign the .app** (removes Gatekeeper warnings):
   ```bash
   codesign --force --deep --sign "Developer ID Application: Your Name" "dist/LabDojo Installer.app"
   codesign --verify --deep --strict --verbose=2 "dist/LabDojo Installer.app"
   ```

   Then rebuild the .dmg with the signed .app.

   **Where to get a certificate:**
   - [Apple Developer Program](https://developer.apple.com/programs/) ($99/year)

## Alternative: Notarization (macOS)

For maximum trust on macOS, also **notarize** the .dmg:

```bash
xcrun notarytool submit LabDojo_v0.1.2_Installer.dmg \
    --apple-id your@email.com \
    --team-id TEAMID \
    --password app-specific-password \
    --wait

xcrun stapler staple LabDojo_v0.1.2_Installer.dmg
```

## Free Alternative: Self-Signing (Development Only)

For testing, you can self-sign (but users will still see warnings):

**Windows:**
```cmd
makecert -r -pe -n "CN=Lab Dojo" -ss My -sr CurrentUser
signtool sign /n "Lab Dojo" /t http://timestamp.digicert.com dist\LabDojo_v0.1.2_Setup.exe
```

**macOS:**
```bash
codesign --force --deep --sign - "dist/LabDojo Installer.app"
```

## What the Installer Does

The GUI installer (`installer_gui.py`):
1. Checks Python 3.9+ is installed
2. Installs dependencies (fastapi, uvicorn, aiohttp, etc.)
3. Checks for Ollama (optional)
4. Creates config directory (`~/.labdojo`)
5. Launches `labdojo.py`

## Distribution

Once signed, upload the installers to GitHub Releases:
- `LabDojo_v0.1.2_Setup.exe` (Windows)
- `LabDojo_v0.1.2_Installer.dmg` (macOS)

Users can download and run without virus warnings.
