; Lab Dojo Windows Installer
; NSIS Script for creating a professional Windows installer
; Requires: NSIS 3.0+ (https://nsis.sourceforge.io)
; Build: makensis LabDojo_Installer.nsi

!include "MUI2.nsh"
!include "x64.nsh"

; Basic Settings
Name "Lab Dojo v0.1.2"
OutFile "LabDojo_v0.1.2_Setup.exe"
InstallDir "$PROGRAMFILES\LabDojo"
InstallDirRegKey HKLM "Software\LabDojo" "Install_Dir"

; Request admin privileges
RequestExecutionLevel admin

; MUI Settings
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

; Installer Sections
Section "Install"
  SetOutPath "$INSTDIR"
  
  ; Download and extract Lab Dojo
  DetailPrint "Downloading Lab Dojo..."
  nsExec::ExecToLog "powershell -Command `
    $ProgressPreference = 'SilentlyContinue'; `
    Invoke-WebRequest -Uri 'https://github.com/SooparAI/labdojo/releases/download/v0.1.2/labdojo-v0.1.2.zip' `
    -OutFile '$INSTDIR\labdojo.zip'; `
    Expand-Archive -Path '$INSTDIR\labdojo.zip' -DestinationPath '$INSTDIR'; `
    Remove-Item '$INSTDIR\labdojo.zip'"
  
  ; Check Python installation
  DetailPrint "Checking Python installation..."
  nsExec::ExecToLog "python --version"
  ${If} ${Errors}
    MessageBox MB_OK "Python 3.8+ is required but not found. Please install Python from python.org"
    Abort
  ${EndIf}
  
  ; Install dependencies
  DetailPrint "Installing dependencies..."
  nsExec::ExecToLog "pip install -r requirements.txt"
  
  ; Create Start Menu shortcuts
  CreateDirectory "$SMPROGRAMS\Lab Dojo"
  CreateShortCut "$SMPROGRAMS\Lab Dojo\Lab Dojo.lnk" "$INSTDIR\labdojo.py" "" "$INSTDIR\logo.ico"
  CreateShortCut "$SMPROGRAMS\Lab Dojo\Uninstall.lnk" "$INSTDIR\uninstall.exe"
  
  ; Create registry entries
  WriteRegStr HKLM "Software\LabDojo" "Install_Dir" "$INSTDIR"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\LabDojo" "DisplayName" "Lab Dojo"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\LabDojo" "UninstallString" "$INSTDIR\uninstall.exe"
SectionEnd

; Uninstaller
Section "Uninstall"
  RMDir /r "$INSTDIR"
  RMDir /r "$SMPROGRAMS\Lab Dojo"
  DeleteRegKey HKLM "Software\LabDojo"
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\LabDojo"
SectionEnd
