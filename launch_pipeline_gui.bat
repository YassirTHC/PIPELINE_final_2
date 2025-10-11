@echo off
setlocal
set REPO_DIR=%~dp0
cd /d "%REPO_DIR%"
call "C:\Python313\python.exe" tools\pipeline_launcher_gui.py
endlocal
