@echo off
echo ========================================
echo LANCEMENT INTERFACE SIMPLIFIE
echo ========================================
echo.

:: Force UTF-8 code page
chcp 65001 >nul

:: Move to script directory
cd /d %~dp0

echo Current directory: %cd%
echo.

:: Check Python venv
if not exist venv311\Scripts\python.exe (
  echo [ERROR] Python venv not found!
  pause
  exit /b 1
)

echo [OK] Python venv found
echo.

:: Launch GUI directly
echo Launching GUI...
echo Starting video_converter_gui.py...
echo.

venv311\Scripts\python.exe video_converter_gui.py

echo.
echo GUI closed.
echo ========================================
pause 