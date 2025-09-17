@echo off
echo ========================================
echo LANCEMENT INTERFACE AVEC DEBUG
echo ========================================
echo.

:: Force UTF-8 code page
chcp 65001 >nul

:: Move to script directory
cd /d %~dp0

:: Create logs directory if missing
if not exist "logs" mkdir "logs"
set LOGDIR=%CD%\logs

echo Current directory: %cd%
echo Log directory: %LOGDIR%
echo.

:: Check Python venv
if not exist venv311\Scripts\python.exe (
  echo [ERROR] Python venv not found!
  pause
  exit /b 1
)

echo [OK] Python venv found
echo.

:: Test basic imports
echo Testing imports...
venv311\Scripts\python.exe -c "import tkinter; print('[OK] tkinter')" 2>&1
if errorlevel 1 (
  echo [ERROR] tkinter import failed!
  pause
  exit /b 1
)

:: Launch GUI with error capture
echo.
echo Launching GUI with error capture...
echo ========================================
venv311\Scripts\python.exe video_converter_gui.py 2>&1
set EXIT_CODE=%errorlevel%
echo ========================================
echo GUI exited with code: %EXIT_CODE%
echo.

if %EXIT_CODE% neq 0 (
  echo [ERROR] GUI failed with exit code %EXIT_CODE%
  echo Check the output above for error details.
) else (
  echo [OK] GUI completed successfully
)

echo.
echo ========================================
echo DEBUG COMPLETED
echo ========================================
pause 