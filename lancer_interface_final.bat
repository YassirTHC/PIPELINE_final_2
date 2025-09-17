@echo off
:: ========================================
:: LANCEMENT INTERFACE FINAL - CORRIGE
:: ========================================

:: Force UTF-8 code page
chcp 65001 >nul

:: Move to script directory
cd /d %~dp0

:: Create logs directory if missing
if not exist "logs" mkdir "logs"
set LOGDIR=%CD%\logs

echo ========================================
echo LANCEMENT INTERFACE CORRIGE
echo ========================================
echo.
echo Current directory: %cd%
echo Log directory: %LOGDIR%
echo.

:: Check Python venv
if not exist venv311\Scripts\python.exe (
  echo [ERROR] Python venv not found!
  echo Creating Python venv...
  where py >nul 2>&1
  if not errorlevel 1 (
    py -3.11 -m venv venv311 2>>"%LOGDIR%\venv_setup.log" || py -3 -m venv venv311 2>>"%LOGDIR%\venv_setup.log" || python -m venv venv311 2>>"%LOGDIR%\venv_setup.log"
  ) else (
    python -m venv venv311 2>>"%LOGDIR%\venv_setup.log"
  )
)

:: Set Python executable path
set PYEXE=%CD%\venv311\Scripts\python.exe
if not exist "%PYEXE%" (
  echo [ERROR] Failed to locate venv python at %PYEXE%
  pause
  exit /b 1
)

echo [OK] Python venv found: %PYEXE%
echo Python version:
"%PYEXE%" --version
echo.

:: Test basic imports
echo Testing imports...
"%PYEXE%" -c "import tkinter; print('[OK] tkinter imported successfully')" 2>&1
if errorlevel 1 (
  echo [ERROR] tkinter import failed!
  pause
  exit /b 1
)

echo.

:: Launch GUI
echo ========================================
echo LAUNCHING GUI...
echo ========================================
echo.

if exist video_converter_gui.py (
  echo Starting video_converter_gui.py...
  echo.
  "%PYEXE%" video_converter_gui.py
  echo.
  echo GUI closed.
) else (
  echo [ERROR] GUI script video_converter_gui.py not found!
  pause
  exit /b 1
)

echo.
echo ========================================
echo INTERFACE SESSION COMPLETED
echo ========================================
pause 