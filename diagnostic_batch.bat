@echo off
echo ========================================
echo DIAGNOSTIC BATCH - Interface Corrige
echo ========================================
echo.

echo 1. Verifying current directory...
echo Current: %cd%
echo.

echo 2. Checking Python venv...
if exist venv311\Scripts\python.exe (
  echo [OK] Python venv found
  echo Python version:
  venv311\Scripts\python.exe --version
) else (
  echo [ERROR] Python venv not found!
  pause
  exit /b 1
)
echo.

echo 3. Testing basic Python import...
venv311\Scripts\python.exe -c "import tkinter; print('[OK] tkinter imported successfully')"
if errorlevel 1 (
  echo [ERROR] tkinter import failed!
  pause
  exit /b 1
)
echo.

echo 4. Testing GUI script import...
venv311\Scripts\python.exe -c "import video_converter_gui; print('[OK] video_converter_gui imported successfully')"
if errorlevel 1 (
  echo [ERROR] video_converter_gui import failed!
  pause
  exit /b 1
)
echo.

echo 5. Attempting to launch GUI...
echo Starting video_converter_gui.py...
venv311\Scripts\python.exe video_converter_gui.py
echo.

echo 6. GUI execution completed (exit code: %errorlevel%)
echo.
echo ========================================
echo DIAGNOSTIC COMPLETED
echo ========================================
pause 