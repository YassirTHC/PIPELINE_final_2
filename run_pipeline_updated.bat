@echo off

chcp 65001 >nul

setlocal



set PYTHONIOENCODING=utf-8

if not defined ENABLE_PIPELINE_CORE_FETCHER set ENABLE_PIPELINE_CORE_FETCHER=false

set BROLL_FETCH_ALLOW_IMAGES=false

set CONTEXTUAL_BROLL_YML=config\contextual_broll.yml



set "SCRIPT_DIR=%~dp0"

cd /d "%SCRIPT_DIR%"



set "PY_CMD=python"

if exist "venv311\Scripts\python.exe" set "PY_CMD=venv311\Scripts\python.exe"



echo ========================================

echo  Updated Video Pipeline Launcher

echo ========================================

echo.

echo     BROLL_FETCH_ALLOW_IMAGES = %BROLL_FETCH_ALLOW_IMAGES%

echo     CONTEXTUAL_BROLL_YML      = %CONTEXTUAL_BROLL_YML%

echo.



if exist "video_converter_gui.py" goto launch_gui

if exist "main.py" goto launch_main



echo Unable to locate GUI entry point (video_converter_gui.py / main.py).

dir *.py | findstr /i "gui\|main"

pause

goto end



:launch_gui

echo Launching GUI interface...

%PY_CMD% video_converter_gui.py

goto end



:launch_main

echo Launching main interface...

%PY_CMD% main.py



:end

set EXIT_CODE=%ERRORLEVEL%

exit /b %EXIT_CODE%

