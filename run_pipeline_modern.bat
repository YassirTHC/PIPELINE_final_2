@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion

rem --- Force UTF-8 + live stdout ---
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1

rem --- Move to repo root ---
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

rem --- Pick python interpreter (venv first) ---
set "PY_CMD=python"
if exist "venv311\Scripts\python.exe" set "PY_CMD=venv311\Scripts\python.exe"
if exist ".venv\Scripts\python.exe" set "PY_CMD=.venv\Scripts\python.exe"

rem --- Default pipeline_core settings if not overridden ---
set ENABLE_PIPELINE_CORE_FETCHER=true
if not defined BROLL_FETCH_ALLOW_VIDEOS set BROLL_FETCH_ALLOW_VIDEOS=1
if not defined BROLL_FETCH_ALLOW_IMAGES set BROLL_FETCH_ALLOW_IMAGES=0

rem --- Allow --gui flag or no args to launch the GUI ---
if "%~1"=="--gui" (
    shift
    goto launch_gui
)
if "%~1"=="" goto launch_gui

set "FIRST_ARG=%~1"

rem --- Pass through diagnostics / config flags ---
if "%FIRST_ARG:~0,2%"=="--" (
    %PY_CMD% -u run_pipeline.py %*
    goto end
)

set "VIDEO_PATH=%FIRST_ARG%"
set "VIDEO_PATH=%VIDEO_PATH:\=/%"
shift

set "EXTRA_ARGS="
:collect_args
if "%~1"=="" goto run_cli
set "EXTRA_ARGS=%EXTRA_ARGS% %~1"
shift
goto collect_args

:run_cli
echo Running command: %PY_CMD% -u run_pipeline.py --video "!VIDEO_PATH!" !EXTRA_ARGS!
%PY_CMD% -u run_pipeline.py --video "!VIDEO_PATH!" !EXTRA_ARGS!
if errorlevel 1 goto end

goto end

:launch_gui
if not exist "video_converter_gui.py" goto usage

echo Launching modern pipeline GUI...
%PY_CMD% -u video_converter_gui.py %*
if errorlevel 1 goto end

goto end

:usage
echo Usage: %~nx0 ^<video_path^> [extra run_pipeline args]
echo        %~nx0 --diag-broll [options]
echo        %~nx0 --gui

echo Examples:
echo    %~nx0 clips\121.mp4 --verbose

echo    %~nx0 --diag-broll --no-report

echo    %~nx0 --gui
exit /b 1

:end
exit /b %ERRORLEVEL%
