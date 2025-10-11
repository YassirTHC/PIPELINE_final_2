@echo off

chcp 65001 >nul

setlocal

rem ------------------------------------------------------------------
rem  Default environment overrides for the pipeline GUI launcher
rem ------------------------------------------------------------------
if not defined PIPELINE_LLM_PROVIDER set PIPELINE_LLM_PROVIDER=ollama
if not defined PIPELINE_LLM_ENDPOINT set PIPELINE_LLM_ENDPOINT=http://localhost:11434
if not defined PIPELINE_LLM_MODEL_TEXT set PIPELINE_LLM_MODEL_TEXT=qwen3:8b
if not defined PIPELINE_LLM_MODEL_JSON set PIPELINE_LLM_MODEL_JSON=qwen3:8b
if not defined PIPELINE_LLM_FORCE_NON_STREAM set PIPELINE_LLM_FORCE_NON_STREAM=true
if not defined PIPELINE_LLM_JSON_MODE set PIPELINE_LLM_JSON_MODE=true
if not defined PIPELINE_LLM_TIMEOUT_S set PIPELINE_LLM_TIMEOUT_S=60
if not defined PIPELINE_LLM_FALLBACK_TIMEOUT_S set PIPELINE_LLM_FALLBACK_TIMEOUT_S=30
if not defined BROLL_FETCH_PROVIDER set BROLL_FETCH_PROVIDER=pixabay,pexels
if not defined FETCH_MAX set FETCH_MAX=8

set PYTHONIOENCODING=utf-8

rem Allow switching back to legacy orchestrator easily
if not defined ENABLE_PIPELINE_CORE_FETCHER set ENABLE_PIPELINE_CORE_FETCHER=true

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

if exist "venv311\Scripts\python.exe" (
    set "PY_CMD=venv311\Scripts\python.exe"
) else (
    set "PY_CMD=python"
)

if not exist "tools\pipeline_launcher_gui.py" (
    echo [ERROR] tools\pipeline_launcher_gui.py introuvable.^&echo Faites tourner python tools\pipeline_launcher_gui.py pour verifier.^&echo.
    pause
    endlocal
    exit /b 1
)

echo =================================================
echo   Video Pipeline GUI Launcher (updated launcher)
echo =================================================
echo.
echo     PIPELINE_LLM_PROVIDER        = %PIPELINE_LLM_PROVIDER%
echo     PIPELINE_LLM_ENDPOINT        = %PIPELINE_LLM_ENDPOINT%
echo     PIPELINE_LLM_MODEL_TEXT      = %PIPELINE_LLM_MODEL_TEXT%
echo     PIPELINE_LLM_MODEL_JSON      = %PIPELINE_LLM_MODEL_JSON%
echo     PIPELINE_LLM_FORCE_NON_STREAM= %PIPELINE_LLM_FORCE_NON_STREAM%
echo     PIPELINE_LLM_JSON_MODE       = %PIPELINE_LLM_JSON_MODE%
echo     BROLL_FETCH_PROVIDER         = %BROLL_FETCH_PROVIDER%
echo     FETCH_MAX                    = %FETCH_MAX%
echo.
echo  Lance l'interface graphique...^&echo.

%PY_CMD% tools\pipeline_launcher_gui.py
set EXIT_CODE=%ERRORLEVEL%

echo.
echo  Process finished with exit code %EXIT_CODE%.
echo.

endlocal
exit /b %EXIT_CODE%

