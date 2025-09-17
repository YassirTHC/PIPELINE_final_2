@echo off
:: Force UTF-8 code page
chcp 65001 >nul

setlocal EnableExtensions
:: Move to script directory
cd /d %~dp0
if not exist "logs" mkdir "logs"
set LOGDIR=%CD%\logs

echo Current directory: %cd%
echo Preparing Python venv...

:: Create venv if missing
if not exist venv311\Scripts\python.exe (
  echo Creating Python venv...
  where py >nul 2>&1
  if not errorlevel 1 (
    py -3.11 -m venv venv311 2>>"%LOGDIR%\venv_setup.log" || py -3 -m venv venv311 2>>"%LOGDIR%\venv_setup.log" || python -m venv venv311 2>>"%LOGDIR%\venv_setup.log"
  ) else (
    python -m venv venv311 2>>"%LOGDIR%\venv_setup.log"
  )
)

:: Use venv python directly (avoid activate to prevent batch parsing issues)
set PYEXE=%CD%\venv311\Scripts\python.exe
if not exist "%PYEXE%" (
  echo Failed to locate venv python at %PYEXE%
  pause
  exit /b 1
)

:: Ensure minimal GUI dependencies are present (tkinterdnd2, customtkinter)
"%PYEXE%" -c "import tkinterdnd2, customtkinter" >nul 2>&1
if errorlevel 1 (
  echo [SETUP] Installing GUI dependencies ^(tkinterdnd2, customtkinter^)...
  "%PYEXE%" -m pip install --disable-pip-version-check -q tkinterdnd2 customtkinter >>"%LOGDIR%\pip_gui.log" 2>&1
  if errorlevel 1 (
    echo [ERROR] Failed to install GUI dependencies. See "%LOGDIR%\pip_gui.log".
  )
)

:: Fire-and-forget local LLM preparation before GUI (non-blocking)
set OLLAMA_EXE=
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe
if "%OLLAMA_EXE%"=="" (
  for /f "delims=" %%i in ('where ollama 2^>NUL') do set OLLAMA_EXE=%%i
)

:: If Ollama installed, configure and start server; pull and build tuned model in background
if not "%OLLAMA_EXE%"=="" (
  echo [LLM] Configuring Ollama for 4 GB VRAM...
  set GGML_CUDA_DISABLE_GRAPHS=1
  set OLLAMA_KV_CACHE_TYPE=q8_0
  set OLLAMA_CONTEXT_LENGTH=2048
  echo [LLM] Starting Ollama server...
  start "" /MIN "%OLLAMA_EXE%" serve
  echo [LLM] Pulling base model gemma3:4b...
  start "" /MIN "%OLLAMA_EXE%" pull gemma3:4b
  echo [LLM] Using qwen3:8b model (already installed)...
echo [LLM] Model qwen3:8b is ready for use
) else (
  :: If winget is available, try to install Ollama in background (do not block GUI)
  where winget >nul 2>&1
  if not errorlevel 1 (
    echo [LLM] Installing Ollama in background ^(this may take a while^)...
    start "" /MIN winget install -e --id Ollama.Ollama --silent --accept-package-agreements --accept-source-agreements >>"%LOGDIR%\ollama_install.log" 2>&1
  ) else (
    echo [LLM] Winget not found; skipping automatic Ollama install.
  )
)

echo Launching GUI...
if exist video_converter_gui.py (
  "%PYEXE%" video_converter_gui.py
) else (
  echo GUI script video_converter_gui.py not found. Skipping GUI.
)

:: Ask to run contextual pipeline after GUI closes
if defined RUNPIPE (
  set _ANS=%RUNPIPE%
) else (
  set /p _ANS="Run contextual pipeline now? (Y/N): "
)
if /I "%_ANS%"=="Y" (
  echo Launching contextual pipeline (auto-setup)...
  if exist AI-B-roll\run_example.bat (
    call AI-B-roll\run_example.bat
  ) else if exist AI-B-roll\examples\run_pipeline.py (
    "%PYEXE%" AI-B-roll\examples\run_pipeline.py >>"%LOGDIR%\pipeline_run.log" 2>&1
  ) else (
    echo Pipeline entrypoint not found at AI-B-roll\run_example.bat or AI-B-roll\examples\run_pipeline.py.
  )
) else (
  echo Pipeline skipped.
)

echo Done. Check output\ and meta\

endlocal

pause
