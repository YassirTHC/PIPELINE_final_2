@echo off
:: ========================================
:: LANCEMENT PIPELINE CORE (patch 2025)
:: ========================================

chcp 65001 >nulif not exist logs mkdir logs
cd /d %~dp0

echo ========================================
echo  PIPELINE IA (mode core)
echo ========================================
echo.
echo [DIR ] %cd%

:: -- venv ---------------------------------
if not exist venv311\Scripts\python.exe (
  echo [WARN] venv311 introuvable, creation...
  where py >nul 2>&1
  if %errorlevel% equ 0 (
    py -3.11 -m venv venv311 2>>logs\venv_setup.log || py -3 -m venv venv311 2>>logs\venv_setup.log || python -m venv venv311 2>>logs\venv_setup.log
  ) else (
    python -m venv venv311 2>>logs\venv_setup.log
  )
)
set "PYEXE=%cd%\venv311\Scripts\python.exe"
if not exist "%PYEXE%" (
  echo [ERREUR] impossible de trouver %PYEXE%
  pause
  exit /b 1
)

:: -- env ----------------------------------
set "PYTHONPATH=%cd%;%cd%\AI-B-roll;%cd%\AI-B-roll\src;%cd%\utils"
set PYTHONIOENCODING=utf-8
set ENABLE_PIPELINE_CORE_FETCHER=true

echo [INFO] PYTHONPATH=%PYTHONPATH%
echo [INFO] Pipeline core active (ENABLE_PIPELINE_CORE_FETCHER=true)

:: -- sanity import ------------------------
"%PYEXE%" -c "import video_processor; import utils.pipeline_integration" || (
  echo [ERREUR] Import de video_processor/utils.pipeline_integration impossible
  pause
  exit /b 1
)

:: -- lancer l'interface -------------------
if exist video_converter_gui.py (
  echo [INFO] Lancement video_converter_gui.py (mode core)
  "%PYEXE%" video_converter_gui.py %*
) else if exist main.py (
  echo [INFO] Lancement main.py (fallback)
  "%PYEXE%" main.py %*
) else (
  echo [ERREUR] interface introuvable (video_converter_gui.py / main.py)
  pause
  exit /b 1
)

pause
