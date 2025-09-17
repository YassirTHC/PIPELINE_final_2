@echo off
:: ========================================
:: LANCEMENT PIPELINE AVEC B-ROLLS CORRIGÉS
:: ========================================

:: Force UTF-8 code page
chcp 65001 >nul

:: Move to script directory
cd /d %~dp0

echo ========================================
echo 🔧 PIPELINE VIDÉO - B-ROLLS CORRIGÉS
echo ========================================
echo.
echo 🕒 %DATE% %TIME%
echo 📁 Dossier: %cd%
echo.

:: Check Python venv
if not exist venv311\Scripts\python.exe (
  echo ❌ [ERREUR] Python venv non trouvé!
  echo.
  echo 🔧 Veuillez d'abord installer l'environnement Python
  pause
  exit /b 1
)

echo ✅ Python venv trouvé
echo.

:: Test de la correction B-rolls
echo 🔧 Test de la correction B-rolls...
echo.

venv311\Scripts\python.exe test_broll_fix.py

echo.
echo ========================================
echo 🚀 LANCEMENT DE L'INTERFACE
echo ========================================
echo.

echo 🖥️ Démarrage de l'interface graphique avec B-rolls corrigés...
echo.

:: Lancement de l'interface principale
if exist video_converter_gui.py (
  echo ▶️ Lancement video_converter_gui.py...
  echo.
  venv311\Scripts\python.exe video_converter_gui.py
  echo.
  echo ✅ Interface fermée.
) else (
  echo ❌ [ERREUR] video_converter_gui.py non trouvé!
  echo.
  echo 🔧 Essai avec main.py...
  if exist main.py (
    venv311\Scripts\python.exe main.py
  ) else (
    echo ❌ [ERREUR] Aucun fichier d'interface trouvé!
  )
)

echo.
echo ========================================
echo 📊 SESSION TERMINÉE
echo ========================================
echo.
echo 📝 Logs disponibles:
echo    • pipeline_test.log
echo    • gui_debug.log
echo    • logs/
echo.
echo 🎬 MAINTENANT LES B-ROLLS DEVRAIENT S'INSÉRER CORRECTEMENT!
echo.
pause 