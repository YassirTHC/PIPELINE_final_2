@echo off
:: ========================================
:: LANCEMENT PIPELINE AVEC B-ROLLS ACTIVÉS
:: ========================================

chcp 65001 >nul
cd /d %~dp0

echo ========================================
echo 🚀 PIPELINE VIDÉO - B-ROLLS ACTIVÉS
echo ========================================
echo.
echo 🕒 %DATE% %TIME%
echo 📁 Dossier: %cd%
echo.

:: Configurer les clés API automatiquement
echo 🔑 Configuration automatique des clés API...
set PEXELS_API_KEY=pwhBa9K7fa9IQJCmfCy0NfHFWy8QyqoCkGnWLK3NC2SbDTtUeuhxpDoD
set PIXABAY_API_KEY=51724939-ee09a81ccfce0f5623df46a69

:: Configuration B-roll
set BROLL_FETCH_ENABLE=True
set BROLL_FETCH_PROVIDER=pexels
set BROLL_FETCH_ALLOW_VIDEOS=True
set BROLL_FETCH_ALLOW_IMAGES=True
set BROLL_FETCH_MAX_PER_KEYWORD=25
set BROLL_DELETE_AFTER_USE=True
set BROLL_PURGE_AFTER_RUN=True

echo ✅ Clés API configurées:
echo    PEXELS: %PEXELS_API_KEY:~0,8%******
echo    PIXABAY: %PIXABAY_API_KEY:~0,8%******
echo ✅ B-roll fetch: ACTIVÉ
echo.

:: Vérifier Python venv
if not exist venv311\Scripts\python.exe (
  echo ❌ [ERREUR] Python venv non trouvé!
  echo.
  echo 🔧 Veuillez d'abord installer l'environnement Python
  pause
  exit /b 1
)

echo ✅ Python venv trouvé
echo.

:: Test rapide des APIs (optionnel)
set /p "test_apis=Voulez-vous tester les APIs avant de lancer l'interface? (o/n): "

if /i "%test_apis%"=="o" (
    echo.
    echo 🧪 Test des APIs B-roll...
    venv311\Scripts\python.exe test_broll_system_complete.py
    echo.
    pause
    echo.
)

:: Lancer l'interface
echo 🎬 Lancement de l'interface...
echo.
echo ========================================
echo 🎯 INTERFACE PIPELINE VIDÉO IA
echo ========================================
echo.
echo 📋 INSTRUCTIONS:
echo 1. Glissez-déposez votre vidéo dans l'interface
echo 2. Les B-rolls se téléchargeront automatiquement
echo 3. Vérifiez les logs pour "Fetch B-roll"
echo 4. Contrôlez AI-B-roll\broll_library\fetched\
echo.

:: Lancer l'interface principale
if exist "video_converter_gui.py" (
    echo 🚀 Lancement GUI...
    venv311\Scripts\python.exe video_converter_gui.py
) else if exist "main.py" (
    echo 🚀 Lancement main...
    venv311\Scripts\python.exe main.py
) else (
    echo ❌ Interface non trouvée!
    echo 🔍 Fichiers disponibles:
    dir *.py | findstr /i "gui\|main\|interface"
    pause
)

echo.
echo ========================================
echo 📊 NETTOYAGE ET RAPPORTS
echo ========================================
echo.

:: Afficher les statistiques B-roll après utilisation
if exist "AI-B-roll\broll_library\fetched" (
    echo 📊 Statistiques B-roll:
    for /r "AI-B-roll\broll_library\fetched" %%f in (*.mp4) do (
        echo    📹 %%~nxf
    )
) else (
    echo ⚠️ Aucun B-roll téléchargé trouvé
)

echo.
echo ✅ Session terminée
pause 