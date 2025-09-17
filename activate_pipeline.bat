@echo off
REM Script d'activation automatique du venv pour le pipeline
REM Utilisation: activate_pipeline.bat

echo 🚀 Activation de l'environnement pipeline...

REM Vérifier si le venv existe
if exist "venv311\Scripts\activate.bat" (
    echo ✅ Environnement virtuel trouvé
    
    REM Activer le venv
    call "venv311\Scripts\activate.bat"
    
    REM Vérifier Mediapipe
    python -c "import mediapipe; print('✅ Mediapipe disponible')"
    if %errorlevel% equ 0 (
        echo 🎯 Pipeline prêt à l'utilisation!
        echo 💡 Utilisez 'deactivate' pour désactiver l'environnement
    ) else (
        echo ❌ Problème avec Mediapipe
    )
) else (
    echo ❌ Environnement virtuel non trouvé
    echo 💡 Créez d'abord l'environnement: python -m venv venv311
)

pause
