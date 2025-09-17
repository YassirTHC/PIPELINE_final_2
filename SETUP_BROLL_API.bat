@echo off
:: ========================================
:: CONFIGURATION CLÉS API B-ROLL
:: ========================================

chcp 65001 >nul
cd /d %~dp0

echo ========================================
echo 🔑 CONFIGURATION CLÉS API B-ROLL
echo ========================================
echo.
echo Ce script vous aide à configurer les clés API
echo pour le téléchargement automatique de B-rolls
echo.

:: Vérifier les clés existantes
echo 🔍 Vérification des clés existantes...
echo.

if defined PEXELS_API_KEY (
    echo ✅ PEXELS_API_KEY déjà configurée
) else (
    echo ❌ PEXELS_API_KEY non configurée
)

if defined PIXABAY_API_KEY (
    echo ✅ PIXABAY_API_KEY déjà configurée  
) else (
    echo ❌ PIXABAY_API_KEY non configurée
)

echo.
echo ========================================
echo 📋 INSTRUCTIONS RAPIDES
echo ========================================
echo.
echo 1. 🔑 PEXELS (RECOMMANDÉ - GRATUIT):
echo    - Allez sur: https://www.pexels.com/api/
echo    - Créez un compte gratuit
echo    - Obtenez votre clé API
echo    - Quotas: 200 requêtes/heure, 20k/mois
echo.
echo 2. 🔑 PIXABAY (ALTERNATIF):
echo    - Allez sur: https://pixabay.com/api/docs/
echo    - Créez un compte gratuit  
echo    - Obtenez votre clé API
echo    - Quotas: 5k requêtes/heure
echo.

:: Configuration interactive
echo ========================================
echo ⚙️ CONFIGURATION INTERACTIVE
echo ========================================
echo.

set /p "config_choice=Voulez-vous configurer les clés API maintenant? (o/n): "

if /i "%config_choice%"=="o" (
    echo.
    echo 🔑 Configuration PEXELS API:
    echo.
    set /p "pexels_key=Entrez votre clé API Pexels (ou Enter pour ignorer): "
    
    if not "!pexels_key!"=="" (
        setx PEXELS_API_KEY "!pexels_key!" >nul
        echo ✅ PEXELS_API_KEY configurée pour cette session et les futures
        set PEXELS_API_KEY=!pexels_key!
    )
    
    echo.
    echo 🔑 Configuration PIXABAY API:
    echo.
    set /p "pixabay_key=Entrez votre clé API Pixabay (ou Enter pour ignorer): "
    
    if not "!pixabay_key!"=="" (
        setx PIXABAY_API_KEY "!pixabay_key!" >nul
        echo ✅ PIXABAY_API_KEY configurée pour cette session et les futures
        set PIXABAY_API_KEY=!pixabay_key!
    )
    
    echo.
    echo ========================================
    echo 🧪 TEST DES CLÉS CONFIGURÉES
    echo ========================================
    echo.
    
    :: Lancer le diagnostic
    if exist "venv311\Scripts\python.exe" (
        echo 🔍 Test des clés API...
        venv311\Scripts\python.exe DIAGNOSTIC_BROLL_FETCHING.py
    ) else (
        echo 🔍 Test des clés API...
        python DIAGNOSTIC_BROLL_FETCHING.py
    )
    
) else (
    echo.
    echo ⚠️ Configuration ignorée
    echo.
    echo 📋 Pour configurer manuellement:
    echo    set PEXELS_API_KEY=votre_cle_pexels
    echo    set PIXABAY_API_KEY=votre_cle_pixabay
    echo.
    echo 💾 Pour une configuration permanente:
    echo    setx PEXELS_API_KEY "votre_cle_pexels"
    echo    setx PIXABAY_API_KEY "votre_cle_pixabay"
)

echo.
echo ========================================
echo 📝 FICHIER .ENV (ALTERNATIF)
echo ========================================
echo.
echo Vous pouvez aussi créer un fichier .env avec:
echo.
echo PEXELS_API_KEY=votre_cle_pexels
echo PIXABAY_API_KEY=votre_cle_pixabay
echo BROLL_FETCH_ENABLE=True
echo BROLL_FETCH_PROVIDER=pexels
echo.

echo ========================================
echo ✅ CONFIGURATION TERMINÉE
echo ========================================
echo.
echo 🚀 PROCHAINES ÉTAPES:
echo 1. Redémarrez votre terminal (pour les nouvelles variables)
echo 2. Lancez une vidéo test via l'interface
echo 3. Vérifiez les logs pour "Fetch B-roll"
echo 4. Contrôlez AI-B-roll\broll_library\fetched\
echo.

pause 