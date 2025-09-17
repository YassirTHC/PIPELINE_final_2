# Script d'activation automatique du venv pour le pipeline
# Utilisation: .\activate_pipeline.ps1

Write-Host "🚀 Activation de l'environnement pipeline..." -ForegroundColor Green

# Vérifier si le venv existe
if (Test-Path "venv311\Scripts\Activate.ps1") {
    Write-Host "✅ Environnement virtuel trouvé" -ForegroundColor Green
    
    # Activer le venv
    & "venv311\Scripts\Activate.ps1"
    
    # Vérifier Mediapipe
    try {
        python -c "import mediapipe; print('✅ Mediapipe disponible')"
        Write-Host "🎯 Pipeline prêt à l'utilisation!" -ForegroundColor Green
        Write-Host "💡 Utilisez 'deactivate' pour désactiver l'environnement" -ForegroundColor Yellow
    } catch {
        Write-Host "❌ Problème avec Mediapipe" -ForegroundColor Red
    }
} else {
    Write-Host "❌ Environnement virtuel non trouvé" -ForegroundColor Red
    Write-Host "💡 Créez d'abord l'environnement: python -m venv venv311" -ForegroundColor Yellow
}
