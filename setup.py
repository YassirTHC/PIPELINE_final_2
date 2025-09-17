import subprocess
import sys
import os

def install_requirements():
    """Installe les dépendances Python"""
    print("📦 Installation des dépendances Python...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def install_ffmpeg():
    """Guide pour installer FFmpeg"""
    print("""
    🎬 Installation FFmpeg requise:
    
    Windows:
    1. Téléchargez FFmpeg depuis https://ffmpeg.org/download.html
    2. Ajoutez le dossier bin à votre PATH

    macOS:
    brew install ffmpeg

    Linux:
    sudo apt update && sudo apt install ffmpeg
    """)

def setup_directories():
    """Crée la structure de dossiers"""
    folders = ["clips", "output", "temp", "scripts"]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"📁 Dossier créé: {folder}/")

def main():
    print("🚀 Configuration du pipeline de clips viraux")
    install_requirements()
    setup_directories()
    install_ffmpeg()
    print("✅ Setup terminé!")

if __name__ == "__main__":
    main()
