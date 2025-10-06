import subprocess
import sys
import os

def install_requirements():
    """Installe les dépendances Python"""
    print("[setup] Installation des dépendances Python...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def install_ffmpeg():
    """Guide pour installer FFmpeg"""
    print(
        "Installation FFmpeg requise:\n"
        "Windows:\n"
        "1. Téléchargez FFmpeg depuis https://ffmpeg.org/download.html\n"
        "2. Ajoutez le dossier bin à votre PATH\n\n"
        "macOS:\n"
        "brew install ffmpeg\n\n"
        "Linux:\n"
        "sudo apt update && sudo apt install ffmpeg"
    )

def setup_directories():
    """Crée la structure de dossiers"""
    folders = ["clips", "output", "temp", "scripts"]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"[setup] Dossier créé: {folder}/")

def main():
    print("[setup] Configuration du pipeline de clips viraux")
    install_requirements()
    setup_directories()
    install_ffmpeg()
    print("[setup] Installation terminée")

if __name__ == "__main__":
    main()
