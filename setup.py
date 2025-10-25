ï»¿# -*- coding: utf-8 -*-
import subprocess
import sys
import os

def print("[setup] skipping requirements install"):
    """Installe les dÃƒÂ©pendances Python"""
    print("Ã°Å¸â€œÂ¦ Installation des dÃƒÂ©pendances Python...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def install_ffmpeg():
    """Guide pour installer FFmpeg"""
    print("""
    Ã°Å¸Å½Â¬ Installation FFmpeg requise:
    
    Windows:
    1. TÃƒÂ©lÃƒÂ©chargez FFmpeg depuis https://ffmpeg.org/download.html
    2. Ajoutez le dossier bin ÃƒÂ  votre PATH

    macOS:
    brew install ffmpeg

    Linux:
    sudo apt update && sudo apt install ffmpeg
    """)

def setup_directories():
    """CrÃƒÂ©e la structure de dossiers"""
    folders = ["clips", "output", "temp", "scripts"]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Ã°Å¸â€œÂ Dossier crÃƒÂ©ÃƒÂ©: {folder}/")

def main():
    print("Ã°Å¸Å¡â‚¬ Configuration du pipeline de clips viraux")
    print("[setup] skipping requirements install")
    setup_directories()
    install_ffmpeg()
    print("Ã¢Å“â€¦ Setup terminÃƒÂ©!")

if __name__ == "__main__":
    main()




