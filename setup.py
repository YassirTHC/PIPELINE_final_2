import subprocess
import sys
import os

def print("[setup] skipping requirements install"):
    """Installe les dÃ©pendances Python"""
    print("ðŸ“¦ Installation des dÃ©pendances Python...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def install_ffmpeg():
    """Guide pour installer FFmpeg"""
    print("""
    ðŸŽ¬ Installation FFmpeg requise:
    
    Windows:
    1. TÃ©lÃ©chargez FFmpeg depuis https://ffmpeg.org/download.html
    2. Ajoutez le dossier bin Ã  votre PATH

    macOS:
    brew install ffmpeg

    Linux:
    sudo apt update && sudo apt install ffmpeg
    """)

def setup_directories():
    """CrÃ©e la structure de dossiers"""
    folders = ["clips", "output", "temp", "scripts"]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"ðŸ“ Dossier crÃ©Ã©: {folder}/")

def main():
    print("ðŸš€ Configuration du pipeline de clips viraux")
    print("[setup] skipping requirements install")
    setup_directories()
    install_ffmpeg()
    print("âœ… Setup terminÃ©!")

if __name__ == "__main__":
    main()


