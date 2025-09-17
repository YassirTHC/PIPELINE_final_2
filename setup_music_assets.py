#!/usr/bin/env python3
"""
🎵 CONFIGURATEUR MUSIQUE LIBRE DE DROITS
Configure automatiquement les dossiers et télécharge de la musique libre
"""

import os
import requests
from pathlib import Path
import zipfile
import shutil

def setup_music_folders():
    """Configure la structure des dossiers musique."""
    music_root = Path("assets/music")
    
    # Créer la structure
    folders = {
        "free": {
            "low": ["ambient", "calm", "gentle"],
            "medium": ["upbeat", "modern", "positive"],
            "high": ["energetic", "electronic", "motivational"]
        },
        "licensed": ["commercial", "youtube", "tiktok"],
        "temp": ["downloads", "processing"]
    }
    
    for main_folder, sub_folders in folders.items():
        main_path = music_root / main_folder
        main_path.mkdir(parents=True, exist_ok=True)
        
        if isinstance(sub_folders, dict):
            for intensity, categories in sub_folders.items():
                intensity_path = main_path / intensity
                intensity_path.mkdir(exist_ok=True)
                
                for category in categories:
                    category_path = intensity_path / category
                    category_path.mkdir(exist_ok=True)
        else:
            for sub_folder in sub_folders:
                sub_path = main_path / sub_folder
                sub_path.mkdir(exist_ok=True)
    
    print("✅ Structure des dossiers musique créée")

def download_sample_music():
    """Télécharge de la musique libre de droits d'exemple."""
    # URLs de musique libre de droits (exemples)
    sample_music = {
        "low/ambient": "https://example.com/free/ambient_soft.mp3",
        "medium/upbeat": "https://example.com/free/upbeat_inspirational.mp3",
        "high/energetic": "https://example.com/free/energetic_rock.mp3"
    }
    
    music_root = Path("assets/music/free")
    
    for path, url in sample_music.items():
        try:
            folder_path = music_root / path
            folder_path.mkdir(parents=True, exist_ok=True)
            
            filename = f"sample_{path.replace('/', '_')}.mp3"
            file_path = folder_path / filename
            
            print(f"📥 Téléchargement: {filename}")
            # response = requests.get(url)
            # file_path.write_bytes(response.content)
            
            # Créer un fichier placeholder pour l'exemple
            file_path.write_text(f"Placeholder pour {filename}\nURL: {url}")
            
        except Exception as e:
            print(f"❌ Erreur téléchargement {path}: {e}")

def create_music_config():
    """Crée le fichier de configuration musique."""
    config = {
        "music_settings": {
            "auto_add": True,
            "default_intensity": "medium",
            "volume_reduction": 0.2,
            "fade_in_duration": 1.0,
            "fade_out_duration": 1.0
        },
        "intensity_mapping": {
            "low": {
                "description": "Calme, réfléchi, méditation",
                "folders": ["ambient", "calm", "gentle"],
                "volume": 0.15
            },
            "medium": {
                "description": "Équilibré, professionnel, informatif",
                "folders": ["upbeat", "modern", "positive"],
                "volume": 0.2
            },
            "high": {
                "description": "Énergique, motivant, action",
                "folders": ["energetic", "electronic", "motivational"],
                "volume": 0.25
            }
        },
        "auto_detection": {
            "sentiment_threshold": 0.3,
            "content_keywords": {
                "low": ["calm", "meditation", "relaxation", "sleep"],
                "medium": ["business", "education", "information", "professional"],
                "high": ["motivation", "action", "energy", "success"]
            }
        }
    }
    
    config_path = Path("config/music_config.json")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        import json
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("✅ Configuration musique créée: config/music_config.json")

def main():
    """Configuration complète."""
    print("🎵 CONFIGURATION MUSIQUE LIBRE DE DROITS")
    print("=" * 50)
    
    setup_music_folders()
    download_sample_music()
    create_music_config()
    
    print("\n🎉 Configuration terminée !")
    print("\n📁 Dossiers créés:")
    print("   assets/music/free/ - Musique libre de droits")
    print("   assets/music/licensed/ - Musique sous licence")
    print("   config/music_config.json - Configuration")
    
    print("\n🚀 Utilisation:")
    print("   1. Ajoutez vos musiques dans assets/music/free/")
    print("   2. Organisez par intensité (low/medium/high)")
    print("   3. Le pipeline ajoutera automatiquement la musique")

if __name__ == "__main__":
    main() 