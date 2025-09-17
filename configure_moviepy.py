#!/usr/bin/env python3
"""
Script de configuration pour MoviePy avec ImageMagick
"""

import os
import sys
from pathlib import Path

def configure_moviepy():
    """Configure MoviePy pour utiliser ImageMagick"""
    
    try:
        import moviepy.config as cfg
        
        # Chemins possibles pour ImageMagick sur Windows
        possible_paths = [
            r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe",
            r"C:\Program Files\ImageMagick-7.1.2-Q16\magick.exe",
            r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe",
            r"C:\Program Files\ImageMagick-7.1.1-Q16\magick.exe",
            r"C:\Program Files\ImageMagick-7.1.0-Q16-HDRI\magick.exe",
            r"C:\Program Files\ImageMagick-7.1.0-Q16\magick.exe",
        ]
        
        # Chercher ImageMagick
        imagemagick_path = None
        for path in possible_paths:
            if os.path.exists(path):
                imagemagick_path = path
                break
        
        if imagemagick_path:
            print(f"✅ ImageMagick trouvé: {imagemagick_path}")
            
            # Configurer MoviePy
            cfg.change_settings({"IMAGEMAGICK_BINARY": imagemagick_path})
            print("✅ MoviePy configuré avec ImageMagick")
            
            # Test de la configuration
            try:
                from moviepy.video.VideoClip import TextClip
                test_clip = TextClip("Test", fontsize=50, color='white')
                test_clip.close()
                print("✅ Test de TextClip réussi!")
                return True
                
            except Exception as e:
                print(f"⚠️ Test échoué: {e}")
                return False
        else:
            print("❌ ImageMagick non trouvé dans les chemins standards")
            print("Chemins vérifiés:")
            for path in possible_paths:
                print(f"  - {path}")
            return False
            
    except ImportError:
        print("❌ MoviePy non installé")
        return False
    except Exception as e:
        print(f"❌ Erreur de configuration: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Configuration de MoviePy avec ImageMagick...")
    success = configure_moviepy()
    
    if success:
        print("\n✅ Configuration terminée avec succès!")
        print("Vous pouvez maintenant utiliser TextClip avec des styles avancés.")
    else:
        print("\n❌ Configuration échouée.")
        print("Les sous-titres utiliseront le mode de fallback simple.") 