ï»¿# -*- coding: utf-8 -*-
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
            print(f"Ã¢Å“â€¦ ImageMagick trouvÃƒÂ©: {imagemagick_path}")
            
            # Configurer MoviePy
            cfg.change_settings({"IMAGEMAGICK_BINARY": imagemagick_path})
            print("Ã¢Å“â€¦ MoviePy configurÃƒÂ© avec ImageMagick")
            
            # Test de la configuration
            try:
                from moviepy.video.VideoClip import TextClip
                test_clip = TextClip("Test", fontsize=50, color='white')
                test_clip.close()
                print("Ã¢Å“â€¦ Test de TextClip rÃƒÂ©ussi!")
                return True
                
            except Exception as e:
                print(f"Ã¢Å¡Â Ã¯Â¸Â Test ÃƒÂ©chouÃƒÂ©: {e}")
                return False
        else:
            print("Ã¢ÂÅ’ ImageMagick non trouvÃƒÂ© dans les chemins standards")
            print("Chemins vÃƒÂ©rifiÃƒÂ©s:")
            for path in possible_paths:
                print(f"  - {path}")
            return False
            
    except ImportError:
        print("Ã¢ÂÅ’ MoviePy non installÃƒÂ©")
        return False
    except Exception as e:
        print(f"Ã¢ÂÅ’ Erreur de configuration: {e}")
        return False

if __name__ == "__main__":
    print("Ã°Å¸â€Â§ Configuration de MoviePy avec ImageMagick...")
    success = configure_moviepy()
    
    if success:
        print("\nÃ¢Å“â€¦ Configuration terminÃƒÂ©e avec succÃƒÂ¨s!")
        print("Vous pouvez maintenant utiliser TextClip avec des styles avancÃƒÂ©s.")
    else:
        print("\nÃ¢ÂÅ’ Configuration ÃƒÂ©chouÃƒÂ©e.")
        print("Les sous-titres utiliseront le mode de fallback simple.") 

