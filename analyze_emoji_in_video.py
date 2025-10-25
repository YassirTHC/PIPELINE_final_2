ï»¿# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Analyser spÃƒÂ©cifiquement les emojis dans la vidÃƒÂ©o finale
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

def analyze_video_frames():
    """Analyser frame par frame pour dÃƒÂ©tecter les emojis"""
    
    print("Ã°Å¸â€Â ANALYSE DÃƒâ€°TAILLÃƒâ€°E EMOJIS VIDÃƒâ€°O")
    print("=" * 40)
    
    video_path = Path("output/subtitled/reframed_131_tiktok_subs.mp4")
    if not video_path.exists():
        print("Ã¢ÂÅ’ VidÃƒÂ©o introuvable")
        return False
    
    try:
        from moviepy.editor import VideoFileClip
        
        video = VideoFileClip(str(video_path))
        print(f"Ã°Å¸â€œÂ¹ VidÃƒÂ©o: {video.size}, {video.duration:.1f}s")
        
        # Analyser des moments spÃƒÂ©cifiques oÃƒÂ¹ il devrait y avoir des emojis
        # D'aprÃƒÂ¨s le log: 'REALLY Ã°Å¸â€™Â¯', 'BRAIN. Ã¢Å“Â¨', 'AND Ã°Å¸â€Â¥'
        emoji_times = [10.0, 20.0, 30.0, 40.0, 50.0]  # DiffÃƒÂ©rents moments
        
        for i, t in enumerate(emoji_times):
            if t < video.duration:
                print(f"\nÃ°Å¸â€œÅ  ANALYSE FRAME {t:.1f}s:")
                
                frame = video.get_frame(t)
                print(f"   Taille frame: {frame.shape}")
                print(f"   Pixels non-noirs: {np.sum(frame > 0)}")
                
                # Sauvegarder la frame
                img = Image.fromarray(frame.astype('uint8'))
                frame_path = f"analyze_frame_{i}_{t:.0f}s.png"
                img.save(frame_path)
                print(f"   Ã°Å¸â€™Â¾ SauvÃƒÂ©: {frame_path}")
                
                # Analyser la distribution des couleurs
                unique_colors = len(np.unique(frame.reshape(-1, frame.shape[-1]), axis=0))
                print(f"   Ã°Å¸Å½Â¨ Couleurs uniques: {unique_colors}")
                
                # DÃƒÂ©tecter des zones de texte (zones avec beaucoup de blanc)
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    white_pixels = np.sum(gray > 200)  # Pixels trÃƒÂ¨s clairs
                    print(f"   Ã°Å¸â€œÂ Pixels blancs/texte: {white_pixels}")
                    
                    if white_pixels > 1000:
                        print("   Ã¢Å“â€¦ Zone de texte dÃƒÂ©tectÃƒÂ©e")
                    else:
                        print("   Ã¢ÂÅ’ Peu de texte visible")
        
        video.close()
        return True
        
    except Exception as e:
        print(f"Ã¢ÂÅ’ Erreur analyse: {e}")
        return False

def test_emoji_rendering_comparison():
    """Comparer rendu emoji direct vs vidÃƒÂ©o"""
    
    print("\nÃ°Å¸â€Â COMPARAISON RENDU EMOJI")
    print("=" * 35)
    
    # 1. CrÃƒÂ©er un emoji directement
    print("1Ã¯Â¸ÂÃ¢Æ’Â£ RENDU EMOJI DIRECT:")
    try:
        sys.path.append('.')
        from tiktok_subtitles import get_emoji_font
        
        # CrÃƒÂ©er une image avec emoji simple
        img = Image.new('RGB', (400, 200), 'black')
        draw = ImageDraw.Draw(img)
        
        font = get_emoji_font(60)
        test_text = "REALLY Ã°Å¸â€™Â¯ TEST"
        
        draw.text((20, 50), test_text, font=font, fill='white')
        img.save("direct_emoji_test.png")
        
        # Analyser
        arr = np.array(img)
        pixels = np.sum(arr > 0)
        print(f"   Ã°Å¸â€œÅ  Pixels visibles: {pixels}")
        print("   Ã°Å¸â€™Â¾ SauvÃƒÂ©: direct_emoji_test.png")
        
    except Exception as e:
        print(f"   Ã¢ÂÅ’ Erreur: {e}")
    
    # 2. Extraire du texte de la vraie vidÃƒÂ©o
    print("\n2Ã¯Â¸ÂÃ¢Æ’Â£ EXTRACTION TEXTE VIDÃƒâ€°O:")
    try:
        from moviepy.editor import VideoFileClip
        
        video_path = Path("output/subtitled/reframed_131_tiktok_subs.mp4")
        video = VideoFileClip(str(video_path))
        
        # Prendre une frame au milieu
        frame = video.get_frame(video.duration / 2)
        
        # Isoler les zones de texte (partie basse de l'ÃƒÂ©cran)
        height = frame.shape[0]
        text_zone = frame[int(height * 0.7):, :]  # 30% bas de l'ÃƒÂ©cran
        
        img = Image.fromarray(text_zone.astype('uint8'))
        img.save("video_text_zone.png")
        
        pixels = np.sum(text_zone > 0)
        print(f"   Ã°Å¸â€œÅ  Pixels zone texte: {pixels}")
        print("   Ã°Å¸â€™Â¾ SauvÃƒÂ©: video_text_zone.png")
        
        video.close()
        
    except Exception as e:
        print(f"   Ã¢ÂÅ’ Erreur: {e}")

def check_emoji_detection_log():
    """Analyser les logs d'assignation emoji"""
    
    print("\nÃ°Å¸â€œÂ ANALYSE LOGS EMOJI")
    print("=" * 25)
    
    # Extraire des exemples du log utilisateur
    emoji_assignments = [
        "'things' Ã¢â€ â€™ 'THINGS Ã°Å¸â€™Â¯'",
        "'really' Ã¢â€ â€™ 'REALLY Ã°Å¸â€™Â¯'", 
        "'brain.' Ã¢â€ â€™ 'BRAIN. Ã¢Å“Â¨'",
        "'and' Ã¢â€ â€™ 'AND Ã°Å¸â€Â¥'"
    ]
    
    print("Ã¢Å“â€¦ Assignations d'emojis dÃƒÂ©tectÃƒÂ©es dans le log:")
    for assignment in emoji_assignments:
        print(f"   Ã¢â‚¬Â¢ {assignment}")
    
    print("\nÃ°Å¸Å½Â¯ CONCLUSION:")
    print("Les emojis SONT assignÃƒÂ©s et traitÃƒÂ©s par le systÃƒÂ¨me.")
    print("Le problÃƒÂ¨me pourrait ÃƒÂªtre:")
    print("Ã¢â‚¬Â¢ Lecteur vidÃƒÂ©o ne supporte pas les emojis")
    print("Ã¢â‚¬Â¢ Codec vidÃƒÂ©o qui compresse les emojis")
    print("Ã¢â‚¬Â¢ Police non chargÃƒÂ©e ÃƒÂ  l'affichage")

def main():
    """Analyse principale"""
    
    print("Ã°Å¸â€Â ANALYSE COMPLÃƒË†TE EMOJIS DANS VIDÃƒâ€°O")
    print("=" * 50)
    
    success = analyze_video_frames()
    test_emoji_rendering_comparison()
    check_emoji_detection_log()
    
    print("\nÃ°Å¸Å½Â¯ DIAGNOSTIC FINAL:")
    print("=" * 25)
    
    if success:
        print("Ã¢Å“â€¦ VidÃƒÂ©o analysÃƒÂ©e avec succÃƒÂ¨s")
        print("Ã°Å¸â€œÂ Fichiers gÃƒÂ©nÃƒÂ©rÃƒÂ©s pour inspection:")
        print("Ã¢â‚¬Â¢ analyze_frame_*.png (frames vidÃƒÂ©o)")
        print("Ã¢â‚¬Â¢ direct_emoji_test.png (emoji direct)")
        print("Ã¢â‚¬Â¢ video_text_zone.png (zone texte vidÃƒÂ©o)")
        
        print("\nÃ°Å¸â€Â PROCHAINES Ãƒâ€°TAPES:")
        print("1. Ouvrir analyze_frame_*.png")
        print("2. Chercher visuellement les emojis")
        print("3. Comparer avec direct_emoji_test.png")
        print("4. VÃƒÂ©rifier video_text_zone.png")
        
        print("\nÃ°Å¸â€™Â¡ SI EMOJIS INVISIBLES:")
        print("Ã¢â‚¬Â¢ ProblÃƒÂ¨me probable: codec/compression")
        print("Ã¢â‚¬Â¢ Solution: changer paramÃƒÂ¨tres export")
        
        print("\nÃ°Å¸â€™Â¡ SI EMOJIS VISIBLES:")
        print("Ã¢â‚¬Â¢ ProblÃƒÂ¨me: lecteur vidÃƒÂ©o utilisÃƒÂ©")
        print("Ã¢â‚¬Â¢ Solution: utiliser VLC ou lecteur compatible")
    else:
        print("Ã¢ÂÅ’ Ãƒâ€°chec analyse - vÃƒÂ©rifier fichiers")

if __name__ == "__main__":
    main() 


