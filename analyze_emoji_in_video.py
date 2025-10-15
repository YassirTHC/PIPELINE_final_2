#!/usr/bin/env python3
"""
Analyser spécifiquement les emojis dans la vidéo finale
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

def analyze_video_frames():
    """Analyser frame par frame pour détecter les emojis"""
    
    print("🔍 ANALYSE DÉTAILLÉE EMOJIS VIDÉO")
    print("=" * 40)
    
    video_path = Path("output/subtitled/reframed_131_tiktok_subs.mp4")
    if not video_path.exists():
        print("❌ Vidéo introuvable")
        return False
    
    try:
        from moviepy import VideoFileClip
        
        video = VideoFileClip(str(video_path))
        print(f"📹 Vidéo: {video.size}, {video.duration:.1f}s")
        
        # Analyser des moments spécifiques où il devrait y avoir des emojis
        # D'après le log: 'REALLY 💯', 'BRAIN. ✨', 'AND 🔥'
        emoji_times = [10.0, 20.0, 30.0, 40.0, 50.0]  # Différents moments
        
        for i, t in enumerate(emoji_times):
            if t < video.duration:
                print(f"\n📊 ANALYSE FRAME {t:.1f}s:")
                
                frame = video.get_frame(t)
                print(f"   Taille frame: {frame.shape}")
                print(f"   Pixels non-noirs: {np.sum(frame > 0)}")
                
                # Sauvegarder la frame
                img = Image.fromarray(frame.astype('uint8'))
                frame_path = f"analyze_frame_{i}_{t:.0f}s.png"
                img.save(frame_path)
                print(f"   💾 Sauvé: {frame_path}")
                
                # Analyser la distribution des couleurs
                unique_colors = len(np.unique(frame.reshape(-1, frame.shape[-1]), axis=0))
                print(f"   🎨 Couleurs uniques: {unique_colors}")
                
                # Détecter des zones de texte (zones avec beaucoup de blanc)
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    white_pixels = np.sum(gray > 200)  # Pixels très clairs
                    print(f"   📝 Pixels blancs/texte: {white_pixels}")
                    
                    if white_pixels > 1000:
                        print("   ✅ Zone de texte détectée")
                    else:
                        print("   ❌ Peu de texte visible")
        
        video.close()
        return True
        
    except Exception as e:
        print(f"❌ Erreur analyse: {e}")
        return False

def test_emoji_rendering_comparison():
    """Comparer rendu emoji direct vs vidéo"""
    
    print("\n🔍 COMPARAISON RENDU EMOJI")
    print("=" * 35)
    
    # 1. Créer un emoji directement
    print("1️⃣ RENDU EMOJI DIRECT:")
    try:
        sys.path.append('.')
        from tiktok_subtitles import get_emoji_font
        
        # Créer une image avec emoji simple
        img = Image.new('RGB', (400, 200), 'black')
        draw = ImageDraw.Draw(img)
        
        font = get_emoji_font(60)
        test_text = "REALLY 💯 TEST"
        
        draw.text((20, 50), test_text, font=font, fill='white')
        img.save("direct_emoji_test.png")
        
        # Analyser
        arr = np.array(img)
        pixels = np.sum(arr > 0)
        print(f"   📊 Pixels visibles: {pixels}")
        print("   💾 Sauvé: direct_emoji_test.png")
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    # 2. Extraire du texte de la vraie vidéo
    print("\n2️⃣ EXTRACTION TEXTE VIDÉO:")
    try:
        from moviepy import VideoFileClip
        
        video_path = Path("output/subtitled/reframed_131_tiktok_subs.mp4")
        video = VideoFileClip(str(video_path))
        
        # Prendre une frame au milieu
        frame = video.get_frame(video.duration / 2)
        
        # Isoler les zones de texte (partie basse de l'écran)
        height = frame.shape[0]
        text_zone = frame[int(height * 0.7):, :]  # 30% bas de l'écran
        
        img = Image.fromarray(text_zone.astype('uint8'))
        img.save("video_text_zone.png")
        
        pixels = np.sum(text_zone > 0)
        print(f"   📊 Pixels zone texte: {pixels}")
        print("   💾 Sauvé: video_text_zone.png")
        
        video.close()
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")

def check_emoji_detection_log():
    """Analyser les logs d'assignation emoji"""
    
    print("\n📝 ANALYSE LOGS EMOJI")
    print("=" * 25)
    
    # Extraire des exemples du log utilisateur
    emoji_assignments = [
        "'things' → 'THINGS 💯'",
        "'really' → 'REALLY 💯'", 
        "'brain.' → 'BRAIN. ✨'",
        "'and' → 'AND 🔥'"
    ]
    
    print("✅ Assignations d'emojis détectées dans le log:")
    for assignment in emoji_assignments:
        print(f"   • {assignment}")
    
    print("\n🎯 CONCLUSION:")
    print("Les emojis SONT assignés et traités par le système.")
    print("Le problème pourrait être:")
    print("• Lecteur vidéo ne supporte pas les emojis")
    print("• Codec vidéo qui compresse les emojis")
    print("• Police non chargée à l'affichage")

def main():
    """Analyse principale"""
    
    print("🔍 ANALYSE COMPLÈTE EMOJIS DANS VIDÉO")
    print("=" * 50)
    
    success = analyze_video_frames()
    test_emoji_rendering_comparison()
    check_emoji_detection_log()
    
    print("\n🎯 DIAGNOSTIC FINAL:")
    print("=" * 25)
    
    if success:
        print("✅ Vidéo analysée avec succès")
        print("📁 Fichiers générés pour inspection:")
        print("• analyze_frame_*.png (frames vidéo)")
        print("• direct_emoji_test.png (emoji direct)")
        print("• video_text_zone.png (zone texte vidéo)")
        
        print("\n🔍 PROCHAINES ÉTAPES:")
        print("1. Ouvrir analyze_frame_*.png")
        print("2. Chercher visuellement les emojis")
        print("3. Comparer avec direct_emoji_test.png")
        print("4. Vérifier video_text_zone.png")
        
        print("\n💡 SI EMOJIS INVISIBLES:")
        print("• Problème probable: codec/compression")
        print("• Solution: changer paramètres export")
        
        print("\n💡 SI EMOJIS VISIBLES:")
        print("• Problème: lecteur vidéo utilisé")
        print("• Solution: utiliser VLC ou lecteur compatible")
    else:
        print("❌ Échec analyse - vérifier fichiers")

if __name__ == "__main__":
    main() 