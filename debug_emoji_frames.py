#!/usr/bin/env python3
"""
Debug des frames emoji : vérifier si les emojis sont présents avant l'export FFmpeg
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

def debug_emoji_frames():
    """Debug les frames avec emojis pour identifier où ils sont perdus"""
    
    print("🔍 DEBUG FRAMES EMOJI")
    print("=" * 30)
    
    # 1. Tester notre création de frame directement
    print("\n1️⃣ TEST CRÉATION FRAME PIL:")
    try:
        sys.path.append('.')
        from tiktok_subtitles import create_text_with_emoji_frame
        
        # Créer une frame avec emoji
        frame = create_text_with_emoji_frame(
            "REALLY 🎯 TEST", 
            None,
            (255, 255, 255),  # blanc
            1.0, 1.0, 1.0,
            (720, 1280)
        )
        
        # Sauvegarder comme image
        img = Image.fromarray(frame.astype('uint8'))
        img.save("debug_emoji_frame.png")
        
        # Compter les pixels non-noirs
        non_black = np.sum(frame > 0)
        print(f"✅ Frame créée: {non_black} pixels visibles")
        print("✅ Sauvé: debug_emoji_frame.png")
        
        # Vérifier si l'emoji est visible
        if non_black > 1000:  # Suffisamment de pixels pour du texte
            print("✅ La frame contient du contenu visible")
        else:
            print("❌ La frame semble vide")
            
    except Exception as e:
        print(f"❌ Erreur création frame: {e}")
        return False
    
    # 2. Tester les clips MoviePy
    print("\n2️⃣ TEST CLIP MOVIEPY:")
    try:
        from moviepy.editor import ImageClip, CompositeVideoClip, ColorClip
        
        # Créer un clip à partir de notre frame emoji
        emoji_clip = ImageClip(frame, duration=2.0)
        
        # Vérifier le premier frame du clip
        test_frame = emoji_clip.get_frame(0)
        clip_pixels = np.sum(test_frame > 0)
        print(f"✅ Clip MoviePy: {clip_pixels} pixels visibles")
        
        if clip_pixels > 1000:
            print("✅ Le clip MoviePy preserve les emojis")
        else:
            print("❌ Le clip MoviePy perd les emojis")
        
        emoji_clip.close()
        
    except Exception as e:
        print(f"❌ Erreur clip MoviePy: {e}")
        return False
    
    # 3. Test export simple
    print("\n3️⃣ TEST EXPORT SIMPLE:")
    try:
        from moviepy.editor import ImageClip
        
        # Créer une vidéo test très simple
        test_clip = ImageClip(frame, duration=1.0)
        test_output = "debug_emoji_export.mp4"
        
        print("🎬 Export en cours...")
        test_clip.write_videofile(
            test_output,
            fps=30,
            codec='libx264',
            verbose=False,
            logger=None
        )
        
        if Path(test_output).exists():
            size = Path(test_output).stat().st_size
            print(f"✅ Export réussi: {size} bytes")
            
            # Recharger et vérifier
            from moviepy.editor import VideoFileClip
            reloaded = VideoFileClip(test_output)
            reloaded_frame = reloaded.get_frame(0)
            reloaded_pixels = np.sum(reloaded_frame > 0)
            print(f"📊 Pixels après rechargement: {reloaded_pixels}")
            
            if reloaded_pixels > 1000:
                print("✅ L'export preserve les emojis")
            else:
                print("❌ L'export fait disparaître les emojis")
            
            reloaded.close()
        else:
            print("❌ Export échoué")
            
        test_clip.close()
        
    except Exception as e:
        print(f"❌ Erreur export: {e}")
        return False
    
    return True

def check_actual_video():
    """Vérifier la dernière vidéo générée"""
    
    print("\n4️⃣ VÉRIFICATION VIDÉO RÉELLE:")
    
    # Chercher la dernière vidéo générée
    output_dir = Path("output/subtitled")
    if not output_dir.exists():
        print("❌ Dossier output/subtitled introuvable")
        return
    
    # Trouver le fichier le plus récent
    video_files = list(output_dir.glob("*.mp4"))
    if not video_files:
        print("❌ Aucune vidéo trouvée")
        return
    
    latest_video = max(video_files, key=lambda f: f.stat().st_mtime)
    print(f"📹 Vidéo analysée: {latest_video.name}")
    
    try:
        from moviepy.editor import VideoFileClip
        
        video = VideoFileClip(str(latest_video))
        
        # Prendre plusieurs frames à différents moments
        times = [1.0, video.duration/2, video.duration-1]
        
        for i, t in enumerate(times):
            if t < video.duration:
                frame = video.get_frame(t)
                pixels = np.sum(frame > 0)
                print(f"📊 Frame {t:.1f}s: {pixels} pixels")
                
                # Sauvegarder frame
                img = Image.fromarray(frame.astype('uint8'))
                img.save(f"debug_video_frame_{i}.png")
        
        video.close()
        print("✅ Frames extraites et sauvées")
        
    except Exception as e:
        print(f"❌ Erreur analyse vidéo: {e}")

def main():
    """Debug principal"""
    
    print("🔍 DEBUG COMPLET EMOJI PIPELINE")
    print("=" * 50)
    
    success = debug_emoji_frames()
    check_actual_video()
    
    print("\n📋 RÉSUMÉ:")
    if success:
        print("✅ Méthode PIL fonctionne")
        print("🔍 Vérifiez les images générées:")
        print("• debug_emoji_frame.png")
        print("• debug_emoji_export.mp4") 
        print("• debug_video_frame_*.png")
        
        print("\n💡 PROCHAINES ÉTAPES:")
        print("• Ouvrir debug_emoji_frame.png")
        print("• Si emojis visibles → problème export")
        print("• Si emojis invisibles → problème création")
    else:
        print("❌ Problème dans la création des frames")

if __name__ == "__main__":
    main() 
