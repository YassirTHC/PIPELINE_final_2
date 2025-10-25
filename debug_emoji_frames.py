ï»¿# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Debug des frames emoji : vÃƒÂ©rifier si les emojis sont prÃƒÂ©sents avant l'export FFmpeg
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

def debug_emoji_frames():
    """Debug les frames avec emojis pour identifier oÃƒÂ¹ ils sont perdus"""
    
    print("Ã°Å¸â€Â DEBUG FRAMES EMOJI")
    print("=" * 30)
    
    # 1. Tester notre crÃƒÂ©ation de frame directement
    print("\n1Ã¯Â¸ÂÃ¢Æ’Â£ TEST CRÃƒâ€°ATION FRAME PIL:")
    try:
        sys.path.append('.')
        from tiktok_subtitles import create_text_with_emoji_frame
        
        # CrÃƒÂ©er une frame avec emoji
        frame = create_text_with_emoji_frame(
            "REALLY Ã°Å¸Å½Â¯ TEST", 
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
        print(f"Ã¢Å“â€¦ Frame crÃƒÂ©ÃƒÂ©e: {non_black} pixels visibles")
        print("Ã¢Å“â€¦ SauvÃƒÂ©: debug_emoji_frame.png")
        
        # VÃƒÂ©rifier si l'emoji est visible
        if non_black > 1000:  # Suffisamment de pixels pour du texte
            print("Ã¢Å“â€¦ La frame contient du contenu visible")
        else:
            print("Ã¢ÂÅ’ La frame semble vide")
            
    except Exception as e:
        print(f"Ã¢ÂÅ’ Erreur crÃƒÂ©ation frame: {e}")
        return False
    
    # 2. Tester les clips MoviePy
    print("\n2Ã¯Â¸ÂÃ¢Æ’Â£ TEST CLIP MOVIEPY:")
    try:
        from moviepy.editor import ImageClip, CompositeVideoClip, ColorClip
        
        # CrÃƒÂ©er un clip ÃƒÂ  partir de notre frame emoji
        emoji_clip = ImageClip(frame, duration=2.0)
        
        # VÃƒÂ©rifier le premier frame du clip
        test_frame = emoji_clip.get_frame(0)
        clip_pixels = np.sum(test_frame > 0)
        print(f"Ã¢Å“â€¦ Clip MoviePy: {clip_pixels} pixels visibles")
        
        if clip_pixels > 1000:
            print("Ã¢Å“â€¦ Le clip MoviePy preserve les emojis")
        else:
            print("Ã¢ÂÅ’ Le clip MoviePy perd les emojis")
        
        emoji_clip.close()
        
    except Exception as e:
        print(f"Ã¢ÂÅ’ Erreur clip MoviePy: {e}")
        return False
    
    # 3. Test export simple
    print("\n3Ã¯Â¸ÂÃ¢Æ’Â£ TEST EXPORT SIMPLE:")
    try:
        from moviepy.editor import ImageClip
        
        # CrÃƒÂ©er une vidÃƒÂ©o test trÃƒÂ¨s simple
        test_clip = ImageClip(frame, duration=1.0)
        test_output = "debug_emoji_export.mp4"
        
        print("Ã°Å¸Å½Â¬ Export en cours...")
        test_clip.write_videofile(
            test_output,
            fps=30,
            codec='libx264',
            verbose=False,
            logger=None
        )
        
        if Path(test_output).exists():
            size = Path(test_output).stat().st_size
            print(f"Ã¢Å“â€¦ Export rÃƒÂ©ussi: {size} bytes")
            
            # Recharger et vÃƒÂ©rifier
            from moviepy.editor import VideoFileClip
            reloaded = VideoFileClip(test_output)
            reloaded_frame = reloaded.get_frame(0)
            reloaded_pixels = np.sum(reloaded_frame > 0)
            print(f"Ã°Å¸â€œÅ  Pixels aprÃƒÂ¨s rechargement: {reloaded_pixels}")
            
            if reloaded_pixels > 1000:
                print("Ã¢Å“â€¦ L'export preserve les emojis")
            else:
                print("Ã¢ÂÅ’ L'export fait disparaÃƒÂ®tre les emojis")
            
            reloaded.close()
        else:
            print("Ã¢ÂÅ’ Export ÃƒÂ©chouÃƒÂ©")
            
        test_clip.close()
        
    except Exception as e:
        print(f"Ã¢ÂÅ’ Erreur export: {e}")
        return False
    
    return True

def check_actual_video():
    """VÃƒÂ©rifier la derniÃƒÂ¨re vidÃƒÂ©o gÃƒÂ©nÃƒÂ©rÃƒÂ©e"""
    
    print("\n4Ã¯Â¸ÂÃ¢Æ’Â£ VÃƒâ€°RIFICATION VIDÃƒâ€°O RÃƒâ€°ELLE:")
    
    # Chercher la derniÃƒÂ¨re vidÃƒÂ©o gÃƒÂ©nÃƒÂ©rÃƒÂ©e
    output_dir = Path("output/subtitled")
    if not output_dir.exists():
        print("Ã¢ÂÅ’ Dossier output/subtitled introuvable")
        return
    
    # Trouver le fichier le plus rÃƒÂ©cent
    video_files = list(output_dir.glob("*.mp4"))
    if not video_files:
        print("Ã¢ÂÅ’ Aucune vidÃƒÂ©o trouvÃƒÂ©e")
        return
    
    latest_video = max(video_files, key=lambda f: f.stat().st_mtime)
    print(f"Ã°Å¸â€œÂ¹ VidÃƒÂ©o analysÃƒÂ©e: {latest_video.name}")
    
    try:
        from moviepy.editor import VideoFileClip
        
        video = VideoFileClip(str(latest_video))
        
        # Prendre plusieurs frames ÃƒÂ  diffÃƒÂ©rents moments
        times = [1.0, video.duration/2, video.duration-1]
        
        for i, t in enumerate(times):
            if t < video.duration:
                frame = video.get_frame(t)
                pixels = np.sum(frame > 0)
                print(f"Ã°Å¸â€œÅ  Frame {t:.1f}s: {pixels} pixels")
                
                # Sauvegarder frame
                img = Image.fromarray(frame.astype('uint8'))
                img.save(f"debug_video_frame_{i}.png")
        
        video.close()
        print("Ã¢Å“â€¦ Frames extraites et sauvÃƒÂ©es")
        
    except Exception as e:
        print(f"Ã¢ÂÅ’ Erreur analyse vidÃƒÂ©o: {e}")

def main():
    """Debug principal"""
    
    print("Ã°Å¸â€Â DEBUG COMPLET EMOJI PIPELINE")
    print("=" * 50)
    
    success = debug_emoji_frames()
    check_actual_video()
    
    print("\nÃ°Å¸â€œâ€¹ RÃƒâ€°SUMÃƒâ€°:")
    if success:
        print("Ã¢Å“â€¦ MÃƒÂ©thode PIL fonctionne")
        print("Ã°Å¸â€Â VÃƒÂ©rifiez les images gÃƒÂ©nÃƒÂ©rÃƒÂ©es:")
        print("Ã¢â‚¬Â¢ debug_emoji_frame.png")
        print("Ã¢â‚¬Â¢ debug_emoji_export.mp4") 
        print("Ã¢â‚¬Â¢ debug_video_frame_*.png")
        
        print("\nÃ°Å¸â€™Â¡ PROCHAINES Ãƒâ€°TAPES:")
        print("Ã¢â‚¬Â¢ Ouvrir debug_emoji_frame.png")
        print("Ã¢â‚¬Â¢ Si emojis visibles Ã¢â€ â€™ problÃƒÂ¨me export")
        print("Ã¢â‚¬Â¢ Si emojis invisibles Ã¢â€ â€™ problÃƒÂ¨me crÃƒÂ©ation")
    else:
        print("Ã¢ÂÅ’ ProblÃƒÂ¨me dans la crÃƒÂ©ation des frames")

if __name__ == "__main__":
    main() 


