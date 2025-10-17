#!/usr/bin/env python3
"""
Crée des animations simples pour le dossier animations_assets
"""

import sys
sys.path.append('.')
from pathlib import Path
from moviepy.editor import ColorClip, CompositeVideoClip
import numpy as np

def create_simple_glow_animation(filename: str, color: tuple = (255, 255, 0)):
    """Crée une animation glow simple"""
    print(f"🎬 Création animation: {filename}")
    
    # Créer un fond noir
    background = ColorClip(size=(400, 200), color=(0, 0, 0), duration=3)
    
    # Créer un cercle coloré avec effet de pulsation
    def create_circle_frame(t):
        # Créer une image avec un cercle qui pulse
        size = 400 + int(50 * np.sin(t * 2))  # Pulsation
        circle_clip = ColorClip(size=(size, size), color=color, duration=0.1)
        circle_clip = circle_clip.set_position('center')
        return circle_clip
    
    # Créer plusieurs frames pour l'animation
    frames = []
    for t in np.arange(0, 3, 0.1):
        frame = create_circle_frame(t)
        frames.append(frame)
    
    # Combiner les frames
    final_clip = CompositeVideoClip([background] + frames)
    
    # Sauvegarder
    output_path = Path("animations_assets") / filename
    output_path.parent.mkdir(exist_ok=True)
    final_clip.write_videofile(str(output_path), fps=10, codec='libx264')
    final_clip.close()
    
    print(f"✅ Animation créée: {filename}")

def create_all_simple_animations():
    """Crée toutes les animations simples"""
    print("🎬 Création des animations simples...")
    
    animations = [
        ("brain_glow.mp4", (255, 100, 100)),
        ("school_glow.mp4", (100, 255, 100)),
        ("papers_glow.mp4", (100, 100, 255)),
        ("calendar_glow.mp4", (255, 255, 100)),
        ("home_glow.mp4", (255, 100, 255)),
        ("digital_glow.mp4", (100, 255, 255)),
        ("heart_glow.mp4", (255, 100, 100)),
        ("medical_glow.mp4", (100, 255, 100)),
        ("family_glow.mp4", (255, 200, 100)),
        ("star_glow.mp4", (255, 255, 100)),
        ("fire_glow.mp4", (255, 100, 50)),
        ("trophy_glow.mp4", (255, 215, 0))
    ]
    
    for filename, color in animations:
        create_simple_glow_animation(filename, color)
    
    print("✅ Toutes les animations simples ont été créées !")

if __name__ == "__main__":
    create_all_simple_animations() 
