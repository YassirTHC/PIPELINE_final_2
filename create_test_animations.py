#!/usr/bin/env python3
"""
Crée des animations de test pour le dossier animations_assets
"""

import sys
sys.path.append('.')
from pathlib import Path
from moviepy.editor import ColorClip, TextClip, CompositeVideoClip
import numpy as np

def create_glow_animation(text: str, filename: str, color: tuple = (255, 255, 0)):
    """Crée une animation glow avec du texte"""
    print(f"🎬 Création animation: {filename}")
    
    # Créer un fond noir
    background = ColorClip(size=(400, 200), color=(0, 0, 0), duration=3)
    
    # Créer le texte avec effet glow
    text_clip = TextClip(text, fontsize=40, color='white', size=(400, 200))
    text_clip = text_clip.set_duration(3).set_position('center')
    
    # Ajouter un effet de pulsation
    def pulse_effect(t):
        scale = 1 + 0.1 * np.sin(t * 4)  # Pulsation
        return scale
    
    text_clip = text_clip.resize(pulse_effect)
    
    # Combiner
    final_clip = CompositeVideoClip([background, text_clip])
    
    # Sauvegarder
    output_path = Path("animations_assets") / filename
    output_path.parent.mkdir(exist_ok=True)
    final_clip.write_videofile(str(output_path), fps=24, codec='libx264')
    final_clip.close()
    
    print(f"✅ Animation créée: {filename}")

def create_all_test_animations():
    """Crée toutes les animations de test"""
    print("🎬 Création des animations de test...")
    
    animations = [
        ("brain_glow.mp4", "🧠 BRAIN", (255, 100, 100)),
        ("school_glow.mp4", "🎓 SCHOOL", (100, 255, 100)),
        ("papers_glow.mp4", "📝 PAPERS", (100, 100, 255)),
        ("calendar_glow.mp4", "📅 CALENDAR", (255, 255, 100)),
        ("home_glow.mp4", "🏠 HOME", (255, 100, 255)),
        ("digital_glow.mp4", "💻 DIGITAL", (100, 255, 255)),
        ("heart_glow.mp4", "❤️ HEART", (255, 100, 100)),
        ("medical_glow.mp4", "🏥 MEDICAL", (100, 255, 100)),
        ("family_glow.mp4", "👨‍👩‍👧‍👦 FAMILY", (255, 200, 100)),
        ("star_glow.mp4", "⭐ STAR", (255, 255, 100)),
        ("fire_glow.mp4", "🔥 FIRE", (255, 100, 50)),
        ("trophy_glow.mp4", "🏆 TROPHY", (255, 215, 0))
    ]
    
    for filename, text, color in animations:
        create_glow_animation(text, filename, color)
    
    print("✅ Toutes les animations de test ont été créées !")

if __name__ == "__main__":
    create_all_test_animations() 
