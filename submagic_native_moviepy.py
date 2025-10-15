#!/usr/bin/env python3
"""
SOLUTION DÉFINITIVE SUBMAGIC - TextClip NATIF MoviePy
Résout le problème de vidéo noire en utilisant UNIQUEMENT les TextClip natifs
"""

import re
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from moviepy import *

class SubmagicNativeConfig:
    """Configuration pour TextClip natifs"""
    def __init__(self):
        # Tailles de police
        self.font_base_size = 60
        self.font_keyword_size = 70
        self.font_emphasis_size = 80
        
        # Couleurs (format MoviePy)
        self.color_white = 'white'
        self.color_green = '#32FF32'      # Actions
        self.color_red = '#FF3232'        # Emphase
        self.color_yellow = '#FFFF32'     # Important
        self.color_orange = '#FFA500'     # Argent
        self.color_blue = '#3296FF'       # Social
        self.color_purple = '#9632FF'     # Tech
        self.color_light_gray = '#C8C8C8' # Neutre
        
        # Police (MoviePy)
        self.font_name = 'Arial-Bold'
        
        # Position
        self.bottom_margin = 0.15  # 15% du bas
        
        # Animation
        self.word_appear_duration = 0.4
        self.bounce_intensity = 0.3
        self.persistence_enabled = True

def classify_word_simple(word: str) -> str:
    """Classification simplifiée"""
    word_lower = word.lower().strip()
    
    # Actions (VERT)
    if word_lower in ['behavior', 'lift', 'running', 'run', 'exercise', 'move', 'work', 'play', 'win']:
        return 'action'
    
    # Emphase (ROUGE)
    elif word_lower in ['why', 'what', 'how', 'quit', 'stop', 'never', 'problem', 'wrong'] or word.isupper():
        return 'emphasis'
    
    # Argent/Résultats (ORANGE)
    elif word_lower in ['money', 'profit', 'rich', 'success', 'result', 'benefit', 'advantage']:
        return 'money'
    
    # Social/Émotions (BLEU)
    elif word_lower in ['people', 'love', 'feel', 'happy', 'friend', 'you', 'we', 'together']:
        return 'social'
    
    # Important/Temps (JAUNE)
    elif word_lower in ['time', 'now', 'today', 'every', 'always', 'first', 'about', 'when']:
        return 'important'
    
    # Tech (VIOLET)
    elif word_lower in ['ai', 'technology', 'digital', 'future', 'system', 'method']:
        return 'tech'
    
    # Neutre (GRIS CLAIR)
    else:
        return 'neutral'

def get_text_style_native(word: str, config: SubmagicNativeConfig) -> Dict:
    """Style pour TextClip natif"""
    word_type = classify_word_simple(word)
    
    if word_type == 'action':
        return {'color': config.color_green, 'fontsize': config.font_keyword_size}
    elif word_type == 'emphasis':
        return {'color': config.color_red, 'fontsize': config.font_emphasis_size}
    elif word_type == 'money':
        return {'color': config.color_orange, 'fontsize': config.font_keyword_size}
    elif word_type == 'social':
        return {'color': config.color_blue, 'fontsize': config.font_keyword_size}
    elif word_type == 'important':
        return {'color': config.color_yellow, 'fontsize': config.font_keyword_size}
    elif word_type == 'tech':
        return {'color': config.color_purple, 'fontsize': config.font_keyword_size}
    elif word_type == 'neutral':
        return {'color': config.color_light_gray, 'fontsize': config.font_base_size}
    else:
        return {'color': config.color_white, 'fontsize': config.font_base_size}

def create_bounce_effect(clip: TextClip, duration: float, config: SubmagicNativeConfig) -> TextClip:
    """Ajoute effet bounce au TextClip"""
    
    def bounce_resize(t):
        if t <= config.word_appear_duration:
            progress = t / config.word_appear_duration
            # Effet bounce avec easeOutBack
            scale = 0.5 + 0.5 * (1 + config.bounce_intensity * np.sin(progress * np.pi))
            return scale
        else:
            return 1.0
    
    # Appliquer l'effet de resize avec bounce
    bounced_clip = clip.resize(bounce_resize)
    
    # Effet de fade-in
    def fade_opacity(t):
        if t <= config.word_appear_duration:
            return t / config.word_appear_duration
        else:
            return 1.0
    
    return bounced_clip.set_opacity(fade_opacity)

def create_word_textclip(word: str, style: Dict, start_time: float, duration: float, 
                        video_size: tuple, config: SubmagicNativeConfig, 
                        with_animation: bool = True) -> TextClip:
    """Crée un TextClip natif pour un mot"""
    
    # Créer TextClip natif (AUCUNE conversion PIL/numpy)
    text_clip = TextClip(
        word,
        fontsize=style['fontsize'],
        color=style['color'],
        font=config.font_name,
        stroke_color='black',
        stroke_width=3
    ).set_duration(duration)
    
    # Position centrée en bas
    width, height = video_size
    bottom_y = height * (1 - config.bottom_margin)
    text_clip = text_clip.set_position(('center', bottom_y))
    
    # Animation bounce si demandée
    if with_animation:
        text_clip = create_bounce_effect(text_clip, duration, config)
    
    # Définir timing
    text_clip = text_clip.set_start(start_time)
    
    return text_clip

def parse_words_native(transcription_data: List[Dict]) -> List[Dict]:
    """Parse simple pour mots"""
    words_timeline = []
    
    for segment in transcription_data:
        text = segment.get('text', '').strip()
        seg_start = float(segment.get('start', 0))
        seg_end = float(segment.get('end', seg_start + 1))
        
        words = re.findall(r'\b\w+\b', text)
        if not words:
            continue
        
        word_duration = (seg_end - seg_start) / len(words)
        
        for i, word in enumerate(words):
            word_start = seg_start + i * word_duration
            
            words_timeline.append({
                'word': word,
                'start': word_start,
                'duration_base': word_duration
            })
    
    return words_timeline

def add_submagic_native(input_video_path: str, transcription_data: List[Dict], 
                       output_video_path: str) -> str:
    """
    SOLUTION DÉFINITIVE - TextClip natifs MoviePy SEULEMENT
    Garantit préservation de la vidéo source
    """
    config = SubmagicNativeConfig()
    
    print("🎬 SUBMAGIC NATIF - TextClip MoviePy pur...")
    
    # Charger vidéo source
    try:
        main_video = VideoFileClip(input_video_path)
        video_size = main_video.size
        video_duration = main_video.duration
        
        print(f"📊 Source: {video_size[0]}x{video_size[1]}, {video_duration:.1f}s")
        
    except Exception as e:
        print(f"❌ Erreur chargement: {e}")
        return input_video_path
    
    # Parser mots
    words_timeline = parse_words_native(transcription_data)
    print(f"📝 {len(words_timeline)} mots à traiter")
    
    if not words_timeline:
        main_video.close()
        return input_video_path
    
    # APPROCHE NATIVE : Créer TextClips purs
    all_text_clips = []
    
    print("✨ Création TextClips natifs (persistance + animations)...")
    
    for i, word_data in enumerate(words_timeline):
        current_word = word_data['word']
        word_start = word_data['start']
        
        # Calculer durée d'affichage
        if i < len(words_timeline) - 1:
            # Jusqu'au prochain mot
            display_duration = words_timeline[i + 1]['start'] - word_start
        else:
            # Jusqu'à la fin
            display_duration = min(video_duration - word_start, 3.0)
        
        if display_duration <= 0:
            continue
        
        # Style pour ce mot
        style = get_text_style_native(current_word, config)
        
        # CLIP PRINCIPAL : Mot actuel avec animation bounce
        main_clip = create_word_textclip(
            current_word, 
            style, 
            word_start, 
            display_duration,
            video_size, 
            config, 
            with_animation=True
        )
        
        all_text_clips.append(main_clip)
        
        # PERSISTANCE : Mots précédents (statiques)
        if config.persistence_enabled and i > 0:
            # Afficher aussi les mots précédents pendant ce mot
            for j in range(max(0, i-3), i):  # 3 mots précédents max
                prev_word_data = words_timeline[j]
                prev_style = get_text_style_native(prev_word_data['word'], config)
                
                # Position décalée pour éviter superposition
                prev_clip = TextClip(
                    prev_word_data['word'],
                    fontsize=int(prev_style['fontsize'] * 0.9),  # Légèrement plus petit
                    color=prev_style['color'],
                    font=config.font_name,
                    stroke_color='black',
                    stroke_width=2
                ).set_duration(display_duration)
                
                # Position latérale
                offset_x = (j - i) * 200  # Décalage horizontal
                bottom_y = video_size[1] * (1 - config.bottom_margin) + 30  # Légèrement plus haut
                prev_clip = prev_clip.set_position(('center', bottom_y))
                prev_clip = prev_clip.set_start(word_start).set_opacity(0.7)
                
                all_text_clips.append(prev_clip)
    
    # COMPOSITION FINALE : Approche native garantie
    print("🎨 Composition finale native...")
    
    try:
        if all_text_clips:
            # MÉTHODE NATIVE : Video de base + TextClips overlay
            final_video = CompositeVideoClip(
                [main_video] + all_text_clips,
                size=video_size
            )
            
            # Préserver audio
            final_video = final_video.set_audio(main_video.audio)
        else:
            final_video = main_video
        
        # Export avec paramètres optimaux
        output_path = Path(output_video_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Export: {output_path.name}")
        
        final_video.write_videofile(
            str(output_path),
            codec='libx264',
            audio_codec='aac',
            fps=30,
            preset='medium',
            ffmpeg_params=['-crf', '20', '-pix_fmt', 'yuv420p'],
            verbose=False,
            logger=None
        )
        
        print("✅ SUCCÈS TOTAL - Vidéo native générée !")
        return str(output_path)
        
    except Exception as e:
        print(f"❌ Erreur export: {e}")
        return input_video_path
        
    finally:
        # Nettoyage
        main_video.close()
        if 'final_video' in locals():
            final_video.close()
        for clip in all_text_clips:
            if hasattr(clip, 'close'):
                clip.close()

# Test rapide
def test_native_system():
    """Test système natif"""
    print("🧪 TEST SYSTÈME NATIF MoviePy")
    print("=" * 50)
    
    test_data = [
        {'text': 'BEHAVIOR LIFT', 'start': 0.0, 'end': 2.0}
    ]
    
    words = parse_words_native(test_data)
    print(f"📝 {len(words)} mots parsés")
    
    config = SubmagicNativeConfig()
    for word in words:
        style = get_text_style_native(word['word'], config)
        print(f"  '{word['word']}' → {style['color']} ({style['fontsize']}px)")
    
    print("✅ Système natif prêt!")

if __name__ == "__main__":
    test_native_system() 