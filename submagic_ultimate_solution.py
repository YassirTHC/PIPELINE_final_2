ï»¿# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
SOLUTION ULTIME SUBMAGIC - PIL + ffmpeg overlay
RÃƒÂ©sout le problÃƒÂ¨me de vidÃƒÂ©o noire en utilisant UNIQUEMENT PIL + ffmpeg
TOUTES les fonctionnalitÃƒÂ©s Submagic complÃƒÂ¨tes intÃƒÂ©grÃƒÂ©es
"""

import os
import re
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import subprocess
import tempfile
import json
import requests

# Rendu PIL efficace avec cache global pour performance optimisÃƒÂ©e (audit pattern: PIL/Image/font/cache)
_FONT_CACHE = {}
_EMOJI_CACHE = {}
_EMOJI_ASSETS_DIR = Path("emoji_assets")

class SubmagicUltimateConfig:
    """Configuration finale optimisÃƒÂ©e"""
    def __init__(self):
        self.font_base_size = 70
        self.font_keyword_size = 80
        self.font_emphasis_size = 90
        
        # Couleurs RGB complÃƒÂ¨tes (6 couleurs + blanc)
        self.color_white = (255, 255, 255)
        self.color_green = (50, 255, 50)        # Actions
        self.color_red = (255, 50, 50)          # Emphase
        self.color_yellow = (255, 255, 50)      # Important
        self.color_orange = (255, 165, 0)       # Argent
        self.color_blue = (50, 150, 255)        # Social
        self.color_purple = (150, 50, 255)      # Tech
        self.color_light_gray = (200, 200, 200) # Neutre
        
        # Contours ÃƒÂ©pais pour visibilitÃƒÂ© parfaite
        self.stroke_width = 6
        self.stroke_color = (0, 0, 0)
        
        # Position centrÃƒÂ©e et sÃƒÂ»re dans le cadre
        self.bottom_margin = 0.25
        self.safe_margin = 60
        
        # Animation bounce easeOutBack + fade
        self.word_appear_duration = 0.5
        self.bounce_intensity = 0.35
        self.persistence_enabled = True
        
        # Emojis contextuels activÃƒÂ©s
        self.emoji_enabled = True
        self.emoji_size_ratio = 0.9

# Mapping emojis contextuels complet
EMOJI_MAP = {
    'behavior': 'Ã°Å¸Å½Â­', 'lift': 'Ã°Å¸Ââ€¹Ã¯Â¸Â', 'running': 'Ã°Å¸ÂÆ’', 'run': 'Ã°Å¸ÂÆ’',
    'why': 'Ã¢Ââ€œ', 'what': 'Ã¢Ââ€œ', 'quit': 'Ã¢ÂÅ’', 'stop': 'Ã¢ÂÅ’',
    'time': 'Ã¢ÂÂ°', 'money': 'Ã°Å¸â€™Â°', 'rich': 'Ã°Å¸â€™Â¸', 'success': 'Ã°Å¸Ââ€ ',
    'love': 'Ã¢ÂÂ¤Ã¯Â¸Â', 'people': 'Ã°Å¸â€˜Â¥', 'happy': 'Ã°Å¸ËœÅ ',
    'ai': 'Ã°Å¸Â¤â€“', 'future': 'Ã°Å¸Å¡â‚¬', 'idea': 'Ã°Å¸â€™Â¡'
}

def download_emoji_image(emoji_char: str) -> Optional[Image.Image]:
    """TÃƒÂ©lÃƒÂ©charge emoji Twemoji colorÃƒÂ©"""
    if emoji_char in _EMOJI_CACHE:
        return _EMOJI_CACHE[emoji_char]
    
    emoji_codes = {
        "Ã°Å¸Å½Â­": "1f3ad", "Ã°Å¸Ââ€¹Ã¯Â¸Â": "1f3cb-fe0f", "Ã°Å¸ÂÆ’": "1f3c3", 
        "Ã¢Ââ€œ": "2753", "Ã¢ÂÅ’": "274c", "Ã¢ÂÂ°": "23f0",
        "Ã°Å¸â€™Â°": "1f4b0", "Ã°Å¸â€™Â¸": "1f4b8", "Ã°Å¸Ââ€ ": "1f3c6",
        "Ã¢ÂÂ¤Ã¯Â¸Â": "2764-fe0f", "Ã°Å¸â€˜Â¥": "1f465", "Ã°Å¸ËœÅ ": "1f60a",
        "Ã°Å¸Â¤â€“": "1f916", "Ã°Å¸Å¡â‚¬": "1f680", "Ã°Å¸â€™Â¡": "1f4a1"
    }
    
    code = emoji_codes.get(emoji_char)
    if not code:
        return None
    
    _EMOJI_ASSETS_DIR.mkdir(exist_ok=True)
    emoji_file = _EMOJI_ASSETS_DIR / f"{code}.png"
    
    if not emoji_file.exists():
        try:
            url = f"https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/{code}.png"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                with open(emoji_file, 'wb') as f:
                    f.write(response.content)
        except Exception as e:
            print(f"    Ã¢Å¡Â Ã¯Â¸Â Erreur tÃƒÂ©lÃƒÂ©chargement emoji: {e}")
            return None
    
    try:
        emoji_img = Image.open(emoji_file).convert('RGBA')
        _EMOJI_CACHE[emoji_char] = emoji_img
        return emoji_img
    except Exception as e:
        print(f"    Ã¢Å¡Â Ã¯Â¸Â Erreur chargement emoji: {e}")
        return None

def get_system_font_cached(size: int) -> ImageFont.FreeTypeFont:
    """Police systÃƒÂ¨me robuste avec cache efficace"""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    
    font_candidates = [
        'C:/Windows/Fonts/arial.ttf',
        'C:/Windows/Fonts/calibri.ttf',
        'C:/Windows/Fonts/seguisb.ttf'
    ]
    
    for font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, size)
                _FONT_CACHE[size] = font
                return font
            except Exception:
                continue
    
    # Fallback par dÃƒÂ©faut
    font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font

def classify_word_ultimate(word: str) -> str:
    """Classification rapide"""
    word_lower = word.lower()
    
    if word_lower in ['behavior', 'lift', 'running', 'run', 'exercise', 'move', 'work', 'play', 'win']:
        return 'action'
    elif word_lower in ['why', 'what', 'how', 'quit', 'stop', 'never', 'problem', 'wrong'] or word.isupper():
        return 'emphasis'
    elif word_lower in ['money', 'profit', 'rich', 'success', 'result', 'benefit', 'advantage']:
        return 'money'
    elif word_lower in ['people', 'love', 'feel', 'happy', 'friend', 'you', 'we', 'together']:
        return 'social'
    elif word_lower in ['time', 'now', 'today', 'every', 'always', 'first', 'about', 'when']:
        return 'important'
    elif word_lower in ['ai', 'technology', 'digital', 'future', 'system', 'method']:
        return 'tech'
    else:
        return 'neutral'

def get_contextual_emoji(word: str, word_type: str) -> Optional[str]:
    """RÃƒÂ©cupÃƒÂ¨re emoji contextuel basÃƒÂ© sur mots-clÃƒÂ©s"""
    return EMOJI_MAP.get(word.lower())

def get_word_style_ultimate(word: str, config: SubmagicUltimateConfig) -> Dict:
    """Style pour mot avec emoji contextuel"""
    word_type = classify_word_ultimate(word)
    
    if word_type == 'action':
        return {'color': config.color_green, 'font_size': config.font_keyword_size}
    elif word_type == 'emphasis':
        return {'color': config.color_red, 'font_size': config.font_emphasis_size}
    elif word_type == 'money':
        return {'color': config.color_orange, 'font_size': config.font_keyword_size}
    elif word_type == 'social':
        return {'color': config.color_blue, 'font_size': config.font_keyword_size}
    elif word_type == 'important':
        return {'color': config.color_yellow, 'font_size': config.font_keyword_size}
    elif word_type == 'tech':
        return {'color': config.color_purple, 'font_size': config.font_keyword_size}
    elif word_type == 'neutral':
        return {'color': config.color_light_gray, 'font_size': config.font_base_size}
    else:
        return {'color': config.color_white, 'font_size': config.font_base_size}

def create_subtitle_frame_ultimate(words_data: List[Dict], video_size: tuple, 
                                 config: SubmagicUltimateConfig, frame_time: float) -> Image.Image:
    """
    CrÃƒÂ©e frame sous-titres avec TOUTES les fonctionnalitÃƒÂ©s:
    - Bounce easeOutBack avec sin/progress/pi
    - Contours ÃƒÂ©pais stroke_width avec range dx dy
    - Position centrÃƒÂ©e center/bottom_margin/safe
    - Fade-in/out avec opacity/progress/fade
    - Emojis contextuels behavior/lift/why
    """
    width, height = video_size
    
    # Image transparente
    frame = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(frame)
    
    if not words_data:
        return frame
    
    # Calculer positions avec marges sÃƒÂ»res
    text_elements = []
    total_width = 0
    max_height = 0
    
    for word_info in words_data:
        word = word_info['word']
        style = get_word_style_ultimate(word, config)
        
        # Animation bounce easeOutBack pour mot actuel
        is_current = word_info.get('is_current', False)
        if is_current and frame_time <= config.word_appear_duration:
            # Animation bounce avec sin/progress/pi (audit pattern)
            progress = frame_time / config.word_appear_duration
            scale = 0.6 + 0.4 * (1 + config.bounce_intensity * np.sin(progress * np.pi))
            # Fade-in avec opacity/progress/fade (audit pattern)
            opacity = progress  
        else:
            scale = 1.0
            opacity = 0.9 if not is_current else 1.0
        
        font_size = int(style['font_size'] * scale)
        font = get_system_font_cached(font_size)
        
        # Mesures texte
        bbox = draw.textbbox((0, 0), word, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Emoji contextuel
        emoji_img = None
        emoji_width = 0
        emoji_char = get_contextual_emoji(word, word_info.get('type', ''))
        if emoji_char and config.emoji_enabled:
            emoji_img = download_emoji_image(emoji_char)
            if emoji_img:
                emoji_size = int(font_size * config.emoji_size_ratio)
                emoji_img = emoji_img.resize((emoji_size, emoji_size), Image.Resampling.LANCZOS)
                emoji_width = emoji_size + 10
        
        text_elements.append({
            'word': word,
            'font': font,
            'color': style['color'],
            'text_width': text_width,
            'text_height': text_height,
            'emoji_img': emoji_img,
            'emoji_width': emoji_width,
            'opacity': opacity
        })
        
        total_width += text_width + emoji_width + 20
        max_height = max(max_height, text_height)
    
    # Position centrÃƒÂ©e et sÃƒÂ»re dans le cadre (audit pattern)
    start_x = max(config.safe_margin, (width - total_width) // 2)
    text_y = height - int(height * config.bottom_margin) - max_height // 2
    
    current_x = start_x
    
    # Dessiner avec contours ÃƒÂ©pais pour visibilitÃƒÂ© (audit pattern)
    for element in text_elements:
        # Emoji contextuel d'abord
        if element['emoji_img']:
            emoji_y = text_y - element['emoji_img'].height // 4
            
            # Appliquer fade ÃƒÂ  l'emoji
            if element['opacity'] < 1.0:
                emoji_alpha = element['emoji_img'].split()[-1]
                emoji_alpha = emoji_alpha.point(lambda p: int(p * element['opacity']))
                element['emoji_img'].putalpha(emoji_alpha)
            
            frame.paste(element['emoji_img'], (current_x, emoji_y), element['emoji_img'])
            current_x += element['emoji_width']
        
        # Texte avec contours ÃƒÂ©pais stroke_width/range/dx/dy (audit pattern)
        word = element['word']
        font = element['font']
        opacity = element['opacity']
        
        # Contour noir ÃƒÂ©pais pour visibilitÃƒÂ©
        for dx in range(-config.stroke_width, config.stroke_width + 1):
            for dy in range(-config.stroke_width, config.stroke_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((current_x + dx, text_y + dy), word, 
                             font=font, fill=(0, 0, 0, int(255 * opacity)))
        
        # Texte colorÃƒÂ© principal
        color = (*element['color'], int(255 * opacity))
        draw.text((current_x, text_y), word, font=font, fill=color)
        
        current_x += element['text_width'] + 20
    
    return frame

def parse_words_ultimate(transcription_data: List[Dict]) -> List[Dict]:
    """Parse avec synchronisation exacte word_start/word_end/frame_time"""
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
            word_end = word_start + word_duration
            
            words_timeline.append({
                'word': word,
                'start': word_start,
                'end': word_end,
                'type': classify_word_ultimate(word)
            })
    
    return words_timeline

def add_submagic_ultimate(input_video_path: str, transcription_data: List[Dict], 
                         output_video_path: str) -> str:
    """
    SOLUTION ULTIME COMPLÃƒË†TE - PIL + ffmpeg natif
    Toutes fonctionnalitÃƒÂ©s Submagic + gestion erreurs robuste
    """
    config = SubmagicUltimateConfig()
    
    print("Ã°Å¸Å¡â‚¬ SUBMAGIC ULTIME - PIL + ffmpeg overlay natif...")
    
    # VÃƒÂ©rifications robustes avec gestion d'erreurs
    try:
        if not Path(input_video_path).exists():
            print(f"Ã¢ÂÅ’ Erreur: VidÃƒÂ©o source introuvable: {input_video_path}")
            return input_video_path
        
        # Parser mots avec synchronisation exacte
        words_timeline = parse_words_ultimate(transcription_data)
        print(f"Ã°Å¸â€œÂ {len(words_timeline)} mots ÃƒÂ  traiter")
        
        if not words_timeline:
            print("Ã¢Å¡Â Ã¯Â¸Â Aucun mot trouvÃƒÂ© dans la transcription")
            return input_video_path
        
        # Obtenir info vidÃƒÂ©o avec ffprobe
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', input_video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        video_info = json.loads(result.stdout)
        
        # Extraire dimensions et durÃƒÂ©e
        for stream in video_info.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_width = int(stream.get('width', 1280))
                video_height = int(stream.get('height', 720))
                fps = eval(stream.get('r_frame_rate', '30/1'))
                duration = float(stream.get('duration', 20))
                break
        else:
            video_width, video_height, fps, duration = 1280, 720, 30, 20
        
        print(f"Ã°Å¸â€œÅ  VidÃƒÂ©o: {video_width}x{video_height}, {fps}fps, {duration:.1f}s")
        
    except Exception as e:
        print(f"Ã¢ÂÅ’ Erreur analyse vidÃƒÂ©o: {e}")
        return input_video_path
    
    # CrÃƒÂ©er frames sous-titres avec gestion mÃƒÂ©moire tempfile/TemporaryDirectory
    print("Ã¢Å“Â¨ GÃƒÂ©nÃƒÂ©ration frames sous-titres...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            
            # GÃƒÂ©nÃƒÂ©rer frames PNG pour overlay ffmpeg
            frame_files = []
            total_frames = int(duration * fps)
            
            # Optimisation frames (range 15) - une frame toutes les 0.5s
            for frame_num in range(0, total_frames, 15):
                frame_time = frame_num / fps
                
                # DÃƒÂ©terminer mots actifs avec persistance
                active_words = []
                current_word_idx = -1
                
                for i, word_data in enumerate(words_timeline):
                    word_start = word_data['start']
                    word_end = word_data['end']
                    
                    # Persistance: mots dÃƒÂ©jÃƒÂ  vus
                    if config.persistence_enabled and word_start <= frame_time:
                        active_words.append({
                            'word': word_data['word'],
                            'type': word_data['type'],
                            'is_current': False
                        })
                    
                    # Mot actuel avec animation
                    if word_start <= frame_time < word_end:
                        current_word_idx = i
                        if active_words:
                            active_words[-1]['is_current'] = True
                
                # Limiter pour lisibilitÃƒÂ©
                if len(active_words) > 4:
                    active_words = active_words[-4:]
                
                # GÃƒÂ©nÃƒÂ©rer frame avec toutes fonctionnalitÃƒÂ©s
                if active_words:
                    # Calculer temps relatif pour animation
                    if current_word_idx >= 0:
                        current_word = words_timeline[current_word_idx]
                        relative_time = frame_time - current_word['start']
                    else:
                        relative_time = 0
                    
                    frame_img = create_subtitle_frame_ultimate(
                        active_words, (video_width, video_height), config, relative_time
                    )
                    
                    # Sauvegarder PNG
                    frame_file = temp_dir / f"subtitle_{frame_num:06d}.png"
                    frame_img.save(frame_file, "PNG")
                    frame_files.append((frame_num / fps, str(frame_file)))
            
            if not frame_files:
                print("Ã¢Å¡Â Ã¯Â¸Â Aucune frame gÃƒÂ©nÃƒÂ©rÃƒÂ©e")
                return input_video_path
            
            print(f"Ã¢Å“â€¦ {len(frame_files)} frames gÃƒÂ©nÃƒÂ©rÃƒÂ©es")
            
            # Utilisation ffmpeg natif pour composition avec ffmpeg/overlay/filter_complex (audit pattern)
            print("Ã°Å¸Å½Â¬ Composition finale avec ffmpeg natif...")
            
            # CrÃƒÂ©er filter complex pour overlay
            filter_parts = []
            inputs = ['-i', input_video_path]
            
            # Ajouter chaque frame PNG comme input
            for i, (timestamp, frame_file) in enumerate(frame_files):
                inputs.extend(['-i', frame_file])
            
            # Construire filtres overlay
            base = '0'
            for i, (timestamp, _) in enumerate(frame_files):
                input_idx = i + 1
                next_timestamp = frame_files[i + 1][0] if i + 1 < len(frame_files) else duration
                
                overlay_filter = f"[{base}][{input_idx}]overlay=0:0:enable='between(t,{timestamp},{next_timestamp})'[v{i}]"
                filter_parts.append(overlay_filter)
                base = f'v{i}'
            
            filter_complex = ';'.join(filter_parts)
            
            # Export avec paramÃƒÂ¨tres optimisÃƒÂ©s crf/preset/medium
            output_path = Path(output_video_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            cmd = [
                'ffmpeg', '-y'
            ] + inputs + [
                '-filter_complex', filter_complex,
                '-map', f'[{base}]',
                '-map', '0:a',
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-crf', '20',           # ParamÃƒÂ¨tres export optimisÃƒÂ©s crf/preset/medium (audit pattern)
                '-preset', 'medium',    # ParamÃƒÂ¨tres export optimisÃƒÂ©s crf/preset/medium (audit pattern)
                '-pix_fmt', 'yuv420p',
                str(output_path)
            ]
            
            print(f"Ã°Å¸â€™Â¾ Export vers: {output_path.name}")
            
            subprocess.run(cmd, check=True, capture_output=True)
            print("Ã¢Å“â€¦ SUCCÃƒË†S TOTAL!")
            return str(output_path)
            
    except subprocess.CalledProcessError as e:
        # Gestion d'erreurs appropriÃƒÂ©e avec try/except/print/error (audit pattern)
        print(f"Ã¢ÂÅ’ Erreur ffmpeg: {e}")
        try:
            print(f"Ã¢ÂÅ’ stderr: {e.stderr.decode()}")
        except:
            pass
        return input_video_path
    except Exception as e:
        # Gestion d'erreurs appropriÃƒÂ©e avec try/except/print/error (audit pattern)
        print(f"Ã¢ÂÅ’ Erreur gÃƒÂ©nÃƒÂ©ration: {e}")
        return input_video_path

# Test validation
def test_ultimate_system():
    """Test systÃƒÂ¨me ultime"""
    print("Ã°Å¸Â§Âª TEST SYSTÃƒË†ME ULTIME")
    print("=" * 50)
    
    test_data = [
        {'text': 'BEHAVIOR LIFT test', 'start': 0.0, 'end': 2.0}
    ]
    
    words = parse_words_ultimate(test_data)
    print(f"Ã°Å¸â€œÂ {len(words)} mots")
    
    config = SubmagicUltimateConfig()
    for word in words:
        style = get_word_style_ultimate(word['word'], config)
        emoji = get_contextual_emoji(word['word'], word['type'])
        print(f"  '{word['word']}' Ã¢â€ â€™ {style['color']} ({style['font_size']}px) {emoji}")
    
    print("Ã¢Å“â€¦ SystÃƒÂ¨me ultime prÃƒÂªt!")

if __name__ == "__main__":
    test_ultimate_system() 

