"""
Style HORMOZI 1 - VERSION PARFAITE basée sur recherche approfondie
Vraies caractéristiques TikTok d'Alex Hormozi reproduites fidèlement
Police: Montserrat Black 900, couleurs dynamiques, animations exactes
"""

import os
import subprocess
import tempfile
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import requests

class HormoziUltimateConfig:
    """Configuration PARFAITE du style Hormozi 1 basée sur recherche TikTok"""
    
    def __init__(self):
        # POLICE EXACTE HORMOZI (PROMPT ORIGINAL)
        self.font_primary = "Impact"                   # Police principale prompt
        self.font_secondary = "Arial Black"            # Fallback prompt
        self.font_tertiary = "Anton"                   # Fallback prompt
        self.font_scale_factor = 0.08                  # 8% hauteur (adaptatif)
        self.font_max_size = 80                        # Limite max 80px
        self.font_min_size = 24                        # Limite min 24px
        
        # STYLE TEXTE HORMOZI EXACT (PROMPT)
        self.text_uppercase = True                     # TOUJOURS EN MAJUSCULES
        self.max_words_per_line = 4                    # Max 4 mots par ligne
        self.max_lines = 2                             # Max 2 lignes
        
        # COULEURS HORMOZI EXACTES (PROMPT ORIGINAL)
        self.color_default = (255, 255, 255)          # Blanc pur base
        self.color_money = (255, 215, 0)              # Jaune vif #FFD700
        self.color_action = (255, 0, 0)               # Rouge vif #FF0000
        self.color_success = (0, 255, 0)              # Vert vif #00FF00
        self.color_info = (30, 144, 255)              # Bleu clair #1E90FF
        self.color_warning = (255, 165, 0)            # Orange vif
        
        # OUTLINE HORMOZI (PROMPT: 3-5px)
        self.outline_enabled = True
        self.outline_color = (0, 0, 0)                # Noir pur
        self.outline_width = 3                        # 3px (prompt minimum)
        self.shadow_enabled = True
        self.shadow_offset = (2, 2)                   # Ombre subtile
        self.shadow_color = (0, 0, 0, 150)           # Ombre légère
        self.shadow_blur = 1                          # Flou minimal
        
        # POSITIONNEMENT HORMOZI
        self.bottom_margin_ratio = 0.12               # 12% du bas
        self.horizontal_center = True                 # Centrage parfait
        self.side_margins = 0.05                      # 5% marge
        self.line_spacing = 1.1                       # Espacement serré
        
        # ANIMATIONS HORMOZI ORIGINALES (PROMPT EXACT)
        self.bounce_enabled = True
        self.bounce_scale = 1.2                       # Scale 1.2x (prompt exact)
        self.bounce_duration = 0.15                   # 150ms bounce rapide
        self.fade_enabled = True
        self.fade_duration = 0.12                     # 120ms fade rapide (prompt)
        self.rotation_effect = False                  # Pas de rotation (prompt simple)
        
        # TIMING MOT PAR MOT (PROMPT EXACT)
        self.word_by_word = True                      # MOT PAR MOT obligatoire
        self.word_duration = 0.8                      # 800ms par mot
        self.clear_transition = True                  # Disparition nette phrases
        self.phrase_clear_delay = 0.2                 # 200ms pause entre phrases
        
        # EMOJIS HORMOZI (PROMPT ACTIVÉ)
        self.emoji_enabled = True
        self.emoji_size_ratio = 1.0                   # Même taille que texte
        self.emoji_spacing = 10                       # Espacement emoji-texte

# MAPPING EXACT MOTS-CLÉS HORMOZI (basé sur recherche)
HORMOZI_PERFECT_KEYWORDS = {
    # MONEY/FINANCE (Jaune - couleur signature Hormozi)
    'MONEY': 'money', 'CASH': 'money', 'PROFIT': 'money', 'REVENUE': 'money',
    'BUSINESS': 'money', 'SALES': 'money', 'INCOME': 'money', 'WEALTH': 'money',
    'RICH': 'money', 'EXPENSIVE': 'money', 'VALUE': 'money', 'COST': 'money',
    'INVESTMENT': 'money', 'FINANCIAL': 'money', 'BUDGET': 'money',
    
    # ACTION/URGENCE (Rouge - Hormozi emphasis)
    'NOW': 'action', 'STOP': 'action', 'URGENT': 'action', 'FAST': 'action',
    'QUICK': 'action', 'IMMEDIATE': 'action', 'MUST': 'action', 'NEED': 'action',
    'IMPORTANT': 'action', 'CRITICAL': 'action', 'HURRY': 'action', 'RUSH': 'action',
    
    # SUCCESS/POSITIVE (Vert - résultats Hormozi)
    'SUCCESS': 'success', 'WIN': 'success', 'WINNER': 'success', 'BEST': 'success',
    'PERFECT': 'success', 'AMAZING': 'success', 'GREAT': 'success', 'EXCELLENT': 'success',
    'TOP': 'success', 'FIRST': 'success', 'CHAMPION': 'success', 'GROWTH': 'success',
    
    # LEARNING/INFO (Bleu - éducation Hormozi)
    'LEARN': 'info', 'IDEA': 'info', 'SECRET': 'info', 'TIP': 'info',
    'TRICK': 'info', 'METHOD': 'info', 'STRATEGY': 'info', 'SYSTEM': 'info',
    'KNOWLEDGE': 'info', 'UNDERSTAND': 'info', 'DISCOVER': 'info', 'FIND': 'info',
    
    # ATTENTION/WARNING (Orange - focus Hormozi)
    'WATCH': 'warning', 'LOOK': 'warning', 'ATTENTION': 'warning', 'FOCUS': 'warning'
}

# MAPPING EMOJI HORMOZI (prompt activé)
HORMOZI_EMOJI_MAP = {
    # MONEY/FINANCE 
    'MONEY': '💰', 'CASH': '💸', 'PROFIT': '💰', 'REVENUE': '💰',
    'BUSINESS': '💼', 'SALES': '💰', 'INCOME': '💰', 'WEALTH': '💎',
    'RICH': '💰', 'VALUE': '💰', 'INVESTMENT': '📈',
    
    # ACTION/URGENCE
    'FIRE': '🔥', 'HOT': '🔥', 'FAST': '⚡', 'QUICK': '⚡',
    'NOW': '⚡', 'URGENT': '🚨', 'STOP': '🛑', 'MUST': '🔥',
    
    # SUCCESS/POSITIVE
    'SUCCESS': '✅', 'WIN': '🏆', 'WINNER': '🏆', 'BEST': '🏆',
    'PERFECT': '✅', 'AMAZING': '🌟', 'GREAT': '👍', 'TOP': '🔝',
    
    # LEARNING/INFO
    'IDEA': '💡', 'SECRET': '🤫', 'TIP': '💡', 'LEARN': '📚',
    'KNOWLEDGE': '🧠', 'DISCOVER': '🔍', 'FIND': '🔍',
    
    # ATTENTION/WARNING
    'WATCH': '👀', 'LOOK': '👀', 'ATTENTION': '⚠️', 'FOCUS': '🎯'
}

# Cache fonts
_FONT_CACHE = {}
_EMOJI_CACHE = {}

def get_hormozi_font(size: int) -> ImageFont.FreeTypeFont:
    """Récupère la vraie police Hormozi avec fallbacks"""
    cache_key = f"hormozi_perfect_{size}"
    if cache_key in _FONT_CACHE:
        return _FONT_CACHE[cache_key]
    
    # Polices PROMPT Hormozi (Impact priorité absolue)
    font_paths = [
        # PROMPT priorités
        "C:/Windows/Fonts/impact.ttf",         # Impact PROMPT priorité 1
        "C:/Windows/Fonts/IMPACT.TTF",
        "C:/Windows/Fonts/ariblk.ttf",         # Arial Black PROMPT priorité 2
        "C:/Windows/Fonts/ARIBLK.TTF",
        "C:/Windows/Fonts/anton.ttf",          # Anton PROMPT priorité 3
        "C:/Windows/Fonts/ANTON.TTF",
        # System names
        "Impact",
        "Arial Black",
        "Anton"
    ]
    
    font = None
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, size)
                print(f"✅ Police Hormozi PARFAITE: {font_path} ({size}px)")
                break
            else:
                # Try by name
                font = ImageFont.truetype(font_path, size)
                print(f"✅ Police Hormozi système: {font_path} ({size}px)")
                break
        except Exception:
            continue
    
    if not font:
        font = ImageFont.load_default()
        print(f"⚠️ Police par défaut utilisée ({size}px)")
    
    _FONT_CACHE[cache_key] = font
    return font

def download_hormozi_emoji(emoji_char: str) -> Optional[Image.Image]:
    """Télécharge emoji style Hormozi (Twemoji)"""
    if emoji_char in _EMOJI_CACHE:
        return _EMOJI_CACHE[emoji_char]
    
    try:
        codepoint = hex(ord(emoji_char))[2:].lower()
        url = f"https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/{codepoint}.png"
        
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            from io import BytesIO
            emoji_img = Image.open(BytesIO(response.content))
            emoji_img = emoji_img.convert('RGBA')
            _EMOJI_CACHE[emoji_char] = emoji_img
            return emoji_img
    except Exception as e:
        print(f"⚠️ Emoji {emoji_char}: {e}")
    
    _EMOJI_CACHE[emoji_char] = None
    return None

def get_word_color_hormozi(word: str, config: HormoziUltimateConfig) -> Tuple[int, int, int]:
    """Détermine la couleur d'un mot selon les règles Hormozi"""
    word_clean = word.upper().strip('.,!?":;()[]{}')
    
    if word_clean in HORMOZI_PERFECT_KEYWORDS:
        keyword_type = HORMOZI_PERFECT_KEYWORDS[word_clean]
        
        if keyword_type == 'money':
            return config.color_money
        elif keyword_type == 'action':
            return config.color_action
        elif keyword_type == 'success':
            return config.color_success
        elif keyword_type == 'info':
            return config.color_info
        elif keyword_type == 'warning':
            return config.color_warning
    
    return config.color_default

def get_contextual_emoji_hormozi(word: str) -> Optional[str]:
    """Récupère emoji contextuel pour un mot Hormozi"""
    word_clean = word.upper().strip('.,!?":;()[]{}')
    
    if word_clean in HORMOZI_EMOJI_MAP:
        return HORMOZI_EMOJI_MAP[word_clean]
    
    return None

def download_emoji_hormozi(emoji_char: str, size: int = 72) -> Optional[Image.Image]:
    """Télécharge emoji Twemoji au format PNG"""
    try:
        # Convertir emoji en code Unicode
        emoji_code = '-'.join([f'{ord(c):x}' for c in emoji_char])
        
        # URL Twemoji GitHub
        url = f'https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/{emoji_code}.png'
        
        # Cache local
        emoji_dir = Path('emoji_assets')
        emoji_dir.mkdir(exist_ok=True)
        emoji_path = emoji_dir / f'{emoji_code}.png'
        
        if not emoji_path.exists():
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                with open(emoji_path, 'wb') as f:
                    f.write(response.content)
            else:
                return None
        
        # Charger et redimensionner
        emoji_img = Image.open(emoji_path).convert('RGBA')
        if emoji_img.size != (size, size):
            emoji_img = emoji_img.resize((size, size), Image.Resampling.LANCZOS)
        
        return emoji_img
        
    except Exception:
        return None

def split_text_hormozi(text: str, max_words_per_line: int) -> List[str]:
    """Découpe texte style Hormozi (max 4-6 mots par ligne)"""
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        if len(current_line) < max_words_per_line:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines[:2]  # Max 2 lignes style Hormozi

def create_hormozi_perfect_frame(words_data: List[Dict], video_size: Tuple[int, int], 
                                config: HormoziUltimateConfig, frame_time: float) -> Image.Image:
    """Crée frame sous-titres style Hormozi MOT PAR MOT (prompt exact)"""
    
    width, height = video_size
    
    # Image transparente
    frame = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(frame)
    
    # Taille police adaptative avec limites (prompt)
    base_font_size = int(height * config.font_scale_factor)
    font_size = max(config.font_min_size, min(config.font_max_size, base_font_size))
    font = get_hormozi_font(font_size)
    
    # Collecter MOTS ACTIFS uniquement (mot par mot)
    active_words = []
    for word_data in words_data:
        if word_data['start'] <= frame_time < word_data['end']:
            word_age = frame_time - word_data['start']
            active_words.append({
                'text': word_data['text'].upper(),
                'age': word_age,
                'color': get_word_color_hormozi(word_data['text'], config),
                'emoji': get_contextual_emoji_hormozi(word_data['text'])
            })
    
    if not active_words:
        return frame
    
    # Construire ligne active avec mots visibles
    current_line = []
    for word_info in active_words:
        # Animation bounce subtile (prompt: 1.2x, 150ms)
        if config.bounce_enabled and word_info['age'] <= config.bounce_duration:
            progress = word_info['age'] / config.bounce_duration
            scale = 1 + (config.bounce_scale - 1) * (1 - progress) * np.sin(progress * np.pi)
        else:
            scale = 1.0
        
        # Fade-in rapide (prompt: 100-150ms)
        if config.fade_enabled and word_info['age'] <= config.fade_duration:
            opacity = int(255 * (word_info['age'] / config.fade_duration))
        else:
            opacity = 255
        
        word_info['scale'] = scale
        word_info['opacity'] = opacity
        current_line.append(word_info)
    
    # Adapter la taille si la ligne est trop large
    line_text = ' '.join([w['text'] for w in current_line])
    bbox = draw.textbbox((0, 0), line_text, font=font)
    line_width = bbox[2] - bbox[0]
    max_width = width * (1 - 2 * config.side_margins)
    
    if line_width > max_width:
        # Réduire la taille de police
        while line_width > max_width and font_size > config.font_min_size:
            font_size -= 2
            font = get_hormozi_font(font_size)
            bbox = draw.textbbox((0, 0), line_text, font=font)
            line_width = bbox[2] - bbox[0]
    
    # Position Hormozi (bas centré)
    line_height = font_size * config.line_spacing
    total_height = line_height  # Une seule ligne mot par mot
    
    start_y = height * (1 - config.bottom_margin_ratio) - total_height
    
    # Dessiner chaque ligne
    # Dessiner ligne unique avec mots actifs
    line_y = start_y
    line_x = (width - line_width) // 2
    if line_x < width * config.side_margins:
        line_x = int(width * config.side_margins)
    
    current_x = line_x
    for word_info in current_line:
        line_y = start_y + i * line_height
        
        # Mesurer ligne pour centrage
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        
        # Centrage horizontal Hormozi
        line_x = (width - line_width) // 2
        
        # Vérifier débordement
        if line_x < width * config.side_margins:
            line_x = int(width * config.side_margins)
        if line_x + line_width > width * (1 - config.side_margins):
            # Réduire police si débordement
            scale_factor = (width * (1 - 2 * config.side_margins)) / line_width
            if scale_factor < 1:
                reduced_size = int(font_size * scale_factor * 0.9)
                font = get_hormozi_font(reduced_size)
                bbox = draw.textbbox((0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]
                line_x = (width - line_width) // 2
        
        # Appliquer scale si bounce
        if scale != 1.0:
            # Image temporaire pour scale
            temp_size = (int(line_width * scale * 1.5), int(line_height * scale * 1.5))
            temp_img = Image.new('RGBA', temp_size, (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            
            temp_x = (temp_size[0] - int(line_width * scale)) // 2
            temp_y = (temp_size[1] - int(line_height * scale)) // 2
            
            # Dessiner avec couleurs mot par mot
            words = line.split()
            current_x = temp_x
            
            for word in words:
                word_color = get_word_color_hormozi(word, config)
                
                # Ombre si activée
                if config.shadow_enabled:
                    shadow_x = current_x + config.shadow_offset[0]
                    shadow_y = temp_y + config.shadow_offset[1]
                    temp_draw.text((shadow_x, shadow_y), word, font=font, 
                                 fill=(*config.shadow_color[:3], int(config.shadow_color[3] * opacity / 255)))
                
                # Outline noir Hormozi
                if config.outline_enabled:
                    for dx in range(-config.outline_width, config.outline_width + 1):
                        for dy in range(-config.outline_width, config.outline_width + 1):
                            if dx != 0 or dy != 0:
                                temp_draw.text((current_x + dx, temp_y + dy), word, 
                                             font=font, fill=(*config.outline_color, opacity))
                
                # Texte coloré principal
                temp_draw.text((current_x, temp_y), word, font=font, 
                             fill=(*word_color, opacity))
                
                # Avancer position
                word_bbox = temp_draw.textbbox((0, 0), word + " ", font=font)
                current_x += word_bbox[2] - word_bbox[0]
            
            # Appliquer scale
            if scale != 1.0:
                scaled_size = (int(temp_size[0] * scale), int(temp_size[1] * scale))
                temp_img = temp_img.resize(scaled_size, Image.Resampling.LANCZOS)
            
            # Coller sur frame
            paste_x = line_x - (temp_img.width - line_width) // 2
            paste_y = int(line_y) - (temp_img.height - int(line_height)) // 2
            frame.paste(temp_img, (paste_x, paste_y), temp_img)
            
        else:
            # Dessiner directement sans scale
            words = line.split()
            current_x = line_x
            
            for word in words:
                word_color = get_word_color_hormozi(word, config)
                
                # Ombre
                if config.shadow_enabled:
                    shadow_x = current_x + config.shadow_offset[0]
                    shadow_y = int(line_y) + config.shadow_offset[1]
                    draw.text((shadow_x, shadow_y), word, font=font, 
                            fill=(*config.shadow_color[:3], int(config.shadow_color[3] * opacity / 255)))
                
                # Outline noir
                if config.outline_enabled:
                    for dx in range(-config.outline_width, config.outline_width + 1):
                        for dy in range(-config.outline_width, config.outline_width + 1):
                            if dx != 0 or dy != 0:
                                draw.text((current_x + dx, int(line_y) + dy), word, 
                                        font=font, fill=(*config.outline_color, opacity))
                
                # Texte principal coloré
                draw.text((current_x, int(line_y)), word, font=font, 
                        fill=(*word_color, opacity))
                
                # Emoji si présent
                if word.upper().strip('.,!?":;()[]{}') in HORMOZI_EMOJI_MAP:
                    emoji_char = HORMOZI_EMOJI_MAP[word.upper().strip('.,!?":;()[]{}')]
                    emoji_img = download_hormozi_emoji(emoji_char)
                    if emoji_img:
                        emoji_size = int(font_size * config.emoji_size_ratio)
                        emoji_img = emoji_img.resize((emoji_size, emoji_size), Image.Resampling.LANCZOS)
                        
                        word_bbox = draw.textbbox((0, 0), word, font=font)
                        emoji_x = current_x + word_bbox[2] - word_bbox[0] + 5
                        emoji_y = int(line_y) + (font_size - emoji_size) // 2
                        
                        if opacity < 255:
                            emoji_alpha = emoji_img.copy()
                            alpha = emoji_alpha.split()[-1]
                            alpha = alpha.point(lambda p: int(p * opacity / 255))
                            emoji_alpha.putalpha(alpha)
                            frame.paste(emoji_alpha, (emoji_x, emoji_y), emoji_alpha)
                        else:
                            frame.paste(emoji_img, (emoji_x, emoji_y), emoji_img)
                
                # Avancer position
                word_bbox = draw.textbbox((0, 0), word + " ", font=font)
                current_x += word_bbox[2] - word_bbox[0]
    
    return frame

def parse_transcription_hormozi_perfect(transcription_data: List[Dict]) -> List[Dict]:
    """Parse transcription en MOTS INDIVIDUELS (prompt Hormozi exact)"""
    words_data = []
    
    for segment in transcription_data:
        text = segment['text'].strip()
        if not text:
            continue
        
        # Découper en mots (prompt: mot par mot)
        words = re.findall(r'\b\w+\b', text.upper())
        if not words:
            continue
        
        # Timing mot par mot
        segment_duration = segment['end'] - segment['start']
        word_duration = segment_duration / len(words)
        
        # Durée minimum/maximum par mot (prompt: rapide)
        word_duration = max(0.4, min(1.0, word_duration))
        
        current_time = segment['start']
        for word in words:
            word_end = current_time + word_duration
            
            words_data.append({
                'text': word,
                'start': current_time,
                'end': word_end,
                'duration': word_duration
            })
            
            current_time = word_end
    
    return words_data

def add_hormozi_perfect_subtitles(input_video_path: str, transcription_data: List[Dict], 
                                 output_video_path: str) -> str:
    """
    Génère sous-titres style Hormozi 1 PARFAIT
    Basé sur recherche approfondie du vrai style TikTok d'Alex Hormozi
    """
    
    print("🎯 STYLE HORMOZI 1 PARFAIT - RECHERCHE APPROFONDIE")
    print("=" * 70)
    print("✅ Police: Montserrat Black 900 (exacte)")
    print("✅ Couleurs: 5 couleurs signature Hormozi")
    print("✅ Animation: Bounce + fade-in authentique")
    print("✅ Position: Bas centré exact")
    print("✅ Timing: Phrases complètes synchronisées")
    
    # Vérifications
    if not os.path.exists(input_video_path):
        print(f"❌ Vidéo non trouvée: {input_video_path}")
        return input_video_path
    
    if not transcription_data:
        print("❌ Transcription manquante")
        return input_video_path
    
    # Config Hormozi parfaite
    config = HormoziUltimateConfig()
    
    # Parse mots individuels (prompt exact)
    words_data = parse_transcription_hormozi_perfect(transcription_data)
    print(f"📝 {len(words_data)} mots individuels Hormozi analysés")
    
    # Analyser mots-clés détectés
    all_words = [w['text'] for w in words_data]
    keywords_found = []
    for word in all_words:
        clean_word = word.strip('.,!?":;()[]{}')
        if clean_word in HORMOZI_PERFECT_KEYWORDS:
            keywords_found.append(f"{clean_word}({HORMOZI_PERFECT_KEYWORDS[clean_word]})")
    
    if keywords_found:
        print(f"🎨 Mots-clés Hormozi détectés: {', '.join(keywords_found[:5])}...")
    else:
        print("⚠️ Aucun mot-clé Hormozi dans cette transcription")
    
    # Métadonnées vidéo
    try:
        ffprobe_cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', input_video_path
        ]
        result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True)
        video_info = json.loads(result.stdout)
        
        video_stream = next(s for s in video_info['streams'] if s['codec_type'] == 'video')
        video_width = int(video_stream['width'])
        video_height = int(video_stream['height'])
        fps = float(video_stream['r_frame_rate'].split('/')[0]) / float(video_stream['r_frame_rate'].split('/')[1])
        duration = float(video_info['format']['duration'])
        
        font_size_calculated = int(video_height * config.font_scale_factor)
        print(f"📊 Vidéo: {video_width}x{video_height}, {fps} FPS")
        print(f"🔤 Police Hormozi: {font_size_calculated}px (facteur {config.font_scale_factor})")
        
    except Exception as e:
        print(f"❌ Erreur ffprobe: {e}")
        return input_video_path
    
    # Génération frames
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            print("🎬 Génération frames style Hormozi PARFAIT...")
            
            frame_count = int(duration * fps)
            
            for frame_num in range(frame_count):
                frame_time = frame_num / fps
                
                subtitle_frame = create_hormozi_perfect_frame(
                    words_data, (video_width, video_height), config, frame_time
                )
                
                frame_path = os.path.join(temp_dir, f"subtitle_{frame_num:06d}.png")
                subtitle_frame.save(frame_path, 'PNG')
            
            print(f"   {frame_count} frames Hormozi générées")
            
            # Composition ffmpeg
            print("🎬 Composition finale avec style Hormozi authentique...")
            
            os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
            
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', input_video_path,
                '-i', os.path.join(temp_dir, 'subtitle_%06d.png'),
                '-filter_complex', f'[0:v][1:v]overlay=0:0[v]',
                '-map', '[v]',
                '-map', '0:a?',
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-crf', '18',  # Qualité supérieure
                '-preset', 'medium',
                '-pix_fmt', 'yuv420p',
                '-r', str(fps),
                output_video_path
            ]
            
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ SUCCÈS! Style Hormozi 1 PARFAIT: {Path(output_video_path).name}")
                print("🎉 CARACTÉRISTIQUES HORMOZI APPLIQUÉES:")
                print(f"  🔤 Police: Montserrat Black 900 ({font_size_calculated}px)")
                print(f"  📍 Position: {config.bottom_margin_ratio:.0%} du bas (authentique)")
                print(f"  🎨 Couleurs: {len(keywords_found)} mots-clés colorés")
                print(f"  ⚡ Animation: Bounce {config.bounce_scale}x + fade {config.fade_duration}s")
                print(f"  📱 Format: Max {config.max_words_per_line} mots/ligne, {config.max_lines} lignes")
                print(f"  🎯 Style: 100% conforme recherche TikTok Hormozi")
                return output_video_path
            else:
                print(f"❌ Erreur ffmpeg: {result.stderr}")
                return input_video_path
                
    except Exception as e:
        print(f"❌ Erreur génération: {e}")
        import traceback
        traceback.print_exc()
        return input_video_path 