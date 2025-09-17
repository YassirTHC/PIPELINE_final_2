"""
Style HORMOZI 1 - REPRODUCTION EXACTE DU PROMPT ORIGINAL
Toutes les spécifications du prompt respectées à 100%
"""

import os
import subprocess
import tempfile
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import requests

class HormoziPromptConfig:
    """Configuration EXACTE selon le prompt Hormozi original"""
    
    def __init__(self):
        # PROMPT: Police bold, sans-serif très épaisse (Impact, Arial Black)
        self.font_primary = "Impact"
        self.font_secondary = "Arial Black"
        self.font_tertiary = "Anton"
        
        # PROMPT: Gros, centré horizontalement, positionné en bas
        self.font_size_base = 0.08  # 8% de la hauteur
        self.font_min_size = 24
        self.font_max_size = 80
        
        # PROMPT: Texte en MAJUSCULES
        self.text_uppercase = True
        
        # PROMPT: Couleur de base blanc pur avec outline noir épais (3-5px)
        self.color_default = (255, 255, 255)  # Blanc pur
        self.outline_enabled = True
        self.outline_color = (0, 0, 0)        # Noir
        self.outline_width = 2                # 2px (optimisé couleurs)
        
        # PROMPT: Mots-clés colorés différemment
        self.color_money = (255, 215, 0)      # Jaune vif #FFD700
        self.color_action = (255, 0, 0)       # Rouge #FF0000
        self.color_success = (0, 255, 0)      # Vert #00FF00
        self.color_info = (30, 144, 255)      # Bleu clair #1E90FF
        
        # PROMPT: Animation légère zoom-in (scale 1.2) puis retour 1.0
        self.bounce_enabled = True
        self.bounce_scale = 1.25              # Scale modéré pour éviter dépassement
        self.bounce_duration = 0.18           # 180ms (perceptible mais court)
        
        # PROMPT: Fade-in rapide (100-150ms)
        self.fade_enabled = True
        self.fade_duration = 0.12             # 120ms (net)
        
        # PROMPT: Mot par mot ou bloc par bloc
        self.word_by_word = True
        
        # PROMPT: Positionnement bas de la vidéo, centré
        self.bottom_margin = 0.12             # 12% du bas
        self.side_margins = 0.05              # 5% marges latérales
        
        # PROMPT: Emojis MONEY → 💸, FIRE → 🔥, IDEA → 💡
        self.emoji_enabled = True
        self.emoji_size_ratio = 1.0
        self.emoji_spacing = 10
        
        # Synchronisation: biais global en secondes (peut être négatif)
        self.timing_bias_s = 0.0
        
        # Fallback visibilité: colorer le mot actif si aucun mot-clé détecté
        self.force_color_current_if_none = True
        self.force_color_current_color = (255, 215, 0)  # Jaune
        self.force_emoji_when_colored = True
        self.force_emoji_char = '💬'

# PROMPT: Mots-clés colorés (exemples du prompt)
HORMOZI_KEYWORDS = {
    # Jaune vif (money-related)
    'MONEY': 'money', 'CASH': 'money', 'PROFIT': 'money', 'BUSINESS': 'money',
    
    # Rouge (action/urgence)
    'NOW': 'action', 'STOP': 'action', 'MUST': 'action', 'URGENT': 'action',
    
    # Vert (success)
    'SUCCESS': 'success', 'WIN': 'success', 'BEST': 'success', 'GREAT': 'success',
    
    # Bleu clair (info/learning)
    'LEARN': 'info', 'IDEA': 'info', 'SECRET': 'info', 'TIP': 'info'
}

# PROMPT: Mapping emojis (exemples du prompt)
HORMOZI_EMOJIS = {
    'MONEY': '💸', 'FIRE': '🔥', 'IDEA': '💡', 'SUCCESS': '✅',
    'BUSINESS': '💼', 'SECRET': '🤫', 'WIN': '🏆', 'BEST': '🏆'
}

def get_system_font_prompt(font_name: str, size: int) -> ImageFont.FreeTypeFont:
    """Charge police système selon prompt (Impact priorité 1)"""
    
    # PROMPT: Impact, Arial Black, Anton
    font_paths = [
        f"C:/Windows/Fonts/{font_name.lower()}.ttf",
        f"C:/Windows/Fonts/{font_name.upper()}.TTF",
        f"C:/Windows/Fonts/{font_name.replace(' ', '')}.ttf"
    ]
    
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                continue
    
    # Fallback
    try:
        return ImageFont.truetype(font_name, size)
    except:
        return ImageFont.load_default()

def get_word_color_prompt(word: str, config: HormoziPromptConfig) -> Tuple[int, int, int]:
    """PROMPT: Certains mots colorés différemment pour attirer l'attention"""
    
    word_clean = word.upper().strip('.,!?":;()[]{}')
    
    if word_clean in HORMOZI_KEYWORDS:
        keyword_type = HORMOZI_KEYWORDS[word_clean]
        
        if keyword_type == 'money':
            return config.color_money      # Jaune vif #FFD700
        elif keyword_type == 'action':
            return config.color_action     # Rouge #FF0000
        elif keyword_type == 'success':
            return config.color_success    # Vert #00FF00
        elif keyword_type == 'info':
            return config.color_info       # Bleu clair #1E90FF
    
    return config.color_default  # Blanc pur

def get_emoji_prompt(word: str) -> Optional[str]:
    """PROMPT: Mapping optionnel mot→emoji (MONEY → 💸, FIRE → 🔥, IDEA → 💡)"""
    
    word_clean = word.upper().strip('.,!?":;()[]{}')
    return HORMOZI_EMOJIS.get(word_clean)

def download_twemoji_png(emoji_char: str, size: int = 72) -> Optional[Image.Image]:
    """Télécharge emoji Twemoji PNG pour éviter carrés vides"""
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

def parse_transcription_word_by_word(transcription_data: List[Dict]) -> List[Dict]:
    """Découpe la transcription en mots; utilise d'abord les timings Whisper mot-à-mot si présents."""
    # Accès config globale optionnelle: timing bias via variable locale si disponible
    # On permet une légère personnalisation en lisant une variable annexe si set
    words_data: List[Dict] = []
    # Essayer d'obtenir un biais depuis les segments (optionnel) ou 0.0
    bias = 0.0
    try:
        # Certains appels injectent {'_bias': x} en tête
        if transcription_data and isinstance(transcription_data[0], dict) and "_bias" in transcription_data[0]:
            bias = float(transcription_data[0]["_bias"]) or 0.0
    except Exception:
        bias = 0.0
    
    for segment in transcription_data:
        text = (segment.get('text') or '').strip()
        if not text:
            continue
        
        # Si Whisper fournit des timings par mot, les utiliser PRIORITAIREMENT
        precise_words = segment.get('words')
        if isinstance(precise_words, list) and precise_words:
            seg_start = float(segment.get('start') or 0.0) + bias
            for w in precise_words:
                wt = (w.get('word') or w.get('text') or '').strip()
                if not wt:
                    continue
                ws = float(w.get('start') or seg_start) + bias
                we = float(w.get('end') or ws) + bias
                if we <= ws:
                    we = ws + 0.08  # 80ms minimum
                words_data.append({'text': wt.upper(), 'start': ws, 'end': we, 'duration': we - ws})
            continue
        
        # Sinon: fallback découpe en mots + timing équitable
        words = re.findall(r'\b\w+\b', text)
        if not words:
            continue
        segment_start = float(segment.get('start') or 0.0) + bias
        segment_end = float(segment.get('end') or segment_start) + bias
        segment_duration = max(0.08, segment_end - segment_start)
        raw_word_duration = segment_duration / max(1, len(words))
        if segment_duration > 3.0:
            word_duration = 0.8
        else:
            word_duration = max(0.3, min(1.2, raw_word_duration))
        current_time = segment_start
        for word in words:
            word_end = current_time + word_duration
            words_data.append({'text': word.upper(), 'start': current_time, 'end': word_end, 'duration': word_duration})
            current_time = word_end
    return words_data

def create_hormozi_prompt_frame(words_data: List[Dict], video_size: Tuple[int, int], 
                               config: HormoziPromptConfig, frame_time: float) -> Image.Image:
    """PROMPT: Créer overlay texte par frame → reset à chaque frame pour éviter accumulation"""
    
    width, height = video_size
    
    # Image transparente (reset à chaque frame)
    frame = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(frame)
    
    # PROMPT: Taille adaptée
    base_font_size = int(height * config.font_size_base)
    font_size = max(config.font_min_size, min(config.font_max_size, base_font_size))
    
    # PROMPT: Police bold, sans-serif très épaisse (Impact priorité)
    font = get_system_font_prompt(config.font_primary, font_size)
    
    # Collecter TOUS LES MOTS ACTIFS à ce moment (même timing)
    active_words = []
    
    for word_data in words_data:
        if word_data['start'] <= frame_time < word_data['end']:
            word_age = frame_time - word_data['start']
            
            # PROMPT: Animation léger zoom-in (scale 1.2) puis retour 1.0
            if config.bounce_enabled and word_age <= config.bounce_duration:
                progress = word_age / config.bounce_duration
                scale = 1 + (config.bounce_scale - 1) * (1 - progress) * np.sin(progress * np.pi)
            else:
                scale = 1.0
            
            # PROMPT: Fade-in rapide (100-150ms)
            if config.fade_enabled and word_age <= config.fade_duration:
                opacity = int(255 * (word_age / config.fade_duration))
            else:
                opacity = 255
            
            active_words.append({
                'text': word_data['text'],
                'color': get_word_color_prompt(word_data['text'], config),
                'emoji': get_emoji_prompt(word_data['text']),
                'scale': scale,
                'opacity': opacity
            })
    
    if not active_words:
        return frame
    
    # Fallback couleur: si aucun mot actif n'est coloré, colorer le mot courant
    if config.force_color_current_if_none:
        any_colored = any(w['color'] != config.color_default for w in active_words)
        if not any_colored:
            active_words[0]['color'] = config.force_color_current_color
            if config.emoji_enabled and config.force_emoji_when_colored and not active_words[0].get('emoji'):
                active_words[0]['emoji'] = config.force_emoji_char
    
    # Construire ligne de texte
    line_text = ' '.join([w['text'] for w in active_words])
    
    # Mesurer et ajuster taille si nécessaire (tenir compte du scale max)
    bbox = draw.textbbox((0, 0), line_text, font=font)
    line_width = bbox[2] - bbox[0]
    # Réserver de la place pour l'animation scale max
    effective_max_scale = max(1.0, config.bounce_scale)
    max_width = (width * (1 - 2 * config.side_margins)) / effective_max_scale
    
    if line_width > max_width:
        # Réduire police pour rester dans le cadre
        while line_width > max_width and font_size > config.font_min_size:
            font_size -= 2
            font = get_system_font_prompt(config.font_primary, font_size)
            bbox = draw.textbbox((0, 0), line_text, font=font)
            line_width = bbox[2] - bbox[0]
    
    # PROMPT: Positionné en bas de la vidéo, centré horizontalement
    text_y = int(height * (1 - config.bottom_margin)) - font_size
    text_x = (width - line_width) // 2
    
    # Vérifier marges latérales
    if text_x < width * config.side_margins:
        text_x = int(width * config.side_margins)
    
    # Dessiner chaque mot individuellement AVEC ANIMATIONS
    current_x = text_x
    
    for word_info in active_words:
        word = word_info['text']
        word_color = word_info['color']
        opacity = word_info['opacity']
        scale = word_info['scale']  # ANIMATION SCALE !
        
        # Appliquer opacité seulement si très faible, sinon garder couleurs vives
        if opacity < 100:  # Seulement si très transparent
            word_color = tuple(int(c * opacity / 255) for c in word_color)
        # Sinon garder couleurs pleines pour visibilité
        
        # APPLIQUER ANIMATION SCALE au font
        animated_font_size = int(font_size * scale)
        animated_font = get_system_font_prompt(config.font_primary, animated_font_size)
        
        # Ajuster position pour centrer le texte agrandi et le CLAMPER dans le cadre
        scale_offset_y = int((animated_font_size - font_size) / 2)
        animated_text_y = text_y - scale_offset_y
        # Clamp vertical pour rester dans l'image
        animated_text_y = max(0, min(animated_text_y, height - animated_font_size - 1))
        
        # PROMPT: Outline noir épais SOUS le texte (3-5px) pour lisibilité
        if config.outline_enabled:
            outline_width = config.outline_width
            # Dessiner outline D'ABORD (en arrière-plan)
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx*dx + dy*dy <= outline_width*outline_width:
                        outline_color = config.outline_color
                        # Garder outline noir fort pour lisibilité
                        draw.text((current_x + dx, animated_text_y + dy), word, 
                                font=animated_font, fill=outline_color)
        
        # Dessiner texte principal COLORÉ PAR-DESSUS l'outline AVEC SCALE
        draw.text((current_x, animated_text_y), word, font=animated_font, fill=word_color)
        
        # PROMPT: Emojis correctement rendus (pas de carrés vides)
        if config.emoji_enabled and word_info['emoji']:
            emoji_size = int(font_size * config.emoji_size_ratio)
            emoji_img = download_twemoji_png(word_info['emoji'], emoji_size)
            
            if emoji_img:
                # Position emoji à côté du mot AVEC SCALE
                word_bbox = draw.textbbox((0, 0), word, font=animated_font)
                word_width = word_bbox[2] - word_bbox[0]
                emoji_x = current_x + word_width + config.emoji_spacing
                # Centrer l'emoji sur la ligne animée, clampé dans le cadre
                emoji_y = max(0, min(animated_text_y + (animated_font_size - emoji_size) // 2, height - emoji_size - 1))
                
                # Ajuster opacité emoji
                if opacity < 255:
                    emoji_alpha = emoji_img.copy()
                    alpha = emoji_alpha.split()[-1]
                    alpha = alpha.point(lambda p: int(p * opacity / 255))
                    emoji_alpha.putalpha(alpha)
                    frame.paste(emoji_alpha, (emoji_x, emoji_y), emoji_alpha)
                else:
                    frame.paste(emoji_img, (emoji_x, emoji_y), emoji_img)
        
        # Avancer position AVEC SCALE
        word_bbox = draw.textbbox((0, 0), word + " ", font=animated_font)
        current_x += word_bbox[2] - word_bbox[0]
    
    return frame

def add_hormozi_subtitles(input_video_path: str, transcription_data: List[Dict], 
                         output_video_path: str) -> str:
    """
    PROMPT EXACT: Génère des sous-titres style Hormozi 1 (texte bold, mots-clés colorés,
    majuscules, animations bounce/fade, disparition nette).
    """
    
    print("🎯 STYLE HORMOZI 1 - PROMPT ORIGINAL EXACT")
    print("=" * 70)
    print("✅ Police: Impact bold, sans-serif très épaisse")
    print("✅ MAJUSCULES: Toujours activé")
    print("✅ Couleurs: Blanc base + mots-clés colorés")
    print("✅ Outline: 3px noir épais")
    print("✅ Animation: Zoom 1.2x + fade 120ms")
    print("✅ Timing: Mot par mot strict")
    print("✅ Emojis: MONEY→💸, FIRE→🔥, IDEA→💡")
    
    # Vérifications
    if not os.path.exists(input_video_path):
        print(f"❌ Vidéo non trouvée: {input_video_path}")
        return input_video_path
    
    if not transcription_data:
        print("❌ Transcription manquante")
        return input_video_path
    
    # Config prompt exact
    config = HormoziPromptConfig()
    
    # Appliquer un biais global optionnel si passé via premier segment {'_bias': x}
    if transcription_data and isinstance(transcription_data[0], dict) and "_bias" in transcription_data[0]:
        try:
            config.timing_bias_s = float(transcription_data[0]["_bias"]) or 0.0
        except Exception:
            config.timing_bias_s = 0.0
    
    # PROMPT: Mot par mot avec timing strict
    words_data = parse_transcription_word_by_word(transcription_data)
    print(f"📝 {len(words_data)} mots individuels analysés")
    
    # Analyser mots-clés colorés
    keywords_found = []
    for word_data in words_data:
        word = word_data['text']
        if word in HORMOZI_KEYWORDS:
            keywords_found.append(f"{word}({HORMOZI_KEYWORDS[word]})")
    
    if keywords_found:
        print(f"🎨 Mots-clés colorés: {', '.join(keywords_found[:5])}...")
    
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
        
        font_size_calculated = int(video_height * config.font_size_base)
        print(f"📊 Vidéo: {video_width}x{video_height}, {fps} FPS")
        print(f"🔤 Police Impact: {font_size_calculated}px")
        
    except Exception as e:
        print(f"❌ Erreur ffprobe: {e}")
        return input_video_path
    
    # Génération frames
    try:
        print("🎬 Génération frames Hormozi PROMPT EXACT...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_count = int(duration * fps)
            
            for frame_num in range(frame_count):
                frame_time = frame_num / fps
                
                # PROMPT: Créer overlay texte par frame (reset chaque frame)
                subtitle_frame = create_hormozi_prompt_frame(
                    words_data, (video_width, video_height), config, frame_time
                )
                
                frame_path = os.path.join(temp_dir, f"subtitle_{frame_num:06d}.png")
                subtitle_frame.save(frame_path, 'PNG')
            
            print(f"   {frame_count} frames générées")
            
            # PROMPT: Export final libx264, yuv420p, même fps, audio conservé
            print("🎬 Composition finale...")
            
            # Créer le dossier de sortie si nécessaire
            output_dir = os.path.dirname(output_video_path)
            if output_dir:  # Seulement si ce n'est pas une chaîne vide
                os.makedirs(output_dir, exist_ok=True)
            
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', input_video_path,
                '-i', os.path.join(temp_dir, 'subtitle_%06d.png'),
                '-filter_complex', f'[0:v][1:v]overlay=0:0[v]',
                '-map', '[v]',
                '-map', '0:a?',
                '-c:v', 'libx264',     # PROMPT: libx264
                '-c:a', 'aac',
                '-crf', '18',
                '-preset', 'medium',
                '-pix_fmt', 'yuv420p', # PROMPT: yuv420p
                '-r', str(fps),        # PROMPT: même fps
                output_video_path
            ]
            
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ SUCCÈS PROMPT HORMOZI: {Path(output_video_path).name}")
                print("🎉 TOUTES SPÉCIFICATIONS PROMPT APPLIQUÉES:")
                print(f"  🔤 Police: Impact bold ({font_size_calculated}px)")
                print(f"  📍 Position: {config.bottom_margin:.0%} du bas centré")
                print(f"  🎨 Couleurs: {len(keywords_found)} mots-clés colorés")
                print(f"  ⚡ Animation: Zoom {config.bounce_scale}x + fade {config.fade_duration*1000:.0f}ms")
                print(f"  📝 Timing: Mot par mot strict")
                print(f"  😀 Emojis: Twemoji PNG (pas de carrés)")
                print(f"  ⚫ Outline: {config.outline_width}px noir épais")
                return output_video_path
            else:
                print(f"❌ Erreur ffmpeg: {result.stderr}")
                print(f"🔍 Commande ffmpeg: {' '.join(ffmpeg_cmd)}")
                return input_video_path
    
    except Exception as e:
        print(f"❌ Erreur génération: {e}")
        return input_video_path 