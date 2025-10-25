ï»¿# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
SystÃƒÂ¨me de sous-titres dynamiques style Submagic
BasÃƒÂ© sur l'analyse approfondie de JEVEUXCA.mp4 et des images fournies
"""

import os
import sys
import re
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from moviepy.editor import *
import json

# Configuration Submagic
class SubmagicConfig:
    """Configuration pour les sous-titres style Submagic"""
    
    def __init__(self):
        # Polices et tailles
        self.font_base_size = 45         # Taille de base
        self.font_keyword_size = 55      # Taille pour mots-clÃƒÂ©s
        self.font_emphasis_size = 65     # Taille pour emphase maximum
        self.font_family = "Arial Black"  # Police trÃƒÂ¨s grasse
        
        # NOUVELLE PALETTE 6 COULEURS - SYSTÃƒË†ME AVANCÃƒâ€°
        self.color_white = (255, 255, 255)      # Ã¢Å¡Âª Blanc - RARE maintenant
        self.color_green = (50, 255, 50)        # Ã°Å¸Å¸Â¢ Vert - Actions/SuccÃƒÂ¨s
        self.color_red = (255, 50, 50)          # Ã°Å¸â€Â´ Rouge - Questions/ProblÃƒÂ¨mes
        self.color_yellow = (255, 255, 50)      # Ã°Å¸Å¸Â¡ Jaune - Temps/Important
        self.color_orange = (255, 165, 0)       # Ã°Å¸Å¸Â  Orange - Argent/RÃƒÂ©sultats
        self.color_blue = (50, 150, 255)        # Ã°Å¸â€Âµ Bleu - Ãƒâ€°motions/Social
        self.color_purple = (150, 50, 255)      # Ã°Å¸Å¸Â£ Violet - Tech/Innovation
        self.color_light_gray = (180, 180, 180) # Ã°Å¸â€Ëœ Gris clair - Mots neutres
        
        # Contours et effets
        self.stroke_width = 4                   # Contour noir ÃƒÂ©pais
        self.stroke_color = (0, 0, 0)          # Noir
        self.shadow_offset = 2                  # Ombre portÃƒÂ©e
        self.glow_radius = 3                    # Effet glow
        
        # Positionnement
        self.bottom_margin = 0.15               # 15% du bas
        self.horizontal_center = True           # CentrÃƒÂ© horizontalement
        self.max_width_percent = 0.9            # 90% de la largeur max
        
        # Animations et timing
        self.word_appear_duration = 0.3         # DurÃƒÂ©e apparition mot
        self.bounce_intensity = 0.2             # IntensitÃƒÂ© rebond
        self.color_transition_speed = 0.5       # Vitesse transition couleur
        self.persistence_enabled = True         # Les mots restent visibles
        
        # Emojis contextuels
        self.emoji_enabled = True
        self.emoji_size_ratio = 0.8             # 80% de la taille du texte
        self.emoji_spacing = 10                 # Espacement emoji-texte

# Mappings emojis contextuels (basÃƒÂ©s sur l'analyse)
SUBMAGIC_EMOJI_MAP = {
    # Mouvement et action
    'run': 'Ã°Å¸ÂÆ’', 'running': 'Ã°Å¸ÂÆ’', 'walk': 'Ã°Å¸Å¡Â¶', 'move': 'Ã°Å¸Å¡Â¶', 'go': 'Ã°Å¸Å¡Â¶',
    'lift': 'Ã°Å¸Ââ€¹Ã¯Â¸Â', 'exercise': 'Ã°Å¸Ââ€¹Ã¯Â¸Â', 'workout': 'Ã°Å¸â€™Âª', 'train': 'Ã°Å¸â€™Âª', 'gym': 'Ã°Å¸Ââ€¹Ã¯Â¸Â',
    
    # Ãƒâ€°motions et rÃƒÂ©actions
    'behavior': 'Ã°Å¸Å½Â­', 'act': 'Ã°Å¸Å½Â­', 'react': 'Ã°Å¸Å½Â­', 'feel': 'Ã¢ÂÂ¤Ã¯Â¸Â',
    'quit': 'Ã¢ÂÅ’', 'stop': 'Ã¢â€ºâ€', 'end': 'Ã°Å¸â€Å¡', 'finish': 'Ã¢Å“â€¦',
    
    # Questions et rÃƒÂ©flexion
    'why': 'Ã¢Ââ€œ', 'what': 'Ã¢Ââ€œ', 'how': 'Ã¢Ââ€œ', 'when': 'Ã¢Ââ€œ', 'where': 'Ã¢Ââ€œ',
    'think': 'Ã°Å¸Â¤â€', 'know': 'Ã°Å¸Â§Â ', 'understand': 'Ã°Å¸â€™Â¡', 'learn': 'Ã°Å¸â€œÅ¡',
    
    # IntensitÃƒÂ© et emphase
    'every': 'Ã°Å¸â€™Â¯', 'all': 'Ã°Å¸â€™Â¯', 'always': 'Ã°Å¸â€™Â¯', 'never': 'Ã¢ÂÅ’',
    'time': 'Ã¢ÂÂ°', 'moment': 'Ã¢ÂÂ±Ã¯Â¸Â', 'now': 'Ã¢Å¡Â¡', 'today': 'Ã°Å¸â€œâ€¦',
    
    # Argent et succÃƒÂ¨s
    'money': 'Ã°Å¸â€™Â°', 'rich': 'Ã°Å¸â€™Â¸', 'success': 'Ã°Å¸Ââ€ ', 'win': 'Ã°Å¸Â¥â€¡',
    'lose': 'Ã°Å¸ËœÅ¾', 'fail': 'Ã¢ÂÅ’', 'problem': 'Ã¢Å¡Â Ã¯Â¸Â',
    
    # Communication
    'say': 'Ã°Å¸â€™Â¬', 'tell': 'Ã°Å¸â€™Â¬', 'speak': 'Ã°Å¸â€”Â£Ã¯Â¸Â', 'listen': 'Ã°Å¸â€˜â€š',
    'look': 'Ã°Å¸â€˜â‚¬', 'see': 'Ã°Å¸â€˜ÂÃ¯Â¸Â', 'watch': 'Ã°Å¸â€œÂº', 'show': 'Ã°Å¸â€˜â€ ',
    
    # Ãƒâ€°tats et sentiments
    'happy': 'Ã°Å¸ËœÅ ', 'sad': 'Ã°Å¸ËœÂ¢', 'angry': 'Ã°Å¸ËœÂ ', 'surprised': 'Ã°Å¸ËœÂ²',
    'love': 'Ã¢ÂÂ¤Ã¯Â¸Â', 'hate': 'Ã°Å¸â€™â€', 'like': 'Ã°Å¸â€˜Â', 'dislike': 'Ã°Å¸â€˜Å½',
}

def get_system_fonts():
    """RÃƒÂ©cupÃƒÂ¨re les polices systÃƒÂ¨me disponibles"""
    font_paths = []
    
    # Windows
    if os.name == 'nt':
        fonts_dir = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
        font_candidates = [
            'arial.ttf', 'arialbd.ttf',  # Arial Bold
            'calibrib.ttf',  # Calibri Bold
            'seguisb.ttf',   # Segoe UI Bold
            'impact.ttf',    # Impact
        ]
    # macOS
    elif sys.platform == 'darwin':
        fonts_dir = Path('/System/Library/Fonts')
        font_candidates = [
            'Arial Bold.ttf', 'Arial.ttc',
            'Helvetica.ttc', 'Impact.ttf'
        ]
    # Linux
    else:
        fonts_dir = Path('/usr/share/fonts/truetype')
        font_candidates = [
            'liberation/LiberationSans-Bold.ttf',
            'dejavu/DejaVuSans-Bold.ttf'
        ]
    
    for font_file in font_candidates:
        font_path = fonts_dir / font_file
        if font_path.exists():
            font_paths.append(str(font_path))
    
    return font_paths

def get_submagic_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    """RÃƒÂ©cupÃƒÂ¨re une police style Submagic (grasse et moderne)"""
    font_paths = get_system_fonts()
    
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            continue
    
    # Fallback vers police par dÃƒÂ©faut
    try:
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()

def detect_keyword_type(word: str, context: str = "") -> str:
    """DÃƒÂ©tecte le type de mot pour appliquer le bon style - SYSTÃƒË†ME AMÃƒâ€°LIORÃƒâ€°"""
    word_lower = word.lower().strip()
    context_lower = context.lower()
    
    # Ã°Å¸Å¸Â¢ VERT - Actions/Mouvements/SuccÃƒÂ¨s/Positif
    action_success_words = {
        'run', 'running', 'walk', 'move', 'go', 'come', 'lift', 'exercise', 
        'work', 'play', 'dance', 'jump', 'climb', 'swim', 'drive', 'behavior',
        'win', 'success', 'good', 'great', 'best', 'perfect', 'amazing',
        'grow', 'build', 'create', 'make', 'start', 'begin', 'achieve',
        'power', 'strong', 'energy', 'force', 'push', 'pull', 'grab'
    }
    
    # Ã°Å¸â€Â´ ROUGE - Questions/ProblÃƒÂ¨mes/Urgence/NÃƒÂ©gatif
    question_problem_words = {
        'why', 'what', 'how', 'when', 'where', 'who', 'which',
        'problem', 'issue', 'wrong', 'bad', 'terrible', 'awful', 'hate',
        'quit', 'stop', 'fail', 'failure', 'lose', 'losing', 'lost',
        'never', 'no', 'not', 'cant', 'cannot', 'impossible', 'difficult',
        'pain', 'hurt', 'struggle', 'fight', 'war', 'battle', 'against'
    }
    
    # Ã°Å¸Å¸Â¡ JAUNE - Temps/Chiffres/Transition/Important
    time_transition_words = {
        'time', 'moment', 'day', 'week', 'month', 'year', 'today', 'tomorrow',
        'now', 'then', 'after', 'before', 'during', 'while', 'when',
        'first', 'second', 'third', 'last', 'final', 'end', 'finish',
        'about', 'around', 'every', 'each', 'all', 'some', 'many', 'few',
        'next', 'previous', 'future', 'past', 'always', 'sometimes', 'often'
    }
    
    # Ã°Å¸Å¸Â  ORANGE - Argent/BÃƒÂ©nÃƒÂ©fices/RÃƒÂ©sultats/Conclusion
    money_benefit_words = {
        'money', 'cash', 'dollar', 'euro', 'price', 'cost', 'pay', 'buy',
        'sell', 'profit', 'income', 'salary', 'rich', 'wealth', 'expensive',
        'benefit', 'advantage', 'result', 'outcome', 'effect', 'impact',
        'change', 'difference', 'better', 'improvement', 'upgrade', 'boost',
        'finally', 'conclusion', 'summary', 'total', 'complete', 'done'
    }
    
    # Ã°Å¸â€Âµ BLEU - Ãƒâ€°motions/Relations/Personnel/Social
    emotion_social_words = {
        'feel', 'feeling', 'emotion', 'happy', 'sad', 'angry', 'excited',
        'love', 'like', 'enjoy', 'fun', 'funny', 'laugh', 'smile', 'cry',
        'people', 'person', 'friend', 'family', 'team', 'group', 'together',
        'relationship', 'connect', 'share', 'help', 'support', 'care',
        'me', 'you', 'we', 'us', 'they', 'them', 'everyone', 'someone'
    }
    
    # Ã°Å¸Å¸Â£ VIOLET - Innovation/Futur/Technique/SpÃƒÂ©cialisÃƒÂ©
    tech_innovation_words = {
        'technology', 'tech', 'digital', 'online', 'internet', 'app', 'software',
        'ai', 'artificial', 'intelligence', 'robot', 'machine', 'computer',
        'innovation', 'new', 'modern', 'advanced', 'latest', 'cutting', 'edge',
        'system', 'method', 'technique', 'strategy', 'approach', 'way', 'process',
        'future', 'tomorrow', 'evolution', 'revolution', 'transformation'
    }
    
    # Analyse contextuelle pour affiner la dÃƒÂ©tection
    word_length = len(word)
    is_caps = word.isupper() and word_length > 2
    is_number = word.isdigit()
    
    # PrioritÃƒÂ©s de dÃƒÂ©tection (ordre important)
    if word_lower in question_problem_words or is_caps:
        return 'emphasis'  # ROUGE - Questions/ProblÃƒÂ¨mes/Emphase
    elif word_lower in action_success_words:
        return 'action'   # VERT - Actions/SuccÃƒÂ¨s
    elif word_lower in money_benefit_words:
        return 'money'    # ORANGE - Argent/BÃƒÂ©nÃƒÂ©fices
    elif word_lower in emotion_social_words:
        return 'social'   # BLEU - Ãƒâ€°motions/Social
    elif word_lower in tech_innovation_words:
        return 'tech'     # VIOLET - Tech/Innovation
    elif word_lower in time_transition_words or is_number:
        return 'important' # JAUNE - Temps/Important
    else:
        # RÃƒâ€°DUCTION DRASTIQUE DU BLANC - Analyse contextuelle avancÃƒÂ©e
        
        # Si le mot est dans un contexte d'action, le marquer comme action
        if any(action in context_lower for action in ['run', 'move', 'go', 'do', 'make']):
            return 'action'
        
        # Si contexte questionneur, marquer comme emphase
        if any(q in context_lower for q in ['why', 'what', 'how', 'problem']):
            return 'emphasis'
        
        # Si contexte temporel, marquer comme important
        if any(t in context_lower for t in ['time', 'when', 'now', 'today']):
            return 'important'
        
        # Mots courts frÃƒÂ©quents - leur donner une couleur selon position
        if word_lower in ['the', 'and', 'or', 'but', 'so', 'if', 'is', 'are', 'was', 'were']:
            return 'neutral'  # GRIS CLAIR au lieu de blanc
        
        # Verbes courants - les marquer comme action
        if word_lower in ['do', 'did', 'does', 'get', 'got', 'have', 'had', 'take', 'took']:
            return 'action'
        
        # Adverbes d'intensitÃƒÂ© - les marquer comme emphase
        if word_lower in ['very', 'really', 'super', 'totally', 'completely', 'absolutely']:
            return 'emphasis'
        
        # Par dÃƒÂ©faut, plus de couleur, moins de blanc
        if word_length >= 5:  # Mots longs = importants
            return 'important'
        else:
            return 'neutral'  # GRIS au lieu de blanc pur

def get_contextual_emoji(word: str, word_type: str) -> Optional[str]:
    """RÃƒÂ©cupÃƒÂ¨re un emoji contextuel pour un mot"""
    word_lower = word.lower().strip()
    
    # Chercher dans le mapping direct
    if word_lower in SUBMAGIC_EMOJI_MAP:
        return SUBMAGIC_EMOJI_MAP[word_lower]
    
    # Chercher par type de mot
    if word_type == 'action':
        action_emojis = ['Ã°Å¸â€™Âª', 'Ã°Å¸ÂÆ’', 'Ã¢Å¡Â¡', 'Ã°Å¸Å¡â‚¬', 'Ã°Å¸Å½Â¯']
        return random.choice(action_emojis)
    elif word_type == 'emphasis':
        emphasis_emojis = ['Ã¢Ââ€”', 'Ã°Å¸â€Â¥', 'Ã°Å¸â€™Â¯', 'Ã¢Å¡Â Ã¯Â¸Â', 'Ã¢Ââ€œ']
        return random.choice(emphasis_emojis)
    
    return None

def calculate_word_style(word: str, word_type: str, config: SubmagicConfig) -> Dict:
    """Calcule le style complet pour un mot - SYSTÃƒË†ME 6 COULEURS"""
    
    # Nouvelle palette ÃƒÂ©tendue
    if word_type == 'action':
        color = config.color_green        # Ã°Å¸Å¸Â¢ VERT - Actions/SuccÃƒÂ¨s
        font_size = config.font_keyword_size
    elif word_type == 'emphasis':
        color = config.color_red          # Ã°Å¸â€Â´ ROUGE - Questions/ProblÃƒÂ¨mes
        font_size = config.font_emphasis_size
    elif word_type == 'money':
        color = config.color_orange       # Ã°Å¸Å¸Â  ORANGE - Argent/BÃƒÂ©nÃƒÂ©fices
        font_size = config.font_keyword_size
    elif word_type == 'social':
        color = getattr(config, 'color_blue', (50, 150, 255))  # Ã°Å¸â€Âµ BLEU - Ãƒâ€°motions/Social
        font_size = config.font_keyword_size
    elif word_type == 'tech':
        color = getattr(config, 'color_purple', (150, 50, 255))  # Ã°Å¸Å¸Â£ VIOLET - Tech/Innovation
        font_size = config.font_keyword_size
    elif word_type == 'important':
        color = config.color_yellow       # Ã°Å¸Å¸Â¡ JAUNE - Temps/Transition
        font_size = config.font_keyword_size
    elif word_type == 'neutral':
        color = getattr(config, 'color_light_gray', (180, 180, 180))  # Gris clair au lieu de blanc
        font_size = config.font_base_size
    else:
        color = config.color_white        # Ã¢Å¡Âª BLANC - TrÃƒÂ¨s rare maintenant
        font_size = config.font_base_size
    
    return {
        'color': color,
        'font_size': font_size,
        'stroke_width': config.stroke_width,
        'stroke_color': config.stroke_color,
        'emoji': get_contextual_emoji(word, word_type) if config.emoji_enabled else None
    }

def create_submagic_frame(words_data: List[Dict], video_size: tuple, config: SubmagicConfig) -> Image.Image:
    """CrÃƒÂ©e une frame de sous-titres style Submagic"""
    width, height = video_size
    
    # Image transparente
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    if not words_data:
        return img
    
    # Calculer la disposition des mots
    total_text = ""
    styled_words = []
    
    for word_info in words_data:
        word = word_info['word']
        word_type = word_info.get('type', 'normal')
        style = calculate_word_style(word, word_type, config)
        
        styled_words.append({
            'word': word,
            'style': style,
            'emoji': style['emoji']
        })
        
        total_text += word + " "
    
    # Calculer la largeur totale et les positions
    y_position = height - int(height * config.bottom_margin)
    
    # Mesurer chaque mot pour le positionnement
    word_positions = []
    total_width = 0
    line_height = 0
    
    for word_info in styled_words:
        font = get_submagic_font(word_info['style']['font_size'])
        
        # Mesurer le mot
        bbox = draw.textbbox((0, 0), word_info['word'], font=font)
        word_width = bbox[2] - bbox[0]
        word_height = bbox[3] - bbox[1]
        
        # Ajouter espace pour emoji si prÃƒÂ©sent
        emoji_width = 0
        if word_info['emoji']:
            emoji_width = int(word_info['style']['font_size'] * config.emoji_size_ratio) + config.emoji_spacing
        
        word_positions.append({
            'word': word_info['word'],
            'style': word_info['style'],
            'emoji': word_info['emoji'],
            'width': word_width + emoji_width,
            'height': word_height,
            'font': font
        })
        
        total_width += word_width + emoji_width + 20  # Espacement entre mots
        line_height = max(line_height, word_height)
    
    # Position de dÃƒÂ©part centrÃƒÂ©e
    start_x = (width - total_width) // 2
    current_x = start_x
    
    # Dessiner chaque mot
    for word_pos in word_positions:
        word = word_pos['word']
        style = word_pos['style']
        font = word_pos['font']
        emoji = word_pos['emoji']
        
        # Position Y centrÃƒÂ©e sur la ligne
        word_y = y_position - line_height // 2
        
        # Dessiner l'emoji d'abord si prÃƒÂ©sent
        if emoji:
            emoji_size = int(style['font_size'] * config.emoji_size_ratio)
            emoji_y = word_y - emoji_size // 4  # LÃƒÂ©gÃƒÂ¨rement au-dessus
            
            # Dessiner l'emoji (simplifiÃƒÂ© - remplacer par une vraie image d'emoji si possible)
            emoji_font = get_submagic_font(emoji_size)
            draw.text((current_x, emoji_y), emoji, font=emoji_font, fill=(255, 255, 255, 255))
            current_x += emoji_size + config.emoji_spacing
        
        # Dessiner le contour du texte (effet stroke)
        stroke_width = style['stroke_width']
        stroke_color = (*style['stroke_color'], 255)
        
        for dx in range(-stroke_width, stroke_width + 1):
            for dy in range(-stroke_width, stroke_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((current_x + dx, word_y + dy), word, 
                             font=font, fill=stroke_color)
        
        # Dessiner le texte principal
        text_color = (*style['color'], 255)
        draw.text((current_x, word_y), word, font=font, fill=text_color)
        
        # Avancer la position
        current_x += word_pos['width'] + 20
    
    return img

def create_submagic_word_clip(word_data: Dict, start_time: float, end_time: float, 
                             video_size: tuple, config: SubmagicConfig) -> VideoClip:
    """CrÃƒÂ©e un clip animÃƒÂ© pour un mot style Submagic"""
    duration = end_time - start_time
    
    def make_frame(t):
        # Animation d'apparition avec bounce
        if t <= config.word_appear_duration:
            progress = t / config.word_appear_duration
            # Effet bounce (easeOutBack)
            scale = 0.3 + 0.7 * (1 + config.bounce_intensity * np.sin(progress * np.pi))
            opacity = progress
        else:
            scale = 1.0
            opacity = 1.0
        
        # CrÃƒÂ©er la frame avec le mot
        frame_img = create_submagic_frame([word_data], video_size, config)
        
        # Appliquer l'ÃƒÂ©chelle et l'opacitÃƒÂ©
        if scale != 1.0 or opacity != 1.0:
            # Redimensionner si nÃƒÂ©cessaire
            if scale != 1.0:
                new_size = (int(frame_img.width * scale), int(frame_img.height * scale))
                frame_img = frame_img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Appliquer l'opacitÃƒÂ©
            if opacity != 1.0:
                # Ajuster le canal alpha
                alpha = frame_img.split()[-1]
                alpha = alpha.point(lambda p: int(p * opacity))
                frame_img.putalpha(alpha)
        
        # CORRECTION: Convertir RGBA en RGB avec fond transparent
        rgba_array = np.array(frame_img)
        if rgba_array.shape[2] == 4:  # Si RGBA
            # Extraire RGB et alpha
            rgb = rgba_array[:, :, :3]
            alpha = rgba_array[:, :, 3] / 255.0
            
            # CrÃƒÂ©er fond transparent (noir)
            background = np.zeros_like(rgb)
            
            # Blending avec alpha
            alpha = alpha[:, :, np.newaxis]
            result = rgb * alpha + background * (1 - alpha)
            return result.astype(np.uint8)
        else:
            return rgba_array
    
    # CrÃƒÂ©er le clip
    clip = VideoClip(make_frame, duration=duration)
    clip = clip.set_start(start_time).set_position((0, 0))
    
    return clip

def parse_transcript_to_words(transcription_data: List[Dict]) -> List[Dict]:
    """Parse la transcription en mots individuels avec timing et style"""
    words_timeline = []
    
    for segment in transcription_data:
        text = segment.get('text', '').strip()
        seg_start = float(segment.get('start', 0))
        seg_end = float(segment.get('end', seg_start))
        
        # DÃƒÂ©couper en mots
        words = re.findall(r'\b\w+\b', text)
        if not words:
            continue
        
        # Calculer le timing pour chaque mot
        word_duration = (seg_end - seg_start) / len(words)
        
        for i, word in enumerate(words):
            word_start = seg_start + i * word_duration
            word_end = word_start + word_duration
            
            # DÃƒÂ©terminer le type de mot
            word_type = detect_keyword_type(word, text)
            
            words_timeline.append({
                'word': word,
                'start': word_start,
                'end': word_end,
                'type': word_type,
                'segment_text': text
            })
    
    return words_timeline

def add_submagic_subtitles(input_video_path: str, transcription_data: List[Dict], 
                          output_video_path: str, config: SubmagicConfig = None) -> str:
    """
    Ajoute des sous-titres style Submagic ÃƒÂ  une vidÃƒÂ©o
    
    Args:
        input_video_path: Chemin vers la vidÃƒÂ©o source
        transcription_data: DonnÃƒÂ©es de transcription avec timecodes
        output_video_path: Chemin de sortie
        config: Configuration Submagic (optionnel)
    
    Returns:
        str: Chemin du fichier gÃƒÂ©nÃƒÂ©rÃƒÂ©
    """
    if config is None:
        config = SubmagicConfig()
    
    print("Ã°Å¸Å½Â¬ GÃƒÂ©nÃƒÂ©ration sous-titres style Submagic...")
    
    # Charger la vidÃƒÂ©o
    video = VideoFileClip(input_video_path)
    video_size = video.size
    
    print(f"Ã°Å¸â€œÅ  VidÃƒÂ©o: {video_size[0]}x{video_size[1]}, {video.duration:.1f}s")
    
    # Parser la transcription
    words_timeline = parse_transcript_to_words(transcription_data)
    print(f"Ã°Å¸â€œÂ {len(words_timeline)} mots ÃƒÂ  animer")
    
    # CrÃƒÂ©er les clips de sous-titres
    subtitle_clips = []
    
    if config.persistence_enabled:
        # MODE HYBRIDE : Persistance + Animations bounce individuelles
        print("Ã¢Å“Â¨ Mode persistance avec animations bounce...")
        
        for i, word_data in enumerate(words_timeline):
            # Mots dÃƒÂ©jÃƒÂ  affichÃƒÂ©s (statiques)
            previous_words = words_timeline[:i] if i > 0 else []
            
            # Mot actuel (avec animation)
            current_word = word_data
            
            # DurÃƒÂ©e d'affichage : jusqu'au prochain mot ou fin
            start_time = word_data['start']
            if i < len(words_timeline) - 1:
                end_time = words_timeline[i + 1]['start']
            else:
                end_time = word_data['end']
            
            duration = end_time - start_time
            
            def make_hybrid_frame(t, prev_words=previous_words.copy(), curr_word=current_word, word_duration=duration):
                """Frame avec persistance + animation du nouveau mot"""
                
                # CrÃƒÂ©er une image vide
                img = Image.new('RGBA', video_size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                
                all_words_to_display = []
                
                # 1. Ajouter les mots prÃƒÂ©cÃƒÂ©dents (STATIQUES, sans animation)
                for prev_word in prev_words:
                    all_words_to_display.append({
                        'word': prev_word['word'],
                        'type': prev_word['type'],
                        'animated': False,  # Pas d'animation
                        'scale': 1.0,
                        'opacity': 1.0
                    })
                
                # 2. Ajouter le mot actuel (ANIMÃƒâ€° avec bounce)
                # Animation du mot actuel
                if t <= config.word_appear_duration:
                    progress = t / config.word_appear_duration
                    # Effet bounce (easeOutBack)
                    scale = 0.3 + 0.7 * (1 + config.bounce_intensity * np.sin(progress * np.pi))
                    opacity = progress
                else:
                    scale = 1.0
                    opacity = 1.0
                
                all_words_to_display.append({
                    'word': curr_word['word'],
                    'type': curr_word['type'],
                    'animated': True,  # Animation active
                    'scale': scale,
                    'opacity': opacity
                })
                
                # 3. Dessiner tous les mots avec leurs styles respectifs
                return render_mixed_words_frame(all_words_to_display, video_size, config, draw)
            
            clip = VideoClip(make_hybrid_frame, duration=duration)
            clip = clip.set_start(start_time).set_position((0, 0))
            subtitle_clips.append(clip)
    
    else:
        # Mode classique : un mot ÃƒÂ  la fois avec animations complÃƒÂ¨tes
        print("Ã¢Å“Â¨ Mode classique avec animations individuelles...")
        for word_data in words_timeline:
            clip = create_submagic_word_clip(
                word_data, word_data['start'], word_data['end'], 
                video_size, config
            )
            subtitle_clips.append(clip)
    
    # Composer la vidÃƒÂ©o finale
    print("Ã°Å¸Å½Â¨ Composition de la vidÃƒÂ©o finale...")
    
    if subtitle_clips:
        # CORRECTION: Assurer compatibilitÃƒÂ© RGB pour tous les clips
        final_video = CompositeVideoClip([video] + subtitle_clips, size=video_size)
        final_video = final_video.set_audio(video.audio)
    else:
        final_video = video
    
    # Export
    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Ã°Å¸â€™Â¾ Export vers: {output_path}")
    
    try:
        final_video.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            fps=30,
            preset="medium",
            ffmpeg_params=["-crf", "20", "-pix_fmt", "yuv420p"],
            verbose=False,
            logger=None
        )
        print("Ã¢Å“â€¦ Export Submagic terminÃƒÂ© !")
        
    except Exception as e:
        print(f"Ã¢ÂÅ’ Erreur export: {e}")
        # Fallback
        final_video.write_videofile(
            str(output_path),
            codec="libx264", 
            audio_codec="aac",
            verbose=False
        )
    
    finally:
        video.close()
        if 'final_video' in locals():
            final_video.close()
    
    return str(output_path)

def render_mixed_words_frame(words_display_data: List[Dict], video_size: tuple, config: SubmagicConfig, draw) -> np.ndarray:
    """Rend une frame avec un mÃƒÂ©lange de mots statiques et animÃƒÂ©s"""
    width, height = video_size
    
    # CrÃƒÂ©er l'image
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    if not words_display_data:
        return np.array(img)
    
    # Calculer les positions et styles
    word_positions = []
    total_width = 0
    line_height = 0
    
    for word_info in words_display_data:
        word = word_info['word']
        word_type = word_info['type']
        style = calculate_word_style(word, word_type, config)
        
        # Ajuster la taille selon l'animation
        if word_info['animated']:
            font_size = int(style['font_size'] * word_info['scale'])
        else:
            font_size = style['font_size']
        
        font = get_submagic_font(font_size)
        
        # Mesurer le mot
        bbox = draw.textbbox((0, 0), word, font=font)
        word_width = bbox[2] - bbox[0]
        word_height = bbox[3] - bbox[1]
        
        # Emoji
        emoji_width = 0
        if style['emoji']:
            emoji_width = int(font_size * config.emoji_size_ratio) + config.emoji_spacing
        
        word_positions.append({
            'word': word,
            'style': style,
            'font': font,
            'width': word_width + emoji_width,
            'height': word_height,
            'animated': word_info['animated'],
            'opacity': word_info['opacity']
        })
        
        total_width += word_width + emoji_width + 20
        line_height = max(line_height, word_height)
    
    # Position centrÃƒÂ©e
    y_position = height - int(height * config.bottom_margin)
    start_x = (width - total_width) // 2
    current_x = start_x
    
    # Dessiner chaque mot
    for word_pos in word_positions:
        word = word_pos['word']
        style = word_pos['style']
        font = word_pos['font']
        opacity = word_pos['opacity']
        
        word_y = y_position - line_height // 2
        
        # Dessiner emoji si prÃƒÂ©sent
        if style['emoji']:
            emoji_size = int(style['font_size'] * config.emoji_size_ratio)
            emoji_font = get_submagic_font(emoji_size)
            emoji_color = (*style['color'], int(255 * opacity))
            draw.text((current_x, word_y - emoji_size // 4), style['emoji'], 
                     font=emoji_font, fill=emoji_color)
            current_x += emoji_size + config.emoji_spacing
        
        # Contour
        stroke_width = style['stroke_width']
        stroke_color = (*style['stroke_color'], int(255 * opacity))
        
        for dx in range(-stroke_width, stroke_width + 1):
            for dy in range(-stroke_width, stroke_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((current_x + dx, word_y + dy), word, 
                             font=font, fill=stroke_color)
        
        # Texte principal
        text_color = (*style['color'], int(255 * opacity))
        draw.text((current_x, word_y), word, font=font, fill=text_color)
        
        current_x += word_pos['width'] + 20
    
    # Conversion RGBA vers RGB
    rgba_array = np.array(img)
    if rgba_array.shape[2] == 4:
        rgb = rgba_array[:, :, :3]
        alpha = rgba_array[:, :, 3] / 255.0
        alpha = alpha[:, :, np.newaxis]
        result = rgb * alpha
        return result.astype(np.uint8)
    else:
        return rgba_array

# Fonctions de configuration personnalisÃƒÂ©e
def create_submagic_config(
    font_base_size: int = 45,
    font_keyword_size: int = 55, 
    color_green: tuple = (50, 255, 50),
    color_red: tuple = (255, 50, 50),
    emoji_enabled: bool = True,
    persistence_enabled: bool = True
) -> SubmagicConfig:
    """CrÃƒÂ©e une configuration Submagic personnalisÃƒÂ©e"""
    config = SubmagicConfig()
    config.font_base_size = font_base_size
    config.font_keyword_size = font_keyword_size
    config.color_green = color_green
    config.color_red = color_red
    config.emoji_enabled = emoji_enabled
    config.persistence_enabled = persistence_enabled
    return config

def test_submagic_style():
    """Test du systÃƒÂ¨me Submagic avec donnÃƒÂ©es d'exemple"""
    print("Ã°Å¸Â§Âª Test style Submagic...")
    
    # DonnÃƒÂ©es de test basÃƒÂ©es sur vos images
    test_data = [
        {'text': 'AT ANY BEHAVIOR', 'start': 0.0, 'end': 2.0},
        {'text': 'I CAN\'T LIFT', 'start': 2.5, 'end': 4.0},
        {'text': 'ABOUT RUNNING OR WE\'RE', 'start': 4.5, 'end': 6.5},
        {'text': 'WHY DO WE QUIT', 'start': 7.0, 'end': 9.0},
        {'text': 'OUT THAT EVERY TIME', 'start': 9.5, 'end': 11.5}
    ]
    
    # Test de parsing
    words = parse_transcript_to_words(test_data)
    
    print(f"Ã°Å¸â€œÅ  {len(words)} mots analysÃƒÂ©s:")
    for word in words[:10]:  # Premiers 10 mots
        print(f"  '{word['word']}' ({word['type']}) {word['start']:.1f}s-{word['end']:.1f}s")
    
    # Test de dÃƒÂ©tection emojis
    test_words = ['BEHAVIOR', 'LIFT', 'RUNNING', 'WHY', 'QUIT']
    print("\nÃ°Å¸Å½Â­ Emojis dÃƒÂ©tectÃƒÂ©s:")
    for word in test_words:
        word_type = detect_keyword_type(word)
        emoji = get_contextual_emoji(word, word_type)
        print(f"  '{word}' ({word_type}) Ã¢â€ â€™ {emoji}")

def diagnose_video_issue(video_path: str) -> Dict:
    """Diagnostic rapide d'une vidÃƒÂ©o gÃƒÂ©nÃƒÂ©rÃƒÂ©e"""
    from pathlib import Path
    import os
    
    video_file = Path(video_path)
    
    diagnosis = {
        'file_exists': video_file.exists(),
        'file_size_mb': 0,
        'duration': 0,
        'issues': []
    }
    
    if video_file.exists():
        try:
            # Taille du fichier
            file_size = video_file.stat().st_size
            diagnosis['file_size_mb'] = round(file_size / (1024 * 1024), 2)
            
            # Diagnostic basique
            if file_size < 1000:  # Moins de 1KB
                diagnosis['issues'].append("Fichier trop petit (probable corruption)")
            elif file_size > 500 * 1024 * 1024:  # Plus de 500MB
                diagnosis['issues'].append("Fichier trÃƒÂ¨s volumineux")
            
            # Essayer de lire la durÃƒÂ©e avec moviepy
            try:
                from moviepy.editor import VideoFileClip
                with VideoFileClip(str(video_file)) as clip:
                    diagnosis['duration'] = round(clip.duration, 1)
                    if clip.duration < 1:
                        diagnosis['issues'].append("DurÃƒÂ©e trÃƒÂ¨s courte")
                    elif clip.duration > 300:
                        diagnosis['issues'].append("DurÃƒÂ©e trÃƒÂ¨s longue")
            except Exception as e:
                diagnosis['issues'].append(f"Erreur lecture vidÃƒÂ©o: {str(e)[:50]}")
            
        except Exception as e:
            diagnosis['issues'].append(f"Erreur analyse: {str(e)[:50]}")
    else:
        diagnosis['issues'].append("Fichier non trouvÃƒÂ©")
    
    return diagnosis

if __name__ == "__main__":
    test_submagic_style() 


