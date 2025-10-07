"""
Système de sous-titres style "Hormozi 1" - ENRICHI avec couleurs intelligentes et emojis contextuels
Corrections: taille adaptée, synchronisation audio, mots-clés colorés, positionnement exact
Intégration: SmartColorSystem + ContextualEmojiSystem
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import moviepy.editor as mp
import os
import re
import requests
import unicodedata
import logging
from typing import Dict, List, Optional, Sequence, Tuple
from pathlib import Path
from collections import deque
import json

try:
    from video_pipeline.config import SubtitleSettings, get_settings  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SubtitleSettings = None  # type: ignore
    get_settings = None  # type: ignore


logger = logging.getLogger(__name__)


def _resolve_typed_subtitle_settings() -> Optional["SubtitleSettings"]:
    if get_settings is None or SubtitleSettings is None:  # pragma: no cover - optional dependency
        return None
    try:
        settings = get_settings()
    except Exception:
        return None
    return getattr(settings, "subtitles", None)

class HormoziSubtitles:
    """Générateur de sous-titres style Hormozi avec animations et effets"""
    
    def __init__(
        self,
        subtitle_settings: Optional["SubtitleSettings"] = None,
        font_candidates: Optional[Sequence[str]] = None,
    ):
        # 🎨 Import des NOUVEAUX systèmes intelligents COMPLETS UNIQUEMENT
        try:
            from smart_color_system_complete import SmartColorSystemComplete
            from contextual_emoji_system_complete import ContextualEmojiSystemComplete
            self.smart_colors = SmartColorSystemComplete()
            self.contextual_emojis = ContextualEmojiSystemComplete()
            self.SMART_SYSTEMS_AVAILABLE = True
            print("🚀 NOUVEAUX SYSTÈMES INTELLIGENTS COMPLETS ACTIVÉS AVEC SUCCÈS !")
        except ImportError as e:
            print(f"❌ ERREUR CRITIQUE: Nouveaux systèmes non disponibles: {e}")
            print("🔧 Vérifiez que smart_color_system_complete.py et contextual_emoji_system_complete.py existent")
            self.SMART_SYSTEMS_AVAILABLE = False
            raise ImportError("Les nouveaux systèmes améliorés sont requis pour fonctionner")

        self.subtitle_settings: Optional["SubtitleSettings"] = (
            subtitle_settings or _resolve_typed_subtitle_settings()
        )
        self._font_primary: Optional[str] = None
        self._font_candidates: List[str] = []
        self._font_logged = False
        self._last_render_metadata: Dict[str, object] = {}

        # 🖼️ NOUVEAU : Système de chargement d'emojis PNG amélioré
        self.emoji_png_cache = {}
        self.emoji_mapping = {
            # 🚨 Services d'urgence
            '🚨': '1f6a8.png',      # Emergency
            '🚒': '1f692.png',      # Fire truck
            '👮‍♂️': '1f46e-200d-2642-fe0f.png',  # Police officer
            '🚑': '1f691.png',      # Ambulance
            '👨‍🚒': '1f468-200d-1f692.png',  # Male firefighter
            '👩‍🚒': '1f469-200d-1f692.png',  # Female firefighter
            
            # 🦸‍♂️ Héros et personnes
            '🦸‍♂️': '1f9b8-200d-2642-fe0f.png',  # Male hero
            '🦸‍♀️': '1f9b8-200d-2640-fe0f.png',  # Female hero
            '👥': '1f465.png',      # People
            '👤': '1f464.png',      # Person
            
            # 😠 Émotions
            '😠': '1f620.png',      # Angry
            '😡': '1f621.png',      # Pissed off
            '😤': '1f624.png',      # Triumph
            '😤': '1f624.png',      # Triumph
            
            # 🔥 Situations d'urgence
            '🔥': '1f525.png',      # Fire
            '🏠': '1f3e0.png',      # House
            '🐱': '1f431.png',      # Cat
            '🌳': '1f333.png',      # Tree
            '👶': '1f476.png',      # Baby
            '💪': '1f4aa.png',      # Biceps (force)
            '⚡': '26a1.png',       # Lightning (urgence)
            '🚨': '1f6a8.png',      # Emergency light
        }
        
        # Configuration du style Hormozi – version Montserrat virale
        self.config = {
            'font_size': 85,
            'font_color': (255, 255, 255),
            'stroke_color': (0, 0, 0),
            'stroke_px': 6,
            'position': 'bottom',
            'margin_bottom': 200,
            'line_spacing': 10,
            'animation_duration': 0.15,
            'bounce_scale': 1.3,
            'fade_duration': 0.1,
            'shadow_opacity': 0.35,
            'shadow_offset': 3,
            'emoji_scale_ratio': 0.9,
            'emoji_gap_px': 8,
            'enable_emojis': True,
            'emoji_png_only': True,
            'emoji_boost': 1.0,
            'keyword_background': False,
            'emoji_prefetch_common': True,
            'emoji_target_per_10': 5,
            'emoji_min_gap_groups': 2,
            'emoji_max_per_segment': 3,
            'emoji_no_context_fallback': "",
            'hero_emoji_enable': True,
            'hero_emoji_max_per_segment': 1,
            'emoji_history_window': 4,
            'use_twemoji_local': True,
        }
        if self.subtitle_settings is not None:
            self.config['font_size'] = int(self.subtitle_settings.font_size)
            self.config['margin_bottom'] = int(self.subtitle_settings.subtitle_safe_margin_px)
            self.config['keyword_background'] = bool(self.subtitle_settings.keyword_background)
            self.config['enable_emojis'] = bool(self.subtitle_settings.enable_emojis)
            self.config['stroke_px'] = max(0, int(self.subtitle_settings.stroke_px))
            self.config['shadow_opacity'] = float(max(0.0, self.subtitle_settings.shadow_opacity))
            self.config['shadow_offset'] = max(0, int(self.subtitle_settings.shadow_offset))
            self.config['emoji_target_per_10'] = max(0, int(self.subtitle_settings.emoji_target_per_10))
            self.config['emoji_min_gap_groups'] = max(0, int(self.subtitle_settings.emoji_min_gap_groups))
            self.config['emoji_max_per_segment'] = max(0, int(self.subtitle_settings.emoji_max_per_segment))
            self.config['emoji_no_context_fallback'] = str(self.subtitle_settings.emoji_no_context_fallback or "")
            self.config['hero_emoji_enable'] = bool(self.subtitle_settings.hero_emoji_enable)
            self.config['hero_emoji_max_per_segment'] = max(0, int(self.subtitle_settings.hero_emoji_max_per_segment))
            if getattr(self.subtitle_settings, 'font', None):
                preferred_font = str(self.subtitle_settings.font)
                if preferred_font:
                    self.config['preferred_font_name'] = preferred_font

        self._font_candidates = self._build_font_candidates(font_candidates)
        # Presets de marque (brand kits)
        self.brand_presets = {
            'default': {'font_size': 85, 'outline_color': (0,0,0), 'outline_width': 4},
            'clean_white': {'font_size': 80, 'outline_color': (0,0,0), 'outline_width': 3},
            'yellow_pop': {'font_size': 90, 'outline_color': (20,20,20), 'outline_width': 5},
        }
        
        palette = {
            'finance': '#FFD700',
            'business': '#00E5FF',
            'sales': '#FF8C00',
            'content': '#FF1493',
            'actions': '#FF8C00',
            'success': '#32CD32',
            'urgency': '#FF8C00',
            'emotions': '#FF1493',
            'tech': '#8A2BE2',
            'mobile': '#00E5FF',
            'personal': '#32CD32',
            'solutions': '#00E5FF',
            'problems': '#FF8C00',
            'health': '#32CD32',
            'sports': '#8A2BE2',
            'education': '#8A2BE2',
        }
        aliases = {
            'money': 'finance',
            'investment': 'finance',
            'profit': 'finance',
            'wealth': 'finance',
            'revenue': 'finance',
            'corporate': 'business',
            'strategy': 'business',
            'leadership': 'business',
            'team': 'business',
            'marketing': 'content',
            'creative': 'content',
            'action': 'actions',
            'energy': 'actions',
            'power': 'actions',
            'movement': 'actions',
            'victory': 'success',
            'achievement': 'success',
            'winning': 'success',
            'time': 'urgency',
            'deadline': 'urgency',
            'pressure': 'urgency',
            'attention': 'urgency',
            'important': 'urgency',
            'critical': 'urgency',
            'stop': 'urgency',
            'urgent': 'urgency',
            'passion': 'emotions',
            'excitement': 'emotions',
            'inspiration': 'emotions',
            'digital': 'tech',
            'innovation': 'tech',
            'future': 'tech',
            'mindset': 'personal',
            'growth': 'personal',
            'learning': 'personal',
            'fix': 'solutions',
            'resolve': 'solutions',
            'improve': 'solutions',
            'challenges': 'problems',
            'obstacles': 'problems',
            'difficulties': 'problems',
            'wellness': 'health',
            'fitness': 'health',
            'mindfulness': 'health',
        }
        self.category_colors: Dict[str, str] = {}
        for key, value in palette.items():
            self.category_colors[key] = value
        for alias, target in aliases.items():
            self.category_colors[alias] = palette[target]
        
        base_emojis: Dict[str, List[str]] = {
            'finance': ['💰', '💸', '📈', '🤑', '🏦', '💳'],
            'business': ['💼', '📊', '📈', '🤝', '🏢', '📝'],
            'sales': ['🛒', '🤝', '💳', '📦', '📈', '💸'],
            'content': ['🎬', '📝', '📹', '🎧', '🎨', '📱'],
            'actions': ['⚡', '🚀', '🔥', '💪', '🏃', '💥'],
            'success': ['🏆', '🎯', '🌟', '✅', '🎉', '💯'],
            'urgency': ['⏰', '⚠️', '🚨', '⏳', '❗', '🕒'],
            'emotions': ['🔥', '🤯', '😍', '✨', '😁', '🥳'],
            'tech': ['🤖', '💻', '🧠', '🛰️', '🔌', '📡'],
            'mobile': ['📱', '📲', '💬', '🕹️', '📟', '📶'],
            'personal': ['🧠', '💡', '🧘', '📚', '📝', '🎯'],
            'solutions': ['✅', '🧩', '🔑', '🛠️', '💡', '🔧'],
            'problems': ['⚠️', '❌', '🛑', '💣', '🤔', '😬'],
            'health': ['❤️', '💪', '🥗', '🧘', '🏥', '🩺'],
            'sports': ['🏀', '⚽', '🏃', '💪', '🎽', '🥇'],
            'education': ['📚', '🎓', '✏️', '🧠', '📖', '📝'],
        }
        self.category_emojis: Dict[str, List[str]] = {
            key: list(values) for key, values in base_emojis.items()
        }
        for alias, target in aliases.items():
            if target in base_emojis:
                self.category_emojis[alias] = list(base_emojis[target])
        
        # Dictionnaire mots-clés -> catégorie (liste élargie de synonymes/variations)
        self.keyword_to_category: Dict[str, str] = {}
        self._bootstrap_categories()
        # Charger un lexique externe optionnel pour enrichir alias/catégories/émoticônes
        try:
            self._load_external_emoji_lexicon(Path('config/emoji_lexicon.json'))
        except Exception:
            pass
        
        # Alias supplémentaires (FR/EN) pour améliorer la couverture sémantique → catégorie
        self.emoji_alias: Dict[str, str] = {
            # Finance
            'ARGENT':'finance','EURO':'finance','EUROS':'finance','REVENU':'finance','REVENUS':'finance','BENEFICE':'finance','BENEFICES':'finance','VENTE':'finance','VENTES':'finance','ACHAT':'finance','ACHETER':'finance','PRICE':'finance','PRICING':'finance','CASH':'finance','MONEY':'finance','PROFIT':'finance','REVENUE':'finance','WEALTH':'finance','BUDGET':'finance','INVEST':'finance','INVESTIR':'finance','ROI':'finance','LTV':'finance','AOV':'finance','BITCOIN':'finance','CRYPTO':'finance','ETHEREUM':'finance',
            # Success/Growth
            'SUCCES':'success','SUCCESS':'success','WIN':'success','VICTOIRE':'success','RESULTAT':'success','RESULTATS':'success','GROWTH':'success','GROW':'success','SCALE':'success','EXPAND':'success','PERF':'success','PERFORMANCE':'success','RECORD':'success','TOP':'success','BEST':'success','MILLION':'finance','MILLIONS':'finance',
            # Actions/Urgency
            'FAST':'urgency','QUICK':'urgency','RAPIDE':'urgency','IMMEDIAT':'urgency','NOW':'urgency','TODAY':'urgency','MAINTENANT':'urgency','VITE':'urgency','HURRY':'urgency','DEADLINE':'urgency','URGENT':'urgency','ACTION':'actions','ACT':'actions','BUILD':'actions','CREATE':'actions','LAUNCH':'actions','START':'actions','IMPLEMENT':'actions','EXECUTE':'actions','OPTIMIZE':'actions','IMPROVE':'actions',
            # Business/Team/Client
            'BUSINESS':'business','ENTREPRISE':'business','SOCIETE':'business','COMPANY':'business','TEAM':'business','CLIENT':'business','CUSTOMER':'business','BRAND':'business','MARKETING':'business','STRATEGY':'business','SYSTEM':'business','PROCESS':'business',
            # Emotions
            'WOW':'emotions','INCROYABLE':'emotions','AMAZING':'emotions','INCREDIBLE':'emotions','FIRE':'emotions','🔥':'emotions','CRAZY':'emotions','INSANE':'emotions','MOTIVATION':'emotions','ENERGY':'emotions','PASSION':'emotions','LOVE':'emotions','❤️':'emotions',
            # Tech
            'AI':'tech','IA':'tech','AUTOMATION':'tech','ALGORITHME':'tech','ALGORITHM':'tech','CODE':'tech','SOFTWARE':'tech','APP':'tech','DIGITAL':'tech','API':'tech','CLOUD':'tech','DATA':'tech',
            # Problems/Solutions
            'PROBLEME':'problems','PROBLEM':'problems','ISSUE':'problems','ERREUR':'problems','FAIL':'problems','BUG':'problems','SOLUTION':'solutions','SOLUTIONS':'solutions','FIX':'solutions','PATCH':'solutions','HOWTO':'solutions','SECRET':'solutions','TIP':'solutions','TRICK':'solutions',
            # Personal/Health
            'BRAIN':'personal','MENTAL':'personal','NEUROSCIENCE':'personal','DOPAMINE':'personal','ANXIETY':'personal','STRESS':'personal','TRAUMA':'personal','MINDSET':'personal','DISCIPLINE':'personal','HABITS':'personal','GOALS':'personal','FOCUS':'personal','PRODUCTIVITY':'personal','EXERCISE':'personal','MOVEMENT':'personal'
        }

        self._hero_triggers: Dict[str, Sequence[str]] = {
            '🔥': ('OFFER', 'OFFRE', 'DEAL'),
            '⚡': ('ENERGY', 'ENERGIE', 'POWER'),
            '💰': ('PROFIT', 'PROFITS', 'MONEY', 'ARGENT', 'CASH', 'REVENU', 'REVENUE'),
        }
        
        # Mémoire pour éviter la répétition immédiate d'un même emoji
        self._last_emoji: str = ""
        history_window = max(1, int(self.config.get('emoji_history_window', 4)))
        self._recent_emojis = deque(maxlen=history_window)
        self._global_group_index = 0
        self._last_emoji_global_index = -999
        # Mémoire pour lisser la position verticale des sous-titres
        self._y_ema: float | None = None
        self._line_h_ema: float | None = None

        # Détection visage (placement intelligent): initialiser un cascade si dispo
        self._face_cascade = None
        try:
            import cv2 as _cv
            cascade_path = getattr(_cv.data, 'haarcascades', '') + 'haarcascade_frontalface_default.xml'
            if cascade_path and os.path.exists(cascade_path):
                self._face_cascade = _cv.CascadeClassifier(cascade_path)
        except Exception:
            self._face_cascade = None
        
        # Mappings finaux
        self.keyword_colors: Dict[str, str] = {}
        self.emoji_mapping: Dict[str, str] = {}
        for kw, cat in self.keyword_to_category.items():
            self.keyword_colors[kw] = self.category_colors.get(cat, '#FFFFFF')
            if cat in self.category_emojis and self.category_emojis[cat]:
                self.emoji_mapping[kw] = self.category_emojis[cat][0]
        
        # Préchargement d'emojis fréquents (PNG) pour éviter latences
        if self.config.get('emoji_prefetch_common', False) and self.config.get('enable_emojis', False):
            common = ['🔥','💸','🚀','💼','📈','🏆','⏳','⚡','✅','💯']
            for ch in common:
                try:
                    self._load_emoji_png(ch, 64)
                except Exception:
                    pass

    def _normalize(self, s: str) -> str:
        """Supprime les accents et met en MAJUSCULE pour comparaison robuste."""
        if not s:
            return ""
        s = str(s)
        s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        s = re.sub(r"[^A-Za-z0-9']+", '', s)
        return s.upper()

    def _get_category_for_word(self, word: str):
        """Retourne la config de catégorie (couleur/emoji) si le mot appartient à une catégorie FR/EN."""
        word_norm = self._normalize(word)
        # Définition FR « Hormozi 1 »
        self.keyword_categories = getattr(self, 'keyword_categories', None) or {
            "MONEY": {
                "words": ["ARGENT","EUROS","DOLLARS","REVENU","CHER","COUT","INVESTIR","BENEFICE","VENDRE","ACHETER"],
                "color": self.category_colors['finance'],
                "emoji": '💰'
            },
            "ACTION": {
                "words": ["CREER","DETRUIRE","MULTIPLIER","AUGMENTER","ECRASER","TRANSFORMER","POUSSER"],
                "color": self.category_colors['actions'],
                "emoji": '⚡'
            },
            "RESULT": {
                "words": ["SUCCES","RESULTAT","GAGNER","VICTOIRE","SOMMET","LEADER","NUMERO","TOP"],
                "color": self.category_colors['success'],
                "emoji": '🏆'
            },
            "TIME": {
                "words": ["HEURE","TEMPS","JOUR","MINUTE","RAPIDE","VITE","IMMEDIAT","AUJOURDHUI"],
                "color": self.category_colors['urgency'],
                "emoji": '⏳'
            },
            "EMOTION": {
                "words": ["PEUR","MOTIVATION","CROYANCE","PASSION","DETERMINATION","ENERGIE","AMOUR"],
                "color": self.category_colors['emotions'],
                "emoji": '❤️'
            }
        }
        for cat, data in self.keyword_categories.items():
            for w in data.get("words", []):
                if word_norm == self._normalize(w):
                    return data
        return None

    def parse_transcription_to_word_groups(self, transcription_data: List[Dict], group_size: int = 2) -> List[Dict]:
        """
        Parse la transcription en groupes de mots (2–3) style Hormozi 1.
        Chaque groupe est stylisé; seul le premier mot-clé est coloré, les autres restent blancs.
        Gère aussi le cas SRT (pas de mots horodatés) en répartissant le temps uniformément.
        """
        words: List[Dict] = []
        for segment in transcription_data:
            seg_text = (segment.get("text") or "").strip()
            seg_start = float(segment.get("start", 0.0))
            seg_end = float(segment.get("end", seg_start))
            if seg_end < seg_start:
                seg_end = seg_start
            duration = max(0.01, seg_end - seg_start)
            raw_words = segment.get("words", [])
            if not raw_words:
                # Fallback SRT: découper le texte en tokens alphanumériques et répartir le temps
                tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+", seg_text)
                if not tokens:
                    continue
                step = duration / max(1, len(tokens))
                cur = seg_start
                raw_words = []
                for tok in tokens:
                    w_start = cur
                    w_end = min(seg_end, w_start + step)
                    raw_words.append({"text": tok, "start": w_start, "end": w_end})
                    cur = w_end
            # Regroupement intelligent par ponctuation/respiration
            # Découpe en unités au niveau ponctuation forte ; sinon groupe de 2 mots
            boundaries = []
            try:
                text_lower = seg_text.lower()
                for idx, w in enumerate(raw_words):
                    word_txt = (w.get('word') or w.get('text') or '').strip()
                    if any(p in word_txt for p in [',', ';', ':']) and idx > 0:
                        boundaries.append(idx)
            except Exception:
                pass
            def next_cut(i:int)->int:
                if i+2 <= len(raw_words):
                    return min(i+2, len(raw_words))
                return len(raw_words)
            i = 0
            segment_groups: List[Dict] = []
            while i < len(raw_words):
                j = next_cut(i)
                if i in boundaries:
                    j = i+1
                chunk = raw_words[i:j]
                i = j
                start_time = float(chunk[0].get("start", seg_start))
                end_time = float(chunk[-1].get("end", seg_end))
                tokens = []
                group_categories: List[str] = []
                candidate_emojis: List[str] = []
                hero_candidates: List[str] = []
                has_keyword = False
                colored_quota = 3
                colored_used = 0
                linking_words = {
                    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
                    'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
                    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may', 'might',
                    'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their', 'we',
                    'us', 'our', 'you', 'your', 'i', 'me', 'my'
                }
                token_norms: List[str] = []
                for w in chunk:
                    base = (str(w.get("word") or w.get("text") or "").strip())
                    clean = self._normalize(base)
                    color_hex = "#FFFFFF"
                    is_keyword = False
                    category: Optional[str] = None
                    if display_base := base.strip():
                        display_text = display_base.upper()
                    else:
                        display_text = ""
                    if not display_text:
                        continue
                    if (colored_used < colored_quota and clean.lower() not in linking_words):
                        category = self.keyword_to_category.get(clean)
                        if not category and clean in self.emoji_alias:
                            category = self.emoji_alias[clean]
                        if not category:
                            cat_data = self._get_category_for_word(base)
                            category = self._category_from_color(cat_data.get("color") if cat_data else None)
                        if category:
                            color_hex = self.category_colors.get(category, "#FFFFFF")
                            is_keyword = True
                            has_keyword = True
                            colored_used += 1
                            if category not in group_categories:
                                group_categories.append(category)
                            if category in self.category_emojis:
                                candidate_emojis.extend(self.category_emojis[category])
                    display_text = base.upper()
                    tokens.append({
                        "text": display_text,
                        "normalized": clean,
                        "is_keyword": is_keyword,
                        "color": color_hex,
                        "category": category,
                    })
                    if clean:
                        token_norms.append(clean)
                candidate_emojis = list(dict.fromkeys(candidate_emojis))
                token_norm_set = set(token_norms)
                for emoji_char, triggers in self._hero_triggers.items():
                    for trigger in triggers:
                        if self._normalize(trigger) in token_norm_set:
                            hero_candidates.append(emoji_char)
                            break
                chunk_text = " ".join(t["text"] for t in tokens)
                segment_groups.append({
                    "text": chunk_text,
                    "original": chunk_text,
                    "start": start_time,
                    "end": end_time,
                    "is_keyword": any(t["is_keyword"] for t in tokens),
                    "color": "#FFFFFF",
                    "emoji": "",
                    "tokens": tokens,
                    "emojis": [],
                    "categories": group_categories,
                    "candidate_emojis": candidate_emojis,
                    "hero_candidates": list(dict.fromkeys(hero_candidates)),
                    "has_keyword": has_keyword,
                })
            if segment_groups:
                self._plan_emojis_for_segment(segment_groups)
                words.extend(segment_groups)
        return words

    def export_tokens_json(self, groups: List[Dict], out_path: str) -> None:
        try:
            data = []
            for g in groups:
                data.append({
                    'text': g.get('text',''),
                    'start': float(g.get('start',0.0)),
                    'end': float(g.get('end',0.0)),
                    'tokens': g.get('tokens',[]),
                    'emojis': g.get('emojis',[]),
                })
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convertit couleur hex en RGB"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _category_from_color(self, color_hex: Optional[str]) -> Optional[str]:
        if not color_hex:
            return None
        target = str(color_hex).lower()
        for key, value in self.category_colors.items():
            if str(value).lower() == target:
                return key
        return None

    def _load_emoji_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Charge une police emoji système (Segoe UI Emoji/Noto) pour fallback texte."""
        candidates = [
            "C:/Windows/Fonts/seguiemj.ttf",
            "C:/Windows/Fonts/seguiui.ttf",
            "/System/Library/Fonts/Apple Color Emoji.ttc",
            "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        ]
        for p in candidates:
            try:
                if os.path.exists(p):
                    return ImageFont.truetype(p, size)
            except Exception:
                continue
        return ImageFont.load_default()

    def _build_font_candidates(self, extra: Optional[Sequence[str]]) -> List[str]:
        def _push(target: Optional[str | Path]) -> None:
            if not target:
                return
            try:
                path_obj = Path(str(target)).expanduser()
            except Exception:
                return
            self._font_candidates.append(str(path_obj))

        self._font_candidates = []
        if extra:
            for candidate in extra:
                if candidate:
                    _push(candidate)
        if self.subtitle_settings is not None and getattr(self.subtitle_settings, "font_path", None):
            _push(self.subtitle_settings.font_path)

        base_dir = Path(__file__).resolve().parent
        asset_fonts = [
            base_dir / "assets" / "fonts" / "Montserrat-ExtraBold.ttf",
            base_dir / "assets" / "fonts" / "Montserrat-Bold.ttf",
        ]
        for candidate in asset_fonts:
            _push(candidate)

        fallback_fonts = [
            "/System/Library/Fonts/Montserrat-ExtraBold.ttf",
            "/System/Library/Fonts/Montserrat-Bold.ttf",
            "/Library/Fonts/Montserrat-ExtraBold.ttf",
            "/Library/Fonts/Montserrat-Bold.ttf",
            "C:/Windows/Fonts/Montserrat-ExtraBold.ttf",
            "C:/Windows/Fonts/Montserrat-Bold.ttf",
            "C:/Windows/Fonts/impact.ttf",
            "/Windows/Fonts/impact.ttf",
            "/System/Library/Fonts/Impact.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/arial.ttf",
        ]
        for candidate in fallback_fonts:
            _push(candidate)

        ordered_unique = list(dict.fromkeys(self._font_candidates))
        preferred_name = str(self.config.get('preferred_font_name', '') or '').lower()
        if preferred_name:
            ordered_unique.sort(
                key=lambda path: (0 if preferred_name in Path(path).name.lower() else 1, Path(path).name.lower())
            )
        self._font_candidates = ordered_unique
        return ordered_unique

    def _resolve_font_path(self) -> Optional[str]:
        if self._font_primary is not None:
            return self._font_primary
        for candidate in self._font_candidates:
            try:
                if Path(candidate).expanduser().exists():
                    self._font_primary = str(Path(candidate).expanduser())
                    if not self._font_logged:
                        logger.info("[Subtitles] Using font: %s", self._font_primary)
                        self._font_logged = True
                    return self._font_primary
            except OSError:
                continue
        self._font_primary = None
        return None

    def _invalidate_font_candidate(self, path: str) -> None:
        self._font_primary = None
        self._font_logged = False
        normalized: List[str] = []
        for candidate in self._font_candidates:
            try:
                if Path(candidate).resolve() == Path(path).resolve():
                    continue
            except Exception:
                if candidate == path:
                    continue
            normalized.append(candidate)
        self._font_candidates = normalized

    def get_font_path(self) -> Optional[str]:
        return self._resolve_font_path()

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont:
        tried: set[str] = set()
        while True:
            path = self._resolve_font_path()
            if not path or path in tried:
                break
            try:
                return ImageFont.truetype(path, int(size))
            except Exception:
                tried.add(path)
                self._invalidate_font_candidate(path)
        return ImageFont.load_default()

    def _load_emoji_png(self, emoji_char: str, target_h: int) -> Image.Image | None:
        """Charge un emoji PNG depuis emoji_assets/<codepoint>.png; télécharge via Twemoji si manquant et PNG-only."""
        try:
            if not emoji_char:
                return None
            assets_dir = Path("emoji_assets"); assets_dir.mkdir(parents=True, exist_ok=True)
            # Support simple et séquences: joindre les codepoints par '-'
            codepoints = "-".join([f"{ord(ch):x}" for ch in emoji_char])
            img_path = assets_dir / f"{codepoints}.png"
            if not img_path.exists() and self.config.get('emoji_png_only', False) and self.config.get('use_twemoji_local', True):
                # Tentative de téléchargement Twemoji
                urls = [
                    f"https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/assets/72x72/{codepoints}.png",
                    f"https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/{codepoints}.png",
                ]
                for url in urls:
                    try:
                        r = requests.get(url, timeout=10)
                        r.raise_for_status()
                        with open(img_path, 'wb') as f:
                            f.write(r.content)
                        break
                    except Exception:
                        continue
            if not img_path.exists():
                return None
            img = Image.open(str(img_path)).convert("RGBA")
            if target_h > 0:
                w, h = img.size
                ratio = float(target_h) / float(h)
                new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            return img
        except Exception:
            return None

    def get_active_words_at_time(self, words: List[Dict], current_time: float) -> List[Dict]:
        """Retourne les groupes actifs et injecte animation_progress si absent."""
        active: List[Dict] = []
        for w in words:
            try:
                start = float(w.get('start', 0.0))
                end = float(w.get('end', 0.0))
                if start <= current_time <= end:
                    duration = max(1e-6, end - start)
                    time_in = max(0.0, current_time - start)
                    anim = min(time_in / max(1e-6, self.config['animation_duration']), 1.0)
                    wc = dict(w)
                    wc['animation_progress'] = float(anim)
                    active.append(wc)
            except Exception:
                continue
        return active

    def _bootstrap_categories(self) -> None:
        """Initialise les catégories et assigne un grand nombre de mots-clés."""
        cat = {}
        # Business
        cat['business'] = [
            'BUSINESS','COMPANY','STARTUP','STRATEGY','SYSTEM','FRAMEWORK','METHOD','PROCESS','MODEL',
            'MARKETING','ADVERTISING','BRANDING','CONTENT','COPYWRITING','FUNNEL','CHECKOUT','CONVERSION',
            'CUSTOMER','CLIENT','TEAM','LEADERSHIP','MANAGEMENT','COACHING','MENTOR','INFLUENCE',
            'ECOMMERCE','SHOPIFY','DROPSHIP','PORTFOLIO','KPI','METRICS','ANALYTICS','RETENTION','UPSELL','CROSSSELL'
        ]
        # Marketing / Sales (plus précis)
        cat['sales'] = ['SALES','SELL','CLOSE','DEAL','NEGOCIATE','NEGOCIER','PRICING','PRICE','QUOTE','PANIER','CART','CONVERSION','LEAD','PIPELINE']
        cat['content'] = ['VIDEO','TITRE','THUMB','THUMBNAIL','HOOK','RETENTION','SCRIPTS','SCRIPT','EDIT','CUT','SUBTITLE','SUBTITLES','CAPTION']
        # Emotions (fortes)
        cat['emotions'] = [
            'FIRE','INSANE','CRAZY','AMAZING','INCREDIBLE','UNBELIEVABLE','MINDBLOWING','EXPLOSIVE',
            'WOW','PASSION','ENERGY','VIBES','HYPE'
        ]
        # Actions
        cat['actions'] = [
            'BUILD','CREATE','LAUNCH','START','IMPLEMENT','EXECUTE','LEARN','APPLY','GROW','SCALE','EXPAND',
            'MOVE','ACT','DO','MAKE','SHIP','OPTIMIZE','IMPROVE','TEST','ITERATE'
        ]
        # Urgency
        cat['urgency'] = [
            'IMPORTANT','CRITICAL','URGENT','MUST','NEED','REQUIRED','ESSENTIAL','CRUCIAL','VITAL','NOW','IMMEDIATE','FAST','QUICK'
        ]
        # Succès
        cat['success'] = [
            'SUCCESS','WIN','WINNER','VICTORY','CHAMPION','BEST','PERFECT','CRUSHING','DOMINATING','ACHIEVE','RESULTS'
        ]
        # Problems
        cat['problems'] = [
            'PROBLEM','PROBLEMS','ISSUE','ISSUES','PAIN','BLOCKER','OBSTACLE','FAIL','FAILING','FAILURE','MISTAKE','ERROR'
        ]
        # Solutions
        cat['solutions'] = [
            'SOLUTION','SOLUTIONS','FIX','PATCH','ANSWER','HOWTO','SECRET','TIP','TRICK','METHOD','SYSTEMATIC'
        ]
        # Finance
        cat['finance'] = [
            'MONEY','CASH','DOLLARS','PROFIT','REVENUE','INCOME','WEALTH','RICH','MILLION','MILLIONS',
            'BILLION','BILLIONS','BANK','PAY','ROI','LTV','AOV','PRICING','PRICE'
        ]
        # Tech
        cat['tech'] = [
            'AI','CHATGPT','OPENAI','MACHINE','LEARNING','AUTOMATION','ALGORITHM','DATA','DIGITAL','SOFTWARE',
            'CODING','PROGRAMMING','PYTHON','JAVASCRIPT','API','DATABASE','CLOUD','DEVOPS','LLM'
        ]
        cat['mobile'] = ['APP','APPLICATION','MOBILE','PHONE','SMARTPHONE','ANDROID','IOS']
        # Personnel / Santé / Fitness / Neuro
        cat['personal'] = [
            'NEUROSCIENCE','DOPAMINE','SEROTONIN','PSYCHOLOGY','MINDSET','MENTAL','BRAIN','NEURAL','ANXIETY','STRESS',
            'PANIC','DEPRESSION','TRAUMA','PTSD','BURNOUT','MOVEMENT','EXERCISE','MOBILITY','STRENGTH','CARDIO','FLEXIBILITY',
            'RECOVERY','PERFORMANCE','NUTRITION','PROTEIN','CARBS','FASTING','METABOLISM','CALORIES','SUPPLEMENTS',
            'MOTIVATION','DISCIPLINE','HABITS','GOALS','FOCUS','PRODUCTIVITY','MINDFULNESS','MEDITATION'
        ]
        # Santé
        cat['health'] = ['HEALTH','SANTE','DIET','NUTRITION','SLEEP','SOMMEIL','REST','RECOVERY','STRESS','ANXIETY','THERAPY','THERAPIE']
        # Sport
        cat['sports'] = ['SPORT','GYM','FITNESS','RUN','RUNNING','MARATHON','SWIM','SWIMMING','CYCLE','CYCLING','FOOTBALL','SOCCER','BASKET','TENNIS']
        # Education
        cat['education'] = ['LEARN','LEARNING','EDUCATION','SCHOOL','UNIVERSITY','COURSE','COURS','LESSON','STUDY','STUDIES','MENTOR','COACH']
        # Temps / Planning
        cat['time'] = ['TIME','TEMPS','DEADLINE','SCHEDULE','PLANNING','CALENDAR','WEEK','MONTH','YEAR','TODAY','TOMORROW']
        # Enregistrer
        for category, words in cat.items():
            for w in words:
                self.keyword_to_category[w] = category

    def _load_external_emoji_lexicon(self, path: Path) -> None:
        """Charge un lexique externe JSON et fusionne: catégories, alias, emojis, mots-clés.
        Format attendu (tous facultatifs):
        {
          "category_emojis": {"category": ["🔥","..."]},
          "emoji_alias": {"WORD":"category"},
          "keyword_to_category": {"WORD":"category"},
          "categories": {"category": ["WORD1","WORD2"]}
        }
        """
        try:
            if not path.exists():
                return
            data = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(data, dict):
                # Emojis par catégorie
                ce = data.get('category_emojis') or {}
                if isinstance(ce, dict):
                    for k,v in ce.items():
                        if isinstance(v, list) and v:
                            self.category_emojis[k] = list(dict.fromkeys((self.category_emojis.get(k, []) + v)))
                # Alias mots -> catégorie
                ea = data.get('emoji_alias') or {}
                if isinstance(ea, dict):
                    for k, v in ea.items():
                        if isinstance(k, str) and isinstance(v, str):
                            self.emoji_alias[self._normalize(k)] = v
                # Mots-clés -> catégorie
                km = data.get('keyword_to_category') or {}
                if isinstance(km, dict):
                    for k, v in km.items():
                        if isinstance(k, str) and isinstance(v, str):
                            self.keyword_to_category[self._normalize(k)] = v
                # Catégories supplémentaires
                cat = data.get('categories') or {}
                if isinstance(cat, dict):
                    for cat_name, words in cat.items():
                        if isinstance(words, list):
                            for w in words:
                                if isinstance(w, str):
                                    self.keyword_to_category[self._normalize(w)] = cat_name
        except Exception:
            pass

    def _choose_emoji_for_tokens(self, tokens: List[Dict], group_text: str) -> str:
        """Choisit un emoji contextuel à partir des catégories déjà évaluées."""
        candidates: List[str] = []
        for token in tokens:
            category = token.get('category')
            if category and category in self.category_emojis:
                candidates.extend(self.category_emojis[category])
        unique_candidates = [c for c in dict.fromkeys(candidates) if c]
        emoji = self._select_from_candidates(unique_candidates, group_text, avoid=self._recent_emojis)
        if not emoji:
            fallback = str(self.config.get('emoji_no_context_fallback') or "")
            return fallback
        return emoji

    def _select_from_candidates(
        self,
        candidates: Sequence[str],
        group_text: str,
        avoid: Optional[Sequence[str]] = None,
    ) -> str:
        if not candidates:
            return ""
        avoid_set = {c for c in (avoid or []) if c}
        pool = [c for c in candidates if c and c not in avoid_set]
        if not pool:
            pool = [c for c in candidates if c]
        if not pool:
            return ""
        seed = abs(hash((group_text or "", tuple(pool))))
        return pool[seed % len(pool)]

    def _select_group_emoji(self, group: Dict) -> str:
        emoji = self._select_from_candidates(
            group.get('candidate_emojis') or [],
            group.get('text', ""),
            avoid=self._recent_emojis,
        )
        if emoji:
            return emoji
        fallback = str(self.config.get('emoji_no_context_fallback') or "")
        if fallback and fallback not in self._recent_emojis:
            return fallback
        return ""

    def _select_hero_emoji(self, candidates: Sequence[str]) -> str:
        if not candidates:
            return ""
        for candidate in candidates:
            if candidate and candidate not in self._recent_emojis:
                return candidate
        return candidates[0] if candidates and candidates[0] not in {""} else ""

    def _plan_emojis_for_segment(self, groups: List[Dict]) -> None:
        if not groups or not self.config.get('enable_emojis', False):
            return
        target = max(0, int(round(len(groups) * float(self.config.get('emoji_target_per_10', 0)) / 10.0)))
        target = min(target, int(self.config.get('emoji_max_per_segment', target)))
        if target <= 0:
            return
        min_gap = max(0, int(self.config.get('emoji_min_gap_groups', 0)))
        hero_remaining = int(self.config.get('hero_emoji_max_per_segment', 0)) if self.config.get('hero_emoji_enable', True) else 0
        placements = 0
        for idx, group in enumerate(groups):
            group['emojis'] = []
            global_index = self._global_group_index + idx
            if placements >= target:
                continue
            if global_index - self._last_emoji_global_index < min_gap:
                continue
            hero_emoji = ""
            if hero_remaining > 0:
                hero_emoji = self._select_hero_emoji(group.get('hero_candidates') or [])
            if hero_emoji:
                emoji = hero_emoji
                hero_remaining -= 1
            else:
                emoji = self._select_group_emoji(group)
            if not emoji:
                continue
            group['emojis'] = [emoji]
            placements += 1
            self._last_emoji_global_index = global_index
            self._recent_emojis.append(emoji)
        self._global_group_index += len(groups)

    def create_subtitle_frame(self, frame: np.ndarray, words: List[Dict], 
                              current_time: float) -> np.ndarray:
        """Crée une frame avec sous-titres overlay (coloration du mot-clé; emojis en fin de groupe)."""
        height, width = frame.shape[:2]
        if words and isinstance(words, list) and isinstance(words[0], dict) and ("animation_progress" in words[0]):
            active_words = words
        else:
            active_words = self.get_active_words_at_time(words, current_time)
        if not active_words:
            return frame
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        items = []
        total_w = 0
        max_h = 0
        group_meta: List[Dict[str, object]] = []
        for wobj in active_words:
            prog = float(wobj.get('animation_progress', 1.0))
            scale = 1.0 + (self.config['bounce_scale'] - 1.0) * (1.0 - min(prog,1.0))
            fsize = max(1, int(self.config['font_size'] * scale))
            font = self._load_font(fsize)
            group_idx = len(group_meta)
            group_meta.append({
                'emojis': list(wobj.get('emojis') or []),
                'font_size': fsize,
                'max_height': 0,
            })
            tokens = wobj.get("tokens") if isinstance(wobj, dict) else None
            # 1) Mots
            if tokens:
                for j, tok in enumerate(tokens):
                    ttext = (tok.get("text") or "").strip()
                    bbox = draw.textbbox((0, 0), ttext, font=font)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    color_hex = tok.get("color", "#FFFFFF")
                    is_keyword = bool(tok.get("is_keyword"))
                    base_rgb = self.hex_to_rgb(color_hex) if is_keyword else (255, 255, 255)
                    bg_rgb = None
                    pad_x = 0
                    pad_y = 0
                    text_rgb = base_rgb if is_keyword and not self.config.get('keyword_background', False) else (255, 255, 255)
                    if is_keyword and self.config.get('keyword_background', False):
                        bg_rgb = base_rgb
                        pad_x = max(4, int(fsize * 0.18))
                        pad_y = max(2, int(fsize * 0.12))
                    w_total = tw + pad_x * 2
                    h_total = th + pad_y * 2
                    items.append({
                        'type': 'word',
                        'text': ttext,
                        'font': font,
                        'w': w_total,
                        'h': h_total,
                        'rgb': text_rgb,
                        'prog': prog,
                        'fs': fsize,
                        'bg_rgb': bg_rgb,
                        'pad_x': pad_x,
                        'pad_y': pad_y,
                        'keyword': is_keyword,
                        'color_hex': color_hex,
                        'group_index': group_idx,
                    })
                    total_w += w_total
                    max_h = max(max_h, h_total)
                    group_meta[group_idx]['max_height'] = max(group_meta[group_idx]['max_height'], h_total)
                    group_meta[group_idx]['font_size'] = max(group_meta[group_idx]['font_size'], fsize)
                    if j < len(tokens) - 1:
                        # Espace basé sur la taille de police actuelle
                        try:
                            space_w = int(max(1, draw.textlength(" ", font=font)))
                        except Exception:
                            space_w = int(max(1, 0.33 * fsize))
                        items.append({'type': 'space', 'w': space_w, 'h': th, 'fs': fsize, 'group_index': group_idx})
                        total_w += space_w
                        max_h = max(max_h, h_total)
                        group_meta[group_idx]['max_height'] = max(group_meta[group_idx]['max_height'], h_total)
            else:
                text = (wobj.get('text') or '').strip()
                bbox = draw.textbbox((0, 0), text, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                rgb = (255, 255, 255)
                items.append({
                    'type': 'word',
                    'text': text,
                    'font': font,
                    'w': tw,
                    'h': th,
                    'rgb': rgb,
                    'prog': prog,
                    'fs': fsize,
                    'bg_rgb': None,
                    'pad_x': 0,
                    'pad_y': 0,
                    'keyword': False,
                    'color_hex': '#FFFFFF',
                    'group_index': group_idx,
                })
                total_w += tw
                max_h = max(max_h, th)
        # Auto-scale si trop large (> 92% de la largeur)
        target_w = int(width * 0.92)
        if total_w > target_w and items:
            shrink = max(0.6, min(1.0, target_w / float(total_w)))
            new_items = []
            total_w = 0
            max_h = 0
            for it in items:
                if it['type'] == 'word':
                    new_fs = max(1, int(it.get('fs', self.config['font_size']) * shrink))
                    nfont = self._load_font(new_fs)
                    nb = draw.textbbox((0, 0), it['text'], font=nfont)
                    nw = nb[2] - nb[0]; nh = nb[3] - nb[1]
                    keyword = bool(it.get('keyword'))
                    color_hex = it.get('color_hex', '#FFFFFF')
                    base_rgb = self.hex_to_rgb(color_hex) if keyword else (255, 255, 255)
                    bg_rgb = None
                    pad_x = 0
                    pad_y = 0
                    text_rgb = base_rgb if keyword and not self.config.get('keyword_background', False) else (255, 255, 255)
                    if keyword and self.config.get('keyword_background', False):
                        bg_rgb = base_rgb
                        pad_x = max(4, int(new_fs * 0.18))
                        pad_y = max(2, int(new_fs * 0.12))
                    nw_total = nw + pad_x * 2
                    nh_total = nh + pad_y * 2
                    new_items.append({
                        'type': 'word',
                        'text': it['text'],
                        'font': nfont,
                        'w': nw_total,
                        'h': nh_total,
                        'rgb': text_rgb,
                        'prog': it.get('prog', 1.0),
                        'fs': new_fs,
                        'bg_rgb': bg_rgb,
                        'pad_x': pad_x,
                        'pad_y': pad_y,
                        'keyword': keyword,
                        'color_hex': color_hex,
                        'group_index': it.get('group_index'),
                    })
                    total_w += nw_total
                    max_h = max(max_h, nh_total)
                elif it['type'] == 'space':
                    new_fs = max(1, int(it.get('fs', self.config['font_size']) * shrink))
                    nfont = self._load_font(new_fs)
                    try:
                        nw = int(max(1, draw.textlength(" ", font=nfont)))
                    except Exception:
                        nw = int(max(1, 0.33 * new_fs))
                    new_items.append({'type':'space','w':nw,'h':it['h'],'fs':new_fs,'group_index': it.get('group_index')})
                    total_w += nw; max_h = max(max_h, it['h'])
                elif it['type'] == 'emoji':
                    em = it['img']
                    nw = max(1, int(em.size[0] * shrink)); nh = max(1, int(em.size[1] * shrink))
                    em_resized = em.resize((nw, nh), Image.Resampling.LANCZOS)
                    new_items.append({'type':'emoji','img':em_resized,'w':nw + self.config['emoji_gap_px'],'h':nh,'fs':int(it.get('fs', self.config['font_size']) * shrink),'group_index': it.get('group_index')})
                    total_w += (nw + self.config['emoji_gap_px']); max_h = max(max_h, nh)
                else:
                    new_items.append(it)
                    total_w += it.get('w', 0); max_h = max(max_h, it.get('h', 0))
            items = new_items
        for meta in group_meta:
            meta['max_height'] = 0
        for it in items:
            group_idx = it.get('group_index') if isinstance(it, dict) else None
            if group_idx is None:
                continue
            if it['type'] in {'word', 'space'}:
                group_meta[group_idx]['max_height'] = max(group_meta[group_idx].get('max_height', 0), it.get('h', 0))
                group_meta[group_idx]['font_size'] = max(group_meta[group_idx].get('font_size', 0), it.get('fs', 0))
        # Position
        x = (width - total_w) // 2
        # Marge adaptative: au moins un pourcentage de la hauteur
        margin_bottom_px = max(int(self.config.get('margin_bottom', 80)), int(height * 0.06))
        # Lissage de la hauteur de ligne pour éviter les sauts verticaux liés à l'animation/bounce
        line_h_target = float(max_h)
        if self._line_h_ema is None:
            self._line_h_ema = line_h_target
        else:
            alpha_line = 0.12
            self._line_h_ema = (1 - alpha_line) * self._line_h_ema + alpha_line * line_h_target
        y_target = float(height - margin_bottom_px - int(self._line_h_ema))
        # Position fixe : aucun ajustement pour faces
        # Lissage EMA de la position Y pour atténuer tout jitter restant
        if self._y_ema is None:
            self._y_ema = y_target
        else:
            alpha_y = 0.15
            self._y_ema = (1 - alpha_y) * self._y_ema + alpha_y * y_target
        y = int(self._y_ema)
        stroke_px = max(0, int(self.config.get('stroke_px', 6)))
        stroke_color = tuple(self.config.get('stroke_color', (0, 0, 0)))
        shadow_offset_px = max(0, int(self.config.get('shadow_offset', 3)))
        shadow_opacity = float(self.config.get('shadow_opacity', 0.35))
        group_positions: Dict[int, Dict[str, Optional[int]]] = {
            idx: {'start_x': None, 'end_x': None, 'baseline_y': None, 'max_h': 0}
            for idx in range(len(group_meta))
        }
        # Rendu
        for it in items:
            if it['type'] == 'word':
                word_text = it['text']
                font = it['font']
                rgb = it['rgb']
                prog = float(it.get('prog', 1.0))
                alpha_prog = prog if prog <= 1.0 else 1.0
                alpha = int(255 * min(alpha_prog / max(1e-6, self.config['fade_duration']), 1.0))
                fill = (*rgb, alpha)
                pad_x = int(it.get('pad_x', 0))
                pad_y = int(it.get('pad_y', 0))
                word_w = int(it.get('w', 0))
                word_h = int(it.get('h', 0))
                draw_x = x + pad_x
                draw_y = y + pad_y
                bg_rgb = it.get('bg_rgb')
                group_idx = it.get('group_index')
                if group_idx is not None:
                    pos = group_positions[group_idx]
                    if pos['start_x'] is None:
                        pos['start_x'] = x
                        pos['baseline_y'] = draw_y
                    pos['end_x'] = x + word_w
                    pos['max_h'] = max(pos['max_h'], word_h)
                    if pos['baseline_y'] is None or draw_y < pos['baseline_y']:
                        pos['baseline_y'] = draw_y
                if self.config.get('keyword_background', False) and bg_rgb:
                    radius = max(2, int(it.get('fs', self.config['font_size']) * 0.25))
                    rect = [int(x), int(y), int(x + word_w), int(y + word_h)]
                    draw.rounded_rectangle(rect, radius=radius, fill=(*bg_rgb, int(alpha * 0.92)))
                if shadow_offset_px > 0 and shadow_opacity > 0:
                    shadow_alpha = max(0, min(255, int(alpha * shadow_opacity)))
                    if shadow_alpha > 0:
                        draw.text((draw_x + shadow_offset_px, draw_y + shadow_offset_px), word_text, font=font, fill=(0, 0, 0, shadow_alpha))
                if stroke_px > 0:
                    stroke_fill = (*stroke_color, alpha)
                    for dx in range(-stroke_px, stroke_px + 1):
                        for dy in range(-stroke_px, stroke_px + 1):
                            if dx == 0 and dy == 0:
                                continue
                            if dx * dx + dy * dy > stroke_px * stroke_px:
                                continue
                            draw.text((draw_x + dx, draw_y + dy), word_text, font=font, fill=stroke_fill)
                # Dessin du texte une seule fois (pas de gradient/ombre pour éviter le sur-noircissement)
                draw.text((draw_x, draw_y), word_text, font=font, fill=fill)
                x += word_w
            elif it['type'] == 'space':
                # Avancer la position horizontale pour l'espace calculé
                group_idx = it.get('group_index')
                if group_idx is not None:
                    pos = group_positions[group_idx]
                    if pos['start_x'] is None:
                        pos['start_x'] = x
                        pos['baseline_y'] = y
                    pos['end_x'] = x + it['w']
                    pos['max_h'] = max(pos['max_h'], it.get('h', 0))
                x += it['w']

        for group_idx, meta in enumerate(group_meta):
            emojis = meta.get('emojis') or []
            if not emojis:
                continue
            pos = group_positions.get(group_idx)
            if not pos:
                continue
            start_x = pos.get('start_x')
            end_x = pos.get('end_x')
            baseline_y = pos.get('baseline_y', y)
            if start_x is None or end_x is None:
                continue
            target_base_fs = max(int(meta.get('font_size') or self.config['font_size']), 1)
            target_h = max(1, int(target_base_fs * self.config.get('emoji_scale_ratio', 0.9) * self.config.get('emoji_boost', 1.0)))
            for idx, emoji_char in enumerate(emojis):
                if not emoji_char:
                    continue
                emoji_img = self._load_emoji_png(emoji_char, target_h)
                if emoji_img is None and not self.config.get('emoji_png_only', False):
                    efont = self._load_emoji_font(target_h)
                    temp = Image.new('RGBA', (target_h * 2, target_h * 2), (0, 0, 0, 0))
                    temp_draw = ImageDraw.Draw(temp)
                    temp_draw.text((0, 0), emoji_char, font=efont, fill=(255, 255, 255, 255))
                    bbox = temp.getbbox()
                    if bbox:
                        emoji_img = temp.crop(bbox)
                if emoji_img is None:
                    continue
                ew, eh = emoji_img.size
                offset_x = int(end_x - ew * (0.6 - 0.15 * idx))
                offset_x = min(max(0, offset_x), width - ew)
                offset_y = int(max(0, (baseline_y or y) - eh * 0.45))
                img.alpha_composite(emoji_img, (offset_x, offset_y))

        text_array = np.array(img)
        if text_array.shape[2] == 4:
            alpha_channel = text_array[:, :, 3] / 255.0
            rgb = text_array[:, :, :3]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for c in range(3):
                frame[:, :, c] = (
                    frame[:, :, c] * (1 - alpha_channel) +
                    bgr[:, :, c] * alpha_channel
                ).astype(frame.dtype)
        self._last_render_metadata = {
            'items': items,
            'stroke_px': stroke_px,
            'margin_bottom': margin_bottom_px,
        }
        return frame

    def add_hormozi_subtitles(self, input_video_path: str, 
                               transcription_data: List[Dict], 
                               output_video_path: str) -> None:
        """
        Ajoute des sous-titres style Hormozi 1 (groupes 2–3 mots, multi-couleurs sur une ligne, emojis PNG en surimpression)
        """
        print("🔥 Génération sous-titres style Hormozi 1...")
        # Groupes plus dynamiques (2–3 mots)
        groups = self.parse_transcription_to_word_groups(transcription_data, group_size=2)
        try:
            self._enrich_keywords_from_transcript(groups)
        except Exception:
            pass
        print(f"📝 {len(groups)} groupes de mots extraits")
        video = mp.VideoFileClip(input_video_path)
        def apply_subtitles(get_frame, t):
            frame = get_frame(t)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            active = []
            for w in groups:
                if float(w.get("start",0.0)) <= t <= float(w.get("end",0.0)):
                    duration = max(0.01, float(w.get("end",0.0)) - float(w.get("start",0.0)))
                    progress = (t - float(w.get("start",0.0))) / duration
                    if progress < 0.3:
                        anim_prog = progress / 0.3
                    elif progress < 0.5:
                        anim_prog = 1.0
                    else:
                        anim_prog = 1.0 + 0.1 * abs(np.sin(progress * np.pi * 3))
                    w_active = dict(w)
                    w_active["animation_progress"] = float(anim_prog)
                    active.append(w_active)
            # Rendu texte + overlay éventuel d'emoji PNG pour mots clés boostés
            out_bgr = self.create_subtitle_frame(frame_bgr, active, t)
            try:
                # Si une palette avec 'emoji' a été fournie, overlay à droite du texte
                if isinstance(getattr(self, 'span_style_map', None), dict):
                    for w in active[:2]:
                        word = str(w.get('text','')).strip().lower()
                        style = self.span_style_map.get(word)
                        if isinstance(style, dict) and 'emoji' in style:
                            # On attend un fichier PNG local correspondant au nom de l'emoji (ex: rocket.png)
                            emo = style.get('emoji') or ''
                            name = None
                            if isinstance(emo, str):
                                # Mapping étendu pour tous les émojis du span_style_map
                                m = {
                                    # Business & Croissance
                                    '📈': 'emoji_chart.png', '🌱': 'emoji_growth.png',
                                    '🔑': 'emoji_key.png', '🌟': 'emoji_star.png',
                                    '⚡': 'emoji_lightning.png', '💡': 'emoji_bulb.png',
                                    '🧭': 'emoji_compass.png', '🗺️': 'emoji_map.png',
                                    # Argent & Finance
                                    '💰': 'emoji_money.png', '📊': 'emoji_chart.png',
                                    '🏦': 'emoji_bank.png', '📉': 'emoji_down.png',
                                    '❌': 'emoji_cross.png', '🧾': 'emoji_receipt.png',
                                    '🪙': 'emoji_coin.png',
                                    # Relation & Client
                                    '🤝': 'emoji_handshake.png', '🫱🏼‍🫲🏽': 'emoji_handshake.png',
                                    '🌍': 'emoji_earth.png', '👥': 'emoji_group.png',
                                    '🔒': 'emoji_lock.png', '🛒': 'emoji_cart.png',
                                    '📦': 'emoji_package.png', '📋': 'emoji_contract.png',
                                    # Motivation & Succès
                                    '🔥': 'emoji_fire.png', '⚡': 'emoji_lightning.png',
                                    '🏆': 'emoji_trophy.png', '🎯': 'emoji_target.png',
                                    '⏳': 'emoji_hourglass.png', '🥋': 'emoji_karate.png',
                                    '🚀': 'emoji_rocket.png', '🌐': 'emoji_globe.png',
                                    '💥': 'emoji_explosion.png',
                                    # Risque & Erreurs
                                    '⚠️': 'emoji_warning.png', '🛑': 'emoji_stop.png',
                                    '🧱': 'emoji_wall.png', '⛔': 'emoji_blocked.png',
                                    '🔧': 'emoji_tools.png', '🪄': 'emoji_magic.png',
                                    '📚': 'emoji_book.png', '': 'emoji_brain.png'
                                }
                                name = m.get(emo)
                            if name:
                                # Chercher dans plusieurs dossiers possibles
                                png_paths = [
                                    Path('assets/emojis')/name,
                                    Path('AI-B-roll/assets/emojis')/name,
                                    Path('emojis')/name
                                ]
                                png = None
                                for p in png_paths:
                                    if p.exists():
                                        png = p
                                        break
                                if png:
                                    # Position approx: coin inférieur droit sécurisé
                                    h,w_ = out_bgr.shape[:2]
                                    out_bgr = self.overlay_big_emoji(out_bgr, str(png), max(32, w_ - 380), max(64, h - 520), scale=1.6)
                                else:
                                    # Fallback: afficher l'emoji unicode en petit si PNG manquant
                                    try:
                                        from PIL import ImageFont, ImageDraw
                                        emoji_font = ImageFont.truetype("arial.ttf", 48)
                                        emoji_img = Image.new('RGBA', (64, 64), (0,0,0,0))
                                        draw = ImageDraw.Draw(emoji_img)
                                        draw.text((8, 8), emo, font=emoji_font, fill=(255,255,255,255))
                                        # Convertir et overlay
                                        emoji_arr = np.array(emoji_img)
                                        # Position plus discrète
                                        h,w_ = out_bgr.shape[:2]
                                        x_pos = max(32, w_ - 120)
                                        y_pos = max(64, h - 120)
                                        # Overlay simple
                                        for c in range(3):
                                            out_bgr[y_pos:y_pos+64, x_pos:x_pos+64, c] = (
                                                out_bgr[y_pos:y_pos+64, x_pos:x_pos+64, c] * 0.3 +
                                                emoji_arr[:,:,c] * 0.7
                                            ).astype(out_bgr.dtype)
                                    except Exception:
                                        pass  # Fallback silencieux si émoji unicode échoue
            except Exception:
                pass
            return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        # 🚨 CORRECTION BUG: Utiliser apply_to=None pour éviter les problèmes de dimensions
        final_video = video.fl(apply_subtitles, apply_to=None)
        print("💾 Export vidéo finale...")
        
        # 🚨 CORRECTION BUG: S'assurer que les dimensions finales sont paires pour H.264
        try:
            # Vérifier les dimensions de la vidéo finale
            final_width = final_video.w
            final_height = final_video.h
            
            # Forcer des dimensions paires si nécessaire
            if final_width % 2 != 0:
                final_width = final_width - 1 if final_width > 1 else final_width + 1
            if final_height % 2 != 0:
                final_height = final_height - 1 if final_height > 1 else final_height + 1
            
            # Redimensionner si les dimensions ont changé
            if final_width != video.w or final_height != video.h:
                print(f"    🔧 Correction dimensions: {video.w}x{video.h} → {final_width}x{final_height}")
                final_video = final_video.resize((final_width, final_height))
        except Exception as e:
            print(f"    ⚠️ Erreur correction dimensions: {e}")
            # Fallback: redimensionner à la taille cible standard
            try:
                final_video = final_video.resize((720, 1280))
                print("    🔧 Fallback: redimensionnement à 720x1280")
            except Exception:
                pass
        
        final_video.write_videofile(
            output_video_path,
            codec='libx264',
            audio_codec='aac',
            fps=video.fps,
            preset='medium',
            ffmpeg_params=['-pix_fmt', 'yuv420p', '-movflags', '+faststart']
        )
        video.close(); final_video.close()
        print(f"✅ Sous-titres Hormozi ajoutés : {output_video_path}") 
        # Export tokens JSON à côté
        try:
            self.export_tokens_json(groups, str(Path(output_video_path).with_suffix('.tokens.json')))
        except Exception:
            pass

    def apply_brand_kit(self, kit_id: str | None):
        """Applique un preset brand si fourni."""
        if not kit_id:
            return
        kit = self.brand_presets.get(str(kit_id), None)
        if not kit:
            return
        for k,v in kit.items():
            self.config[k] = v

    def apply_span_style_map(self, mapping: Dict[str, Dict[str, object]]):
        """Applique une palette riche multi-couleurs pour certains mots-clés.
        mapping ex: {"argent": {"color":"#FFD700","bold":True}, ...}
        """
        try:
            for kw, style in (mapping or {}).items():
                col = style.get("color")
                if isinstance(col, str) and col.startswith('#') and len(col) in (4,7):
                    # Convert hex to RGB
                    h = col.lstrip('#')
                    if len(h)==3:
                        r,g,b = [int(c*2,16) for c in h]
                    else:
                        r = int(h[0:2],16); g=int(h[2:4],16); b=int(h[4:6],16)
                    self.keyword_colors[kw.lower()] = (r,g,b)
        except Exception:
            pass
    
    def get_smart_color_for_keyword(self, keyword: str, text: str = "", intensity: float = 1.0) -> str:
        """Obtient une couleur intelligente pour un mot-clé (nouveau système)"""
        if self.SMART_SYSTEMS_AVAILABLE:
            try:
                result = self.smart_colors.get_color_for_keyword(keyword, text, intensity)
                return result
            except Exception as e:
                print(f"🔍 DEBUG SMART: Erreur: {e}")
                pass
        
        # Fallback : système classique
        if keyword.lower() in self.keyword_colors:
            r, g, b = self.keyword_colors[keyword.lower()]
            return f"#{r:02x}{g:02x}{b:02x}"
        
        # Fallback : couleur par défaut
        return "#FFFFFF"
    
    def get_contextual_emoji_for_keyword(self, keyword: str, text: str = "", sentiment: str = "neutral", intensity: float = 1.0) -> str:
        """MAPPING AUTHENTIQUE HORMOZI 1 POUR TIKTOK VIRAL"""
        keyword_lower = keyword.lower().strip()
        
        # 🔥 MAPPING HORMOZI 1 AUTHENTIQUE BASÉ SUR TIKTOK
        hormozi_emoji_map = {
            # 💰 ARGENT & BUSINESS (couleur signature Hormozi)
            'money': '💰', 'cash': '💸', 'profit': '💰', 'revenue': '💰', 'wealth': '💰',
            'business': '💼', 'sales': '💰', 'income': '💰', 'rich': '💰', 'expensive': '💸',
            'investment': '📈', 'financial': '💰', 'budget': '💰', 'value': '💎',
            
            # 🚨 ATTENTION & URGENCE (style Hormozi)
            'attention': '👀', 'look': '👀', 'watch': '👀', 'see': '👀', 'focus': '🎯',
            'important': '🚨', 'urgent': '🚨', 'critical': '🚨', 'must': '❗', 'need': '❗',
            'stop': '✋', 'wait': '✋', 'listen': '👂', 'hear': '👂',
            
            # ⚡ ACTION & ÉNERGIE
            'action': '⚡', 'move': '🏃', 'go': '🚀', 'start': '🚀', 'begin': '🚀',
            'work': '💪', 'effort': '💪', 'push': '💪', 'fight': '⚔️', 'battle': '⚔️',
            'power': '⚡', 'energy': '⚡', 'force': '💪', 'strength': '💪',
            
            # 🏆 SUCCÈS & VICTOIRE
            'success': '🏆', 'win': '🏆', 'winner': '🏆', 'victory': '🏆', 'champion': '🏆',
            'best': '👑', 'top': '👑', 'first': '🥇', 'great': '🔥', 'amazing': '🤯',
            'incredible': '🤯', 'fantastic': '🔥', 'perfect': '💯', 'excellent': '⭐',
            
            # 🧠 INTELLIGENCE & APPRENTISSAGE  
            'learn': '🧠', 'study': '📚', 'education': '🎓', 'knowledge': '🧠', 'smart': '🧠',
            'understand': '💡', 'idea': '💡', 'think': '🤔', 'brain': '🧠', 'mind': '🧠',
            'wisdom': '🦉', 'insight': '💡', 'discovery': '🔍', 'find': '🔍',
            
                         # ❤️ ÉMOTIONS POSITIVES
             'love': '❤️', 'like': '👍', 'enjoy': '😊', 'happy': '😊', 'joy': '😊',
             'excited': '🤩', 'wonderful': '✨', 'beautiful': '✨',
             'good': '👍', 'positive': '🌟', 'hope': '🌟', 'dream': '✨',
            
            # 😡 ÉMOTIONS NÉGATIVES (mapping CORRECT!)
            'hate': '😡', 'angry': '😡', 'mad': '😡', 'furious': '🤬', 'rage': '🤬',
            'bad': '👎', 'terrible': '💀', 'awful': '💀', 'horrible': '💀',
            'problem': '⚠️', 'issue': '⚠️', 'trouble': '⚠️', 'difficulty': '😤',
            'challenge': '💪', 'struggle': '😤', 'pain': '😣', 'hurt': '😣',
            
            # 🎯 OBJECTIFS & CIBLES
            'goal': '🎯', 'target': '🎯', 'objective': '🎯', 'aim': '🎯', 'focus': '🎯',
            'plan': '📋', 'strategy': '🧭', 'method': '⚙️', 'system': '⚙️',
            
            # 👥 PERSONNES & RELATIONS
            'people': '👥', 'person': '👤', 'man': '👨', 'woman': '👩', 'women': '👩',
            'team': '👥', 'group': '👥', 'community': '🌍', 'family': '👨‍👩‍👧‍👦',
            'friend': '👫', 'relationship': '💕', 'partner': '🤝',
            
            # ⏰ TEMPS & URGENCE
            'time': '⏰', 'now': '⏰', 'today': '📅', 'tomorrow': '📅', 'future': '🔮',
            'past': '📜', 'present': '⏰', 'quick': '⚡', 'fast': '⚡', 'slow': '🐌',
            'wait': '⏳', 'delay': '⏳', 'hurry': '💨', 'rush': '💨',
            
            # 🚀 CROISSANCE & PROGRÈS
            'growth': '📈', 'progress': '📈', 'improve': '📈', 'better': '📈',
            'upgrade': '⬆️', 'level': '📊', 'scale': '📈', 'expand': '📈',
            'develop': '🌱', 'evolution': '🦋', 'change': '🔄', 'transform': '🦋',
            
            # 💯 QUALITÉ & PERFORMANCE
            'quality': '💎', 'premium': '👑', 'luxury': '💎', 'elite': '👑',
            'professional': '💼', 'expert': '🎓', 'master': '👑', 'pro': '💯',
            
            # 🦁 FORCE & PUISSANCE (style alpha Hormozi)
            'beast': '🦁', 'monster': '👹', 'savage': '🦁', 'alpha': '👑', 'lion': '🦁',
            'tiger': '🐅', 'warrior': '⚔️', 'killer': '💀', 'machine': '🤖', 'unstoppable': '🚀',
            'invincible': '💪', 'legendary': '👑', 'godlike': '⚡', 'superior': '👑',
            
            # 🔥 VIRAL & TENDANCE (spécial TikTok)
            'viral': '🔥', 'trending': '📈', 'hot': '🔥', 'fire': '🔥', 'lit': '🔥',
            'crazy': '🤯', 'insane': '🤯', 'wild': '🤯', 'epic': '🔥', 'sick': '🔥'
        }
        
        # Recherche directe dans le mapping Hormozi
        if keyword_lower in hormozi_emoji_map:
            return hormozi_emoji_map[keyword_lower]
        
        # Mots de liaison = PAS d'emoji (style Hormozi authentique)
        linking_words = {
            'the', 'and', 'or', 'but', 'if', 'then', 'when', 'where', 'how', 'why',
            'what', 'who', 'which', 'that', 'this', 'these', 'those', 'with', 'without',
            'from', 'to', 'for', 'of', 'in', 'on', 'at', 'by', 'about', 'into', 'through'
        }
        if keyword_lower in linking_words:
            return ""
        
        # Fallback intelligent par contexte
        if 'money' in text.lower() or 'business' in text.lower():
            return '💰'
        elif 'success' in text.lower() or 'win' in text.lower():
            return '🏆'
        elif 'problem' in text.lower() or 'issue' in text.lower():
            return '⚠️'
        elif 'learn' in text.lower() or 'education' in text.lower():
            return '🧠'
        
        # Pas d'emoji pour les mots non pertinents (style Hormozi)
        return ""
    
    def load_emoji_png_improved(self, emoji_char: str, size: int = 64) -> Path | None:
        """NOUVEAU : Chargement amélioré des emojis PNG avec fallback robuste"""
        try:
            # Vérifier le cache d'abord
            if emoji_char in self.emoji_png_cache:
                return self.emoji_png_cache[emoji_char]
            
            # Essayer le mapping direct
            filename = self.emoji_mapping.get(emoji_char)
            if not filename:
                # Fallback : générer le nom de fichier à partir du code Unicode
                filename = f"{ord(emoji_char):x}.png"
            
            # Construire le chemin
            emoji_path = Path("emoji_assets") / filename
            
            # Vérifier l'existence
            if emoji_path.exists():
                # Mettre en cache
                self.emoji_png_cache[emoji_char] = emoji_path
                print(f"✅ Emoji PNG chargé: {emoji_char} → {filename}")
                return emoji_path
            else:
                print(f"⚠️ Emoji PNG manquant: {emoji_char} → {filename}")
                return None
                
        except Exception as e:
            print(f"❌ Erreur chargement emoji PNG: {e}")
            return None
    
    def get_emoji_display_improved(self, emoji_char: str, fallback_to_text: bool = True) -> str:
        """NOUVEAU : Obtient l'affichage optimal d'un emoji avec fallback"""
        # Essayer PNG d'abord
        png_path = self.load_emoji_png_improved(emoji_char)
        if png_path:
            return f"PNG:{png_path}"
        
        # Fallback vers police système
        if fallback_to_text:
            return emoji_char
        
        # Fallback vers emoji générique
        return "✨"

    def overlay_big_emoji(self, frame_bgr: np.ndarray, emoji_png_path: str, x: int, y: int, scale: float = 1.0) -> np.ndarray:
        """Superpose un gros emoji PNG sur le frame pour simuler un big-emoji.
        """
        try:
            if not os.path.exists(emoji_png_path):
                return frame_bgr
            em = Image.open(emoji_png_path).convert('RGBA')
            if scale != 1.0:
                em = em.resize((int(em.size[0]*scale), int(em.size[1]*scale)), Image.LANCZOS)
            h,w = frame_bgr.shape[:2]
            canvas = Image.new('RGBA', (w,h), (0,0,0,0))
            canvas.alpha_composite(em, (x, y))
            arr = np.array(canvas)
            alpha = arr[:,:,3] / 255.0
            rgb = arr[:,:,:3][:,:,::-1]  # RGBA->BGR
            out = frame_bgr.copy()
            for c in range(3):
                out[:,:,c] = (out[:,:,c]*(1-alpha) + rgb[:,:,c]*alpha).astype(out.dtype)
            return out
        except Exception:
            return frame_bgr

# Compat pour import externe
HormoziSubtitleProcessor = HormoziSubtitles


def add_hormozi_subtitles(input_video_path: str,
                          transcription_data: List[Dict],
                          output_video_path: str,
                          **kwargs) -> None:
    """Wrapper compatible attendu par video_processor.py."""
    subtitle_settings = kwargs.pop('subtitle_settings', None)
    font_path_override = kwargs.pop('font_path', None)
    font_candidates = [font_path_override] if font_path_override else None
    proc = HormoziSubtitles(subtitle_settings=subtitle_settings, font_candidates=font_candidates)
    # Appliquer options simples si fournies
    for key in ['font_size', 'margin_bottom', 'bounce_scale', 'enable_emojis', 'emoji_boost', 'keyword_background', 'emoji_png_only', 'emoji_density_non_keyword', 'emoji_density_keyword', 'emoji_min_gap_groups']:
        if key in kwargs:
            proc.config[key] = kwargs[key]
    # Brand kit si fourni
    if 'brand_kit' in kwargs:
        proc.apply_brand_kit(kwargs['brand_kit'])
    # Mises à jour mapping
    if 'keyword_colors' in kwargs and isinstance(kwargs['keyword_colors'], dict):
        proc.keyword_colors.update(kwargs['keyword_colors'])
    # Palette riche multi-couleurs
    if 'span_style_map' in kwargs and isinstance(kwargs['span_style_map'], dict):
        proc.apply_span_style_map(kwargs['span_style_map'])
        # stocker pour overlay emoji
        proc.span_style_map = kwargs['span_style_map']
    if 'emoji_mapping' in kwargs and isinstance(kwargs['emoji_mapping'], dict):
        proc.emoji_mapping.update(kwargs['emoji_mapping'])
    # Exécuter
    proc.add_hormozi_subtitles(input_video_path, transcription_data, output_video_path) 
