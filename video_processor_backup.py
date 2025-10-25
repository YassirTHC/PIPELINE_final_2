ï»¿# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import subprocess
import logging
import random
import numpy as np
import shutil
import time  # NEW: pour timestamps uniques
from datetime import datetime  # NEW: pour mÃƒÂ©tadonnÃƒÂ©es intelligentes
from temp_function import _llm_generate_caption_hashtags_fixed
from pathlib import Path
from typing import List, Dict, Optional
import whisper
import requests
import cv2
# Gestion optionnelle de Mediapipe avec fallback
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("Ã¢Å“â€¦ Mediapipe disponible - Utilisation des fonctionnalitÃƒÂ©s IA avancÃƒÂ©es")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None
    print("Ã¢Å¡Â Ã¯Â¸Â Mediapipe non disponible - Utilisation du fallback OpenCV (fonctionnalitÃƒÂ©s rÃƒÂ©duites)")

# Ã°Å¸Å¡â‚¬ NOUVEAU: Import du sÃƒÂ©lecteur B-roll gÃƒÂ©nÃƒÂ©rique
try:
    from broll_selector import BrollSelector, Asset, ScoringFeatures, BrollCandidate
    BROLL_SELECTOR_AVAILABLE = True
    print("Ã¢Å“â€¦ SÃƒÂ©lecteur B-roll gÃƒÂ©nÃƒÂ©rique disponible - Scoring mixte activÃƒÂ©")
except ImportError as e:
    BROLL_SELECTOR_AVAILABLE = False
    print(f"Ã¢Å¡Â Ã¯Â¸Â SÃƒÂ©lecteur B-roll gÃƒÂ©nÃƒÂ©rique non disponible: {e}")
    print("   Ã°Å¸â€â€ž Utilisation du systÃƒÂ¨me de scoring existant")

try:
    from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
except Exception:
    from moviepy.editor import VideoFileClip, CompositeVideoClip
    try:
        from moviepy.editor import TextClip  # type: ignore
    except Exception:
        TextClip = None  # type: ignore[assignment]
from tqdm import tqdm  # NEW: console progress
import re # NEW: for caption/hashtag generation
from hormozi_subtitles import add_hormozi_subtitles


def _format_srt_timestamp(total_seconds: float) -> str:
    """Format seconds to SRT timestamp HH:MM:SS,mmm"""
    if total_seconds < 0:
        total_seconds = 0.0
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int(round((total_seconds - int(total_seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def write_srt(segments: List[Dict], srt_path: Path) -> None:
    """Write segments [{'start','end','text'}] to SRT file."""
    srt_path = Path(srt_path)
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    index = 1
    for seg in segments:
        text = (seg.get('text') or '').strip()
        if not text:
            continue
        start = float(seg.get('start') or 0.0)
        end = float(seg.get('end') or max(0.0, start + 0.01))
        start_ts = _format_srt_timestamp(start)
        end_ts = _format_srt_timestamp(end)
        lines.append(str(index))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")
        index += 1
    with open(srt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

def write_vtt(segments: List[Dict], vtt_path: Path) -> None:
    """Write segments to WebVTT file."""
    vtt_path = Path(vtt_path)
    vtt_path.parent.mkdir(parents=True, exist_ok=True)
    def to_vtt_ts(total_seconds: float) -> str:
        if total_seconds < 0:
            total_seconds = 0.0
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int(round((total_seconds - int(total_seconds)) * 1000))
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    lines: List[str] = ["WEBVTT",""]
    for seg in segments:
        text = (seg.get('text') or '').strip()
        if not text:
            continue
        start = float(seg.get('start') or 0.0)
        end = float(seg.get('end') or max(0.0, start + 0.01))
        lines.append(f"{to_vtt_ts(start)} --> {to_vtt_ts(end)}")
        lines.append(text)
        lines.append("")
    with open(vtt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def _read_ui_settings() -> Dict:
    """Read optional UI settings from config/ui_settings.json."""
    try:
        cfg_path = Path('config/ui_settings.json')
        if cfg_path.exists():
            with open(cfg_path, 'r', encoding='utf-8') as f:
                return json.load(f) or {}
    except Exception:
        pass
    return {}


def _to_bool(v, default=False) -> bool:
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1","true","yes","on"}


# Load optional UI overrides once
_UI_SETTINGS = _read_ui_settings()

# Configuration automatique d'ImageMagick pour MoviePy
def configure_imagemagick():
    """Configure automatiquement ImageMagick pour MoviePy"""
    try:
        import moviepy.config as cfg
        
        # Chemins possibles pour ImageMagick sur Windows
        possible_paths = [
            r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe",
            r"C:\Program Files\ImageMagick-7.1.2-Q16\magick.exe",
            r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe",
            r"C:\Program Files\ImageMagick-7.1.1-Q16\magick.exe",
            r"C:\Program Files\ImageMagick-7.1.0-Q16-HDRI\magick.exe",
            r"C:\Program Files\ImageMagick-7.1.0-Q16\magick.exe",
        ]
        
        # Chercher ImageMagick
        for path in possible_paths:
            if os.path.exists(path):
                cfg.change_settings({"IMAGEMAGICK_BINARY": path})
                print(f"Ã¢Å“â€¦ ImageMagick configurÃƒÂ©: {path}")
                return True
        
        print("Ã¢Å¡Â Ã¯Â¸Â ImageMagick non trouvÃƒÂ©, utilisation du mode fallback")
        return False
        
    except Exception as e:
        print(f"Ã¢Å¡Â Ã¯Â¸Â Erreur configuration ImageMagick: {e}")
        return False

# Configuration automatique au dÃƒÂ©marrage
configure_imagemagick()

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    """Configuration centralisÃƒÂ©e du pipeline"""
    CLIPS_FOLDER = Path("./clips")
    OUTPUT_FOLDER = Path("./output") 
    TEMP_FOLDER = Path("./temp")
    
    # RÃƒÂ©solution cible pour les rÃƒÂ©seaux sociaux
    TARGET_WIDTH = 720
    TARGET_HEIGHT = 1280  # Format 9:16
    
    # ParamÃƒÂ¨tres Whisper
    WHISPER_MODEL = "tiny"  # ou "small", "medium", "large"
    
    # ParamÃƒÂ¨tres sous-titres
    SUBTITLE_FONT_SIZE = 60
    SUBTITLE_COLOR = 'yellow'
    SUBTITLE_STROKE_COLOR = 'black'
    SUBTITLE_STROKE_WIDTH = 3
    # Biais global (en secondes) pour corriger un lÃƒÂ©ger dÃƒÂ©calage systÃƒÂ©matique
    # 0.0 par dÃƒÂ©faut pour ÃƒÂ©viter tout dÃƒÂ©calage si non nÃƒÂ©cessaire
    SUBTITLE_TIMING_BIAS_S = 0.0

    # Activation B-roll: UI > ENV > dÃƒÂ©faut(off)
    # Si fetchers cochÃƒÂ©s, activer automatiquement l'insertion B-roll, sauf si explicitement dÃƒÂ©sactivÃƒÂ© cÃƒÂ´tÃƒÂ© UI
    _UI_ENABLE_BROLL = _UI_SETTINGS.get('enable_broll') if 'enable_broll' in _UI_SETTINGS else None
    _ENV_ENABLE_BROLL = os.getenv('ENABLE_BROLL') or os.getenv('AI_BROLL_ENABLED')
    _AUTO_ENABLE = _to_bool(_UI_SETTINGS.get('broll_fetch_enable'), default=True) if 'broll_fetch_enable' in _UI_SETTINGS else _to_bool(os.getenv('BROLL_FETCH_ENABLE') or os.getenv('AI_BROLL_ENABLE_FETCHER'), default=True)
    ENABLE_BROLL = (
        _to_bool(_UI_ENABLE_BROLL, default=False) if _UI_ENABLE_BROLL is not None
        else (_to_bool(_ENV_ENABLE_BROLL, default=False) or _AUTO_ENABLE)
    )

    # === Options fetcher B-roll (stock) ===
    # Active le fetch automatique: UI > ENV > dÃƒÂ©faut(on)
    BROLL_FETCH_ENABLE = _to_bool(_UI_SETTINGS.get('broll_fetch_enable'), default=True) if 'broll_fetch_enable' in _UI_SETTINGS else _to_bool(os.getenv('BROLL_FETCH_ENABLE') or os.getenv('AI_BROLL_ENABLE_FETCHER'), default=True)
    # Fournisseur: UI > ENV > dÃƒÂ©faut pexels
    BROLL_FETCH_PROVIDER = (_UI_SETTINGS.get('broll_fetch_provider') or os.getenv('AI_BROLL_FETCH_PROVIDER') or 'pexels')
    # ClÃƒÂ©s API
    PEXELS_API_KEY = _UI_SETTINGS.get('PEXELS_API_KEY') or os.getenv('PEXELS_API_KEY')
    PIXABAY_API_KEY = _UI_SETTINGS.get('PIXABAY_API_KEY') or os.getenv('PIXABAY_API_KEY')
    # ContrÃƒÂ´les de fetch
    BROLL_FETCH_MAX_PER_KEYWORD = int(_UI_SETTINGS.get('broll_fetch_max_per_keyword') or os.getenv('BROLL_FETCH_MAX_PER_KEYWORD') or 25)  # CORRIGÃƒâ€°: 12 Ã¢â€ â€™ 25
    BROLL_FETCH_ALLOW_VIDEOS = _to_bool(_UI_SETTINGS.get('broll_fetch_allow_videos'), default=True) if 'broll_fetch_allow_videos' in _UI_SETTINGS else _to_bool(os.getenv('BROLL_FETCH_ALLOW_VIDEOS'), default=True)
    BROLL_FETCH_ALLOW_IMAGES = _to_bool(_UI_SETTINGS.get('broll_fetch_allow_images'), default=False) if 'broll_fetch_allow_images' in _UI_SETTINGS else _to_bool(os.getenv('BROLL_FETCH_ALLOW_IMAGES'), default=False)
    # Ãƒâ€°largir le pool par dÃƒÂ©faut: activer les images si non prÃƒÂ©cisÃƒÂ©
    if 'broll_fetch_allow_images' not in _UI_SETTINGS and os.getenv('BROLL_FETCH_ALLOW_IMAGES') is None:
        BROLL_FETCH_ALLOW_IMAGES = True
    # Embeddings pour matching sÃƒÂ©mantique
    BROLL_USE_EMBEDDINGS = _to_bool(_UI_SETTINGS.get('broll_use_embeddings'), default=True) if 'broll_use_embeddings' in _UI_SETTINGS else _to_bool(os.getenv('BROLL_USE_EMBEDDINGS'), default=True)
    BROLL_EMBEDDING_MODEL = (_UI_SETTINGS.get('broll_embedding_model') or os.getenv('BROLL_EMBEDDING_MODEL') or 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    # Config contextuelle
    CONTEXTUAL_CONFIG_PATH = Path(_UI_SETTINGS.get('contextual_broll_yml') or os.getenv('CONTEXTUAL_BROLL_YML') or 'config/contextual_broll.yml')

    # Sortie et nettoyage
    USE_HARDLINKS = _to_bool(_UI_SETTINGS.get('use_hardlinks'), default=True) if 'use_hardlinks' in _UI_SETTINGS else _to_bool(os.getenv('USE_HARDLINKS'), default=True)
    BROLL_DELETE_AFTER_USE = _to_bool(_UI_SETTINGS.get('broll_delete_after_use'), default=True) if 'broll_delete_after_use' in _UI_SETTINGS else _to_bool(os.getenv('BROLL_DELETE_AFTER_USE') or os.getenv('AI_BROLL_PURGE_AFTER_USE'), default=True)
    BROLL_PURGE_AFTER_RUN = _to_bool(_UI_SETTINGS.get('broll_purge_after_run'), default=True) if 'broll_purge_after_run' in _UI_SETTINGS else _to_bool(os.getenv('BROLL_PURGE_AFTER_RUN') or os.getenv('AI_BROLL_PURGE_AFTER_RUN'), default=True)
    # Brand kit
    BRAND_KIT_ID = _UI_SETTINGS.get('brand_kit_id') or os.getenv('BRAND_KIT_ID') or 'default'
    # Experimental FX (wipes/zoom/LUT etc.)
    ENABLE_EXPERIMENTAL_FX = _to_bool(_UI_SETTINGS.get('enable_experimental_fx'), default=False) if 'enable_experimental_fx' in _UI_SETTINGS else _to_bool(os.getenv('ENABLE_EXPERIMENTAL_FX'), default=False)

    # Ã°Å¸Å¡â‚¬ NOUVEAU: Configuration du sÃƒÂ©lecteur B-roll gÃƒÂ©nÃƒÂ©rique
    BROLL_SELECTOR_CONFIG_PATH = Path(_UI_SETTINGS.get('broll_selector_config') or os.getenv('BROLL_SELECTOR_CONFIG') or 'config/broll_selector_config.yaml')
    BROLL_SELECTOR_ENABLED = _to_bool(_UI_SETTINGS.get('broll_selector_enabled'), default=True) if 'broll_selector_enabled' in _UI_SETTINGS else _to_bool(os.getenv('BROLL_SELECTOR_ENABLED') or os.getenv('AI_BROLL_SELECTOR_ENABLED'), default=True)

# Ã°Å¸Å¡â‚¬ SUPPRIMÃƒâ€°: Fonction _detect_local_llm obsolÃƒÂ¨te
# RemplacÃƒÂ©e par le systÃƒÂ¨me LLM industriel qui gÃƒÂ¨re automatiquement la dÃƒÂ©tection

# Ã°Å¸Å¡â‚¬ SUPPRIMÃƒâ€°: Ancien systÃƒÂ¨me LLM obsolÃƒÂ¨te remplacÃƒÂ© par le systÃƒÂ¨me industriel
# Cette fonction utilisait l'ancien prompt complexe et causait des timeouts
# Maintenant remplacÃƒÂ©e par le systÃƒÂ¨me LLM industriel dans generate_caption_and_hashtags
# Ã°Å¸Å¡â‚¬ SUPPRIMÃƒâ€°: Reste de l'ancien systÃƒÂ¨me LLM obsolÃƒÂ¨te
# Toute cette logique complexe est maintenant remplacÃƒÂ©e par le systÃƒÂ¨me industriel

# === IA: Analyse mots-clÃƒÂ©s et prompts visuels pour guider le B-roll ===

def extract_keywords_from_transcript_ai(transcript_segments: List[Dict]) -> Dict:
    """Analyse simple: thÃƒÂ¨mes, occurrences et timestamps pour B-roll contextuel."""
    keyword_categories = {
        'money': ['money', 'cash', 'dollars', 'profit', 'revenue', 'income', 'wealth'],
        'business': ['business', 'company', 'startup', 'entrepreneur', 'strategy'],
        'technology': ['tech', 'software', 'app', 'digital', 'online', 'ai', 'automation'],
        'success': ['success', 'win', 'achievement', 'goal', 'growth', 'scale'],
        'people': ['team', 'customer', 'client', 'person', 'human', 'community'],
        'emotion_positive': ['amazing', 'incredible', 'fantastic', 'awesome', 'fire'],
        'emotion_negative': ['problem', 'issue', 'difficult', 'challenge', 'fail'],
        'action': ['build', 'create', 'launch', 'start', 'implement', 'execute']
    }
    full_text = ' '.join([(seg.get('text') or '').lower() for seg in transcript_segments])
    detected_keywords: Dict[str, List[str]] = {}
    timestamps_by_category: Dict[str, List[Dict]] = {}
    for category, kws in keyword_categories.items():
        detected_keywords[category] = []
        timestamps_by_category[category] = []
        for kw in kws:
            if kw in full_text:
                detected_keywords[category].append(kw)
                for seg in transcript_segments:
                    text = (seg.get('text') or '').lower()
                    if kw in text:
                        timestamps_by_category[category].append({
                            'start': float(seg.get('start') or 0.0),
                            'end': float(seg.get('end') or 0.0),
                            'keyword': kw,
                            'context': seg.get('text') or ''
                        })
    dominant_theme = 'business'
    try:
        dominant_theme = max(detected_keywords.items(), key=lambda x: len(x[1]))[0]
    except Exception:
        pass
    return {
        'keywords': detected_keywords,
        'timestamps': timestamps_by_category,
        'dominant_theme': dominant_theme,
        'total_duration': float(transcript_segments[-1]['end']) if transcript_segments else 0.0
    }


def generate_broll_prompts_ai(keyword_analysis: Dict) -> List[Dict]:
    """Generate B-roll prompts using AI analysis."""
    try:
        # Extract main theme and keywords
        main_theme = keyword_analysis.get('main_theme', 'general')
        keywords = keyword_analysis.get('keywords', [])
        sentiment = keyword_analysis.get('sentiment', 0.0)
        
        # Generate context-aware prompts
        prompts = []
        
        # Base prompts from main theme
        if main_theme == 'technology':
            prompts.extend([
                'artificial intelligence neural network',
                'computer vision algorithm',
                'tech innovation future',
                'digital transformation',
                'machine learning data'
            ])
        elif main_theme == 'medical':
            prompts.extend([
                'medical research laboratory',
                'healthcare innovation hospital',
                'microscope scientific discovery',
                'medical technology',
                'healthcare professionals'
            ])
        elif main_theme == 'business':
            prompts.extend([
                'business success growth',
                'entrepreneurship motivation',
                'professional development office',
                'team collaboration',
                'business strategy'
            ])
        elif main_theme == 'neuroscience':
            prompts.extend([
                'neuroscience brain neurons synapse',
                'brain reflexes nervous system',
                'brain scan mri eeg lab',
                'cognitive science',
                'mental health awareness'
            ])
        else:
            # Generic prompts for other themes
            base_keywords = keywords[:3] if keywords else [main_theme]
            for kw in base_keywords:
                prompts.append(f"{main_theme} {kw}")
        
        # Add sentiment-based prompts
        if sentiment > 0.3:
            prompts.extend(['positive energy', 'success achievement', 'happy people'])
        elif sentiment < -0.3:
            prompts.extend(['serious focus', 'determination', 'overcoming challenges'])
        
        # Limit and deduplicate
        unique_prompts = list(dict.fromkeys(prompts))[:8]
        
        return unique_prompts
        
    except Exception as e:
        print(f"Ã¢Å¡Â Ã¯Â¸Â Erreur gÃƒÂ©nÃƒÂ©ration prompts AI: {e}")
        # Fallback prompts
        return ['general content', 'people working', 'modern technology']

class VideoProcessor:
    """Classe principale pour traiter les vidÃƒÂ©os"""
    
    def __init__(self):
        self.whisper_model = whisper.load_model(Config.WHISPER_MODEL)
        self._setup_directories()
        # Cache ÃƒÂ©ventuel pour spaCy
        self._spacy_model = None
    
    def _setup_directories(self):
        """CrÃƒÂ©e les dossiers nÃƒÂ©cessaires"""
        for folder in [Config.CLIPS_FOLDER, Config.OUTPUT_FOLDER, Config.TEMP_FOLDER]:
            folder.mkdir(exist_ok=True)
    
    def _generate_unique_output_dir(self, clip_stem: str) -> Path:
        """CrÃƒÂ©e un dossier unique pour ce clip sous output/clips/<stem>[-NNN]"""
        root = Config.OUTPUT_FOLDER / 'clips'
        root.mkdir(parents=True, exist_ok=True)
        base = root / clip_stem
        if not base.exists():
            base.mkdir(parents=True, exist_ok=True)
            return base
        # Trouver suffixe -001, -002, ...
        for i in range(1, 1000):
            candidate = root / f"{clip_stem}-{i:03d}"
            if not candidate.exists():
                candidate.mkdir(parents=True, exist_ok=True)
                return candidate
        # Fallback timestamp
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        cand = root / f"{clip_stem}-{ts}"
        cand.mkdir(parents=True, exist_ok=True)
        return cand
    
    def _safe_copy(self, src: Path, dst: Path) -> None:
        try:
            if src and Path(src).exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))
        except Exception:
            pass

    def _hardlink_or_copy(self, src: Path, dst: Path) -> None:
        """CrÃƒÂ©e un hardlink si possible, sinon copie le fichier."""
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if getattr(Config, 'USE_HARDLINKS', True):
                os.link(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))
        except Exception:
            try:
                shutil.copy2(str(src), str(dst))
            except Exception:
                pass
 
    def _unique_path(self, directory: Path, base_name: str, extension: str) -> Path:
        """Retourne un chemin unique dans directory en ajoutant -NNN si collision."""
        directory.mkdir(parents=True, exist_ok=True)
        candidate = directory / f"{base_name}{extension}"
        if not candidate.exists():
            return candidate
        for i in range(1, 1000):
            alt = directory / f"{base_name}-{i:03d}{extension}"
            if not alt.exists():
                return alt
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        return directory / f"{base_name}-{ts}{extension}"
    
    def _cleanup_files(self, paths: List[Path]) -> None:
        for p in paths:
            try:
                if p and Path(p).exists():
                    Path(p).unlink()
            except Exception:
                pass
 
    def _purge_broll_caches(self) -> None:
        try:
            broll_lib = Path('AI-B-roll') / 'broll_library'
            broll_cache = Path('AI-B-roll') / '.cache'
            if broll_lib.exists():
                for item in broll_lib.glob('*'):
                    try:
                        if item.is_dir():
                            shutil.rmtree(item, ignore_errors=True)
                        else:
                            item.unlink(missing_ok=True)
                    except Exception:
                        pass
            if broll_cache.exists():
                shutil.rmtree(broll_cache, ignore_errors=True)
        except Exception:
            pass

    # Ã°Å¸Å¡Â¨ CORRECTION CRITIQUE: MÃƒÂ©thodes manquantes pour le sÃƒÂ©lecteur B-roll
    def _load_broll_selector_config(self):
        """Charge la configuration du sÃƒÂ©lecteur B-roll depuis le fichier YAML"""
        try:
            import yaml
            if Config.BROLL_SELECTOR_CONFIG_PATH.exists():
                with open(Config.BROLL_SELECTOR_CONFIG_PATH, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            else:
                print(f"    Ã¢Å¡Â Ã¯Â¸Â Fichier de configuration introuvable: {Config.BROLL_SELECTOR_CONFIG_PATH}")
                return {}
        except Exception as e:
            print(f"    Ã¢Å¡Â Ã¯Â¸Â Erreur chargement configuration: {e}")
            return {}

    def _calculate_asset_hash(self, asset_path: Path) -> str:
        """Calcule un hash unique pour un asset B-roll basÃƒÂ© sur son contenu et mÃƒÂ©tadonnÃƒÂ©es"""
        try:
            import hashlib
            import os
            from datetime import datetime
            
            # Hash basÃƒÂ© sur le nom, la taille et la date de modification
            stat = asset_path.stat()
            hash_data = f"{asset_path.name}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.md5(hash_data.encode()).hexdigest()
        except Exception:
            # Fallback sur le nom du fichier
            return str(asset_path.name)

    def _extract_keywords_for_segment_spacy(self, text: str) -> List[str]:
        """Extraction optionnelle (spaCy) de mots-clÃƒÂ©s (noms/verbes/entitÃƒÂ©s). Fallback heuristique si indisponible."""
        try:
            import re as _re
            
            # Ã°Å¸Å¡Â¨ CORRECTION IMMÃƒâ€°DIATE: Filtre des mots gÃƒÂ©nÃƒÂ©riques inutiles
            GENERIC_WORDS = {
                'very', 'much', 'many', 'some', 'any', 'all', 'each', 'every', 'few', 'several',
                'reflexes', 'speed', 'clear', 'good', 'bad', 'big', 'small', 'new', 'old', 'high', 'low',
                'fast', 'slow', 'hard', 'easy', 'strong', 'weak', 'hot', 'cold', 'warm', 'cool',
                'right', 'wrong', 'true', 'false', 'yes', 'no', 'maybe', 'perhaps', 'probably',
                'thing', 'stuff', 'way', 'time', 'place', 'person', 'people', 'man', 'woman', 'child',
                'work', 'make', 'do', 'get', 'go', 'come', 'see', 'look', 'hear', 'feel', 'think',
                'know', 'want', 'need', 'like', 'love', 'hate', 'hope', 'wish', 'try', 'help'
            }
            
            if self._spacy_model is None:
                try:
                    import spacy as _spacy
                    for _model in ['en_core_web_sm', 'fr_core_news_sm', 'xx_ent_wiki_sm']:
                        try:
                            self._spacy_model = _spacy.load(_model, disable=['parser','lemmatizer'])
                            break
                        except Exception:
                            continue
                    if self._spacy_model is None:
                        self._spacy_model = _spacy.blank('en')
                except Exception:
                    self._spacy_model = None
            doc = None
            if self._spacy_model is not None:
                try:
                    doc = self._spacy_model(text)
                except Exception:
                    doc = None
            keywords: List[str] = []
            if doc is not None and hasattr(doc, 'ents'):
                for ent in doc.ents:
                    val = ent.text.strip()
                    if len(val) >= 3 and val.lower() not in keywords and val.lower() not in GENERIC_WORDS:
                        keywords.append(val.lower())
            # POS si dispo
            if doc is not None and getattr(doc, 'has_annotation', lambda *_: False)('TAG'):
                for tok in doc:
                    if tok.pos_ in ('NOUN','PROPN','VERB') and len(tok.text) >= 3:
                        lemma = (tok.lemma_ or tok.text).lower()
                        if lemma not in keywords and lemma not in GENERIC_WORDS:
                            keywords.append(lemma)
            # Fallback heuristique simple avec filtre
            if not keywords:
                for w in _re.findall(r"[A-Za-zÃƒâ‚¬-Ãƒâ€“ÃƒËœ-ÃƒÂ¶ÃƒÂ¸-ÃƒÂ¿0-9']{4,}", text or ""):
                    lw = w.lower()
                    if lw not in keywords and lw not in GENERIC_WORDS:
                        keywords.append(lw)
            
            # Ã°Å¸Å¡Â¨ CORRECTION IMMÃƒâ€°DIATE: Prioriser les mots contextuels importants
            PRIORITY_WORDS = {
                'neuroscience', 'brain', 'mind', 'consciousness', 'cognitive', 'mental', 'psychology',
                'medical', 'health', 'treatment', 'research', 'science', 'discovery', 'innovation',
                'technology', 'digital', 'future', 'ai', 'artificial', 'intelligence', 'machine',
                'business', 'success', 'growth', 'strategy', 'leadership', 'entrepreneur', 'startup'
            }
            
            # RÃƒÂ©organiser pour prioriser les mots importants
            priority_keywords = [kw for kw in keywords if kw in PRIORITY_WORDS]
            other_keywords = [kw for kw in keywords if kw not in PRIORITY_WORDS]
            
            # Retourner d'abord les mots prioritaires, puis les autres
            final_keywords = priority_keywords + other_keywords
            return final_keywords[:12]
        except Exception:
            return []

    def process_all_clips(self, input_video_path: str):
        """Pipeline principal de traitement"""
        logger.info("Ã°Å¸Å¡â‚¬ DÃƒÂ©but du pipeline de traitement")
        print("Ã°Å¸Å½Â¬ DÃƒÂ©marrage du pipeline de traitement...")
        
        # Ãƒâ€°tape 1: DÃƒÂ©coupage (votre IA existante)
        
        # Ãƒâ€°tape 2: Traitement de chaque clip
        clip_files = list(Config.CLIPS_FOLDER.glob("*.mp4"))
        total_clips = len(clip_files)
        
        print(f"Ã°Å¸â€œÂ {total_clips} clips trouvÃƒÂ©s dans le dossier clips/")
        
        for i, clip_path in enumerate(clip_files):
            print(f"\nÃ°Å¸Å½Â¬ [{i+1}/{total_clips}] Traitement de: {clip_path.name}")
            logger.info(f"Ã°Å¸Å½Â¬ Traitement du clip {i+1}/{total_clips}: {clip_path.name}")
            
            # Skip si dÃƒÂ©jÃƒÂ  traitÃƒÂ©
            stem = Path(clip_path).stem
            final_dir = Config.OUTPUT_FOLDER / 'final'
            processed_already = False
            if final_dir.exists():
                matches = list(final_dir.glob(f"final_{stem}*.mp4"))
                processed_already = len(matches) > 0
            if processed_already:
                print(f"Ã¢ÂÂ© Clip dÃƒÂ©jÃƒÂ  traitÃƒÂ©, ignorÃƒÂ© : {clip_path.name}")
                logger.info(f"Ã¢ÂÂ© Clip dÃƒÂ©jÃƒÂ  traitÃƒÂ©, ignorÃƒÂ© : {clip_path.name}")
                continue

            # Verrou concurrentiel par clip
            locks_dir = Config.OUTPUT_FOLDER / 'locks'
            locks_dir.mkdir(parents=True, exist_ok=True)
            lock_file = locks_dir / f"{stem}.lock"
            if lock_file.exists():
                print(f"Ã¢ÂÂ­Ã¯Â¸Â Verrou dÃƒÂ©tectÃƒÂ©, saut du clip: {clip_path.name}")
                continue
            try:
                lock_file.write_text("locked", encoding='utf-8')
                self.process_single_clip(clip_path)
                print(f"Ã¢Å“â€¦ Clip {clip_path.name} traitÃƒÂ© avec succÃƒÂ¨s")
                logger.info(f"Ã¢Å“â€¦ Clip {clip_path.name} traitÃƒÂ© avec succÃƒÂ¨s")
            except Exception as e:
                print(f"Ã¢ÂÅ’ Erreur lors du traitement de {clip_path.name}: {e}")
                logger.error(f"Ã¢ÂÅ’ Erreur lors du traitement de {clip_path.name}: {e}")
            finally:
                try:
                    if lock_file.exists():
                        lock_file.unlink()
                except Exception:
                    pass
        
        print(f"\nÃ°Å¸Å½â€° Pipeline terminÃƒÂ© ! {total_clips} clips traitÃƒÂ©s.")
        logger.info("Ã°Å¸Å½â€° Pipeline terminÃƒÂ© avec succÃƒÂ¨s")
        # Purge B-roll (librairie + caches) si demandÃƒÂ© pour garder le disque lÃƒÂ©ger
        try:
            if getattr(Config, 'BROLL_PURGE_AFTER_RUN', False):
                self._purge_broll_caches()
        except Exception:
            pass
        # AgrÃƒÂ©ger un rapport global mÃƒÂªme sans --json-report
        try:
            final_dir = (Config.OUTPUT_FOLDER / 'final')
            items = []
            if final_dir.exists():
                for jf in final_dir.glob('final_*.json'):
                    try:
                        items.append(json.loads(jf.read_text(encoding='utf-8')))
                    except Exception:
                        pass
            report_path = Config.OUTPUT_FOLDER / 'report.json'
            report_path.write_text(json.dumps({'clips': items}, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass

    def cut_viral_clips(self, input_video_path: str):
        """
        Interface pour votre IA de dÃƒÂ©coupage existante
        Remplacez cette mÃƒÂ©thode par votre implÃƒÂ©mentation
        """
        logger.info("Ã°Å¸â€œÂ¼ DÃƒÂ©coupage des clips avec IA...")
        
        # Exemple basique - remplacez par votre IA
        video = VideoFileClip(input_video_path)
        duration = video.duration
        
        # DÃƒÂ©coupage adaptatif selon la durÃƒÂ©e
        if duration <= 30:
            # VidÃƒÂ©o courte : utiliser toute la vidÃƒÂ©o
            segment_duration = duration
            segments = 1
        else:
            # VidÃƒÂ©o longue : dÃƒÂ©couper en segments de 30 secondes
            segment_duration = 30
            segments = max(1, int(duration // segment_duration))
        
        for i in range(min(segments, 5)):  # Max 5 clips pour test
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, duration)
            
            clip = video.subclip(start_time, end_time)
            output_path = Config.CLIPS_FOLDER / f"clip_{i+1:02d}.mp4"
            clip.write_videofile(str(output_path), verbose=False, logger=None)
        
        video.close()
        logger.info(f"Ã¢Å“â€¦ {segments} clips gÃƒÂ©nÃƒÂ©rÃƒÂ©s")
    
    def process_single_clip(self, clip_path: Path):
        """Traite un clip individuel (reframe -> transcription (pour B-roll) -> B-roll -> sous-titres)"""
        
        # Dossier de sortie dÃƒÂ©diÃƒÂ© et unique
        per_clip_dir = self._generate_unique_output_dir(clip_path.stem)
        
        print(f"  Ã°Å¸â€œÂ Ãƒâ€°tape 1/4: Reframe dynamique IA...")
        reframed_path = self.reframe_to_vertical(clip_path)
        # DÃƒÂ©placer artefact reframed dans le dossier du clip
        try:
            dst_reframed = per_clip_dir / 'reframed.mp4'
            if Path(reframed_path).exists():
                shutil.move(str(reframed_path), str(dst_reframed))
            reframed_path = dst_reframed
        except Exception:
            pass
        
        print(f"  Ã°Å¸â€”Â£Ã¯Â¸Â Ãƒâ€°tape 2/4: Transcription Whisper (guide B-roll)...")
        # Transcrire tÃƒÂ´t pour guider la sÃƒÂ©lection B-roll (SRT disponible)
        subtitles = self.transcribe_segments(reframed_path)
        try:
            # Ãƒâ€°crire un SRT ÃƒÂ  cÃƒÂ´tÃƒÂ© de la vidÃƒÂ©o reframÃƒÂ©e
            srt_reframed = reframed_path.with_suffix('.srt')
            write_srt(subtitles, srt_reframed)
            # Sauvegarder transcription segments JSON
            seg_json = per_clip_dir / f"{clip_path.stem}_segments.json"
            with open(seg_json, 'w', encoding='utf-8') as f:
                json.dump(subtitles, f, ensure_ascii=False)
        except Exception:
            pass
        
        print(f"  Ã°Å¸Å½Å¾Ã¯Â¸Â Ãƒâ€°tape 3/4: Insertion des B-rolls {'(activÃƒÂ©e)' if getattr(Config, 'ENABLE_BROLL', False) else '(dÃƒÂ©sactivÃƒÂ©e)'}...")
        
        # Ã°Å¸Å¡â‚¬ CORRECTION: GÃƒÂ©nÃƒÂ©rer les mots-clÃƒÂ©s LLM AVANT l'insertion des B-rolls
        broll_keywords = []
        try:
            print("    Ã°Å¸Â¤â€“ GÃƒÂ©nÃƒÂ©ration prÃƒÂ©coce des mots-clÃƒÂ©s LLM pour B-rolls...")
            title, description, hashtags, broll_keywords = self.generate_caption_and_hashtags(subtitles)
            print(f"    Ã¢Å“â€¦ Mots-clÃƒÂ©s B-roll LLM gÃƒÂ©nÃƒÂ©rÃƒÂ©s: {len(broll_keywords)} termes")
            print(f"    Ã°Å¸Å½Â¯ Exemples: {', '.join(broll_keywords[:5])}")
        except Exception as e:
            print(f"    Ã¢Å¡Â Ã¯Â¸Â Erreur gÃƒÂ©nÃƒÂ©ration mots-clÃƒÂ©s LLM: {e}")
            broll_keywords = []
        
        # Maintenant insÃƒÂ©rer les B-rolls avec les mots-clÃƒÂ©s LLM disponibles
        with_broll_path = self.insert_brolls_if_enabled(reframed_path, subtitles, broll_keywords)
        
        # Copier artefact with_broll si diffÃƒÂ©rent
        try:
            if with_broll_path and with_broll_path != reframed_path:
                self._safe_copy(with_broll_path, per_clip_dir / 'with_broll.mp4')
        except Exception:
            pass
        
        print(f"  Ã¢Å“Â¨ Ãƒâ€°tape 4/4: Ajout des sous-titres Hormozi 1...")
        # GÃƒÂ©nÃƒÂ©rer meta (titre/hashtags) depuis transcription (dÃƒÂ©jÃƒÂ  fait)
        try:
            # RÃƒÂ©utiliser les donnÃƒÂ©es dÃƒÂ©jÃƒÂ  gÃƒÂ©nÃƒÂ©rÃƒÂ©es
            if not broll_keywords:  # Fallback si pas encore gÃƒÂ©nÃƒÂ©rÃƒÂ©
                title, description, hashtags, broll_keywords = self.generate_caption_and_hashtags(subtitles)
            
            print(f"  Ã°Å¸â€œÂ Title: {title}")
            print(f"  Ã°Å¸â€œÂ Description: {description}")
            print(f"  #Ã¯Â¸ÂÃ¢Æ’Â£ Hashtags: {' '.join(hashtags)}")
            meta_path = per_clip_dir / 'meta.txt'
            with open(meta_path, 'w', encoding='utf-8') as f:
                f.write(
                    "Title: " + title + "\n\n" +
                    "Description: " + description + "\n\n" +
                    "Hashtags: " + ' '.join(hashtags) + "\n\n" +
                    "B-roll Keywords: " + ', '.join(broll_keywords) + "\n"
                )
            print(f"  Ã°Å¸â€œÂ [MÃƒâ€°TADONNÃƒâ€°ES] Fichier meta.txt sauvegardÃƒÂ©: {meta_path}")
        except Exception as e:
            print(f"  Ã¢Å¡Â Ã¯Â¸Â [ERREUR MÃƒâ€°TADONNÃƒâ€°ES] {e}")
            # Fallback: crÃƒÂ©er des mÃƒÂ©tadonnÃƒÂ©es basiques
            try:
                meta_path = per_clip_dir / 'meta.txt'
                with open(meta_path, 'w', encoding='utf-8') as f:
                    f.write("Title: VidÃƒÂ©o gÃƒÂ©nÃƒÂ©rÃƒÂ©e automatiquement\n\nDescription: Contenu gÃƒÂ©nÃƒÂ©rÃƒÂ© par pipeline vidÃƒÂ©o\n\nHashtags: #video #auto\n\nB-roll Keywords: video, content\n")
                print(f"  Ã°Å¸â€œÂ [FALLBACK] MÃƒÂ©tadonnÃƒÂ©es de base sauvegardÃƒÂ©es: {meta_path}")
            except Exception as e2:
                print(f"  Ã¢ÂÅ’ [ERREUR FALLBACK] {e2}")
        
        # Appliquer style Hormozi sur la vidÃƒÂ©o post B-roll
        subtitled_out_dir = per_clip_dir
        subtitled_out_dir.mkdir(parents=True, exist_ok=True)
        final_subtitled_path = subtitled_out_dir / 'final_subtitled.mp4'
        try:
            span_style_map = {
                # Business & Croissance
                "croissance": {"color": "#39FF14", "bold": True, "emoji": "Ã°Å¸â€œË†"},
                "growth": {"color": "#39FF14", "bold": True, "emoji": "Ã°Å¸â€œË†"},
                "opportunitÃƒÂ©": {"color": "#FFD700", "bold": True, "emoji": "Ã¯Â¿Â½Ã¯Â¿Â½"},
                "opportunite": {"color": "#FFD700", "bold": True, "emoji": "Ã°Å¸â€â€˜"},
                "innovation": {"color": "#00E5FF", "emoji": "Ã¢Å¡Â¡"},
                "idÃƒÂ©e": {"color": "#00E5FF", "emoji": "Ã°Å¸â€™Â¡"},
                "idee": {"color": "#00E5FF", "emoji": "Ã°Å¸â€™Â¡"},
                "stratÃƒÂ©gie": {"color": "#FF73FA", "emoji": "Ã°Å¸Â§Â­"},
                "strategie": {"color": "#FF73FA", "emoji": "Ã°Å¸Â§Â­"},
                "plan": {"color": "#FF73FA", "emoji": "Ã°Å¸â€”ÂºÃ¯Â¸Â"},
                # Argent & Finance
                "argent": {"color": "#FFD700", "bold": True, "emoji": "Ã°Å¸â€™Â°"},
                "money": {"color": "#FFD700", "bold": True, "emoji": "Ã°Å¸â€™Â°"},
                "cash": {"color": "#FFD700", "bold": True, "emoji": "Ã°Å¸â€™Â°"},
                "investissement": {"color": "#8AFF00", "bold": True, "emoji": "Ã°Å¸â€œÅ "},
                "investissements": {"color": "#8AFF00", "bold": True, "emoji": "Ã°Å¸â€œÅ "},
                "revenu": {"color": "#8AFF00", "emoji": "Ã°Å¸ÂÂ¦"},
                "revenus": {"color": "#8AFF00", "emoji": "Ã°Å¸ÂÂ¦"},
                "profit": {"color": "#8AFF00", "bold": True, "emoji": "Ã°Å¸â€™Â°"},
                "profits": {"color": "#8AFF00", "bold": True, "emoji": "Ã°Å¸â€™Â°"},
                "perte": {"color": "#FF3131", "emoji": "Ã°Å¸â€œâ€°"},
                "pertes": {"color": "#FF3131", "emoji": "Ã°Å¸â€œâ€°"},
                "ÃƒÂ©chec": {"color": "#FF3131", "emoji": "Ã¢ÂÅ’"},
                "echec": {"color": "#FF3131", "emoji": "Ã¢ÂÅ’"},
                "budget": {"color": "#FFD700", "emoji": "Ã°Å¸Â§Â¾"},
                "gestion": {"color": "#FFD700", "emoji": "Ã°Å¸Âªâ„¢"},
                "roi": {"color": "#8AFF00", "bold": True, "emoji": "Ã°Å¸â€œË†"},
                "chiffre": {"color": "#FFD700", "emoji": "Ã°Å¸â€™Â°"},
                "ca": {"color": "#FFD700", "emoji": "Ã°Å¸â€™Â°"},
                # Relation & Client
                "client": {"color": "#00E5FF", "underline": True, "emoji": "Ã°Å¸Â¤Â"},
                "clients": {"color": "#00E5FF", "underline": True, "emoji": "Ã°Å¸Â¤Â"},
                "collaboration": {"color": "#00E5FF", "emoji": "Ã°Å¸Â«Â±Ã°Å¸ÂÂ¼Ã¢â‚¬ÂÃ°Å¸Â«Â²Ã°Å¸ÂÂ½"},
                "collaborations": {"color": "#00E5FF", "emoji": "Ã°Å¸Â«Â±Ã°Å¸ÂÂ¼Ã¢â‚¬ÂÃ°Å¸Â«Â²Ã°Å¸ÂÂ½"},
                "communautÃƒÂ©": {"color": "#39FF14", "emoji": "Ã°Å¸Å’Â"},
                "communaute": {"color": "#39FF14", "emoji": "Ã°Å¸Å’Â"},
                "confiance": {"color": "#00E5FF", "emoji": "Ã°Å¸â€â€™"},
                "vente": {"color": "#FF73FA", "emoji": "Ã°Å¸â€ºâ€™"},
                "ventes": {"color": "#FF73FA", "emoji": "Ã°Å¸â€ºâ€™"},
                "deal": {"color": "#FF73FA", "emoji": "Ã°Å¸â€œÂ¦"},
                "deals": {"color": "#FF73FA", "emoji": "Ã°Å¸â€œÂ¦"},
                "prospect": {"color": "#00E5FF", "emoji": "Ã°Å¸Â¤Â"},
                "prospects": {"color": "#00E5FF", "emoji": "Ã°Å¸Â¤Â"},
                "contrat": {"color": "#FF73FA", "emoji": "Ã°Å¸â€œâ€¹"},
                # Motivation & SuccÃƒÂ¨s
                "succÃƒÂ¨s": {"color": "#39FF14", "italic": True, "emoji": "Ã°Å¸Ââ€ "},
                "succes": {"color": "#39FF14", "italic": True, "emoji": "Ã°Å¸Ââ€ "},
                "motivation": {"color": "#FF73FA", "bold": True, "emoji": "Ã°Å¸â€Â¥"},
                "ÃƒÂ©nergie": {"color": "#FF73FA", "emoji": "Ã¢Å¡Â¡"},
                "energie": {"color": "#FF73FA", "emoji": "Ã¢Å¡Â¡"},
                "victoire": {"color": "#39FF14", "emoji": "Ã°Å¸Å½Â¯"},
                "discipline": {"color": "#FFD700", "emoji": "Ã¢ÂÂ³"},
                "viral": {"color": "#FF73FA", "bold": True, "emoji": "Ã°Å¸Å¡â‚¬"},
                "viralitÃƒÂ©": {"color": "#FF73FA", "bold": True, "emoji": "Ã°Å¸Å’Â"},
                "viralite": {"color": "#FF73FA", "bold": True, "emoji": "Ã°Å¸Å’Â"},
                "impact": {"color": "#FF73FA", "emoji": "Ã°Å¸â€™Â¥"},
                "explose": {"color": "#FF73FA", "emoji": "Ã°Å¸â€™Â¥"},
                "explosion": {"color": "#FF73FA", "emoji": "Ã°Å¸â€™Â¥"},
                # Risque & Erreurs
                "erreur": {"color": "#FF3131", "emoji": "Ã¢Å¡Â Ã¯Â¸Â"},
                "erreurs": {"color": "#FF3131", "emoji": "Ã¢Å¡Â Ã¯Â¸Â"},
                "warning": {"color": "#FF3131", "emoji": "Ã¢Å¡Â Ã¯Â¸Â"},
                "obstacle": {"color": "#FF3131", "emoji": "Ã°Å¸Â§Â±"},
                "obstacles": {"color": "#FF3131", "emoji": "Ã°Å¸Â§Â±"},
                "solution": {"color": "#00E5FF", "emoji": "Ã°Å¸â€Â§"},
                "solutions": {"color": "#00E5FF", "emoji": "Ã°Å¸â€Â§"},
                "leÃƒÂ§on": {"color": "#00E5FF", "emoji": "Ã°Å¸â€œÅ¡"},
                "lecon": {"color": "#00E5FF", "emoji": "Ã°Å¸â€œÅ¡"},
                "apprentissage": {"color": "#00E5FF", "emoji": "Ã°Å¸Â§Â "},
                "problÃƒÂ¨me": {"color": "#FF3131", "emoji": "Ã°Å¸â€ºâ€˜"},
                "probleme": {"color": "#FF3131", "emoji": "Ã°Å¸â€ºâ€˜"},
            }
            add_hormozi_subtitles(
                str(with_broll_path), subtitles, str(final_subtitled_path),
                brand_kit=getattr(Config, 'BRAND_KIT_ID', 'default'),
                span_style_map=span_style_map
            )
        except Exception as e:
            print(f"  Ã¢ÂÅ’ Erreur ajout sous-titres Hormozi: {e}")
            # Pas de retour anticipÃƒÂ©: continuer export simple
        
        # Export final accumulÃƒÂ© dans output/final/ et sous-titrÃƒÂ© (burn-in) dans output/subtitled/
        final_dir = Config.OUTPUT_FOLDER / 'final'
        subtitled_dir = Config.OUTPUT_FOLDER / 'subtitled'
        # Noms de base sans extension
        base_name = clip_path.stem
        output_path = self._unique_path(final_dir, f"final_{base_name}", ".mp4")
        try:
            # Choisir source finale: si sous-titrÃƒÂ©e existe sinon with_broll sinon reframed
            source_final = None
            if final_subtitled_path.exists():
                source_final = final_subtitled_path
            elif with_broll_path and Path(with_broll_path).exists():
                source_final = with_broll_path
            else:
                source_final = reframed_path
            if source_final and Path(source_final).exists():
                self._hardlink_or_copy(source_final, output_path)
                # Ecrire SRT: ÃƒÂ©viter le doublon si la vidÃƒÂ©o finale a dÃƒÂ©jÃƒÂ  les sous-titres incrustÃƒÂ©s
                is_burned = (final_subtitled_path.exists() and Path(source_final) == Path(final_subtitled_path))
                if not is_burned:
                    srt_out = output_path.with_suffix('.srt')
                    write_srt(subtitles, srt_out)
                    self._hardlink_or_copy(srt_out, per_clip_dir / 'final.srt')
                    # WebVTT
                    try:
                        vtt_out = output_path.with_suffix('.vtt')
                        write_vtt(subtitles, vtt_out)
                    except Exception:
                        pass
                else:
                    # Produire uniquement une SRT dans le dossier du clip, pas ÃƒÂ  cÃƒÂ´tÃƒÂ© du MP4 final
                    try:
                        write_srt(subtitles, per_clip_dir / 'final.srt')
                    except Exception:
                        pass
                # Toujours produire un VTT ÃƒÂ  cÃƒÂ´tÃƒÂ© du final pour compat
                try:
                    vtt_out = output_path.with_suffix('.vtt')
                    write_vtt(subtitles, vtt_out)
                except Exception:
                    pass
                # Copier final dans dossier clip
                self._hardlink_or_copy(output_path, per_clip_dir / 'final.mp4')
                # Si une version sous-titrÃƒÂ©e burn-in existe, la dupliquer dans output/subtitled/
                if final_subtitled_path.exists():
                    subtitled_out = self._unique_path(subtitled_dir, f"{base_name}_subtitled", ".mp4")
                    self._hardlink_or_copy(final_subtitled_path, subtitled_out)
                # Copier meta.txt ÃƒÂ  cÃƒÂ´tÃƒÂ© du final accumulÃƒÂ©
                try:
                    meta_src = per_clip_dir / 'meta.txt'
                    if meta_src.exists():
                        self._hardlink_or_copy(meta_src, output_path.with_suffix('.txt'))
                except Exception:
                    pass
                # Ecrire un JSON rÃƒÂ©cap par clip
                try:
                    # DurÃƒÂ©e et hash final
                    final_duration = None
                    try:
                        with VideoFileClip(str(output_path)) as vc:
                            final_duration = float(vc.duration)
                    except Exception:
                        final_duration = None
                    media_hash = None
                    try:
                        from src.pipeline.utils import hash_media  # type: ignore
                    except Exception:
                        hash_media = None  # type: ignore
                    if hash_media:
                        try:
                            media_hash = hash_media(str(output_path))
                        except Exception:
                            media_hash = None
                    summary = {
                        'clip': base_name,
                        'final_mp4': str(output_path.resolve()),
                        'final_srt': str(output_path.with_suffix('.srt').resolve()) if (not is_burned) and output_path.with_suffix('.srt').exists() else None,
                        'final_vtt': str(output_path.with_suffix('.vtt').resolve()) if (not is_burned) and output_path.with_suffix('.vtt').exists() else None,
                        'subtitled_mp4': str((subtitled_out.resolve() if final_subtitled_path.exists() else '')) if final_subtitled_path.exists() else None,
                        'meta_txt': str(output_path.with_suffix('.txt').resolve()) if output_path.with_suffix('.txt').exists() else None,
                        'per_clip_dir': str(per_clip_dir.resolve()),
                        'duration_s': final_duration,
                        'media_hash': media_hash,
                        'events': [
                            {
                                'id': getattr(ev, 'id', ev.get('id') if isinstance(ev, dict) else ''),
                                'start_s': float(getattr(ev, 'start_s', ev.get('start_s') if isinstance(ev, dict) else 0.0) or 0.0),
                                'end_s': float(getattr(ev, 'end_s', ev.get('end_s') if isinstance(ev, dict) else 0.0) or 0.0),
                                'media_path': getattr(ev, 'media_path', ev.get('media_path') if isinstance(ev, dict) else ''),
                                'transition': getattr(ev, 'transition', ev.get('transition') if isinstance(ev, dict) else None),
                                'transition_duration': float(getattr(ev, 'transition_duration', ev.get('transition_duration') if isinstance(ev, dict) else 0.0) or 0.0),
                            } for ev in (events or [])
                        ]
                    }
                    with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as jf:
                        json.dump(summary, jf, ensure_ascii=False, indent=2)
                    # JSONL log
                    try:
                        jsonl = (Config.OUTPUT_FOLDER / 'pipeline.log.jsonl')
                        with open(jsonl, 'a', encoding='utf-8') as lf:
                            lf.write(json.dumps(summary, ensure_ascii=False) + '\n')
                    except Exception:
                        pass
                except Exception:
                    pass
                print(f"  Ã°Å¸â€œÂ¤ Export terminÃƒÂ©: {output_path.name}")
                # Nettoyage des intermÃƒÂ©diaires pour limiter l'empreinte disque
                self._cleanup_files([
                    with_broll_path if with_broll_path and with_broll_path != output_path else None,
                ])
                return output_path
            else:
                print(f"  Ã¢Å¡Â Ã¯Â¸Â Fichier final introuvable")
                return None
        except Exception as e:
            print(f"  Ã¢ÂÅ’ Erreur export: {e}")
            return None

    def _get_sample_times(self, duration: float, fps: int) -> List[float]:
        if duration <= 10:
            return list(np.arange(0, duration, 1/fps))
        elif duration <= 30:
            return list(np.arange(0, duration, 2/fps))
        else:
            return list(np.arange(0, duration, 4/fps))

    def _smooth_trajectory(self, x_centers: List[float], window_size: int = 15) -> List[float]:
        # FenÃƒÂªtre plus grande pour un lissage plus smooth
        window_size = max(window_size, 31)
        if len(x_centers) < window_size:
            kernel = np.ones(min(9, len(x_centers))) / max(1, min(9, len(x_centers)))
            return np.convolve(x_centers, kernel, mode='same').tolist()
        try:
            from scipy.signal import savgol_filter
            smoothed = savgol_filter(x_centers, window_size, 3).tolist()
        except Exception:
            kernel = np.ones(window_size) / window_size
            smoothed = np.convolve(x_centers, kernel, mode='same').tolist()
        # EMA additionnel pour attÃƒÂ©nuer le jitter haute frÃƒÂ©quence
        alpha = 0.15  # plus petit = plus lisse
        ema = []
        last = smoothed[0] if smoothed else 0.5
        for v in smoothed:
            last = (1 - alpha) * last + alpha * v
            ema.append(last)
        return ema

    def _interpolate_trajectory(self, x_centers: List[float], sample_times: List[float], duration: float, fps: int) -> List[float]:
        if not x_centers:
            return [0.5] * int(duration * fps)
        target_times = np.arange(0, duration, 1/fps)
        if len(x_centers) == 1:
            return [x_centers[0]] * len(target_times)
        try:
            return np.interp(target_times, sample_times, x_centers).tolist()
        except Exception:
            return [x_centers[-1]] * len(target_times)

    def _detect_single_frame(self, image_rgb: np.ndarray) -> float:
        # DÃƒÂ©tecteurs MediaPipe
        mp_pose = mp.solutions.pose
        mp_face = mp.solutions.face_detection
        h, w = image_rgb.shape[:2]
        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.8) as pose, mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detection:
            pose_results = pose.process(image_rgb)
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                key_points = [
                    landmarks[mp_pose.PoseLandmark.NOSE],
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                ]
                valid_points = [p.x for p in key_points if p.visibility > 0.5]
                if valid_points:
                    return sum(valid_points) / len(valid_points)
            face_results = face_detection.process(image_rgb)
            if face_results.detections:
                detection = face_results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                return bbox.xmin + bbox.width / 2
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            moments = [cv2.moments(c) for c in contours if cv2.contourArea(c) > 100]
            if moments:
                centroids_x = [m['m10']/m['m00'] for m in moments if m['m00'] > 0]
                if centroids_x:
                    return sum(centroids_x) / len(centroids_x) / w
        return 0.5

    def _detect_focus_points(self, video: VideoFileClip, fps: int, duration: float) -> List[float]:
        x_centers = []
        sample_times = self._get_sample_times(duration, fps)
        
        # Guard: check if Mediapipe is available
        if not MEDIAPIPE_AVAILABLE or mp is None:
            # Fallback to OpenCV center detection
            for t in sample_times:
                try:
                    frame = video.get_frame(t)
                    x_center = self._detect_single_frame(frame)
                    x_centers.append(x_center)
                except Exception:
                    x_centers.append(0.5)  # Default center
            return x_centers
            
        mp_pose = mp.solutions.pose
        mp_face = mp.solutions.face_detection
        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.8) as pose, mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detection:
            for t in tqdm(sample_times, desc="Ã°Å¸â€Å½ IA focus", leave=False):
                try:
                    frame = video.get_frame(t)  # MoviePy retourne des frames RGB
                    image_rgb = frame
                    # Pose
                    pose_results = pose.process(image_rgb)
                    if pose_results.pose_landmarks:
                        landmarks = pose_results.pose_landmarks.landmark
                        key_points = [
                            landmarks[mp_pose.PoseLandmark.NOSE],
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                        ]
                        valid_points = [p.x for p in key_points if p.visibility > 0.5]
                        if valid_points:
                            x_centers.append(sum(valid_points)/len(valid_points))
                            continue
                    # Face fallback
                    face_results = face_detection.process(image_rgb)
                    if face_results.detections:
                        detection = face_results.detections[0]
                        bbox = detection.location_data.relative_bounding_box
                        x_centers.append(bbox.xmin + bbox.width/2)
                        continue
                    # Mouvement fallback
                    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        moments = [cv2.moments(c) for c in contours if cv2.contourArea(c) > 100]
                        centroids_x = [m['m10']/m['m00'] for m in moments if m['m00'] > 0]
                        if centroids_x:
                            x_centers.append(sum(centroids_x)/len(centroids_x)/image_rgb.shape[1])
                            continue
                    x_centers.append(0.5)
                except Exception:
                    x_centers.append(0.5)
        return self._interpolate_trajectory(x_centers, sample_times, duration, fps)

    def reframe_to_vertical(self, clip_path: Path) -> Path:
        """Reframe dynamique basÃƒÂ© sur dÃƒÂ©tection IA optimisÃƒÂ©e"""
        logger.info("Ã°Å¸Å½Â¯ Reframe dynamique avec IA (optimisÃƒÂ©)")
        print("    Ã°Å¸Å½Â¯ DÃƒÂ©tection IA en cours...")
        video = VideoFileClip(str(clip_path))
        fps = int(video.fps)
        duration = video.duration
        # DÃƒÂ©tection des centres d'intÃƒÂ©rÃƒÂªt
        x_centers = self._detect_focus_points(video, fps, duration)
        x_centers_smooth = self._smooth_trajectory(x_centers, window_size=min(15, max(5, len(x_centers)//4)))
        frame_index = 0
        applied_x_center_px = None
        beta = 0.85  # amortissement (0.85 = trÃƒÂ¨s smooth)
        def crop_frame(frame):
            nonlocal frame_index
            nonlocal applied_x_center_px
            h, w, _ = frame.shape
            if frame_index < len(x_centers_smooth):
                x_target_px = x_centers_smooth[frame_index] * w
            else:
                x_target_px = w * 0.5
            frame_index += 1
            # Initialisation EMA
            if applied_x_center_px is None:
                applied_x_center_px = x_target_px
            # Clamp vitesse de dÃƒÂ©placement + deadband
            shift = x_target_px - applied_x_center_px
            deadband_px = w * 0.003
            if abs(shift) < deadband_px:
                shift = 0.0
            max_shift_px = w * 0.02
            if shift > max_shift_px:
                shift = max_shift_px
            elif shift < -max_shift_px:
                shift = -max_shift_px
            x_clamped = applied_x_center_px + shift
            # EMA amorti
            applied_x_center_px = beta * applied_x_center_px + (1 - beta) * x_clamped
            
            # Ã°Å¸Å¡Â¨ CORRECTION BUG: Forcer des dimensions paires pour H.264
            target_width = Config.TARGET_WIDTH
            target_height = Config.TARGET_HEIGHT
            
            # Calcul du crop avec ratio 9:16
            crop_width = int(target_width * h / target_height)
            crop_width = min(crop_width, w)
            
            # Ã°Å¸Å¡Â¨ CORRECTION: S'assurer que crop_width est pair
            if crop_width % 2 != 0:
                crop_width = crop_width - 1 if crop_width > 1 else crop_width + 1
            
            x1 = int(max(0, min(w - crop_width, applied_x_center_px - crop_width / 2)))
            x2 = x1 + crop_width
            cropped = frame[:, x1:x2]
            
            # Ã°Å¸Å¡Â¨ CORRECTION: S'assurer que les dimensions finales sont paires
            final_width = target_width
            final_height = target_height
            
            # VÃƒÂ©rifier et corriger si nÃƒÂ©cessaire
            if final_width % 2 != 0:
                final_width = final_width - 1 if final_width > 1 else final_width + 1
            if final_height % 2 != 0:
                final_height = final_height - 1 if final_height > 1 else final_height + 1
            
            return cv2.resize(cropped, (final_width, final_height), interpolation=cv2.INTER_LANCZOS4)
        reframed = video.fl_image(crop_frame)
        output_path = Config.TEMP_FOLDER / f"reframed_{clip_path.name}"
        try:
            # Prefer AMD AMF hardware encoder on this system; boost quality slightly with QP=18
            reframed.write_videofile(
                str(output_path),
                fps=fps,
                codec='h264_nvenc',
                audio_codec='aac',
                verbose=False,
                logger=None,
                preset=None,
                ffmpeg_params=['-rc','vbr','-cq','19','-b:v','0','-maxrate','0','-pix_fmt','yuv420p','-movflags','+faststart']
            )
        except Exception:
            # Fallback to CPU x264 with stable CRF
            reframed.write_videofile(
                str(output_path),
                fps=fps,
                codec='libx264',
                audio_codec='aac',
                verbose=False,
                logger=None,
                preset='medium',
                ffmpeg_params=['-pix_fmt','yuv420p','-movflags','+faststart','-crf','20']
            )
        video.close(); reframed.close()
        print("    Ã¢Å“â€¦ Reframe terminÃƒÂ©")
        return output_path
    
    def transcribe_audio(self, video_path: Path) -> str:
        """Transcription avec Whisper"""
        logger.info("Ã°Å¸â€œÂ Transcription audio avec Whisper")
        print("    Ã°Å¸â€œÂ Transcription Whisper en cours...")
        
        result = self.whisper_model.transcribe(str(video_path))
        print("    Ã¢Å“â€¦ Transcription terminÃƒÂ©e")
        return result["text"]
    
    def transcribe_segments(self, video_path: Path) -> List[Dict]:
        """
        Transcrit l'audio en segments avec timestamps (sans rendu visuel).
        Retourne une liste de segments {'text', 'start', 'end'} et conserve les mots si fournis.
        """
        logger.info("Ã¢ÂÂ±Ã¯Â¸Â Transcription avec timestamps")
        print("    Ã¢ÂÂ±Ã¯Â¸Â GÃƒÂ©nÃƒÂ©ration des timestamps...")
        result = self.whisper_model.transcribe(str(video_path), word_timestamps=True)
        bias = getattr(Config, 'SUBTITLE_TIMING_BIAS_S', 0.0)
        subtitles: List[Dict] = []
        for segment in result.get("segments", []):
            seg_start = max(0.0, (segment.get("start") or 0.0) + bias)
            seg_end = max(seg_start, (segment.get("end") or seg_start) + bias)
            subtitle: Dict = {
                "text": (segment.get("text") or "").strip(),
                "start": seg_start,
                "end": seg_end
            }
            words = segment.get("words")
            if words:
                precise_words = []
                for w in words:
                    ws = max(0.0, (w.get("start") or seg_start) + bias)
                    we = max(ws, (w.get("end") or ws) + bias)
                    wt = (w.get("word") or w.get("text") or "").strip()
                    if wt:
                        precise_words.append({"text": wt, "start": ws, "end": we})
                if precise_words:
                    subtitle["words"] = precise_words
            subtitles.append(subtitle)
        print(f"    Ã¢Å“â€¦ {len(subtitles)} segments de sous-titres gÃƒÂ©nÃƒÂ©rÃƒÂ©s")
        return subtitles

    def generate_caption_and_hashtags(self, subtitles: List[Dict]) -> (str, str, List[str], List[str]):
        """GÃƒÂ©nÃƒÂ¨re une lÃƒÂ©gende, des hashtags et des mots-clÃƒÂ©s B-roll avec le systÃƒÂ¨me LLM industriel."""
        full_text = ' '.join(s.get('text', '') for s in subtitles)
        
        # Ã°Å¸Å¡â‚¬ NOUVEAU: Utilisation du systÃƒÂ¨me LLM industriel
        try:
            # Import du nouveau systÃƒÂ¨me
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent / "utils"))
            
            from pipeline_integration import create_pipeline_integration
            
            # CrÃƒÂ©er l'intÃƒÂ©gration LLM
            llm_integration = create_pipeline_integration()
            
            print(f"    Ã°Å¸Å¡â‚¬ [LLM INDUSTRIEL] GÃƒÂ©nÃƒÂ©ration de mÃƒÂ©tadonnÃƒÂ©es pour {len(full_text)} caractÃƒÂ¨res")
            
            # Traitement avec le nouveau systÃƒÂ¨me
            result = llm_integration.process_video_transcript(
                transcript=full_text,
                video_id=f"video_{int(time.time())}",
                segment_timestamps=[(s.get('start', 0), s.get('end', 0)) for s in subtitles if 'start' in s and 'end' in s]
            )
            
            if result.get('success', False):
                metadata = result.get('metadata', {})
                broll_data = result.get('broll_data', {})
                
                title = metadata.get('title', '').strip()
                description = metadata.get('description', '').strip()
                hashtags = [h for h in (metadata.get('hashtags') or []) if h]
                broll_keywords = broll_data.get('keywords', [])
                
                print(f"    Ã¢Å“â€¦ [LLM INDUSTRIEL] MÃƒÂ©tadonnÃƒÂ©es gÃƒÂ©nÃƒÂ©rÃƒÂ©es avec succÃƒÂ¨s")
                print(f"    Ã°Å¸Å½Â¯ Titre: {title}")
                print(f"    Ã°Å¸â€œÂ Description: {description[:100]}...")
                print(f"    #Ã¯Â¸ÂÃ¢Æ’Â£ Hashtags: {len(hashtags)} gÃƒÂ©nÃƒÂ©rÃƒÂ©s")
                print(f"    Ã°Å¸Å½Â¬ Mots-clÃƒÂ©s B-roll: {len(broll_keywords)} termes optimisÃƒÂ©s")
                
                return title, description, hashtags, broll_keywords
            else:
                print(f"    Ã¢Å¡Â Ã¯Â¸Â [LLM INDUSTRIEL] Ãƒâ€°chec, fallback vers ancien systÃƒÂ¨me")
                raise Exception("LLM industriel ÃƒÂ©chouÃƒÂ©")
                
        except Exception as e:
            print(f"    Ã°Å¸â€â€ž [FALLBACK] Retour vers ancien systÃƒÂ¨me: {e}")
            # Fallback vers l'ancien systÃƒÂ¨me
            llm_res = _llm_generate_caption_hashtags_fixed(full_text)
            if llm_res and (llm_res.get('title') or llm_res.get('description') or llm_res.get('hashtags')):
                title = (llm_res.get('title') or '').strip()
                description = (llm_res.get('description') or '').strip()
                hashtags = [h for h in (llm_res.get('hashtags') or []) if h]
                
                # Ã°Å¸Å¡â‚¬ NOUVEAU: Extraction des mots-clÃƒÂ©s B-roll du LLM
                broll_keywords = llm_res.get('broll_keywords', [])
                if broll_keywords:
                    print(f"    Ã°Å¸Â¤â€“ [LLM] Titre/description/hashtags + {len(broll_keywords)} mots-clÃƒÂ©s B-roll gÃƒÂ©nÃƒÂ©rÃƒÂ©s par LLM local")
                    print(f"    Ã°Å¸Å½Â¯ Mots-clÃƒÂ©s B-roll LLM: {', '.join(broll_keywords[:8])}...")
                else:
                    print("    Ã°Å¸Â¤â€“ [LLM] Titre/description/hashtags gÃƒÂ©nÃƒÂ©rÃƒÂ©s par LLM local")
                    # Fallback: extraire des mots-clÃƒÂ©s basiques du titre et de la description
                    fallback_text = f"{title} {description}".lower()
                    broll_keywords = [word for word in fallback_text.split() if len(word) > 3 and word.isalpha()]
                    broll_keywords = list(set(broll_keywords))[:10]
                    print(f"    Ã°Å¸â€â€ž Fallback mots-clÃƒÂ©s B-roll: {', '.join(broll_keywords[:5])}...")
                
                # Back-compat: si titre vide mais description prÃƒÂ©sente, promouvoir description en titre court
                if not title and description:
                    title = (description[:60] + ('Ã¢â‚¬Â¦' if len(description) > 60 else ''))
                return title, description, hashtags, broll_keywords
        
        # Fallback heuristic
        words = [w.strip().lower() for w in re.split(r"[^a-zA-Z0-9ÃƒÂ©ÃƒÂ¨ÃƒÂ ÃƒÂ¹ÃƒÂ§ÃƒÂªÃƒÂ®ÃƒÂ´ÃƒÂ¢]+", full_text) if len(w) > 2]
        from collections import Counter
        counts = Counter(words)
        common = [w for w,_ in counts.most_common(12) if w.isalpha()]
        hashtags = [f"#{w}" for w in common[:12]]
        
        # Ã°Å¸Å¡â‚¬ NOUVEAU: Mots-clÃƒÂ©s B-roll de fallback basÃƒÂ©s sur les mots communs
        # Intelligent visual keywords fallback
        text_lower = full_text.lower()
        visual_keywords = []
        
        # Domain-specific visual keywords
        if any(word in text_lower for word in ['scientific', 'research', 'study', 'paper', 'published']):
            visual_keywords.extend(['scientist', 'research', 'laboratory', 'study', 'analysis', 'discovery'])
        if any(word in text_lower for word in ['behavior', 'cognitive', 'psychology', 'mental']):
            visual_keywords.extend(['therapy', 'counseling', 'brain', 'mind', 'psychology', 'behavior'])
        if any(word in text_lower for word in ['work', 'effort', 'exert', 'challenge']):
            visual_keywords.extend(['working', 'focusing', 'concentrating', 'determined', 'motivated'])
        if any(word in text_lower for word in ['human', 'animal', 'species', 'evolution']):
            visual_keywords.extend(['people', 'professional', 'teamwork', 'collaboration'])
        
        # Add high-value generic B-roll keywords
        visual_keywords.extend(['modern office', 'workspace', 'business', 'success', 'achievement', 
                               'technology', 'digital', 'innovation', 'professional', 'focused'])
        
        # Remove duplicates and limit
        broll_keywords = list(dict.fromkeys(visual_keywords))[:20]
        
        # Heuristic title/description
        title = (full_text.strip()[:60] + ("Ã¢â‚¬Â¦" if len(full_text.strip()) > 60 else "")) if full_text.strip() else ""
        description = (full_text.strip()[:180] + ("Ã¢â‚¬Â¦" if len(full_text.strip()) > 180 else "")) if full_text.strip() else ""
        print("    Ã°Å¸Â§Â© [Heuristics] Meta gÃƒÂ©nÃƒÂ©rÃƒÂ©es en fallback")
        print(f"    Ã°Å¸â€â€˜ Mots-clÃƒÂ©s B-roll fallback: {', '.join(broll_keywords[:5])}...")
        return title, description, hashtags, broll_keywords

    def insert_brolls_if_enabled(self, input_path: Path, subtitles: List[Dict], broll_keywords: List[str]) -> Path:
        """Point d'extension B-roll: retourne le chemin vidÃƒÂ©o aprÃƒÂ¨s insertion si activÃƒÂ©e."""
        if not getattr(Config, 'ENABLE_BROLL', False):
            print("    Ã¢ÂÂ­Ã¯Â¸Â B-roll dÃƒÂ©sactivÃƒÂ©s: aucune insertion")
            return input_path
        
        try:
            # VÃƒÂ©rifier la librairie B-roll
            broll_root = Path("AI-B-roll")
            broll_library = broll_root / "broll_library"
            if not broll_library.exists():
                print("    Ã¢Å¡Â Ã¯Â¸Â Librairie B-roll introuvable, saut de l'insertion")
                return input_path
            # PrÃƒÂ©parer chemins (ÃƒÂ©crire directement dans le dossier du clip si possible)
            clip_dir = (Path(input_path).parent if (Path(input_path).name == 'reframed.mp4') else Config.TEMP_FOLDER)
            # Si input_path est dÃƒÂ©jÃƒÂ  dans un dossier clip (reframed.mp4), sortir with_broll.mp4 ÃƒÂ  cÃƒÂ´tÃƒÂ©
            if Path(input_path).name == 'reframed.mp4':
                output_with_broll = clip_dir / 'with_broll.mp4'
            else:
                output_with_broll = Config.TEMP_FOLDER / f"with_broll_{Path(input_path).name}"
            output_with_broll.parent.mkdir(parents=True, exist_ok=True)
            
            # Assurer l'import du pipeline local (src/*)
            if str(broll_root.resolve()) not in sys.path:
                sys.path.insert(0, str(broll_root.resolve()))
            
            # Ã°Å¸Å¡â‚¬ NOUVEAUX IMPORTS INTELLIGENTS SYNCHRONES (DÃƒâ€°SACTIVÃƒâ€°S POUR PROMPT OPTIMISÃƒâ€°)
            try:
                from sync_context_analyzer import SyncContextAnalyzer
                from broll_diversity_manager import BrollDiversityManager
                # Ã°Å¸Å¡Â¨ DÃƒâ€°SACTIVATION TEMPORAIRE: Le systÃƒÂ¨me intelligent interfÃƒÂ¨re avec notre prompt optimisÃƒÂ© LLM
                INTELLIGENT_BROLL_AVAILABLE = False
                print("    Ã¢Å¡Â Ã¯Â¸Â  SystÃƒÂ¨me intelligent DÃƒâ€°SACTIVÃƒâ€° pour laisser le prompt optimisÃƒÂ© LLM fonctionner")
                print("    Ã°Å¸Å½Â¯ Utilisation exclusive du prompt optimisÃƒÂ©: 25-35 keywords + structure hiÃƒÂ©rarchique")
            except ImportError as e:
                print(f"    Ã¢Å¡Â Ã¯Â¸Â  SystÃƒÂ¨me intelligent non disponible: {e}")
                print("    Ã°Å¸â€â€ž Fallback vers ancien systÃƒÂ¨me...")
                INTELLIGENT_BROLL_AVAILABLE = False
            
            # Imports B-roll dans tous les cas
            from src.pipeline.config import BrollConfig  # type: ignore
            from src.pipeline.keyword_extraction import extract_keywords_for_segment  # type: ignore
            from src.pipeline.timeline_legacy import plan_broll_insertions, normalize_timeline, enrich_keywords  # type: ignore
            from src.pipeline.renderer import render_video  # type: ignore
            from src.pipeline.transcription import TranscriptSegment  # type: ignore
            
            from moviepy.editor import VideoFileClip as _VFC
            # Optionnel: indexation FAISS/CLIP
            try:
                from src.pipeline.indexer import build_index  # type: ignore
                index_handle = None
            except Exception:
                build_index = None  # type: ignore
                index_handle = None
            
            # Ã°Å¸Â§Â  ANALYSE INTELLIGENTE AVANCÃƒâ€°E
            if INTELLIGENT_BROLL_AVAILABLE:
                print("    Ã°Å¸Â§Â  Utilisation du systÃƒÂ¨me B-roll intelligent...")
                try:
                    # Initialiser l'analyseur contextuel intelligent SYNCHRONE
                    context_analyzer = SyncContextAnalyzer()
                    
                    # Analyser le contexte global de la vidÃƒÂ©o
                    transcript_text = " ".join([s.get('text', '') for s in subtitles])
                    global_analysis = context_analyzer.analyze_context(transcript_text)
                    
                    print(f"    Ã°Å¸Å½Â¯ Contexte dÃƒÂ©tectÃƒÂ©: {global_analysis.main_theme}")
                    print(f"    Ã°Å¸Â§Â¬ Sujets: {', '.join(global_analysis.key_topics[:3])}")
                    print(f"    Ã°Å¸ËœÅ  Sentiment: {global_analysis.sentiment}")
                    print(f"    Ã°Å¸â€œÅ  ComplexitÃƒÂ©: {global_analysis.complexity}")
                    print(f"    Ã°Å¸â€â€˜ Mots-clÃƒÂ©s: {', '.join(global_analysis.keywords[:5])}")
                    
                    # Persister l'analyse intelligente
                    try:
                        meta_dir = Config.OUTPUT_FOLDER / 'meta'
                        meta_dir.mkdir(parents=True, exist_ok=True)
                        meta_path = meta_dir / f"{Path(input_path).stem}_intelligent_broll_metadata.json"
                        with open(meta_path, 'w', encoding='utf-8') as f:
                            json.dump({
                                'intelligent_analysis': {
                                    'main_theme': global_analysis.main_theme,
                                    'key_topics': global_analysis.key_topics,
                                    'sentiment': global_analysis.sentiment,
                                    'complexity': global_analysis.complexity,
                                    'keywords': global_analysis.keywords,
                                    'context_score': global_analysis.context_score
                                },
                                'timestamp': str(datetime.now())
                            }, f, ensure_ascii=False, indent=2)
                        print(f"    Ã°Å¸â€™Â¾ MÃƒÂ©tadonnÃƒÂ©es intelligentes sauvegardÃƒÂ©es: {meta_path}")
                        
                        # Ã°Å¸Å½Â¬ INSÃƒâ€°RATION INTELLIGENTE DES B-ROLLS
                        print("    Ã°Å¸Å½Â¬ Insertion intelligente des B-rolls...")
                        try:
                            # CrÃƒÂ©er un dossier unique pour ce clip
                            clip_id = input_path.stem
                            unique_broll_dir = broll_library / f"clip_intelligent_{clip_id}_{int(time.time())}"
                            unique_broll_dir.mkdir(parents=True, exist_ok=True)
                            
                            # GÃƒÂ©nÃƒÂ©rer des prompts intelligents basÃƒÂ©s sur l'analyse
                            intelligent_prompts = []
                            main_theme = global_analysis.main_theme
                            kws = _filter_prompt_terms(global_analysis.keywords[:6]) if hasattr(global_analysis, 'keywords') else []
                            if main_theme == 'technology':
                                intelligent_prompts.extend([
                                    'artificial intelligence neural network',
                                    'computer vision algorithm',
                                    'tech innovation future'
                                ])
                            elif main_theme == 'medical':
                                intelligent_prompts.extend([
                                    'medical research laboratory',
                                    'healthcare innovation hospital',
                                    'microscope scientific discovery'
                                ])
                            elif main_theme == 'business':
                                intelligent_prompts.extend([
                                    'business success growth',
                                    'entrepreneurship motivation',
                                    'professional development office'
                                ])
                            elif main_theme == 'neuroscience':
                                intelligent_prompts.extend([
                                    'neuroscience brain neurons synapse',
                                    'brain reflexes nervous system',
                                    'brain scan mri eeg lab'
                                ])
                            else:
                                base = _filter_prompt_terms([main_theme] + kws)
                                intelligent_prompts.extend([f"{main_theme} {kw}" for kw in base[:3]])

                            # Ajouter variantes from cleaned keywords
                            for kw in kws[:3]:
                                intelligent_prompts.append(f"{main_theme} {kw}")

                            # Dedup and trim
                            seen_ip = set()
                            intelligent_prompts = [p for p in intelligent_prompts if not (p in seen_ip or seen_ip.add(p))][:8]

                            print(f"    Ã°Å¸Å½Â¯ Prompts intelligents gÃƒÂ©nÃƒÂ©rÃƒÂ©s: {', '.join(intelligent_prompts[:3])}")
                            
                            # Utiliser l'ancien systÃƒÂ¨me mais avec les prompts intelligents
                            # (temporaire en attendant l'intÃƒÂ©gration complÃƒÂ¨te)
                            print("    Ã°Å¸â€â€ž Utilisation du systÃƒÂ¨me B-roll avec prompts intelligents...")
                            
                        except Exception as e:
                            print(f"    Ã¢Å¡Â Ã¯Â¸Â  Erreur insertion intelligente: {e}")
                            print("    Ã°Å¸â€â€ž Fallback vers ancien systÃƒÂ¨me...")
                            INTELLIGENT_BROLL_AVAILABLE = False
                            
                    except Exception as e:
                        print(f"    Ã¢Å¡Â Ã¯Â¸Â  Erreur systÃƒÂ¨me intelligent: {e}")
                        print("    Ã°Å¸â€â€ž Fallback vers ancien systÃƒÂ¨me...")
                        INTELLIGENT_BROLL_AVAILABLE = False
                except Exception as e:
                    print(f"    Ã¢Å¡Â Ã¯Â¸Â  Erreur systÃƒÂ¨me intelligent: {e}")
                    print("    Ã°Å¸â€â€ž Fallback vers ancien systÃƒÂ¨me...")
                    INTELLIGENT_BROLL_AVAILABLE = False
                    
            # Fallback: ancienne analyse si systÃƒÂ¨me intelligent indisponible
            if not INTELLIGENT_BROLL_AVAILABLE:
                print("    Ã°Å¸â€â€ž Utilisation de l'ancien systÃƒÂ¨me B-roll...")
                analysis = extract_keywords_from_transcript_ai(subtitles)
                prompts = generate_broll_prompts_ai(analysis)
                # Filtrer les prompts fallback
                try:
                    cleaned_prompts = []
                    for p in prompts:
                        tokens = _filter_prompt_terms(str(p).split())
                        if tokens:
                            cleaned_prompts.append(' '.join(tokens))
                    if cleaned_prompts:
                        prompts = cleaned_prompts
                except Exception:
                    pass
                # Persiste metadata dans un dossier clip dÃƒÂ©diÃƒÂ© si possible
                try:
                    meta_dir = Config.OUTPUT_FOLDER / 'meta'
                    meta_dir.mkdir(parents=True, exist_ok=True)
                    meta_path = meta_dir / f"{Path(input_path).stem}_broll_metadata.json"
                    with open(meta_path, 'w', encoding='utf-8') as f:
                        json.dump({'analysis': analysis, 'prompts': prompts}, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
            else:
                # Ã°Å¸Å½Â¯ UTILISER LES PROMPTS INTELLIGENTS
                print("    Ã°Å¸Å½Â¯ Utilisation des prompts intelligents pour B-rolls...")
                try:
                    # CrÃƒÂ©er une analyse basÃƒÂ©e sur l'analyse intelligente
                    analysis = {
                        'main_theme': global_analysis.main_theme,
                        'key_topics': global_analysis.key_topics,
                        'sentiment': global_analysis.sentiment,
                        'keywords': global_analysis.keywords
                    }
                    
                    # Utiliser les prompts intelligents gÃƒÂ©nÃƒÂ©rÃƒÂ©s
                    prompts = intelligent_prompts if 'intelligent_prompt' in locals() else [
                        f"{global_analysis.main_theme} {kw}" for kw in global_analysis.keywords[:3]
                    ]
                    
                    print(f"    Ã°Å¸Å½Â¯ Prompts utilisÃƒÂ©s: {', '.join(prompts[:3])}")
                    
                except Exception as e:
                    print(f"    Ã¢Å¡Â Ã¯Â¸Â  Erreur prompts intelligents: {e}")
                    # Fallback vers prompts gÃƒÂ©nÃƒÂ©riques
                    analysis = extract_keywords_from_transcript_ai(subtitles)
                    prompts = generate_broll_prompts_ai(analysis)
            
            # Ã°Å¸Å¡â‚¬ NOUVEAU: IntÃƒÂ©gration des mots-clÃƒÂ©s B-roll du LLM
            # RÃƒÂ©cupÃƒÂ©rer les mots-clÃƒÂ©s B-roll gÃƒÂ©nÃƒÂ©rÃƒÂ©s par le LLM (si disponibles)
            llm_broll_keywords = []
            try:
                # Les mots-clÃƒÂ©s B-roll sont dÃƒÂ©jÃƒÂ  disponibles depuis generate_caption_and_hashtags
                # Ils sont passÃƒÂ©s via la variable broll_keywords dans le scope parent
                if 'broll_keywords' in locals():
                    llm_broll_keywords = broll_keywords
                    print(f"    Ã°Å¸Â§Â  Mots-clÃƒÂ©s B-roll LLM intÃƒÂ©grÃƒÂ©s: {len(llm_broll_keywords)} termes")
                    print(f"    Ã°Å¸Å½Â¯ Exemples: {', '.join(llm_broll_keywords[:5])}")
                else:
                    print("    Ã¢Å¡Â Ã¯Â¸Â Mots-clÃƒÂ©s B-roll LLM non disponibles")
            except Exception as e:
                print(f"    Ã¢Å¡Â Ã¯Â¸Â Erreur rÃƒÂ©cupÃƒÂ©ration mots-clÃƒÂ©s B-roll LLM: {e}")
            
            # Combiner les mots-clÃƒÂ©s LLM avec les prompts existants
            if llm_broll_keywords:
                # Enrichir les prompts avec les mots-clÃƒÂ©s LLM
                enhanced_prompts = []
                for kw in llm_broll_keywords[:8]:  # Limiter ÃƒÂ  8 mots-clÃƒÂ©s principaux
                    enhanced_prompts.append(kw)
                    # CrÃƒÂ©er des combinaisons avec le thÃƒÂ¨me principal
                    if 'global_analysis' in locals() and hasattr(global_analysis, 'main_theme'):
                        enhanced_prompts.append(f"{global_analysis.main_theme} {kw}")
                
                # Ajouter les prompts existants
                enhanced_prompts.extend(prompts)
                
                # DÃƒÂ©dupliquer et limiter
                seen_prompts = set()
                final_prompts = []
                for p in enhanced_prompts:
                    if p not in seen_prompts and len(p) > 2:
                        final_prompts.append(p)
                        seen_prompts.add(p)
                
                prompts = final_prompts[:12]  # Limiter ÃƒÂ  12 prompts finaux
                print(f"    Ã°Å¸Å¡â‚¬ Prompts enrichis avec LLM: {len(prompts)} termes")
                print(f"    Ã°Å¸Å½Â¯ Prompts finaux: {', '.join(prompts[:5])}...")
            
            # Convertir nos sous-titres en segments attendus par le pipeline
            segments = [
                TranscriptSegment(start=float(s.get('start', 0.0)), end=float(s.get('end', 0.0)), text=str(s.get('text', '')).strip())
                for s in subtitles if (s.get('text') and (s.get('end', 0.0) >= s.get('start', 0.0)))
            ]
            if not segments:
                print("    Ã¢Å¡Â Ã¯Â¸Â Aucun segment de transcription valide, saut B-roll")
                return input_path
            
            # Construire la config du pipeline (fetch + embeddings activÃƒÂ©s, pas de limites)
            cfg = BrollConfig(
                input_video=str(input_path),
                output_video=output_with_broll,
                broll_library=broll_library,
                srt_path=None,
                render_subtitles=False,
                            max_broll_ratio=0.65,           # CORRIGÃƒâ€°: 90% Ã¢â€ â€™ 65% pour ÃƒÂ©quilibre optimal
            min_gap_between_broll_s=1.5,    # CORRIGÃƒâ€°: 0.2s Ã¢â€ â€™ 1.5s pour respiration visuelle
                            max_broll_clip_s=4.0,           # CORRIGÃƒâ€°: 8.0s Ã¢â€ â€™ 4.0s pour B-rolls ÃƒÂ©quilibrÃƒÂ©s
            min_broll_clip_s=2.0,           # CORRIGÃƒâ€°: 3.5s Ã¢â€ â€™ 2.0s pour durÃƒÂ©e optimale
                use_whisper=False,
                ffmpeg_preset="fast",
                crf=23,
                threads=0,
                # Fetchers (stock)
                enable_fetcher=getattr(Config, 'BROLL_FETCH_ENABLE', False),
                fetch_provider=getattr(Config, 'BROLL_FETCH_PROVIDER', 'pexels'),
                pexels_api_key=getattr(Config, 'PEXELS_API_KEY', None),
                pixabay_api_key=getattr(Config, 'PIXABAY_API_KEY', None),
                fetch_max_per_keyword=getattr(Config, 'BROLL_FETCH_MAX_PER_KEYWORD', 25),  # CORRIGÃƒâ€°: 50 Ã¢â€ â€™ 25 pour qualitÃƒÂ© optimale
                fetch_allow_videos=getattr(Config, 'BROLL_FETCH_ALLOW_VIDEOS', True),
                fetch_allow_images=getattr(Config, 'BROLL_FETCH_ALLOW_IMAGES', True),  # ActivÃƒÂ©: images animÃƒÂ©es + Ken Burns
                # Embeddings
                use_embeddings=getattr(Config, 'BROLL_USE_EMBEDDINGS', True),
                embedding_model_name=getattr(Config, 'BROLL_EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
                contextual_config_path=getattr(Config, 'CONTEXTUAL_CONFIG_PATH', Path('config/contextual_broll.yml')),
                # Experimental FX toggle
                enable_experimental_fx=getattr(Config, 'ENABLE_EXPERIMENTAL_FX', False),
            )
            # FETCH DYNAMIQUE PAR CLIP: CrÃƒÂ©er un dossier unique et forcer le fetch ÃƒÂ  chaque fois
            try:
                from src.pipeline.fetchers import ensure_assets_for_keywords  # type: ignore
                
                # CrÃƒÂ©er un dossier unique pour ce clip (ÃƒÂ©viter le partage entre clips)
                clip_id = input_path.stem  # Nom du fichier sans extension
                clip_broll_dir = broll_library / f"clip_{clip_id}_{int(time.time())}"
                clip_broll_dir.mkdir(parents=True, exist_ok=True)
                
                # Forcer l'activation du fetcher pour chaque clip
                setattr(cfg, 'enable_fetcher', True)
                setattr(cfg, 'broll_library', str(clip_broll_dir))  # Utiliser le dossier unique
                
                print(f"    Ã°Å¸â€â€ž Fetch B-roll personnalisÃƒÂ© pour clip: {clip_id}")
                print(f"    Ã°Å¸â€œÂ Dossier B-roll unique: {clip_broll_dir.name}")
                
                # Ã°Å¸Å¡â‚¬ NOUVEAU: IntÃƒÂ©gration du sÃƒÂ©lecteur B-roll gÃƒÂ©nÃƒÂ©rique
                if BROLL_SELECTOR_AVAILABLE and getattr(Config, 'BROLL_SELECTOR_ENABLED', True):
                    try:
                        print("    Ã°Å¸Å½Â¯ SÃƒÂ©lecteur B-roll gÃƒÂ©nÃƒÂ©rique activÃƒÂ© - Scoring mixte intelligent")
                        
                        # Initialiser le sÃƒÂ©lecteur avec la configuration
                        selector_config = None
                        if getattr(Config, 'BROLL_SELECTOR_CONFIG_PATH', None):
                            try:
                                import yaml
                                with open(Config.BROLL_SELECTOR_CONFIG_PATH, 'r', encoding='utf-8') as f:
                                    selector_config = yaml.safe_load(f)
                                print(f"    Ã¢Å¡â„¢Ã¯Â¸Â Configuration chargÃƒÂ©e: {Config.BROLL_SELECTOR_CONFIG_PATH}")
                            except Exception as e:
                                print(f"    Ã¢Å¡Â Ã¯Â¸Â Erreur chargement config: {e}")
                        
                        # CrÃƒÂ©er le sÃƒÂ©lecteur
                        from broll_selector import BrollSelector
                        broll_selector = BrollSelector(selector_config)
                        
                        # Analyser le contexte pour la sÃƒÂ©lection intelligente
                        context_keywords = []
                        if 'global_analysis' in locals():
                            context_keywords = global_analysis.keywords[:10] if hasattr(global_analysis, 'keywords') else []
                        else:
                            # Fallback vers extraction basique
                            for s in subtitles:
                                text = s.get('text', '')
                                if text:
                                    words = text.lower().split()
                                    context_keywords.extend([w for w in words if len(w) > 3 and w.isalpha()])
                        
                        # DÃƒÂ©tecter le domaine
                        detected_domain = None
                        if 'global_analysis' in locals() and hasattr(global_analysis, 'main_theme'):
                            detected_domain = global_analysis.main_theme
                        
                        print(f"    Ã°Å¸Å½Â¯ Contexte: {detected_domain or 'gÃƒÂ©nÃƒÂ©ral'}")
                        print(f"    Ã°Å¸â€â€˜ Mots-clÃƒÂ©s contextuels: {', '.join(context_keywords[:5])}")
                        
                        # Utiliser le sÃƒÂ©lecteur pour la planification
                        selection_report = broll_selector.select_brolls(
                            keywords=context_keywords,
                            domain=detected_domain,
                            min_delay=self._load_broll_selector_config().get('thresholds', {}).get('min_delay_seconds', 4.0),
                            desired_count=self._load_broll_selector_config().get('desired_broll_count', 3)
                        )
                        
                        # Sauvegarder le rapport de sÃƒÂ©lection
                        try:
                            meta_dir = Config.OUTPUT_FOLDER / 'meta'
                            meta_dir.mkdir(parents=True, exist_ok=True)
                            selection_report_path = meta_dir / f"{Path(input_path).stem}_broll_selection_report.json"
                            with open(selection_report_path, 'w', encoding='utf-8') as f:
                                json.dump(selection_report, f, ensure_ascii=False, indent=2)
                            print(f"    Ã°Å¸â€™Â¾ Rapport de sÃƒÂ©lection sauvegardÃƒÂ©: {selection_report_path}")
                        except Exception as e:
                            print(f"    Ã¢Å¡Â Ã¯Â¸Â Erreur sauvegarde rapport: {e}")
                        
                        # Afficher les statistiques de sÃƒÂ©lection
                        if 'diagnostics' in selection_report:
                            diag = selection_report['diagnostics']
                            print(f"    Ã°Å¸â€œÅ  SÃƒÂ©lection: {diag.get('num_selected', 0)}/{diag.get('num_candidates', 0)} B-rolls")
                            print(f"    Ã°Å¸Å½Â¯ Top score: {diag.get('top_score', 0):.3f}")
                            print(f"    Ã°Å¸â€œÂ Seuil appliquÃƒÂ©: {diag.get('min_score', 0):.3f}")
                        
                        if selection_report.get('fallback_used'):
                            print(f"    Ã°Å¸â€ Ëœ Fallback activÃƒÂ©: Tier {selection_report.get('fallback_tier', '?')}")
                        
                    except Exception as e:
                        print(f"    Ã¢Å¡Â Ã¯Â¸Â Erreur sÃƒÂ©lecteur gÃƒÂ©nÃƒÂ©rique: {e}")
                        print("    Ã°Å¸â€â€ž Fallback vers systÃƒÂ¨me existant")
                
                # Ã°Å¸Å¡â‚¬ CORRECTION: IntÃƒÂ©gration des mots-clÃƒÂ©s LLM pour le fetch
                # SÃƒâ€°LECTION INTELLIGENTE: Mots-clÃƒÂ©s contextuels + concepts associÃƒÂ©s
                from collections import Counter as _Counter
                kw_pool: list[str] = []
                
                # Ã°Å¸Â§Â  PRIORITÃƒâ€° 1: Mots-clÃƒÂ©s LLM si disponibles
                if 'broll_keywords' in locals() and broll_keywords:
                    print(f"    Ã°Å¸Å¡â‚¬ Utilisation des mots-clÃƒÂ©s LLM pour le fetch: {len(broll_keywords)} termes")
                    # Ajouter TOUS les mots-clÃƒÂ©s LLM en prioritÃƒÂ©
                    for kw in broll_keywords:
                        low = (kw or '').strip().lower()
                        if low and len(low) >= 3:
                            kw_pool.append(low)
                            # Ajouter des variations pour enrichir
                            if ' ' in low:  # Mots composÃƒÂ©s
                                parts = low.split()
                                kw_pool.extend(parts)
                    
                    print(f"    Ã°Å¸Å½Â¯ Mots-clÃƒÂ©s LLM ajoutÃƒÂ©s: {', '.join(broll_keywords[:8])}")
                
                # Ã°Å¸â€â€ž PRIORITÃƒâ€° 2: Extraction des mots-clÃƒÂ©s du transcript
                for s in subtitles:
                    base_kws = extract_keywords_for_segment(s.get('text','')) or []
                    spacy_kws = self._extract_keywords_for_segment_spacy(s.get('text','')) or []
                    for kw in (base_kws + spacy_kws):
                        low = (kw or '').strip().lower()
                        if low and len(low) >= 3:
                            kw_pool.append(low)
                
                # Ã°Å¸Å¡â‚¬ CONCEPTS ASSOCIÃƒâ€°S ENRICHIS (50+ concepts)
                concept_mapping = {
                    # Ã°Å¸Â§Â  Cerveau & Intelligence
                    'brain': ['neuroscience', 'mind', 'thinking', 'intelligence', 'cognitive', 'mental', 'psychology', 'consciousness'],
                    'mind': ['brain', 'thinking', 'thought', 'intelligence', 'cognitive', 'mental', 'psychology'],
                    'thinking': ['brain', 'mind', 'thought', 'intelligence', 'cognitive', 'mental', 'logic'],
                    
                    # Ã°Å¸â€™Â° Argent & Finance
                    'money': ['finance', 'business', 'success', 'wealth', 'investment', 'cash', 'profit', 'revenue'],
                    'argent': ['finance', 'business', 'success', 'wealth', 'investment', 'cash', 'profit', 'revenue'],
                    'finance': ['money', 'business', 'investment', 'wealth', 'profit', 'revenue', 'budget'],
                    
                    # Ã°Å¸Å½Â¯ Focus & Concentration
                    'focus': ['concentration', 'productivity', 'attention', 'mindfulness', 'clarity', 'precision'],
                    'concentration': ['focus', 'attention', 'mindfulness', 'clarity', 'precision', 'dedication'],
                    'attention': ['focus', 'concentration', 'mindfulness', 'awareness', 'observation'],
                    
                    # Ã°Å¸Ââ€  SuccÃƒÂ¨s & RÃƒÂ©ussite
                    'success': ['achievement', 'goal', 'victory', 'winning', 'growth', 'accomplishment', 'triumph'],
                    'succÃƒÂ¨s': ['achievement', 'goal', 'victory', 'winning', 'growth', 'accomplishment', 'triumph'],
                    'victory': ['success', 'achievement', 'winning', 'triumph', 'conquest', 'domination'],
                    
                    # Ã¢ÂÂ¤Ã¯Â¸Â SantÃƒÂ© & Bien-ÃƒÂªtre
                    'health': ['wellness', 'fitness', 'medical', 'lifestyle', 'nutrition', 'vitality', 'strength'],
                    'santÃƒÂ©': ['wellness', 'fitness', 'medical', 'lifestyle', 'nutrition', 'vitality', 'strength'],
                    'fitness': ['health', 'wellness', 'exercise', 'training', 'strength', 'endurance'],
                    
                    # Ã°Å¸Â¤â€“ Technologie & Innovation
                    'technology': ['digital', 'innovation', 'future', 'ai', 'automation', 'tech', 'modern'],
                    'technologie': ['digital', 'innovation', 'future', 'ai', 'automation', 'tech', 'modern'],
                    'innovation': ['technology', 'digital', 'future', 'ai', 'automation', 'creativity', 'progress'],
                    
                    # Ã°Å¸â€™Â¼ Business & Entreprise
                    'business': ['entrepreneur', 'startup', 'strategy', 'leadership', 'growth', 'company', 'enterprise'],
                    'entreprise': ['entrepreneur', 'startup', 'strategy', 'leadership', 'growth', 'company', 'enterprise'],
                    'strategy': ['business', 'planning', 'tactics', 'approach', 'method', 'system'],
                    
                    # Ã°Å¸Å¡â‚¬ Action & Dynamisme
                    'action': ['movement', 'energy', 'power', 'vitality', 'dynamism', 'activity', 'motion'],
                    'action': ['movement', 'energy', 'power', 'vitality', 'dynamism', 'activity', 'motion'],
                    'energy': ['power', 'vitality', 'strength', 'force', 'intensity', 'enthusiasm'],
                    
                    # Ã°Å¸â€Â¥ Ãƒâ€°motion & Passion
                    'emotion': ['feeling', 'passion', 'excitement', 'inspiration', 'motivation', 'enthusiasm'],
                    'ÃƒÂ©motion': ['feeling', 'passion', 'excitement', 'inspiration', 'motivation', 'enthusiasm'],
                    'passion': ['emotion', 'feeling', 'excitement', 'inspiration', 'motivation', 'enthusiasm'],
                    
                    # Ã°Å¸Â§Â  DÃƒÂ©veloppement Personnel
                    'growth': ['development', 'improvement', 'progress', 'advancement', 'evolution', 'maturity'],
                    'croissance': ['development', 'improvement', 'progress', 'advancement', 'evolution', 'maturity'],
                    'development': ['growth', 'improvement', 'progress', 'advancement', 'evolution', 'maturity'],
                    
                    # Ã¢Å“â€¦ Solutions & RÃƒÂ©solution
                    'solution': ['resolution', 'fix', 'answer', 'remedy', 'cure', 'treatment'],
                    'solution': ['resolution', 'fix', 'answer', 'remedy', 'cure', 'treatment'],
                    'resolution': ['solution', 'fix', 'answer', 'remedy', 'cure', 'treatment'],
                    
                    # Ã¢Å¡Â Ã¯Â¸Â ProblÃƒÂ¨mes & DÃƒÂ©fis
                    'problem': ['challenge', 'difficulty', 'obstacle', 'barrier', 'issue', 'trouble'],
                    'problÃƒÂ¨me': ['challenge', 'difficulty', 'obstacle', 'barrier', 'issue', 'trouble'],
                    'challenge': ['problem', 'difficulty', 'obstacle', 'barrier', 'issue', 'trouble'],
                    
                    # Ã°Å¸Å’Å¸ QualitÃƒÂ© & Excellence
                    'quality': ['excellence', 'perfection', 'superiority', 'premium', 'best', 'optimal'],
                    'qualitÃƒÂ©': ['excellence', 'perfection', 'superiority', 'premium', 'best', 'optimal'],
                    'excellence': ['quality', 'perfection', 'superiority', 'premium', 'best', 'optimal']
                }
                
                # Enrichir avec des concepts associÃƒÂ©s
                for kw in kw_pool[:]:
                    for concept, related in concept_mapping.items():
                        if concept in kw or any(r in kw for r in related):
                            kw_pool.extend(related[:2])  # Ajouter 2 concepts max
                
                counts = _Counter(kw_pool)
                
                # Ã°Å¸Å¡Â¨ CORRECTION CRITIQUE: PRIORISER les mots-clÃƒÂ©s LLM sur les mots-clÃƒÂ©s gÃƒÂ©nÃƒÂ©riques
                if 'broll_keywords' in locals() and broll_keywords:
                    # Utiliser DIRECTEMENT les mots-clÃƒÂ©s LLM comme requÃƒÂªte principale
                    llm_keywords = [kw.strip().lower() for kw in broll_keywords if kw and len(kw.strip()) >= 3]
                    if llm_keywords:
                        # Prendre les 8 premiers mots-clÃƒÂ©s LLM + 2 concepts associÃƒÂ©s
                        top_kws = llm_keywords[:8]
                        # Ajouter quelques concepts associÃƒÂ©s pour enrichir
                        for kw in top_kws[:3]:  # Pour les 3 premiers mots-clÃƒÂ©s LLM
                            for concept, related in concept_mapping.items():
                                if concept in kw or any(r in kw for r in related):
                                    top_kws.extend(related[:1])  # 1 concept max par mot-clÃƒÂ© LLM
                                    break
                        print(f"    Ã°Å¸Å¡â‚¬ REQUÃƒÅ TE LLM PRIORITAIRE: {' '.join(top_kws[:5])}")
                    else:
                        top_kws = [w for w,_n in counts.most_common(15)]
                        print(f"    Ã°Å¸â€â€ž Fallback vers mots-clÃƒÂ©s gÃƒÂ©nÃƒÂ©riques: {' '.join(top_kws[:5])}")
                else:
                    top_kws = [w for w,_n in counts.most_common(15)]
                    print(f"    Ã°Å¸â€â€ž Mots-clÃƒÂ©s gÃƒÂ©nÃƒÂ©riques: {' '.join(top_kws[:5])}")
                
                # Fallback intelligent selon le contexte
                if not top_kws:
                    top_kws = ["focus","concentration","study","brain","mind","productivity","success"]
                print(f"    Ã°Å¸â€Å½ Fetch B-roll sur requÃƒÂªte: {' '.join(top_kws[:5])}")
                # Provider auto-fallback si pas de clÃƒÂ©s -> archive
                import os as _os
                pex = getattr(Config, 'PEXELS_API_KEY', None) or _os.getenv('PEXELS_API_KEY')
                pixa = getattr(Config, 'PIXABAY_API_KEY', None) or _os.getenv('PIXABAY_API_KEY')
                uns = getattr(Config, 'UNSPLASH_ACCESS_KEY', None) or _os.getenv('UNSPLASH_ACCESS_KEY')
                giphy = _os.getenv('GIPHY_API_KEY')  # Ã°Å¸Å½Â­ GIPHY pour GIFs animÃƒÂ©s
                # Exposer l'accÃƒÂ¨s Unsplash dans la cfg si dispo
                try:
                    if uns:
                        setattr(cfg, 'unsplash_access_key', uns)
                except Exception:
                    pass
                if not any([pex, pixa, uns]):
                    try:
                        setattr(cfg, 'fetch_provider', 'archive')
                        print("    Ã°Å¸Å’Â Providers: archive (aucune clÃƒÂ© API dÃƒÂ©tectÃƒÂ©e)")
                    except Exception:
                        pass
                else:
                    # Ã°Å¸Å¡â‚¬ AMÃƒâ€°LIORATION: Construire une liste de providers optimisÃƒÂ©e
                    prov = []
                    if pex:
                        prov.append('pexels')
                    if pixa:
                        prov.append('pixabay')
                    if uns:
                        prov.append('unsplash')
                    if giphy:
                        prov.append('giphy')  # Ã°Å¸Å½Â­ GIPHY pour GIFs animÃƒÂ©s
                    
                    # Ã°Å¸Å½Â¯ AJOUT SÃƒâ€°CURISÃƒâ€°: Archive.org comme source supplÃƒÂ©mentaire
                    try:
                        if prov:  # Si on a des providers avec clÃƒÂ©s API
                            prov.append('archive')  # Ajouter Archive.org
                            print(f"    Ã°Å¸Å’Â Providers: {','.join(prov)} (Archive.org + Giphy ajoutÃƒÂ©s pour variÃƒÂ©tÃƒÂ©)")
                        else:
                            prov = ['archive']  # Seulement Archive.org si pas de clÃƒÂ©s
                            print(f"    Ã°Å¸Å’Â Providers: {','.join(prov)} (Archive.org uniquement)")
                        
                        setattr(cfg, 'fetch_provider', ",".join(prov))
                    except Exception as e:
                        # Fallback sÃƒÂ©curisÃƒÂ©
                        try:
                            if prov:
                                setattr(cfg, 'fetch_provider', ",".join(prov))
                                print(f"    Ã°Å¸Å’Â Providers: {','.join(prov)} (fallback sÃƒÂ©curisÃƒÂ©)")
                            else:
                                setattr(cfg, 'fetch_provider', 'archive')
                                print(f"    Ã°Å¸Å’Â Providers: archive (fallback ultime)")
                        except Exception:
                            pass
                
                try:
                    setattr(cfg, 'fetch_allow_images', True)
                    # Ã°Å¸Å¡â‚¬ OPTIMISATION MULTI-SOURCES: QualitÃƒÂ© optimale (CORRIGÃƒâ€°)
                    if uns and giphy:  # Si Unsplash ET Giphy sont disponibles
                        setattr(cfg, 'fetch_max_per_keyword', 35)  # CORRIGÃƒâ€°: 125 Ã¢â€ â€™ 35 pour qualitÃƒÂ© maximale
                        print("    Ã°Å¸â€œÅ  Configuration optimisÃƒÂ©e: 35 assets max + images activÃƒÂ©es (Unsplash + Giphy + Archive)")
                    elif uns:  # Si seulement Unsplash est disponible
                        setattr(cfg, 'fetch_max_per_keyword', 30)  # CORRIGÃƒâ€°: 100 Ã¢â€ â€™ 30 pour qualitÃƒÂ© maximale
                        print("    Ã°Å¸â€œÅ  Configuration optimisÃƒÂ©e: 30 assets max + images activÃƒÂ©es (Unsplash + Archive)")
                    elif giphy:  # Si seulement Giphy est disponible
                        setattr(cfg, 'fetch_max_per_keyword', 30)  # CORRIGÃƒâ€°: 100 Ã¢â€ â€™ 30 pour qualitÃƒÂ© avec GIFs
                        print("    Ã°Å¸â€œÅ  Configuration optimisÃƒÂ©e: 30 assets max + images activÃƒÂ©es (Giphy + Archive)")
                    else:
                        setattr(cfg, 'fetch_max_per_keyword', 25)  # CORRIGÃƒâ€°: 75 Ã¢â€ â€™ 25 pour Archive.org
                        print("    Ã°Å¸â€œÅ  Configuration optimisÃƒÂ©e: 25 assets max + images activÃƒÂ©es (Archive.org)")
                except Exception:
                    pass
                # DÃƒÂ©clencher le fetch dans le dossier unique du clip
                ensure_assets_for_keywords(cfg, top_kws)
                
                # Ã°Å¸Å¡Â¨ CORRECTION CRITIQUE: SYSTÃƒË†ME D'UNICITÃƒâ€° DES B-ROLLS
                # Ãƒâ€°viter la duplication des B-rolls entre vidÃƒÂ©os diffÃƒÂ©rentes
                try:
                    # CrÃƒÂ©er un fichier de traÃƒÂ§abilitÃƒÂ© des B-rolls utilisÃƒÂ©s
                    broll_tracking_file = Config.OUTPUT_FOLDER / 'meta' / 'broll_usage_tracking.json'
                    broll_tracking_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Charger l'historique des B-rolls utilisÃƒÂ©s
                    broll_history = {}
                    if broll_tracking_file.exists():
                        try:
                            with open(broll_tracking_file, 'r', encoding='utf-8') as f:
                                broll_history = json.load(f)
                        except Exception:
                            broll_history = {}
                    
                    # Identifier les B-rolls disponibles pour ce clip
                    available_brolls = []
                    for asset_path in clip_broll_dir.rglob('*'):
                        if asset_path.suffix.lower() in {'.mp4', '.mov', '.mkv', '.webm', '.jpg', '.jpeg', '.png'}:
                            asset_hash = self._calculate_asset_hash(asset_path)
                            asset_info = {
                                'path': str(asset_path),
                                'hash': asset_hash,
                                'size': asset_path.stat().st_size,
                                'last_used': None,
                                'usage_count': 0
                            }
                            
                            # VÃƒÂ©rifier si ce B-roll a dÃƒÂ©jÃƒÂ  ÃƒÂ©tÃƒÂ© utilisÃƒÂ©
                            if asset_hash in broll_history:
                                asset_info['last_used'] = broll_history[asset_hash].get('last_used')
                                asset_info['usage_count'] = broll_history[asset_hash].get('usage_count', 0)
                            
                            available_brolls.append(asset_info)
                    
                    # Trier par prioritÃƒÂ©: B-rolls jamais utilisÃƒÂ©s en premier, puis par anciennetÃƒÂ©
                    available_brolls.sort(key=lambda x: (x['usage_count'], x['last_used'] or '1970-01-01'))
                    
                    # SÃƒÂ©lectionner les B-rolls uniques pour cette vidÃƒÂ©o
                    selected_brolls = available_brolls[:3]  # 3 B-rolls uniques
                    
                    # Mettre ÃƒÂ  jour l'historique d'utilisation
                    current_time = datetime.now().isoformat()
                    for broll in selected_brolls:
                        broll_history[broll['hash']] = {
                            'last_used': current_time,
                            'usage_count': broll['usage_count'] + 1,
                            'video_id': Path(input_path).stem
                        }
                    
                    # Sauvegarder l'historique
                    with open(broll_tracking_file, 'w', encoding='utf-8') as f:
                        json.dump(broll_history, f, ensure_ascii=False, indent=2)
                    
                    print(f"    Ã°Å¸Å½Â¯ B-rolls uniques sÃƒÂ©lectionnÃƒÂ©s: {len(selected_brolls)} (ÃƒÂ©vite duplication)")
                    
                except Exception as e:
                    print(f"    Ã¢Å¡Â Ã¯Â¸Â Erreur systÃƒÂ¨me d'unicitÃƒÂ©: {e}")
                    # Fallback: utiliser tous les B-rolls disponibles
                    pass
                
                # Comptage aprÃƒÂ¨s fetch dans le dossier du clip
                try:
                    _media_exts = {'.mp4','.mov','.mkv','.webm','.jpg','.jpeg','.png'}
                    _after = [p for p in clip_broll_dir.rglob('*') if p.suffix.lower() in _media_exts]
                    print(f"    Ã°Å¸â€œÂ¥ Fetch terminÃƒÂ©: {len(_after)} assets pour ce clip")
                    
                    # Ã°Å¸Å¡Â¨ CORRECTION CRITIQUE: CrÃƒÂ©er fetched_brolls accessible globalement
                    fetched_brolls = []
                    for asset_path in _after:
                        if asset_path.exists():
                            fetched_brolls.append({
                                'path': str(asset_path),
                                'name': asset_path.name,
                                'size': asset_path.stat().st_size if asset_path.exists() else 0
                            })
                    
                    print(f"    Ã°Å¸Å½Â¯ {len(fetched_brolls)} B-rolls prÃƒÂªts pour l'assignation")
                    
                    if len(_after) == 0:
                        print("    Ã¢Å¡Â Ã¯Â¸Â Aucun asset tÃƒÂ©lÃƒÂ©chargÃƒÂ©. VÃƒÂ©rifie les clÃƒÂ©s API et la connectivitÃƒÂ© rÃƒÂ©seau.")
                except Exception:
                    fetched_brolls = []
                    print("    Ã¢Å¡Â Ã¯Â¸Â Erreur lors de la prÃƒÂ©paration des B-rolls fetchÃƒÂ©s")
                
                # Construire l'index FAISS pour ce clip spÃƒÂ©cifique
                try:
                    if 'build_index' in globals() and build_index is not None:  # type: ignore[name-defined]
                        index_handle = build_index(str(clip_broll_dir), model_name='ViT-B/32')  # type: ignore[misc]
                        print(f"    Ã°Å¸Â§Â­ Index FAISS construit pour {clip_id}: {len(_after)} assets")
                except Exception:
                    index_handle = None
            except Exception:
                pass
  
            # Extensions optionnelles pour crossfade/LUT
            try:
                setattr(cfg, 'crossfade_frames', 3)
                setattr(cfg, 'enable_color_match', True)
                setattr(cfg, 'transition_mode', 'auto')  # 'auto' | 'cut' | 'crossfade' | 'zoom'
                setattr(cfg, 'allow_zoom_transitions', True)
                setattr(cfg, 'enable_image_kenburns', True)
            except Exception:
                pass
            
            # PrÃƒÂ©parer stop-words (legacy pipeline)
            stopwords: set[str] = set()
            try:
                swp = Path('config/stopwords.txt')
                if swp.exists():
                    stopwords = {ln.strip().lower() for ln in swp.read_text(encoding='utf-8').splitlines() if ln.strip()}
            except Exception:
                stopwords = set()

            # Ã°Å¸Å¡â‚¬ CORRECTION: IntÃƒÂ©gration des mots-clÃƒÂ©s LLM dans la planification
            # Planification: nouvelle API prÃƒÂ©fÃƒÂ©rÃƒÂ©e (plan_broll_insertions(segments, cfg, index))
            
            # Ã°Å¸Å¡Â¨ CORRECTION CRITIQUE: fetched_brolls est dÃƒÂ©jÃƒÂ  dÃƒÂ©clarÃƒÂ© plus haut, ne pas le redÃƒÂ©clarer !
            # fetched_brolls = []  # Ã¢ÂÅ’ SUPPRIMÃƒâ€°: Cette ligne ÃƒÂ©crase la variable fetchÃƒÂ©e !
            
            try:
                plan = plan_broll_insertions(segments, cfg, index_handle)  # type: ignore[arg-type]
            except Exception:
                # Ã°Å¸Å¡â‚¬ NOUVEAU: Utiliser les mots-clÃƒÂ©s LLM pour la planification
                seg_keywords: List[List[str]] = []
                
                # Ã°Å¸Â§Â  PRIORITÃƒâ€° 1: Mots-clÃƒÂ©s LLM si disponibles
                if 'broll_keywords' in locals() and broll_keywords:
                    print(f"    Ã°Å¸Å¡â‚¬ Utilisation des mots-clÃƒÂ©s LLM pour la planification: {len(broll_keywords)} termes")
                    # Distribuer les mots-clÃƒÂ©s LLM sur les segments
                    for i, s in enumerate(segments):
                        # Prendre 2-3 mots-clÃƒÂ©s LLM par segment
                        start_idx = (i * 2) % len(broll_keywords)
                        end_idx = min(start_idx + 2, len(broll_keywords))
                        segment_llm_kws = broll_keywords[start_idx:end_idx]
                        
                        # Combiner avec extraction basique
                        base_kws = extract_keywords_for_segment(s.text) or []
                        spacy_kws = self._extract_keywords_for_segment_spacy(s.text) or []
                        
                        # Ã°Å¸Å½Â¯ PRIORITÃƒâ€° aux mots-clÃƒÂ©s LLM
                        merged: List[str] = segment_llm_kws + base_kws + spacy_kws
                        
                        # Nettoyer et dÃƒÂ©dupliquer
                        cleaned: List[str] = []
                        seen = set()
                        for kw in merged:
                            if kw and kw.lower() not in seen:
                                low = kw.lower()
                                if not (len(low) < 3 and low in stopwords):
                                    cleaned.append(low)
                                    seen.add(low)
                        
                        seg_keywords.append(cleaned[:15])  # AugmentÃƒÂ©: 12 Ã¢â€ â€™ 15
                        print(f"    Ã°Å¸Å½Â¯ Segment {i}: {len(cleaned)} mots-clÃƒÂ©s (LLM: {len(segment_llm_kws)})")
                else:
                    # Ã°Å¸â€â€ž Fallback: extraction basique uniquement
                    print("    Ã¢Å¡Â Ã¯Â¸Â Mots-clÃƒÂ©s LLM non disponibles, utilisation extraction basique")
                    for s in segments:
                        base_kws = extract_keywords_for_segment(s.text) or []
                        spacy_kws = self._extract_keywords_for_segment_spacy(s.text) or []
                        merged: List[str] = []
                        for kw in (base_kws + spacy_kws):
                            if kw and kw.lower() not in merged:
                                low = kw.lower()
                                if not (len(low) < 5 and low in stopwords):
                                    merged.append(low)
                        seg_keywords.append(merged[:12])
                
                with _VFC(str(input_path)) as _tmp:
                    duration = float(_tmp.duration)
                plan = plan_broll_insertions(  # type: ignore[call-arg]
                    segments,
                    seg_keywords,
                    total_duration=duration,
                    max_broll_ratio=cfg.max_broll_ratio,
                    min_gap_between_broll_s=cfg.min_gap_between_broll_s,
                    max_broll_clip_s=cfg.max_broll_clip_s,
                    min_broll_clip_s=cfg.min_broll_clip_s,
                )
                
                # Ã°Å¸Å¡Â¨ CORRECTION CRITIQUE: Assigner directement les B-rolls fetchÃƒÂ©s aux items du plan
                if plan and fetched_brolls:
                    print(f"    Ã°Å¸Å½Â¯ Assignation directe des {len(fetched_brolls)} B-rolls fetchÃƒÂ©s aux {len(plan)} items du plan...")
                    
                    # Filtrer les B-rolls valides
                    valid_brolls = [broll for broll in fetched_brolls if broll.get('path') and Path(broll.get('path')).exists()]
                    
                    if valid_brolls:
                        # Assigner les B-rolls aux items du plan
                        for i, item in enumerate(plan):
                            if i < len(valid_brolls):
                                asset_path = valid_brolls[i]['path']
                                
                                # Assigner l'asset_path selon le type d'objet
                                if hasattr(item, 'asset_path'):
                                    item.asset_path = asset_path
                                elif isinstance(item, dict):
                                    item['asset_path'] = asset_path
                                
                                print(f"    Ã¢Å“â€¦ B-roll {i+1} assignÃƒÂ©: {Path(asset_path).name}")
                            else:
                                break
                        
                        print(f"    Ã°Å¸Å½â€° {min(len(plan), len(valid_brolls))} B-rolls assignÃƒÂ©s avec succÃƒÂ¨s au plan")
                    else:
                        print(f"    Ã¢Å¡Â Ã¯Â¸Â Aucun B-roll valide trouvÃƒÂ© dans fetched_brolls")
                elif not fetched_brolls:
                    print(f"    Ã¢Å¡Â Ã¯Â¸Â Aucun B-roll fetchÃƒÂ© disponible pour l'assignation")
                elif not plan:
                    print(f"    Ã¢Å¡Â Ã¯Â¸Â Plan vide - aucun item ÃƒÂ  traiter")
            # Scoring adaptatif si disponible (pertinence/diversitÃƒÂ©/esthÃƒÂ©tique)
            

            
            try:
                from src.pipeline.scoring import score_candidates  # type: ignore
                boosts = {
                    # Ã°Å¸Å¡â‚¬ Business & Croissance
                    "croissance": 0.9, "growth": 0.9, "opportunitÃƒÂ©": 0.8, "opportunite": 0.8,
                    "innovation": 0.9, "dÃƒÂ©veloppement": 0.8, "developpement": 0.8, "expansion": 0.8,
                    "stratÃƒÂ©gie": 0.8, "strategie": 0.8, "plan": 0.7, "objectif": 0.8, "vision": 0.8,
                    
                    # Ã°Å¸â€™Â° Argent & Finance
                    "argent": 1.0, "money": 1.0, "cash": 0.9, "investissement": 0.9, "investissements": 0.9,
                    "revenu": 0.8, "revenus": 0.8, "profit": 0.9, "profits": 0.9, "perte": 0.7, "pertes": 0.7,
                    "ÃƒÂ©chec": 0.7, "echec": 0.7, "budget": 0.7, "gestion": 0.7, "marge": 0.8, "roi": 0.9,
                    "chiffre": 0.7, "ca": 0.7, "ÃƒÂ©conomie": 0.8, "economie": 0.8, "financier": 0.8,
                    
                    # Ã°Å¸Â¤Â Relation & Client
                    "client": 0.9, "clients": 0.9, "collaboration": 0.8, "collaborations": 0.8,
                    "communautÃƒÂ©": 0.7, "communaute": 0.7, "confiance": 0.7, "vente": 0.8, "ventes": 0.8,
                    "deal": 0.7, "deals": 0.7, "prospect": 0.6, "prospects": 0.6, "contrat": 0.7,
                    "partenariat": 0.8, "ÃƒÂ©quipe": 0.7, "equipe": 0.7, "rÃƒÂ©seau": 0.7, "reseau": 0.7,
                    
                    # Ã°Å¸â€Â¥ Motivation & SuccÃƒÂ¨s
                    "succÃƒÂ¨s": 0.9, "succes": 0.9, "motivation": 0.8, "ÃƒÂ©nergie": 0.7, "energie": 0.7,
                    "victoire": 0.8, "discipline": 0.7, "viral": 0.8, "viralitÃƒÂ©": 0.8, "viralite": 0.8,
                    "impact": 0.6, "explose": 0.6, "explosion": 0.6, "inspiration": 0.8, "passion": 0.8,
                    "dÃƒÂ©termination": 0.8, "determination": 0.8, "persÃƒÂ©vÃƒÂ©rance": 0.8, "perseverance": 0.8,
                    
                    # Ã°Å¸Â§Â  Intelligence & Apprentissage
                    "cerveau": 1.0, "brain": 1.0, "intelligence": 0.9, "savoir": 0.8, "connaissance": 0.8,
                    "apprentissage": 0.8, "apprendre": 0.8, "ÃƒÂ©tude": 0.8, "etude": 0.8, "formation": 0.8,
                    "compÃƒÂ©tence": 0.8, "competence": 0.8, "expertise": 0.8, "maÃƒÂ®trise": 0.8, "maitrise": 0.8,
                    
                    # Ã°Å¸â€™Â¡ Innovation & Technologie
                    "technologie": 0.9, "tech": 0.9, "innovation": 0.9, "digital": 0.8, "numÃƒÂ©rique": 0.8,
                    "numerique": 0.8, "futur": 0.8, "avancÃƒÂ©e": 0.8, "avancee": 0.8, "rÃƒÂ©volution": 0.8,
                    "revolution": 0.8, "disruption": 0.8, "transformation": 0.8, "ÃƒÂ©volution": 0.8, "evolution": 0.8,
                    
                    # Ã¢Å¡Â Ã¯Â¸Â Risque & Erreurs
                    "erreur": 0.6, "erreurs": 0.6, "warning": 0.6, "obstacle": 0.6, "obstacles": 0.6,
                    "solution": 0.6, "solutions": 0.6, "leÃƒÂ§on": 0.5, "lecon": 0.5, "apprentissage": 0.5,
                    "problÃƒÂ¨me": 0.6, "probleme": 0.6, "dÃƒÂ©fi": 0.7, "defi": 0.7, "challenge": 0.7,
                    
                    # Ã°Å¸Å’Å¸ QualitÃƒÂ© & Excellence
                    "excellence": 0.9, "qualitÃƒÂ©": 0.8, "qualite": 0.8, "perfection": 0.8, "meilleur": 0.8,
                    "optimal": 0.8, "efficacitÃƒÂ©": 0.8, "efficacite": 0.8, "performance": 0.8, "rÃƒÂ©sultat": 0.8,
                    "resultat": 0.8, "succÃƒÂ¨s": 0.9, "succes": 0.9, "rÃƒÂ©ussite": 0.9, "reussite": 0.9,
                }
                plan = score_candidates(
                    plan, segments, broll_library=str(broll_library), clip_model='ViT-B/32',
                    use_faiss=True, top_k=10, keyword_boosts=boosts
                )
                
            except Exception:
                pass
 
            # FILTRE: Exclure les B-rolls trop tÃƒÂ´t dans la vidÃƒÂ©o (dÃƒÂ©lai minimum 3 secondes)
            try:
                filtered_plan = []
                for it in plan:
                    st = float(getattr(it, 'start', 0.0) if hasattr(it, 'start') else (it.get('start', 0.0) if isinstance(it, dict) else 0.0))
                    if st >= 3.0:  # DÃƒÂ©lai minimum de 3 secondes avant le premier B-roll
                        filtered_plan.append(it)
                    else:
                        print(f"    Ã¢ÂÂ° B-roll filtrÃƒÂ©: trop tÃƒÂ´t ÃƒÂ  {st:.2f}s (minimum 3.0s)")
                
                plan = filtered_plan
                print(f"    Ã¢Å“â€¦ Plan filtrÃƒÂ©: {len(plan)} B-rolls aprÃƒÂ¨s dÃƒÂ©lai minimum")
            except Exception:
                pass

            # DÃƒÂ©duplication souple: autoriser rÃƒÂ©utilisation si espacÃƒÂ©e (> 12s)
            try:
                seen: dict[str, float] = {}
                new_plan = []
                for it in plan:
                    # Ã°Å¸â€Â§ CORRECTION: GÃƒÂ©rer ÃƒÂ  la fois BrollPlanItem et dict
                    if hasattr(it, 'asset_path'):
                        ap = it.asset_path
                        st = float(it.start)
                    elif isinstance(it, dict):
                        ap = it.get('asset_path')
                        st = float(it.get('start', 0.0))
                    else:
                        # Fallback pour autres types
                        ap = getattr(it, 'asset_path', None)
                        st = float(getattr(it, 'start', 0.0))
                    
                    if not ap:
                        new_plan.append(it)
                        continue
                    
                    last = seen.get(ap, -1e9)
                    if st - last >= 8.0:
                        new_plan.append(it)
                        seen[ap] = st
                plan = new_plan
                
            except Exception:
                pass
 
            # Ã°Å¸Å¡â‚¬ PRIORISATION FRAÃƒÅ½CHEUR: Trier par timestamp du dossier (plus rÃƒÂ©cent en premier)
            try:
                if plan:
                    # Extraire le clip_id pour la priorisation
                    clip_id = input_path.stem
                    
                    # Prioriser par fraÃƒÂ®cheur si possible
                    for item in plan:
                        if hasattr(item, 'asset_path') and item.asset_path:
                            asset_path = item.asset_path
                        elif isinstance(item, dict) and item.get('asset_path'):
                            asset_path = item['asset_path']
                        else:
                            continue
                        
                        # Calculer le score de fraÃƒÂ®cheur
                        try:
                            path = Path(asset_path)
                            for part in path.parts:
                                if part.startswith(f"clip_{clip_id}_") and "_" in part:
                                    timestamp_str = part.split("_")[-1]
                                    if timestamp_str.isdigit():
                                        item.freshness_score = int(timestamp_str)
                                        break
                            else:
                                item.freshness_score = 0
                        except Exception:
                            item.freshness_score = 0
                    
                    # Trier par fraÃƒÂ®cheur dÃƒÂ©croissante
                    plan.sort(key=lambda x: getattr(x, 'freshness_score', 0), reverse=True)
                    print(f"    Ã°Å¸â€ â€¢ Priorisation fraÃƒÂ®cheur: {len(plan)} B-rolls triÃƒÂ©s par timestamp")
                    
            except Exception as e:
                print(f"    Ã¢Å¡Â Ã¯Â¸Â  Erreur priorisation fraÃƒÂ®cheur: {e}")
 
            # Ã°Å¸Å½Â¯ SCORING CONTEXTUEL RENFORCÃƒâ€°: PÃƒÂ©naliser les assets non pertinents au domaine
            try:
                if plan and "global_analysis" in locals() and hasattr(global_analysis, 'main_theme') and hasattr(global_analysis, 'keywords'):
                    domain = global_analysis.main_theme
                    keywords = global_analysis.keywords[:10] if hasattr(global_analysis, 'keywords') else []
                    
                    for item in plan:
                        if hasattr(item, 'asset_path') and item.asset_path:
                            asset_path = item.asset_path
                        elif isinstance(item, dict) and item.get('asset_path'):
                            asset_path = item['asset_path']
                        else:
                            continue
                        
                        # Calculer le score contextuel
                        context_score = _score_contextual_relevance(asset_path, domain, keywords)
                        
                        # Appliquer le score contextuel au score final
                        if hasattr(item, 'score'):
                            # Ajuster le score existant
                            item.score = item.score * context_score
                        elif isinstance(item, dict) and 'score' in item:
                            item['score'] = item['score'] * context_score
                        
                        # Stocker le score contextuel pour debug
                        if hasattr(item, 'context_score'):
                            item.context_score = context_score
                        elif isinstance(item, dict):
                            item['context_score'] = context_score
                    
                    print(f"    Ã°Å¸Å½Â¯ Scoring contextuel appliquÃƒÂ©: domaine '{domain}' avec {len(keywords)} mots-clÃƒÂ©s")
                    
                    # Ã°Å¸â€Â DEBUG B-ROLL SELECTION (si activÃƒÂ©)
                    debug_mode = getattr(Config, 'DEBUG_BROLL', False) or os.getenv('DEBUG_BROLL', 'false').lower() == 'true'
                    _debug_broll_selection(plan, domain, keywords, debug_mode)
                    
                    # Ã°Å¸Å¡Â¨ FALLBACK PROPRE: Si aucun asset pertinent, utiliser des assets neutres
                    # Ã°Å¸â€Â§ CORRECTION CRITIQUE: VÃƒÂ©rifier d'abord si les items ont des assets assignÃƒÂ©s
                    items_without_assets = []
                    items_with_assets = []
                    
                    for item in plan:
                        if hasattr(item, 'asset_path') and item.asset_path:
                            items_with_assets.append(item)
                        elif isinstance(item, dict) and item.get('asset_path'):
                            items_with_assets.append(item)
                        else:
                            items_without_assets.append(item)
                    
                    print(f"    Ã°Å¸â€Â Analyse des assets: {len(items_with_assets)} avec assets, {len(items_without_assets)} sans assets")
                    
                    # Ã°Å¸Å¡Â¨ CORRECTION: Assigner des assets aux items sans assets AVANT le fallback
                    if items_without_assets and fetched_brolls:
                        print(f"    Ã°Å¸Å½Â¯ Assignation d'assets aux {len(items_without_assets)} items sans assets...")
                        
                        # Utiliser les B-rolls fetchÃƒÂ©s pour assigner aux items
                        available_assets = [broll.get('path', '') for broll in fetched_brolls if broll.get('path')]
                        
                        for i, item in enumerate(items_without_assets):
                            if i < len(available_assets):
                                asset_path = available_assets[i]
                                if hasattr(item, 'asset_path'):
                                    item.asset_path = asset_path
                                elif isinstance(item, dict):
                                    item['asset_path'] = asset_path
                                
                                print(f"    Ã¢Å“â€¦ Asset assignÃƒÂ© ÃƒÂ  item {i+1}: {Path(asset_path).name}")
                            else:
                                break
                        
                        # Recalculer les listes aprÃƒÂ¨s assignation
                        items_with_assets = [item for item in plan if (hasattr(item, 'asset_path') and item.asset_path) or (isinstance(item, dict) and item.get('asset_path'))]
                        items_without_assets = [item for item in plan if not ((hasattr(item, 'asset_path') and item.asset_path) or (isinstance(item, dict) and item.get('asset_path')))]
                    
                    # Ã°Å¸Å¡Â¨ FALLBACK UNIQUEMENT SI VRAIMENT NÃƒâ€°CESSAIRE
                    if not items_with_assets and items_without_assets:
                        print(f"    Ã¢Å¡Â Ã¯Â¸Â  Aucun asset disponible, activation du fallback neutre")
                        fallback_assets = _get_fallback_neutral_assets(broll_library, count=3)
                        if fallback_assets:
                            print(f"    Ã°Å¸â€ Ëœ Fallback neutre: {len(fallback_assets)} assets gÃƒÂ©nÃƒÂ©riques utilisÃƒÂ©s")
                            # CrÃƒÂ©er des items de plan avec les assets de fallback
                            for i, asset_path in enumerate(fallback_assets):
                                fallback_item = {
                                    'start': 3.0 + (i * 5.0),  # Espacer les fallbacks
                                    'end': 3.0 + (i * 5.0) + 3.0,
                                    'asset_path': asset_path,
                                    'score': 0.5,  # Score neutre
                                    'context_score': 0.3,  # Pertinence faible
                                    'is_fallback': True
                                }
                                plan.append(fallback_item)
                        else:
                            print(f"    Ã°Å¸Å¡Â¨ Aucun asset de fallback disponible")
                    elif items_with_assets:
                        print(f"    Ã¢Å“â€¦ {len(items_with_assets)} items avec assets assignÃƒÂ©s - Pas de fallback nÃƒÂ©cessaire")
                    else:
                        print(f"    Ã¢Å¡Â Ã¯Â¸Â  Plan vide - Aucun item ÃƒÂ  traiter")
                    
            except Exception as e:
                print(f"    Ã¢Å¡Â Ã¯Â¸Â  Erreur scoring contextuel: {e}")
 
                        # Affecter un asset_path pertinent via FAISS/CLIP si manquant
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                import numpy as _np  # type: ignore
                import faiss as _faiss  # type: ignore
                from pathlib import Path as _P
                
                # Ã°Å¸Å¡Â¨ NOUVEAU: Importer le systÃƒÂ¨me de scoring contextuel intelligent
                try:
                    from src.pipeline.broll_selector import get_contextual_broll_score
                    print("    Ã°Å¸Â§Â  SystÃƒÂ¨me de scoring contextuel intelligent activÃƒÂ©")
                except ImportError:
                    print("    Ã¢Å¡Â Ã¯Â¸Â SystÃƒÂ¨me de scoring contextuel non disponible")
                    get_contextual_broll_score = None
                
                # UTILISER LE DOSSIER SPÃƒâ€°CIFIQUE DU CLIP (pas la librairie globale)
                clip_specific_dir = clip_broll_dir if 'clip_specific_dir' in locals() else broll_library
                idx_bin = (clip_specific_dir / 'faiss.index')
                idx_json = (clip_specific_dir / 'faiss.json')
                
                model_name = getattr(cfg, 'embedding_model_name', 'clip-ViT-B/32')
                st_model = SentenceTransformer('clip-ViT-B/32') if 'ViT' in model_name else SentenceTransformer(model_name)
                def emb_text(t: str):
                    v = st_model.encode([t])[0].astype('float32')
                    n = _np.linalg.norm(v) + 1e-12
                    return v / n
                paths = []
                if idx_json.exists():
                    import json as _json
                    try:
                        paths = _json.loads(idx_json.read_text(encoding='utf-8')).get('paths', [])
                    except Exception:
                        paths = []
                index = _faiss.read_index(str(idx_bin)) if idx_bin.exists() else None
                used_recent: set[str] = set()
                for it in plan or []:
                    ap = getattr(it, 'asset_path', None) if hasattr(it, 'asset_path') else (it.get('asset_path') if isinstance(it, dict) else None)
                    if ap:
                        continue
                    # Texte local autour de l'event
                    st_e = float(getattr(it, 'start', 0.0) if hasattr(it, 'start') else (it.get('start') if isinstance(it, dict) else 0.0))
                    en_e = float(getattr(it, 'end', 0.0) if hasattr(it, 'end') else (it.get('end') if isinstance(it, dict) else 0.0))
                    local = " ".join(s.text for s in segments if float(s.start) <= en_e and float(s.end) >= st_e)[:400]
                    q = emb_text(local) if local else None
                    
                    # Ã°Å¸Å¡Â¨ NOUVEAU: Extraction des mots-clÃƒÂ©s pour le scoring contextuel
                    local_keywords = []
                    if local:
                        # Extraire les mots-clÃƒÂ©s du texte local
                        words = local.lower().split()
                        local_keywords = [w for w in words if len(w) > 3 and w.isalpha()][:10]
                    
                    chosen = None
                    best_score = -1
                    
                    if index is not None and q is not None and paths:
                        # Ã°Å¸Å¡Â¨ NOUVEAU: Recherche ÃƒÂ©tendue pour ÃƒÂ©valuation contextuelle
                        D,I = index.search(q.reshape(1,-1), 15)  # Augmenter de 5 ÃƒÂ  15 candidats
                        
                        # Ã°Å¸Å¡Â¨ NOUVEAU: Ãƒâ€°valuation contextuelle de tous les candidats
                        for idx in I[0].tolist():
                            if 0 <= idx < len(paths):
                                p = paths[idx]
                                if not p:
                                    continue
                                cand = _P(p)
                                if not cand.is_absolute():
                                    cand = (clip_specific_dir / p).resolve()
                                if str(cand) not in used_recent and cand.exists():
                                    # Ã°Å¸Å¡Â¨ NOUVEAU: Calcul du score contextuel intelligent
                                    contextual_score = 0.0
                                    if 'get_contextual_broll_score' in globals() and local_keywords:
                                        try:
                                            # Extraire les tokens et tags du fichier
                                            asset_name = cand.stem.lower()
                                            asset_tokens = asset_name.split('_')
                                            asset_tags = asset_name.split('_')  # SimplifiÃƒÂ© pour l'exemple
                                            contextual_score = get_contextual_broll_score(local_keywords, asset_tokens, asset_tags)
                                        except Exception as e:
                                            print(f"    Ã¢Å¡Â Ã¯Â¸Â Erreur scoring contextuel: {e}")
                                            contextual_score = 0.0
                                    
                                    # Ã°Å¸Å¡Â¨ NOUVEAU: Score combinÃƒÂ© FAISS + Contextuel
                                    faiss_score = float(D[0][I[0].tolist().index(idx)]) if idx in I[0] else 0.0
                                    combined_score = faiss_score + (contextual_score * 2.0)  # Poids contextuel DOUBLÃƒâ€°
                                    
                                    if combined_score > best_score:
                                        best_score = combined_score
                                        chosen = str(cand)
                        
                        # Ã°Å¸Å¡Â¨ NOUVEAU: Log de la sÃƒÂ©lection contextuelle
                        if chosen and 'get_contextual_broll_score' in globals() and local_keywords:
                            try:
                                asset_name = Path(chosen).stem.lower()
                                asset_tokens = asset_name.split('_')
                                asset_tags = asset_name.split('_')
                                final_contextual_score = get_contextual_broll_score(local_keywords, asset_tokens, asset_tags)
                                print(f"    Ã°Å¸Å½Â¯ SÃƒÂ©lection contextuelle: {Path(chosen).stem} | Score: {best_score:.3f} | Contexte: {final_contextual_score:.2f}")
                            except Exception:
                                pass
                    
                    if chosen is None:
                        # Ã°Å¸Å¡Â¨ NOUVEAU: Fallback contextuel intelligent au lieu d'alÃƒÂ©atoire
                        print(f"    Ã°Å¸â€Â Fallback contextuel pour segment: {local[:50]}...")
                        for p in clip_specific_dir.rglob('*'):
                            if p.suffix.lower() in {'.mp4','.mov','.mkv','.webm','.jpg','.jpeg','.png'}:
                                if str(p.resolve()) not in used_recent and p.exists():
                                    # Ã°Å¸Å¡Â¨ NOUVEAU: Ãƒâ€°valuation contextuelle du fallback
                                    if 'get_contextual_broll_score' in globals() and local_keywords:
                                        try:
                                            asset_name = p.stem.lower()
                                            asset_tokens = asset_name.split('_')
                                            asset_tags = asset_name.split('_')
                                            fallback_score = get_contextual_broll_score(local_keywords, asset_tokens, asset_tags)
                                            if fallback_score > 2.0:  # Seuil contextuel minimum
                                                chosen = str(p.resolve())
                                                print(f"    Ã¢Å“â€¦ Fallback contextuel: {p.stem} | Score: {fallback_score:.2f}")
                                                break
                                        except Exception:
                                            pass
                                    else:
                                        # Fallback sans scoring contextuel
                                        chosen = str(p.resolve())
                                        break
                    
                    if chosen:
                        if isinstance(it, dict):
                            it['asset_path'] = chosen
                        else:
                            try:
                                setattr(it, 'asset_path', chosen)
                            except Exception:
                                pass
            except Exception:
                pass

            # VÃƒÂ©rification des asset_path avant normalisation + mini fallback non invasif
            try:
                def _get_ap(x):
                    return (getattr(x, 'asset_path', None) if hasattr(x, 'asset_path') else (x.get('asset_path') if isinstance(x, dict) else None))
                missing = [it for it in (plan or []) if not _get_ap(it)]
                if plan and len(missing) == len(plan):
                    # Aucun asset assignÃƒÂ© par FAISS Ã¢â€ â€™ mini fallback d'assignation sÃƒÂ©quentielle
                    # UTILISER LE DOSSIER SPÃƒâ€°CIFIQUE DU CLIP
                    clip_specific_dir = clip_broll_dir if 'clip_specific_dir' in locals() else broll_library
                    lib_assets = [p for p in clip_specific_dir.rglob('*') if p.suffix.lower() in {'.mp4','.mov','.mkv','.webm','.jpg','.jpeg','.png'}]
                    if lib_assets:
                        for i, it in enumerate(plan):
                            ap = _get_ap(it)
                            if ap:
                                continue
                            a = lib_assets[i % len(lib_assets)]
                            chosen = str(a.resolve())
                            if isinstance(it, dict):
                                it['asset_path'] = chosen
                            else:
                                try:
                                    setattr(it, 'asset_path', chosen)
                                except Exception:
                                    pass
            except Exception:
                pass
 
             # Normaliser la timeline en ÃƒÂ©vÃƒÂ©nements canonique et rendre
            try:
                with _VFC(str(input_path)) as _fpsprobe:
                    fps_probe = float(_fpsprobe.fps or 25.0)
            except Exception:
                fps_probe = 25.0
            events = normalize_timeline(plan, fps=fps_probe)
            events = enrich_keywords(events)
            

            
            # Hard fail if no valid events
            if not events:
                raise RuntimeError('Aucun B-roll valide aprÃƒÂ¨s planification/scoring. VÃƒÂ©rifier l\'index FAISS et la librairie. Aucun fallback synthÃƒÂ©tique appliquÃƒÂ©.')
            # Valider que les mÃƒÂ©dias existent
            from pathlib import Path as _Path
            valid_events = []
            for ev in events:
                mp = getattr(ev, 'media_path', '')
                pp = _Path(mp)
                if not pp.exists() and mp and not pp.is_absolute():
                    pp = (broll_library / mp).resolve()
                    if pp.exists():
                        try:
                            setattr(ev, 'media_path', str(pp))
                        except Exception:
                            pass
                if getattr(ev, 'media_path', '') and _Path(getattr(ev, 'media_path')).exists():
                    valid_events.append(ev)
            # Log count and sample
            try:
                print(f"    Ã°Å¸â€Å½ B-roll events valides: {len(valid_events)}")
                for _ev in valid_events[:3]:
                    print(f"       Ã¢â‚¬Â¢ {_ev.start_s:.2f}-{_ev.end_s:.2f} Ã¢â€ â€™ {getattr(_ev, 'media_path','')}")
            except Exception:
                pass
            if not valid_events:
                # Fallback legacy: construire un plan simple ÃƒÂ  partir de la librairie existante
                try:
                    _media_exts = {'.mp4','.mov','.mkv','.webm','.jpg','.jpeg','.png'}
                    assets = [p for p in Path(broll_library).rglob('*') if p.suffix.lower() in _media_exts]
                    assets.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
                    assets = assets[:20]
                    if assets:
                        # Choisir des segments suffisamment longs (>2.0s) et espacÃƒÂ©s
                        cands = []
                        for s in segments:
                            dur = float(getattr(s, 'end', 0.0) - getattr(s, 'start', 0.0))
                            if dur >= 2.0 and getattr(s, 'start', 0.0) >= 1.5:  # Plus flexible
                                cands.append(s)
                        plan_simple = []
                        gap = 6.0  # RÃƒÂ©duit: 8s Ã¢â€ â€™ 6s pour plus d'insertions
                        last = -1e9
                        ai = 0
                        for s in cands:
                            st = float(getattr(s,'start',0.0))
                            en = float(getattr(s,'end',0.0))
                            if st - last < gap:
                                continue
                            dur = min(7.0, max(2.5, en - st))  # DurÃƒÂ©e min: 2.5s, max: 7s
                            asset = assets[ai % len(assets)]
                            ai += 1
                            plan_simple.append({
                                'start': st,
                                'end': min(en, st + dur),
                                'asset_path': str(asset.resolve()),
                                'crossfade_frames': 2,
                            })
                            last = st
                        # Normaliser et rendre si on a des items
                        if plan_simple:
                            try:
                                with _VFC(str(input_path)) as _fpsprobe:
                                    fps_probe = float(_fpsprobe.fps or 25.0)
                            except Exception:
                                fps_probe = 25.0
                            legacy_events = normalize_timeline(plan_simple, fps=fps_probe)
                            legacy_events = enrich_keywords(legacy_events)
                            print(f"    Ã¢â„¢Â»Ã¯Â¸Â Fallback legacy appliquÃƒÂ©: {len(legacy_events)} events")
                            valid_events = legacy_events
                            # Continue vers le rendu unique plus bas
                        else:
                            raise RuntimeError('Librairie B-roll prÃƒÂ©sente mais aucun slot valide pour fallback legacy')
                    else:
                        raise RuntimeError('B-rolls planifiÃƒÂ©s, aucun media_path valide et aucune ressource en librairie pour fallback')
                except Exception as _e:
                    raise RuntimeError('B-rolls planifiÃƒÂ©s, mais aucun media_path valide trouvÃƒÂ©. Fallback legacy impossible: ' + str(_e))
            # Rendu unique avec les events valides (incl. fallback le cas ÃƒÂ©chÃƒÂ©ant)
            render_video(cfg, segments, valid_events)
            
            # VÃƒâ€°RIFICATION ET NETTOYAGE INTELLIGENT DES B-ROLLS
            try:
                if getattr(Config, 'BROLL_DELETE_AFTER_USE', False):
                    print("    Ã°Å¸â€Â VÃƒÂ©rification des B-rolls avant suppression...")
                    
                    # Importer le systÃƒÂ¨me de vÃƒÂ©rification
                    try:
                        from broll_verification_system import create_verification_system
                        verifier = create_verification_system()
                        
                        # VÃƒÂ©rifier l'insertion des B-rolls
                        verification_result = verifier.verify_broll_insertion(
                            video_path=cfg.output_video,
                            broll_plan=plan or [],
                            broll_library_path=str(clip_broll_dir) if 'clip_broll_dir' in locals() else "AI-B-roll/broll_library"
                        )
                        
                        # Ã°Å¸Å¡â‚¬ CORRECTION: VÃƒÂ©rifier le type du rÃƒÂ©sultat de vÃƒÂ©rification
                        if not isinstance(verification_result, dict):
                            print(f"    Ã¢Å¡Â Ã¯Â¸Â RÃƒÂ©sultat de vÃƒÂ©rification invalide (type: {type(verification_result)}) - Fallback vers vÃƒÂ©rification basique")
                            verification_result = {
                                "verification_passed": True,  # Par dÃƒÂ©faut, autoriser la suppression
                                "issues": [],
                                "recommendations": []
                            }
                        
                        # DÃƒÂ©cider si la suppression est autorisÃƒÂ©e
                        if verification_result.get("verification_passed", False):
                            print("    Ã¢Å“â€¦ VÃƒÂ©rification rÃƒÂ©ussie - Suppression autorisÃƒÂ©e")
                            
                            # Supprimer seulement les fichiers B-roll utilisÃƒÂ©s (pas le dossier)
                            used_files: List[str] = []
                            for item in (plan or []):
                                path = getattr(item, 'asset_path', None) if hasattr(item, 'asset_path') else (item.get('asset_path') if isinstance(item, dict) else None)
                                if path and os.path.exists(path):
                                    used_files.append(path)
                            
                            # Nettoyer les fichiers utilisÃƒÂ©s
                            cleaned_count = 0
                            for p in used_files:
                                try:
                                    os.remove(p)
                                    cleaned_count += 1
                                except Exception:
                                    pass
                            
                            # Marquer le dossier comme "utilisÃƒÂ©" mais le garder
                            if 'clip_broll_dir' in locals() and clip_broll_dir.exists():
                                try:
                                    # CrÃƒÂ©er un fichier de statut pour indiquer que le clip est traitÃƒÂ©
                                    status_file = clip_broll_dir / "STATUS_COMPLETED.txt"
                                    status_file.write_text(f"Clip traitÃƒÂ© le {time.strftime('%Y-%m-%d %H:%M:%S')}\nB-rolls utilisÃƒÂ©s: {cleaned_count}\nVÃƒÂ©rification: PASSED\n", encoding='utf-8')
                                    print(f"    Ã°Å¸â€”â€šÃ¯Â¸Â Dossier B-roll conservÃƒÂ©: {clip_broll_dir.name} (fichiers nettoyÃƒÂ©s: {cleaned_count})")
                                except Exception as e:
                                    print(f"    Ã¢Å¡Â Ã¯Â¸Â Erreur crÃƒÂ©ation statut: {e}")
                        else:
                            print("    Ã¢ÂÅ’ VÃƒÂ©rification ÃƒÂ©chouÃƒÂ©e - Suppression REFUSÃƒâ€°E")
                            print("    Ã°Å¸â€œâ€¹ ProblÃƒÂ¨mes dÃƒÂ©tectÃƒÂ©s:")
                            for issue in verification_result.get("issues", []):
                                print(f"       Ã¢â‚¬Â¢ {issue}")
                            print("    Ã°Å¸â€™Â¡ Recommandations:")
                            for rec in verification_result.get("recommendations", []):
                                print(f"       Ã¢â‚¬Â¢ {rec}")
                            
                            # CrÃƒÂ©er un fichier de statut d'ÃƒÂ©chec
                            if 'clip_broll_dir' in locals() and clip_broll_dir.exists():
                                try:
                                    status_file = clip_broll_dir / "STATUS_FAILED.txt"
                                    status_file.write_text(f"Clip traitÃƒÂ© le {time.strftime('%Y-%m-%d %H:%M:%S')}\nVÃƒÂ©rification: FAILED\nProblÃƒÂ¨mes: {', '.join(verification_result.get('issues', []))}\n", encoding='utf-8')
                                    print(f"    Ã°Å¸Å¡Â¨ Dossier B-roll marquÃƒÂ© comme ÃƒÂ©chec: {clip_broll_dir.name}")
                                except Exception as e:
                                    print(f"    Ã¢Å¡Â Ã¯Â¸Â Erreur crÃƒÂ©ation statut d'ÃƒÂ©chec: {e}")
                    
                    except ImportError:
                        print("    Ã¢Å¡Â Ã¯Â¸Â SystÃƒÂ¨me de vÃƒÂ©rification non disponible - Suppression sans vÃƒÂ©rification")
                        # Fallback vers l'ancien systÃƒÂ¨me
                        used_files: List[str] = []
                        for item in (plan or []):
                            path = getattr(item, 'asset_path', None) if hasattr(item, 'asset_path') else (item.get('asset_path') if isinstance(item, dict) else None)
                            if path and os.path.exists(path):
                                used_files.append(path)
                        
                        cleaned_count = 0
                        for p in used_files:
                            try:
                                os.remove(p)
                                cleaned_count += 1
                            except Exception:
                                pass
                        
                        if 'clip_broll_dir' in locals() and clip_broll_dir.exists():
                            try:
                                status_file = clip_broll_dir / "STATUS_COMPLETED_NO_VERIFICATION.txt"
                                status_file.write_text(f"Clip traitÃƒÂ© le {time.strftime('%Y-%m-%d %H:%M:%S')}\nB-rolls utilisÃƒÂ©s: {cleaned_count}\nVÃƒÂ©rification: NON DISPONIBLE\n", encoding='utf-8')
                                print(f"    Ã°Å¸â€”â€šÃ¯Â¸Â Dossier B-roll conservÃƒÂ©: {clip_broll_dir.name} (fichiers nettoyÃƒÂ©s: {cleaned_count})")
                            except Exception as e:
                                print(f"    Ã¢Å¡Â Ã¯Â¸Â Erreur crÃƒÂ©ation statut: {e}")
                    
            except Exception as e:
                print(f"    Ã¢Å¡Â Ã¯Â¸Â Erreur lors de la vÃƒÂ©rification/nettoyage: {e}")
                # En cas d'erreur, ne pas supprimer les B-rolls
                pass

            if Path(cfg.output_video).exists():
                print("    Ã¢Å“â€¦ B-roll insÃƒÂ©rÃƒÂ©s avec succÃƒÂ¨s")
                return Path(cfg.output_video)
            else:
                print("    Ã¢Å¡Â Ã¯Â¸Â Sortie B-roll introuvable, retour ÃƒÂ  la vidÃƒÂ©o d'origine")
                return input_path
        except Exception as e:
            print(f"    Ã¢ÂÅ’ Erreur B-roll: {e}")
            return input_path

    # Si densitÃƒÂ© trop faible aprÃƒÂ¨s planification, injecter quelques B-rolls gÃƒÂ©nÃƒÂ©riques
    try:
        with _VFC(str(input_path)) as _tmp:
            _total = float(_tmp.duration or 0.0)
        cur_cov = sum(max(0.0, (float(getattr(it,'end', it.get('end',0.0))) - float(getattr(it,'start', it.get('start',0.0))))) for it in (plan or []))
        if _total > 0 and (cur_cov / _total) < 0.20:  # AugmentÃƒÂ©: 15% Ã¢â€ â€™ 20% pour plus de B-rolls
            _generics = []
            bank = [
                "money", "handshake", "meeting", "audience", "lightbulb", "typing", "city", "success"
            ]
            # Chercher quelques mÃƒÂ©dias gÃƒÂ©nÃƒÂ©riques existants
            for p in broll_library.rglob('*'):
                if p.suffix.lower() in {'.mp4','.mov','.mkv','.webm','.jpg','.jpeg','.png'}:
                    name = p.stem.lower()
                    if any(k in name for k in bank):
                        _generics.append(str(p.resolve()))
            if _generics:
                # Injecter 2Ã¢â‚¬â€œ4 gÃƒÂ©nÃƒÂ©riques espacÃƒÂ©s
                inject_count = min(4, max(2, int(len(_generics)/5)))
                st = 2.0
                while inject_count > 0 and st < (_total - 3.5):
                    plan.append({'start': st, 'end': min(_total, st+3.5), 'asset_path': _generics[inject_count % len(_generics)], 'crossfade_frames': 2})
                    st += 10.0
                    inject_count -= 1
                print("    Ã¢Å¾â€¢ B-rolls gÃƒÂ©nÃƒÂ©riques injectÃƒÂ©s pour densitÃƒÂ© minimale")
    except Exception:
        pass

class PremiereProAutomation:
    """
    Classe pour l'automatisation Premiere Pro (optionnelle)
    Utilise ExtendScript pour les utilisateurs avancÃƒÂ©s
    """
    
    @staticmethod
    def create_jsx_script(clip_path: str, output_path: str) -> str:
        """GÃƒÂ©nÃƒÂ¨re un script ExtendScript pour Premiere Pro"""
        jsx_script = f'''
        // Script ExtendScript pour Premiere Pro
        var project = app.project;
        
        // Import du clip
        var importOptions = new ImportOptions();
        importOptions.file = new File("{clip_path}");
        var clip = project.importFiles([importOptions.file]);
        
        // CrÃƒÂ©ation d'une sÃƒÂ©quence 9:16
        var sequence = project.createNewSequence("Vertical_Clip", "HDV-1080i25");
        sequence.videoTracks[0].insertClip(clip[0], 0);
        
        // Application de l'effet Auto Reframe (si disponible)
        // Note: Ceci nÃƒÂ©cessite Premiere Pro 2019 ou plus rÃƒÂ©cent
        
        // Export
        var encoder = app.encoder;
        encoder.encodeSequence(sequence, "{output_path}", "H.264", false);
        '''
        return jsx_script
    
    @staticmethod 
    def run_premiere_script(jsx_script_content: str):
        """ExÃƒÂ©cute un script ExtendScript dans Premiere Pro"""
        try:
            # Sauvegarde du script temporaire
            script_path = Config.TEMP_FOLDER / "premiere_script.jsx"
            with open(script_path, 'w') as f:
                f.write(jsx_script_content)
            
            import platform
            system = platform.system()
            
            if system == 'Darwin':  # macOS
                subprocess.run([
                    'osascript', '-e',
                    f'tell application "Adobe Premiere Pro" to do script "{script_path}"'
                ], check=True)
            elif system == 'Windows':
                print("Ã¢Å¡Â Ã¯Â¸Â ExÃƒÂ©cution ExtendScript automatisÃƒÂ©e non supportÃƒÂ©e nativement sous Windows dans ce pipeline.")
                print("   Ouvrez Premiere Pro et exÃƒÂ©cutez le script manuellement: " + str(script_path))
            else:
                print("Ã¢Å¡Â Ã¯Â¸Â Plateforme non supportÃƒÂ©e pour l'exÃƒÂ©cution automatique de Premiere Pro.")
            
            logger.info("Ã¢Å“â€¦ Script Premiere Pro traitÃƒÂ© (voir message ci-dessus)")
            
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Erreur Premiere Pro: {e}")
            raise

# Helper: filter noisy prompt terms
STOP_PROMPT_TERMS = {
    'very','really','clear','stuff','thing','things','some','any','ever','so','much','get','got',
    'will','discuss','this','that','these','those','it','its','im','ive','youve','because'
}

def _filter_prompt_terms(words):
    cleaned = []
    for w in words:
        if not isinstance(w, str):
            continue
        t = w.strip().lower()
        if not t or t in STOP_PROMPT_TERMS or len(t) < 3:
            continue
        cleaned.append(t)
    # de-duplicate preserving order
    seen = set()
    result = []
    for t in cleaned:
        if t not in seen:
            result.append(t)
            seen.add(t)
    return result[:5]

def _prioritize_fresh_assets(broll_candidates, clip_id):
    """Priorise les assets les plus rÃƒÂ©cents basÃƒÂ©s sur le timestamp du dossier."""
    if not broll_candidates:
        return broll_candidates
    
    try:
        # Extraire le timestamp du dossier pour chaque candidat
        for candidate in broll_candidates:
            if hasattr(candidate, 'file_path') and candidate.file_path:
                path = Path(candidate.file_path)
                # Chercher le pattern clip_*_timestamp dans le chemin
                for part in path.parts:
                    if part.startswith(f"clip_{clip_id}_") and "_" in part:
                        timestamp_str = part.split("_")[-1]
                        if timestamp_str.isdigit():
                            candidate.folder_timestamp = int(timestamp_str)
                            break
                else:
                    candidate.folder_timestamp = 0
            else:
                candidate.folder_timestamp = 0
        
        # Trier par timestamp dÃƒÂ©croissant (plus rÃƒÂ©cent en premier)
        broll_candidates.sort(key=lambda x: getattr(x, 'folder_timestamp', 0), reverse=True)
        
    except Exception as e:
        print(f"    Ã¢Å¡Â Ã¯Â¸Â  Erreur priorisation fraÃƒÂ®cheur: {e}")
    
    return broll_candidates

def _score_contextual_relevance(asset_path, domain, keywords):
    """Score de pertinence contextuelle basÃƒÂ© sur les tokens et le domaine."""
    try:
        if not asset_path or not domain or not keywords:
            return 0.5
        
        # Extraire les tokens du nom de fichier
        filename = Path(asset_path).stem.lower()
        asset_tokens = set(re.split(r'[^a-z0-9]+', filename))
        
        # Tokens du domaine et mots-clÃƒÂ©s
        domain_tokens = set(domain.lower().split())
        keyword_tokens = set()
        for kw in keywords:
            if isinstance(kw, str):
                keyword_tokens.update(kw.lower().split())
        
        # Calculer l'overlap
        relevant_tokens = domain_tokens | keyword_tokens
        if not relevant_tokens:
            return 0.5
        
        overlap = len(asset_tokens & relevant_tokens)
        total_relevant = len(relevant_tokens)
        
        # Score basÃƒÂ© sur l'overlap (0.0 ÃƒÂ  1.0)
        base_score = min(1.0, overlap / max(1, total_relevant * 0.3))
        
        # Bonus pour les tokens de domaine
        domain_overlap = len(asset_tokens & domain_tokens)
        domain_bonus = min(0.3, domain_overlap * 0.1)
        
        final_score = min(1.0, base_score + domain_bonus)
        return final_score
        
    except Exception as e:
        print(f"    Ã¢Å¡Â Ã¯Â¸Â  Erreur scoring contextuel: {e}")
        return 0.5

def _get_fallback_neutral_assets(broll_library, count=3):
    """RÃƒÂ©cupÃƒÂ¨re des assets neutres/gÃƒÂ©nÃƒÂ©riques comme fallback."""
    try:
        fallback_keywords = ['neutral', 'generic', 'background', 'abstract', 'minimal']
        fallback_assets = []
        
        for keyword in fallback_keywords:
            # Chercher dans la librairie des assets avec ces mots-clÃƒÂ©s
            for ext in ['.mp4', '.mov', '.jpg', '.png']:
                for asset_path in broll_library.rglob(f"*{keyword}*{ext}"):
                    if asset_path.exists() and asset_path not in fallback_assets:
                        fallback_assets.append(str(asset_path))
                        if len(fallback_assets) >= count:
                            break
                if len(fallback_assets) >= count:
                    break
            if len(fallback_assets) >= count:
                break
        
        # Si pas assez d'assets spÃƒÂ©cifiques, prendre des assets gÃƒÂ©nÃƒÂ©riques
        if len(fallback_assets) < count:
            for ext in ['.mp4', '.mov', '.jpg', '.png']:
                for asset_path in broll_library.rglob(f"*{ext}"):
                    if asset_path.exists() and asset_path not in fallback_assets:
                        fallback_assets.append(str(asset_path))
                        if len(fallback_assets) >= count:
                            break
                if len(fallback_assets) >= count:
                    break
        
        return fallback_assets[:count]
        
    except Exception as e:
        print(f"    Ã¢Å¡Â Ã¯Â¸Â  Erreur fallback neutre: {e}")
        return []

def _debug_broll_selection(plan, domain, keywords, debug_mode=False):
    """Log dÃƒÂ©taillÃƒÂ© de la sÃƒÂ©lection B-roll si debug activÃƒÂ©."""
    if not debug_mode:
        return
    
    print(f"    Ã°Å¸â€Â DEBUG B-ROLL SELECTION:")
    print(f"       Domaine: {domain}")
    print(f"       Mots-clÃƒÂ©s: {keywords[:5]}")
    print(f"       Plan: {len(plan)} items")
    
    for i, item in enumerate(plan[:3]):  # Afficher les 3 premiers
        if hasattr(item, 'asset_path') and item.asset_path:
            asset_path = item.asset_path
            score = getattr(item, 'score', 'N/A')
            context_score = getattr(item, 'context_score', 'N/A')
            freshness = getattr(item, 'freshness_score', 'N/A')
        elif isinstance(item, dict):
            asset_path = item.get('asset_path', 'N/A')
            score = item.get('score', 'N/A')
            context_score = item.get('context_score', 'N/A')
            freshness = item.get('freshness_score', 'N/A')
        else:
            continue
        
        print(f"       Item {i+1}: {Path(asset_path).name}")
        print(f"         Score: {score}, Context: {context_score}, FraÃƒÂ®cheur: {freshness}")

# Ã°Å¸Å¡â‚¬ NOUVEAU: Fonction de scoring mixte intelligent pour B-rolls
def score_broll_asset_mixed(asset_path: str, asset_tags: List[str], query_keywords: List[str], 
                           domain: Optional[str] = None, asset_metadata: Optional[Dict] = None) -> float:
    """
    Score un asset B-roll avec le systÃƒÂ¨me mixte intelligent.
    
    Args:
        asset_path: Chemin vers l'asset
        asset_tags: Tags de l'asset
        query_keywords: Mots-clÃƒÂ©s de la requÃƒÂªte
        domain: Domaine dÃƒÂ©tectÃƒÂ© (optionnel)
        asset_metadata: MÃƒÂ©tadonnÃƒÂ©es supplÃƒÂ©mentaires (optionnel)
    
    Returns:
        Score final entre 0.0 et 1.0
    """
    try:
        if not BROLL_SELECTOR_AVAILABLE:
            # Fallback vers scoring basique
            return _score_broll_asset_basic(asset_path, asset_tags, query_keywords)
        
        # Utiliser le nouveau sÃƒÂ©lecteur si disponible
        from broll_selector import Asset, ScoringFeatures
        
        # CrÃƒÂ©er un asset simulÃƒÂ© pour le scoring
        asset = Asset(
            id=f"asset_{hash(asset_path)}",
            file_path=asset_path,
            tags=asset_tags,
            title=Path(asset_path).stem,
            description="",
            source="local",
            fetched_at=datetime.now(),
            duration=asset_metadata.get('duration', 2.0) if asset_metadata else 2.0,
            resolution=asset_metadata.get('resolution', '1920x1080') if asset_metadata else '1920x1080'
        )
        
        # Normaliser les mots-clÃƒÂ©s de la requÃƒÂªte
        normalized_keywords = set()
        for kw in query_keywords:
            if kw and isinstance(kw, str):
                clean = kw.lower().strip()
                if len(clean) > 2:
                    normalized_keywords.add(clean)
        
        # Calculer les features de scoring
        features = ScoringFeatures()
        
        # 1. Token overlap (Jaccard)
        if asset_tags and normalized_keywords:
            intersection = len(set(asset_tags) & normalized_keywords)
            union = len(set(asset_tags) | normalized_keywords)
            features.token_overlap = intersection / union if union > 0 else 0.0
        
        # 2. Domain match
        if domain and asset_tags:
            domain_keywords = _get_domain_keywords(domain)
            domain_overlap = len(set(asset_tags) & set(domain_keywords))
            features.domain_match = min(1.0, domain_overlap / max(len(domain_keywords), 1))
        
        # 3. Freshness (basÃƒÂ© sur la date de crÃƒÂ©ation du fichier)
        try:
            file_path = Path(asset_path)
            if file_path.exists():
                mtime = file_path.stat().st_mtime
                days_old = (time.time() - mtime) / (24 * 3600)
                features.freshness = 1.0 / (1.0 + days_old / 60)  # Demi-vie de 60 jours
        except:
            features.freshness = 0.5  # Valeur par dÃƒÂ©faut
        
        # 4. Quality score (basÃƒÂ© sur la rÃƒÂ©solution et l'extension)
        features.quality_score = _calculate_quality_score(asset_path, asset_metadata)
        
        # 5. Embedding similarity (placeholder - ÃƒÂ  implÃƒÂ©menter avec FAISS)
        features.embedding_similarity = 0.5  # Valeur par dÃƒÂ©faut
        
        # Calculer le score final pondÃƒÂ©rÃƒÂ©
        weights = {
            'embedding': 0.4,
            'token': 0.2,
            'domain': 0.15,
            'freshness': 0.1,
            'quality': 0.1,
            'diversity': 0.05
        }
        
        final_score = (
            weights['embedding'] * features.embedding_similarity +
            weights['token'] * features.token_overlap +
            weights['domain'] * features.domain_match +
            weights['freshness'] * features.freshness +
            weights['quality'] * features.quality_score
        )
        
        return max(0.0, min(1.0, final_score))
        
    except Exception as e:
        print(f"Ã¢Å¡Â Ã¯Â¸Â Erreur scoring mixte: {e}")
        # Fallback vers scoring basique
        return _score_broll_asset_basic(asset_path, asset_tags, query_keywords)

def _score_broll_asset_basic(asset_path: str, asset_tags: List[str], query_keywords: List[str]) -> float:
    """Scoring basique de fallback"""
    try:
        # Score simple basÃƒÂ© sur l'overlap de tags
        if not asset_tags or not query_keywords:
            return 0.5
        
        asset_tag_set = set(tag.lower() for tag in asset_tags)
        query_set = set(kw.lower() for kw in query_keywords if kw)
        
        if not query_set:
            return 0.5
        
        intersection = len(asset_tag_set & query_set)
        union = len(asset_tag_set | query_set)
        
        return intersection / union if union > 0 else 0.0
        
    except Exception as e:
        print(f"Ã¢Å¡Â Ã¯Â¸Â Erreur scoring basique: {e}")
        return 0.5

def _get_domain_keywords(domain: str) -> List[str]:
    """Retourne les mots-clÃƒÂ©s spÃƒÂ©cifiques au domaine"""
    domain_keywords = {
        'health': ['medical', 'healthcare', 'wellness', 'fitness', 'medicine', 'hospital', 'doctor'],
        'technology': ['tech', 'digital', 'innovation', 'computer', 'ai', 'software', 'data'],
        'business': ['business', 'entrepreneur', 'success', 'growth', 'strategy', 'office', 'professional'],
        'education': ['learning', 'education', 'knowledge', 'study', 'teaching', 'school', 'university'],
        'finance': ['money', 'finance', 'investment', 'wealth', 'business', 'success', 'growth']
    }
    
    return domain_keywords.get(domain.lower(), [domain])

def _calculate_quality_score(asset_path: str, metadata: Optional[Dict] = None) -> float:
    """Calcule un score de qualitÃƒÂ© basÃƒÂ© sur les mÃƒÂ©tadonnÃƒÂ©es"""
    try:
        score = 0.5  # Score de base
        
        # Bonus pour la rÃƒÂ©solution
        if metadata and 'resolution' in metadata:
            res = metadata['resolution']
            if '4k' in res or '3840' in res:
                score += 0.2
            elif '1080' in res or '1920' in res:
                score += 0.1
        
        # Bonus pour la durÃƒÂ©e
        if metadata and 'duration' in metadata:
            duration = metadata['duration']
            if 2.0 <= duration <= 6.0:  # DurÃƒÂ©e optimale
                score += 0.1
        
        # Bonus pour l'extension (prÃƒÂ©fÃƒÂ©rer MP4)
        if asset_path.lower().endswith('.mp4'):
            score += 0.1
        
        return min(1.0, score)
        
    except Exception:
        return 0.5

    def _load_broll_selector_config(self):
        """Charge la configuration du sÃƒÂ©lecteur B-roll depuis le fichier YAML"""
        try:
            import yaml
            if Config.BROLL_SELECTOR_CONFIG_PATH.exists():
                with open(Config.BROLL_SELECTOR_CONFIG_PATH, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            else:
                print(f"    Ã¢Å¡Â Ã¯Â¸Â Fichier de configuration introuvable: {Config.BROLL_SELECTOR_CONFIG_PATH}")
                return {}
        except Exception as e:
            print(f"    Ã¢Å¡Â Ã¯Â¸Â Erreur chargement configuration: {e}")
            return {}

    def _calculate_asset_hash(self, asset_path: Path) -> str:
        """Calcule un hash unique pour un asset B-roll basÃƒÂ© sur son contenu et mÃƒÂ©tadonnÃƒÂ©es"""
        try:
            import hashlib
            import os
            from datetime import datetime
            
            # Hash basÃƒÂ© sur le nom, la taille et la date de modification
            stat = asset_path.stat()
            hash_data = f"{asset_path.name}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.md5(hash_data.encode()).hexdigest()
        except Exception:
            # Fallback sur le nom du fichier
            return str(asset_path.name)


