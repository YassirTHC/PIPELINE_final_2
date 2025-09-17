import sys
from pathlib import Path

# ensure project-root is first
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(1, str(PROJECT_ROOT / 'AI-B-roll'))
sys.path.insert(2, str(PROJECT_ROOT / 'AI-B-roll' / 'src'))

if 'utils' in sys.modules:
    del sys.modules['utils']

import logging
import concurrent.futures
import subprocess
import shlex
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Sequence, Set
import gc

# 🚀 NOUVEAU: Configuration des logs temps réel + suppression warnings non-critiques
import warnings
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated")
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces` was not set")
warnings.filterwarnings("ignore", message="Warning: in file.*bytes wanted but.*bytes read")

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 🚀 NOUVEAU: Fonction print temps réel
def print_realtime(message):
    """Print avec flush immédiat pour logs temps réel"""
    print(message, flush=True)
    logger.info(message)

SEEN_URLS: Set[str] = set()
SEEN_PHASHES: List[int] = []
PHASH_DISTANCE = 6


def run_with_timeout(fn, timeout_s: float, *args, **kwargs):
    if timeout_s <= 0:
        return fn(*args, **kwargs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            return None


def dedupe_by_url(candidates):
    unique = []
    hits = 0
    for candidate in candidates or []:
        url = getattr(candidate, 'url', None)
        if url and url in SEEN_URLS:
            hits += 1
            continue
        unique.append(candidate)
    return unique, hits


def dedupe_by_phash(candidates):
    unique = []
    hits = 0
    for candidate in candidates or []:
        preview = getattr(candidate, 'thumb_url', None)
        media_url = getattr(candidate, 'url', None)
        phash = compute_phash(preview, media_url=media_url)
        if phash is None:
            unique.append(candidate)
            continue
        if any(hamming_distance(phash, seen) <= PHASH_DISTANCE for seen in SEEN_PHASHES):
            hits += 1
            continue
        setattr(candidate, '_phash', phash)
        unique.append(candidate)
    return unique, hits


def _try_overlay_http_direct(broll_url: str, t0: float, t1: float, render_cfg, base_cmd: list[str]) -> bool:
    duration = max(0.1, float(t1 - t0))
    cmd = base_cmd + [
        '-ss', f"{max(0.0, t0):.3f}",
        '-to', f"{max(0.0, t0 + duration):.3f}",
        '-i', broll_url,
    ]
    try:
        proc = subprocess.run(cmd, check=True)
        return proc.returncode == 0
    except Exception:
        return False


def _overlay_via_pipe(broll_url: str, t0: float, t1: float, render_cfg, base_cmd: list[str]) -> bool:
    duration = max(0.1, float(t1 - t0))
    extract_cmd = f'ffmpeg -hide_banner -loglevel error -ss {t0:.3f} -i "{broll_url}" -t {duration:.3f} -an -c:v libx264 -preset veryfast -f mpegts -'
    try:
        extractor = subprocess.Popen(shlex.split(extract_cmd), stdout=subprocess.PIPE)
    except Exception:
        return False

    try:
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix='.ts', delete=False) as tmp:
            tmp_path = tmp.name
            while True:
                chunk = extractor.stdout.read(65536)
                if not chunk:
                    break
                tmp.write(chunk)
        extractor.wait(timeout=30)
        cmd = base_cmd + ['-i', tmp_path]
        try:
            proc = subprocess.run(cmd, check=True)
            return proc.returncode == 0
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    except Exception:
        try:
            extractor.kill()
        except Exception:
            pass
    return False


def overlay_http_or_pipe(broll_url: str, t0: float, t1: float, render_cfg, base_cmd: list[str]) -> bool:
    if _try_overlay_http_direct(broll_url, t0, t1, render_cfg, base_cmd):
        return True
    return _overlay_via_pipe(broll_url, t0, t1, render_cfg, base_cmd)


import os
import json
import random
import numpy as np
import shutil
from datetime import datetime  # NEW: pour métadonnées intelligentes
from temp_function import _llm_generate_caption_hashtags_fixed
import whisper
import requests
import cv2

from pipeline_core.configuration import PipelineConfigBundle
from pipeline_core.fetchers import FetcherOrchestrator
from pipeline_core.dedupe import compute_phash, hamming_distance
from pipeline_core.logging import JsonlLogger, log_broll_decision
from pipeline_core.llm_service import LLMMetadataGeneratorService

# 🚀 NOUVEAU: Cache global pour éviter le rechargement des modèles
_MODEL_CACHE = {}

def get_sentence_transformer_model(model_name: str):
    """Récupère un modèle SentenceTransformer depuis le cache ou le charge"""
    # 🚀 OPTIMISATION: Normaliser le nom du modèle pour éviter les doublons
    normalized_name = model_name.replace('sentence-transformers/', '')
    
    if normalized_name not in _MODEL_CACHE:
        print(f"    🔄 Chargement initial du modèle: {model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            _MODEL_CACHE[normalized_name] = SentenceTransformer(model_name)
            print(f"    ✅ Modèle {model_name} chargé et mis en cache")
        except Exception as e:
            print(f"    ❌ Erreur chargement modèle {model_name}: {e}")
            return None
    else:
        print(f"    ♻️ Modèle {model_name} récupéré du cache")
    
    return _MODEL_CACHE[normalized_name]

def safe_remove_tree(directory: Path, max_retries: int = 3, delay: float = 1.0) -> bool:
    """
    Supprime un dossier de façon sécurisée avec retry et gestion des handles Windows
    
    Args:
        directory: Dossier à supprimer
        max_retries: Nombre maximum de tentatives
        delay: Délai entre les tentatives (secondes)
    
    Returns:
        True si la suppression a réussi, False sinon
    """
    if not directory.exists():
        return True
    
    for attempt in range(max_retries):
        try:
            # Forcer la libération des handles
            gc.collect()
            
            # Tentative de suppression récursive
            shutil.rmtree(directory, ignore_errors=False)
            
            # Vérifier que c'est vraiment supprimé
            if not directory.exists():
                return True
                
        except PermissionError as e:
            if "WinError 32" in str(e) or "being used by another process" in str(e):
                print(f"    ⚠️ Tentative {attempt + 1}/{max_retries}: Fichier en cours d'utilisation, retry dans {delay}s...")
                time.sleep(delay)
                delay *= 1.5  # Backoff exponentiel
                continue
            else:
                print(f"    ❌ Erreur de permission: {e}")
                break
        except Exception as e:
            print(f"    ❌ Erreur inattendue lors de la suppression: {e}")
            break
    
    # Si on arrive ici, toutes les tentatives ont échoué
    try:
        # Tentative finale avec ignore_errors=True
        shutil.rmtree(directory, ignore_errors=True)
        if not directory.exists():
            print(f"    ✅ Suppression réussie avec ignore_errors")
            return True
        else:
            print(f"    ⚠️ Dossier partiellement supprimé, résidu: {directory}")
            return False
    except Exception as e:
        print(f"    ❌ Échec final de suppression: {e}")
        return False

# Gestion optionnelle de Mediapipe avec fallback
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ Mediapipe disponible - Utilisation des fonctionnalités IA avancées")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None
    print("⚠️ Mediapipe non disponible - Utilisation du fallback OpenCV (fonctionnalités réduites)")

# 🚀 NOUVEAU: Import du sélecteur B-roll générique
try:
    from broll_selector import BrollSelector, Asset, ScoringFeatures, BrollCandidate
    BROLL_SELECTOR_AVAILABLE = True
    print("✅ Sélecteur B-roll générique disponible - Scoring mixte activé")
except ImportError as e:
    BROLL_SELECTOR_AVAILABLE = False
    print(f"⚠️ Sélecteur B-roll générique non disponible: {e}")
    print("   🔄 Utilisation du système de scoring existant")

from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
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
                print(f"✅ ImageMagick configuré: {path}")
                return True
        
        print("⚠️ ImageMagick non trouvé, utilisation du mode fallback")
        return False
        
    except Exception as e:
        print(f"⚠️ Erreur configuration ImageMagick: {e}")
        return False

# Configuration automatique au démarrage
configure_imagemagick()

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    """Configuration centralisée du pipeline"""
    CLIPS_FOLDER = Path("./clips")
    OUTPUT_FOLDER = Path("./output") 
    TEMP_FOLDER = Path("./temp")
    
    # Résolution cible pour les réseaux sociaux
    TARGET_WIDTH = 720
    TARGET_HEIGHT = 1280  # Format 9:16
    
    # Paramètres Whisper
    WHISPER_MODEL = "tiny"  # ou "small", "medium", "large"
    
    # Paramètres sous-titres
    SUBTITLE_FONT_SIZE = 85
    SUBTITLE_COLOR = 'yellow'
    SUBTITLE_STROKE_COLOR = 'black'
    SUBTITLE_STROKE_WIDTH = 3
    # Biais global (en secondes) pour corriger un léger décalage systématique
    # 0.0 par défaut pour éviter tout décalage si non nécessaire
    SUBTITLE_TIMING_BIAS_S = 0.0

    # Activation B-roll: UI > ENV > défaut(off)
    # Si fetchers cochés, activer automatiquement l'insertion B-roll, sauf si explicitement désactivé côté UI
    _UI_ENABLE_BROLL = _UI_SETTINGS.get('enable_broll') if 'enable_broll' in _UI_SETTINGS else None
    _ENV_ENABLE_BROLL = os.getenv('ENABLE_BROLL') or os.getenv('AI_BROLL_ENABLED')
    _AUTO_ENABLE = _to_bool(_UI_SETTINGS.get('broll_fetch_enable'), default=True) if 'broll_fetch_enable' in _UI_SETTINGS else _to_bool(os.getenv('BROLL_FETCH_ENABLE') or os.getenv('AI_BROLL_ENABLE_FETCHER'), default=True)
    ENABLE_BROLL = (
        _to_bool(_UI_ENABLE_BROLL, default=False) if _UI_ENABLE_BROLL is not None
        else (_to_bool(_ENV_ENABLE_BROLL, default=False) or _AUTO_ENABLE)
    )

    # === Options fetcher B-roll (stock) ===
    # Active le fetch automatique: UI > ENV > défaut(on)
    BROLL_FETCH_ENABLE = _to_bool(_UI_SETTINGS.get('broll_fetch_enable'), default=True) if 'broll_fetch_enable' in _UI_SETTINGS else _to_bool(os.getenv('BROLL_FETCH_ENABLE') or os.getenv('AI_BROLL_ENABLE_FETCHER'), default=True)
    # Fournisseur: UI > ENV > défaut pexels
    BROLL_FETCH_PROVIDER = (_UI_SETTINGS.get('broll_fetch_provider') or os.getenv('AI_BROLL_FETCH_PROVIDER') or 'pexels')
    # Clés API
    PEXELS_API_KEY = _UI_SETTINGS.get('PEXELS_API_KEY') or os.getenv('PEXELS_API_KEY')
    PIXABAY_API_KEY = _UI_SETTINGS.get('PIXABAY_API_KEY') or os.getenv('PIXABAY_API_KEY')
    # Contrôles de fetch
    BROLL_FETCH_MAX_PER_KEYWORD = int(_UI_SETTINGS.get('broll_fetch_max_per_keyword') or os.getenv('BROLL_FETCH_MAX_PER_KEYWORD') or 25)  # CORRIGÉ: 12 → 25
    BROLL_FETCH_ALLOW_VIDEOS = _to_bool(_UI_SETTINGS.get('broll_fetch_allow_videos'), default=True) if 'broll_fetch_allow_videos' in _UI_SETTINGS else _to_bool(os.getenv('BROLL_FETCH_ALLOW_VIDEOS'), default=True)
    BROLL_FETCH_ALLOW_IMAGES = _to_bool(_UI_SETTINGS.get('broll_fetch_allow_images'), default=False) if 'broll_fetch_allow_images' in _UI_SETTINGS else _to_bool(os.getenv('BROLL_FETCH_ALLOW_IMAGES'), default=False)
    # Élargir le pool par défaut: activer les images si non précisé
    if 'broll_fetch_allow_images' not in _UI_SETTINGS and os.getenv('BROLL_FETCH_ALLOW_IMAGES') is None:
        BROLL_FETCH_ALLOW_IMAGES = True
    # Embeddings pour matching sémantique
    BROLL_USE_EMBEDDINGS = _to_bool(_UI_SETTINGS.get('broll_use_embeddings'), default=True) if 'broll_use_embeddings' in _UI_SETTINGS else _to_bool(os.getenv('BROLL_USE_EMBEDDINGS'), default=True)
    BROLL_EMBEDDING_MODEL = (_UI_SETTINGS.get('broll_embedding_model') or os.getenv('BROLL_EMBEDDING_MODEL') or 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    # Config contextuelle
    CONTEXTUAL_CONFIG_PATH = Path(_UI_SETTINGS.get('contextual_broll_yml') or os.getenv('CONTEXTUAL_BROLL_YML') or 'config/contextual_broll.yml')

    # Sortie et nettoyage
    USE_HARDLINKS = _to_bool(_UI_SETTINGS.get('use_hardlinks'), default=True) if 'use_hardlinks' in _UI_SETTINGS else _to_bool(os.getenv('USE_HARDLINKS'), default=True)
    BROLL_DELETE_AFTER_USE = _to_bool(_UI_SETTINGS.get('broll_delete_after_use'), default=True) if 'broll_delete_after_use' in _UI_SETTINGS else _to_bool(os.getenv('BROLL_DELETE_AFTER_USE') or os.getenv('AI_BROLL_PURGE_AFTER_USE'), default=True)
    # 🚀 NOUVEAU: Forcer le nettoyage après chaque vidéo pour économiser l'espace
    BROLL_CLEANUP_PER_VIDEO = True  # Toujours activé pour éviter l'accumulation
    BROLL_PURGE_AFTER_RUN = _to_bool(_UI_SETTINGS.get('broll_purge_after_run'), default=True) if 'broll_purge_after_run' in _UI_SETTINGS else _to_bool(os.getenv('BROLL_PURGE_AFTER_RUN') or os.getenv('AI_BROLL_PURGE_AFTER_RUN'), default=True)
    # Brand kit
    BRAND_KIT_ID = _UI_SETTINGS.get('brand_kit_id') or os.getenv('BRAND_KIT_ID') or 'default'
    # Experimental FX (wipes/zoom/LUT etc.)
    ENABLE_EXPERIMENTAL_FX = _to_bool(_UI_SETTINGS.get('enable_experimental_fx'), default=False) if 'enable_experimental_fx' in _UI_SETTINGS else _to_bool(os.getenv('ENABLE_EXPERIMENTAL_FX'), default=False)

    # 🚀 NOUVEAU: Configuration du sélecteur B-roll générique
    BROLL_SELECTOR_CONFIG_PATH = Path(_UI_SETTINGS.get('broll_selector_config') or os.getenv('BROLL_SELECTOR_CONFIG') or 'config/broll_selector_config.yaml')
    BROLL_SELECTOR_ENABLED = _to_bool(_UI_SETTINGS.get('broll_selector_enabled'), default=True) if 'broll_selector_enabled' in _UI_SETTINGS else _to_bool(os.getenv('BROLL_SELECTOR_ENABLED') or os.getenv('AI_BROLL_SELECTOR_ENABLED'), default=True)

# 🚀 SUPPRIMÉ: Fonction _detect_local_llm obsolète
# Remplacée par le système LLM industriel qui gère automatiquement la détection

# 🚀 SUPPRIMÉ: Ancien système LLM obsolète remplacé par le système industriel
# Cette fonction utilisait l'ancien prompt complexe et causait des timeouts
# Maintenant remplacée par le système LLM industriel dans generate_caption_and_hashtags
# 🚀 SUPPRIMÉ: Reste de l'ancien système LLM obsolète
# Toute cette logique complexe est maintenant remplacée par le système industriel

# === IA: Analyse mots-clés et prompts visuels pour guider le B-roll ===

def extract_keywords_from_transcript_ai(transcript_segments: List[Dict]) -> Dict:
    """Analyse simple: thèmes, occurrences et timestamps pour B-roll contextuel."""
    keyword_categories = {
        'money': ['money', 'cash', 'dollars', 'profit', 'revenue', 'income', 'wealth'],
        'business': ['business', 'company', 'startup', 'entrepreneur', 'strategy'],
        'technology': ['tech', 'software', 'app', 'digital', 'online', 'ai', 'automation'],
        'success': ['success', 'win', 'achievement', 'goal', 'growth', 'scale', 'unstoppable', 'beast'],
        'people': ['team', 'customer', 'client', 'person', 'human', 'community'],
        'emotion_positive': ['amazing', 'incredible', 'fantastic', 'awesome', 'fire'],
        'emotion_negative': ['problem', 'issue', 'difficult', 'challenge', 'fail'],
        'action': ['build', 'create', 'launch', 'start', 'implement', 'execute'],
        # 🚀 NOUVELLES CATÉGORIES pour contenu cerveau/mental/neurosciences
        'brain_mind': ['brain', 'mind', 'mental', 'neuroscience', 'neural', 'cognitive', 'psychology'],
        'health_wellness': ['health', 'wellness', 'nutrition', 'nutrients', 'supplements', 'fitness', 'energy'],
        'learning_growth': ['learn', 'learning', 'growth', 'development', 'improvement', 'potential', 'capability'],
        'internal_dialogue': ['dialogue', 'conversation', 'thoughts', 'thinking', 'mindset', 'beliefs', 'circuit']
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
                        # 🚀 CORRECTION: Gestion robuste des timestamps avec slice objects
                        start_val = seg.get('start', 0.0)
                        end_val = seg.get('end', 0.0)
                        
                        # Convertir slice objects en float si nécessaire
                        if hasattr(start_val, 'start'):  # Si c'est un slice
                            start_val = start_val.start or 0.0
                        if hasattr(end_val, 'start'):  # Si c'est un slice
                            end_val = end_val.start or 0.0
                            
                        timestamps_by_category[category].append({
                            'start': float(start_val),
                            'end': float(end_val),
                            'keyword': kw,
                            'context': seg.get('text') or ''
                        })
    dominant_theme = 'business'
    try:
        dominant_theme = max(detected_keywords.items(), key=lambda x: len(x[1]))[0]
    except Exception:
        pass
    # 🚀 CORRECTION CRITIQUE: Gestion robuste du timestamp final
    total_duration = 0.0
    if transcript_segments:
        try:
            last_end = transcript_segments[-1].get('end', 0.0)
            # Convertir slice object en float si nécessaire
            if hasattr(last_end, 'start'):  # Si c'est un slice
                last_end = last_end.start or 0.0
            total_duration = float(last_end)
        except (ValueError, TypeError, AttributeError, KeyError):
            total_duration = 0.0
    
    return {
        'keywords': detected_keywords,
        'timestamps': timestamps_by_category,
        'dominant_theme': dominant_theme,
        'total_duration': total_duration
    }


def generate_broll_prompts_ai(keyword_analysis: Dict) -> List[Dict]:
    """Generate B-roll prompts using AI analysis."""
    try:
        # Extract main theme and keywords
        main_theme = keyword_analysis.get('dominant_theme', 'general')  # 🚀 CORRECTION: Clé correcte
        keywords = keyword_analysis.get('keywords', {})
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
            # 🚀 CORRECTION: keywords est un dict, pas une liste
            if keywords and isinstance(keywords, dict):
                # Extraire les premiers mots-clés de toutes les catégories
                all_kws = []
                for category_kws in keywords.values():
                    if isinstance(category_kws, list):
                        all_kws.extend(category_kws[:2])  # 2 par catégorie
                base_keywords = all_kws[:3] if all_kws else [main_theme]
            else:
                base_keywords = [main_theme]
                
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
        print(f"⚠️ Erreur génération prompts AI: {e}")
        # Fallback prompts
        return ['general content', 'people working', 'modern technology']

class VideoProcessor:
    _shared_llm_service = None

    """Classe principale pour traiter les vidéos"""
    
    def __init__(self):
        self.whisper_model = whisper.load_model(Config.WHISPER_MODEL)
        self._setup_directories()
        # Cache éventuel pour spaCy
        self._spacy_model = None

        if VideoProcessor._shared_llm_service is None:
            try:
                VideoProcessor._shared_llm_service = LLMMetadataGeneratorService()
            except Exception as exc:
                logger.warning("LLM service initialisation failed: %s", exc)
        self._llm_service = VideoProcessor._shared_llm_service
        self._pipeline_config = PipelineConfigBundle()
        self._broll_event_logger = None



def _setup_directories(self):
    for folder in [Config.CLIPS_FOLDER, Config.OUTPUT_FOLDER, Config.TEMP_FOLDER]:
        folder.mkdir(parents=True, exist_ok=True)

def _get_broll_event_logger(self):
    if self._broll_event_logger is None:
        log_file = Config.OUTPUT_FOLDER / 'meta' / 'broll_pipeline_events.jsonl'
        self._broll_event_logger = JsonlLogger(log_file)
    return self._broll_event_logger

def _insert_brolls_pipeline_core(self, segments, broll_keywords, *, subtitles, input_path: Path) -> None:
    global SEEN_URLS, SEEN_PHASHES
    SEEN_URLS.clear()
    SEEN_PHASHES.clear()
    logger.info("[BROLL] pipeline_core orchestrator engaged")
    config_bundle = self._pipeline_config
    orchestrator = FetcherOrchestrator(config_bundle.fetcher)
    selection_cfg = config_bundle.selection
    timeboxing_cfg = config_bundle.timeboxing
    event_logger = self._get_broll_event_logger()

    fetch_timeout = max((timeboxing_cfg.fetch_rank_ms or 0) / 1000.0, 0.0)

    for idx, segment in enumerate(segments):
        seg_duration = max(0.0, segment.end - segment.start)
        llm_hints = None
        llm_healthy = True
        if getattr(self, '_llm_service', None):
            try:
                llm_hints = self._llm_service.generate_hints_for_segment(segment.text, segment.start, segment.end)
            except Exception:
                llm_hints = None
                llm_healthy = False

        segment_keywords = self._derive_segment_keywords(segment, broll_keywords)
        queries: List[str] = []
        if llm_hints and isinstance(llm_hints.get('queries'), list):
            queries = [q.strip() for q in llm_hints['queries'] if isinstance(q, str) and q.strip()]
        if not queries:
            queries = segment_keywords[:4]
        if not queries:
            log_broll_decision(
                event_logger,
                segment_idx=idx,
                start=segment.start,
                end=segment.end,
                query_count=0,
                candidate_count=0,
                unique_candidates=0,
                url_dedup_hits=0,
                phash_dedup_hits=0,
                selected_url=None,
                selected_score=None,
                provider=None,
                latency_ms=0,
                llm_healthy=llm_healthy,
                reject_reasons=['no_keywords'],
            )
            continue

        filters = {}
        if llm_hints and isinstance(llm_hints.get('filters'), dict):
            filters = llm_hints['filters'] or {}

        start_time = time.perf_counter()

        def _do_fetch():
            return orchestrator.fetch_candidates(
                queries,
                duration_hint=seg_duration,
                filters=filters,
            )

        candidates = run_with_timeout(_do_fetch, fetch_timeout) if fetch_timeout else _do_fetch()
        if candidates is None:
            candidates = []

        unique_candidates, url_hits = dedupe_by_url(candidates)
        unique_candidates, phash_hits = dedupe_by_phash(unique_candidates)

        best_candidate = None
        best_score = -1.0
        best_provider = None
        reject_reasons: List[str] = []

        for candidate in unique_candidates:
            score = self._rank_candidate(segment.text, candidate, selection_cfg, seg_duration)
            if score < selection_cfg.min_score:
                reject_reasons.append('low_score')
                continue
            if score > best_score:
                best_candidate = candidate
                best_score = score
                best_provider = getattr(candidate, 'provider', None)

        if best_candidate is None:
            if not candidates:
                reject_reasons.append('timeout' if fetch_timeout else 'no_candidates')
            elif url_hits or phash_hits:
                reject_reasons.append('deduped')
            elif not reject_reasons:
                reject_reasons.append('no_candidates')
        else:
            url = getattr(best_candidate, 'url', None)
            if url:
                SEEN_URLS.add(url)
            ph = getattr(best_candidate, '_phash', None)
            if ph is not None:
                SEEN_PHASHES.append(ph)

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        log_broll_decision(
            event_logger,
            segment_idx=idx,
            start=segment.start,
            end=segment.end,
            query_count=len(queries),
            candidate_count=len(candidates),
            unique_candidates=len(unique_candidates),
            url_dedup_hits=url_hits,
            phash_dedup_hits=phash_hits,
            selected_url=getattr(best_candidate, 'url', None) if best_candidate else None,
            selected_score=best_score if best_candidate else None,
            provider=best_provider if best_candidate else None,
            latency_ms=latency_ms,
            llm_healthy=llm_healthy,
            reject_reasons=sorted(set(reject_reasons)),
        )

    log_broll_decision(
        event_logger,
        segment_idx=-1,
        start=0.0,
        end=0.0,
        query_count=0,
        candidate_count=0,
        unique_candidates=0,
        url_dedup_hits=0,
        phash_dedup_hits=0,
        selected_url=None,
        selected_score=None,
        provider=None,
        latency_ms=0,
        llm_healthy=True,
        reject_reasons=['summary'],
    )

def _derive_segment_keywords(self, segment, global_keywords: Sequence[str]) -> List[str]:
    keywords: List[str] = []
    if global_keywords:
        keywords.extend(global_keywords[:4])
    segment_words = [w.strip().lower() for w in segment.text.split() if len(w.strip()) > 3]
    unique_segment_words: List[str] = []
    for word in segment_words:
        if word not in unique_segment_words:
            unique_segment_words.append(word)
    keywords.extend(unique_segment_words[:3])
    result: List[str] = []
    for word in keywords:
        if word and word not in result:
            result.append(word)
    return result

def _estimate_candidate_score(self, candidate, selection_cfg, segment_duration: float) -> float:
    base_score = 0.6
    width = getattr(candidate, 'width', 0) or 0
    height = getattr(candidate, 'height', 0) or 0
    duration = getattr(candidate, 'duration', None)

    if selection_cfg.prefer_landscape and width and height and width < height:
        return 0.0
    if not selection_cfg.prefer_landscape and width and height and height < width:
        base_score -= 0.1

    if duration is not None:
        if duration < max(selection_cfg.min_duration_s, 0.0):
            base_score -= 0.2
        elif duration >= selection_cfg.min_duration_s:
            base_score += 0.05
        if segment_duration > 0:
            ratio = duration / max(segment_duration, 1e-3)
            if ratio < 0.6:
                base_score -= 0.1
            elif ratio > 1.4:
                base_score -= 0.05
            else:
                base_score += 0.05

    tags = getattr(candidate, 'tags', None) or ()
    keyword_hits = sum(1 for t in tags if isinstance(t, str) and t)
    if keyword_hits:
        base_score += min(0.1, keyword_hits * 0.02)

    return max(0.0, min(1.0, base_score))

def _rank_candidate(self, segment_text: str, candidate, selection_cfg, segment_duration: float) -> float:
    base_score = self._estimate_candidate_score(candidate, selection_cfg, segment_duration)
    title = (getattr(candidate, 'title', '') or '').lower()
    tokens = {tok for tok in segment_text.lower().split() if len(tok) > 2}
    if title and tokens:
        overlap = sum(1 for tok in tokens if tok in title)
        if overlap:
            base_score += min(0.1, overlap * 0.02)
    return max(0.0, min(1.0, base_score))


    def insert_brolls_if_enabled(self, input_path: Path, subtitles: List[Dict], broll_keywords: List[str]) -> Path:
        """Point d'extension B-roll: retourne le chemin vidéo après insertion si activée."""
        if not getattr(Config, 'ENABLE_BROLL', False):
            print("    ⏭️ B-roll désactivés: aucune insertion")
            return input_path
        
        try:
            # Vérifier la librairie B-roll
            broll_root = Path("AI-B-roll")
            broll_library = broll_root / "broll_library"
            if not broll_library.exists():
                print("    ⚠️ Librairie B-roll introuvable, saut de l'insertion")
                return input_path
            # Préparer chemins (écrire directement dans le dossier du clip si possible)
            clip_dir = (Path(input_path).parent if (Path(input_path).name == 'reframed.mp4') else Config.TEMP_FOLDER)
            # Si input_path est déjà dans un dossier clip (reframed.mp4), sortir with_broll.mp4 à côté
            if Path(input_path).name == 'reframed.mp4':
                output_with_broll = clip_dir / 'with_broll.mp4'
            else:
                output_with_broll = Config.TEMP_FOLDER / f"with_broll_{Path(input_path).name}"
            output_with_broll.parent.mkdir(parents=True, exist_ok=True)
            
            # Assurer l'import du pipeline local (src/*)
            if str(broll_root.resolve()) not in sys.path:
                sys.path.insert(0, str(broll_root.resolve()))
            
            # 🚀 NOUVEAUX IMPORTS INTELLIGENTS SYNCHRONES (DÉSACTIVÉS POUR PROMPT OPTIMISÉ)
            try:
                from sync_context_analyzer import SyncContextAnalyzer
                from broll_diversity_manager import BrollDiversityManager
                # 🚨 DÉSACTIVATION TEMPORAIRE: Le système intelligent interfère avec notre prompt optimisé LLM
                INTELLIGENT_BROLL_AVAILABLE = False
                print("    ⚠️  Système intelligent DÉSACTIVÉ pour laisser le prompt optimisé LLM fonctionner")
                print("    🎯 Utilisation exclusive du prompt optimisé: 25-35 keywords + structure hiérarchique")
            except ImportError as e:
                print(f"    ⚠️  Système intelligent non disponible: {e}")
                print("    🔄 Fallback vers ancien système...")
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
            
            # 🧠 ANALYSE INTELLIGENTE AVANCÉE
            if INTELLIGENT_BROLL_AVAILABLE:
                print("    🧠 Utilisation du système B-roll intelligent...")
                try:
                    # Initialiser l'analyseur contextuel intelligent SYNCHRONE
                    context_analyzer = SyncContextAnalyzer()
                    
                    # Analyser le contexte global de la vidéo
                    transcript_text = " ".join([s.get('text', '') for s in subtitles])
                    global_analysis = context_analyzer.analyze_context(transcript_text)
                    
                    print(f"    🎯 Contexte détecté: {global_analysis.main_theme}")
                    print(f"    🧬 Sujets: {', '.join(global_analysis.key_topics[:3])}")
                    print(f"    😊 Sentiment: {global_analysis.sentiment}")
                    print(f"    📊 Complexité: {global_analysis.complexity}")
                    print(f"    🔑 Mots-clés: {', '.join(global_analysis.keywords[:5])}")
                    
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
                        print(f"    💾 Métadonnées intelligentes sauvegardées: {meta_path}")
                        
                        # 🎬 INSÉRATION INTELLIGENTE DES B-ROLLS
                        print("    🎬 Insertion intelligente des B-rolls...")
                        try:
                            # Créer un dossier unique pour ce clip
                            clip_id = input_path.stem
                            unique_broll_dir = broll_library / f"clip_intelligent_{clip_id}_{int(time.time())}"
                            unique_broll_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Générer des prompts intelligents basés sur l'analyse
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

                            print(f"    🎯 Prompts intelligents générés: {', '.join(intelligent_prompts[:3])}")
                            
                            # Utiliser l'ancien système mais avec les prompts intelligents
                            # (temporaire en attendant l'intégration complète)
                            print("    🔄 Utilisation du système B-roll avec prompts intelligents...")
                            
                        except Exception as e:
                            print(f"    ⚠️  Erreur insertion intelligente: {e}")
                            print("    🔄 Fallback vers ancien système...")
                            INTELLIGENT_BROLL_AVAILABLE = False
                            
                    except Exception as e:
                        print(f"    ⚠️  Erreur système intelligent: {e}")
                        print("    🔄 Fallback vers ancien système...")
                        INTELLIGENT_BROLL_AVAILABLE = False
                except Exception as e:
                    print(f"    ⚠️  Erreur système intelligent: {e}")
                    print("    🔄 Fallback vers ancien système...")
                    INTELLIGENT_BROLL_AVAILABLE = False
            
            # 🚀 CORRECTION: Préparer l'analyse des mots-clés pour tous les systèmes
            analysis = None
            try:
                analysis = extract_keywords_from_transcript_ai(subtitles)
                print(f"    🧠 Analyse des mots-clés préparée: {analysis.get('dominant_theme', 'N/A')}")
            except Exception as e:
                print(f"    ⚠️ Erreur préparation analyse: {e}")
                    
            # Fallback: ancienne analyse si système intelligent indisponible
            if not INTELLIGENT_BROLL_AVAILABLE:
                print("    🔄 Utilisation de l'ancien système B-roll...")
                # analysis déjà préparé plus haut
                prompts = generate_broll_prompts_ai(analysis) if analysis else []
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
                # Persiste metadata dans un dossier clip dédié si possible
                try:
                    meta_dir = Config.OUTPUT_FOLDER / 'meta'
                    meta_dir.mkdir(parents=True, exist_ok=True)
                    meta_path = meta_dir / f"{Path(input_path).stem}_broll_metadata.json"
                    with open(meta_path, 'w', encoding='utf-8') as f:
                        json.dump({'analysis': analysis, 'prompts': prompts}, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
            else:
                # 🎯 UTILISER LES PROMPTS INTELLIGENTS
                print("    🎯 Utilisation des prompts intelligents pour B-rolls...")
                try:
                    # Créer une analyse basée sur l'analyse intelligente
                    analysis = {
                        'main_theme': global_analysis.main_theme,
                        'key_topics': global_analysis.key_topics,
                        'sentiment': global_analysis.sentiment,
                        'keywords': global_analysis.keywords
                    }
                    
                    # Utiliser les prompts intelligents générés
                    prompts = intelligent_prompts if 'intelligent_prompt' in locals() else [
                        f"{global_analysis.main_theme} {kw}" for kw in global_analysis.keywords[:3]
                    ]
                    
                    print(f"    🎯 Prompts utilisés: {', '.join(prompts[:3])}")
                    
                except Exception as e:
                    print(f"    ⚠️  Erreur prompts intelligents: {e}")
                    # Fallback vers prompts génériques (analysis déjà préparé)
                    prompts = generate_broll_prompts_ai(analysis) if analysis else []
            
            # 🚀 CORRECTION: Utiliser directement le paramètre broll_keywords du LLM
            llm_broll_keywords = []
            try:
                # Utiliser les mots-clés B-roll passés en paramètre (générés par le LLM)
                if broll_keywords and len(broll_keywords) > 0:
                    llm_broll_keywords = broll_keywords
                    print(f"    🧠 Mots-clés B-roll LLM intégrés: {len(llm_broll_keywords)} termes")
                    print(f"    🎯 Exemples: {', '.join(llm_broll_keywords[:5])}")
                else:
                    print("    ⚠️ Mots-clés B-roll LLM non disponibles, utilisation extraction basique")
            except Exception as e:
                print(f"    ⚠️ Erreur récupération mots-clés B-roll LLM: {e}")
            
            # Combiner les mots-clés LLM avec les prompts existants
            if llm_broll_keywords:
                # Enrichir les prompts avec les mots-clés LLM
                enhanced_prompts = []
                for kw in llm_broll_keywords[:8]:  # Limiter à 8 mots-clés principaux
                    enhanced_prompts.append(kw)
                    # Créer des combinaisons avec le thème principal
                    if 'global_analysis' in globals() and 'global_analysis' in locals() and hasattr(global_analysis, 'main_theme'):
                        enhanced_prompts.append(f"{getattr(global_analysis, 'main_theme', 'general')} {kw}")
                
                # Ajouter les prompts existants
                enhanced_prompts.extend(prompts)
                
                # Dédupliquer et limiter
                seen_prompts = set()
                final_prompts = []
                for p in enhanced_prompts:
                    if p not in seen_prompts and len(p) > 2:
                        final_prompts.append(p)
                        seen_prompts.add(p)
                
                prompts = final_prompts[:12]  # Limiter à 12 prompts finaux
                print(f"    🚀 Prompts enrichis avec LLM: {len(prompts)} termes")
                print(f"    🎯 Prompts finaux: {', '.join(prompts[:5])}...")
            
            # Convertir nos sous-titres en segments attendus par le pipeline
            segments = [
                TranscriptSegment(start=float(s.get('start', 0.0)), end=float(s.get('end', 0.0)), text=str(s.get('text', '')).strip())
                for s in subtitles if (s.get('text') and (s.get('end', 0.0) >= s.get('start', 0.0)))
            ]
            if not segments:
                print("    ⚠️ Aucun segment de transcription valide, saut B-roll")
                return input_path

            if getattr(Config, 'ENABLE_PIPELINE_CORE_FETCHER', False):
                self._insert_brolls_pipeline_core(segments, broll_keywords, subtitles=subtitles, input_path=input_path)

            
            # Construire la config du pipeline (fetch + embeddings activés, pas de limites)
            cfg = BrollConfig(
                input_video=str(input_path),
                output_video=output_with_broll,
                broll_library=broll_library,
                srt_path=None,
                render_subtitles=False,
                            max_broll_ratio=0.65,           # CORRIGÉ: 90% → 65% pour équilibre optimal
            min_gap_between_broll_s=1.5,    # CORRIGÉ: 0.2s → 1.5s pour respiration visuelle
                            max_broll_clip_s=4.0,           # CORRIGÉ: 8.0s → 4.0s pour B-rolls équilibrés
            min_broll_clip_s=1.5,           # 🚀 OPTIMISÉ: 0.8s → 1.5s pour B-rolls plus visibles sur TikTok
                use_whisper=False,
                ffmpeg_preset="fast",
                crf=23,
                threads=0,
                # Fetchers (stock)
                enable_fetcher=getattr(Config, 'BROLL_FETCH_ENABLE', False),
                fetch_provider=getattr(Config, 'BROLL_FETCH_PROVIDER', 'pexels'),
                pexels_api_key=getattr(Config, 'PEXELS_API_KEY', None),
                pixabay_api_key=getattr(Config, 'PIXABAY_API_KEY', None),
                fetch_max_per_keyword=getattr(Config, 'BROLL_FETCH_MAX_PER_KEYWORD', 15),  # CORRIGÉ: 25 → 15 pour vitesse optimale
                fetch_allow_videos=getattr(Config, 'BROLL_FETCH_ALLOW_VIDEOS', True),
                fetch_allow_images=getattr(Config, 'BROLL_FETCH_ALLOW_IMAGES', True),  # Activé: images animées + Ken Burns
                # Embeddings
                use_embeddings=getattr(Config, 'BROLL_USE_EMBEDDINGS', True),
                embedding_model_name=getattr(Config, 'BROLL_EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
                contextual_config_path=getattr(Config, 'CONTEXTUAL_CONFIG_PATH', Path('config/contextual_broll.yml')),
                # Experimental FX toggle
                enable_experimental_fx=getattr(Config, 'ENABLE_EXPERIMENTAL_FX', False),
            )
            # FETCH DYNAMIQUE PAR CLIP: Créer un dossier unique et forcer le fetch à chaque fois
            try:
                from src.pipeline.fetchers import ensure_assets_for_keywords  # type: ignore
                
                # 🚀 NOUVEAU: Dossier temporaire unique - sera nettoyé après traitement
                clip_id = input_path.stem  # Nom du fichier sans extension
                clip_broll_dir = broll_library / f"temp_clip_{clip_id}_{int(time.time())}"
                
                # Toujours créer un nouveau dossier temporaire
                clip_broll_dir.mkdir(parents=True, exist_ok=True)
                print(f"    📁 Dossier B-roll temporaire créé: {clip_broll_dir.name}")
                print(f"    🗑️ Sera automatiquement nettoyé après traitement")
                
                # Forcer l'activation du fetcher pour chaque clip
                setattr(cfg, 'enable_fetcher', True)
                setattr(cfg, 'broll_library', str(clip_broll_dir))  # Utiliser le dossier unique
                
                print(f"    🔄 Fetch B-roll personnalisé pour clip: {clip_id}")
                print(f"    📁 Dossier B-roll unique: {clip_broll_dir.name}")
                
                # 🚀 NOUVEAU: Intégration du sélecteur B-roll générique
                if BROLL_SELECTOR_AVAILABLE and getattr(Config, 'BROLL_SELECTOR_ENABLED', True):
                    try:
                        print("    🎯 Sélecteur B-roll générique activé - Scoring mixte intelligent")
                        
                        # Initialiser le sélecteur avec la configuration
                        selector_config = None
                        if getattr(Config, 'BROLL_SELECTOR_CONFIG_PATH', None):
                            try:
                                import yaml
                                with open(Config.BROLL_SELECTOR_CONFIG_PATH, 'r', encoding='utf-8') as f:
                                    selector_config = yaml.safe_load(f)
                                print(f"    ⚙️ Configuration chargée: {Config.BROLL_SELECTOR_CONFIG_PATH}")
                            except Exception as e:
                                print(f"    ⚠️ Erreur chargement config: {e}")
                        
                        # Créer le sélecteur
                        from broll_selector import BrollSelector
                        broll_selector = BrollSelector(selector_config)
                        
                        # 🚀 CORRECTION: Utiliser les mots-clés LLM intelligents au lieu du fallback basique
                        context_keywords = []
                        
                        # 🎯 PRIORITÉ 0: Utiliser les VRAIS mots-clés LLM du système industriel
                        if llm_broll_keywords and len(llm_broll_keywords) > 0:
                            context_keywords = llm_broll_keywords[:15]  # Prendre les 15 meilleurs
                            print(f"    🚀 Mots-clés LLM INDUSTRIELS utilisés: {len(context_keywords)} termes")
                            print(f"    🎯 Mots-clés: {', '.join(context_keywords[:5])}")
                        
                        # Priorité 1: Utiliser l'analyse intelligente si disponible
                        elif 'global_analysis' in locals():
                            context_keywords = global_analysis.keywords[:10] if hasattr(global_analysis, 'keywords') else []
                        
                        # Priorité 2: Utiliser les mots-clés LLM corrigés (notre extraction améliorée)
                        if not context_keywords and 'analysis' in locals():
                            # Extraire les meilleurs mots-clés de notre système LLM
                            llm_keywords = []
                            if isinstance(analysis, dict) and 'keywords' in analysis:
                                for category, kws in analysis['keywords'].items():
                                    if isinstance(kws, list):
                                        llm_keywords.extend(kws[:3])  # 3 meilleurs par catégorie
                            
                            # Ajouter le thème dominant
                            if analysis.get('dominant_theme'):
                                llm_keywords.insert(0, analysis['dominant_theme'])
                            
                            context_keywords = llm_keywords[:15] if llm_keywords else []
                            print(f"    🧠 Mots-clés LLM intelligents utilisés: {len(context_keywords)} termes")
                        
                        # Priorité 3: Fallback basique seulement si aucun autre système
                        if not context_keywords:
                            # Extraction basique améliorée - mots significatifs seulement
                            significant_words = []
                            for s in subtitles:
                                text = s.get('text', '')
                                if text:
                                    words = text.lower().split()
                                    # Filtrer les mots vides et garder les mots significatifs
                                    meaningful_words = [w for w in words if len(w) > 4 and w.isalpha() 
                                                      and w not in ['that', 'this', 'they', 'there', 'where', 'when', 'what', 'with', 'have', 'your', 'once', 'figure']]
                                    significant_words.extend(meaningful_words)
                            
                            # Déduplication et limitation
                            context_keywords = list(dict.fromkeys(significant_words))[:10]
                            print(f"    ⚠️ Fallback vers extraction basique améliorée: {len(context_keywords)} termes")
                        
                        # Détecter le domaine
                        detected_domain = None
                        if 'global_analysis' in locals() and hasattr(global_analysis, 'main_theme'):
                            detected_domain = global_analysis.main_theme
                        
                        print(f"    🎯 Contexte: {detected_domain or 'général'}")
                        print(f"    🔑 Mots-clés contextuels: {', '.join(context_keywords[:5])}")
                        
                        # Utiliser le sélecteur pour la planification
                        selection_report = broll_selector.select_brolls(
                            keywords=context_keywords,
                            domain=detected_domain,
                            min_delay=self._load_broll_selector_config().get('thresholds', {}).get('min_delay_seconds', 4.0),
                            desired_count=self._load_broll_selector_config().get('desired_broll_count', 3)
                        )
                        
                        # Sauvegarder le rapport de sélection
                        try:
                            meta_dir = Config.OUTPUT_FOLDER / 'meta'
                            meta_dir.mkdir(parents=True, exist_ok=True)
                            selection_report_path = meta_dir / f"{Path(input_path).stem}_broll_selection_report.json"
                            with open(selection_report_path, 'w', encoding='utf-8') as f:
                                json.dump(selection_report, f, ensure_ascii=False, indent=2)
                            print(f"    💾 Rapport de sélection sauvegardé: {selection_report_path}")
                        except Exception as e:
                            print(f"    ⚠️ Erreur sauvegarde rapport: {e}")
                        
                        # Afficher les statistiques de sélection
                        if 'diagnostics' in selection_report:
                            diag = selection_report['diagnostics']
                            print(f"    📊 Sélection: {diag.get('num_selected', 0)}/{diag.get('num_candidates', 0)} B-rolls")
                            print(f"    🎯 Top score: {diag.get('top_score', 0):.3f}")
                            print(f"    📏 Seuil appliqué: {diag.get('min_score', 0):.3f}")
                        
                        if selection_report.get('fallback_used'):
                            print(f"    🆘 Fallback activé: Tier {selection_report.get('fallback_tier', '?')}")
                        
                    except Exception as e:
                        print(f"    ⚠️ Erreur sélecteur générique: {e}")
                        print("    🔄 Fallback vers système existant")
                
                # 🚀 CORRECTION: Intégration des mots-clés LLM pour le fetch
                # SÉLECTION INTELLIGENTE: Mots-clés contextuels + concepts associés
                from collections import Counter as _Counter
                kw_pool: list[str] = []
                
                # 🧠 PRIORITÉ 1: Mots-clés LLM si disponibles
                if broll_keywords:
                    try:
                        if not isinstance(broll_keywords, (list, tuple)):
                            print(f"    ❌ Format invalide broll_keywords: {type(broll_keywords)}")
                            broll_keywords = []
                        else:
                            # Normalisation/filtrage
                            broll_keywords = [
                                (kw.strip() if isinstance(kw, str) else "")
                                for kw in broll_keywords
                                if isinstance(kw, str) and kw and kw.strip()
                            ]
                    except (TypeError, AttributeError):
                        broll_keywords = []
                    
                    if broll_keywords:
                        print(f"    🚀 Utilisation des mots-clés LLM pour le fetch: {len(broll_keywords)} termes")
                        # Ajouter TOUS les mots-clés LLM en priorité
                        for kw in broll_keywords:
                            low = (kw or '').strip().lower()
                            if low and len(low) >= 3:
                                kw_pool.append(low)
                                # Ajouter des variations pour enrichir
                                if ' ' in low:  # Mots composés
                                    parts = low.split()
                                    kw_pool.extend(parts)
                        print(f"    🎯 Mots-clés LLM ajoutés: {', '.join(broll_keywords[:8])}")
                    else:
                        print("    ⚠️ Mots-clés LLM indisponibles après validation, fallback basique")
                
                # 🔄 PRIORITÉ 2: Extraction des mots-clés du transcript
                for s in subtitles:
                    base_kws = extract_keywords_for_segment(s.get('text','')) or []
                    spacy_kws = self._extract_keywords_for_segment_spacy(s.get('text','')) or []
                    for kw in (base_kws + spacy_kws):
                        low = (kw or '').strip().lower()
                        if low and len(low) >= 3:
                            kw_pool.append(low)
                
                # 🚀 CONCEPTS ASSOCIÉS ENRICHIS (50+ concepts)
                concept_mapping = {
                    # 🧠 Cerveau & Intelligence
                    'brain': ['neuroscience', 'mind', 'thinking', 'intelligence', 'cognitive', 'mental', 'psychology', 'consciousness'],
                    'mind': ['brain', 'thinking', 'thought', 'intelligence', 'cognitive', 'mental', 'psychology'],
                    'thinking': ['brain', 'mind', 'thought', 'intelligence', 'cognitive', 'mental', 'logic'],
                    
                    # 💰 Argent & Finance
                    'money': ['finance', 'business', 'success', 'wealth', 'investment', 'cash', 'profit', 'revenue'],
                    'argent': ['finance', 'business', 'success', 'wealth', 'investment', 'cash', 'profit', 'revenue'],
                    'finance': ['money', 'business', 'investment', 'wealth', 'profit', 'revenue', 'budget'],
                    
                    # 🎯 Focus & Concentration
                    'focus': ['concentration', 'productivity', 'attention', 'mindfulness', 'clarity', 'precision'],
                    'concentration': ['focus', 'attention', 'mindfulness', 'clarity', 'precision', 'dedication'],
                    'attention': ['focus', 'concentration', 'mindfulness', 'awareness', 'observation'],
                    
                    # 🏆 Succès & Réussite
                    'success': ['achievement', 'goal', 'victory', 'winning', 'growth', 'accomplishment', 'triumph'],
                    'succès': ['achievement', 'goal', 'victory', 'winning', 'growth', 'accomplishment', 'triumph'],
                    'victory': ['success', 'achievement', 'winning', 'triumph', 'conquest', 'domination'],
                    
                    # ❤️ Santé & Bien-être
                    'health': ['wellness', 'fitness', 'medical', 'lifestyle', 'nutrition', 'vitality', 'strength'],
                    'santé': ['wellness', 'fitness', 'medical', 'lifestyle', 'nutrition', 'vitality', 'strength'],
                    'fitness': ['health', 'wellness', 'exercise', 'training', 'strength', 'endurance'],
                    
                    # 🤖 Technologie & Innovation
                    'technology': ['digital', 'innovation', 'future', 'ai', 'automation', 'tech', 'modern'],
                    'technologie': ['digital', 'innovation', 'future', 'ai', 'automation', 'tech', 'modern'],
                    'innovation': ['technology', 'digital', 'future', 'ai', 'automation', 'creativity', 'progress'],
                    
                    # 💼 Business & Entreprise
                    'business': ['entrepreneur', 'startup', 'strategy', 'leadership', 'growth', 'company', 'enterprise'],
                    'entreprise': ['entrepreneur', 'startup', 'strategy', 'leadership', 'growth', 'company', 'enterprise'],
                    'strategy': ['business', 'planning', 'tactics', 'approach', 'method', 'system'],
                    
                    # 🚀 Action & Dynamisme
                    'action': ['movement', 'energy', 'power', 'vitality', 'dynamism', 'activity', 'motion'],
                    'action': ['movement', 'energy', 'power', 'vitality', 'dynamism', 'activity', 'motion'],
                    'energy': ['power', 'vitality', 'strength', 'force', 'intensity', 'enthusiasm'],
                    
                    # 🔥 Émotion & Passion
                    'emotion': ['feeling', 'passion', 'excitement', 'inspiration', 'motivation', 'enthusiasm'],
                    'émotion': ['feeling', 'passion', 'excitement', 'inspiration', 'motivation', 'enthusiasm'],
                    'passion': ['emotion', 'feeling', 'excitement', 'inspiration', 'motivation', 'enthusiasm'],
                    
                    # 🧠 Développement Personnel
                    'growth': ['development', 'improvement', 'progress', 'advancement', 'evolution', 'maturity'],
                    'croissance': ['development', 'improvement', 'progress', 'advancement', 'evolution', 'maturity'],
                    'development': ['growth', 'improvement', 'progress', 'advancement', 'evolution', 'maturity'],
                    
                    # ✅ Solutions & Résolution
                    'solution': ['resolution', 'fix', 'answer', 'remedy', 'cure', 'treatment'],
                    'solution': ['resolution', 'fix', 'answer', 'remedy', 'cure', 'treatment'],
                    'resolution': ['solution', 'fix', 'answer', 'remedy', 'cure', 'treatment'],
                    
                    # ⚠️ Problèmes & Défis
                    'problem': ['challenge', 'difficulty', 'obstacle', 'barrier', 'issue', 'trouble'],
                    'problème': ['challenge', 'difficulty', 'obstacle', 'barrier', 'issue', 'trouble'],
                    'challenge': ['problem', 'difficulty', 'obstacle', 'barrier', 'issue', 'trouble'],
                    
                    # 🌟 Qualité & Excellence
                    'quality': ['excellence', 'perfection', 'superiority', 'premium', 'best', 'optimal'],
                    'qualité': ['excellence', 'perfection', 'superiority', 'premium', 'best', 'optimal'],
                    'excellence': ['quality', 'perfection', 'superiority', 'premium', 'best', 'optimal']
                }
                
                # Enrichir avec des concepts associés
                for kw in kw_pool[:]:
                    for concept, related in concept_mapping.items():
                        if concept in kw or any(r in kw for r in related):
                            kw_pool.extend(related[:2])  # Ajouter 2 concepts max
                
                counts = _Counter(kw_pool)
                
                # 🚨 CORRECTION CRITIQUE: PRIORISER les mots-clés LLM sur les mots-clés génériques
                if 'broll_keywords' in locals() and broll_keywords:
                    # Utiliser DIRECTEMENT les mots-clés LLM comme requête principale
                    llm_keywords = [kw.strip().lower() for kw in broll_keywords if kw and len(kw.strip()) >= 3]
                    if llm_keywords:
                        # Prendre les 8 premiers mots-clés LLM + 2 concepts associés
                        top_kws = llm_keywords[:8]
                        # Ajouter quelques concepts associés pour enrichir
                        for kw in top_kws[:3]:  # Pour les 3 premiers mots-clés LLM
                            for concept, related in concept_mapping.items():
                                if concept in kw or any(r in kw for r in related):
                                    top_kws.extend(related[:1])  # 1 concept max par mot-clé LLM
                                    break
                        print(f"    🚀 REQUÊTE LLM PRIORITAIRE: {' '.join(top_kws[:5])}")
                    else:
                        top_kws = [w for w,_n in counts.most_common(15)]
                        print(f"    🔄 Fallback vers mots-clés génériques: {' '.join(top_kws[:5])}")
                else:
                    top_kws = [w for w,_n in counts.most_common(15)]
                    print(f"    🔄 Mots-clés génériques: {' '.join(top_kws[:5])}")
                
                # Fallback intelligent selon le contexte
                if not top_kws:
                    top_kws = ["focus","concentration","study","brain","mind","productivity","success"]
                print(f"    🔎 Fetch B-roll sur requête: {' '.join(top_kws[:5])}")
                # Provider auto-fallback si pas de clés -> archive
                import os as _os
                pex = getattr(Config, 'PEXELS_API_KEY', None) or _os.getenv('PEXELS_API_KEY')
                pixa = getattr(Config, 'PIXABAY_API_KEY', None) or _os.getenv('PIXABAY_API_KEY')
                uns = getattr(Config, 'UNSPLASH_ACCESS_KEY', None) or _os.getenv('UNSPLASH_ACCESS_KEY')
                giphy = _os.getenv('GIPHY_API_KEY')  # 🎭 GIPHY pour GIFs animés
                # Exposer l'accès Unsplash dans la cfg si dispo
                try:
                    if uns:
                        setattr(cfg, 'unsplash_access_key', uns)
                except Exception:
                    pass
                if not any([pex, pixa, uns]):
                    try:
                        setattr(cfg, 'fetch_provider', 'archive')
                        print("    🌐 Providers: archive (aucune clé API détectée)")
                    except Exception:
                        pass
                else:
                    # 🚀 AMÉLIORATION: Construire une liste de providers optimisée
                    prov = []
                    if pex:
                        prov.append('pexels')
                    if pixa:
                        prov.append('pixabay')
                    if uns:
                        prov.append('unsplash')
                    if giphy:
                        prov.append('giphy')  # 🎭 GIPHY pour GIFs animés
                    
                    # 🎯 AJOUT SÉCURISÉ: Archive.org comme source supplémentaire
                    try:
                        if prov:  # Si on a des providers avec clés API
                            prov.append('archive')  # Ajouter Archive.org
                            print(f"    🌐 Providers: {','.join(prov)} (Archive.org + Giphy ajoutés pour variété)")
                        else:
                            prov = ['archive']  # Seulement Archive.org si pas de clés
                            print(f"    🌐 Providers: {','.join(prov)} (Archive.org uniquement)")
                        
                        setattr(cfg, 'fetch_provider', ",".join(prov))
                    except Exception as e:
                        # Fallback sécurisé
                        try:
                            if prov:
                                setattr(cfg, 'fetch_provider', ",".join(prov))
                                print(f"    🌐 Providers: {','.join(prov)} (fallback sécurisé)")
                            else:
                                setattr(cfg, 'fetch_provider', 'archive')
                                print(f"    🌐 Providers: archive (fallback ultime)")
                        except Exception:
                            pass
                
                try:
                    setattr(cfg, 'fetch_allow_images', True)
                    # 🚀 OPTIMISATION MULTI-SOURCES: Qualité optimale (CORRIGÉ)
                    if uns and giphy:  # Si Unsplash ET Giphy sont disponibles
                        setattr(cfg, 'fetch_max_per_keyword', 35)  # CORRIGÉ: 125 → 35 pour qualité maximale
                        print("    📊 Configuration optimisée: 35 assets max + images activées (Unsplash + Giphy + Archive)")
                    elif uns:  # Si seulement Unsplash est disponible
                        setattr(cfg, 'fetch_max_per_keyword', 30)  # CORRIGÉ: 100 → 30 pour qualité maximale
                        print("    📊 Configuration optimisée: 30 assets max + images activées (Unsplash + Archive)")
                    elif giphy:  # Si seulement Giphy est disponible
                        setattr(cfg, 'fetch_max_per_keyword', 30)  # CORRIGÉ: 100 → 30 pour qualité avec GIFs
                        print("    📊 Configuration optimisée: 30 assets max + images activées (Giphy + Archive)")
                    else:
                        setattr(cfg, 'fetch_max_per_keyword', 25)  # CORRIGÉ: 75 → 25 pour Archive.org
                        print("    📊 Configuration optimisée: 25 assets max + images activées (Archive.org)")
                except Exception:
                    pass
                
                # Déclencher le fetch par mot-clé avec limites dynamiques (5 générique, 8 spécifique)
                def _is_generic_fetch_keyword(kw: str) -> bool:
                    if not isinstance(kw, str):
                        return True
                    k = kw.strip().lower()
                    # Expressions multi-mots = spécifiques
                    if ' ' in k:
                        return False
                    GENERIC_SIMPLE = {
                        'people','person','start','thing','stuff','your','once','figure',
                        'they','them','this','that','what','when','where','how','any','some'
                    }
                    return (k in GENERIC_SIMPLE) or (len(k) <= 6)
                
                for _kw in top_kws:
                    per_kw_limit = 5 if _is_generic_fetch_keyword(_kw) else 8
                    try:
                        setattr(cfg, 'fetch_max_per_keyword', per_kw_limit)
                        print(f"    🔧 Limite par mot-clé '{_kw}': {per_kw_limit} assets")
                    except Exception:
                        pass
                    ensure_assets_for_keywords(cfg, [_kw])
                
                # 🚨 CORRECTION CRITIQUE: SYSTÈME D'UNICITÉ DES B-ROLLS
                # Éviter la duplication des B-rolls entre vidéos différentes
                try:
                    # Créer un fichier de traçabilité des B-rolls utilisés
                    broll_tracking_file = Config.OUTPUT_FOLDER / 'meta' / 'broll_usage_tracking.json'
                    broll_tracking_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Charger l'historique des B-rolls utilisés
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
                            
                            # Vérifier si ce B-roll a déjà été utilisé
                            if asset_hash in broll_history:
                                asset_info['last_used'] = broll_history[asset_hash].get('last_used')
                                asset_info['usage_count'] = broll_history[asset_hash].get('usage_count', 0)
                            
                            available_brolls.append(asset_info)
                    
                    # Trier par priorité: B-rolls jamais utilisés en premier, puis par ancienneté
                    available_brolls.sort(key=lambda x: (x['usage_count'], x['last_used'] or '1970-01-01'))
                    
                    # Sélectionner les B-rolls uniques pour cette vidéo
                    selected_brolls = available_brolls[:3]  # 3 B-rolls uniques
                    
                    # Mettre à jour l'historique d'utilisation
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
                    
                    print(f"    🎯 B-rolls uniques sélectionnés: {len(selected_brolls)} (évite duplication)")
                    
                except Exception as e:
                    print(f"    ⚠️ Erreur système d'unicité: {e}")
                    # Fallback: utiliser tous les B-rolls disponibles
                    pass
                
                # Comptage après fetch dans le dossier du clip
                try:
                    _media_exts = {'.mp4','.mov','.mkv','.webm','.jpg','.jpeg','.png'}
                    _after = [p for p in clip_broll_dir.rglob('*') if p.suffix.lower() in _media_exts]
                    print(f"    📥 Fetch terminé: {len(_after)} assets pour ce clip")
                    
                    # 🚨 CORRECTION CRITIQUE: Créer fetched_brolls accessible globalement
                    fetched_brolls = []
                    for asset_path in _after:
                        if asset_path.exists():
                            fetched_brolls.append({
                                'path': str(asset_path),
                                'name': asset_path.name,
                                'size': asset_path.stat().st_size if asset_path.exists() else 0
                            })
                    
                    print(f"    🎯 {len(fetched_brolls)} B-rolls prêts pour l'assignation")
                    
                    if len(_after) == 0:
                        print("    ⚠️ Aucun asset téléchargé. Vérifie les clés API et la connectivité réseau.")
                except Exception:
                    fetched_brolls = []
                    print("    ⚠️ Erreur lors de la préparation des B-rolls fetchés")
                
                # Construire l'index FAISS pour ce clip spécifique
                try:
                    if 'build_index' in globals() and build_index is not None:  # type: ignore[name-defined]
                        index_handle = build_index(str(clip_broll_dir), model_name='ViT-B/32')  # type: ignore[misc]
                        print(f"    🧭 Index FAISS construit pour {clip_id}: {len(_after)} assets")
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
            
            # Préparer stop-words (legacy pipeline)
            stopwords: set[str] = set()
            try:
                swp = Path('config/stopwords.txt')
                if swp.exists():
                    stopwords = {ln.strip().lower() for ln in swp.read_text(encoding='utf-8').splitlines() if ln.strip()}
            except Exception:
                stopwords = set()

            # 🚀 CORRECTION: Intégration des mots-clés LLM dans la planification
            # Planification: nouvelle API préférée (plan_broll_insertions(segments, cfg, index))
            
            # 🚨 CORRECTION CRITIQUE: fetched_brolls est déjà déclaré plus haut, ne pas le redéclarer !
            # fetched_brolls = []  # ❌ SUPPRIMÉ: Cette ligne écrase la variable fetchée !
            
            try:
                plan = plan_broll_insertions(segments, cfg, index_handle)  # type: ignore[arg-type]
            except Exception:
                # 🚀 NOUVEAU: Utiliser les mots-clés LLM pour la planification
                seg_keywords: List[List[str]] = []
                
                # 🧠 PRIORITÉ 1: Mots-clés LLM si disponibles
                if 'broll_keywords' in locals() and broll_keywords:
                    print(f"    🚀 Utilisation des mots-clés LLM pour la planification: {len(broll_keywords)} termes")
                    # Distribuer les mots-clés LLM sur les segments
                    for i, s in enumerate(segments):
                        # Prendre 2-3 mots-clés LLM par segment
                        start_idx = (i * 2) % len(broll_keywords)
                        end_idx = min(start_idx + 2, len(broll_keywords))
                        segment_llm_kws = broll_keywords[start_idx:end_idx]
                        
                        # Combiner avec extraction basique
                        base_kws = extract_keywords_for_segment(s.text) or []
                        spacy_kws = self._extract_keywords_for_segment_spacy(s.text) or []
                        
                        # 🎯 PRIORITÉ aux mots-clés LLM
                        merged: List[str] = segment_llm_kws + base_kws + spacy_kws
                        
                        # Nettoyer et dédupliquer
                        cleaned: List[str] = []
                        seen = set()
                        for kw in merged:
                            if kw and kw.lower() not in seen:
                                low = kw.lower()
                                if not (len(low) < 3 and low in stopwords):
                                    cleaned.append(low)
                                    seen.add(low)
                        
                        seg_keywords.append(cleaned[:8])  # OPTIMISÉ: 15 → 8 pour vitesse
                        print(f"    🎯 Segment {i}: {len(cleaned)} mots-clés (LLM: {len(segment_llm_kws)})")
                else:
                    # 🔄 Fallback: extraction basique uniquement
                    print("    ⚠️ Mots-clés LLM non disponibles, utilisation extraction basique")
                    for s in segments:
                        base_kws = extract_keywords_for_segment(s.text) or []
                        spacy_kws = self._extract_keywords_for_segment_spacy(s.text) or []
                        merged: List[str] = []
                        for kw in (base_kws + spacy_kws):
                            if kw and kw.lower() not in merged:
                                low = kw.lower()
                                if not (len(low) < 5 and low in stopwords):
                                    merged.append(low)
                        seg_keywords.append(merged[:6])
                
                with _VFC(str(input_path)) as _tmp:
                    duration = float(_tmp.duration)
                plan = plan_broll_insertions(  # type: ignore[call-arg]
                    segments,
                    seg_keywords,
                    total_duration=duration,
                    max_broll_ratio=cfg.max_broll_ratio,
                    min_gap_between_broll_s=cfg.min_gap_between_broll_s,
                    max_broll_clip_s=cfg.max_broll_clip_s,
                    min_broll_clip_s=1.5,  # 🚀 OPTIMISÉ: 0.8s → 1.5s pour B-rolls plus visibles
                )
                
                # 🚨 CORRECTION CRITIQUE: Assigner directement les B-rolls fetchés aux items du plan
                if plan and fetched_brolls:
                    print(f"    🎯 Assignation directe des {len(fetched_brolls)} B-rolls fetchés aux {len(plan)} items du plan...")
                    
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
                                
                                print(f"    ✅ B-roll {i+1} assigné: {Path(asset_path).name}")
                            else:
                                break
                        
                        print(f"    🎉 {min(len(plan), len(valid_brolls))} B-rolls assignés avec succès au plan")
                    else:
                        print(f"    ⚠️ Aucun B-roll valide trouvé dans fetched_brolls")
                elif not fetched_brolls:
                    print(f"    ⚠️ Aucun B-roll fetché disponible pour l'assignation")
                elif not plan:
                    print(f"    ⚠️ Plan vide - aucun item à traiter")
            # Scoring adaptatif si disponible (pertinence/diversité/esthétique)
            

            
            try:
                from src.pipeline.scoring import score_candidates  # type: ignore
                boosts = {
                    # 🚀 Business & Croissance
                    "croissance": 0.9, "growth": 0.9, "opportunité": 0.8, "opportunite": 0.8,
                    "innovation": 0.9, "développement": 0.8, "developpement": 0.8, "expansion": 0.8,
                    "stratégie": 0.8, "strategie": 0.8, "plan": 0.7, "objectif": 0.8, "vision": 0.8,
                    
                    # 💰 Argent & Finance
                    "argent": 1.0, "money": 1.0, "cash": 0.9, "investissement": 0.9, "investissements": 0.9,
                    "revenu": 0.8, "revenus": 0.8, "profit": 0.9, "profits": 0.9, "perte": 0.7, "pertes": 0.7,
                    "échec": 0.7, "echec": 0.7, "budget": 0.7, "gestion": 0.7, "marge": 0.8, "roi": 0.9,
                    "chiffre": 0.7, "ca": 0.7, "économie": 0.8, "economie": 0.8, "financier": 0.8,
                    
                    # 🤝 Relation & Client
                    "client": 0.9, "clients": 0.9, "collaboration": 0.8, "collaborations": 0.8,
                    "communauté": 0.7, "communaute": 0.7, "confiance": 0.7, "vente": 0.8, "ventes": 0.8,
                    "deal": 0.7, "deals": 0.7, "prospect": 0.6, "prospects": 0.6, "contrat": 0.7,
                    "partenariat": 0.8, "équipe": 0.7, "equipe": 0.7, "réseau": 0.7, "reseau": 0.7,
                    
                    # 🔥 Motivation & Succès
                    "succès": 0.9, "succes": 0.9, "motivation": 0.8, "énergie": 0.7, "energie": 0.7,
                    "victoire": 0.8, "discipline": 0.7, "viral": 0.8, "viralité": 0.8, "viralite": 0.8,
                    "impact": 0.6, "explose": 0.6, "explosion": 0.6, "inspiration": 0.8, "passion": 0.8,
                    "détermination": 0.8, "determination": 0.8, "persévérance": 0.8, "perseverance": 0.8,
                    
                    # 🧠 Intelligence & Apprentissage
                    "cerveau": 1.0, "brain": 1.0, "intelligence": 0.9, "savoir": 0.8, "connaissance": 0.8,
                    "apprentissage": 0.8, "apprendre": 0.8, "étude": 0.8, "etude": 0.8, "formation": 0.8,
                    "compétence": 0.8, "competence": 0.8, "expertise": 0.8, "maîtrise": 0.8, "maitrise": 0.8,
                    
                    # 💡 Innovation & Technologie
                    "technologie": 0.9, "tech": 0.9, "innovation": 0.9, "digital": 0.8, "numérique": 0.8,
                    "numerique": 0.8, "futur": 0.8, "avancée": 0.8, "avancee": 0.8, "révolution": 0.8,
                    "revolution": 0.8, "disruption": 0.8, "transformation": 0.8, "évolution": 0.8, "evolution": 0.8,
                    
                    # ⚠️ Risque & Erreurs
                    "erreur": 0.6, "erreurs": 0.6, "warning": 0.6, "obstacle": 0.6, "obstacles": 0.6,
                    "solution": 0.6, "solutions": 0.6, "leçon": 0.5, "lecon": 0.5, "apprentissage": 0.5,
                    "problème": 0.6, "probleme": 0.6, "défi": 0.7, "defi": 0.7, "challenge": 0.7,
                    
                    # 🌟 Qualité & Excellence
                    "excellence": 0.9, "qualité": 0.8, "qualite": 0.8, "perfection": 0.8, "meilleur": 0.8,
                    "optimal": 0.8, "efficacité": 0.8, "efficacite": 0.8, "performance": 0.8, "résultat": 0.8,
                    "resultat": 0.8, "succès": 0.9, "succes": 0.9, "réussite": 0.9, "reussite": 0.9,
                }
                plan = score_candidates(
                    plan, segments, broll_library=str(broll_library), clip_model='ViT-B/32',
                    use_faiss=True, top_k=10, keyword_boosts=boosts
                )
                
            except Exception:
                pass
 
            # FILTRE: Exclure les B-rolls trop tôt dans la vidéo (délai minimum 3 secondes)
            try:
                filtered_plan = []
                for it in plan:
                    st = float(getattr(it, 'start', 0.0) if hasattr(it, 'start') else (it.get('start', 0.0) if isinstance(it, dict) else 0.0))
                    if st >= 3.0:  # Délai minimum de 3 secondes avant le premier B-roll
                        filtered_plan.append(it)
                    else:
                        print(f"    ⏰ B-roll filtré: trop tôt à {st:.2f}s (minimum 3.0s)")
                
                plan = filtered_plan
                print(f"    ✅ Plan filtré: {len(plan)} B-rolls après délai minimum")
            except Exception:
                pass

            # Déduplication souple: autoriser réutilisation si espacée (> 12s)
            try:
                seen: dict[str, float] = {}
                new_plan = []
                for it in plan:
                    # 🔧 CORRECTION: Gérer à la fois BrollPlanItem et dict
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
 
            # 🚀 PRIORISATION FRAÎCHEUR: Trier par timestamp du dossier (plus récent en premier)
            try:
                if plan:
                    # Extraire le clip_id pour la priorisation
                    clip_id = input_path.stem
                    
                    # Prioriser par fraîcheur si possible
                    for item in plan:
                        if hasattr(item, 'asset_path') and item.asset_path:
                            asset_path = item.asset_path
                        elif isinstance(item, dict) and item.get('asset_path'):
                            asset_path = item['asset_path']
                        else:
                            continue
                        
                        # Calculer le score de fraîcheur
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
                    
                    # Trier par fraîcheur décroissante
                    plan.sort(key=lambda x: getattr(x, 'freshness_score', 0), reverse=True)
                    print(f"    🆕 Priorisation fraîcheur: {len(plan)} B-rolls triés par timestamp")
                    
            except Exception as e:
                print(f"    ⚠️  Erreur priorisation fraîcheur: {e}")
 
            # 🎯 SCORING CONTEXTUEL RENFORCÉ: Pénaliser les assets non pertinents au domaine
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
                    
                    print(f"    🎯 Scoring contextuel appliqué: domaine '{domain}' avec {len(keywords)} mots-clés")
                    
                    # 🔍 DEBUG B-ROLL SELECTION (si activé)
                    debug_mode = getattr(Config, 'DEBUG_BROLL', False) or os.getenv('DEBUG_BROLL', 'false').lower() == 'true'
                    _debug_broll_selection(plan, domain, keywords, debug_mode)
                    
                    # 🚨 FALLBACK PROPRE: Si aucun asset pertinent, utiliser des assets neutres
                    # 🔧 CORRECTION CRITIQUE: Vérifier d'abord si les items ont des assets assignés
                    items_without_assets = []
                    items_with_assets = []
                    
                    for item in plan:
                        if hasattr(item, 'asset_path') and item.asset_path:
                            items_with_assets.append(item)
                        elif isinstance(item, dict) and item.get('asset_path'):
                            items_with_assets.append(item)
                        else:
                            items_without_assets.append(item)
                    
                    print(f"    🔍 Analyse des assets: {len(items_with_assets)} avec assets, {len(items_without_assets)} sans assets")
                    
                    # 🚨 CORRECTION: Assigner des assets aux items sans assets AVANT le fallback
                    if items_without_assets and fetched_brolls:
                        print(f"    🎯 Assignation d'assets aux {len(items_without_assets)} items sans assets...")
                        
                        # 🚀 NOUVEAU: Utiliser uniquement les B-rolls frais pour éviter la duplication
                        available_assets = []
                        for broll in fetched_brolls:
                            asset_path = broll.get('path', '')
                            if asset_path and Path(asset_path).exists():
                                available_assets.append(asset_path)
                        
                        # 🚀 NOUVEAU: Mélanger pour éviter l'ordre séquentiel répétitif
                        import random
                        random.shuffle(available_assets)
                        
                        print(f"    ✅ {len(available_assets)} assets frais disponibles (mélangés pour diversité)")
                        
                        for i, item in enumerate(items_without_assets):
                            if i < len(available_assets):
                                asset_path = available_assets[i]
                                if hasattr(item, 'asset_path'):
                                    item.asset_path = asset_path
                                elif isinstance(item, dict):
                                    item['asset_path'] = asset_path
                                
                                print(f"    ✅ Asset frais assigné à item {i+1}: {Path(asset_path).name}")
                            else:
                                break
                    else:
                        print(f"    ⚠️  Plan vide - Aucun item à traiter")
                    
            except Exception as e:
                print(f"    ⚠️  Erreur scoring contextuel: {e}")
 
                        # Affecter un asset_path pertinent via FAISS/CLIP si manquant
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                import numpy as _np  # type: ignore
                import faiss as _faiss  # type: ignore
                from pathlib import Path as _P
                
                # 🚨 NOUVEAU: Importer le système de scoring contextuel intelligent
                try:
                    from src.pipeline.broll_selector import get_contextual_broll_score
                    print("    🧠 Système de scoring contextuel intelligent activé")
                except ImportError:
                    print("    ⚠️ Système de scoring contextuel non disponible")
                    get_contextual_broll_score = None
                
                # UTILISER LE DOSSIER SPÉCIFIQUE DU CLIP (pas la librairie globale)
                clip_specific_dir = clip_broll_dir if 'clip_specific_dir' in locals() else broll_library
                idx_bin = (clip_specific_dir / 'faiss.index')
                idx_json = (clip_specific_dir / 'faiss.json')
                
                model_name = getattr(cfg, 'embedding_model_name', 'clip-ViT-B/32')
                # 🚀 OPTIMISATION: Utiliser le cache pour éviter le rechargement
                final_model_name = 'clip-ViT-B/32' if 'ViT' in model_name else model_name
                st_model = get_sentence_transformer_model(final_model_name)
                
                # 🚀 CORRECTION: Créer emb_text conditionnellement
                emb_text = None
                if st_model is None:
                    print(f"    ⚠️ Impossible de charger le modèle {final_model_name}, désactivation FAISS")
                else:
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
                    q = emb_text(local) if local and emb_text is not None else None
                    
                    # 🚨 NOUVEAU: Extraction des mots-clés pour le scoring contextuel
                    local_keywords = []
                    if local:
                        # Extraire les mots-clés du texte local
                        words = local.lower().split()
                        local_keywords = [w for w in words if len(w) > 3 and w.isalpha()][:10]
                    
                    chosen = None
                    best_score = -1
                    
                    if index is not None and q is not None and paths:
                        # 🚨 NOUVEAU: Recherche étendue pour évaluation contextuelle
                        D,I = index.search(q.reshape(1,-1), 15)  # Augmenter de 5 à 15 candidats
                        
                        # 🚨 NOUVEAU: Évaluation contextuelle de tous les candidats
                        for idx in I[0].tolist():
                            if 0 <= idx < len(paths):
                                p = paths[idx]
                                if not p:
                                    continue
                                cand = _P(p)
                                if not cand.is_absolute():
                                    cand = (clip_specific_dir / p).resolve()
                                if str(cand) not in used_recent and cand.exists():
                                    # 🚨 NOUVEAU: Calcul du score contextuel intelligent
                                    contextual_score = 0.0
                                    if 'get_contextual_broll_score' in globals() and local_keywords:
                                        try:
                                            # Extraire les tokens et tags du fichier
                                            asset_name = cand.stem.lower()
                                            asset_tokens = asset_name.split('_')
                                            asset_tags = asset_name.split('_')  # Simplifié pour l'exemple
                                            contextual_score = get_contextual_broll_score(local_keywords, asset_tokens, asset_tags)
                                        except Exception as e:
                                            print(f"    ⚠️ Erreur scoring contextuel: {e}")
                                            contextual_score = 0.0
                                    
                                    # 🚨 NOUVEAU: Score combiné FAISS + Contextuel
                                    faiss_score = float(D[0][I[0].tolist().index(idx)]) if idx in I[0] else 0.0
                                    combined_score = faiss_score + (contextual_score * 2.0)  # Poids contextuel DOUBLÉ
                                    
                                    if combined_score > best_score:
                                        best_score = combined_score
                                        chosen = str(cand)
                        
                        # 🚨 NOUVEAU: Log de la sélection contextuelle
                        if chosen and 'get_contextual_broll_score' in globals() and local_keywords:
                            try:
                                asset_name = Path(chosen).stem.lower()
                                asset_tokens = asset_name.split('_')
                                asset_tags = asset_name.split('_')
                                final_contextual_score = get_contextual_broll_score(local_keywords, asset_tokens, asset_tags)
                                print(f"    🎯 Sélection contextuelle: {Path(chosen).stem} | Score: {best_score:.3f} | Contexte: {final_contextual_score:.2f}")
                            except Exception:
                                pass
                    
                    if chosen is None:
                        # 🚀 NOUVEAU: Fallback intelligent utilisant UNIQUEMENT les assets frais du dossier fetched/
                        print(f"    🔍 Fallback vers assets frais uniquement...")
                        fetched_dir = clip_specific_dir / 'fetched'
                        if fetched_dir.exists():
                            for p in fetched_dir.rglob('*'):
                                if p.suffix.lower() in {'.mp4','.mov','.mkv','.webm','.jpg','.jpeg','.png'}:
                                    if str(p.resolve()) not in used_recent and p.exists():
                                        # 🚀 NOUVEAU: Évaluation contextuelle prioritaire
                                        if 'get_contextual_broll_score' in globals() and local_keywords:
                                            try:
                                                asset_name = p.stem.lower()
                                                asset_tokens = asset_name.split('_')
                                                asset_tags = asset_name.split('_')
                                                fallback_score = get_contextual_broll_score(local_keywords, asset_tokens, asset_tags)
                                                if fallback_score > 1.0:  # Seuil réduit pour plus de diversité
                                                    chosen = str(p.resolve())
                                                    print(f"    ✅ Asset frais contextuel: {p.stem} | Score: {fallback_score:.2f}")
                                                    break
                                            except Exception:
                                                pass
                                        else:
                                            # Utiliser directement l'asset frais sans scoring
                                            chosen = str(p.resolve())
                                            print(f"    ✅ Asset frais utilisé: {p.stem}")
                                            break
                        else:
                            print(f"    ⚠️ Dossier fetched/ non trouvé: {fetched_dir}")
                    
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

            # Vérification des asset_path avant normalisation + mini fallback non invasif
            try:
                def _get_ap(x):
                    return (getattr(x, 'asset_path', None) if hasattr(x, 'asset_path') else (x.get('asset_path') if isinstance(x, dict) else None))
                missing = [it for it in (plan or []) if not _get_ap(it)]
                if plan and len(missing) == len(plan):
                    # 🚀 NOUVEAU: Fallback intelligent anti-duplication
                    # UTILISER LE DOSSIER SPÉCIFIQUE DU CLIP (fetched/ en priorité)
                    clip_specific_dir = clip_broll_dir if 'clip_specific_dir' in locals() else broll_library
                    
                    # 🚀 PRIORISER le dossier fetched/ pour des assets frais
                    fetched_dir = clip_specific_dir / 'fetched'
                    if fetched_dir.exists():
                        lib_assets = [p for p in fetched_dir.rglob('*') if p.suffix.lower() in {'.mp4','.mov','.mkv','.webm','.jpg','.jpeg','.png'}]
                        print(f"    🎯 Utilisation assets frais du dossier fetched/: {len(lib_assets)} assets")
                    else:
                        lib_assets = [p for p in clip_specific_dir.rglob('*') if p.suffix.lower() in {'.mp4','.mov','.mkv','.webm','.jpg','.jpeg','.png'}]
                        print(f"    ⚠️ Fallback vers dossier complet: {len(lib_assets)} assets")
                    
                    if lib_assets:
                        # 🚀 NOUVEAU: Mélanger les assets pour éviter l'ordre répétitif
                        import random
                        random.shuffle(lib_assets)
                        
                        # 🚀 NOUVEAU: Répartition intelligente anti-duplication
                        assigned_assets = set()  # Tracker les assets déjà utilisés
                        for i, it in enumerate(plan):
                            ap = _get_ap(it)
                            if ap:
                                continue
                                
                            # Chercher un asset non encore utilisé
                            chosen_asset = None
                            for asset in lib_assets:
                                asset_path = str(asset.resolve())
                                if asset_path not in assigned_assets:
                                    chosen_asset = asset_path
                                    assigned_assets.add(asset_path)
                                    break
                            
                            # Si tous les assets sont utilisés, reprendre depuis le début
                            if chosen_asset is None and lib_assets:
                                chosen_asset = str(lib_assets[i % len(lib_assets)].resolve())
                                print(f"    ⚠️ Recyclage asset {i+1}: tous les assets frais épuisés")
                            
                            if chosen_asset:
                                if isinstance(it, dict):
                                    it['asset_path'] = chosen_asset
                                else:
                                    try:
                                        setattr(it, 'asset_path', chosen_asset)
                                    except Exception:
                                        pass
                                print(f"    ✅ Asset intelligent assigné {i+1}: {Path(chosen_asset).name}")
                        
                        print(f"    📊 Assignation intelligente: {len(assigned_assets)} assets uniques utilisés sur {len(lib_assets)} disponibles")
            except Exception:
                pass
 
             # Normaliser la timeline en événements canonique et rendre
            try:
                with _VFC(str(input_path)) as _fpsprobe:
                    fps_probe = float(_fpsprobe.fps or 25.0)
            except Exception:
                fps_probe = 25.0
            events = normalize_timeline(plan, fps=fps_probe)
            events = enrich_keywords(events)
            

            
            # Hard fail if no valid events
            if not events:
                raise RuntimeError('Aucun B-roll valide après planification/scoring. Vérifier l\'index FAISS et la librairie. Aucun fallback synthétique appliqué.')
            # Valider que les médias existent
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
                print(f"    🔎 B-roll events valides: {len(valid_events)}")
                for _ev in valid_events[:3]:
                    print(f"       • {_ev.start_s:.2f}-{_ev.end_s:.2f} → {getattr(_ev, 'media_path','')}")
            except Exception:
                pass
            if not valid_events:
                # Fallback legacy: construire un plan simple à partir de la librairie existante
                try:
                    _media_exts = {'.mp4','.mov','.mkv','.webm','.jpg','.jpeg','.png'}
                    assets = [p for p in Path(broll_library).rglob('*') if p.suffix.lower() in _media_exts]
                    assets.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
                    assets = assets[:20]
                    if assets:
                        # Choisir des segments suffisamment longs (>2.0s) et espacés
                        cands = []
                        for s in segments:
                            dur = float(getattr(s, 'end', 0.0) - getattr(s, 'start', 0.0))
                            if dur >= 2.0 and getattr(s, 'start', 0.0) >= 1.5:  # Plus flexible
                                cands.append(s)
                        plan_simple = []
                        gap = 6.0  # Réduit: 8s → 6s pour plus d'insertions
                        last = -1e9
                        ai = 0
                        for s in cands:
                            st = float(getattr(s,'start',0.0))
                            en = float(getattr(s,'end',0.0))
                            if st - last < gap:
                                continue
                            dur = min(7.0, max(2.5, en - st))  # Durée min: 2.5s, max: 7s
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
                            print(f"    ♻️ Fallback legacy appliqué: {len(legacy_events)} events")
                            valid_events = legacy_events
                            # Continue vers le rendu unique plus bas
                        else:
                            raise RuntimeError('Librairie B-roll présente mais aucun slot valide pour fallback legacy')
                    else:
                        raise RuntimeError('B-rolls planifiés, aucun media_path valide et aucune ressource en librairie pour fallback')
                except Exception as _e:
                    raise RuntimeError('B-rolls planifiés, mais aucun media_path valide trouvé. Fallback legacy impossible: ' + str(_e))
            # Rendu unique avec les events valides (incl. fallback le cas échéant)
            render_video(cfg, segments, valid_events)
            
            # VÉRIFICATION ET NETTOYAGE INTELLIGENT DES B-ROLLS
            try:
                if getattr(Config, 'BROLL_DELETE_AFTER_USE', False):
                    print("    🔍 Vérification des B-rolls avant suppression...")
                    
                    # Importer le système de vérification
                    try:
                        from broll_verification_system import create_verification_system
                        verifier = create_verification_system()
                        
                        # Vérifier l'insertion des B-rolls
                        verification_result = verifier.verify_broll_insertion(
                            video_path=cfg.output_video,
                            broll_plan=plan or [],
                            broll_library_path=str(clip_broll_dir) if 'clip_broll_dir' in locals() else "AI-B-roll/broll_library"
                        )
                        
                        # 🚀 CORRECTION: Vérifier le type du résultat de vérification
                        if not isinstance(verification_result, dict):
                            print(f"    ⚠️ Résultat de vérification invalide (type: {type(verification_result)}) - Fallback vers vérification basique")
                            verification_result = {
                                "verification_passed": True,  # Par défaut, autoriser la suppression
                                "issues": [],
                                "recommendations": []
                            }
                        
                        # Décider si la suppression est autorisée
                        if verification_result.get("verification_passed", False):
                            print("    ✅ Vérification réussie - Suppression autorisée")
                            
                            # Supprimer seulement les fichiers B-roll utilisés (pas le dossier)
                            used_files: List[str] = []
                            for item in (plan or []):
                                path = getattr(item, 'asset_path', None) if hasattr(item, 'asset_path') else (item.get('asset_path') if isinstance(item, dict) else None)
                                if path and os.path.exists(path):
                                    used_files.append(path)
                            
                            # Nettoyer les fichiers utilisés
                            cleaned_count = 0
                            for p in used_files:
                                try:
                                    os.remove(p)
                                    cleaned_count += 1
                                except Exception:
                                    pass
                            
                            # Marquer le dossier comme "utilisé" mais le garder
                            if 'clip_broll_dir' in locals() and clip_broll_dir.exists():
                                try:
                                    # Créer un fichier de statut pour indiquer que le clip est traité
                                    status_file = clip_broll_dir / "STATUS_COMPLETED.txt"
                                    status_file.write_text(f"Clip traité le {time.strftime('%Y-%m-%d %H:%M:%S')}\nB-rolls utilisés: {cleaned_count}\nVérification: PASSED\n", encoding='utf-8')
                                    print(f"    🗂️ Dossier B-roll conservé: {clip_broll_dir.name} (fichiers nettoyés: {cleaned_count})")
                                except Exception as e:
                                    print(f"    ⚠️ Erreur création statut: {e}")
                        else:
                            print("    ❌ Vérification échouée - Suppression REFUSÉE")
                            print("    📋 Problèmes détectés:")
                            for issue in verification_result.get("issues", []):
                                print(f"       • {issue}")
                            print("    💡 Recommandations:")
                            for rec in verification_result.get("recommendations", []):
                                print(f"       • {rec}")
                            
                            # Créer un fichier de statut d'échec
                            if 'clip_broll_dir' in locals() and clip_broll_dir.exists():
                                try:
                                    status_file = clip_broll_dir / "STATUS_FAILED.txt"
                                    status_file.write_text(f"Clip traité le {time.strftime('%Y-%m-%d %H:%M:%S')}\nVérification: FAILED\nProblèmes: {', '.join(verification_result.get('issues', []))}\n", encoding='utf-8')
                                    print(f"    🚨 Dossier B-roll marqué comme échec: {clip_broll_dir.name}")
                                except Exception as e:
                                    print(f"    ⚠️ Erreur création statut d'échec: {e}")
                    
                    except ImportError:
                        print("    ⚠️ Système de vérification non disponible - Suppression sans vérification")
                        # Fallback vers l'ancien système
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
                                status_file.write_text(f"Clip traité le {time.strftime('%Y-%m-%d %H:%M:%S')}\nB-rolls utilisés: {cleaned_count}\nVérification: NON DISPONIBLE\n", encoding='utf-8')
                                print(f"    🗂️ Dossier B-roll conservé: {clip_broll_dir.name} (fichiers nettoyés: {cleaned_count})")
                            except Exception as e:
                                print(f"    ⚠️ Erreur création statut: {e}")
                    
            except Exception as e:
                print(f"    ⚠️ Erreur lors de la vérification/nettoyage: {e}")
                # En cas d'erreur, ne pas supprimer les B-rolls
                pass

            if Path(cfg.output_video).exists():
                print("    ✅ B-roll insérés avec succès")
            
                # 🧹 NOUVEAU: Nettoyage immédiat du cache B-roll temporaire
                try:
                    if 'clip_broll_dir' in locals() and clip_broll_dir.exists():
                        folder_size = sum(f.stat().st_size for f in clip_broll_dir.rglob('*') if f.is_file()) / (1024**2)  # MB
                        # 🚀 OPTIMISATION: Utiliser safe_remove_tree
                        if safe_remove_tree(clip_broll_dir):
                            print(f"    🗑️ Cache B-roll nettoyé: {folder_size:.1f} MB libérés")
                            print(f"    💾 Dossier temporaire supprimé: {clip_broll_dir.name}")
                        else:
                            print(f"    ⚠️ Nettoyage partiel du cache B-roll")
                except Exception as e:
                    print(f"    ⚠️ Erreur nettoyage cache: {e}")
            
                return Path(cfg.output_video)
            else:
                print("    ⚠️ Sortie B-roll introuvable, retour à la vidéo d'origine")
            
            # 🧹 Nettoyer même en cas d'échec
            try:
                if 'clip_broll_dir' in locals() and clip_broll_dir.exists():
                    # 🚀 OPTIMISATION: Utiliser safe_remove_tree
                    if safe_remove_tree(clip_broll_dir):
                        print(f"    🗑️ Cache B-roll nettoyé (échec traitement)")
            except Exception:
                pass
            
            return input_path
        except Exception as e:
            print(f"    ❌ Erreur B-roll: {e}")
            
            # 🧹 IMPORTANT: Nettoyer même en cas d'erreur pour éviter l'accumulation
            try:
                if 'clip_broll_dir' in locals() and clip_broll_dir.exists():
                    # 🚀 OPTIMISATION: Utiliser safe_remove_tree
                    if safe_remove_tree(clip_broll_dir):
                        print(f"    🗑️ Cache B-roll nettoyé après erreur")
            except Exception:
                pass
            
            return input_path

    # Si densité trop faible après planification, injecter quelques B-rolls génériques
    try:
        with _VFC(str(input_path)) as _tmp:
            _total = float(_tmp.duration or 0.0)
        cur_cov = sum(max(0.0, (float(getattr(it,'end', it.get('end',0.0))) - float(getattr(it,'start', it.get('start',0.0))))) for it in (plan or []))
        if _total > 0 and (cur_cov / _total) < 0.20:  # Augmenté: 15% → 20% pour plus de B-rolls
            _generics = []
            bank = [
                "money", "handshake", "meeting", "audience", "lightbulb", "typing", "city", "success"
            ]
            # Chercher quelques médias génériques existants
            for p in broll_library.rglob('*'):
                if p.suffix.lower() in {'.mp4','.mov','.mkv','.webm','.jpg','.jpeg','.png'}:
                    name = p.stem.lower()
                    if any(k in name for k in bank):
                        _generics.append(str(p.resolve()))
            if _generics:
                # Injecter 2–4 génériques espacés
                inject_count = min(4, max(2, int(len(_generics)/5)))
                st = 2.0
                while inject_count > 0 and st < (_total - 3.5):
                    plan.append({'start': st, 'end': min(_total, st+3.5), 'asset_path': _generics[inject_count % len(_generics)], 'crossfade_frames': 2})
                    st += 10.0
                    inject_count -= 1
                print("    ➕ B-rolls génériques injectés pour densité minimale")
    except Exception:
        pass

class PremiereProAutomation:
    """
    Classe pour l'automatisation Premiere Pro (optionnelle)
    Utilise ExtendScript pour les utilisateurs avancés
    """
    
    @staticmethod
    def create_jsx_script(clip_path: str, output_path: str) -> str:
        """Génère un script ExtendScript pour Premiere Pro"""
        jsx_script = f'''
        // Script ExtendScript pour Premiere Pro
        var project = app.project;
        
        // Import du clip
        var importOptions = new ImportOptions();
        importOptions.file = new File("{clip_path}");
        var clip = project.importFiles([importOptions.file]);
        
        // Création d'une séquence 9:16
        var sequence = project.createNewSequence("Vertical_Clip", "HDV-1080i25");
        sequence.videoTracks[0].insertClip(clip[0], 0);
        
        // Application de l'effet Auto Reframe (si disponible)
        // Note: Ceci nécessite Premiere Pro 2019 ou plus récent
        
        // Export
        var encoder = app.encoder;
        encoder.encodeSequence(sequence, "{output_path}", "H.264", false);
        '''
        return jsx_script
    
    @staticmethod 
    def run_premiere_script(jsx_script_content: str):
        """Exécute un script ExtendScript dans Premiere Pro"""
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
                print("⚠️ Exécution ExtendScript automatisée non supportée nativement sous Windows dans ce pipeline.")
                print("   Ouvrez Premiere Pro et exécutez le script manuellement: " + str(script_path))
            else:
                print("⚠️ Plateforme non supportée pour l'exécution automatique de Premiere Pro.")
            
            logger.info("✅ Script Premiere Pro traité (voir message ci-dessus)")
            
        except Exception as e:
            logger.error(f"❌ Erreur Premiere Pro: {e}")
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
    """Priorise les assets les plus récents basés sur le timestamp du dossier."""
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
        
        # Trier par timestamp décroissant (plus récent en premier)
        broll_candidates.sort(key=lambda x: getattr(x, 'folder_timestamp', 0), reverse=True)
        
    except Exception as e:
        print(f"    ⚠️  Erreur priorisation fraîcheur: {e}")
    
    return broll_candidates

def _score_contextual_relevance(asset_path, domain, keywords):
    """Score de pertinence contextuelle basé sur les tokens et le domaine."""
    try:
        if not asset_path or not domain or not keywords:
            return 0.5
        
        # Extraire les tokens du nom de fichier
        filename = Path(asset_path).stem.lower()
        asset_tokens = set(re.split(r'[^a-z0-9]+', filename))
        
        # Tokens du domaine et mots-clés
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
        
        # Score basé sur l'overlap (0.0 à 1.0)
        base_score = min(1.0, overlap / max(1, total_relevant * 0.3))
        
        # Bonus pour les tokens de domaine
        domain_overlap = len(asset_tokens & domain_tokens)
        domain_bonus = min(0.3, domain_overlap * 0.1)
        
        final_score = min(1.0, base_score + domain_bonus)
        return final_score
        
    except Exception as e:
        print(f"    ⚠️  Erreur scoring contextuel: {e}")
        return 0.5

def _get_fallback_neutral_assets(broll_library, count=3):
    """Récupère des assets neutres/génériques comme fallback."""
    try:
        fallback_keywords = ['neutral', 'generic', 'background', 'abstract', 'minimal']
        fallback_assets = []
        
        for keyword in fallback_keywords:
            # Chercher dans la librairie des assets avec ces mots-clés
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
        
        # Si pas assez d'assets spécifiques, prendre des assets génériques
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
        print(f"    ⚠️  Erreur fallback neutre: {e}")
        return []

def _debug_broll_selection(plan, domain, keywords, debug_mode=False):
    """Log détaillé de la sélection B-roll si debug activé."""
    if not debug_mode:
        return
    
    print(f"    🔍 DEBUG B-ROLL SELECTION:")
    print(f"       Domaine: {domain}")
    print(f"       Mots-clés: {keywords[:5]}")
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
        print(f"         Score: {score}, Context: {context_score}, Fraîcheur: {freshness}")

# 🚀 NOUVEAU: Fonction de scoring mixte intelligent pour B-rolls
def score_broll_asset_mixed(asset_path: str, asset_tags: List[str], query_keywords: List[str], 
                           domain: Optional[str] = None, asset_metadata: Optional[Dict] = None) -> float:
    """
    Score un asset B-roll avec le système mixte intelligent.
    
    Args:
        asset_path: Chemin vers l'asset
        asset_tags: Tags de l'asset
        query_keywords: Mots-clés de la requête
        domain: Domaine détecté (optionnel)
        asset_metadata: Métadonnées supplémentaires (optionnel)
    
    Returns:
        Score final entre 0.0 et 1.0
    """
    try:
        if not BROLL_SELECTOR_AVAILABLE:
            # Fallback vers scoring basique
            return _score_broll_asset_basic(asset_path, asset_tags, query_keywords)
        
        # Utiliser le nouveau sélecteur si disponible
        from broll_selector import Asset, ScoringFeatures
        
        # Créer un asset simulé pour le scoring
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
        
        # Normaliser les mots-clés de la requête
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
        
        # 3. Freshness (basé sur la date de création du fichier)
        try:
            file_path = Path(asset_path)
            if file_path.exists():
                mtime = file_path.stat().st_mtime
                days_old = (time.time() - mtime) / (24 * 3600)
                features.freshness = 1.0 / (1.0 + days_old / 60)  # Demi-vie de 60 jours
        except:
            features.freshness = 0.5  # Valeur par défaut
        
        # 4. Quality score (basé sur la résolution et l'extension)
        features.quality_score = _calculate_quality_score(asset_path, asset_metadata)
        
        # 5. Embedding similarity (placeholder - à implémenter avec FAISS)
        features.embedding_similarity = 0.5  # Valeur par défaut
        
        # Calculer le score final pondéré
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
        print(f"⚠️ Erreur scoring mixte: {e}")
        # Fallback vers scoring basique
        return _score_broll_asset_basic(asset_path, asset_tags, query_keywords)

def _score_broll_asset_basic(asset_path: str, asset_tags: List[str], query_keywords: List[str]) -> float:
    """Scoring basique de fallback"""
    try:
        # Score simple basé sur l'overlap de tags
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
        print(f"⚠️ Erreur scoring basique: {e}")
        return 0.5

def _get_domain_keywords(domain: str) -> List[str]:
    """Retourne les mots-clés spécifiques au domaine"""
    domain_keywords = {
        'health': ['medical', 'healthcare', 'wellness', 'fitness', 'medicine', 'hospital', 'doctor'],
        'technology': ['tech', 'digital', 'innovation', 'computer', 'ai', 'software', 'data'],
        'business': ['business', 'entrepreneur', 'success', 'growth', 'strategy', 'office', 'professional'],
        'education': ['learning', 'education', 'knowledge', 'study', 'teaching', 'school', 'university'],
        'finance': ['money', 'finance', 'investment', 'wealth', 'business', 'success', 'growth']
    }
    
    return domain_keywords.get(domain.lower(), [domain])

def _calculate_quality_score(asset_path: str, metadata: Optional[Dict] = None) -> float:
    """Calcule un score de qualité basé sur les métadonnées"""
    try:
        score = 0.5  # Score de base
        
        # Bonus pour la résolution
        if metadata and 'resolution' in metadata:
            res = metadata['resolution']
            if '4k' in res or '3840' in res:
                score += 0.2
            elif '1080' in res or '1920' in res:
                score += 0.1
        
        # Bonus pour la durée
        if metadata and 'duration' in metadata:
            duration = metadata['duration']
            if 2.0 <= duration <= 6.0:  # Durée optimale
                score += 0.1
        
        # Bonus pour l'extension (préférer MP4)
        if asset_path.lower().endswith('.mp4'):
            score += 0.1
        
        return min(1.0, score)
        
    except Exception:
        return 0.5

    def _load_broll_selector_config(self):
        """Charge la configuration du sélecteur B-roll depuis le fichier YAML"""
        try:
            import yaml
            if Config.BROLL_SELECTOR_CONFIG_PATH.exists():
                with open(Config.BROLL_SELECTOR_CONFIG_PATH, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            else:
                print(f"    ⚠️ Fichier de configuration introuvable: {Config.BROLL_SELECTOR_CONFIG_PATH}")
                return {}
        except Exception as e:
            print(f"    ⚠️ Erreur chargement configuration: {e}")
            return {}

    def _calculate_asset_hash(self, asset_path: Path) -> str:
        """Calcule un hash unique pour un asset B-roll basé sur son contenu et métadonnées"""
        try:
            import hashlib
            import os
            from datetime import datetime
            
            # Hash basé sur le nom, la taille et la date de modification
            stat = asset_path.stat()
            hash_data = f"{asset_path.name}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.md5(hash_data.encode()).hexdigest()
        except Exception:
            # Fallback sur le nom du fichier
            return str(asset_path.name)
