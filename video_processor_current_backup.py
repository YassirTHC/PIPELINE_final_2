import os

import sys

from pathlib import Path

os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

os.environ.setdefault('PYTHONUTF8', '1')

for stream in (sys.stdout, sys.stderr):

    if hasattr(stream, 'reconfigure'):

        try:

            stream.reconfigure(encoding='utf-8')

        except Exception:

            pass

# ensure project-root is first

PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(1, str(PROJECT_ROOT / 'AI-B-roll'))

sys.path.insert(2, str(PROJECT_ROOT / 'AI-B-roll' / 'src'))

if 'utils' in sys.modules:

    del sys.modules['utils']

import logging

from dotenv import load_dotenv

from utils.logging_setup import get_console_logger

load_dotenv(PROJECT_ROOT / '.env', override=False)

load_dotenv(PROJECT_ROOT / '.env.local', override=True)

import concurrent.futures

import subprocess

import shlex

import time

from pathlib import Path

from typing import List, Dict, Any, Optional, Union, Sequence, Set, Tuple

from collections import Counter

import gc

# 🚀 NOUVEAU: Configuration des logs temps réel + suppression warnings non-critiques

import warnings

warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated")

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces` was not set")

warnings.filterwarnings("ignore", message="Warning: in file.*bytes wanted but.*bytes read")

logger = get_console_logger(name=__name__)

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

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    future = executor.submit(fn, *args, **kwargs)

    try:

        return future.result(timeout=timeout_s)

    except concurrent.futures.TimeoutError:

        try:

            future.cancel()

        except Exception:

            pass

        # Do not wait for background task to finish to keep call non-blocking

        try:

            executor.shutdown(wait=False, cancel_futures=True)  # type: ignore[call-arg]

        except TypeError:  # cancel_futures not available

            executor.shutdown(wait=False)

        return None

    except Exception:

        try:

            executor.shutdown(wait=False)

        except Exception:

            pass

        raise

    finally:

        # Ensure we don't leave threads around on normal completion

        try:

            executor.shutdown(wait=False)

        except Exception:

            pass

def dedupe_by_url(candidates):

    unique = []

    hits = 0

    local_seen: Set[str] = set()

    for candidate in candidates or []:

        url = getattr(candidate, 'url', None)

        if not url:

            unique.append(candidate)

            continue

        if url in SEEN_URLS or url in local_seen:

            hits += 1

            continue

        local_seen.add(url)

        unique.append(candidate)

    return unique, hits

def dedupe_by_phash(candidates):

    unique = []

    hits = 0

    local_seen: list[int] = []

    for candidate in candidates or []:

        preview = getattr(candidate, 'thumb_url', None)

        media_url = getattr(candidate, 'url', None)

        phash = compute_phash(preview, media_url=media_url)

        if phash is None:

            unique.append(candidate)

            continue

        pools = SEEN_PHASHES + local_seen

        if any(hamming_distance(phash, seen) <= PHASH_DISTANCE for seen in pools):

            hits += 1

            continue

        setattr(candidate, '_phash', phash)

        local_seen.append(phash)

        unique.append(candidate)

    return unique, hits

# --- Query normalization for external providers (Pexels/Pixabay)

_STOPWORDS: Set[str] = {

    'the','a','an','to','of','in','on','at','for','and','or','but',

    'first','thing','that','this','those','these','you','we','they',

    'with','from','by','as','is','are','be','was','were','it','its'

}

_SYN_PREFIX_MAP: Dict[str, List[str]] = {

    'brain_scan_monitor': ['mri scan', 'ct scanner', 'brain imaging'],

    'therapy_session': ['therapy session', 'psychology session', 'counseling'],

    'dopamine_release': ['brain chemistry', 'neurotransmitter lab', 'neuroscience research'],

    'path_visualization': ['goal roadmap', 'strategy roadmap', 'progress timeline'],

    'internal_reward': ['self reward', 'intrinsic motivation']

}

def _normalize_queries(llm_keywords: List[str], transcript_tokens: List[str], *, max_queries: int = 8) -> List[str]:

    """Produce a compact, deduplicated list of provider queries.

    - Prefer LLM keywords; fallback to transcript tokens if empty

    - Normalize underscores, strip punctuation, drop stopwords

    - Expand with a small synonym map; cap list size

    """

    import re as _re_clean

    def _yield_terms(source):

        for term in source or []:

            if not isinstance(term, str):

                continue

            cleaned = term.strip().lower().replace('_', ' ')

            cleaned = _re_clean.sub(r"[^a-z0-9\s]", ' ', cleaned)

            cleaned = _re_clean.sub(r"\s+", ' ', cleaned).strip()

            if cleaned and cleaned not in _STOPWORDS:

                yield cleaned

    base = list(dict.fromkeys(_yield_terms(llm_keywords)))

    if not base:

        base = list(dict.fromkeys(_yield_terms(transcript_tokens)))

    enriched: List[str] = []

    seen: Set[str] = set()

    for q in base:

        if q in seen:

            continue

        enriched.append(q)

        seen.add(q)

        key = q.replace(' ', '_')

        for alt in _SYN_PREFIX_MAP.get(key, []):

            alt_c = alt.strip().lower()

            if alt_c and alt_c not in seen:

                enriched.append(alt_c)

                seen.add(alt_c)

    return enriched[:max_queries]

def _setup_directories() -> None:

    for folder in (Config.CLIPS_FOLDER, Config.OUTPUT_FOLDER, Config.TEMP_FOLDER):

        folder.mkdir(parents=True, exist_ok=True)

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

from pipeline_core.logging import JsonlLogger, log_broll_decision, log_pipeline_summary, log_pipeline_error, log_stage_start, log_stage_end

from pipeline_core.runtime import PipelineResult, Stage

from pipeline_core.llm_service import DynamicCompletionError, LLMMetadataGeneratorService

# 🚀 NOUVEAU: Cache global pour éviter le rechargement des modèles

_MODEL_CACHE = {}

# --- Dynamic LLM context toggle (default: on)

ENABLE_DYNAMIC_CONTEXT = os.getenv("ENABLE_DYNAMIC_CONTEXT", "true").lower() not in {"0","false","no"}

# ---- Feature flags / knobs for per-segment refinement

ENABLE_SEGMENT_REFINEMENT = os.getenv("ENABLE_SEGMENT_REFINEMENT", "true").lower() not in {"0","false","no"}

SEGMENT_REFINEMENT_MAX_TERMS = int(os.getenv("SEGMENT_REFINEMENT_MAX_TERMS", "4"))

SEGMENT_FETCH_TIMEOUT_S = float(os.getenv("SEGMENT_FETCH_TIMEOUT_S", "1.5"))

SEGMENT_PARALLEL_REQUESTS = int(os.getenv("SEGMENT_PARALLEL_REQUESTS", "2"))

# ---- Feature flag: prefer dynamic LLM domain for selector

ENABLE_SELECTOR_DYNAMIC_DOMAIN = os.getenv("ENABLE_SELECTOR_DYNAMIC_DOMAIN", "true").lower() not in {"0","false","no"}

# --- Provider anti-terms and helpers for query building

_ANTI_TERMS = {"people","thing","nice","background","start","generic","template","stock"}

def _norm_query_term(s: str) -> str:

    s = (s or "").strip().lower().replace("_"," ")

    try:

        s = re.sub(r"[^\w\s\-]", "", s)

    except Exception:

        pass

    return s

def _dedupe_queries(seq, cap: int) -> list[str]:

    seen, out = set(), []

    for x in (seq or []):

        x = _norm_query_term(x)

        if len(x) < 3 or x in _ANTI_TERMS or not any(c.isalpha() for c in x):

            continue

        # Filter if any token is an anti-term (e.g., 'nice background')

        tokens = [t for t in x.split() if t]

        if any(t in _ANTI_TERMS for t in tokens):

            continue

        if x not in seen:

            out.append(x); seen.add(x)

        if len(out) >= cap:

            break

    return out

def enforce_broll_schedule_rules(

    plan: Sequence[Any],

    *,

    min_duration: float = 1.5,

    max_duration: float = 6.0,

    min_gap: float = 1.5,

) -> Tuple[List[Any], List[Dict[str, Any]]]:

    """Filter plan items based on duration/gap rules while avoiding empty results."""

    def _extract_times(entry: Any) -> Tuple[float, float, float]:

        start_val = float(

            getattr(entry, 'start', None)

            if hasattr(entry, 'start')

            else (entry.get('start') if isinstance(entry, dict) else 0.0)

        )

        end_val = float(

            getattr(entry, 'end', None)

            if hasattr(entry, 'end')

            else (entry.get('end') if isinstance(entry, dict) else 0.0)

        )

        duration_val = max(0.0, end_val - start_val)

        return start_val, end_val, duration_val

    def _apply_rules(min_duration_val: float, min_gap_val: float) -> Tuple[List[Any], List[Dict[str, Any]]]:

        filtered_local: List[Any] = []

        removals_local: List[Dict[str, Any]] = []

        last_end_local: Optional[float] = None

        for entry in plan or []:

            start, end, duration = _extract_times(entry)

            if duration < min_duration_val:

                removals_local.append({'start': start, 'end': end, 'reason': 'duration_short', 'duration': duration})

                continue

            if duration > max_duration:

                end = start + max_duration

                duration = max_duration

                if hasattr(entry, 'end'):

                    try:

                        setattr(entry, 'end', end)

                    except Exception:

                        pass

                elif isinstance(entry, dict):

                    entry['end'] = end

            if last_end_local is not None and start - last_end_local < min_gap_val:

                removals_local.append({'start': start, 'end': end, 'reason': 'gap_violation', 'gap': start - last_end_local})

                continue

            if hasattr(entry, 'start'):

                try:

                    setattr(entry, 'start', start)

                except Exception:

                    pass

            elif isinstance(entry, dict):

                entry['start'] = start

                entry['end'] = end

            filtered_local.append(entry)

            last_end_local = end

        return filtered_local, removals_local

    filtered, removals = _apply_rules(min_duration, min_gap)

    if filtered or not plan:

        return filtered, removals

    durations = [max(0.0, _extract_times(entry)[2]) for entry in plan or []]

    relaxed_min_duration = min(min_duration, max(0.75, min(durations) if durations else min_duration))

    relaxed_min_gap = max(0.0, min_gap * 0.5)

    relaxed_filtered, relaxed_removals = _apply_rules(relaxed_min_duration, relaxed_min_gap)

    if relaxed_filtered:

        removals.append({

            'start': None,

            'end': None,

            'reason': 'relaxed_rules_applied',

            'min_duration': relaxed_min_duration,

            'min_gap': relaxed_min_gap,

            'initial_count': len(plan or []),

            'remaining': len(relaxed_filtered),

        })

        removals.extend(relaxed_removals)

        return relaxed_filtered, removals

    longest_item = None

    longest_duration = -1.0

    for entry in plan or []:

        _, _, duration = _extract_times(entry)

        if duration > longest_duration:

            longest_duration = duration

            longest_item = entry

    if longest_item is not None:

        removals.append({'start': None, 'end': None, 'reason': 'forced_keep', 'kept_duration': longest_duration})

        return [longest_item], removals

    return filtered, removals

def _segment_terms_from_briefs(dyn: dict, seg_idx: int, cap: int) -> list[str]:

    """Extract up to `cap` clean terms (keywords+queries) for a given segment index."""

    if not isinstance(dyn, dict):

        return []

    briefs = dyn.get("segment_briefs") or []

    pool: list[str] = []

    for br in briefs:

        if not isinstance(br, dict):

            continue

        raw_idx = br.get("segment_index")

        try:

            idx_val = int(raw_idx)

        except Exception:

            idx_val = None

        matches = False

        if idx_val is not None:

            if idx_val == seg_idx:

                matches = True

            elif idx_val == seg_idx + 1:

                matches = True

            elif idx_val == seg_idx - 1 and seg_idx == 0:

                matches = True

        if not matches:

            continue

        pool.extend(br.get("keywords") or [])

        pool.extend(br.get("queries") or [])

        break  # only first matching brief for determinism

    return _dedupe_queries(pool, cap)

def _choose_dynamic_domain(dyn: dict):

    """Pick the best domain from LLM dynamic context.

    Returns (name, confidence) or (None, None).

    """

    try:

        domains = dyn.get("detected_domains") or []

        best_name = None

        best_conf = -1.0

        for d in domains:

            if not isinstance(d, dict):

                continue

            name = str(d.get("name") or "").strip()

            if not name:

                continue

            try:

                conf = float(d.get("confidence", 0.0) or 0.0)

            except Exception:

                conf = 0.0

            if conf > best_conf:

                best_conf = conf

                best_name = name

        if best_name:

            return best_name, (best_conf if best_conf >= 0 else None)

    except Exception:

        pass

    return None, None

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

# Mediapipe availability (optional)

try:

    import mediapipe as mp  # type: ignore

    _MEDIAPIPE_AVAILABLE = True

except Exception:

    mp = None  # type: ignore

    _MEDIAPIPE_AVAILABLE = False

if _MEDIAPIPE_AVAILABLE:

    logger.info("Mediapipe available - advanced features enabled")

else:

    logger.warning("Mediapipe not installed - continuing without Mediapipe features")

try:

    from broll_selector import BrollSelector, Asset, ScoringFeatures, BrollCandidate

    BROLL_SELECTOR_AVAILABLE = True

    print("✅ Sélecteur B-roll générique disponible - Scoring mixte activé")

except ImportError as e:

    BROLL_SELECTOR_AVAILABLE = False

    print(f"⚠️ Sélecteur B-roll générique non disponible: {e}")

    print("   🔄 Utilisation du système de scoring existant")

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

def _pipeline_core_fetcher_enabled() -> bool:

    """Resolve the pipeline_core toggle at runtime (env/UI overrides)."""

    override = os.getenv("ENABLE_PIPELINE_CORE_FETCHER")

    if override is not None:

        return _to_bool(override)

    legacy = os.getenv("AI_PIPELINE_CORE_FETCHER")

    if legacy is not None:

        return _to_bool(legacy)

    ui_value = _UI_SETTINGS.get("pipeline_core_fetcher") if isinstance(_UI_SETTINGS, dict) else None

    if ui_value is not None:

        return _to_bool(ui_value)

    return getattr(Config, "ENABLE_PIPELINE_CORE_FETCHER", False)

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

def _mask_api_key(value: Optional[str]) -> Optional[str]:

    if not value:

        return None

    masked = str(value).strip()

    if not masked:

        return None

    tail = masked[-4:] if len(masked) >= 4 else masked

    return f"****{tail}"

_PROVIDER_API_ENV_MAP = {

    'pexels': 'PEXELS_API_KEY',

    'pixabay': 'PIXABAY_API_KEY',

}

SUPPORTED_BROLL_PROVIDERS: Set[str] = {'pexels', 'pixabay'}

def _normalize_provider_list(raw: Any) -> List[str]:

    """Return supported provider names from a raw config value."""

    normalized: List[str] = []

    if raw is None:

        return normalized

    if isinstance(raw, str):

        tokens = raw.replace(';', ',').split(',')

    elif isinstance(raw, (list, tuple, set)):

        tokens = list(raw)

    else:

        tokens = [raw]

    for token in tokens:

        name = str(token).strip().lower()

        if not name or name not in SUPPORTED_BROLL_PROVIDERS:

            continue

        if name not in normalized:

            normalized.append(name)

    return normalized

def _resolve_provider_api_key(provider_name: str) -> Optional[str]:

    name = (provider_name or '').strip().lower()

    env_key = _PROVIDER_API_ENV_MAP.get(name)

    if not env_key:

        return None

    candidate = None

    try:

        config_cls = getattr(__import__('config', fromlist=['Config']), 'Config')

        candidate = getattr(config_cls, env_key, None)

    except Exception:

        candidate = None

    candidate = candidate or os.environ.get(env_key)

    if candidate is None:

        return None

    cleaned = str(candidate).strip()

    return cleaned or None

def _iter_provider_configs(providers) -> List[Any]:

    if providers is None:

        return []

    if isinstance(providers, dict):

        return list(providers.values())

    if isinstance(providers, (list, tuple, set)):

        return list(providers)

    return [providers]

class VideoProcessor:

    _shared_llm_service = None

    """Classe principale pour traiter les vidéos"""

    def _collect_broll_env_snapshot(self):

        fetcher_cfg = getattr(self._pipeline_config, 'fetcher', None) if hasattr(self, '_pipeline_config') else None

        providers_meta: List[Dict[str, Any]] = []

        warnings: List[Tuple[str, str]] = []

        seen_names: Set[str] = set()

        if fetcher_cfg:

            raw_providers = getattr(fetcher_cfg, 'providers', None)

            for provider_conf in _iter_provider_configs(raw_providers):

                name = str(getattr(provider_conf, 'name', '') or '').strip().lower()

                if not name:

                    continue

                if name not in SUPPORTED_BROLL_PROVIDERS:

                    warnings.append((name, 'unsupported_provider'))

                    try:

                        setattr(provider_conf, 'enabled', False)

                    except Exception:

                        pass

                    continue

                seen_names.add(name)

                enabled = bool(getattr(provider_conf, 'enabled', True))

                requires_key = name in _PROVIDER_API_ENV_MAP

                key_value = _resolve_provider_api_key(name)

                key_present = bool(key_value) if requires_key else True

                providers_meta.append({

                    'name': name,

                    'enabled': enabled,

                    'key_present': key_present,

                    'masked_key': _mask_api_key(key_value),

                    'source': 'core',

                    'requires_key': requires_key,

                })

                if requires_key and enabled and not key_present:

                    warnings.append((name, 'missing_api_key'))

                    try:

                        setattr(provider_conf, 'enabled', False)

                    except Exception:

                        pass

        legacy_raw = getattr(Config, 'BROLL_FETCH_PROVIDER', '') or ''

        legacy_candidates = _normalize_provider_list(legacy_raw)

        if legacy_raw and not legacy_candidates:

            warnings.append((str(legacy_raw).strip().lower(), 'unsupported_provider'))

        for legacy_provider in legacy_candidates:

            if legacy_provider in seen_names:

                continue

            requires_key = legacy_provider in _PROVIDER_API_ENV_MAP

            key_value = _resolve_provider_api_key(legacy_provider)

            key_present = bool(key_value) if requires_key else True

            providers_meta.append({

                'name': legacy_provider,

                'enabled': True,

                'key_present': key_present,

                'masked_key': _mask_api_key(key_value),

                'source': 'legacy',

                'requires_key': requires_key,

            })

            seen_names.add(legacy_provider)

            if requires_key and not key_present:

                warnings.append((legacy_provider, 'missing_api_key'))

        allow_images = bool(getattr(Config, 'BROLL_FETCH_ALLOW_IMAGES', False))

        allow_videos = bool(getattr(Config, 'BROLL_FETCH_ALLOW_VIDEOS', True))

        fetch_max_raw = getattr(Config, 'BROLL_FETCH_MAX_PER_KEYWORD', 0) or 0

        try:

            fetch_max = int(fetch_max_raw)

        except (TypeError, ValueError):

            fetch_max = 0

        return providers_meta, allow_images, allow_videos, fetch_max, warnings

    def _emit_env_boot_event(self, event_logger: JsonlLogger, *, session_id: str) -> None:

        try:

            providers_meta, allow_images, allow_videos, fetch_max, warnings = self._collect_broll_env_snapshot()

        except Exception as exc:

            logger.warning('[BROLL] failed to build environment snapshot: %s', exc)

            return

        active = sorted({entry['name'] for entry in providers_meta if entry.get('enabled') and entry.get('key_present')})

        payload = {

            'event': 'broll_env_ready',

            'session_id': session_id,

            'providers_actifs': active,

            'providers': providers_meta,

            'allow_images': bool(allow_images),

            'allow_videos': bool(allow_videos),

            'fetch_max': int(fetch_max),

        }

        event_logger.log(payload)

        providers_display = ','.join(active) if active else 'none'

        print_realtime(f"[ENV] providers={providers_display} | allow_images={str(bool(allow_images)).lower()} | fetch_max={int(fetch_max)}")

        seen: Set[Tuple[str, str]] = set()

        for provider_name, reason in warnings:

            key = (provider_name, reason)

            if key in seen:

                continue

            seen.add(key)

            human_reason = 'missing API key' if reason == 'missing_api_key' else reason

            logger.warning("[BROLL] provider '%s' skipped: %s", provider_name, human_reason)

            event_logger.log({

                'event': 'provider_skipped_missing_key',

                'provider': provider_name,

                'reason': reason,

            })

    def __init__(self):

        self.whisper_model = whisper.load_model(Config.WHISPER_MODEL)

        _setup_directories()

        # Cache éventuel pour spaCy

        self._spacy_model = None

        if VideoProcessor._shared_llm_service is None:

            try:

                VideoProcessor._shared_llm_service = LLMMetadataGeneratorService()

            except Exception as exc:

                logger.warning("LLM service initialisation failed: %s", exc)

        self._llm_service = VideoProcessor._shared_llm_service

        self._core_last_run_used = False

        self._pipeline_config = PipelineConfigBundle()

        self._broll_event_logger = None

    

    def _setup_directories(self):

        """Crée les dossiers nécessaires"""

        for folder in [Config.CLIPS_FOLDER, Config.OUTPUT_FOLDER, Config.TEMP_FOLDER]:

            folder.mkdir(parents=True, exist_ok=True)

    

    def _generate_unique_output_dir(self, clip_stem: str) -> Path:

        """Crée un dossier unique pour ce clip sous output/clips/<stem>[-NNN]"""

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

    def process_single_clip(self, clip_path: Path, *, enable_core: Optional[bool] = None, verbose: bool = False):

        """Entry point used by the CLI wrapper with JSONL boot events."""

        clip_path = Path(clip_path)

        if not clip_path.exists():

            raise FileNotFoundError(f"Source video not found: {clip_path}")

        _setup_directories()

        output_root = Config.OUTPUT_FOLDER

        meta_dir = output_root / 'meta'

        final_dir = output_root / 'final'

        subtitled_dir = output_root / 'subtitled'

        for folder in (meta_dir, final_dir, subtitled_dir):

            folder.mkdir(parents=True, exist_ok=True)

        session_id = f"{clip_path.stem}-{int(time.time() * 1000)}"

        log_file = meta_dir / 'broll_pipeline_events.jsonl'

        event_logger = JsonlLogger(log_file)

        self._broll_event_logger = event_logger

        self._emit_env_boot_event(event_logger, session_id=session_id)

        result = PipelineResult()

        self._last_pipeline_result = result

        use_core = enable_core if enable_core is not None else _pipeline_core_fetcher_enabled()

        logger.info('[CORE] Orchestrator %s', 'enabled' if use_core else 'disabled (legacy path)')

        boot_payload = {

            'event': 'pipeline_core_boot',

            'video': clip_path.name,

            'enable_core': bool(use_core),

            'session_id': session_id,

        }

        event_logger.log(boot_payload)

        event_logger.log({'event': 'core_start' if use_core else 'legacy_start', 'video': clip_path.name, 'session_id': session_id})

        stage_details = {

            'video': clip_path.name,

            'cwd': str(Path.cwd()),

            'core_requested': bool(use_core),

        }

        pipeline_stage_started = log_stage_start(event_logger, Stage.PIPELINE.value, stage_details)

        previous_flag = os.environ.get('ENABLE_PIPELINE_CORE_FETCHER')

        os.environ['ENABLE_PIPELINE_CORE_FETCHER'] = 'true' if use_core else 'false'

        start_time = time.time()

        pipeline_error: Optional[Exception] = None

        final_output = None

        try:

            try:

                final_output = self._process_single_clip_impl(clip_path, verbose=verbose)

            except Exception as exc:

                pipeline_error = exc

                log_pipeline_error(

                    event_logger,

                    Stage.PIPELINE.value,

                    error_type=exc.__class__.__name__,

                    message=str(exc),

                )

                result.errors.append(str(exc))

                result.final_export_ok = False

                result.final_export_path = None

        finally:

            if previous_flag is None:

                os.environ.pop('ENABLE_PIPELINE_CORE_FETCHER', None)

            else:

                os.environ['ENABLE_PIPELINE_CORE_FETCHER'] = previous_flag

        final_path_obj = Path(final_output) if final_output else None

        if final_path_obj is not None:

            result.final_export_path = str(final_path_obj)

            result.final_export_ok = final_path_obj.exists()

        elif result.final_export_ok is None:

            result.final_export_ok = False

            result.final_export_path = None

        event_logger.log({

            'event': 'finalized',

            'video': clip_path.name,

            'session_id': session_id,

            'elapsed_s': round(time.time() - start_time, 3),

            'final_mp4': str(final_path_obj) if final_path_obj else None,

            'core_requested': bool(use_core),

            'core_effective': bool(getattr(self, '_core_last_run_used', False)),

        })

        log_path = getattr(event_logger, 'path', None)

        if result.end_ts is None:

            result.finish()

        log_stage_end(

            event_logger,

            Stage.PIPELINE.value,

            started_at=pipeline_stage_started,

            ok=bool(result.final_export_ok),

            details={'final_path': result.final_export_path},

        )

        summary_extra = {

            'video': clip_path.name,

            'session_id': session_id,

            'core_requested': bool(use_core),

            'core_effective': bool(getattr(self, '_core_last_run_used', False)),

        }

        effective_domain = getattr(self, '_effective_domain', None)

        if effective_domain:

            summary_extra['effective_domain'] = effective_domain

        log_pipeline_summary(event_logger, result, extra=summary_extra)

        if pipeline_error is None and isinstance(log_path, Path) and (not log_path.exists() or log_path.stat().st_size == 0):

            raise RuntimeError('[CORE] No JSONL events written for this run.')

        if pipeline_error is not None:

            raise pipeline_error

        return result

    def _get_broll_event_logger(self):

        if getattr(self, '_broll_event_logger', None) is None:

            log_file = Config.OUTPUT_FOLDER / 'meta' / 'broll_pipeline_events.jsonl'

            self._broll_event_logger = JsonlLogger(log_file)

        return self._broll_event_logger

    def _maybe_use_pipeline_core(

        self,

        segments,

        broll_keywords,

        *,

        subtitles,

        input_path: Path,

    ) -> Optional[Path]:

        """Attempt to run the pipeline_core orchestrator if configured."""

        event_logger = self._get_broll_event_logger()

        if not _pipeline_core_fetcher_enabled():

            print("    [CORE] pipeline_core fetcher disabled -> falling back to legacy compositor")

            event_logger.log({'event': 'core_disabled', 'reason': 'flag_disabled'})

            self._core_last_run_used = False

            return None

        fetcher_cfg = getattr(self._pipeline_config, 'fetcher', None)

        if fetcher_cfg is None:

            logger.warning(

                "pipeline_core fetcher enabled but fetcher configuration is missing; "

                "falling back to legacy pipeline",

            )

            print("    [CORE] pipeline_core fetcher missing configuration -> falling back to legacy")

            event_logger.log({'event': 'core_disabled', 'reason': 'missing_fetcher_config'})

            self._core_last_run_used = False

            return None

        providers = getattr(fetcher_cfg, 'providers', None)

        providers_list: List[Any]

        misconfigured = False

        if providers is None:

            providers_list = []

        else:

            try:

                providers_list = list(providers or ())

            except TypeError:

                misconfigured = True

                providers_list = [providers]

        if misconfigured:

            logger.warning(

                "pipeline_core fetcher misconfigured: expected iterable `fetcher.providers`, "

                "got %s; falling back to legacy pipeline",

                type(providers).__name__,

            )

            print("    [CORE] pipeline_core fetcher misconfigured -> falling back to legacy")

            event_logger.log({'event': 'core_disabled', 'reason': 'providers_misconfigured', 'provider_type': type(providers).__name__})

            self._core_last_run_used = False

            return None

        if not providers_list:

            logger.warning(

                "pipeline_core fetcher enabled but no providers configured; falling back to legacy pipeline",

            )

            print("    [CORE] pipeline_core fetcher has no providers -> falling back to legacy")

            event_logger.log({'event': 'core_disabled', 'reason': 'no_providers'})

            self._core_last_run_used = False

            return None

        event_logger.log({'event': 'core_engaged', 'video': input_path.name, 'providers': len(providers_list)})

        core_output = self._insert_brolls_pipeline_core(

            segments,

            broll_keywords,

            subtitles=subtitles,

            input_path=input_path,

        )

        if not core_output:

            print("    [CORE] pipeline_core returned no artefact -> falling back to legacy")

            event_logger.log({'event': 'core_disabled', 'reason': 'core_output_missing', 'video': input_path.name})

            self._core_last_run_used = False

            return None

        core_path = Path(core_output)

        try:

            same_as_input = core_path.resolve() == input_path.resolve()

        except Exception:

            same_as_input = False

        if same_as_input:

            print("    [CORE] pipeline_core returned source video -> falling back to legacy")

            event_logger.log({'event': 'core_disabled', 'reason': 'core_output_same_as_input', 'video': input_path.name})

            self._core_last_run_used = False

            return None

        if not core_path.exists():

            print(f"    [CORE] pipeline_core output missing on disk -> falling back to legacy (path={core_path})")

            event_logger.log({'event': 'core_disabled', 'reason': 'core_output_missing', 'video': input_path.name, 'path': str(core_path)})

            self._core_last_run_used = False

            return None

        self._core_last_run_used = True

        return core_path

    def _insert_brolls_pipeline_core(self, segments, broll_keywords, *, subtitles, input_path: Path) -> Optional[Path]:

        global SEEN_URLS, SEEN_PHASHES

        SEEN_URLS.clear()

        SEEN_PHASHES.clear()

        logger.info("[BROLL] pipeline_core orchestrator engaged")

        config_bundle = self._pipeline_config

        event_logger = self._get_broll_event_logger()

        orchestrator = FetcherOrchestrator(config_bundle.fetcher, event_logger=event_logger)

        selection_cfg = config_bundle.selection

        timeboxing_cfg = config_bundle.timeboxing

        event_logger.log(

            {

                "event": "broll_session_start",

                "segment": -1,

                "total_segments": len(segments),

                "llm_healthy": bool(self._llm_service),

            }

        )

        provider_usage: Dict[str, int] = {}

        refined_segments = 0

        brief_segments = 0

        selected_durations: List[float] = []

        provider_counts: Counter = Counter()

        total_url_hits = 0

        total_phash_hits = 0

        # Prepare selection report (once per clip)

        report = None

        try:

            dyn_ctx = getattr(self, '_dyn_context', None)

            dom_name, dom_conf = _choose_dynamic_domain(dyn_ctx) if ENABLE_SELECTOR_DYNAMIC_DOMAIN else (None, None)

            dom_source = 'dyn' if dom_name else 'none'

            sk = list(getattr(self, '_selector_keywords', []))

            fk = list(getattr(self, '_fetch_keywords', []))

            report = {

                'video_stem': Path(input_path).stem,

                'effective_domain': dom_name,

                'domain_confidence': dom_conf,

                'domain_source': dom_source,

                'selector_keywords': sk,

                'fetch_keywords': fk,

                'segments': [],

            }

        except Exception:

            report = None

        sanitized_segments = []

        dropped_invalid = 0

        for segment in segments or []:

            try:

                start = float(getattr(segment, 'start', 0.0))

                end = float(getattr(segment, 'end', start))

            except (TypeError, ValueError):

                dropped_invalid += 1

                continue

            text = getattr(segment, 'text', '')

            if not str(text).strip():

                dropped_invalid += 1

                continue

            if start < 0 or end < 0 or end < start:

                dropped_invalid += 1

                continue

            sanitized_segments.append(segment)

        if dropped_invalid:

            event_logger.log({

                'event': 'core_segment_filtered',

                'dropped': dropped_invalid,

                'remaining': len(sanitized_segments),

                'video': input_path.name,

            })

            logger.warning(

                "Filtered %s invalid transcript segments before core fetcher (remaining=%s)",

                dropped_invalid,

                len(sanitized_segments),

            )

        segments = sanitized_segments

        if not segments:

            event_logger.log({'event': 'core_no_segments', 'reason': 'no_valid_segments', 'video': input_path.name})

            self._core_last_run_used = False

            return

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

            # Optional per-segment refinement using dynamic LLM context briefs

            refined = False

            try:

                dyn = getattr(self, '_dyn_context', None)

            except Exception:

                dyn = None

            reason = "no_brief"

            if ENABLE_SEGMENT_REFINEMENT and dyn:

                try:

                    seg_terms = _segment_terms_from_briefs(dyn, idx, SEGMENT_REFINEMENT_MAX_TERMS)

                except Exception:

                    seg_terms = []

                if seg_terms:

                    brief_segments += 1

                    if seg_terms != queries:

                        queries = seg_terms

                        refined = True

                        refined_segments += 1

                        reason = "brief_applied"

                    else:

                        reason = "no_change"

                else:

                    reason = "empty_terms"

            # Observability: show per-segment queries

            try:

                print(f"[BROLL] segment #{idx}: queries={queries} (refined={refined})")

            except Exception:

                pass

            try:

                ev = self._get_broll_event_logger()

                if ev:

                    ev.log({

                        "event": "broll_segment_queries",

                        "segment": idx,

                        "queries": queries,

                        "refined": bool(refined),

                        "reason": reason,

                    })

            except Exception:

                pass

            filters = {}

            if llm_hints and isinstance(llm_hints.get('filters'), dict):

                filters = llm_hints['filters'] or {}

            start_time = time.perf_counter()

            def _do_fetch():

                return orchestrator.fetch_candidates(

                    queries,
                    segment_index=idx,
                    duration_hint=seg_duration,
                    filters=filters,

                    segment_timeout_s=eff_timeout,

                )

            # Timebox the per-segment fetch using the strictest timeout

            eff_timeout = fetch_timeout

            # Config timeboxing: per-request timeout if available

            try:

                per_request = float(getattr(self._pipeline_config.timeboxing, "request_timeout_s", SEGMENT_FETCH_TIMEOUT_S) or SEGMENT_FETCH_TIMEOUT_S)

            except Exception:

                per_request = SEGMENT_FETCH_TIMEOUT_S

            # Environment guard

            try:

                guard = float(SEGMENT_FETCH_TIMEOUT_S)

            except Exception:

                guard = SEGMENT_FETCH_TIMEOUT_S

            # Start from a positive base if none set

            if not eff_timeout or eff_timeout <= 0.0:

                eff_timeout = per_request

            # Apply the strictest of all timeouts

            eff_timeout = min(v for v in (eff_timeout, per_request, guard) if v and v > 0.0)

            fetch_start = time.perf_counter()

            candidates = run_with_timeout(_do_fetch, eff_timeout) if eff_timeout else _do_fetch()

            fetch_latency_ms = int((time.perf_counter() - fetch_start) * 1000)

            try:

                ev = self._get_broll_event_logger()

                if ev:

                    ev.log({

                        "event": "broll_segment_fetch_latency",

                        "segment": idx,

                        "latency_ms": fetch_latency_ms,

                        "query_count": len(queries or []),

                        "timeout_s": eff_timeout,

                    })

            except Exception:

                pass

            if candidates is None:

                candidates = []

            unique_candidates, url_hits = dedupe_by_url(candidates)

            unique_candidates, phash_hits = dedupe_by_phash(unique_candidates)

            total_url_hits += url_hits

            total_phash_hits += phash_hits

            best_candidate = None

            best_score = -1.0

            best_provider = None

            reject_reasons: List[str] = []

            for candidate in unique_candidates:

                score = self._rank_candidate(

                    segment.text,

                    candidate,

                    selection_cfg,

                    seg_duration,

                    queries=queries,

                    provider_usage=provider_usage,

                )

                if score < selection_cfg.min_score:

                    reject_reasons.append('low_score')

                    continue

                if score > best_score:

                    best_candidate = candidate

                    best_score = score

                    best_provider = getattr(candidate, 'provider', None)

            if best_candidate:

                url = getattr(best_candidate, 'url', None)

                if url:

                    SEEN_URLS.add(url)

                ph = getattr(best_candidate, '_phash', None)

                if ph is not None:

                    SEEN_PHASHES.append(ph)

                if best_provider:

                    provider_usage[best_provider] = provider_usage.get(best_provider, 0) + 1

                    provider_counts[best_provider] += 1

                duration_selected = max(0.0, min(seg_duration, 6.0))

                if duration_selected >= 0.05:

                    selected_durations.append(duration_selected)

                # Append to selection report

                try:

                    if report is not None:

                        report['segments'].append({

                            'segment': idx,

                            't0': float(getattr(segment, 'start', 0.0)),

                            't1': float(getattr(segment, 'end', 0.0)),

                            'selected_url': getattr(best_candidate, 'url', None),

                            'provider': getattr(best_candidate, 'provider', None),

                            'score': float(best_score) if best_score is not None else None,

                        })

                except Exception:

                    pass

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

                queries=list(queries or []),

            )

        # Add effective domain fields to the summary line for easy scraping

        clip_duration = 0.0

        try:

            clip_duration = float(max((getattr(seg, 'end', 0.0) for seg in segments), default=0.0))

        except Exception:

            clip_duration = 0.0

        selection_count = len(selected_durations)

        total_segments = len(segments) if segments else 0

        selection_rate = round(selection_count / total_segments, 3) if total_segments else 0.0

        avg_duration = round(sum(selected_durations) / selection_count, 3) if selection_count else 0.0

        broll_per_min = 0.0

        if clip_duration > 0 and selection_count:

            broll_per_min = round(selection_count / (clip_duration / 60.0), 3) if clip_duration >= 1.0 else float(selection_count)

        refined_ratio = round(refined_segments / brief_segments, 3) if brief_segments else 0.0

        dup_rate_phash = round(total_phash_hits / max(1, total_phash_hits + selection_count), 3)

        provider_status = {}

        try:

            dom_name = report.get('effective_domain') if isinstance(report, dict) else None

            dom_source = report.get('domain_source') if isinstance(report, dict) else None

            dom_conf = report.get('domain_confidence') if isinstance(report, dict) else None

            provider_status.update({

                'effective_domain': dom_name,

                'domain_source': dom_source,

                'domain_confidence': dom_conf,

            })

        except Exception:

            pass

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

            provider_status=provider_status or None,

        )

        # Persist compact selection report next to JSONL

        try:

            ENABLE_SELECTION_REPORT = os.getenv("ENABLE_SELECTION_REPORT", "true").lower() not in {"0","false","no"}

        except Exception:

            ENABLE_SELECTION_REPORT = True

        if ENABLE_SELECTION_REPORT and report is not None:

            try:

                seg_total = len(segments) if segments else 0

                seg_sel = len(report.get('segments') or [])

                report['selection_rate'] = selection_rate

                report['avg_broll_duration'] = avg_duration

                report['broll_per_min'] = broll_per_min

                report['refined_ratio'] = refined_ratio

                report['dup_rate_phash'] = dup_rate_phash

                report['segments_refined'] = refined_segments

                report['segments_with_brief'] = brief_segments

                report['selected_segments'] = selection_count

                report['total_segments'] = total_segments

                report['clip_duration_s'] = clip_duration

                report['total_url_dedup_hits'] = total_url_hits

                report['total_phash_dedup_hits'] = total_phash_hits

                report['by_provider'] = dict(provider_counts)

                meta_dir = Config.OUTPUT_FOLDER / 'meta'

                meta_dir.mkdir(parents=True, exist_ok=True)

                name = f"selection_report_{report.get('video_stem') or 'clip'}.json"

                out_path = meta_dir / name

                with open(out_path, 'w', encoding='utf-8') as f:

                    json.dump(report, f, ensure_ascii=False, indent=2)

                print(f"[REPORT] wrote {out_path}")

            except Exception as e:

                print(f"[REPORT] failed: {e}")

        return None

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

    def _rank_candidate(

        self,

        segment_text: str,

        candidate,

        selection_cfg,

        segment_duration: float,

        *,

        queries: Optional[Sequence[str]] = None,

        provider_usage: Optional[Dict[str, int]] = None,

    ) -> float:

        base_score = self._estimate_candidate_score(candidate, selection_cfg, segment_duration)

        segment_tokens = {tok for tok in _norm_query_term(segment_text).split() if len(tok) > 2}

        candidate_title = (getattr(candidate, 'title', '') or '').lower()

        candidate_tags = [t.lower() for t in getattr(candidate, 'tags', []) or []]

        candidate_text = ' '.join([candidate_title] + candidate_tags + [str(getattr(candidate, 'description', '') or '').lower()])

        if segment_tokens and candidate_title:

            overlap = sum(1 for tok in segment_tokens if tok in candidate_title)

            if overlap:

                base_score += min(0.12, overlap * 0.03)

        if queries:

            normalized_queries = [_norm_query_term(q) for q in queries if q]

            query_hits = sum(1 for q in normalized_queries if q and q in candidate_text)

            base_score += min(0.18, query_hits * 0.06)

        tag_hits = len(segment_tokens.intersection(set(candidate_tags)))

        if tag_hits:

            base_score += min(0.15, tag_hits * 0.04)

        provider = getattr(candidate, 'provider', None) or ''

        if provider_usage and provider:

            repeat = provider_usage.get(provider, 0)

            if repeat:

                base_score -= min(0.15, repeat * 0.05)

        ph = getattr(candidate, '_phash', None)

        if ph is not None and any(hamming_distance(ph, seen) <= PHASH_DISTANCE for seen in SEEN_PHASHES):

            base_score -= 0.3

        return max(0.0, min(1.0, base_score))

    

    def _safe_copy(self, src: Path, dst: Path) -> None:

        try:

            if src and Path(src).exists():

                dst.parent.mkdir(parents=True, exist_ok=True)

                shutil.copy2(str(src), str(dst))

        except Exception:

            pass

    def _hardlink_or_copy(self, src: Path, dst: Path) -> None:

        """Crée un hardlink si possible, sinon copie le fichier."""

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

    # 🚨 CORRECTION CRITIQUE: Méthodes manquantes pour le sélecteur B-roll

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

    def _extract_keywords_for_segment_spacy(self, text: str) -> List[str]:

        """Extraction optionnelle (spaCy) de mots-clés (noms/verbes/entités). Fallback heuristique si indisponible."""

        try:

            import re as _re

            

            # 🚨 CORRECTION IMMÉDIATE: Filtre des mots génériques inutiles

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

                for w in _re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']{4,}", text or ""):

                    lw = w.lower()

                    if lw not in keywords and lw not in GENERIC_WORDS:

                        keywords.append(lw)

            

                        # Build clean selector/fetch keyword lists from LLM + transcript

                        llm_keywords_input = list(broll_keywords or [])

                        transcript_tokens: List[str] = []

                        for _seg in subtitles:

                            _t = str(_seg.get('text','') or '')

                            transcript_tokens.extend(_t.split())

                        selector_keywords = _normalize_queries(llm_keywords_input, transcript_tokens, max_queries=12)

                        fetch_keywords = _normalize_queries(llm_keywords_input, transcript_tokens, max_queries=8)

                        if not selector_keywords:

                            selector_keywords = fetch_keywords[:]

                        if not fetch_keywords:

                            fetch_keywords = selector_keywords[:8] if selector_keywords else ['motivation','reward','focus','success','mindset']

            # 🚨 CORRECTION IMMÉDIATE: Prioriser les mots contextuels importants

            PRIORITY_WORDS = {

                'neuroscience', 'brain', 'mind', 'consciousness', 'cognitive', 'mental', 'psychology',

                'medical', 'health', 'treatment', 'research', 'science', 'discovery', 'innovation',

                'technology', 'digital', 'future', 'ai', 'artificial', 'intelligence', 'machine',

                'business', 'success', 'growth', 'strategy', 'leadership', 'entrepreneur', 'startup'

            }

            

            # Réorganiser pour prioriser les mots importants

            priority_keywords = [kw for kw in keywords if kw in PRIORITY_WORDS]

            other_keywords = [kw for kw in keywords if kw not in PRIORITY_WORDS]

            

            # Retourner d'abord les mots prioritaires, puis les autres

            final_keywords = priority_keywords + other_keywords

            return final_keywords[:12]

        except Exception:

            return []

    def process_all_clips(self, input_video_path: str):

        """Pipeline principal de traitement"""

        logger.info("🚀 Début du pipeline de traitement")

        print("🎬 Démarrage du pipeline de traitement...")

        

        # Étape 1: Découpage (votre IA existante)

        

        # Étape 2: Traitement de chaque clip

        clip_files = list(Config.CLIPS_FOLDER.glob("*.mp4"))

        total_clips = len(clip_files)

        

        print(f"📁 {total_clips} clips trouvés dans le dossier clips/")

        

        for i, clip_path in enumerate(clip_files):

            print(f"\n🎬 [{i+1}/{total_clips}] Traitement de: {clip_path.name}")

            logger.info(f"🎬 Traitement du clip {i+1}/{total_clips}: {clip_path.name}")

            

            # Skip si déjà traité

            stem = Path(clip_path).stem

            final_dir = Config.OUTPUT_FOLDER / 'final'

            processed_already = False

            if final_dir.exists():

                matches = list(final_dir.glob(f"final_{stem}*.mp4"))

                processed_already = len(matches) > 0

            if processed_already:

                print(f"⏩ Clip déjà traité, ignoré : {clip_path.name}")

                logger.info(f"⏩ Clip déjà traité, ignoré : {clip_path.name}")

                continue

            # Verrou concurrentiel par clip

            locks_dir = Config.OUTPUT_FOLDER / 'locks'

            locks_dir.mkdir(parents=True, exist_ok=True)

            lock_file = locks_dir / f"{stem}.lock"

            if lock_file.exists():

                print(f"⏭️ Verrou détecté, saut du clip: {clip_path.name}")

                continue

            try:

                lock_file.write_text("locked", encoding='utf-8')

                self.process_single_clip(clip_path)

                print(f"✅ Clip {clip_path.name} traité avec succès")

                logger.info(f"✅ Clip {clip_path.name} traité avec succès")

            except Exception as e:

                print(f"❌ Erreur lors du traitement de {clip_path.name}: {e}")

                logger.error(f"❌ Erreur lors du traitement de {clip_path.name}: {e}")

            finally:

                try:

                    if lock_file.exists():

                        lock_file.unlink()

                except Exception:

                    pass

        

        print(f"\n🎉 Pipeline terminé ! {total_clips} clips traités.")

        logger.info("🎉 Pipeline terminé avec succès")

        # Purge B-roll (librairie + caches) si demandé pour garder le disque léger

        try:

            if getattr(Config, 'BROLL_PURGE_AFTER_RUN', False):

                self._purge_broll_caches()

        except Exception:

            pass

        # Agréger un rapport global même sans --json-report

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

        Interface pour votre IA de découpage existante

        Remplacez cette méthode par votre implémentation

        """

        logger.info("📼 Découpage des clips avec IA...")

        

        # Exemple basique - remplacez par votre IA

        video = VideoFileClip(input_video_path)

        duration = video.duration

        

        # Découpage adaptatif selon la durée

        if duration <= 30:

            # Vidéo courte : utiliser toute la vidéo

            segment_duration = duration

            segments = 1

        else:

            # Vidéo longue : découper en segments de 30 secondes

            segment_duration = 30

            segments = max(1, int(duration // segment_duration))

        

        for i in range(min(segments, 5)):  # Max 5 clips pour test

            start_time = i * segment_duration

            end_time = min((i + 1) * segment_duration, duration)

            

            clip = video.subclip(start_time, end_time)

            output_path = Config.CLIPS_FOLDER / f"clip_{i+1:02d}.mp4"

            clip.write_videofile(str(output_path), verbose=False, logger=None)

        

        video.close()

        logger.info(f"✅ {segments} clips générés")

    

    def _process_single_clip_impl(self, clip_path: Path, *, verbose: bool = False):

        """Traite un clip individuel (reframe -> transcription (pour B-roll) -> B-roll -> sous-titres)"""

        

        # Dossier de sortie dédié et unique

        per_clip_dir = self._generate_unique_output_dir(clip_path.stem)

        

        print(f"  📐 Étape 1/4: Reframe dynamique IA...")

        reframed_path = self.reframe_to_vertical(clip_path)

        # Déplacer artefact reframed dans le dossier du clip

        try:

            dst_reframed = per_clip_dir / 'reframed.mp4'

            if Path(reframed_path).exists():

                shutil.move(str(reframed_path), str(dst_reframed))

            reframed_path = dst_reframed

        except Exception:

            pass

        

        print(f"  🗣️ Étape 2/4: Transcription Whisper (guide B-roll)...")

        # Transcrire tôt pour guider la sélection B-roll (SRT disponible)

        subtitles = self.transcribe_segments(reframed_path)

        try:

            # Écrire un SRT à côté de la vidéo reframée

            srt_reframed = reframed_path.with_suffix('.srt')

            write_srt(subtitles, srt_reframed)

            # Sauvegarder transcription segments JSON

            seg_json = per_clip_dir / f"{clip_path.stem}_segments.json"

            with open(seg_json, 'w', encoding='utf-8') as f:

                json.dump(subtitles, f, ensure_ascii=False)

        except Exception:

            pass

        

        print(f"  🎞️ Étape 3/4: Insertion des B-rolls {'(activée)' if getattr(Config, 'ENABLE_BROLL', False) else '(désactivée)'}...")

        

        # 🚀 CORRECTION: Générer les mots-clés LLM AVANT l'insertion des B-rolls

        broll_keywords = []

        try:

            print("    🤖 Génération précoce des mots-clés LLM pour B-rolls...")

            title, description, hashtags, broll_keywords = self.generate_caption_and_hashtags(subtitles)

            print(f"    ✅ Mots-clés B-roll LLM générés: {len(broll_keywords)} termes")

            print(f"    🎯 Exemples: {', '.join(broll_keywords[:5])}")

        except Exception as e:

            print(f"    ⚠️ Erreur génération mots-clés LLM: {e}")

            broll_keywords = []

        

        # Maintenant insérer les B-rolls avec les mots-clés LLM disponibles

        with_broll_path = self.insert_brolls_if_enabled(reframed_path, subtitles, broll_keywords)

        

        # Copier artefact with_broll si différent

        try:

            if with_broll_path and with_broll_path != reframed_path:

                self._safe_copy(with_broll_path, per_clip_dir / 'with_broll.mp4')

        except Exception:

            pass

        

        print(f"  ✨ Étape 4/4: Ajout des sous-titres Hormozi 1...")

        # Générer meta (titre/hashtags) depuis transcription (déjà fait)

        try:

            # Réutiliser les données déjà générées

            if not broll_keywords:  # Fallback si pas encore généré

                title, description, hashtags, broll_keywords = self.generate_caption_and_hashtags(subtitles)

            

            print(f"  📝 Title: {title}")

            print(f"  📝 Description: {description}")

            print(f"  #️⃣ Hashtags: {' '.join(hashtags)}")

            meta_path = per_clip_dir / 'meta.txt'

            with open(meta_path, 'w', encoding='utf-8') as f:

                f.write(

                    "Title: " + title + "\n\n" +

                    "Description: " + description + "\n\n" +

                    "Hashtags: " + ' '.join(hashtags) + "\n\n" +

                    "B-roll Keywords: " + ', '.join(broll_keywords) + "\n"

                )

            print(f"  📝 [MÉTADONNÉES] Fichier meta.txt sauvegardé: {meta_path}")

        except Exception as e:

            print(f"  ⚠️ [ERREUR MÉTADONNÉES] {e}")

            # Fallback: créer des métadonnées basiques

            try:

                meta_path = per_clip_dir / 'meta.txt'

                with open(meta_path, 'w', encoding='utf-8') as f:

                    f.write("Title: Vidéo générée automatiquement\n\nDescription: Contenu généré par pipeline vidéo\n\nHashtags: #video #auto\n\nB-roll Keywords: video, content\n")

                print(f"  📝 [FALLBACK] Métadonnées de base sauvegardées: {meta_path}")

            except Exception as e2:

                print(f"  ❌ [ERREUR FALLBACK] {e2}")

        

        # Appliquer style Hormozi sur la vidéo post B-roll

        subtitled_out_dir = per_clip_dir

        subtitled_out_dir.mkdir(parents=True, exist_ok=True)

        final_subtitled_path = subtitled_out_dir / 'final_subtitled.mp4'

        try:

            span_style_map = {

                # Business & Croissance

                "croissance": {"color": "#39FF14", "bold": True, "emoji": "📈"},

                "growth": {"color": "#39FF14", "bold": True, "emoji": "📈"},

                "opportunité": {"color": "#FFD700", "bold": True, "emoji": "��"},

                "opportunite": {"color": "#FFD700", "bold": True, "emoji": "🔑"},

                "innovation": {"color": "#00E5FF", "emoji": "⚡"},

                "idée": {"color": "#00E5FF", "emoji": "💡"},

                "idee": {"color": "#00E5FF", "emoji": "💡"},

                "stratégie": {"color": "#FF73FA", "emoji": "🧭"},

                "strategie": {"color": "#FF73FA", "emoji": "🧭"},

                "plan": {"color": "#FF73FA", "emoji": "🗺️"},

                # Argent & Finance

                "argent": {"color": "#FFD700", "bold": True, "emoji": "💰"},

                "money": {"color": "#FFD700", "bold": True, "emoji": "💰"},

                "cash": {"color": "#FFD700", "bold": True, "emoji": "💰"},

                "investissement": {"color": "#8AFF00", "bold": True, "emoji": "📊"},

                "investissements": {"color": "#8AFF00", "bold": True, "emoji": "📊"},

                "revenu": {"color": "#8AFF00", "emoji": "🏦"},

                "revenus": {"color": "#8AFF00", "emoji": "🏦"},

                "profit": {"color": "#8AFF00", "bold": True, "emoji": "💰"},

                "profits": {"color": "#8AFF00", "bold": True, "emoji": "💰"},

                "perte": {"color": "#FF3131", "emoji": "📉"},

                "pertes": {"color": "#FF3131", "emoji": "📉"},

                "échec": {"color": "#FF3131", "emoji": "❌"},

                "echec": {"color": "#FF3131", "emoji": "❌"},

                "budget": {"color": "#FFD700", "emoji": "🧾"},

                "gestion": {"color": "#FFD700", "emoji": "🪙"},

                "roi": {"color": "#8AFF00", "bold": True, "emoji": "📈"},

                "chiffre": {"color": "#FFD700", "emoji": "💰"},

                "ca": {"color": "#FFD700", "emoji": "💰"},

                # Relation & Client

                "client": {"color": "#00E5FF", "underline": True, "emoji": "🤝"},

                "clients": {"color": "#00E5FF", "underline": True, "emoji": "🤝"},

                "collaboration": {"color": "#00E5FF", "emoji": "🫱🏼‍🫲🏽"},

                "collaborations": {"color": "#00E5FF", "emoji": "🫱🏼‍🫲🏽"},

                "communauté": {"color": "#39FF14", "emoji": "🌍"},

                "communaute": {"color": "#39FF14", "emoji": "🌍"},

                "confiance": {"color": "#00E5FF", "emoji": "🔒"},

                "vente": {"color": "#FF73FA", "emoji": "🛒"},

                "ventes": {"color": "#FF73FA", "emoji": "🛒"},

                "deal": {"color": "#FF73FA", "emoji": "📦"},

                "deals": {"color": "#FF73FA", "emoji": "📦"},

                "prospect": {"color": "#00E5FF", "emoji": "🤝"},

                "prospects": {"color": "#00E5FF", "emoji": "🤝"},

                "contrat": {"color": "#FF73FA", "emoji": "📋"},

                # Motivation & Succès

                "succès": {"color": "#39FF14", "italic": True, "emoji": "🏆"},

                "succes": {"color": "#39FF14", "italic": True, "emoji": "🏆"},

                "motivation": {"color": "#FF73FA", "bold": True, "emoji": "🔥"},

                "énergie": {"color": "#FF73FA", "emoji": "⚡"},

                "energie": {"color": "#FF73FA", "emoji": "⚡"},

                "victoire": {"color": "#39FF14", "emoji": "🎯"},

                "discipline": {"color": "#FFD700", "emoji": "⏳"},

                "viral": {"color": "#FF73FA", "bold": True, "emoji": "🚀"},

                "viralité": {"color": "#FF73FA", "bold": True, "emoji": "🌐"},

                "viralite": {"color": "#FF73FA", "bold": True, "emoji": "🌐"},

                "impact": {"color": "#FF73FA", "emoji": "💥"},

                "explose": {"color": "#FF73FA", "emoji": "💥"},

                "explosion": {"color": "#FF73FA", "emoji": "💥"},

                # Risque & Erreurs

                "erreur": {"color": "#FF3131", "emoji": "⚠️"},

                "erreurs": {"color": "#FF3131", "emoji": "⚠️"},

                "warning": {"color": "#FF3131", "emoji": "⚠️"},

                "obstacle": {"color": "#FF3131", "emoji": "🧱"},

                "obstacles": {"color": "#FF3131", "emoji": "🧱"},

                "solution": {"color": "#00E5FF", "emoji": "🔧"},

                "solutions": {"color": "#00E5FF", "emoji": "🔧"},

                "leçon": {"color": "#00E5FF", "emoji": "📚"},

                "lecon": {"color": "#00E5FF", "emoji": "📚"},

                "apprentissage": {"color": "#00E5FF", "emoji": "🧠"},

                "problème": {"color": "#FF3131", "emoji": "🛑"},

                "probleme": {"color": "#FF3131", "emoji": "🛑"},

            }

            add_hormozi_subtitles(

                str(with_broll_path), subtitles, str(final_subtitled_path),

                brand_kit=getattr(Config, 'BRAND_KIT_ID', 'default'),

                span_style_map=span_style_map

            )

        except Exception as e:

            print(f"  ❌ Erreur ajout sous-titres Hormozi: {e}")

            # Pas de retour anticipé: continuer export simple

        

        # Export final accumulé dans output/final/ et sous-titré (burn-in) dans output/subtitled/

        final_dir = Config.OUTPUT_FOLDER / 'final'

        subtitled_dir = Config.OUTPUT_FOLDER / 'subtitled'

        # Noms de base sans extension

        base_name = clip_path.stem

        output_path = self._unique_path(final_dir, f"final_{base_name}", ".mp4")

        try:

            # Choisir source finale: si sous-titrée existe sinon with_broll sinon reframed

            source_final = None

            if final_subtitled_path.exists():

                source_final = final_subtitled_path

            elif with_broll_path and Path(with_broll_path).exists():

                source_final = with_broll_path

            else:

                source_final = reframed_path

            try:

                event_logger = self._get_broll_event_logger()

                if event_logger:

                    event_logger.log({

                        'event': 'broll_compositor_path_selected',

                        'path': str(source_final) if source_final else None,

                        'source': 'pipeline_core' if getattr(self, '_core_last_run_used', False) else 'legacy',

                        'exists': bool(source_final and Path(source_final).exists()),

                    })

            except Exception:

                pass

            if source_final and Path(source_final).exists():

                self._hardlink_or_copy(source_final, output_path)

                # Ecrire SRT: éviter le doublon si la vidéo finale a déjà les sous-titres incrustés

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

                    # Produire uniquement une SRT dans le dossier du clip, pas à côté du MP4 final

                    try:

                        write_srt(subtitles, per_clip_dir / 'final.srt')

                    except Exception:

                        pass

                # Toujours produire un VTT à côté du final pour compat

                try:

                    vtt_out = output_path.with_suffix('.vtt')

                    write_vtt(subtitles, vtt_out)

                except Exception:

                    pass

                # Copier final dans dossier clip

                self._hardlink_or_copy(output_path, per_clip_dir / 'final.mp4')

                # Si une version sous-titrée burn-in existe, la dupliquer dans output/subtitled/

                if final_subtitled_path.exists():

                    subtitled_out = self._unique_path(subtitled_dir, f"{base_name}_subtitled", ".mp4")

                    self._hardlink_or_copy(final_subtitled_path, subtitled_out)

                # Copier meta.txt à côté du final accumulé

                try:

                    meta_src = per_clip_dir / 'meta.txt'

                    if meta_src.exists():

                        self._hardlink_or_copy(meta_src, output_path.with_suffix('.txt'))

                except Exception:

                    pass

                # Ecrire un JSON récap par clip

                try:

                    # Durée et hash final

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

                print(f"  📤 Export terminé: {output_path.name}")

                # Nettoyage des intermédiaires pour limiter l'empreinte disque

                self._cleanup_files([

                    with_broll_path if with_broll_path and with_broll_path != output_path else None,

                ])

                return output_path

            else:

                print(f"  ⚠️ Fichier final introuvable")

                return None

        except Exception as e:

            print(f"  ❌ Erreur export: {e}")

            return None

    def _get_sample_times(self, duration: float, fps: int) -> List[float]:

        if duration <= 10:

            return list(np.arange(0, duration, 1/fps))

        elif duration <= 30:

            return list(np.arange(0, duration, 2/fps))

        else:

            return list(np.arange(0, duration, 4/fps))

    def _smooth_trajectory(self, x_centers: List[float], window_size: int = 15) -> List[float]:

        # Fenêtre plus grande pour un lissage plus smooth

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

        # EMA additionnel pour atténuer le jitter haute fréquence

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

        # Détecteurs MediaPipe

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

        mp_pose = mp.solutions.pose

        mp_face = mp.solutions.face_detection

        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.8) as pose, mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detection:

            for t in tqdm(sample_times, desc="🔎 IA focus", leave=False):

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

        """Reframe dynamique basé sur détection IA optimisée"""

        logger.info("🎯 Reframe dynamique avec IA (optimisé)")

        print("    🎯 Détection IA en cours...")

        video = VideoFileClip(str(clip_path))

        fps = int(video.fps)

        duration = video.duration

        # Détection des centres d'intérêt

        x_centers = self._detect_focus_points(video, fps, duration)

        x_centers_smooth = self._smooth_trajectory(x_centers, window_size=min(15, max(5, len(x_centers)//4)))

        frame_index = 0

        applied_x_center_px = None

        beta = 0.85  # amortissement (0.85 = très smooth)

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

            # Clamp vitesse de déplacement + deadband

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

            

            # 🚨 CORRECTION BUG: Forcer des dimensions paires pour H.264

            target_width = Config.TARGET_WIDTH

            target_height = Config.TARGET_HEIGHT

            

            # Calcul du crop avec ratio 9:16

            crop_width = int(target_width * h / target_height)

            crop_width = min(crop_width, w)

            

            # 🚨 CORRECTION: S'assurer que crop_width est pair

            if crop_width % 2 != 0:

                crop_width = crop_width - 1 if crop_width > 1 else crop_width + 1

            

            x1 = int(max(0, min(w - crop_width, applied_x_center_px - crop_width / 2)))

            x2 = x1 + crop_width

            cropped = frame[:, x1:x2]

            

            # 🚨 CORRECTION: S'assurer que les dimensions finales sont paires

            final_width = target_width

            final_height = target_height

            

            # Vérifier et corriger si nécessaire

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

        print("    ✅ Reframe terminé")

        return output_path

    

    def transcribe_audio(self, video_path: Path) -> str:

        """Transcription avec Whisper"""

        logger.info("📝 Transcription audio avec Whisper")

        print("    📝 Transcription Whisper en cours...")

        

        result = self.whisper_model.transcribe(str(video_path))

        print("    ✅ Transcription terminée")

        return result["text"]

    

    def transcribe_segments(self, video_path: Path) -> List[Dict]:

        """

        Transcrit l'audio en segments avec timestamps (sans rendu visuel).

        Retourne une liste de segments {'text', 'start', 'end'} et conserve les mots si fournis.

        """

        logger.info("⏱️ Transcription avec timestamps")

        print("    ⏱️ Génération des timestamps...")

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

        print(f"    ✅ {len(subtitles)} segments de sous-titres générés")

        return subtitles

    def generate_caption_and_hashtags(self, subtitles: List[Dict]) -> (str, str, List[str], List[str]):

        """Génère une légende, des hashtags et des mots-clés B-roll avec le système LLM industriel."""

        full_text = ' '.join(s.get('text', '') for s in subtitles)

        

        # 🚀 NOUVEAU: Utilisation du système LLM industriel

        try:

            # Import du nouveau système

            import sys

            from pathlib import Path

            sys.path.insert(0, str(Path(__file__).parent / "utils"))

            

            from pipeline_integration import create_pipeline_integration

            

            # Créer l'intégration LLM

            llm_integration = create_pipeline_integration()

            

            print(f"    🚀 [LLM INDUSTRIEL] Génération de métadonnées pour {len(full_text)} caractères")

            

            # Traitement avec le nouveau système

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

                

                print(f"    ✅ [LLM INDUSTRIEL] Métadonnées générées avec succès")

                print(f"    🎯 Titre: {title}")

                print(f"    📝 Description: {description[:100]}...")

                print(f"    #️⃣ Hashtags: {len(hashtags)} générés")

                print(f"    🎬 Mots-clés B-roll: {len(broll_keywords)} termes optimisés")

                

                return title, description, hashtags, broll_keywords

            else:

                print(f"    ⚠️ [LLM INDUSTRIEL] Échec, fallback vers ancien système")

                raise Exception("LLM industriel échoué")

                

        except Exception as e:

            print(f"    🔄 [FALLBACK] Retour vers ancien système: {e}")

            # Fallback vers l'ancien système

            llm_res = _llm_generate_caption_hashtags(full_text)

            if llm_res and (llm_res.get('title') or llm_res.get('description') or llm_res.get('hashtags')):

                title = (llm_res.get('title') or '').strip()

                description = (llm_res.get('description') or '').strip()

                hashtags = [h for h in (llm_res.get('hashtags') or []) if h]

                

                # 🚀 NOUVEAU: Extraction des mots-clés B-roll du LLM

                broll_keywords = llm_res.get('broll_keywords', [])

                if broll_keywords:

                    print(f"    🤖 [LLM] Titre/description/hashtags + {len(broll_keywords)} mots-clés B-roll générés par LLM local")

                    print(f"    🎯 Mots-clés B-roll LLM: {', '.join(broll_keywords[:8])}...")

                else:

                    print("    🤖 [LLM] Titre/description/hashtags générés par LLM local")

                    # Fallback: extraire des mots-clés basiques du titre et de la description

                    fallback_text = f"{title} {description}".lower()

                    broll_keywords = [word for word in fallback_text.split() if len(word) > 3 and word.isalpha()]

                    broll_keywords = list(set(broll_keywords))[:10]

                    print(f"    🔄 Fallback mots-clés B-roll: {', '.join(broll_keywords[:5])}...")

                

                # Back-compat: si titre vide mais description présente, promouvoir description en titre court

                if not title and description:

                    title = (description[:60] + ('…' if len(description) > 60 else ''))

                return title, description, hashtags, broll_keywords

        

        # Fallback heuristic

        words = [w.strip().lower() for w in re.split(r"[^a-zA-Z0-9éèàùçêîôâ]+", full_text) if len(w) > 2]

        from collections import Counter

        counts = Counter(words)

        common = [w for w,_ in counts.most_common(12) if w.isalpha()]

        hashtags = [f"#{w}" for w in common[:12]]

        

        # 🚀 NOUVEAU: Mots-clés B-roll de fallback basés sur les mots communs

        broll_keywords = [w for w in common if len(w) > 3][:15]

        

        # Heuristic title/description

        title = (full_text.strip()[:60] + ("…" if len(full_text.strip()) > 60 else "")) if full_text.strip() else ""

        description = (full_text.strip()[:180] + ("…" if len(full_text.strip()) > 180 else "")) if full_text.strip() else ""

        print("    🧩 [Heuristics] Meta générées en fallback")

        print(f"    🔑 Mots-clés B-roll fallback: {', '.join(broll_keywords[:5])}...")

        return title, description, hashtags, broll_keywords

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

            event_logger = self._get_broll_event_logger()

            fetched_brolls: List[Dict[str, Any]] = []

            # --- Build dynamic LLM context once per clip (no hardcoded domains)

            selector_keywords: List[str] = []

            fetch_keywords: List[str] = []

            dyn_context: Dict[str, Any] = {}

            try:

                transcript_text_full = " ".join(str(s.get("text", "")) for s in (subtitles or []))

            except Exception:

                transcript_text_full = ""

            if ENABLE_DYNAMIC_CONTEXT and getattr(self, "_llm_service", None):

                try:

                    dyn_context = self._llm_service.generate_dynamic_context(transcript_text_full)

                except DynamicCompletionError as exc:

                    dyn_context = exc.payload or {}

                except Exception:

                    dyn_context = {}

            llm_kw = list(dyn_context.get('keywords') or [])

            llm_queries = list(dyn_context.get('search_queries') or [])

            syn_map = dyn_context.get('synonyms', {}) or {}

            seg_queries: List[str] = []

            for br in (dyn_context.get('segment_briefs') or []):

                if not isinstance(br, dict):

                    continue

                seg_queries.extend(br.get('queries') or [])

            synonyms_flat = [v for vs in (syn_map.values() if isinstance(syn_map, dict) else []) for v in (vs or [])]

            llm_terms = _dedupe_queries(llm_kw + llm_queries + synonyms_flat + seg_queries, cap=12)

            if not llm_terms:

                try:

                    transcript_tokens = [

                        w

                        for s in (subtitles or [])

                        for w in str(s.get('text', '')).split()

                        if isinstance(w, str)

                    ]

                except Exception:

                    transcript_tokens = []

                llm_terms = _dedupe_queries(transcript_tokens, cap=8)

            selector_keywords = llm_terms[:8]

            fetch_keywords = llm_terms[:8]

            # Ensure bilingual queries when original language isn't English, if LLM provided EN

            try:

                lang = str(dyn_context.get("language") or "").strip().lower()

            except Exception:

                lang = ""

            def _is_english(q: str) -> bool:

                try:

                    q2 = q.strip()

                    return bool(q2) and all((ord(c) < 128 and (c.isalnum() or c.isspace() or c in "-_") ) for c in q2) and any(c.isalpha() for c in q2)

                except Exception:

                    return False

            if lang and lang != "en":

                # If no English queries made it into fetch_keywords, try to inject 1-2 EN ones from LLM output

                has_en = any(_is_english(q) for q in fetch_keywords)

                if not has_en:

                    en_from_llm = [q for q in (dyn_context.get("search_queries") or []) if isinstance(q, str) and _is_english(q)]

                    if en_from_llm:

                        fetch_keywords = _dedupe_queries(list(fetch_keywords) + en_from_llm[:2], cap=8)

            try:

                dom_names = [d.get("name") for d in (dyn_context.get("detected_domains", []) or []) if isinstance(d, dict)]

            except Exception:

                dom_names = []

            print(f"Detected domains (free): {dom_names}")

            print(f"Selector keywords: {selector_keywords}")

            print(f"Fetch keywords: {fetch_keywords}")

            # Expose dynamic context to pipeline_core segment loop

            try:

                self._dyn_context = dyn_context

            except Exception:

                pass

            # Expose keyword lists for downstream reporting

            try:

                self._selector_keywords = list(selector_keywords)

                self._fetch_keywords = list(fetch_keywords)

            except Exception:

                pass

            # Observability: log dynamic context summary if logger available

            try:

                event_logger = self._get_broll_event_logger()

                if event_logger:

                    segment_queries_payload = []

                    for br in dyn_context.get('segment_briefs') or []:

                        if isinstance(br, dict):

                            segment_queries_payload.append(br.get('queries') or [])

                    event_logger.log({

                        'event': 'broll_dynamic_context',

                        'detected_domains': dyn_context.get('detected_domains', []),

                        'language': dyn_context.get('language'),

                        'keywords': llm_kw,

                        'search_queries': llm_queries,

                        'segment_queries': segment_queries_payload,

                        'selector_keywords': selector_keywords,

                        'fetch_keywords': fetch_keywords,

                    })

            except Exception:

                pass

            

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

                    

            # Fallback: ancienne analyse si système intelligent indisponible

            if not INTELLIGENT_BROLL_AVAILABLE:

                print("    🔄 Utilisation de l'ancien système B-roll...")

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

                    # Fallback vers prompts génériques

                    analysis = extract_keywords_from_transcript_ai(subtitles)

                    prompts = generate_broll_prompts_ai(analysis)

            

            # 🚀 NOUVEAU: Intégration des mots-clés B-roll du LLM

            # Récupérer les mots-clés B-roll générés par le LLM (si disponibles)

            llm_broll_keywords = []

            try:

                # Les mots-clés B-roll sont déjà disponibles depuis generate_caption_and_hashtags

                # Ils sont passés via la variable broll_keywords dans le scope parent

                if 'broll_keywords' in locals():

                    llm_broll_keywords = broll_keywords

                    print(f"    🧠 Mots-clés B-roll LLM intégrés: {len(llm_broll_keywords)} termes")

                    print(f"    🎯 Exemples: {', '.join(llm_broll_keywords[:5])}")

                else:

                    print("    ⚠️ Mots-clés B-roll LLM non disponibles")

            except Exception as e:

                print(f"    ⚠️ Erreur récupération mots-clés B-roll LLM: {e}")

            

            # Combiner les mots-clés LLM avec les prompts existants

            if llm_broll_keywords:

                # Enrichir les prompts avec les mots-clés LLM

                enhanced_prompts = []

                for kw in llm_broll_keywords[:8]:  # Limiter à 8 mots-clés principaux

                    enhanced_prompts.append(kw)

                    # Créer des combinaisons avec le thème principal

                    if 'global_analysis' in locals() and hasattr(global_analysis, 'main_theme'):

                        enhanced_prompts.append(f"{global_analysis.main_theme} {kw}")

                

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

            core_path = self._maybe_use_pipeline_core(

                segments,

                broll_keywords,

                subtitles=subtitles,

                input_path=input_path,

            )

            if core_path is not None:

                return core_path

            # Construire la config du pipeline (fetch + embeddings activés, pas de limites)

            raw_fetch_provider = getattr(Config, 'BROLL_FETCH_PROVIDER', 'pexels')

            fetch_provider_names = _normalize_provider_list(raw_fetch_provider)

            if not fetch_provider_names:

                fetch_provider_names = ['pexels']

            fetch_provider_value = ','.join(fetch_provider_names)

            cfg = BrollConfig(

                input_video=str(input_path),

                output_video=output_with_broll,

                broll_library=broll_library,

                srt_path=None,

                render_subtitles=False,

                            max_broll_ratio=0.65,           # CORRIGÉ: 90% → 65% pour équilibre optimal

            min_gap_between_broll_s=1.5,    # CORRIGÉ: 0.2s → 1.5s pour respiration visuelle

                            max_broll_clip_s=4.0,           # CORRIGÉ: 8.0s → 4.0s pour B-rolls équilibrés

            min_broll_clip_s=2.0,           # CORRIGÉ: 3.5s → 2.0s pour durée optimale

                use_whisper=False,

                ffmpeg_preset="fast",

                crf=23,

                threads=0,

                # Fetchers (stock)

                enable_fetcher=getattr(Config, 'BROLL_FETCH_ENABLE', False),

                fetch_provider=fetch_provider_value,

                pexels_api_key=getattr(Config, 'PEXELS_API_KEY', None),

                pixabay_api_key=getattr(Config, 'PIXABAY_API_KEY', None),

                fetch_max_per_keyword=getattr(Config, 'BROLL_FETCH_MAX_PER_KEYWORD', 25),  # CORRIGÉ: 50 → 25 pour qualité optimale

                fetch_allow_videos=getattr(Config, 'BROLL_FETCH_ALLOW_VIDEOS', True),

                fetch_allow_images=getattr(Config, 'BROLL_FETCH_ALLOW_IMAGES', True),  # Activé: images animées + Ken Burns

                # Embeddings

                use_embeddings=getattr(Config, 'BROLL_USE_EMBEDDINGS', True),

                embedding_model_name=getattr(Config, 'BROLL_EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),

                contextual_config_path=getattr(Config, 'CONTEXTUAL_CONFIG_PATH', Path('config/contextual_broll.yml')),

                # Experimental FX toggle

                enable_experimental_fx=getattr(Config, 'ENABLE_EXPERIMENTAL_FX', False),

            )

            schedule_drop_events: List[Dict[str, Any]] = []

            # FETCH DYNAMIQUE PAR CLIP: Créer un dossier unique et forcer le fetch à chaque fois

            try:

                from src.pipeline.fetchers import ensure_assets_for_keywords  # type: ignore

                

                # Créer un dossier unique pour ce clip (éviter le partage entre clips)

                clip_id = input_path.stem  # Nom du fichier sans extension

                clip_broll_dir = broll_library / f"clip_{clip_id}_{int(time.time())}"

                clip_broll_dir.mkdir(parents=True, exist_ok=True)

                

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

                        

                        # Analyser le contexte pour la sélection intelligente

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

                        

                        # Détecter le domaine

                        detected_domain = None

                        if 'global_analysis' in locals() and hasattr(global_analysis, 'main_theme'):

                            detected_domain = global_analysis.main_theme

                        

                        print(f"    🎯 Contexte: {detected_domain or 'général'}")

                        print(f"    🔑 Mots-clés contextuels: {', '.join(context_keywords[:5])}")

                        

                        # Utiliser le sélecteur pour la planification

                        # Domaine effectif pour le sélecteur (dyn > global)

                        dyn_ctx = getattr(self, '_dyn_context', None)

                        dyn_dom_name, dyn_dom_conf = (None, None)

                        if ENABLE_SELECTOR_DYNAMIC_DOMAIN and dyn_ctx:

                            dyn_dom_name, dyn_dom_conf = _choose_dynamic_domain(dyn_ctx)

                        effective_domain = dyn_dom_name or detected_domain

                        source = 'dyn' if dyn_dom_name else ('global' if detected_domain else 'none')

                        try:

                            print(f"    [SEL] domain={effective_domain or 'None'} source={source} conf={dyn_dom_conf if dyn_dom_conf is not None else 'n/a'}")

                        except Exception:

                            pass

                        try:

                            ev = self._get_broll_event_logger()

                            if ev:

                                ev.log({'event': 'broll_selector_domain', 'domain': effective_domain, 'source': source, 'confidence': dyn_dom_conf})

                        except Exception:

                            pass

                        try:

                            selection_report = broll_selector.select_brolls(

                                keywords=selector_keywords,

                                domain=effective_domain,

                                min_delay=self._load_broll_selector_config().get('thresholds', {}).get('min_delay_seconds', 4.0),

                                desired_count=self._load_broll_selector_config().get('desired_broll_count', 3)

                            )

                        except TypeError:

                            selection_report = broll_selector.select_brolls(

                                keywords=selector_keywords,

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

                if 'broll_keywords' in locals() and broll_keywords:

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

                # Provider selection limited to Pexels/Pixabay

                import os as _os

                pex = getattr(Config, 'PEXELS_API_KEY', None) or _os.getenv('PEXELS_API_KEY')

                pixa = getattr(Config, 'PIXABAY_API_KEY', None) or _os.getenv('PIXABAY_API_KEY')

                providers = []

                if pex:

                    providers.append('pexels')

                if pixa:

                    providers.append('pixabay')

                if providers:

                    try:

                        setattr(cfg, 'fetch_provider', ','.join(providers))

                        print(f"    Providers activés: {','.join(providers)}")

                    except Exception:

                        pass

                else:

                    print('    Providers: aucun (clés Pexels/Pixabay manquantes)')

                try:

                    setattr(cfg, 'fetch_allow_images', False)

                    setattr(cfg, 'fetch_allow_videos', True)

                    setattr(cfg, 'fetch_max_per_keyword', 12)

                except Exception:

                    pass

                ensure_assets_for_keywords(cfg, fetch_keywords, top_kws)

                

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

            keywords = fetch_keywords or selector_keywords or keywords

            

            try:

                plan = plan_broll_insertions(segments, cfg, index_handle)  # type: ignore[arg-type]

                plan, schedule_drops = enforce_broll_schedule_rules(plan)

                if schedule_drops:

                    schedule_drop_events.extend(schedule_drops)

                if schedule_drops and event_logger:

                    for drop in schedule_drops:

                        try:

                            event_logger.log({

                                'event': 'broll_plan_item_dropped',

                                **drop,

                            })

                        except Exception:

                            pass

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

                        

                        seg_keywords.append(cleaned[:15])  # Augmenté: 12 → 15

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

                plan, schedule_drops = enforce_broll_schedule_rules(plan)

                if schedule_drops:

                    schedule_drop_events.extend(schedule_drops)

                if schedule_drops and event_logger:

                    for drop in schedule_drops:

                        try:

                            event_logger.log({

                                'event': 'broll_plan_item_dropped',

                                **drop,

                            })

                        except Exception:

                            pass

                

            duplicate_drops: List[Dict[str, Any]] = []

            if plan:

                filtered_plan = []

                seen_asset_paths: Set[str] = set()

                for item in plan:

                    if hasattr(item, 'asset_path'):

                        asset_path = getattr(item, 'asset_path', None)

                    elif isinstance(item, dict):

                        asset_path = item.get('asset_path')

                    else:

                        asset_path = None

                    normalized_path = str(asset_path) if asset_path else None

                    if normalized_path:

                        if normalized_path in seen_asset_paths:

                            drop_entry: Dict[str, Any] = {'reason': 'duplicate_asset', 'asset_path': normalized_path}

                            start_val = getattr(item, 'start', None) if hasattr(item, 'start') else (item.get('start') if isinstance(item, dict) else None)

                            end_val = getattr(item, 'end', None) if hasattr(item, 'end') else (item.get('end') if isinstance(item, dict) else None)

                            if start_val is not None:

                                try:

                                    drop_entry['start'] = float(start_val)

                                except Exception:

                                    pass

                            if end_val is not None:

                                try:

                                    drop_entry['end'] = float(end_val)

                                except Exception:

                                    pass

                            duplicate_drops.append(drop_entry)

                            continue

                        seen_asset_paths.add(normalized_path)

                    filtered_plan.append(item)

                plan = filtered_plan

            if duplicate_drops:

                schedule_drop_events.extend(duplicate_drops)

                if event_logger:

                    for drop in duplicate_drops:

                        try:

                            event_logger.log({'event': 'broll_plan_item_dropped', **drop})

                        except Exception:

                            pass

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

                if plan and 'global_analysis' in locals() and hasattr(global_analysis, 'main_theme') and hasattr(global_analysis, 'keywords'):

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

                        

                        # Utiliser les B-rolls fetchés pour assigner aux items

                        available_assets = [broll.get('path', '') for broll in fetched_brolls if broll.get('path')]

                        

                        for i, item in enumerate(items_without_assets):

                            if i < len(available_assets):

                                asset_path = available_assets[i]

                                if hasattr(item, 'asset_path'):

                                    item.asset_path = asset_path

                                elif isinstance(item, dict):

                                    item['asset_path'] = asset_path

                                

                                print(f"    ✅ Asset assigné à item {i+1}: {Path(asset_path).name}")

                            else:

                                break

                        

                        # Recalculer les listes après assignation

                        items_with_assets = [item for item in plan if (hasattr(item, 'asset_path') and item.asset_path) or (isinstance(item, dict) and item.get('asset_path'))]

                        items_without_assets = [item for item in plan if not ((hasattr(item, 'asset_path') and item.asset_path) or (isinstance(item, dict) and item.get('asset_path')))]

                    

                    # 🚨 FALLBACK UNIQUEMENT SI VRAIMENT NÉCESSAIRE

                    if not items_with_assets and items_without_assets:

                        print(f"    ⚠️  Aucun asset disponible, activation du fallback neutre")

                        fallback_assets = _get_fallback_neutral_assets(broll_library, count=3)

                        if fallback_assets:

                            print(f"    🆘 Fallback neutre: {len(fallback_assets)} assets génériques utilisés")

                            # Créer des items de plan avec les assets de fallback

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

                            print(f"    🚨 Aucun asset de fallback disponible")

                    elif items_with_assets:

                        print(f"    ✅ {len(items_with_assets)} items avec assets assignés - Pas de fallback nécessaire")

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

                        # 🚨 NOUVEAU: Fallback contextuel intelligent au lieu d'aléatoire

                        print(f"    🔍 Fallback contextuel pour segment: {local[:50]}...")

                        for p in clip_specific_dir.rglob('*'):

                            if p.suffix.lower() in {'.mp4','.mov','.mkv','.webm','.jpg','.jpeg','.png'}:

                                if str(p.resolve()) not in used_recent and p.exists():

                                    # 🚨 NOUVEAU: Évaluation contextuelle du fallback

                                    if 'get_contextual_broll_score' in globals() and local_keywords:

                                        try:

                                            asset_name = p.stem.lower()

                                            asset_tokens = asset_name.split('_')

                                            asset_tags = asset_name.split('_')

                                            fallback_score = get_contextual_broll_score(local_keywords, asset_tokens, asset_tags)

                                            if fallback_score > 2.0:  # Seuil contextuel minimum

                                                chosen = str(p.resolve())

                                                print(f"    ✅ Fallback contextuel: {p.stem} | Score: {fallback_score:.2f}")

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

            # Vérification des asset_path avant normalisation + mini fallback non invasif

            try:

                def _get_ap(x):

                    return (getattr(x, 'asset_path', None) if hasattr(x, 'asset_path') else (x.get('asset_path') if isinstance(x, dict) else None))

                missing = [it for it in (plan or []) if not _get_ap(it)]

                if plan and len(missing) == len(plan):

                    # Aucun asset assigné par FAISS → mini fallback d'assignation séquentielle

                    # UTILISER LE DOSSIER SPÉCIFIQUE DU CLIP

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

                    assets = [p for p in broll_library.rglob('*') if p.suffix.lower() in _media_exts]

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

            # Capturer la liste des assets rendus avant le nettoyage

            rendered_asset_paths: List[str] = []

            for ev in valid_events:

                media_path = getattr(ev, 'media_path', None)

                if media_path is None and isinstance(ev, dict):

                    media_path = ev.get('media_path')

                if media_path:

                    rendered_asset_paths.append(str(media_path))

            seen_assets: Set[str] = set()

            rendered_asset_paths = [p for p in rendered_asset_paths if not (p in seen_assets or seen_assets.add(p))]

            rendered_event_count = len(valid_events)

            selected_assets_snapshot: List[str] = list(rendered_asset_paths)

            pipeline_result_ref = getattr(self, '_last_pipeline_result', None)

            if not isinstance(pipeline_result_ref, PipelineResult):

                pipeline_result_ref = None

            if pipeline_result_ref is not None:

                pipeline_result_ref.broll_inserted_count = len(selected_assets_snapshot)

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

            output_path = Path(cfg.output_video)

            selected_assets: List[str] = list(selected_assets_snapshot) if 'selected_assets_snapshot' in locals() else []

            rendered_events = rendered_event_count if 'rendered_event_count' in locals() else 0

            drop_counts = Counter()

            try:

                drop_counts = Counter(drop.get('reason', 'unknown') for drop in schedule_drop_events if isinstance(drop, dict))

            except Exception:

                drop_counts = Counter()

            if output_path.exists() and selected_assets and rendered_events:

                if pipeline_result_ref is not None:

                    pipeline_result_ref.broll_legacy_ok = True

                print("    [OK] B-roll inseres avec succes")

                try:

                    event_logger.log({

                        'event': 'broll_compositor_success',

                        'source': 'legacy',

                        'output': str(output_path),

                        'selected_count': len(selected_assets),

                        'rendered_events': rendered_events,

                    'drops_by_reason': dict(drop_counts),

                        'drops_by_reason': dict(drop_counts),

                    })

                except Exception:

                    pass

                return output_path

            reason = 'no_selected_assets' if not selected_assets or not rendered_events else 'output_missing'

            if pipeline_result_ref is not None:

                pipeline_result_ref.broll_inserted_count = 0

                pipeline_result_ref.broll_legacy_ok = False

            print(f"    [WARN] B-roll non inseres ({reason}) - retour a la video d'origine")

            try:

                event_logger.log({

                    'event': 'broll_compositor_failed',

                    'source': 'legacy',

                    'reason': reason,

                    'output': str(output_path),

                    'selected_count': len(selected_assets),

                    'rendered_events': rendered_events,

                })

            except Exception:

                pass

            return input_path

        except Exception as e:

            print(f"    ❌ Erreur B-roll: {e}")

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

import argparse

import os

import time

from pathlib import Path

from typing import Optional, Sequence

def main(argv: Optional[Sequence[str]] = None, *, return_result: bool = False) -> Union[int, PipelineResult]:

    parser = argparse.ArgumentParser(description='Run the video pipeline on a single clip.')

    parser.add_argument('--video', required=True, help='Chemin du clip source (mp4, mov, etc.)')

    parser.add_argument('--verbose', action='store_true', help='Affiche des informations supplementaires pendant le run.')

    parser.add_argument('--no-report', action='store_true', help='Disable selection report JSON sidecar')

    parser.add_argument('--no-emoji', action='store_true', help='Desactive les emojis dans les logs console.')

    args = parser.parse_args(list(argv) if argv is not None else None)

    print(f"[CLI] cwd={os.getcwd()}")

    print(f"[CLI] video={args.video}")

    print(f"[CLI] ENABLE_PIPELINE_CORE_FETCHER={os.getenv('ENABLE_PIPELINE_CORE_FETCHER')}")

    if getattr(args, 'no_report', False):

        os.environ['ENABLE_SELECTION_REPORT'] = 'false'

        print('[CLI] selection report disabled')

    global logger

    logger = get_console_logger(no_emoji=args.no_emoji, name=__name__)

    start_time = time.time()

    processor = VideoProcessor()

    pipeline_result: Optional[PipelineResult] = None

    try:

        pipeline_result = processor.process_single_clip(Path(args.video), verbose=args.verbose)

    except Exception as exc:

        pipeline_result = getattr(processor, '_last_pipeline_result', None)

        if pipeline_result is None:

            pipeline_result = PipelineResult()

            pipeline_result.errors.append(str(exc))

            pipeline_result.final_export_ok = False

            pipeline_result.finish()

        if return_result:

            return pipeline_result

        raise

    elapsed = time.time() - start_time

    final_path_display = pipeline_result.final_export_path if pipeline_result.final_export_path else 'None'

    print(f"[CLI] Done in {elapsed:.1f}s -> {final_path_display}")

    if return_result:

        return pipeline_result

    return 0 if pipeline_result.final_export_ok else 1

if __name__ == '__main__':

    raise SystemExit(main())
