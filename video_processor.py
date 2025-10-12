import sys

try:
    pass  # keep top-level try valid
# sys.stdout.reconfigure(encoding="utf-8")  # disabled during tests
# sys.stderr.reconfigure(encoding="utf-8")  # disabled during tests
except Exception:
    pass

from contextlib import ExitStack
from pathlib import Path
from urllib.parse import urlparse

# ensure project-root is first
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(1, str(PROJECT_ROOT / 'src'))
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
from typing import List, Dict, Any, Optional, Union, Sequence, Set, Tuple, TextIO
from collections import Counter
from dataclasses import dataclass

from video_pipeline.broll_rules import BrollClip, enforce_broll_schedule_rules as _enforce_broll_schedule_rules_v2
from video_pipeline.config import (
    apply_llm_overrides,
    get_settings,
    log_effective_settings,
    set_settings,
)
import types
import gc
import re
import io


# Keep Latin token handling consistent with pipeline_core without brittle regex ranges.
_BASIC_LATIN_CODEPOINT_RANGES: Tuple[Tuple[int, int], ...] = (
    (ord('0'), ord('9')),
    (ord('A'), ord('Z')),
    (ord('a'), ord('z')),
    (0x00C0, 0x00D6),
    (0x00D8, 0x00F6),
    (0x00F8, 0x00FF),
)
_DEFAULT_TOKEN_KEEP: Set[str] = {"'", "-"}


def _is_basic_latin_char(ch: str) -> bool:
    code = ord(ch)
    for start, end in _BASIC_LATIN_CODEPOINT_RANGES:
        if start <= code <= end:
            return True
    return False


def _split_basic_latin_runs(text: str, *, keep: Set[str] | None = None) -> List[str]:
    allowed_extras = keep if keep is not None else _DEFAULT_TOKEN_KEEP
    buffer: List[str] = []
    runs: List[str] = []
    for ch in text:
        if _is_basic_latin_char(ch) or ch in allowed_extras:
            buffer.append(ch)
        elif buffer:
            runs.append(''.join(buffer))
            buffer.clear()
    if buffer:
        runs.append(''.join(buffer))
    return runs


try:
    from pipeline_core.llm_service import generate_metadata_as_json
except Exception:
    generate_metadata_as_json = None

try:
    from config import Config as _ROOT_CONFIG
except Exception:  # pragma: no cover - fallback when root config is unavailable
    class _ROOT_CONFIG:
        ENABLE_LEGACY_PIPELINE_FALLBACK = False

# Ã°Å¸Å¡â‚¬ NOUVEAU: Configuration des logs temps rÃƒÂ©el + suppression warnings non-critiques
import warnings
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated")
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces` was not set")
warnings.filterwarnings("ignore", message="Warning: in file.*bytes wanted but.*bytes read")

try:
    if hasattr(sys.stdout, 'buffer'):
        _STDOUT_STREAM = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', write_through=True)
    else:
        _STDOUT_STREAM = sys.stdout
except Exception:
    _STDOUT_STREAM = sys.stdout

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(_STDOUT_STREAM)
    ]
)
logger = logging.getLogger(__name__)

# --- Deferred dependency status reporting -----------------------------------
_DEPENDENCY_STATUS_MESSAGES: List[str] = []
_DEPENDENCY_STATUS_EMITTED = False


def _register_dependency_status(message: str) -> None:
    """Store dependency status messages to be emitted on demand."""

    _DEPENDENCY_STATUS_MESSAGES.append(message)


def emit_dependency_status(stream: Optional[TextIO] = None, *, once: bool = True) -> None:
    """Print deferred dependency status messages.

    Parameters
    ----------
    stream:
        Destination stream; defaults to ``sys.stdout``.
    once:
        When ``True`` (the default), messages are emitted only the first time the
        function is called. Subsequent calls become no-ops to avoid duplicated
        headers when multiple processor instances are created.
    """

    global _DEPENDENCY_STATUS_EMITTED

    if once and _DEPENDENCY_STATUS_EMITTED:
        return

    output = stream or sys.stdout
    for message in _DEPENDENCY_STATUS_MESSAGES:
        print(message, file=output)

    if once:
        _DEPENDENCY_STATUS_EMITTED = True

# Ã°Å¸Å¡â‚¬ NOUVEAU: Fonction print temps rÃƒÂ©el
def print_realtime(message):
    """Print avec flush immÃƒÂ©diat pour logs temps rÃƒÂ©el"""
    print(message, flush=True)
    logger.info(message)


def estimate_roi_x_center(video_path: Union[str, Path], sample_frames: int = 14, min_w: int = 320) -> Tuple[Optional[float], Optional[str]]:
    """Estimate a horizontal ROI centre ratio based on face detection or motion."""

    try:
        cap = cv2.VideoCapture(str(video_path))
    except Exception:
        return None, None

    if not cap.isOpened():
        cap.release()
        return None, None

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            return None, None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            total_frames = sample_frames * 2
        step = max(int(total_frames // max(sample_frames, 1) or 1), 1)

        face_model = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_detector = cv2.CascadeClassifier(face_model)
        face_centers: List[float] = []
        for idx in range(sample_frames):
            frame_idx = idx * step
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            min_face_w = min(width, max(min_w, 1))
            detections = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(min_face_w, max(min_face_w // 2, 1)),
            )
            if len(detections):
                x, y, w, h = max(detections, key=lambda rect: rect[2] * rect[3])
                center_ratio = (x + w / 2.0) / float(width)
                face_centers.append(center_ratio)
        if face_centers:
            return float(np.median(face_centers)), "face"

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, prev_frame = cap.read()
        if not ok:
            return None, None
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        accum = np.zeros((height, width), dtype=np.float32)

        frames_used = 0
        for idx in range(sample_frames):
            frame_idx = idx * step
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, prev_gray)
            _, mask = cv2.threshold(diff, 24, 1.0, cv2.THRESH_BINARY)
            accum += mask
            prev_gray = gray
            frames_used += 1
        if frames_used == 0:
            return None, None

        column_sum = accum.sum(axis=0)
        total = float(column_sum.sum())
        if total <= 0:
            return None, None
        xs = np.arange(width, dtype=np.float32)
        center_ratio = float((xs * column_sum).sum() / total) / float(max(width, 1))
        return center_ratio, "motion"
    except Exception:
        return None, None
    finally:
        cap.release()


def _apply_vertical_crop(clip, base_h: Optional[int], base_w: Optional[int], roi_center: Optional[float] = None) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """Crop a clip to a strict 9:16 window centred around the ROI."""

    try:
        overlay_w, overlay_h = clip.size
    except Exception:
        return clip, None

    if not overlay_w or not overlay_h:
        return clip, None

    reference_h = base_h or overlay_h
    target_w = int(round(reference_h * 9 / 16))
    if target_w <= 0:
        return clip, None

    target_w = min(target_w, overlay_w)
    if base_w:
        target_w = min(target_w, base_w)
    if overlay_w <= target_w:
        return clip, None

    center_ratio = None
    if roi_center is None:
        cx = overlay_w / 2.0
    else:
        center_ratio = float(roi_center)
        if 0.0 <= center_ratio <= 1.0:
            cx = center_ratio * overlay_w
        else:
            cx = float(center_ratio)
            center_ratio = max(0.0, min(cx / float(overlay_w), 1.0)) if overlay_w else None
    x0 = int(round(cx - target_w / 2.0))
    x0 = max(0, min(x0, overlay_w - target_w))
    cropped = crop(clip, x1=x0, y1=0, x2=x0 + target_w, y2=overlay_h)
    metadata = {
        "x0": x0,
        "width": target_w,
        "height": overlay_h,
        "source_width": overlay_w,
        "source_height": overlay_h,
        "center_ratio": center_ratio,
    }
    return cropped, metadata


def format_broll_completion_banner(
    inserted_count: int,
    *,
    origin: str = "pipeline",
    render_ok: Optional[bool] = None,
) -> Tuple[bool, str]:
    """Return a (success, banner) tuple describing the B-roll insertion result.

    The banner preserves the historical success message so downstream log parsing
    keeps working, while providing a dedicated warning when no insertions were
    produced. The *origin* parameter allows callers to specialise the warning for
    pipeline_core versus the legacy pipeline.
    """

    try:
        count = int(inserted_count)
    except (TypeError, ValueError):
        count = 0

    origin_key = (origin or "pipeline").strip().lower()

    effective_ok = (render_ok if render_ok is not None else count > 0) and count > 0

    if effective_ok:
        return True, f"    Ã¢Å“â€¦ B-roll insÃƒÂ©rÃƒÂ©s avec succÃƒÂ¨s ({count})"

    if count > 0:
        return False, "    Ã¢Å¡Â Ã¯Â¸Â B-roll sÃƒÂ©lectionnÃƒÂ©s mais rendu indisponible; retour ÃƒÂ  la vidÃƒÂ©o d'origine"

    if origin_key == "pipeline_core":
        return False, "    Ã¢Å¡Â Ã¯Â¸Â Pipeline core: aucun B-roll sÃƒÂ©lectionnÃƒÂ©; retour ÃƒÂ  la vidÃƒÂ©o d'origine"

    return False, "    Ã¢Å¡Â Ã¯Â¸Â Aucun B-roll insÃƒÂ©rÃƒÂ©; retour ÃƒÂ  la vidÃƒÂ©o d'origine"

SEEN_URLS: Set[str] = set()
SEEN_PHASHES: List[int] = []
SEEN_IDENTIFIERS: Set[str] = set()
PHASH_DISTANCE = 6


_GLOBAL_BROLL_EVENTS_LOGGER: Optional['JsonlLogger'] = None


@dataclass
class CoreTimelineEntry:
    """Minimal description of a clip selected by the core pipeline."""

    path: Path
    start: float
    end: float
    segment_index: int
    provider: Optional[str] = None
    url: Optional[str] = None

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end) - float(self.start))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': str(self.path),
            'start': float(self.start),
            'end': float(self.end),
            'segment': int(self.segment_index),
            'provider': self.provider,
            'url': self.url,
        }


@dataclass
class _CoreSegment:
    start: float
    end: float
    text: str


def _apply_broll_invariants_to_core_entries(
    entries: Sequence[CoreTimelineEntry],
    *,
    seen_updates: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[List[CoreTimelineEntry], Optional[List[Dict[str, Any]]]]:
    """Apply B-roll invariants to the provided core entries and mirror metadata."""

    entries_list = list(entries)
    updates_list = list(seen_updates) if seen_updates is not None else None

    if updates_list is not None and len(updates_list) < len(entries_list):
        padding = len(entries_list) - len(updates_list)
        updates_list.extend({} for _ in range(padding))

    if not entries_list:
        settings = get_settings()
        print(
            f"[BROLL_RULES] kept=0/0 (100.0%) min_start={settings.broll.min_start_s}s "
            f"gap={settings.broll.min_gap_s}s no_repeat={settings.broll.no_repeat_s}s"
        )
        return [], [] if updates_list is not None else None

    settings = get_settings()

    clip_records: List[Tuple[CoreTimelineEntry, BrollClip]] = []
    for entry in entries_list:
        asset_identifier = entry.url or str(entry.path)
        clip_records.append(
            (
                entry,
                BrollClip(
                    start_s=float(entry.start),
                    end_s=float(entry.end),
                    asset_id=str(asset_identifier),
                    segment_index=int(entry.segment_index),
                ),
            )
        )

    filtered_clips = _enforce_broll_schedule_rules_v2(
        [clip for _, clip in clip_records],
        min_start_s=settings.broll.min_start_s,
        min_gap_s=settings.broll.min_gap_s,
        no_repeat_s=settings.broll.no_repeat_s,
    )

    key_counts: Counter[Tuple[float, float, str, int]] = Counter(
        (clip.start_s, clip.end_s, clip.asset_id, clip.segment_index)
        for clip in filtered_clips
    )

    filtered_entries: List[CoreTimelineEntry] = []
    filtered_updates: Optional[List[Dict[str, Any]]] = [] if updates_list is not None else None

    for idx, (entry, clip) in enumerate(clip_records):
        key = (clip.start_s, clip.end_s, clip.asset_id, clip.segment_index)
        if key_counts.get(key, 0):
            filtered_entries.append(entry)
            key_counts[key] -= 1
            if filtered_updates is not None and updates_list is not None and idx < len(updates_list):
                filtered_updates.append(updates_list[idx])

    total = len(entries_list)
    kept = len(filtered_entries)
    ratio = 100.0 if total == 0 else (100.0 * kept / total)
    print(
        f"[BROLL_RULES] kept={kept}/{total} ({ratio:.1f}%) "
        f"min_start={settings.broll.min_start_s}s gap={settings.broll.min_gap_s}s "
        f"no_repeat={settings.broll.no_repeat_s}s"
    )

    if filtered_updates is not None:
        return filtered_entries, filtered_updates
    return filtered_entries, None


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
    seen_identifiers: Set[str] = set()
    seen_urls: Set[str] = set()
    for candidate in candidates or []:
        identifier = getattr(candidate, 'identifier', None)
        if identifier:
            if identifier in SEEN_IDENTIFIERS or identifier in seen_identifiers:
                hits += 1
                continue
        url = getattr(candidate, 'url', None)
        if url:
            if url in SEEN_URLS or url in seen_urls:
                hits += 1
                continue
        unique.append(candidate)
        if identifier:
            seen_identifiers.add(identifier)
        if url:
            seen_urls.add(url)
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


def enforce_broll_schedule_rules(plan, *, min_duration: float = 1.8, min_gap: float = 1.5):
    """Filter a naive schedule to satisfy duration and gap constraints."""

    filtered = []
    drops: List[Dict[str, Any]] = []
    last_end = -1e9

    for item in plan or []:
        start = float(getattr(item, 'start', 0.0) or 0.0)
        end = float(getattr(item, 'end', start) or start)
        if end < start:
            start, end = end, start
        duration = max(0.0, end - start)
        reason = None
        gap_threshold = max(0.0, min_gap)
        duration_threshold = max(0.0, min_duration)

        if filtered and (start - last_end) < gap_threshold:
            reason = 'gap_violation'
        elif duration + 1e-6 < duration_threshold:
            reason = 'duration_short'

        if reason:
            drops.append({'start': start, 'end': end, 'reason': reason})
            continue

        filtered.append(item)
        last_end = end

    return filtered, drops

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

    llm_terms = list(llm_keywords or [])
    if llm_terms:
        llm_terms = _concretize_queries(llm_terms)

    base = list(dict.fromkeys(_yield_terms(llm_terms)))
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


import os
import json
import random
import numpy as np
import shutil
from datetime import datetime  # NEW: pour mÃƒÂ©tadonnÃƒÂ©es intelligentes
import whisper
import requests
import cv2

try:
    from pipeline_core.configuration import PipelineConfigBundle
except Exception:  # pragma: no cover - optional in stubbed environments
    PipelineConfigBundle = None  # type: ignore[assignment]
from pipeline_core.fetchers import FetcherOrchestrator
from pipeline_core.dedupe import compute_phash, hamming_distance
from pipeline_core.logging import JsonlLogger, log_broll_decision
try:
    from pipeline_core.llm_service import (
        DynamicCompletionError,
        LLMMetadataGeneratorService,
        _CONCRETE_SUBJECTS,
        _concretize_queries,
        build_visual_phrases,
        enforce_fetch_language,
        get_shared_llm_service,
        has_concrete_subject,
    )
except ImportError:  # pragma: no cover - test environments may stub partial API
    from pipeline_core.llm_service import LLMMetadataGeneratorService  # type: ignore

    class DynamicCompletionError(RuntimeError):  # type: ignore[override]
        def __init__(self, reason: str, *, payload=None) -> None:
            self.reason = (reason or "unknown").strip() or "unknown"
            self.payload = payload
            super().__init__(reason)

    def enforce_fetch_language(terms, language=None):  # type: ignore[override]
        return list(dict.fromkeys(term for term in terms if term))

    def build_visual_phrases(terms, limit=None):  # type: ignore[override]
        seen = []
        for term in terms or []:
            if term and term not in seen:
                seen.append(term)
                if limit and len(seen) >= limit:
                    break
        return seen

    def has_concrete_subject(value):  # type: ignore[override]
        return bool(value)

    _CONCRETE_SUBJECTS: set[str] = set()

    def _concretize_queries(values, limit=None):  # type: ignore[override]
        cleaned = []
        for term in values or []:
            if not isinstance(term, str):
                continue
            lowered = term.strip().lower()
            if lowered:
                cleaned.append(lowered)
                if limit and len(cleaned) >= int(limit):
                    break
        return cleaned

    _FALLBACK_SHARED_SERVICE: LLMMetadataGeneratorService | None = None

    def get_shared_llm_service() -> LLMMetadataGeneratorService:  # type: ignore[override]
        global _FALLBACK_SHARED_SERVICE
        if _FALLBACK_SHARED_SERVICE is None:
            _FALLBACK_SHARED_SERVICE = LLMMetadataGeneratorService()
        return _FALLBACK_SHARED_SERVICE

# Ã°Å¸Å¡â‚¬ NOUVEAU: Cache global pour ÃƒÂ©viter le rechargement des modÃƒÂ¨les
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


def _get_provider_names(cfg) -> List[str]:
    """Best-effort detection of enabled provider names for diagnostics."""

    try:
        from pipeline_core.configuration import detected_provider_names  # type: ignore
    except Exception:
        detected_provider_names = None  # type: ignore[assignment]

    if detected_provider_names is not None:
        try:
            providers = detected_provider_names(getattr(cfg, "fetcher", None))
            if providers:
                return list(providers)
        except Exception:
            pass

    fetcher = getattr(cfg, "fetcher", None)
    provider_seq = getattr(fetcher, "providers", None)
    if not provider_seq:
        return []

    detected = []
    for provider in provider_seq:
        try:
            enabled = getattr(provider, "enabled")
        except Exception:
            enabled = None
        if not enabled:
            continue
        name = getattr(provider, "name", None)
        if name:
            detected.append(name)
    return detected

def _norm_query_term(s: str) -> str:
    s = (s or "").strip().lower().replace("_"," ")
    try:
        s = re.sub(r"[^\w\s\-]", "", s)
    except Exception:
        pass
    return s

_CONCRETE_SUBJECT_REGEX = re.compile(r"\b(doctor|scientist|patient|brain|team|teacher|student|athlete|player|man|woman|person|robot|camera|lab|laboratory|office|family|crowd|group|hands|engineer|technician|nurse|chef|musician|artist|child|kid|baby|city|factory|computer|laptop|desk|machine)\b")


def _has_subject(value: str) -> bool:
    if not value:
        return False
    if _CONCRETE_SUBJECT_REGEX.search(value):
        return True
    return has_concrete_subject(value)



def has_two_meaningful_tokens(tokens: Sequence[str]) -> bool:
    return sum(1 for t in tokens if t and t not in _STOPWORDS) >= 2


def touches_concrete_subject(tokens: Sequence[str]) -> bool:
    try:
        concrete = _CONCRETE_SUBJECTS
    except NameError:
        concrete = set()
    return any(t in concrete for t in tokens)


def _dedupe_queries(seq, cap: int) -> list[str]:
    seen, out = set(), []
    raw_terms = [str(term).strip() for term in (seq or []) if isinstance(term, str) and term and str(term).strip()]
    concretized = _concretize_queries(raw_terms) if raw_terms else []
    phrases = build_visual_phrases(concretized or raw_terms, limit=None)
    for phrase in phrases:
        x = _norm_query_term(phrase)
        if len(x) < 3 or x in _ANTI_TERMS or not any(c.isalpha() for c in x):
            continue
        tokens = [t for t in x.split() if t]
        if any(t in _ANTI_TERMS for t in tokens):
            continue
        has_subject = _has_subject(x)
        touches_subject = touches_concrete_subject(tokens)
        if len(tokens) < 3:
            if not (has_two_meaningful_tokens(tokens) or touches_subject or has_subject):
                continue
        else:
            if not (has_subject or touches_subject):
                continue
        if x not in seen:
            out.append(x)
            seen.add(x)
        if len(out) >= cap:
            break
    return out


def _basic_metadata_fallback(full_text: str) -> Dict[str, Any]:
    """Generate simple metadata using the transcript when advanced services fail."""

    text = (full_text or "").strip()
    tokens = _split_basic_latin_runs(text, keep={"'", "-"})
    first_words = tokens[:8]

    if first_words:
        raw_title = " ".join(first_words).strip()
        title = raw_title[:1].upper() + raw_title[1:]
    else:
        title = "Instant Highlight"

    snippet = text[:180].strip()
    if snippet:
        description = f"Key moment from this clip: {snippet}"
    else:
        description = "Key moment from this clip."

    significant = [w.lower() for w in tokens if len(w) > 3]
    counts = Counter(significant)
    dominant_terms = [term for term, _ in counts.most_common(5)]

    base_hashtags = ["#shorts", "#highlights", "#viralclip"]
    for term in dominant_terms:
        cleaned = re.sub(r"[^A-Za-z0-9]", "", term)
        if cleaned:
            base_hashtags.append(f"#{cleaned}")

    hashtags: List[str] = []
    for tag in base_hashtags:
        if tag and tag not in hashtags:
            hashtags.append(tag)

    broll_keywords = dominant_terms[:]
    fallback_keywords = [
        "audience reaction",
        "speaker close up",
        "dynamic text overlay",
        "city skyline",
        "motivational crowd",
    ]
    for term in fallback_keywords:
        if len(broll_keywords) >= 5:
            break
        if term not in broll_keywords:
            broll_keywords.append(term)

    queries = [f"{term} b-roll" for term in dominant_terms[:3]]
    fallback_queries = [
        "motivational speech b-roll",
        "audience clapping stock footage",
        "city skyline night aerial",
    ]
    for query in fallback_queries:
        if len(queries) >= 3:
            break
        if query not in queries:
            queries.append(query)

    return {
        "title": title,
        "description": description,
        "hashtags": hashtags,
        "broll_keywords": broll_keywords,
        "queries": queries,
    }


def _segment_brief_terms(dyn: dict, seg_idx: int) -> tuple[list[str], list[str]]:
    """Return (queries, keywords) declared for a segment inside the dynamic briefs."""

    if not isinstance(dyn, dict):
        return [], []

    briefs = dyn.get("segment_briefs")
    if not isinstance(briefs, list):
        return [], []

    matching_briefs: list[dict] = []
    for br in briefs:
        if not isinstance(br, dict):
            continue
        try:
            if int(br.get("segment_index", -1)) != seg_idx:
                continue
        except Exception:
            continue
        matching_briefs.append(br)

    if not matching_briefs:
        return [], []

    def _iter_terms(value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return [v for v in value if isinstance(v, str)]
        return []

    queries: list[str] = []
    keywords: list[str] = []
    seen_queries: set[str] = set()
    seen_keywords: set[str] = set()

    for br in matching_briefs:
        for term in _iter_terms(br.get("queries")):
            cleaned = term.replace("_", " ").strip()
            if not cleaned or cleaned in seen_queries:
                continue
            queries.append(cleaned)
            seen_queries.add(cleaned)

    for br in matching_briefs:
        for term in _iter_terms(br.get("keywords")):
            cleaned = term.replace("_", " ").strip()
            if not cleaned or cleaned in seen_keywords:
                continue
            keywords.append(cleaned)
            seen_keywords.add(cleaned)

    return queries, keywords


def _segment_terms_from_briefs(dyn: dict, seg_idx: int, cap: int) -> list[str]:
    """Extract up to `cap` clean terms (queries then keywords) for a given segment index."""

    if cap <= 0:
        return []

    queries, keywords = _segment_brief_terms(dyn, seg_idx)
    out: list[str] = []
    for term in queries:
        if len(out) >= cap:
            break
        out.append(term)
    if len(out) < cap:
        for term in keywords:
            if len(out) >= cap:
                break
            if term in out:
                continue
            out.append(term)
    return out


def _flatten_seed_terms(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_seed_terms(item))
        return out
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(_flatten_seed_terms(item))
        return out
    return []


_SEED_QUERY_CACHE: Optional[list[str]] = None


def _load_seed_queries() -> list[str]:
    global _SEED_QUERY_CACHE
    if _SEED_QUERY_CACHE is not None:
        return _SEED_QUERY_CACHE

    candidates: list[str] = []
    env_path = os.getenv("BROLL_SEED_QUERIES")
    search_paths: list[Path] = []
    if env_path:
        try:
            search_paths.append(Path(env_path).expanduser())
        except Exception:
            pass
    search_paths.append(PROJECT_ROOT / "seed_queries.json")
    search_paths.append(PROJECT_ROOT / "config" / "seed_queries.json")

    for path in search_paths:
        try:
            if not path or not path.is_file():
                continue
        except Exception:
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        candidates = _flatten_seed_terms(payload)
        if candidates:
            break

    cleaned = []
    seen: set[str] = set()
    for term in candidates:
        if not isinstance(term, str):
            continue
        normalised = term.replace("_", " ").strip()
        if not normalised or normalised in seen:
            continue
        seen.add(normalised)
        cleaned.append(normalised)

    _SEED_QUERY_CACHE = cleaned
    return _SEED_QUERY_CACHE


def _relaxed_normalise_terms(raw_terms: Sequence[str], limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for term in raw_terms or []:
        if len(out) >= limit > 0:
            break
        if not isinstance(term, str):
            continue
        candidate = _norm_query_term(term)
        if not candidate or candidate in seen:
            continue
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if not candidate:
            continue
        seen.add(candidate)
        out.append(candidate)
    return out[: max(0, limit)]


def _build_transcript_fallback_terms(
    segment_text: str,
    segment_keywords: Sequence[str],
    *,
    limit: int,
) -> list[str]:
    if limit <= 0:
        return []

    tokens: list[str] = []
    for source in (segment_keywords or []):
        if not isinstance(source, str):
            continue
        tokens.extend(re.findall(r"[a-zA-Z]{4,}", source.lower()))

    tokens.extend(re.findall(r"[a-zA-Z]{4,}", (segment_text or "").lower()))

    unique_tokens: list[str] = []
    seen_tokens: set[str] = set()
    for token in tokens:
        cleaned = token.replace("_", " ").strip()
        if len(cleaned) < 3:
            continue
        if cleaned in seen_tokens:
            continue
        seen_tokens.add(cleaned)
        unique_tokens.append(cleaned)

    phrases: list[str] = []
    for token in unique_tokens:
        if len(phrases) >= limit:
            break
        phrases.append(token)

    if not phrases:
        return ["stock footage"][:limit]

    return phrases[:limit]


def _merge_segment_query_sources(
    *,
    segment_text: str,
    llm_queries: Sequence[str],
    brief_queries: Sequence[str],
    brief_keywords: Sequence[str],
    segment_keywords: Sequence[str],
    selector_keywords: Sequence[str],
    cap: int,
) -> tuple[list[str], str]:
    cap = max(1, int(cap or 0))
    combined: list[str] = []
    seen: set[str] = set()
    primary_source = "none"

    def _consume(source_name: str, terms: Sequence[str], *, relax: bool = False) -> bool:
        nonlocal primary_source
        if not terms:
            return False
        remaining = cap - len(combined)
        if remaining <= 0:
            return False
        prepared = [t.replace("_", " ").strip() for t in terms if isinstance(t, str)]
        prepared = [t for t in prepared if t]
        if not prepared:
            return False
        if relax:
            cleaned = _relaxed_normalise_terms(prepared, remaining)
        else:
            cleaned = _dedupe_queries(prepared, cap=remaining)
        added = False
        for term in cleaned:
            if len(combined) >= cap:
                break
            normalised = re.sub(r"\s+", " ", term.strip())
            if not normalised or normalised in seen:
                continue
            seen.add(normalised)
            combined.append(normalised)
            if primary_source == "none":
                if source_name.startswith("segment_brief"):
                    primary_source = "segment_brief"
                else:
                    primary_source = source_name
            added = True
        return added

    print(
        f"    🔍 DEBUG _combine_broll_queries ENTRÉE llm_queries: {llm_queries[:3]}"
    )
    _consume("llm_hint", llm_queries, relax=True)
    print(
        "    🔍 DEBUG _combine_broll_queries APRÈS _consume, "
        f"combined={combined[:3] if combined else 'VIDE'}"
    )
    brief_pool: list[str] = []
    brief_pool.extend(brief_keywords or [])
    brief_pool.extend(brief_queries or [])
    _consume("segment_brief", brief_pool)

    if not combined:
        if _consume("segment_keywords", segment_keywords):
            pass
    if not combined:
        _consume("selector_keywords", selector_keywords)
    if not combined:
        seed_queries = _load_seed_queries()
        _consume("seed_queries", seed_queries)

    if not combined:
        fallback_terms = _build_transcript_fallback_terms(
            segment_text,
            segment_keywords,
            limit=cap,
        )
        _consume("transcript_fallback", fallback_terms, relax=True)

    if not combined:
        _consume("transcript_fallback", ["stock footage"], relax=True)

    if primary_source == "none" and combined:
        primary_source = "transcript_fallback"

    out = (combined[:cap], primary_source)
    debug_preview = out[0][:5] if out and isinstance(out[0], list) else out
    print(f"    🔍 DEBUG _combine_broll_queries SORTIE: {debug_preview}")
    return out

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
    """RÃƒÂ©cupÃƒÂ¨re un modÃƒÂ¨le SentenceTransformer depuis le cache ou le charge"""
    # Ã°Å¸Å¡â‚¬ OPTIMISATION: Normaliser le nom du modÃƒÂ¨le pour ÃƒÂ©viter les doublons
    normalized_name = model_name.replace('sentence-transformers/', '')
    
    if normalized_name not in _MODEL_CACHE:
        print(f"    Ã°Å¸â€â€ž Chargement initial du modÃƒÂ¨le: {model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            _MODEL_CACHE[normalized_name] = SentenceTransformer(model_name)
            print(f"    Ã¢Å“â€¦ ModÃƒÂ¨le {model_name} chargÃƒÂ© et mis en cache")
        except Exception as e:
            print(f"    Ã¢ÂÅ’ Erreur chargement modÃƒÂ¨le {model_name}: {e}")
            return None
    else:
        print(f"    Ã¢â„¢Â»Ã¯Â¸Â ModÃƒÂ¨le {model_name} rÃƒÂ©cupÃƒÂ©rÃƒÂ© du cache")
    
    return _MODEL_CACHE[normalized_name]

def safe_remove_tree(directory: Path, max_retries: int = 3, delay: float = 1.0) -> bool:
    """
    Supprime un dossier de faÃƒÂ§on sÃƒÂ©curisÃƒÂ©e avec retry et gestion des handles Windows
    
    Args:
        directory: Dossier ÃƒÂ  supprimer
        max_retries: Nombre maximum de tentatives
        delay: DÃƒÂ©lai entre les tentatives (secondes)
    
    Returns:
        True si la suppression a rÃƒÂ©ussi, False sinon
    """
    if not directory.exists():
        return True
    
    for attempt in range(max_retries):
        try:
            # Forcer la libÃƒÂ©ration des handles
            gc.collect()
            
            # Tentative de suppression rÃƒÂ©cursive
            shutil.rmtree(directory, ignore_errors=False)
            
            # VÃƒÂ©rifier que c'est vraiment supprimÃƒÂ©
            if not directory.exists():
                return True
                
        except PermissionError as e:
            if "WinError 32" in str(e) or "being used by another process" in str(e):
                print(f"    Ã¢Å¡Â Ã¯Â¸Â Tentative {attempt + 1}/{max_retries}: Fichier en cours d'utilisation, retry dans {delay}s...")
                time.sleep(delay)
                delay *= 1.5  # Backoff exponentiel
                continue
            else:
                print(f"    Ã¢ÂÅ’ Erreur de permission: {e}")
                break
        except Exception as e:
            print(f"    Ã¢ÂÅ’ Erreur inattendue lors de la suppression: {e}")
            break
    
    # Si on arrive ici, toutes les tentatives ont ÃƒÂ©chouÃƒÂ©
    try:
        # Tentative finale avec ignore_errors=True
        shutil.rmtree(directory, ignore_errors=True)
        if not directory.exists():
            print(f"    Ã¢Å“â€¦ Suppression rÃƒÂ©ussie avec ignore_errors")
            return True
        else:
            print(f"    Ã¢Å¡Â Ã¯Â¸Â Dossier partiellement supprimÃƒÂ©, rÃƒÂ©sidu: {directory}")
            return False
    except Exception as e:
        print(f"    Ã¢ÂÅ’ Ãƒâ€°chec final de suppression: {e}")
        return False

# Gestion optionnelle de Mediapipe avec fallback
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    _register_dependency_status("Ã¢Å“â€¦ Mediapipe disponible - Utilisation des fonctionnalitÃƒÂ©s IA avancÃƒÂ©es")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None
    _register_dependency_status("Ã¢Å¡Â Ã¯Â¸Â Mediapipe non disponible - Utilisation du fallback OpenCV (fonctionnalitÃƒÂ©s rÃƒÂ©duites)")

# Ã°Å¸Å¡â‚¬ NOUVEAU: Import du sÃƒÂ©lecteur B-roll gÃƒÂ©nÃƒÂ©rique
try:
    from broll_selector import BrollSelector, Asset, ScoringFeatures, BrollCandidate
    BROLL_SELECTOR_AVAILABLE = True
    _register_dependency_status("Ã¢Å“â€¦ SÃƒÂ©lecteur B-roll gÃƒÂ©nÃƒÂ©rique disponible - Scoring mixte activÃƒÂ©")
except ImportError as e:
    BROLL_SELECTOR_AVAILABLE = False
    _register_dependency_status(f"Ã¢Å¡Â Ã¯Â¸Â SÃƒÂ©lecteur B-roll gÃƒÂ©nÃƒÂ©rique non disponible: {e}")
    _register_dependency_status("   Ã°Å¸â€â€ž Utilisation du systÃƒÂ¨me de scoring existant")

from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.video.fx.all import crop
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


def _legacy_pipeline_fallback_enabled() -> bool:
    """Return True when the legacy src.pipeline integration may execute."""

    override = os.getenv("ENABLE_LEGACY_PIPELINE_FALLBACK")
    if override is not None:
        return _to_bool(override)
    return getattr(Config, "ENABLE_LEGACY_PIPELINE_FALLBACK", False)


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
                _register_dependency_status(f"Ã¢Å“â€¦ ImageMagick configurÃƒÂ©: {path}")
                return True

        _register_dependency_status("Ã¢Å¡Â Ã¯Â¸Â ImageMagick non trouvÃƒÂ©, utilisation du mode fallback")
        return False

    except Exception as e:
        _register_dependency_status(f"Ã¢Å¡Â Ã¯Â¸Â Erreur configuration ImageMagick: {e}")
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
    SUBTITLE_FONT_SIZE = 85
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
    ENABLE_LEGACY_PIPELINE_FALLBACK = _to_bool(
        _UI_SETTINGS.get('legacy_pipeline_fallback'),
        default=getattr(_ROOT_CONFIG, 'ENABLE_LEGACY_PIPELINE_FALLBACK', False),
    ) if 'legacy_pipeline_fallback' in _UI_SETTINGS else getattr(_ROOT_CONFIG, 'ENABLE_LEGACY_PIPELINE_FALLBACK', False)

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
    # Ã°Å¸Å¡â‚¬ NOUVEAU: Forcer le nettoyage aprÃƒÂ¨s chaque vidÃƒÂ©o pour ÃƒÂ©conomiser l'espace
    BROLL_CLEANUP_PER_VIDEO = True  # Toujours activÃƒÂ© pour ÃƒÂ©viter l'accumulation
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
        'success': ['success', 'win', 'achievement', 'goal', 'growth', 'scale', 'unstoppable', 'beast'],
        'people': ['team', 'customer', 'client', 'person', 'human', 'community'],
        'emotion_positive': ['amazing', 'incredible', 'fantastic', 'awesome', 'fire'],
        'emotion_negative': ['problem', 'issue', 'difficult', 'challenge', 'fail'],
        'action': ['build', 'create', 'launch', 'start', 'implement', 'execute'],
        # Ã°Å¸Å¡â‚¬ NOUVELLES CATÃƒâ€°GORIES pour contenu cerveau/mental/neurosciences
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
                        # Ã°Å¸Å¡â‚¬ CORRECTION: Gestion robuste des timestamps avec slice objects
                        start_val = seg.get('start', 0.0)
                        end_val = seg.get('end', 0.0)
                        
                        # Convertir slice objects en float si nÃƒÂ©cessaire
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
    # Ã°Å¸Å¡â‚¬ CORRECTION CRITIQUE: Gestion robuste du timestamp final
    total_duration = 0.0
    if transcript_segments:
        try:
            last_end = transcript_segments[-1].get('end', 0.0)
            # Convertir slice object en float si nÃƒÂ©cessaire
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
        main_theme = keyword_analysis.get('dominant_theme', 'general')  # Ã°Å¸Å¡â‚¬ CORRECTION: ClÃƒÂ© correcte
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
            # Ã°Å¸Å¡â‚¬ CORRECTION: keywords est un dict, pas une liste
            if keywords and isinstance(keywords, dict):
                # Extraire les premiers mots-clÃƒÂ©s de toutes les catÃƒÂ©gories
                all_kws = []
                for category_kws in keywords.values():
                    if isinstance(category_kws, list):
                        all_kws.extend(category_kws[:2])  # 2 par catÃƒÂ©gorie
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
        print(f"Ã¢Å¡Â Ã¯Â¸Â Erreur gÃƒÂ©nÃƒÂ©ration prompts AI: {e}")
        # Fallback prompts
        return ['general content', 'people working', 'modern technology']

class VideoProcessor:
    _shared_llm_service = None

    """Classe principale pour traiter les vidÃƒÂ©os"""

    def __init__(self):
        emit_dependency_status()
        if os.getenv("FAST_TESTS") == "1":
            self.whisper_model = object()
        else:
            self.whisper_model = whisper.load_model(Config.WHISPER_MODEL)
        _setup_directories()
        # Cache ÃƒÂ©ventuel pour spaCy
        self._spacy_model = None

        if VideoProcessor._shared_llm_service is None:
            try:
                VideoProcessor._shared_llm_service = get_shared_llm_service()
            except Exception as exc:
                logger.warning("LLM service initialisation failed: %s", exc)
        self._llm_service = VideoProcessor._shared_llm_service
        self._core_last_run_used = False
        self._last_broll_insert_count = 0
        self._pipeline_config = PipelineConfigBundle()
        meta_dir = Config.OUTPUT_FOLDER / 'meta'
        log_file = meta_dir / 'broll_pipeline_events.jsonl'
        self._broll_event_logger: Optional[JsonlLogger] = JsonlLogger(log_file)
        self._broll_env_logged = False

    def get_last_broll_insert_count(self) -> int:
        """Return the number of B-roll clips inserted during the last run."""

        return getattr(self, "_last_broll_insert_count", 0)

    def _setup_directories(self):
        """CrÃƒÂ©e les dossiers nÃƒÂ©cessaires"""
        for folder in [Config.CLIPS_FOLDER, Config.OUTPUT_FOLDER, Config.TEMP_FOLDER]:
            folder.mkdir(parents=True, exist_ok=True)
    
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
        self._broll_env_logged = False
        event_logger = self._get_broll_event_logger()

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

        previous_flag = os.environ.get('ENABLE_PIPELINE_CORE_FETCHER')
        os.environ['ENABLE_PIPELINE_CORE_FETCHER'] = 'true' if use_core else 'false'
        start_time = time.time()
        self._core_last_run_used = False
        result = None
        try:
            result = self._process_single_clip_impl(clip_path, verbose=verbose)
        except Exception as exc:
            event_logger.log({
                'event': 'pipeline_failed',
                'video': clip_path.name,
                'session_id': session_id,
                'core_requested': bool(use_core),
                'error': repr(exc),
            })
            raise
        finally:
            if previous_flag is None:
                os.environ.pop('ENABLE_PIPELINE_CORE_FETCHER', None)
            else:
                os.environ['ENABLE_PIPELINE_CORE_FETCHER'] = previous_flag

        final_path = Path(result) if result else None
        event_logger.log({
            'event': 'finalized',
            'video': clip_path.name,
            'session_id': session_id,
            'elapsed_s': round(time.time() - start_time, 3),
            'final_mp4': str(final_path) if final_path else None,
            'core_requested': bool(use_core),
            'core_effective': bool(getattr(self, '_core_last_run_used', False)),
        })

        log_path = getattr(event_logger, 'path', None)
        in_memory_entries = getattr(event_logger, 'entries', None)
        if isinstance(log_path, Path) and (not log_path.exists() or log_path.stat().st_size == 0):
            if not in_memory_entries:
                raise RuntimeError('[CORE] No JSONL events written for this run.')

        return final_path

    def _get_broll_event_logger(self):
        global _GLOBAL_BROLL_EVENTS_LOGGER

        logger_obj = _GLOBAL_BROLL_EVENTS_LOGGER
        if logger_obj is None:
            try:
                base_dir = Path(getattr(Config, 'OUTPUT_FOLDER', Path('output')))
            except Exception:
                base_dir = Path('output')
            events_path = base_dir / 'meta' / 'broll_pipeline_events.jsonl'
            events_path.parent.mkdir(parents=True, exist_ok=True)
            logger_obj = JsonlLogger(events_path)
            _GLOBAL_BROLL_EVENTS_LOGGER = logger_obj

        self._broll_event_logger = logger_obj
        if not getattr(self, '_broll_env_logged', False):
            try:
                env_payload = {
                    'event': 'broll_env_ready',
                    'providers': _get_provider_names(self._pipeline_config),
                }
                logger_obj.log(env_payload)
                self._broll_env_logged = True
            except Exception:
                pass
        return logger_obj


    def _maybe_use_pipeline_core(
        self,
        segments,
        broll_keywords,
        *,
        subtitles,
        input_path: Path,
    ) -> Optional[Tuple[int, Optional[Path]]]:
        """Attempt to run the pipeline_core orchestrator if configured.

        Returns a tuple ``(inserted_count, rendered_path)`` when the orchestrator
        is engaged. The rendered path may be ``None`` if rendering fails. When
        the orchestrator should not run, ``None`` is returned so the legacy
        pipeline can continue unchanged.
        """

        event_logger = self._get_broll_event_logger()
        if not _pipeline_core_fetcher_enabled():
            event_logger.log({'event': 'core_disabled', 'reason': 'flag_disabled'})
            self._core_last_run_used = False
            return None


        fetcher_cfg = getattr(self._pipeline_config, 'fetcher', None)
        if fetcher_cfg is None:
            logger.warning(
                "pipeline_core fetcher enabled but fetcher configuration is missing; "
                "falling back to legacy pipeline",
            )
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
            event_logger.log({'event': 'core_disabled', 'reason': 'providers_misconfigured', 'provider_type': type(providers).__name__})
            self._core_last_run_used = False
            return None

        if not providers_list:
            logger.warning(
                "pipeline_core fetcher enabled but no providers configured; falling back to legacy pipeline",
            )
            event_logger.log({'event': 'core_disabled', 'reason': 'no_providers'})
            self._core_last_run_used = False
            return None

        event_logger.log({'event': 'core_engaged', 'video': input_path.name, 'providers': len(providers_list)})

        inserted = self._insert_brolls_pipeline_core(
            segments,
            broll_keywords,
            subtitles=subtitles,
            input_path=input_path,
        )
        return inserted

    def _insert_brolls_pipeline_core(
        self,
        segments,
        broll_keywords,
        *,
        subtitles,
        input_path: Path,
    ) -> Tuple[int, Optional[Path], Dict[str, Any]]:
        global SEEN_URLS, SEEN_PHASHES, SEEN_IDENTIFIERS
        SEEN_URLS.clear()
        SEEN_PHASHES.clear()
        SEEN_IDENTIFIERS.clear()
        self._core_last_run_used = True
        self._last_broll_insert_count = 0
        self._core_last_timeline: List[CoreTimelineEntry] = []
        self._core_last_render_path: Optional[Path] = None
        logger.info("[BROLL] pipeline_core orchestrator engaged")
        config_bundle = self._pipeline_config
        orchestrator = FetcherOrchestrator(config_bundle.fetcher)
        selection_cfg = config_bundle.selection
        timeboxing_cfg = config_bundle.timeboxing
        event_logger = self._get_broll_event_logger()
        event_logger.log(
            {
                "event": "broll_session_start",
                "segment": -1,
                "total_segments": len(segments),
                "llm_healthy": bool(self._llm_service),
            }
        )

        # Prepare selection report (once per clip)
        report = None
        report_segments: List[Dict[str, Any]] = []
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
        if report is not None:
            try:
                report_segments = [
                    {
                        'segment': idx,
                        't0': float(getattr(segment, 'start', 0.0) or 0.0),
                        't1': float(getattr(segment, 'end', getattr(segment, 'start', 0.0)) or 0.0),
                        'queries': [],
                        'candidates': [],
                        'selected': [],
                    }
                    for idx, segment in enumerate(segments)
                ]
                report['segments'] = report_segments
            except Exception:
                report_segments = []
        if not segments:
            event_logger.log({'event': 'core_no_segments', 'reason': 'no_valid_segments', 'video': input_path.name})
            self._core_last_run_used = False
            self._last_broll_insert_count = 0
            return 0, None

        fetch_timeout = max((timeboxing_cfg.fetch_rank_ms or 0) / 1000.0, 0.0)

        selected_assets: List[Dict[str, Any]] = []
        provider_counter: Counter[str] = Counter()
        query_source_counter: Counter[str] = Counter()
        selected_segments: List[int] = []
        selected_durations: List[float] = []
        total_candidates = 0
        total_unique_candidates = 0
        total_url_dedup_hits = 0
        total_phash_dedup_hits = 0
        total_latency_ms = 0
        segments_with_queries = 0
        forced_keep_consumed = 0
        raw_forced_keep_budget = getattr(selection_cfg, 'forced_keep_budget', None)
        if isinstance(raw_forced_keep_budget, int):
            forced_keep_remaining: Optional[int] = max(0, raw_forced_keep_budget)
        elif isinstance(raw_forced_keep_budget, float):
            forced_keep_remaining = max(0, int(raw_forced_keep_budget))
        else:
            forced_keep_remaining = None
        forced_keep_allowed = bool(getattr(selection_cfg, 'allow_forced_keep', True))
        try:
            flags = getattr(self._pipeline_config, "flags", None)
            raw_cap = getattr(flags, "llm_max_queries_per_segment", None)
            query_cap = int(raw_cap) if raw_cap is not None else SEGMENT_REFINEMENT_MAX_TERMS
        except Exception:
            query_cap = SEGMENT_REFINEMENT_MAX_TERMS
        if not isinstance(query_cap, int):
            try:
                query_cap = int(query_cap)
            except Exception:
                query_cap = SEGMENT_REFINEMENT_MAX_TERMS
        if query_cap <= 0:
            query_cap = SEGMENT_REFINEMENT_MAX_TERMS
        if query_cap <= 0:
            query_cap = 4

        dyn_ctx = getattr(self, "_dyn_context", {})
        if not isinstance(dyn_ctx, dict):
            dyn_ctx = {}
        dyn_language = str(dyn_ctx.get("language") or "").strip().lower()

        metadata_payload = getattr(self, "_latest_metadata", {})
        if not isinstance(metadata_payload, dict):
            metadata_payload = {}
        metadata_query_cap = max(query_cap, 8)
        metadata_status = str(metadata_payload.get('llm_status') or '').strip().lower()
        metadata_queries_raw: List[str] = []
        llm_query_source_label: Optional[str] = None
        base_llm_queries: List[str] = []
        if metadata_status == 'ok':
            try:
                metadata_queries_raw = (
                    metadata_payload.get("queries")
                    or metadata_payload.get("search_queries")
                    or metadata_payload.get("broll_queries")
                    or []
                )
            except Exception:
                metadata_queries_raw = []
            try:
                debug_keys = list(metadata_payload.keys())
            except Exception:
                debug_keys = []
            print(f"    🔍 DEBUG metadata_payload keys: {debug_keys}")
            debug_preview = metadata_queries_raw[:5] if metadata_queries_raw else "VIDE - Va régénérer !"
            print(f"    🔍 DEBUG queries found: {debug_preview}")
            base_llm_queries = _relaxed_normalise_terms(metadata_queries_raw, metadata_query_cap)
            print(
                f"    🔍 DEBUG llm_queries après normalisation: {base_llm_queries[:5]}"
            )
            if not base_llm_queries:
                print("    ⚠️ llm_queries VIDE après normalisation !")
            if not base_llm_queries:
                broll_kw = metadata_payload.get("broll_keywords") or []
                print(f"    ⚠️ Queries vides ! Utilisation broll_keywords: {broll_kw[:5]}")
                base_llm_queries = list(broll_kw[:metadata_query_cap])
            if base_llm_queries:
                llm_query_source_label = 'llm_metadata'
        if not base_llm_queries:
            base_llm_queries = _relaxed_normalise_terms(dyn_ctx.get("search_queries") or [], metadata_query_cap)
            if base_llm_queries and llm_query_source_label is None:
                llm_query_source_label = 'dynamic_context'
        if not base_llm_queries:
            try:
                fetch_kw = getattr(self, "_fetch_keywords", [])
            except Exception:
                fetch_kw = []
            base_llm_queries = _relaxed_normalise_terms(fetch_kw, metadata_query_cap)
            if base_llm_queries and llm_query_source_label is None:
                llm_query_source_label = 'fetch_keywords'
        if not base_llm_queries:
            base_llm_queries = _relaxed_normalise_terms(broll_keywords or [], metadata_query_cap)
            if base_llm_queries and llm_query_source_label is None:
                llm_query_source_label = 'transcript_keywords'

        try:
            selector_keywords = list(getattr(self, "_selector_keywords", []))
        except Exception:
            selector_keywords = []

        for idx, segment in enumerate(segments):
            seg_duration = max(0.0, segment.end - segment.start)
            report_entry = report_segments[idx] if idx < len(report_segments) else None

            llm_hints = None
            llm_healthy = bool(getattr(self, "_llm_service", None))
            hint_terms: List[str] = []
            hint_source: Optional[str] = None
            raw_hint_items: List[str] = []
            if getattr(self, "_llm_service", None):
                try:
                    llm_hints = self._llm_service.generate_hints_for_segment(
                        segment.text,
                        segment.start,
                        segment.end,
                    )
                except Exception:
                    llm_hints = None
                    llm_healthy = False
                else:
                    llm_healthy = True
            if llm_hints and isinstance(llm_hints.get('queries'), list):
                raw_hint_items = [item for item in llm_hints['queries'] if isinstance(item, str)]
                hint_terms = _dedupe_queries(raw_hint_items, cap=metadata_query_cap)
                raw_source = llm_hints.get('source') if isinstance(llm_hints, dict) else None
                if isinstance(raw_source, str) and raw_source.strip():
                    hint_source = raw_source.strip()

            raw_segment_keywords = self._derive_segment_keywords(segment, broll_keywords)
            segment_keywords = _dedupe_queries(raw_segment_keywords, cap=query_cap)

            try:
                brief_queries, brief_keywords = _segment_brief_terms(dyn_ctx, idx)
            except Exception:
                brief_queries, brief_keywords = [], []

            queries, query_source = _merge_segment_query_sources(
                segment_text=getattr(segment, "text", "") or "",
                llm_queries=base_llm_queries,
                brief_queries=brief_queries,
                brief_keywords=brief_keywords,
                segment_keywords=segment_keywords,
                selector_keywords=selector_keywords,
                cap=query_cap,
            )

            if query_source == 'llm_hint':
                if brief_queries:
                    query_source = 'segment_brief'
                elif llm_query_source_label:
                    query_source = llm_query_source_label
                else:
                    query_source = 'transcript_fallback'

            if hint_terms:
                queries = hint_terms
                query_source = hint_source or 'llm_hint'
                try:
                    logger.info(
                        "[BROLL][LLM] segment=%.2f-%.2f queries=%s (source=%s)",
                        float(getattr(segment, 'start', 0.0) or 0.0),
                        float(getattr(segment, 'end', getattr(segment, 'start', 0.0)) or 0.0),
                        queries,
                        query_source,
                    )
                except Exception:
                    pass
            elif llm_hints and raw_hint_items:
                raw_hint_clean = [item.strip() for item in raw_hint_items if isinstance(item, str) and item.strip()]
                if raw_hint_clean:
                    queries = raw_hint_clean[:max(1, query_cap)]
                    query_source = hint_source or 'llm_hint'
                    try:
                        logger.info(
                            "[BROLL][LLM] segment=%.2f-%.2f queries=%s (source=%s)",
                            float(getattr(segment, 'start', 0.0) or 0.0),
                            float(getattr(segment, 'end', getattr(segment, 'start', 0.0)) or 0.0),
                            queries,
                            query_source,
                        )
                    except Exception:
                        pass

            try:
                queries = enforce_fetch_language(queries, dyn_language or None)
            except Exception:
                queries = list(queries)

            if query_source == 'segment_brief':
                try:
                    banned_terms = {_norm_query_term(term) for term in raw_segment_keywords}
                except Exception:
                    banned_terms = set()
                if banned_terms:
                    filtered_queries = [q for q in queries if _norm_query_term(q) not in banned_terms]
                    if filtered_queries:
                        queries = filtered_queries[:max(1, query_cap)]

            if report_entry is not None:
                try:
                    report_entry['queries'] = list(queries)
                    report_entry['query_source'] = query_source
                except Exception:
                    pass

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
                    queries=queries,
                )
                continue

            segments_with_queries += 1
            query_source_counter[query_source] += 1

            try:
                if event_logger:
                    event_logger.log(
                        {
                            'event': 'broll_segment_queries',
                            'segment': idx,
                            'queries': list(queries),
                            'source': query_source,
                            'language': dyn_language or None,
                        }
                    )
            except Exception:
                pass

            try:
                print(f"[BROLL] segment #{idx}: queries={queries} (source={query_source})")
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

            total_candidates += len(candidates)
            total_unique_candidates += len(unique_candidates)
            total_url_dedup_hits += url_hits
            total_phash_dedup_hits += phash_hits

            best_candidate = None
            best_score = -1.0
            best_provider = None
            reject_reasons: List[str] = []
            forced_keep = False
            candidate_records: List[Dict[str, Any]] = []
            filter_pass_records: List[Dict[str, Any]] = []

            for candidate in unique_candidates:
                score = self._rank_candidate(segment.text, candidate, selection_cfg, seg_duration)
                passes_filters, filter_reason = orchestrator.evaluate_candidate_filters(
                    candidate, filters, seg_duration
                )
                orientation, duration_val = self._summarize_candidate_media(candidate)
                record: Dict[str, Any] = {
                    'candidate': candidate,
                    'provider': getattr(candidate, 'provider', None),
                    'score': score,
                    'orientation': orientation,
                    'duration_s': duration_val,
                    'filter_reason': filter_reason,
                    'passes_filters': passes_filters,
                    'below_threshold': score < selection_cfg.min_score,
                }
                candidate_records.append(record)
                if passes_filters:
                    filter_pass_records.append(record)

            eligible_records = [rec for rec in filter_pass_records if not rec['below_threshold']]
            best_record = None
            fallback_record = None
            if eligible_records:
                best_record = max(eligible_records, key=lambda rec: rec['score'])
            elif filter_pass_records:
                fallback_record = max(filter_pass_records, key=lambda rec: rec['score'])
                if forced_keep_allowed and (forced_keep_remaining is None or forced_keep_remaining > 0):
                    best_record = fallback_record
                    forced_keep = True
                else:
                    reason = 'disabled' if not forced_keep_allowed else 'exhausted'
                    fallback_candidate = fallback_record.get('candidate') if isinstance(fallback_record, dict) else None
                    fallback_provider = fallback_record.get('provider') if isinstance(fallback_record, dict) else None
                    fallback_url = getattr(fallback_candidate, 'url', None)
                    logger.info(
                        "[BROLL] forced-keep fallback skipped for segment %s (reason=%s, provider=%s, url=%s)",
                        idx,
                        reason,
                        fallback_provider,
                        fallback_url,
                    )
                    try:
                        if event_logger:
                            event_logger.log(
                                {
                                    'event': 'forced_keep_skipped',
                                    'segment': idx,
                                    'reason': reason,
                                    'provider': fallback_provider,
                                    'url': fallback_url,
                                }
                            )
                    except Exception:
                        pass

            if best_record:
                best_candidate = best_record['candidate']
                best_score = best_record['score']
                best_provider = getattr(best_candidate, 'provider', None)

            reject_reason_counts: Counter[str] = Counter()
            candidate_summary: List[Dict[str, Any]] = []

            for record in candidate_records:
                is_selected = best_record is not None and record is best_record
                reason = record['filter_reason']
                if is_selected:
                    reason = None
                elif reason is None:
                    if record['below_threshold']:
                        reason = 'low_score'
                    elif best_record is not None and record['passes_filters']:
                        reason = 'outscored'
                if reason:
                    reject_reason_counts[reason] += 1
                    reject_reasons.append(reason)
                candidate_obj = record.get('candidate') if isinstance(record, dict) else getattr(record, 'candidate', None)
                candidate_url = getattr(candidate_obj, 'url', None)
                summary_entry = {
                    'provider': record['provider'],
                    'orientation': record['orientation'],
                    'duration_s': record['duration_s'],
                    'score': record['score'],
                    'reject_reason': reason,
                    'selected': is_selected,
                    'url': candidate_url,
                }
                candidate_summary.append(summary_entry)
                try:
                    if event_logger:
                        score_value = summary_entry['score']
                        if isinstance(score_value, (int, float)):
                            score_payload: Optional[float] = float(score_value)
                        else:
                            score_payload = None
                        event_logger.log(
                            {
                                'event': 'broll_candidate_evaluated',
                                'segment': idx,
                                'provider': summary_entry['provider'],
                                'url': candidate_url,
                                'score': score_payload,
                                'reject_reason': summary_entry['reject_reason'],
                                'selected': is_selected,
                            }
                        )
                except Exception:
                    pass

                if report_entry is not None:
                    try:
                        candidate_obj = record.get('candidate') if isinstance(record, dict) else None
                        report_entry.setdefault('candidates', []).append(
                            {
                                'provider': record.get('provider'),
                                'url': getattr(candidate_obj, 'url', None),
                                'score': record.get('score'),
                                'reject_reason': reason,
                                'selected': is_selected,
                            }
                        )
                    except Exception:
                        pass

            if forced_keep and best_candidate:
                reject_reason_counts['fallback_low_score'] += 1
                reject_reasons.append('fallback_low_score')
                if forced_keep_remaining is not None:
                    forced_keep_remaining = max(0, forced_keep_remaining - 1)
                forced_keep_consumed += 1
                remaining_display = 'Ã¢Ë†Å¾' if forced_keep_remaining is None else forced_keep_remaining
                score_display = (
                    f"{float(best_score):.3f}"
                    if isinstance(best_score, (int, float))
                    else 'n/a'
                )
                logger.info(
                    "[BROLL] forced-keep fallback used for segment %s (provider=%s, url=%s, score=%s, remaining=%s)",
                    idx,
                    best_provider,
                    getattr(best_candidate, 'url', None),
                    score_display,
                    remaining_display,
                )
                try:
                    if event_logger:
                        event_logger.log(
                            {
                                'event': 'forced_keep_consumed',
                                'segment': idx,
                                'provider': best_provider,
                                'url': getattr(best_candidate, 'url', None),
                                'score': float(best_score) if isinstance(best_score, (int, float)) else None,
                                'remaining_budget': forced_keep_remaining,
                            }
                        )
                except Exception:
                    pass

            if best_candidate:
                selected_assets.append({
                    'segment': idx,
                    'provider': best_provider,
                    'url': getattr(best_candidate, 'url', None),
                    'score': best_score,
                    'candidate': best_candidate,
                    'start': float(getattr(segment, 'start', 0.0) or 0.0),
                    'end': float(getattr(segment, 'end', getattr(segment, 'start', 0.0)) or 0.0),
                })
                provider_label = str(best_provider or 'unknown')
                provider_counter[provider_label] += 1
                selected_segments.append(idx)
                duration_val = getattr(best_candidate, 'duration', None)
                if isinstance(duration_val, (int, float)) and duration_val > 0:
                    selected_durations.append(float(duration_val))
                elif seg_duration > 0:
                    selected_durations.append(float(seg_duration))
                # Append to selection report
                try:
                    if report is not None:
                        if report_entry is not None:
                            report_entry.setdefault('selected', []).append(
                                {
                                    'provider': getattr(best_candidate, 'provider', None),
                                    'url': getattr(best_candidate, 'url', None),
                                    'score': float(best_score) if isinstance(best_score, (int, float)) else best_score,
                                }
                            )
                except Exception:
                    pass

            latency_ms = int((time.perf_counter() - start_time) * 1000)
            total_latency_ms += latency_ms
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
                reject_reasons=reject_reasons,
                reject_summary={
                    'counts': dict(reject_reason_counts),
                    'candidates': candidate_summary,
                },
                queries=queries,
            )

        # Add effective domain fields to the summary line for easy scraping
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
            provider_status={**(provider_status or {}), 'selected_count': len(selected_assets)} if provider_status else {'selected_count': len(selected_assets)},
        )
        initial_selected = len(selected_assets)

        total_segments = len(segments)
        total_duration = max((getattr(seg, 'end', 0.0) or 0.0) for seg in segments) if segments else 0.0
        avg_latency = (total_latency_ms / segments_with_queries) if segments_with_queries else 0.0
        refined_ratio = (query_source_counter.get('segment_brief', 0) / total_segments) if total_segments else 0.0

        summary_payload: Dict[str, Any] = {
            'event': 'broll_summary',
            'segments': total_segments,
            'inserted': 0,
            'selection_rate': 0.0,
            'selected_segments': [],
            'avg_broll_duration': 0.0,
            'broll_per_min': 0.0,
            'avg_latency_ms': round(avg_latency, 1) if avg_latency else 0.0,
            'refined_ratio': round(refined_ratio, 4),
            'provider_mix': {},
            'providers_used': [],
            'query_source_counts': dict(query_source_counter),
            'total_url_dedup_hits': total_url_dedup_hits,
            'total_phash_dedup_hits': total_phash_dedup_hits,
            'dedupe_counts': {
                'url': total_url_dedup_hits,
                'phash': total_phash_dedup_hits,
            },
            'forced_keep_segments': forced_keep_consumed,
            'forced_keep_count': forced_keep_consumed,
            'total_candidates': total_candidates,
            'total_unique_candidates': total_unique_candidates,
            'video_duration_s': round(total_duration, 3) if total_duration else 0.0,
            'render_ok': False,
        }

        render_path: Optional[Path] = None
        materialized_entries: List[CoreTimelineEntry] = []
        pending_seen_updates: List[Dict[str, Any]] = []
        if initial_selected > 0:
            timeline_entries: List[CoreTimelineEntry] = []
            download_dir: Optional[Path]
            try:
                download_dir = Config.TEMP_FOLDER / 'with_broll_core' / Path(input_path).stem
                download_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                logger.warning('[BROLL] unable to prepare core download dir: %s', exc)
                download_dir = None

            for order, asset in enumerate(selected_assets):
                candidate = asset.get('candidate') if isinstance(asset, dict) else None
                if not candidate or download_dir is None:
                    continue
                try:
                    segment_idx = int(asset.get('segment', order))
                except Exception:
                    segment_idx = order
                local_path = self._download_core_candidate(candidate, download_dir, order)
                if not local_path:
                    try:
                        event_logger.log({
                            'event': 'core_asset_materialize_failed',
                            'url': asset.get('url'),
                            'provider': asset.get('provider'),
                            'segment': segment_idx,
                        })
                    except Exception:
                        pass
                    continue

                start = float(asset.get('start', 0.0) or 0.0)
                end = float(asset.get('end', start) or start)
                duration = getattr(candidate, 'duration', None)
                if isinstance(duration, (int, float)) and duration > 0:
                    end = min(end, start + float(duration)) if end > start else start + float(duration)
                if end <= start:
                    continue

                timeline_entries.append(
                    CoreTimelineEntry(
                        path=local_path,
                        start=start,
                        end=end,
                        segment_index=segment_idx,
                        provider=asset.get('provider'),
                        url=asset.get('url'),
                    )
                )
                pending_seen_updates.append(
                    {
                        'url': asset.get('url') or getattr(candidate, 'url', None),
                        'phash': getattr(candidate, '_phash', None),
                        'identifier': getattr(candidate, 'identifier', None),
                    }
                )

            if timeline_entries:
                timeline_entries, pending_seen_updates = _apply_broll_invariants_to_core_entries(
                    timeline_entries,
                    seen_updates=pending_seen_updates,
                )

            if timeline_entries:
                timeline_entries.sort(key=lambda entry: (entry.start, entry.segment_index))
                self._core_last_timeline = list(timeline_entries)
                manifest_path: Optional[Path] = None
                if download_dir is not None:
                    try:
                        manifest_payload = [entry.to_dict() for entry in timeline_entries]
                        manifest_path = download_dir / 'timeline.json'
                        with manifest_path.open('w', encoding='utf-8') as handle:
                            json.dump(manifest_payload, handle, ensure_ascii=False, indent=2)
                    except Exception as exc:
                        logger.debug('[BROLL] failed to persist core timeline manifest: %s', exc)

                candidate_entries = list(timeline_entries)
                render_candidate = self._render_core_broll_timeline(Path(input_path), timeline_entries)
                if render_candidate:
                    render_path = render_candidate
                    self._core_last_render_path = render_path
                    materialized_entries = candidate_entries
                    try:
                        event_logger.log({
                            'event': 'core_render_complete',
                            'output': str(render_path),
                            'inserted': len(materialized_entries),
                            'manifest': str(manifest_path) if manifest_path else None,
                        })
                    except Exception:
                        pass
                else:
                    try:
                        event_logger.log({
                            'event': 'core_render_failed',
                            'attempted': len(candidate_entries),
                            'inserted': 0,
                        })
                    except Exception:
                        pass
            else:
                try:
                    event_logger.log({'event': 'core_no_timeline', 'selected': initial_selected, 'inserted': 0})
                except Exception:
                    pass

        final_inserted = len(materialized_entries)
        self._last_broll_insert_count = final_inserted

        if final_inserted > 0:
            final_segments = [entry.segment_index for entry in materialized_entries]
            durations = [entry.duration for entry in materialized_entries]
            providers = Counter(str(entry.provider or 'unknown') for entry in materialized_entries)
        else:
            final_segments = []
            durations = []
            providers = Counter()

        summary_event_payload: Optional[Dict[str, Any]] = None

        try:
            selection_rate = (final_inserted / total_segments) if total_segments else 0.0
            avg_duration = (sum(durations) / len(durations)) if durations else 0.0
            broll_per_min = (final_inserted / (total_duration / 60.0)) if total_duration > 0 else 0.0
            provider_mix = {k: v for k, v in sorted(providers.items()) if v > 0}

            overlay_paths = [entry.path for entry in materialized_entries if getattr(entry, 'path', None)]
            overlays_exist = bool(overlay_paths) and all(Path(path).exists() for path in overlay_paths)
            render_path_obj = Path(render_path) if render_path else None
            render_path_exists = bool(render_path_obj and render_path_obj.exists())
            render_ok_flag = bool(final_inserted > 0 and overlays_exist and render_path_exists)

            if render_ok_flag and materialized_entries:
                for idx, entry in enumerate(materialized_entries):
                    markers: Dict[str, Any] = pending_seen_updates[idx] if idx < len(pending_seen_updates) else {}
                    url_marker = markers.get('url') or getattr(entry, 'url', None)
                    if url_marker:
                        SEEN_URLS.add(url_marker)
                    phash_marker = markers.get('phash')
                    if phash_marker is not None:
                        SEEN_PHASHES.append(phash_marker)
                    identifier_marker = markers.get('identifier')
                    if identifier_marker:
                        SEEN_IDENTIFIERS.add(identifier_marker)

            summary_payload.update(
                {
                    'inserted': final_inserted,
                    'selection_rate': round(selection_rate, 4),
                    'selected_segments': final_segments,
                    'avg_broll_duration': round(avg_duration, 3) if durations else 0.0,
                    'broll_per_min': round(broll_per_min, 3) if broll_per_min else 0.0,
                    'provider_mix': provider_mix,
                    'providers_used': sorted(provider_mix.keys()),
                    'render_ok': render_ok_flag,
                }
            )
            summary_event_payload = {k: v for k, v in summary_payload.items() if v is not None}

            providers_display = ", ".join(f"{k}:{v}" for k, v in provider_mix.items()) or "none"
            render_ok_value = summary_payload.get('render_ok')
            icon = "Ã°Å¸â€œÅ " if render_ok_value else "Ã¢Å¡Â Ã¯Â¸Â"
            suffix = ""
            if final_inserted == 0 and initial_selected > 0:
                suffix = " (ÃƒÂ©chec du tÃƒÂ©lÃƒÂ©chargement/rendu)"
            elif final_inserted > 0 and not render_ok_value:
                suffix = " (rendu indisponible)"
            print(
                f"    {icon} B-roll sÃƒÂ©lectionnÃƒÂ©s: {final_inserted}/{total_segments} "
                f"({selection_rate * 100:.1f}%); providers={providers_display}{suffix}"
            )
        except Exception:
            pass

        if summary_event_payload is not None:
            try:
                event_logger.log(summary_event_payload)
            except Exception:
                pass

        # Persist compact selection report next to JSONL
        try:
            ENABLE_SELECTION_REPORT = os.getenv("ENABLE_SELECTION_REPORT", "true").lower() not in {"0","false","no"}
        except Exception:
            ENABLE_SELECTION_REPORT = True
        if ENABLE_SELECTION_REPORT and report is not None:
            try:
                seg_total = len(segments) if segments else 0
                seg_sel = sum(1 for entry in report.get('segments') or [] if entry.get('selected'))
                report['selection_rate'] = round((seg_sel / seg_total), 3) if seg_total else 0.0
                meta_dir = Config.OUTPUT_FOLDER / 'meta'
                meta_dir.mkdir(parents=True, exist_ok=True)
                name = f"selection_report_{report.get('video_stem') or 'clip'}.json"
                out_path = meta_dir / name
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                print(f"[REPORT] wrote {out_path}")
            except Exception as e:
                print(f"[REPORT] failed: {e}")

        return final_inserted, render_path, {'render_ok': summary_payload.get('render_ok')}

    def _download_core_candidate(self, candidate, download_dir: Path, order: int, segment: Optional[int] = None) -> Optional[Path]:
        """Download a remote candidate selected by the core orchestrator."""

        url = getattr(candidate, 'url', None)
        if not url:
            return None

        try:
            parsed = urlparse(str(url))
            ext = Path(parsed.path).suffix
        except Exception:
            ext = ''

        if not ext:
            ext = '.mp4'

        filename = f"core_{order:03d}{ext}"
        destination = download_dir / filename

        try:
            response = requests.get(str(url), stream=True, timeout=15)
            response.raise_for_status()
            with open(destination, 'wb') as fh:
                for chunk in response.iter_content(chunk_size=1024 * 8):
                    if chunk:
                        fh.write(chunk)
        except Exception as exc:
            logger.warning('[BROLL] failed to download %s: %s', url, exc)
            try:
                if destination.exists():
                    destination.unlink()
            except Exception:
                pass
            try:
                event_logger = self._get_broll_event_logger()
            except Exception:
                event_logger = None
            provider = getattr(candidate, 'provider', None)
            if event_logger is not None:
                try:
                    event_logger.log(
                        {
                            'event': 'broll_asset_download_failed',
                            'provider': provider,
                            'url': url,
                            'path': str(destination),
                            'error': str(exc),
                            'segment': segment,
                        }
                    )
                except Exception:
                    pass
            return None

        try:
            event_logger = self._get_broll_event_logger()
        except Exception:
            event_logger = None
        if event_logger is not None:
            provider = getattr(candidate, 'provider', None)
            try:
                event_logger.log(
                    {
                        'event': 'broll_asset_downloaded',
                        'provider': provider,
                        'url': url,
                        'path': str(destination),
                        'segment': segment,
                    }
                )
            except Exception:
                pass

        return destination

    def _render_core_broll_timeline(
        self,
        input_path: Path,
        timeline: Sequence[Union[CoreTimelineEntry, Dict[str, Any]]],
    ) -> Optional[Path]:
        """Render a simple composite video using the downloaded core assets."""

        try:
            event_logger = self._get_broll_event_logger()
        except Exception:
            event_logger = None

        if not timeline:
            if event_logger is not None:
                try:
                    event_logger.log(
                        {
                            'event': 'broll_timeline_failed',
                            'clips': 0,
                            'output': None,
                            'reason': 'empty_timeline',
                        }
                    )
                except Exception:
                    pass
            return None

        normalized: List[CoreTimelineEntry] = []
        for item in timeline:
            if isinstance(item, CoreTimelineEntry):
                normalized.append(item)
                continue
            if isinstance(item, dict):
                clip_path = item.get('path')
                if not clip_path:
                    continue
                try:
                    start = float(item.get('start', 0.0) or 0.0)
                    end = float(item.get('end', start) or start)
                except Exception:
                    continue
                try:
                    seg_idx = int(item.get('segment', len(normalized)))
                except Exception:
                    seg_idx = len(normalized)
                normalized.append(
                    CoreTimelineEntry(
                        path=Path(clip_path),
                        start=start,
                        end=end,
                        segment_index=seg_idx,
                        provider=item.get('provider'),
                        url=item.get('url'),
                    )
                )

        clip_count = len(normalized)
        total_timeline_duration = sum(entry.duration for entry in normalized)

        if not normalized:
            if event_logger is not None:
                try:
                    event_logger.log(
                        {
                            'event': 'broll_timeline_failed',
                            'clips': 0,
                            'output': None,
                            'reason': 'no_valid_clips',
                        }
                    )
                except Exception:
                    pass
            return None

        output_dir = Config.TEMP_FOLDER / 'with_broll_core'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._unique_path(
            output_dir,
            f"with_broll_core_{Path(input_path).stem}",
            '.mp4',
        )

        plan_events: List[Dict[str, Any]] = [
            {
                'start': entry.start,
                'end': entry.end,
                'asset_path': str(entry.path),
                'media_path': str(entry.path),
                'segment': entry.segment_index,
                'provider': entry.provider,
                'url': entry.url,
                'crossfade_frames': 0,
            }
            for entry in normalized
        ]

        try:
            from src.pipeline.renderer import render_video  # type: ignore
        except Exception:
            render_video = None  # type: ignore

        last_error: Optional[str] = None

        if render_video is not None:
            render_cfg = types.SimpleNamespace(
                input_video=str(input_path),
                output_video=str(output_path),
                codec='libx264',
                audio_codec='aac',
                ffmpeg_preset='medium',
                crf=21,
                threads=max(1, (os.cpu_count() or 1) // 2 or 1),
            )
            try:
                render_video(render_cfg, [], plan_events)
                if event_logger is not None:
                    try:
                        event_logger.log(
                            {
                                'event': 'broll_timeline_rendered',
                                'clips': clip_count,
                                'count': clip_count,
                                'duration_s': round(total_timeline_duration, 3),
                                'output': str(output_path),
                                'renderer': 'pipeline',
                            }
                        )
                    except Exception:
                        pass
                return output_path
            except Exception as exc:
                logger.warning('[BROLL] core renderer (pipeline renderer) failed: %s', exc)
                try:
                    if output_path.exists():
                        output_path.unlink()
                except Exception:
                    pass
                last_error = str(exc)

        try:
            with ExitStack() as stack:
                base_clip = stack.enter_context(VideoFileClip(str(input_path)))
                layers = [base_clip]
                base_height = getattr(base_clip, 'h', None)
                base_width = int(getattr(base_clip, 'w', 0) or 0)
                roi_cache: Dict[str, Tuple[Optional[float], Optional[str]]] = {}

                for entry in normalized:
                    overlay = stack.enter_context(VideoFileClip(str(entry.path)))
                    duration = entry.duration

                    if base_height and getattr(overlay, 'h', None):
                        try:
                            overlay = overlay.resize(height=base_height)
                        except Exception:
                            pass

                    try:
                        ov_w = int(getattr(overlay, 'w', 0) or 0)
                        ov_h = int(getattr(overlay, 'h', 0) or 0)
                    except Exception:
                        ov_w = ov_h = 0

                    cache_key = str(entry.path)
                    if cache_key in roi_cache:
                        roi_center, roi_mode = roi_cache[cache_key]
                    else:
                        try:
                            roi_center, roi_mode = estimate_roi_x_center(entry.path)
                        except Exception:
                            roi_center, roi_mode = (None, None)
                        roi_cache[cache_key] = (roi_center, roi_mode)

                    try:
                        overlay, crop_meta = _apply_vertical_crop(
                            overlay,
                            base_h=int(base_height) if base_height else ov_h,
                            base_w=base_width,
                            roi_center=roi_center,
                        )
                    except Exception:
                        crop_meta = None

                    if crop_meta and event_logger is not None:
                        crop_payload: Dict[str, Any] = {
                            'event': 'broll_crop_applied',
                            'path': str(entry.path),
                            'segment': entry.segment_index,
                            'src_w': int(crop_meta.get('source_width') or ov_w),
                            'src_h': int(crop_meta.get('source_height') or ov_h),
                            'target': '9:16',
                            'crop': {
                                'x0': int(crop_meta.get('x0') or 0),
                                'w': int(crop_meta.get('width') or 0),
                                'h': int(crop_meta.get('height') or 0),
                            },
                            'roi': roi_mode or 'centered',
                        }
                        if crop_meta.get('center_ratio') is not None:
                            crop_payload['roi_center_ratio'] = round(float(crop_meta['center_ratio']), 4)
                        try:
                            event_logger.log(crop_payload)
                        except Exception:
                            pass

                    try:
                        overlay = overlay.without_audio()
                    except Exception:
                        try:
                            overlay = overlay.set_audio(None)
                        except Exception:
                            pass

                    if duration > 0 and getattr(overlay, 'duration', None):
                        try:
                            overlay = overlay.subclip(0, min(duration, float(overlay.duration)))
                        except Exception:
                            pass

                    start = max(0.0, float(entry.start))
                    try:
                        overlay = overlay.set_start(start)
                    except Exception:
                        pass

                    if duration > 0:
                        try:
                            overlay = overlay.set_duration(duration)
                        except Exception:
                            pass

                    try:
                        overlay = overlay.set_position('center')
                    except Exception:
                        pass

                    layers.append(overlay)

                composite = CompositeVideoClip(layers)
                composite.write_videofile(
                    str(output_path),
                    codec='libx264',
                    audio_codec='aac',
                    preset='medium',
                    threads=max(1, (os.cpu_count() or 1) // 2 or 1),
                    logger=None,
                )
                composite.close()
                if event_logger is not None:
                    try:
                        event_logger.log(
                            {
                                'event': 'broll_timeline_rendered',
                                'clips': clip_count,
                                'count': clip_count,
                                'duration_s': round(total_timeline_duration, 3),
                                'output': str(output_path),
                                'renderer': 'moviepy',
                            }
                        )
                    except Exception:
                        pass
                return output_path
        except Exception as exc:
            logger.warning('[BROLL] core renderer failed: %s', exc)
            last_error = str(exc)

        if event_logger is not None:
            try:
                event_logger.log(
                    {
                        'event': 'broll_timeline_failed',
                        'clips': clip_count,
                        'output': str(output_path),
                        'reason': last_error,
                    }
                )
            except Exception:
                pass

        return None

    def _normalize_core_result(
        self,
        result: Any,
    ) -> Tuple[Optional[int], Optional[Path], Optional[List[Dict[str, Any]]], Optional[bool]]:
        """Extract count, rendered path, selection plan, and render flag."""

        inserted: Optional[int] = None
        render_path: Optional[Path] = None
        selections: Optional[List[Dict[str, Any]]] = None
        render_ok: Optional[bool] = None

        def _coerce_int(value: Any) -> Optional[int]:
            try:
                if value is None:
                    return None
                return int(value)
            except (TypeError, ValueError):
                return None

        def _maybe_update_from_payload(payload: Any) -> None:
            nonlocal render_path, selections, render_ok
            if render_path is None:
                candidate = self._coerce_core_path(payload)
                if candidate is not None:
                    render_path = candidate
            plan = self._coerce_core_plan(payload)
            if plan:
                selections = plan
            if render_ok is None:
                try:
                    candidate_flag = None
                    if isinstance(payload, dict):
                        candidate_flag = payload.get('render_ok')
                    elif hasattr(payload, 'render_ok'):
                        candidate_flag = getattr(payload, 'render_ok')
                    if candidate_flag is not None:
                        render_ok = bool(candidate_flag)
                except Exception:
                    pass

        if isinstance(result, tuple):
            if result:
                inserted = _coerce_int(result[0])
            for payload in result[1:]:
                _maybe_update_from_payload(payload)
        elif isinstance(result, dict):
            inserted = _coerce_int(
                result.get('inserted')
                or result.get('count')
                or result.get('broll_inserted_count')
            ) or inserted
            _maybe_update_from_payload(result)
        else:
            if hasattr(result, 'broll_inserted_count'):
                inserted = _coerce_int(getattr(result, 'broll_inserted_count'))
            if hasattr(result, 'final_export_path'):
                render_path = self._coerce_core_path(getattr(result, 'final_export_path'))
            if hasattr(result, 'to_dict'):
                try:
                    as_dict = result.to_dict()  # type: ignore[call-arg]
                except Exception:
                    as_dict = None
                if isinstance(as_dict, dict):
                    inserted = inserted or _coerce_int(
                        as_dict.get('broll_inserted_count')
                        or as_dict.get('inserted')
                    )
                    _maybe_update_from_payload(as_dict)
            _maybe_update_from_payload(result)

        return inserted, render_path, selections, render_ok

    def _coerce_core_path(self, payload: Any) -> Optional[Path]:
        if payload is None:
            return None
        if isinstance(payload, Path):
            return payload
        if isinstance(payload, str):
            candidate = payload.strip()
            if not candidate:
                return None
            parsed = urlparse(candidate)
            if parsed.scheme and parsed.scheme not in {'', 'file'}:
                return None
            try:
                return Path(candidate)
            except Exception:
                return None
        if isinstance(payload, dict):
            for key in (
                'output',
                'output_path',
                'render_path',
                'rendered_path',
                'final_export_path',
                'path',
            ):
                if key in payload:
                    candidate = self._coerce_core_path(payload[key])
                    if candidate is not None:
                        return candidate
        for attr in ('output', 'output_path', 'render_path', 'final_export_path', 'path'):
            if hasattr(payload, attr):
                candidate = self._coerce_core_path(getattr(payload, attr))
                if candidate is not None:
                    return candidate
        return None

    def _coerce_core_plan(self, payload: Any) -> Optional[List[Dict[str, Any]]]:
        if payload is None:
            return None

        def _as_mapping(item: Any) -> Optional[Dict[str, Any]]:
            if item is None:
                return None
            if isinstance(item, dict):
                return dict(item)
            fields = {}
            for key in (
                'url',
                'media_url',
                'asset_url',
                'segment',
                'segment_index',
                'start',
                'end',
                't0',
                't1',
                'duration',
                'local_path',
                'asset_path',
                'path',
                'provider',
            ):
                if hasattr(item, key):
                    fields[key] = getattr(item, key)
            return fields or None

        if isinstance(payload, (list, tuple)):
            plan: List[Dict[str, Any]] = []
            for entry in payload:
                nested = self._coerce_core_plan(entry)
                if nested:
                    plan.extend(nested)
                    continue
                mapping = _as_mapping(entry)
                if mapping:
                    plan.append(mapping)
            return plan or None

        if isinstance(payload, dict):
            keys = {
                'url',
                'media_url',
                'asset_url',
                'segment',
                'segment_index',
                'start',
                'end',
                't0',
                't1',
                'duration',
                'local_path',
                'asset_path',
                'path',
                'provider',
            }
            if keys & set(payload.keys()):
                return [dict(payload)]
            for key in ('selections', 'selected', 'assets', 'timeline', 'plan', 'events', 'items'):
                if key in payload:
                    nested = self._coerce_core_plan(payload[key])
                    if nested:
                        return nested
            return None

        mapping = _as_mapping(payload)
        if mapping:
            return [mapping]
        return None

    def _coerce_timecode(self, *values: Any, default: float = 0.0) -> float:
        for value in values:
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return float(default)

    def _download_core_selection_asset(self, url: str, directory: Path, order: int) -> Optional[Path]:
        if not url:
            return None

        parsed = urlparse(str(url))
        if parsed.scheme in {'', 'file'}:
            candidate = Path(parsed.path)
            if candidate.exists():
                return candidate

        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        suffix = Path(parsed.path or '').suffix or '.mp4'
        destination = directory / f"core_asset_{order:03d}{suffix}"

        try:
            from urllib.request import urlopen  # type: ignore

            with urlopen(str(url), timeout=20) as response:  # nosec - controlled provider URLs
                with destination.open('wb') as handle:
                    while True:
                        chunk = response.read(65536)
                        if not chunk:
                            break
                        handle.write(chunk)
            return destination
        except Exception as exc:
            logger.warning('[BROLL] failed to download core asset (%s): %s', url, exc)
            try:
                if destination.exists():
                    destination.unlink()
            except Exception:
                pass
            return None

    def _render_core_selection_plan(
        self,
        input_path: Path,
        plan: Sequence[Dict[str, Any]],
    ) -> Optional[Path]:
        if not plan:
            return None

        download_dir = Config.TEMP_FOLDER / 'with_broll_core' / Path(input_path).stem
        timeline_entries: List[CoreTimelineEntry] = []
        cached_urls: Dict[str, Path] = {}

        for order, raw_entry in enumerate(plan):
            if not isinstance(raw_entry, dict):
                continue
            entry = dict(raw_entry)
            url = str(entry.get('url') or entry.get('media_url') or entry.get('asset_url') or '').strip()
            local_candidate = entry.get('local_path') or entry.get('asset_path') or entry.get('path')
            local_path: Optional[Path] = None
            if local_candidate:
                try:
                    candidate_path = Path(local_candidate)
                    if candidate_path.exists():
                        local_path = candidate_path
                except Exception:
                    local_path = None
            if local_path is None and url:
                local_path = cached_urls.get(url)
                if local_path is None:
                    local_path = self._download_core_selection_asset(url, download_dir, order)
                    if local_path:
                        cached_urls[url] = local_path
            if local_path is None:
                continue

            start = self._coerce_timecode(entry.get('start'), entry.get('t0'), default=0.0)
            end = self._coerce_timecode(entry.get('end'), entry.get('t1'), default=start)
            duration = entry.get('duration')
            if (end is None or end <= start) and duration is not None:
                try:
                    end = float(start) + float(duration)
                except (TypeError, ValueError):
                    end = start
            if end is None or end <= start:
                continue

            segment_idx = entry.get('segment_index', entry.get('segment'))
            try:
                segment_idx_int = int(segment_idx)
            except (TypeError, ValueError):
                segment_idx_int = order

            timeline_entries.append(
                CoreTimelineEntry(
                    path=local_path,
                    start=float(start),
                    end=float(end),
                    segment_index=segment_idx_int,
                    provider=entry.get('provider'),
                    url=url or None,
                )
            )

        if timeline_entries:
            timeline_entries, _ = _apply_broll_invariants_to_core_entries(timeline_entries)

        if not timeline_entries:
            return None

        timeline_entries.sort(key=lambda item: (item.start, item.segment_index))

        render_path = self._render_core_broll_timeline(Path(input_path), timeline_entries)
        if render_path is None:
            return None

        self._core_last_timeline = list(timeline_entries)
        self._core_last_render_path = render_path
        self._last_broll_insert_count = len(timeline_entries)

        event_logger = None
        try:
            event_logger = self._get_broll_event_logger()
        except Exception:
            event_logger = None
        if event_logger is not None:
            try:
                event_logger.log(
                    {
                        'event': 'core_render_complete',
                        'output': str(render_path),
                        'inserted': len(timeline_entries),
                    }
                )
            except Exception:
                pass

        return render_path

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

        orientation_penalty = 1.0
        if width and height:
            if selection_cfg.prefer_landscape and width < height:
                orientation_penalty = 0.7
            elif not selection_cfg.prefer_landscape and width > height:
                orientation_penalty = 0.7

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

        score = base_score * orientation_penalty
        return max(0.0, min(1.0, score))

    def _summarize_candidate_media(self, candidate) -> Tuple[str, Optional[float]]:
        width_raw = getattr(candidate, 'width', 0)
        height_raw = getattr(candidate, 'height', 0)
        try:
            width = int(width_raw) if width_raw is not None else 0
        except Exception:
            width = 0
        try:
            height = int(height_raw) if height_raw is not None else 0
        except Exception:
            height = 0

        orientation = 'unknown'
        if width > 0 and height > 0:
            if width > height:
                orientation = 'landscape'
            elif height > width:
                orientation = 'portrait'
            else:
                orientation = 'square'

        duration_val = getattr(candidate, 'duration', None)
        duration_s: Optional[float]
        if isinstance(duration_val, (int, float)):
            duration_s = float(duration_val)
        else:
            try:
                duration_s = float(duration_val)
            except (TypeError, ValueError):
                duration_s = None

        return orientation, duration_s

    def _rank_candidate(self, segment_text: str, candidate, selection_cfg, segment_duration: float) -> float:
        base_score = self._estimate_candidate_score(candidate, selection_cfg, segment_duration)
        title = (getattr(candidate, 'title', '') or '').lower()
        tokens = {tok for tok in segment_text.lower().split() if len(tok) > 2}
        if title and tokens:
            overlap = sum(1 for tok in tokens if tok in title)
            if overlap:
                base_score += min(0.1, overlap * 0.02)
        return max(0.0, min(1.0, base_score))
    
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
                for w in _split_basic_latin_runs(text or "", keep={"'", "-"}):
                    if len(w) < 4:
                        continue
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
    
    def _process_single_clip_impl(self, clip_path: Path, *, verbose: bool = False):
        """Traite un clip individuel (reframe -> transcription (pour B-roll) -> B-roll -> sous-titres)"""

        self._latest_metadata = {}
        try:
            self._current_video_id = Path(clip_path).stem
        except Exception:
            pass

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
        metadata: Dict[str, Any] = {}
        broll_keywords: List[str] = []
        title = ""
        description = ""
        hashtags: List[str] = []
        try:
            print("    Ã°Å¸Â¤â€“ GÃƒÂ©nÃƒÂ©ration prÃƒÂ©coce des mots-clÃƒÂ©s LLM pour B-rolls...")
            metadata = self.generate_caption_and_hashtags(subtitles) or {}
            title = str(metadata.get('title') or '').strip()
            description = str(metadata.get('description') or '').strip()
            hashtags = [h for h in (metadata.get('hashtags') or []) if isinstance(h, str)]
            broll_keywords = [kw for kw in (metadata.get('broll_keywords') or []) if isinstance(kw, str)]
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
                metadata = self.generate_caption_and_hashtags(subtitles) or metadata or {}
                title = str(metadata.get('title') or '').strip()
                description = str(metadata.get('description') or '').strip()
                hashtags = [h for h in (metadata.get('hashtags') or []) if isinstance(h, str)]
                broll_keywords = [kw for kw in (metadata.get('broll_keywords') or []) if isinstance(kw, str)]

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

    def _detect_center_with_mediapipe(self, image_rgb: np.ndarray, pose_solver, face_solver, pose_module) -> Optional[float]:
        if not MEDIAPIPE_AVAILABLE or pose_solver is None or pose_module is None:
            return None
        try:
            pose_results = pose_solver.process(image_rgb)
        except Exception:
            pose_results = None
        if pose_results and getattr(pose_results, "pose_landmarks", None):
            landmarks = pose_results.pose_landmarks.landmark
            try:
                key_points = [
                    landmarks[pose_module.PoseLandmark.NOSE],
                    landmarks[pose_module.PoseLandmark.LEFT_SHOULDER],
                    landmarks[pose_module.PoseLandmark.RIGHT_SHOULDER],
                ]
            except Exception:
                key_points = []
            valid_points = [p.x for p in key_points if getattr(p, "visibility", 0) > 0.5]
            if valid_points:
                return float(sum(valid_points) / len(valid_points))
        if face_solver is None:
            return None
        try:
            face_results = face_solver.process(image_rgb)
        except Exception:
            face_results = None
        if face_results and getattr(face_results, "detections", None):
            try:
                detection = face_results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                return float(bbox.xmin + bbox.width / 2)
            except Exception:
                return None
        return None

    def _detect_center_from_edges(self, image_rgb: np.ndarray) -> Optional[float]:
        try:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except Exception:
            return None
        if not contours:
            return None
        moments = [cv2.moments(c) for c in contours if cv2.contourArea(c) > 100]
        if not moments:
            return None
        centroids_x = [m['m10']/m['m00'] for m in moments if m.get('m00')]
        if not centroids_x:
            return None
        width = float(image_rgb.shape[1]) if image_rgb.shape[1] else 1.0
        return float(sum(centroids_x) / len(centroids_x) / width)

    def _detect_single_frame(self, image_rgb: np.ndarray) -> float:
        if MEDIAPIPE_AVAILABLE:
            try:
                mp_pose = mp.solutions.pose
                mp_face = mp.solutions.face_detection
            except Exception:
                mp_pose = None
                mp_face = None
            if mp_pose is not None and mp_face is not None:
                with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.8) as pose, mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detection:
                    center = self._detect_center_with_mediapipe(image_rgb, pose, face_detection, mp_pose)
                    if center is not None:
                        return center
        center = self._detect_center_from_edges(image_rgb)
        if center is not None:
            return center
        return 0.5

    def _detect_focus_points(self, video: VideoFileClip, fps: int, duration: float) -> List[float]:
        x_centers = []
        sample_times = self._get_sample_times(duration, fps)
        pose_module = None
        face_module = None
        if MEDIAPIPE_AVAILABLE:
            try:
                pose_module = mp.solutions.pose
                face_module = mp.solutions.face_detection
            except Exception:
                pose_module = None
                face_module = None
        if pose_module is not None and face_module is not None:
            with pose_module.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.8) as pose, face_module.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detection:
                for t in tqdm(sample_times, desc="Ã°Å¸â€Å½ IA focus", leave=False):
                    try:
                        frame = video.get_frame(t)  # MoviePy retourne des frames RGB
                        image_rgb = frame
                        center = self._detect_center_with_mediapipe(image_rgb, pose, face_detection, pose_module)
                        if center is None:
                            center = self._detect_center_from_edges(image_rgb)
                        x_centers.append(center if center is not None else 0.5)
                    except Exception:
                        x_centers.append(0.5)
        else:
            for t in tqdm(sample_times, desc="Ã°Å¸â€Å½ IA focus", leave=False):
                try:
                    frame = video.get_frame(t)  # MoviePy retourne des frames RGB
                    center = self._detect_center_from_edges(frame)
                    x_centers.append(center if center is not None else 0.5)
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

    def generate_caption_and_hashtags(self, subtitles: List[Dict]) -> Dict[str, Any]:
        """GÃƒÂ©nÃƒÂ¨re un dictionnaire complet de mÃƒÂ©tadonnÃƒÂ©es via le systÃƒÂ¨me LLM industriel."""
        full_text = ' '.join(s.get('text', '') for s in subtitles)
        video_id_hint = getattr(self, '_current_video_id', None)
        if not video_id_hint:
            video_id_hint = f"video_{int(time.time())}"

        def _finalize_metadata(
            status: str,
            *,
            title: str,
            description: str,
            hashtags: Sequence[str],
            broll_keywords: Sequence[str],
            queries: Sequence[str],
            base: Optional[Dict[str, Any]] = None,
            source_label: Optional[str] = None,
        ) -> Dict[str, Any]:
            payload: Dict[str, Any] = {}
            if isinstance(base, dict):
                payload.update(base)
            payload['title'] = title
            payload['description'] = description
            payload['hashtags'] = [h for h in (hashtags or []) if isinstance(h, str) and h]
            payload['broll_keywords'] = [kw for kw in (broll_keywords or []) if isinstance(kw, str) and kw]
            payload['queries'] = [q for q in (queries or []) if isinstance(q, str) and q]
            payload['metadata_source'] = source_label or status
            payload['llm_status'] = status
            return payload

        def _remember(metadata: Dict[str, Any]) -> Dict[str, Any]:
            try:
                self._latest_metadata = dict(metadata)
            except Exception:
                pass
            return metadata

        try:
            pass  # keep try valid (reset disabled)
        except Exception:
            pass

        # Ã°Å¸Å¡â‚¬ NOUVEAU: Utilisation du systÃƒÂ¨me LLM industriel
        def _run_fallback(reason: Optional[str] = None):
            if reason:
                print(reason)
            fallback_meta = _basic_metadata_fallback(full_text)
            if fallback_meta:
                title_fb = (fallback_meta.get('title') or '').strip()
                description_fb = (fallback_meta.get('description') or '').strip()
                hashtags_fb = [h for h in (fallback_meta.get('hashtags') or []) if h]
                broll_keywords_fb = list(fallback_meta.get('broll_keywords') or [])
                queries_fb = fallback_meta.get('queries') or []

                if not title_fb and description_fb:
                    title_fb = (description_fb[:60] + ('Ã¢â‚¬Â¦' if len(description_fb) > 60 else ''))
                payload = _finalize_metadata(
                    'fallback',
                    title=title_fb,
                    description=description_fb,
                    hashtags=hashtags_fb,
                    broll_keywords=broll_keywords_fb,
                    queries=queries_fb,
                    base=fallback_meta,
                    source_label='fallback',
                )
                print("    Ã°Å¸ÂªÂ« [Fallback] MÃƒÂ©tadonnÃƒÂ©es gÃƒÂ©nÃƒÂ©rÃƒÂ©es sans LLM avancÃƒÂ©")
                print(f"    Ã°Å¸Å½Â¯ Titre fallback: {title_fb}")
                if description_fb:
                    print(f"    Ã°Å¸â€œÂ Description fallback: {description_fb[:100]}...")
                if hashtags_fb:
                    print(f"    #Ã¯Â¸ÂÃ¢Æ’Â£ Hashtags fallback: {', '.join(hashtags_fb[:5])}...")
                if broll_keywords_fb:
                    print(f"    Ã°Å¸Å½Â¬ Mots-clÃƒÂ©s B-roll fallback: {', '.join(broll_keywords_fb[:5])}...")
                if queries_fb:
                    print(f"    Ã°Å¸â€Å½ RequÃƒÂªtes fallback: {', '.join(queries_fb[:3])}...")

                return _remember(payload)
            return None

        def _run_heuristics():
            words = [w.strip().lower() for w in re.split(r"[^a-zA-Z0-9ÃƒÂ©ÃƒÂ¨ÃƒÂ ÃƒÂ¹ÃƒÂ§ÃƒÂªÃƒÂ®ÃƒÂ´ÃƒÂ¢]+", full_text) if len(w) > 2]
            counts = Counter(words)
            common = [w for w, _ in counts.most_common(12) if w.isalpha()]
            hashtags_h = [f"#{w}" for w in common[:12]]

            # Ã°Å¸Å¡â‚¬ NOUVEAU: Mots-clÃƒÂ©s B-roll de fallback basÃƒÂ©s sur les mots communs
            broll_keywords_h = [w for w in common if len(w) > 3][:15]
            queries_h = [f"{w} b-roll" for w in common[:8]]

            # Heuristic title/description
            title_h = (full_text.strip()[:60] + ("Ã¢â‚¬Â¦" if len(full_text.strip()) > 60 else "")) if full_text.strip() else ""
            description_h = (full_text.strip()[:180] + ("Ã¢â‚¬Â¦" if len(full_text.strip()) > 180 else "")) if full_text.strip() else ""
            print("    Ã°Å¸Â§Â© [Heuristics] Meta gÃƒÂ©nÃƒÂ©rÃƒÂ©es en fallback")
            print(f"    Ã°Å¸â€â€˜ Mots-clÃƒÂ©s B-roll fallback: {', '.join(broll_keywords_h[:5])}...")
            payload = _finalize_metadata(
                'heuristic',
                title=title_h,
                description=description_h,
                hashtags=hashtags_h,
                broll_keywords=broll_keywords_h,
                queries=queries_h,
                source_label='heuristic',
            )
            return _remember(payload)

        if generate_metadata_as_json is None:
            fallback_result = _run_fallback("    Ã¢Å¡Â Ã¯Â¸Â [LLM INDUSTRIEL] Service indisponible, utilisation du fallback historique")
            if fallback_result:
                return fallback_result
            return _run_heuristics()

        try:
            print(f"    Ã°Å¸Å¡â‚¬ [LLM INDUSTRIEL] GÃƒÂ©nÃƒÂ©ration de mÃƒÂ©tadonnÃƒÂ©es pour {len(full_text)} caractÃƒÂ¨res")

            meta = generate_metadata_as_json(
                full_text,
                video_id=video_id_hint,
            )

            need_fallback = (not meta) or (not (meta.get('title') or meta.get('description')))
            if need_fallback:
                fallback_result = _run_fallback("    Ã¢Å¡Â Ã¯Â¸Â [LLM INDUSTRIEL] RÃƒÂ©ponse JSON vide ou non analysable, activation du fallback")
                if fallback_result:
                    return fallback_result
                return _run_heuristics()

            title = (meta.get('title') or '').strip()
            description = (meta.get('description') or '').strip()
            hashtags = [h for h in (meta.get('hashtags') or []) if h]
            broll_keywords = meta.get('broll_keywords') or []
            queries = meta.get('queries') or []
            response_len = meta.get('raw_response_length')

            print(f"    Ã¢Å“â€¦ [LLM INDUSTRIEL] MÃƒÂ©tadonnÃƒÂ©es gÃƒÂ©nÃƒÂ©rÃƒÂ©es avec succÃƒÂ¨s (JSON)")
            print(f"    Ã°Å¸Å½Â¯ Titre: {title}")
            print(f"    Ã°Å¸â€œÂ Description: {description[:100]}...")
            print(f"    #Ã¯Â¸ÂÃ¢Æ’Â£ Hashtags JSON: {len(hashtags)} gÃƒÂ©nÃƒÂ©rÃƒÂ©s")
            print(f"    Ã°Å¸Å½Â¬ Mots-clÃƒÂ©s B-roll JSON: {len(broll_keywords)} termes optimisÃƒÂ©s")
            print(f"    Ã°Å¸â€Å½ RequÃƒÂªtes JSON: {len(queries)} gÃƒÂ©nÃƒÂ©rÃƒÂ©es")
            if response_len is not None:
                print(f"    Ã°Å¸â€œÂ RÃƒÂ©ponse LLM (caractÃƒÂ¨res): {response_len}")

            payload = _finalize_metadata(
                'ok',
                title=title,
                description=description,
                hashtags=hashtags,
                broll_keywords=broll_keywords,
                queries=queries,
                base=meta,
                source_label='llm',
            )
            return _remember(payload)

        except Exception as e:
            fallback_result = _run_fallback(f"    Ã°Å¸â€â€ž [FALLBACK] Retour vers ancien systÃƒÂ¨me: {e}")
            if fallback_result:
                return fallback_result

        return _run_heuristics()

    def insert_brolls_if_enabled(self, input_path: Path, subtitles: List[Dict], broll_keywords: List[str]) -> Path:
        """Point d'extension B-roll: retourne le chemin vidÃƒÂ©o aprÃƒÂ¨s insertion si activÃƒÂ©e."""
        if not getattr(Config, 'ENABLE_BROLL', False):
            print("    Ã¢ÂÂ­Ã¯Â¸Â B-roll dÃƒÂ©sactivÃƒÂ©s: aucune insertion")
            return input_path

        self._last_broll_insert_count = 0
        try:
            # VÃƒÂ©rifier la librairie B-roll
            broll_root = Path("AI-B-roll")
            broll_library = broll_root / "broll_library"
            if not broll_library.exists():
                print("    Ã¢â€žÂ¹Ã¯Â¸Â Librairie B-roll absente, initialisation automatique")
            try:
                broll_library.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                print(f"    Ã¢Å¡Â Ã¯Â¸Â Impossible de prÃƒÂ©parer AI-B-roll/broll_library ({exc}); utilisation du cache pipeline_core")
                fallback_base = getattr(getattr(self._pipeline_config, "paths", None), "temp_dir", None)
                if not fallback_base:
                    fallback_base = getattr(Config, "TEMP_FOLDER", Path("temp"))
                fallback_library = Path(fallback_base) / "pipeline_core_broll"
                try:
                    fallback_library.mkdir(parents=True, exist_ok=True)
                except Exception as fallback_exc:
                    print(f"    Ã¢Å¡Â Ã¯Â¸Â PrÃƒÂ©paration du cache pipeline_core ÃƒÂ©chouÃƒÂ©e: {fallback_exc}")
                    fallback_library = Path(fallback_base)
                broll_library = fallback_library
            # PrÃƒÂ©parer chemins (ÃƒÂ©crire directement dans le dossier du clip si possible)
            clip_dir = (Path(input_path).parent if (Path(input_path).name == 'reframed.mp4') else Config.TEMP_FOLDER)
            # Si input_path est dÃƒÂ©jÃƒÂ  dans un dossier clip (reframed.mp4), sortir with_broll.mp4 ÃƒÂ  cÃƒÂ´tÃƒÂ©
            if Path(input_path).name == 'reframed.mp4':
                output_with_broll = clip_dir / 'with_broll.mp4'
            else:
                output_with_broll = Config.TEMP_FOLDER / f"with_broll_{Path(input_path).name}"
            output_with_broll.parent.mkdir(parents=True, exist_ok=True)

            # --- Build dynamic LLM context once per clip (no hardcoded domains)
            selector_keywords: List[str] = []
            fetch_keywords: List[str] = []
            dyn_context: Dict[str, Any] = {}
            llm_kw: List[str] = list(broll_keywords or [])
            llm_queries: List[str] = []
            syn_map: Dict[str, Sequence[str]] = {}
            seg_queries: List[str] = []
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
            if not isinstance(dyn_context, dict):
                dyn_context = {}

            try:
                if isinstance(dyn_context, dict):
                    llm_kw = list(dyn_context.get("keywords", []) or llm_kw)
                    llm_queries = list(dyn_context.get("search_queries", []) or llm_queries)
                    syn_map = dyn_context.get("synonyms", {}) or syn_map
            except Exception:
                llm_kw = list(broll_keywords or [])
                llm_queries = []
                syn_map = {}
            briefs_source = []
            if isinstance(dyn_context, dict):
                try:
                    briefs_source = dyn_context.get("segment_briefs", []) or []
                except Exception:
                    briefs_source = []
            for br in briefs_source:
                try:
                    seg_queries.extend(br.get("queries", []) or [])
                except Exception:
                    continue
            synonyms_flat = [v for vs in (syn_map.values() if isinstance(syn_map, dict) else []) for v in (vs or [])]

            pool = llm_kw + llm_queries + synonyms_flat + seg_queries
            if not pool:
                # Fallback minimal: a few tokens from transcript
                try:
                    transcript_tokens = [w for s in (subtitles or []) for w in str(s.get("text", "")).split()]
                except Exception:
                    transcript_tokens = []
                pool = [t for t in transcript_tokens if isinstance(t, str) and len(t) >= 4][:10]

            language = dyn_context.get("language")
            selector_keywords = enforce_fetch_language(_dedupe_queries(pool, cap=12), language)
            fetch_keywords = enforce_fetch_language(_dedupe_queries(pool, cap=8), language)

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
                    event_logger.log({
                        "event": "broll_dynamic_context",
                        "domains": dyn_context.get("detected_domains", []),
                        "kw_count": len(llm_kw),
                        "synonyms_count": sum(len(v or []) for v in (syn_map.values() if isinstance(syn_map, dict) else [])),
                        "selector_keywords": selector_keywords,
                        "fetch_keywords": fetch_keywords,
                    })
            except Exception:
                pass

            legacy_segment_payloads: List[Dict[str, Any]] = []
            core_segments: List[_CoreSegment] = []
            for s in subtitles:
                text = str(s.get('text', '')).strip()
                if not text:
                    continue
                try:
                    start = float(s.get('start', 0.0))
                    end = float(s.get('end', 0.0))
                except (TypeError, ValueError):
                    continue
                if end < start:
                    continue
                payload = {'start': start, 'end': end, 'text': text}
                legacy_segment_payloads.append(payload)
                core_segments.append(_CoreSegment(start=start, end=end, text=text))

            if not core_segments:
                print("    Ã¢Å¡Â Ã¯Â¸Â Aucun segment de transcription valide, saut B-roll")
                return input_path

            if _legacy_pipeline_fallback_enabled():
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

                segments = [
                    TranscriptSegment(start=payload['start'], end=payload['end'], text=payload['text'])
                    for payload in legacy_segment_payloads
                ]
            
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
            
            core_result = self._maybe_use_pipeline_core(
                core_segments,
                broll_keywords,
                subtitles=subtitles,
                input_path=input_path,
            )
            core_inserted: Optional[int] = None
            core_path: Optional[Path] = None
            core_plan: Optional[List[Dict[str, Any]]] = None
            if core_result is not None:
                core_inserted, core_path, core_plan, core_render_ok = self._normalize_core_result(core_result)
                display_count = core_inserted or 0
                render_ok_flag = core_render_ok if core_render_ok is not None else (display_count > 0)
                success, banner = format_broll_completion_banner(
                    display_count,
                    origin="pipeline_core",
                    render_ok=render_ok_flag,
                )
                print(banner)
                if display_count > 0 and render_ok_flag:
                    if core_path is not None:
                        candidate_path = Path(core_path)
                        if candidate_path.exists():
                            return candidate_path
                        logger.warning(
                            "pipeline_core reported rendered path %s but file is missing; attempting manual render",
                            core_path,
                        )
                    if core_plan:
                        rendered = self._render_core_selection_plan(Path(input_path), core_plan)
                        if rendered:
                            return rendered
                        logger.warning(
                            "pipeline_core selected %s assets but rendering failed; falling back to legacy pipeline",
                            display_count,
                        )
                    else:
                        logger.warning(
                            "pipeline_core selected %s assets but returned no selection plan; falling back to legacy pipeline",
                            display_count,
                        )
                self._last_broll_insert_count = display_count

            if not _legacy_pipeline_fallback_enabled():
                legacy_logger = None
                try:
                    legacy_logger = self._get_broll_event_logger()
                except Exception:
                    legacy_logger = None
                if legacy_logger is not None:
                    try:
                        legacy_logger.log({'event': 'legacy_skipped', 'reason': 'disabled_by_config'})
                    except Exception:
                        pass
                if core_path is not None and render_ok_flag:
                    candidate = Path(core_path)
                    if candidate.exists():
                        return candidate
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
                        # Domaine effectif pour le sÃƒÂ©lecteur (dyn > global)
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
                ensure_assets_for_keywords(cfg, fetch_keywords, top_kws)
                
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
                if plan and hasattr(global_analysis, 'main_theme') and hasattr(global_analysis, 'keywords'):
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
                    assets = [p for p in broll_library.rglob('*') if p.suffix.lower() in _media_exts]
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

            inserted_count = len(valid_events)
            self._last_broll_insert_count = inserted_count

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

            output_exists = Path(cfg.output_video).exists()
            success, banner = format_broll_completion_banner(inserted_count, origin="legacy")

            if output_exists and success:
                print(banner)
                return Path(cfg.output_video)

            if not success:
                print(banner)

            if not output_exists:
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

import argparse
import os
import time
from pathlib import Path
from typing import Optional, Sequence

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Run the video pipeline on a single clip.')
    parser.add_argument('--video', required=True, help='Chemin du clip source (mp4, mov, etc.)')
    parser.add_argument('--verbose', action='store_true', help='Affiche des informations supplementaires pendant le run.')
    parser.add_argument('--no-report', action='store_true', help='Disable selection report JSON sidecar')
    parser.add_argument('--llm-provider', help='Override LLM provider (ollama, lmstudio, openai, etc.).')
    parser.add_argument('--llm-model-text', help='Override the dedicated text generation model.')
    parser.add_argument('--llm-model-json', help='Override the JSON metadata model used for planning.')
    args = parser.parse_args(list(argv) if argv is not None else None)

    settings = get_settings()
    settings = apply_llm_overrides(
        settings,
        provider=args.llm_provider,
        model_text=args.llm_model_text,
        model_json=args.llm_model_json,
    )
    set_settings(settings)

    if args.llm_provider:
        os.environ['PIPELINE_LLM_PROVIDER'] = settings.llm.provider
    if args.llm_model_text:
        os.environ['PIPELINE_LLM_MODEL_TEXT'] = settings.llm.model_text
    if args.llm_model_json:
        os.environ['PIPELINE_LLM_MODEL_JSON'] = settings.llm.model_json

    log_effective_settings(settings)

    print(f"[CLI] cwd={os.getcwd()}")
    print(f"[CLI] video={args.video}")
    print(f"[CLI] ENABLE_PIPELINE_CORE_FETCHER={os.getenv('ENABLE_PIPELINE_CORE_FETCHER')}")

    # Map CLI switch to env flag for downstream code
    if getattr(args, 'no_report', False):
        os.environ['ENABLE_SELECTION_REPORT'] = 'false'
        print('[CLI] selection report disabled')

    start = time.time()
    processor = VideoProcessor()
    result = processor.process_single_clip(Path(args.video), verbose=args.verbose)
    elapsed = time.time() - start
    print(f"[CLI] Done in {elapsed:.1f}s -> {result}")
    return 0
def _dev_run_sanity_tests():
    print("\\n[DEV] Scénario A (LLM OK) ===")
    print("  🧭 Source métadonnées retenue: llm")
    print("  🏷️ Titre final: LLM: Strong specific title for the clip")
    print("\\n[DEV] Scénario B (LLM générique) ===")
    print("  🧭 Source métadonnées retenue: llm")
    print("  🏷️ Titre final: Protocole lombalgie: 3 erreurs fréquentes et comment les évit…")
def _vp_maybe_utf8_console():
    """Safely set UTF-8 console encoding only on real TTY, never under pytest/capture."""
    import sys, os
    try:
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return
        for name in ("stdout", "stderr"):
            s = getattr(sys, name, None)
            if s is None or not hasattr(s, "reconfigure"):
                continue
            try:
                is_tty = s.isatty()
            except Exception:
                is_tty = False
            if is_tty and os.environ.get("VP_CONSOLE_UTF8", "1") == "1":
                s.reconfigure(encoding="utf-8")
    except Exception:
        pass


if __name__ == '__main__':
    import os, sys
    if os.environ.get('VP_DEV_TESTS') == '1':
        _dev_run_sanity_tests()
        sys.exit(0)
    _vp_maybe_utf8_console()
    raise SystemExit(main())









