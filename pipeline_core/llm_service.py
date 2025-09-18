"""LLM metadata helper built on top of the existing integration stack."""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
import logging
import sys
from pathlib import Path
import importlib.util
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
UTILS_DIR = PROJECT_ROOT / 'utils'
if UTILS_DIR.exists():
    sys.path.insert(1, str(UTILS_DIR))
if 'utils' in sys.modules:
    del sys.modules['utils']
if UTILS_DIR.exists():
    spec = importlib.util.spec_from_file_location('utils', UTILS_DIR / '__init__.py', submodule_search_locations=[str(UTILS_DIR)])
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules['utils'] = module

from typing import Dict, List, Optional, Sequence, Tuple

if UTILS_DIR.exists():
    spec_pi = importlib.util.spec_from_file_location('utils.pipeline_integration', UTILS_DIR / 'pipeline_integration.py')
    if spec_pi and spec_pi.loader:
        _module_pi = importlib.util.module_from_spec(spec_pi)
        spec_pi.loader.exec_module(_module_pi)
        create_pipeline_integration = _module_pi.create_pipeline_integration
    else:
        raise ImportError('Cannot load utils.pipeline_integration')
else:
    raise ImportError('utils directory missing: ' + str(UTILS_DIR))

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class SubtitleSegment:
    """Lightweight representation of a subtitle segment."""

    start: float
    end: float
    text: str

    @classmethod
    def from_mapping(cls, payload: Dict) -> "SubtitleSegment":
        return cls(
            start=float(payload.get("start", 0.0) or 0.0),
            end=float(payload.get("end", 0.0) or 0.0),
            text=str(payload.get("text", "")).strip(),
        )


@dataclass(slots=True)
class LLMMetadata:
    """Structured result returned by the metadata generator."""

    title: str
    description: str
    hashtags: List[str]
    broll_keywords: List[str]
    raw_payload: Dict


class LLMMetadataGeneratorService:
    """Thin wrapper that memoises the heavy integration initialisation."""

    _lock: Lock = Lock()
    _shared_integration = None
    _init_count = 0

    def __init__(self, *, reuse_shared: bool = True):
        self._reuse_shared = reuse_shared
        self._integration = None

    def _get_integration(self):
        if self._reuse_shared:
            with self._lock:
                if self._shared_integration is None:
                    logger.info("[LLM] Initialising shared pipeline integration")
                    try:
                        self._shared_integration = create_pipeline_integration()
                    except Exception:  # pragma: no cover - defensive logging
                        logger.exception("[LLM] Failed to initialise shared pipeline integration")
                        raise
                    LLMMetadataGeneratorService._init_count += 1
            return self._shared_integration
        if self._integration is None:
            logger.info("[LLM] Initialising local pipeline integration")
            try:
                self._integration = create_pipeline_integration()
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("[LLM] Failed to initialise local pipeline integration")
                raise
        return self._integration

    def generate_metadata(
        self,
        segments: Sequence[Dict],
        *,
        video_id: Optional[str] = None,
    ) -> Optional[LLMMetadata]:
        """Generate metadata and B-roll keywords from subtitle segments."""

        if not segments:
            return None

        # Normalise input for the integration layer
        normalised: List[SubtitleSegment] = [
            SubtitleSegment.from_mapping(seg)
            for seg in segments
            if seg and seg.get("text")
        ]
        if not normalised:
            return None

        transcript = " ".join(seg.text for seg in normalised if seg.text)
        timestamps: List[Tuple[float, float]] = [
            (seg.start, seg.end) for seg in normalised if seg.end >= seg.start
        ]

        integration = self._get_integration()
        result = integration.process_video_transcript(
            transcript=transcript,
            video_id=video_id or f"video_{id(integration):x}",
            segment_timestamps=timestamps if len(timestamps) > 1 else None,
        )

        if not result or not result.get("success"):
            return None

        metadata = result.get("metadata") or {}
        broll_data = result.get("broll_data") or {}

        title = str(metadata.get("title", "")).strip()
        description = str(metadata.get("description", "")).strip()
        hashtags = [h for h in metadata.get("hashtags", []) if isinstance(h, str) and h]
        broll_keywords = [
            kw
            for kw in (broll_data.get("keywords") or metadata.get("keywords") or [])
            if isinstance(kw, str) and kw.strip()
        ]

        logger.info("[LLM] Metadata generated", extra={'hashtags': len(hashtags), 'broll_keywords': len(broll_keywords)})
        return LLMMetadata(
            title=title,
            description=description,
            hashtags=hashtags,
            broll_keywords=broll_keywords,
            raw_payload=result,
        )

    def generate_hints_for_segment(self, text: str, start: float, end: float) -> Dict:
        """Produce visual search hints for a transcript segment."""
        try:
            tokens = [t.lower() for t in re.findall(r"[A-Za-z']+", text) if len(t) > 3]
        except Exception:
            tokens = []
        stopwords = {
            'this', 'that', 'with', 'have', 'there', 'their', 'would', 'could', 'where', 'when',
            'these', 'those', 'which', 'about', 'because', 'being', 'while', 'after', 'before',
        }
        keywords = [t for t in tokens if t not in stopwords]
        base_queries = [
            "doctor talking with patient in clinic",
            "close-up stethoscope on chest",
            "nurse writing notes at hospital desk",
            "medical team walking in hospital corridor",
            "therapist speaking with client",
            "hands typing on laptop in hospital office",
            "MRI scanner room",
        ]
        mapping = {
            'patient': "doctor explaining results to patient in clinic",
            'doctor': "doctor discussing treatment plan at hospital desk",
            'brain': "brain scan visuals with medical monitors",
            'therapy': "therapist talking with client in calm office",
            'motivation': "person giving motivational talk on stage",
            'business': "entrepreneur presenting idea to team in office",
            'data': "data analyst reviewing charts on large screen",
            'training': "coach guiding person during workout session",
            'emotion': "close-up face expressing emotion indoors",
        }
        dynamic_queries = []
        for token in keywords:
            if token in mapping:
                dynamic_queries.append(mapping[token])
        if not dynamic_queries and keywords:
            take = keywords[:3]
            dynamic_queries.append(f"person discussing {' '.join(take)} in modern workspace")
        queries = list(dict.fromkeys(dynamic_queries + base_queries))[:8]
        synonyms = sorted(set(keywords[:6]))
        return {
            "queries": queries,
            "synonyms": synonyms or ["healthcare", "therapy", "consultation"],
            "filters": {"orientation": "landscape", "min_duration_s": 3.0},
        }

