"""LLM metadata helper built on top of the existing integration stack."""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
import importlib.util
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

FAST_TESTS = os.getenv('PIPELINE_FAST_TESTS') == '1'

if UTILS_DIR.exists() and not FAST_TESTS:
    spec_pi = importlib.util.spec_from_file_location('utils.pipeline_integration', UTILS_DIR / 'pipeline_integration.py')
    if spec_pi and spec_pi.loader:
        _module_pi = importlib.util.module_from_spec(spec_pi)
        spec_pi.loader.exec_module(_module_pi)
        create_pipeline_integration = _module_pi.create_pipeline_integration
    else:
        raise ImportError('Cannot load utils.pipeline_integration')
elif FAST_TESTS:
    def create_pipeline_integration(config=None):  # type: ignore[override]
        return object()
else:
    raise ImportError('utils directory missing: ' + str(UTILS_DIR))


logger = logging.getLogger(__name__)


# --- Robust JSON extraction from LLM responses -------------------------------
def _safe_parse_json(s: str) -> Dict[str, Any]:
    """Extract the first valid JSON object from text; return {} on failure.

    Uses a simple brace-matching scan to find the first top-level JSON object.
    """
    if not isinstance(s, str):
        return {}
    try:
        # Fast path: already valid JSON
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    if start < 0:
        return {}
    depth = 0
    for i, ch in enumerate(s[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                snippet = s[start:i+1]
                try:
                    return json.loads(snippet)
                except Exception:
                    break
    return {}

def build_dynamic_prompt(transcript_text: str, *, max_len: int = 1800) -> str:
    tx = (transcript_text or "")[:max_len]
    return f"""
RÔLE
Tu es planificateur B-roll pour vidéos verticales (TikTok/Shorts, 9:16).

OBJECTIF
À partir de la transcription, détecte le(s) domaine(s) librement (pas de liste fixe), puis génère :
1) des mots-clés et phrases-clés visuelles (scènes filmables) utiles aux banques vidéos,
2) des synonymes/variantes/termes proches pour CHAQUE mot-clé (2–4 max),
3) des requêtes de recherche (2–4 mots, provider-friendly),
4) des briefs segmentaires facultatifs.

CONTRAINTES
- Zéro domaine prédéfini. Déduis librement 1–3 “detected_domains” + confidence (0–1).
- Évite les anti-termes génériques : people, thing, nice, background, start, generic, template, stock.
- Priorise des requêtes concrètes et filmables : « sujet_action_contexte », objets précis, lieux identifiables.
- Fenêtres visuelles recommandées : 3–6 secondes. Format vertical.
- Si la langue de la transcription n’est pas l’anglais, produis les requêtes en langue d’origine + anglais.

RÉPONDS UNIQUEMENT EN JSON:
{{
  "detected_domains": [{{"name": "...", "confidence": 0.0}}],
  "language": "fr|en|…",
  "keywords": ["..."],
  "synonyms": {{ "keyword": ["variante1","variante2"] }},
  "search_queries": ["..."],
  "segment_briefs": [
    {{"segment_index": 0, "window_s": 4, "keywords": ["..."], "queries": ["..."]}}
  ],
  "notes": "pièges, anti-termes, risques"
}}

TRANSCRIPT (tronqué à 1500–2000 caractères):
{tx}
"""


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
    _shared_config = None
    _init_count = 0

    def __init__(self, *, reuse_shared: bool = True, config: Optional[Any] = None):
        self._reuse_shared = reuse_shared
        self._config = config
        self._integration = None

    @classmethod
    def get_shared(cls, config: Optional[Any] = None):
        """Return a service bound to the shared integration, initialising once."""
        service = cls(reuse_shared=True, config=config)
        service._get_integration()
        return service

    def _get_integration(self):
        config = self._config
        if FAST_TESTS:
            if self._reuse_shared:
                with self._lock:
                    if self._shared_integration is None:
                        logger.info('[LLM] FAST_TESTS stub integration initialised')
                        LLMMetadataGeneratorService._shared_integration = create_pipeline_integration(config)
                        LLMMetadataGeneratorService._shared_config = config
                        LLMMetadataGeneratorService._init_count += 1
                return self._shared_integration
            if self._integration is None:
                logger.info('[LLM] FAST_TESTS stub integration (local) initialised')
                self._integration = create_pipeline_integration(config)
            return self._integration

        if self._reuse_shared:
            with self._lock:
                if self._shared_integration is None:
                    logger.info('[LLM] Initialising shared pipeline integration')
                    try:
                        LLMMetadataGeneratorService._shared_integration = create_pipeline_integration(config)
                    except Exception:  # pragma: no cover - defensive logging
                        logger.exception('[LLM] Failed to initialise shared pipeline integration')
                        raise
                    LLMMetadataGeneratorService._shared_config = config
                    LLMMetadataGeneratorService._init_count += 1
            return self._shared_integration

        if self._integration is None:
            logger.info('[LLM] Initialising local pipeline integration')
            try:
                self._integration = create_pipeline_integration(config)
            except Exception:  # pragma: no cover - defensive logging
                logger.exception('[LLM] Failed to initialise local pipeline integration')
                raise
        return self._integration

    # --- Compatibility layer for plain text completions ---------------------
    def _complete_text(self, prompt: str) -> str:
        """Attempt to invoke a completion method on the underlying integration.

        Tries common method names; falls back to calling the optimized LLM engine
        when available (integration.llm._call_llm). Returns raw text.
        """
        integration = self._get_integration()

        # 1) Try common completion-shaped methods on integration directly
        for attr in ("complete_json", "complete", "chat", "generate"):
            fn = getattr(integration, attr, None)
            if callable(fn):
                try:
                    return fn(prompt)  # type: ignore[misc]
                except Exception:
                    continue

        # 2) Try going through the optimized LLM engine if exposed
        llm = getattr(integration, "llm", None)
        if llm is not None:
            # Prefer a public method if one exists
            for attr in ("complete_json", "complete", "generate"):
                fn = getattr(llm, attr, None)
                if callable(fn):
                    try:
                        return fn(prompt)  # type: ignore[misc]
                    except Exception:
                        continue
            # Fallback to private _call_llm used in optimized_llm.py
            call = getattr(llm, "_call_llm", None)
            if callable(call):
                ok, text, _err = call(prompt, temperature=0.1, max_tokens=800)
                if ok and isinstance(text, str):
                    return text

        raise RuntimeError("No compatible completion method available on integration")

    def generate_dynamic_context(self, transcript_text: str, *, max_len: int = 1800) -> Dict[str, Any]:
        """Domain detection + dynamic expansions (no hardcoded domains)."""
        prompt = build_dynamic_prompt(transcript_text, max_len=max_len)
        try:
            raw = self._complete_text(prompt)
            data = _safe_parse_json(raw)
        except Exception:
            logger.exception("[LLM] dynamic context failed")
            data = {}
        # Guard rails
        data.setdefault("detected_domains", [])
        data.setdefault("language", None)
        data.setdefault("keywords", [])
        data.setdefault("synonyms", {})
        data.setdefault("search_queries", [])
        data.setdefault("segment_briefs", [])
        return data

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

        logger.info('[LLM] Metadata generated', extra={'hashtags': len(hashtags), 'broll_keywords': len(broll_keywords)})
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
