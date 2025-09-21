"""LLM metadata helper built on top of the existing integration stack."""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
import copy
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
        # Remember the last usable dynamic context so we can gracefully
        # recover when the primary LLM endpoint times out.
        self._last_dynamic_context: Dict[str, Any] = {}

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

    # --- Dynamic context fallback -------------------------------------------------
    def _fallback_dynamic_context(self, transcript_text: str) -> Dict[str, Any]:
        """Generate a lightweight context when the primary LLM call fails.

        We first try to reuse the optimised LLM helper exposed by the
        integration (``generate_broll_keywords_and_queries``). When that is not
        available we fall back to the heuristic keyword extractor so that we
        still return actionable queries for the downstream fetchers.
        """

        transcript_text = transcript_text or ""
        data: Dict[str, Any] = {}

        try:
            integration = self._get_integration()
        except Exception:
            integration = None

        if integration is not None:
            llm = getattr(integration, "llm", None)
            generator = getattr(llm, "generate_broll_keywords_and_queries", None)
            if callable(generator):
                try:
                    ok, payload = generator(transcript_text, max_keywords=12)
                except Exception:
                    ok, payload = False, {}
                if ok and isinstance(payload, dict):
                    keywords = [
                        str(kw).strip().lower()
                        for kw in payload.get("broll_keywords", [])
                        if isinstance(kw, str) and kw.strip()
                    ]
                    if not keywords:
                        keywords = [
                            str(kw).strip().lower()
                            for kw in payload.get("keywords", [])
                            if isinstance(kw, str) and kw.strip()
                        ]
                    queries = [
                        str(q).strip()
                        for q in payload.get("search_queries", [])
                        if isinstance(q, str) and q.strip()
                    ]
                    if keywords or queries:
                        synonyms = {
                            kw: [kw.replace("_", " "), kw]
                            for kw in keywords
                        }
                        data = {
                            "detected_domains": payload.get("detected_domains") or [],
                            "language": payload.get("language"),
                            "keywords": keywords,
                            "synonyms": synonyms,
                            "search_queries": queries,
                            "segment_briefs": payload.get("segment_briefs") or [],
                        }

        if not data:
            try:
                from fallback_heuristic import HeuristicKeywordExtractor  # type: ignore

                extractor = HeuristicKeywordExtractor()
                heuristic_keywords = extractor.extract_keywords_from_text(
                    transcript_text,
                    target_count=12,
                )
            except Exception:
                heuristic_keywords = []

            clean_keywords = [
                str(kw).strip().lower()
                for kw in heuristic_keywords
                if isinstance(kw, str) and kw.strip()
            ]

            if clean_keywords:
                base_queries: List[str] = []
                for kw in clean_keywords:
                    humanised = kw.replace("_", " ")
                    base_queries.append(f"{humanised} vertical video")
                    base_queries.append(f"{humanised} cinematic portrait")
                # Keep the most relevant queries while preserving order
                deduped_queries = list(dict.fromkeys(base_queries))[:12]
                data = {
                    "detected_domains": [{"name": "general", "confidence": 0.2}],
                    "language": None,
                    "keywords": clean_keywords,
                    "synonyms": {
                        kw: [kw.replace("_", " "), kw.split("_")[0]]
                        for kw in clean_keywords
                    },
                    "search_queries": deduped_queries,
                    "segment_briefs": [],
                }

        return data

    def generate_dynamic_context(self, transcript_text: str, *, max_len: int = 1800) -> Dict[str, Any]:
        """Domain detection + dynamic expansions (no hardcoded domains)."""
        prompt = build_dynamic_prompt(transcript_text, max_len=max_len)
        try:
            raw = self._complete_text(prompt)
            data = _safe_parse_json(raw)
        except Exception:
            logger.exception("[LLM] dynamic context failed")
            data = self._fallback_dynamic_context(transcript_text)

        if not isinstance(data, dict):
            data = {}

        # Guard rails + normalisation
        detected_domains = data.get("detected_domains") or []
        language = data.get("language") if isinstance(data.get("language"), str) else None
        keywords = [
            str(kw).strip().lower()
            for kw in data.get("keywords", [])
            if isinstance(kw, str) and kw.strip()
        ]
        raw_synonyms = data.get("synonyms") or {}
        synonyms: Dict[str, List[str]] = {}
        if isinstance(raw_synonyms, dict):
            for key, values in raw_synonyms.items():
                if not isinstance(key, str):
                    continue
                cleaned_key = key.strip().lower()
                if not cleaned_key:
                    continue
                syn_list: List[str] = []
                for value in values or []:
                    if isinstance(value, str) and value.strip():
                        norm = value.strip()
                        if norm not in syn_list:
                            syn_list.append(norm)
                if cleaned_key not in syn_list:
                    syn_list.append(cleaned_key)
                synonyms[cleaned_key] = syn_list
        queries = [
            str(q).strip()
            for q in data.get("search_queries", [])
            if isinstance(q, str) and q.strip()
        ]
        segment_briefs = [
            brief
            for brief in data.get("segment_briefs", [])
            if isinstance(brief, dict)
        ]

        normalised = {
            "detected_domains": detected_domains,
            "language": language,
            "keywords": keywords,
            "synonyms": synonyms,
            "search_queries": queries,
            "segment_briefs": segment_briefs,
        }

        # If the normalised context is empty, reuse the previous successful one
        if not keywords and not queries and self._last_dynamic_context:
            logger.warning("[LLM] dynamic context empty – reusing last known context")
            normalised = copy.deepcopy(self._last_dynamic_context)
        else:
            self._last_dynamic_context = copy.deepcopy(normalised)

        return normalised

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
        """Produce visual search hints for a transcript segment.

        The goal is to stay aligned with short-form vertical platforms. We favour
        action-driven queries derived from the LLM context and enrich them with
        segment-specific terms so that fetchers can retrieve relevant portrait
        footage.
        """

        ctx = getattr(self, "_last_dynamic_context", {}) or {}
        ctx_queries: List[str] = [
            str(q).strip()
            for q in ctx.get("search_queries", [])
            if isinstance(q, str) and q.strip()
        ]
        ctx_keywords: List[str] = [
            str(kw).strip().lower()
            for kw in ctx.get("keywords", [])
            if isinstance(kw, str) and kw.strip()
        ]
        ctx_synonyms = ctx.get("synonyms", {}) if isinstance(ctx.get("synonyms"), dict) else {}

        token_pattern = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")
        try:
            raw_tokens = [t.lower() for t in token_pattern.findall(text or "")]
        except Exception:
            raw_tokens = []

        stopwords = {
            "this", "that", "with", "have", "there", "their", "would", "could", "where", "when",
            "these", "those", "which", "about", "because", "being", "while", "after", "before",
            "people", "thing", "start", "really", "just", "then", "they", "them", "your", "you're",
        }
        segment_tokens: List[str] = []
        for tok in raw_tokens:
            if tok in stopwords or len(tok) <= 3:
                continue
            if tok not in segment_tokens:
                segment_tokens.append(tok)

        # Build dynamic queries prioritising context keywords and queries
        combined_queries: List[str] = []

        # 1) Direct queries coming from the LLM context
        for q in ctx_queries:
            if q not in combined_queries:
                combined_queries.append(q)

        # 2) Pair segment tokens with LLM keywords to create descriptive prompts
        def _humanise(term: str) -> str:
            return term.replace('_', ' ').strip()

        for token in segment_tokens[:4]:
            for kw in ctx_keywords[:6]:
                human_kw = _humanise(kw)
                candidate = f"{token} {human_kw} vertical video"
                if candidate not in combined_queries:
                    combined_queries.append(candidate)

        # 3) Add a few safety templates when there is very little context
        if not combined_queries:
            templates = [
                "{token} cinematic portrait",
                "{token} pov vertical video",
                "{token} motivational lifestyle vertical",
            ]
            base_tokens = segment_tokens or ctx_keywords or ["motivation"]
            for token in base_tokens[:4]:
                for tpl in templates:
                    candidate = tpl.format(token=_humanise(token))
                    if candidate not in combined_queries:
                        combined_queries.append(candidate)

        # Keep queries concise and provider-friendly
        queries = [q.strip() for q in combined_queries if q.strip()]
        queries = list(dict.fromkeys(queries))[:8]

        # Build synonyms merging context knowledge and fresh tokens
        synonyms: List[str] = []
        if isinstance(ctx_synonyms, dict):
            for values in ctx_synonyms.values():
                for value in values or []:
                    if isinstance(value, str):
                        clean_value = value.strip()
                        if clean_value and clean_value not in synonyms:
                            synonyms.append(clean_value)
        for tok in segment_tokens:
            readable = tok.replace('_', ' ')
            if readable not in synonyms:
                synonyms.append(readable)
        synonyms = synonyms[:10]

        if not queries:
            queries = ["motivational speaker vertical video", "focus training portrait shot"]
        if not synonyms:
            synonyms = ["motivation", "focus", "progress"]

        return {
            "queries": queries,
            "synonyms": synonyms,
            "filters": {"orientation": "portrait", "min_duration_s": 3.0},
        }
