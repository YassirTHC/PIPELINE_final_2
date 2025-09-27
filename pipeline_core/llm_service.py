"""LLM metadata helper built on top of the existing integration stack."""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
import importlib.util
import json
import logging
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
import unicodedata
import inspect
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    """Extract the first valid JSON object from text; return {} on failure."""

    if not isinstance(s, str):
        return {}
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find('{')
    if start < 0:
        return {}
    depth = 0
    for idx, ch in enumerate(s[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                snippet = s[start:idx + 1]
                try:
                    return json.loads(snippet)
                except Exception:
                    break
    return {}


_TERM_MIN_LEN = 3
_GENERIC_TERMS = {
    'people',
    'thing',
    'nice',
    'background',
    'start',
    'generic',
    'template',
    'stock',
    'first',
    'occurs',
    'stuff',
    'video',
    'clip',
}

_VISUAL_ABSTRACT_TOKENS = {
    "abstract",
    "analysis",
    "approach",
    "buffer",
    "concept",
    "design",
    "idea",
    "insight",
    "method",
    "network",
    "professional",
    "signal",
    "strategy",
    "structure",
    "system",
}

_CONCRETE_SUBJECTS = {
    "actor",
    "adult",
    "athlete",
    "audience",
    "baby",
    "brain",
    "camera",
    "chef",
    "city",
    "class",
    "classroom",
    "clinic",
    "coach",
    "computer",
    "crowd",
    "desk",
    "doctor",
    "dopamine",
    "dog",
    "engineer",
    "family",
    "farmer",
    "factory",
    "forest",
    "group",
    "hands",
    "hospital",
    "human",
    "kid",
    "lab",
    "laboratory",
    "laptop",
    "machine",
    "manager",
    "man",
    "microscope",
    "musician",
    "nurse",
    "office",
    "patient",
    "person",
    "player",
    "researcher",
    "robot",
    "scientist",
    "student",
    "team",
    "teacher",
    "technician",
    "woman",
    "worker",
}

_CONCRETE_SUBJECT_PATTERN = re.compile(
    r"\b(" +
    r"|".join(
        [
            "actor",
            "adult",
            "athlete",
            "audience",
            "baby",
            "brain",
            "camera",
            "chef",
            "city",
            "classroom",
            "clinic",
            "coach",
            "computer",
            "crowd",
            "doctor",
            "dopamine",
            "dog",
            "engineer",
            "family",
            "farmer",
            "factory",
            "forest",
            "group",
            "hands",
            "hospital",
            "human",
            "kid",
            "lab",
            "laboratory",
            "laptop",
            "machine",
            "manager",
            "man",
            "microscope",
            "musician",
            "nurse",
            "office",
            "patient",
            "person",
            "player",
            "researcher",
            "robot",
            "scientist",
            "student",
            "team",
            "teacher",
            "technician",
            "woman",
            "worker",
        ]
    )
    + r")\b"
)

_ACTION_HINTS = {
    "analyze",
    "analyzing",
    "build",
    "building",
    "check",
    "checking",
    "code",
    "coding",
    "collaborate",
    "collaborating",
    "create",
    "creating",
    "discuss",
    "discussing",
    "experiment",
    "experimenting",
    "explain",
    "explaining",
    "hold",
    "holding",
    "inspect",
    "inspecting",
    "learn",
    "learning",
    "meet",
    "meeting",
    "mix",
    "mixing",
    "observe",
    "observing",
    "plan",
    "planning",
    "point",
    "pointing",
    "present",
    "presenting",
    "review",
    "reviewing",
    "run",
    "running",
    "scan",
    "scanning",
    "study",
    "studying",
    "teach",
    "teaching",
    "test",
    "testing",
    "use",
    "using",
    "walk",
    "walking",
    "work",
    "working",
}

_DEFAULT_ACTION = "showing"


_STOPWORDS_EN = {
    'a',
    'an',
    'and',
    'any',
    'are',
    'as',
    'at',
    'be',
    'been',
    'being',
    'but',
    'by',
    'can',
    'could',
    'did',
    'do',
    'does',
    'doing',
    'for',
    'from',
    'had',
    'has',
    'have',
    'having',
    'he',
    'her',
    'hers',
    'him',
    'his',
    'i',
    'if',
    'in',
    'into',
    'is',
    'it',
    'its',
    'may',
    'me',
    'might',
    'mine',
    'must',
    'my',
    'of',
    'on',
    'or',
    'our',
    'ours',
    'she',
    'should',
    'so',
    'some',
    'such',
    'than',
    'that',
    'the',
    'their',
    'theirs',
    'them',
    'then',
    'there',
    'these',
    'they',
    'this',
    'those',
    'through',
    'to',
    'too',
    'up',
    'was',
    'we',
    'were',
    'when',
    'where',
    'which',
    'who',
    'whom',
    'why',
    'will',
    'with',
    'would',
    'you',
    'your',
    'yours',
}


_EN_EQUIVALENTS = {
    "adrnaline": "adrenaline",
    "adrenaline": "adrenaline",
    "adrenalin": "adrenaline",
    "controle": "control",
    "contrle": "control",
    "processus": "process",
    "recompense": "reward",
    "recompenses": "rewards",
    "rcompense": "reward",
    "rcompenses": "rewards",
    "dure": "duration",
    "duree": "duration",
    "durees": "durations",
    "objectif": "goal",
    "objectifs": "goals",
    "reussite": "success",
    "russite": "success",
    "succs": "success",
    "succes": "success",
    "russites": "successes",
    "reussites": "successes",
}


def _tokenise(text: str) -> List[str]:
    if not text:
        return []
    try:
        raw = re.findall(r"[A-Za-zÀ-ÿ'\-]+", text.lower())
    except Exception:
        raw = []
    tokens: List[str] = []
    for token in raw:
        token = token.replace('_', ' ').strip()
        if len(token) >= _TERM_MIN_LEN and token not in _GENERIC_TERMS:
            tokens.append(token)
    return tokens


def _normalise_terms(values: Iterable[str], *, limit: Optional[int] = None) -> List[str]:
    seen: set[str] = set()
    normalised: List[str] = []
    for value in values or []:
        if not isinstance(value, str):
            continue
        term = value.strip().lower().replace('_', ' ')
        term = re.sub(r"\s+", ' ', term)
        term = re.sub(r"[^a-z0-9\s-]", '', term)
        tokens = [t for t in term.split() if t]
        filtered_tokens = [t for t in tokens if t not in _STOPWORDS_EN]
        if not filtered_tokens:
            continue
        if any(t in _GENERIC_TERMS for t in filtered_tokens):
            continue
        cleaned = " ".join(filtered_tokens)
        if len(cleaned) < _TERM_MIN_LEN:
            continue
        if cleaned in _GENERIC_TERMS:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        normalised.append(cleaned)
        if limit is not None and len(normalised) >= limit:
            break
    return normalised


def _pick_subject(tokens: List[str]) -> Optional[str]:
    for token in tokens:
        if token in _CONCRETE_SUBJECTS or _CONCRETE_SUBJECT_PATTERN.search(token):
            return token
    return None


def _pick_action(tokens: List[str], *, exclude: set[str]) -> Optional[str]:
    for token in tokens:
        if token in exclude:
            continue
        if token in _ACTION_HINTS or token.endswith("ing") or token.endswith("ed"):
            return token
    return None


def has_concrete_subject(value: str) -> bool:
    tokens = [tok for tok in (value or "").lower().split() if tok]
    return _pick_subject(tokens) is not None


def build_visual_phrases(terms: Iterable[str], *, limit: Optional[int] = None) -> List[str]:
    phrases: List[str] = []
    seen: set[str] = set()
    for raw in terms or []:
        if not isinstance(raw, str):
            continue
        tokens = [tok for tok in raw.lower().split() if tok]
        if not tokens:
            continue
        tokens = [tok for tok in tokens if tok not in _VISUAL_ABSTRACT_TOKENS]
        if len(tokens) < 2:
            continue
        subject = _pick_subject(tokens)
        if not subject:
            continue
        used = {subject}
        action = _pick_action(tokens, exclude=used)
        if action:
            used.add(action)
        else:
            action = _DEFAULT_ACTION
        context_tokens: List[str] = []
        for token in tokens:
            if token in used:
                continue
            if token in _VISUAL_ABSTRACT_TOKENS:
                continue
            context_tokens.append(token)
            if len(context_tokens) >= 3:
                break
        phrase_tokens = [subject, action]
        phrase_tokens.extend(context_tokens)
        phrase_tokens = phrase_tokens[:5]
        if len(phrase_tokens) < 2:
            continue
        phrase = " ".join(phrase_tokens)
        if phrase in seen:
            continue
        seen.add(phrase)
        phrases.append(phrase)
        if limit is not None and len(phrases) >= limit:
            break
    return phrases


def _strip_diacritics(value: str) -> str:
    if not value:
        return value
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def enforce_fetch_language(terms: Iterable[str], language: Optional[str]) -> List[str]:
    """Ensure fetch terms use English tokens when the payload is English."""

    if language not in (None, "", "en"):
        return list(dict.fromkeys(term for term in terms if term))

    normalised: List[str] = []
    seen: set[str] = set()
    for raw in terms:
        if not raw:
            continue
        tokens: List[str] = []
        for token in raw.split():
            token_norm = token.lower()
            base = _EN_EQUIVALENTS.get(token_norm)
            if base is None:
                ascii_token = _strip_diacritics(token_norm)
                ascii_token = ascii_token.lower()
                base = _EN_EQUIVALENTS.get(ascii_token)
                if base is None:
                    base = ascii_token or token
            tokens.append(base)
        candidate = " ".join(tok for tok in tokens if tok)
        if candidate and candidate not in seen:
            seen.add(candidate)
            normalised.append(candidate)
    return normalised


def _normalise_synonyms(raw: Dict[str, Any]) -> Dict[str, List[str]]:
    clean: Dict[str, List[str]] = {}
    for key, variants in (raw or {}).items():
        base = _normalise_terms([key], limit=1)
        variants_norm = _normalise_terms(variants, limit=4)
        if base and variants_norm:
            clean[base[0]] = variants_norm
    return clean


def _normalise_briefs(raw: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    briefs: List[Dict[str, Any]] = []
    for entry in raw or []:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get('segment_index'))
        except Exception:
            continue
        keywords_raw = _normalise_terms(entry.get('keywords') or [], limit=6)
        queries_raw = _normalise_terms(entry.get('queries') or [], limit=6)
        keywords = build_visual_phrases(keywords_raw, limit=6)
        queries = build_visual_phrases(queries_raw, limit=6)
        if not keywords and not queries:
            continue
        briefs.append({
            'segment_index': idx,
            'keywords': keywords,
            'queries': queries,
        })
    return briefs


def _tfidf_fallback(transcript: str, *, top_k: int = 12) -> Tuple[List[str], List[str]]:
    segments: List[List[str]] = []
    for chunk in re.split(r"[\.!?\n]+", transcript or ''):
        tokens = _tokenise(chunk)
        if tokens:
            segments.append(tokens)
    if not segments:
        tokens = _tokenise(transcript)
        if tokens:
            segments.append(tokens)
    if not segments:
        return [], []
    df = Counter()
    for tokens in segments:
        df.update(set(tokens))
    tfidf: Dict[str, float] = {}
    for tokens in segments:
        counts = Counter(tokens)
        length = max(1, len(tokens))
        for term, count in counts.items():
            tf = count / length
            idf = math.log((len(segments) + 1) / (df[term] + 1)) + 1.0
            score = tf * idf
            tfidf[term] = max(tfidf.get(term, 0.0), score)
    keywords = [term for term, _ in sorted(tfidf.items(), key=lambda item: item[1], reverse=True) if term not in _GENERIC_TERMS][:top_k]
    bigrams: Counter[str] = Counter()
    for tokens in segments:
        for first, second in zip(tokens, tokens[1:]):
            if first == second:
                continue
            if first in _GENERIC_TERMS or second in _GENERIC_TERMS:
                continue
            bigrams[f'{first} {second}'] += 1
    queries = [term for term, _ in bigrams.most_common(max(4, top_k // 2))]
    return _normalise_terms(keywords, limit=top_k), _normalise_terms(queries, limit=max(4, top_k // 2))


def _normalise_dynamic_payload(raw: Dict[str, Any], *, transcript: str) -> Dict[str, Any]:
    domains: List[Dict[str, Any]] = []
    for entry in raw.get('detected_domains') or []:
        if not isinstance(entry, dict):
            continue
        names = _normalise_terms([entry.get('name', '')], limit=1)
        if not names:
            continue
        confidence = entry.get('confidence')
        try:
            conf = float(confidence) if confidence is not None else None
        except Exception:
            conf = None
        domains.append({'name': names[0], 'confidence': conf})

    language = raw.get('language')
    if isinstance(language, str):
        language = language.strip().lower() or None
    else:
        language = None

    keywords = enforce_fetch_language(_normalise_terms(raw.get('keywords') or [], limit=20), language)
    search_queries = enforce_fetch_language(_normalise_terms(raw.get('search_queries') or [], limit=12), language)
    synonyms = _normalise_synonyms(raw.get('synonyms') or {})
    briefs = _normalise_briefs(raw.get('segment_briefs') or [])

    if not keywords and not search_queries:
        fallback_kw, fallback_q = _tfidf_fallback(transcript)
        keywords = fallback_kw[:12]
        if not search_queries:
            search_queries = fallback_q[:8]

    return {
        'detected_domains': domains,
        'language': language,
        'keywords': keywords,
        'synonyms': synonyms,
        'search_queries': search_queries,
        'segment_briefs': briefs,
    }


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
        timeout_env = os.getenv("PIPELINE_LLM_TIMEOUT")
        try:
            timeout_val = int(timeout_env) if timeout_env else 35
        except (TypeError, ValueError):
            timeout_val = 35
        self._llm_timeout = max(5, timeout_val)

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
    def _complete_text(self, prompt: str, *, max_tokens: int = 800) -> str:
        """Attempt to invoke a completion method on the underlying integration.

        Tries common method names; falls back to calling the optimized LLM engine
        when available (integration.llm._call_llm). Returns raw text.
        """
        integration = self._get_integration()

        # 1) Try common completion-shaped methods on integration directly
        last_error: Optional[BaseException] = None
        timed_out = False
        for attr in ("complete_json", "complete", "chat", "generate"):
            fn = getattr(integration, attr, None)
            if callable(fn):
                try:
                    return self._invoke_completion(fn, prompt, max_tokens=max_tokens)
                except Exception as exc:  # pragma: no cover - robustness
                    last_error = exc
                    if self._is_timeout_error(exc):
                        timed_out = True
                        break

        # 2) Try going through the optimized LLM engine if exposed
        if not timed_out:
            llm = getattr(integration, "llm", None)
            if llm is not None:
                # Prefer a public method if one exists
                for attr in ("complete_json", "complete", "generate"):
                    fn = getattr(llm, attr, None)
                    if callable(fn):
                        try:
                            return self._invoke_completion(fn, prompt, max_tokens=max_tokens)
                        except Exception as exc:  # pragma: no cover - robustness
                            last_error = exc
                            if self._is_timeout_error(exc):
                                timed_out = True
                                break
                    if timed_out:
                        break
                if not timed_out:
                    # Fallback to private _call_llm used in optimized_llm.py
                    call = getattr(llm, "_call_llm", None)
                    if callable(call):
                        ok, text, err = call(
                            prompt,
                            temperature=0.1,
                            max_tokens=max_tokens,
                            timeout=self._llm_timeout,
                        )
                        if ok and isinstance(text, str):
                            return text
                        if err == "timeout":
                            timed_out = True
                        last_error = RuntimeError(f"LLM _call_llm failed: {err or 'unknown'}")

        if timed_out:
            raise TimeoutError("LLM completion timed out") from last_error

        if last_error is not None:
            raise RuntimeError("No compatible completion method available on integration") from last_error

        raise RuntimeError("No compatible completion method available on integration")

    def _invoke_completion(self, fn, prompt: str, *, max_tokens: int) -> Any:
        """Call a completion function with a bounded timeout when supported."""

        kwargs: Dict[str, Any] = {}
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):  # pragma: no cover - builtins
            signature = None

        if signature is not None:
            params = signature.parameters
            if "timeout" in params:
                kwargs["timeout"] = self._llm_timeout
            if "max_tokens" in params and "max_tokens" not in kwargs:
                kwargs["max_tokens"] = max_tokens
        try:
            return fn(prompt, **kwargs)  # type: ignore[misc]
        except TypeError as exc:
            if kwargs:
                return fn(prompt)  # type: ignore[misc]
            raise exc

    @staticmethod
    def _is_timeout_error(exc: BaseException) -> bool:
        message = str(exc).lower()
        return "timeout" in message or "timed out" in message

    def generate_dynamic_context(self, transcript_text: str, *, max_len: int = 1800) -> Dict[str, Any]:
        """Domain detection + dynamic expansions (no hardcoded domains)."""
        transcript = transcript_text or ''
        attempt_limits: List[int] = []
        for limit in (max_len, max_len // 2, max_len // 3):
            if limit and limit > 0:
                attempt_limits.append(min(max_len, max(limit, 350)))

        # Keep order from largest to smallest while removing duplicates
        seen_limits = set()
        ordered_limits: List[int] = []
        for limit in attempt_limits:
            if limit not in seen_limits:
                ordered_limits.append(limit)
                seen_limits.add(limit)

        if not ordered_limits:
            ordered_limits = [max_len or 600]

        raw_payload: Dict[str, Any] = {}
        for idx, limit in enumerate(ordered_limits, start=1):
            prompt = build_dynamic_prompt(transcript, max_len=limit)
            if limit >= 1400:
                token_budget = 700
            elif limit >= 900:
                token_budget = 550
            elif limit >= 600:
                token_budget = 400
            else:
                token_budget = 300
            try:
                raw_text = self._complete_text(prompt, max_tokens=token_budget)
            except TimeoutError:
                logger.warning(
                    '[LLM] dynamic context attempt %s timed out (limit=%s, tokens=%s)',
                    idx,
                    limit,
                    token_budget,
                )
                continue
            except Exception:
                logger.exception('[LLM] dynamic context attempt %s failed', idx)
                continue

            if isinstance(raw_text, dict):
                raw_payload = raw_text
            else:
                raw_payload = _safe_parse_json(raw_text)

            if raw_payload:
                break
        else:
            if transcript:
                logger.warning('[LLM] dynamic context fell back to TF-IDF (no structured payload)')
        return _normalise_dynamic_payload(raw_payload, transcript=transcript_text or '')


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
        anti_terms = {
            'person',
            'doctor',
            'stethoscope',
            'patient',
            'clinic',
            'workspace',
            'discussing',
            'nurse',
        }

        candidates: List[str] = []
        for window in (3, 2):
            if len(keywords) < window:
                continue
            for i in range(len(keywords) - window + 1):
                phrase = " ".join(keywords[i : i + window])
                if phrase:
                    candidates.append(phrase)

        if not candidates and keywords:
            span = min(3, len(keywords))
            candidates.append(" ".join(keywords[:span]))

        queries: List[str] = []
        seen: set[str] = set()
        for phrase in candidates:
            tokens = [tok for tok in phrase.split() if tok]
            if any(tok in anti_terms for tok in tokens):
                continue
            if phrase not in seen:
                seen.add(phrase)
                queries.append(phrase)
            if len(queries) >= 8:
                break
        synonyms = sorted(set(keywords[:6]))
        return {
            "queries": queries or keywords[:3],
            "synonyms": synonyms or ["healthcare", "therapy", "consultation"],
            "filters": {"orientation": "landscape", "min_duration_s": 3.0},
        }
