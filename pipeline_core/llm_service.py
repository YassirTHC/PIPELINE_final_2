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
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests

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


class DynamicCompletionError(RuntimeError):
    """Raised when the dedicated text completion returns no usable payload."""

    def __init__(self, reason: str, *, payload: Optional[Dict[str, Any]] = None) -> None:
        self.reason = (reason or "unknown").strip() or "unknown"
        self.payload = payload
        super().__init__(f"dynamic completion failed: {self.reason}")


class TfidfFallbackDisabled(RuntimeError):
    """Raised when TF-IDF fallback is disabled for the current run."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.reason = message


_SHARED_LOCK: Lock = Lock()
_SHARED: LLMMetadataGeneratorService | None = None


_DEFAULT_STOP_TOKENS: Tuple[str, ...] = ("```", "\n\n\n", "END_OF_CONTEXT", "</json>")

_LAST_METADATA_KEYWORDS: Dict[str, Any] = {"values": [], "updated_at": 0.0}
_LAST_METADATA_QUERIES: Dict[str, Any] = {"values": [], "updated_at": 0.0}

_DEFAULT_OLLAMA_ENDPOINT = "http://127.0.0.1:11434"
_DEFAULT_OLLAMA_MODEL = "mistral:7b-instruct"
_DEFAULT_OLLAMA_KEEP_ALIVE = "5m"


def _keywords_first_enabled() -> bool:
    flag = _env_to_bool(os.getenv("PIPELINE_LLM_KEYWORDS_FIRST"))
    if flag is None:
        return True
    return flag


def _target_language_default() -> str:
    raw = os.getenv("PIPELINE_LLM_TARGET_LANG", "en")
    if raw is None:
        return "en"
    cleaned = raw.strip().lower()
    return cleaned or "en"


def _hashtags_disabled() -> bool:
    flag = _env_to_bool(os.getenv("PIPELINE_LLM_DISABLE_HASHTAGS"))
    if flag is None:
        return False
    return flag


def _parse_stop_tokens(value: Optional[str]) -> Sequence[str]:
    if not value:
        return _DEFAULT_STOP_TOKENS
    try:
        parsed = json.loads(value)
        if isinstance(parsed, (list, tuple)):
            tokens = [str(token).strip() for token in parsed if str(token).strip()]
            if tokens:
                return tokens
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    tokens = [token.strip() for token in value.split("|") if token.strip()]
    return tokens or _DEFAULT_STOP_TOKENS


def _ollama_json(prompt: str, *, model: Optional[str] = None, endpoint: Optional[str] = None) -> Dict[str, Any]:
    """Send a prompt to Ollama and try to extract the largest JSON object."""

    prompt = (prompt or "").strip()
    if not prompt:
        return {}

    model_name = (model or os.getenv("PIPELINE_LLM_MODEL") or "qwen2.5:7b").strip() or "qwen2.5:7b"
    base_url = (
        endpoint
        or os.getenv("PIPELINE_LLM_ENDPOINT")
        or os.getenv("PIPELINE_LLM_BASE_URL")
        or os.getenv("OLLAMA_HOST")
        or "http://localhost:11434"
    )
    base_url = base_url.strip() or "http://localhost:11434"
    url = f"{base_url.rstrip('/')}/api/generate"

    def _parse_float(value: Optional[str], *, default: float, minimum: float) -> float:
        try:
            parsed = float(value) if value is not None else default
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, parsed)

    def _parse_int(value: Optional[str], *, default: int, minimum: int) -> int:
        try:
            parsed = int(value) if value is not None else default
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, parsed)

    timeout_env = os.getenv("PIPELINE_LLM_TIMEOUT_S")
    timeout = _parse_float(timeout_env, default=60.0, minimum=1.0)
    num_predict = _parse_int(os.getenv("PIPELINE_LLM_NUM_PREDICT"), default=256, minimum=1)
    temperature = _parse_float(os.getenv("PIPELINE_LLM_TEMP"), default=0.3, minimum=0.0)
    top_p = _parse_float(os.getenv("PIPELINE_LLM_TOP_P"), default=0.9, minimum=0.0)

    payload: Dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "num_predict": num_predict,
            "temperature": temperature,
            "top_p": max(0.0, min(1.0, top_p)),
        },
    }

    started = time.perf_counter()

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.Timeout:
        logger.warning(
            "[LLM] Ollama request timed out",
            extra={"model": model_name, "duration_s": round(time.perf_counter() - started, 3)},
        )
        return {}
    except requests.RequestException as exc:
        logger.warning(
            "[LLM] Ollama request failed",
            extra={"model": model_name, "error": str(exc)},
        )
        return {}

    response_text = response.text or ""
    try:
        payload_json = response.json()
    except ValueError:
        payload_json = None

    candidates: List[str] = []
    if isinstance(payload_json, dict):
        # Common Ollama schema: {"response": "..."}
        for key in ("response", "content", "data", "message", "result", "metadata"):
            value = payload_json.get(key)
            if isinstance(value, dict):
                return value
            if isinstance(value, str) and value.strip():
                candidates.append(value)
        if not candidates:
            try:
                serialised = json.dumps(payload_json)
            except Exception:
                serialised = ""
            if serialised:
                candidates.append(serialised)
    if response_text and response_text not in candidates:
        candidates.append(response_text)

    best_match: Dict[str, Any] = {}
    best_length = -1
    pattern = re.compile(r"\{.*?\}", re.DOTALL)
    for chunk in candidates:
        if not chunk:
            continue
        for match in pattern.finditer(chunk):
            snippet = match.group(0)
            try:
                parsed = json.loads(snippet)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(parsed, dict) and len(snippet) > best_length:
                best_match = parsed
                best_length = len(snippet)

    return best_match if best_match else {}


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


# --- Query sanitization & augmentation ---------------------------------------
BANNED_GENERIC = {
    "that",
    "this",
    "it",
    "they",
    "we",
    "you",
    "thing",
    "stuff",
    "very",
    "just",
    "really",
}

BANNED_PROVIDER_NOISE = {
    "stock",
    "footage",
    "b-roll",
    "broll",
    "roll",
    "cinematic",
    "timelapse",
    "background",
    "background footage",
    "visual",
    "visuals",
    "clip",
    "clips",
    "video",
    "videos",
    "shot",
    "shots",
    "scene",
    "scenes",
    "sequence",
    "sequences",
    "demonstration",
    "demonstrations",
    "visualization",
    "visualizations"
}

BANNED_ALL = BANNED_GENERIC | BANNED_PROVIDER_NOISE


def compile_token_blocklist_regex(terms: Iterable[str]) -> re.Pattern:
    escaped: List[str] = []
    for term in terms:
        if term is None:
            continue
        text = str(term).strip()
        if not text:
            continue
        escaped.append(re.escape(text))
    if not escaped:
        return re.compile(r'^\b$')
    pattern = r'\b(?:' + '|'.join(escaped) + r')\b'
    return re.compile(pattern, flags=re.IGNORECASE | re.UNICODE)


RX_BANNED = compile_token_blocklist_regex(BANNED_ALL)


def remove_blocklisted_tokens(text: str, pattern: Optional[re.Pattern] = None) -> str:
    if text is None:
        return ''
    rx = pattern or RX_BANNED
    try:
        cleaned = rx.sub(' ', str(text))
    except re.error:
        return str(text)
    return re.sub(r'\s+', ' ', cleaned).strip()



# Normalise extended Latin letters without regex ranges to avoid Windows mojibake errors.
_BASIC_LATIN_CODEPOINT_RANGES: Tuple[Tuple[int, int], ...] = (
    (ord('0'), ord('9')),
    (ord('A'), ord('Z')),
    (ord('a'), ord('z')),
    (0x00C0, 0x00D6),
    (0x00D8, 0x00F6),
    (0x00F8, 0x00FF),
)

_EXTRA_TOKEN_CHARS: Set[str] = {"'", "-"}


def _is_basic_latin_char(ch: str) -> bool:
    code = ord(ch)
    for start, end in _BASIC_LATIN_CODEPOINT_RANGES:
        if start <= code <= end:
            return True
    return False


def _filter_basic_latin(text: str) -> str:
    return ''.join(ch for ch in text if _is_basic_latin_char(ch))


def _split_basic_latin_runs(text: str, *, keep: Set[str] | None = None) -> List[str]:
    allowed_extras = keep or set()
    runs: List[str] = []
    buffer: List[str] = []
    for ch in text:
        if _is_basic_latin_char(ch) or ch in allowed_extras:
            buffer.append(ch)
        elif buffer:
            runs.append(''.join(buffer))
            buffer.clear()
    if buffer:
        runs.append(''.join(buffer))
    return runs


_NEGATIVE_QUERY_TERMS: Set[str] = {
    "visual",
    "visuals",
    "clip",
    "clips",
    "video",
    "videos",
    "shot",
    "shots",
    "scene",
    "scenes",
    "sequence",
    "sequences",
    "demonstration",
    "demonstrations",
    "visualization",
    "visualizations",
}

_CONCRETIZE_RULES: Tuple[Tuple[re.Pattern[str], Tuple[str, ...]], ...] = (
    (re.compile(r"internal rewards?", flags=re.IGNORECASE), ("journaling at desk", "smiling after workout", "deep breath at window")),
    (re.compile(r"mindset shift|conceptual mapping", flags=re.IGNORECASE), ("whiteboard planning", "sticky notes wall", "drawing flowchart")),
    (re.compile(r"time passing|duration", flags=re.IGNORECASE), ("clock close up", "sunset sky", "hourglass closeup")),
    (re.compile(r"energy", flags=re.IGNORECASE), ("city night traffic", "powerlines closeup", "spark plug closeup")),
    (re.compile(r"adrenaline|tension|focus", flags=re.IGNORECASE), ("runner tying shoes", "boxing training", "eyes closeup focus")),
)

_CONCEPT_FALLBACKS: Tuple[str, ...] = (
    "typing at desk",
    "whiteboard sketching",
    "team huddle meeting",
    "city street walking",
)

_DEFAULT_CONCRETE_QUERIES: Tuple[str, ...] = (
    "typing at desk",
    "whiteboard planning",
    "runner tying shoes",
    "city night traffic",
)

_ABSTRACT_HINTS: Tuple[str, ...] = (
    "process",
    "realization",
    "formation",
    "mapping",
    "framework",
    "concept",
    "awareness",
)


def _concretize_queries(values: Sequence[str]) -> List[str]:
    """Map abstract LLM queries to concrete, filmable prompts."""

    concrete: List[str] = []
    seen: Set[str] = set()

    for raw in values or []:
        if raw is None:
            continue
        base = str(raw).strip().lower()
        if not base:
            continue
        filtered_tokens = [token for token in base.split() if token and token not in _NEGATIVE_QUERY_TERMS]
        filtered = " ".join(filtered_tokens).strip()
        target = filtered or base
        matched = False
        for pattern, replacements in _CONCRETIZE_RULES:
            if pattern.search(target):
                for replacement in replacements[:2]:
                    if replacement not in seen:
                        concrete.append(replacement)
                        seen.add(replacement)
                matched = True
                break
        if matched:
            continue
        if any(hint in target for hint in _ABSTRACT_HINTS):
            for replacement in _CONCEPT_FALLBACKS[:2]:
                if replacement not in seen:
                    concrete.append(replacement)
                    seen.add(replacement)
            continue
        if target and target not in seen:
            concrete.append(target)
            seen.add(target)

    if not concrete:
        for replacement in _DEFAULT_CONCRETE_QUERIES:
            if replacement not in seen:
                concrete.append(replacement)
                seen.add(replacement)
    return concrete


def strip_banned(text: str) -> str:
    return remove_blocklisted_tokens(text or '', RX_BANNED)


def _sanitize_queries(
    queries: Sequence[str],
    *,
    min_words: int = 2,
    max_words: int = 4,
    max_len: Optional[int] = 12,
) -> List[str]:
    limit = max_len if isinstance(max_len, int) and max_len > 0 else None
    cleaned: List[str] = []
    seen: Set[str] = set()
    for raw in queries or []:
        candidate = remove_blocklisted_tokens(str(raw or ''), RX_BANNED)
        candidate = re.sub(r'\s+', ' ', candidate).strip()
        if not candidate:
            continue
        words = candidate.split()
        if len(words) < min_words or len(words) > max_words:
            continue
        key = candidate.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(candidate)
        if limit is not None and len(cleaned) >= limit:
            break
    return cleaned

SEGMENT_JSON_PROMPT = (
    "You are a JSON API. Return ONLY one JSON object with keys: broll_keywords, queries. "
    "broll_keywords: 8–12 visual noun phrases (2–3 words), concrete and shootable. "
    "queries: 8–12 short, filmable search queries (2–4 words), provider-friendly. "
    "Banned tokens: that, this, it, they, we, you, thing, stuff, very, just, really, "
    "stock, footage, b-roll, broll, roll, cinematic, timelapse, background, background footage. "
    "Segment transcript:\n{segment_text}"
)


_QUERY_SYNONYMS: Dict[str, List[str]] = {
    "running": ["jogging", "sprinting"],
    "workout": ["gym weights", "barbell training"],
    "typing": ["keyboard typing", "typing at desk"],
    "planning": ["whiteboard planning", "sticky notes planning"],
    "coffee": ["coffee brewing", "pour over coffee"],
}

def _augment_with_synonyms(queries: Sequence[str], *, max_extra_per: int = 1, limit: int = 12) -> List[str]:
    """Add 0–1 short synonym per base query from a static table, capped by ``limit``."""

    out: List[str] = []
    seen: Set[str] = set()
    for q in queries or []:
        cand = " ".join(str(q).split()).lower()
        if not cand or cand in seen:
            continue
        seen.add(cand)
        out.append(cand)
        key = cand.split()[0]
        added = 0
        for syn in _QUERY_SYNONYMS.get(key, []):
            s = " ".join(str(syn).split()).lower()
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
            added += 1
            if added >= max_extra_per or len(out) >= limit:
                break
        if len(out) >= limit:
            break
    return out[:limit]

def _coerce_ollama_json(payload: Any) -> Dict[str, Any]:
    """Best-effort conversion of Ollama responses into dictionaries."""

    if isinstance(payload, dict):
        return payload

    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return {}

        if text.startswith('"') and text.endswith('"'):
            try:
                unescaped = json.loads(text)
            except json.JSONDecodeError:
                unescaped = None
            if isinstance(unescaped, str) and unescaped:
                text = unescaped.strip()

        fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if fence_match:
            fenced = fence_match.group(1).strip()
            if fenced:
                text = fenced

        parsed = _safe_parse_json(text)
        if isinstance(parsed, dict) and parsed:
            return parsed

        snippet = _first_balanced_json_block(text)
        if snippet:
            try:
                parsed_snippet = json.loads(snippet)
            except json.JSONDecodeError:
                parsed_snippet = None
            if isinstance(parsed_snippet, dict) and parsed_snippet:
                return parsed_snippet

        parsed = _extract_json_braces(text)
        if isinstance(parsed, dict) and parsed:
            return parsed

        try:
            parsed_direct = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed_direct = None
        if isinstance(parsed_direct, dict) and parsed_direct:
            return parsed_direct

    return {}


def _extract_json_braces(text: str) -> Dict[str, Any]:
    """Fallback extraction that keeps searching for balanced ``{...}`` blocks."""

    if not isinstance(text, str):
        return {}

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        snippet = match.group(0)
        try:
            return json.loads(snippet)
        except Exception:
            pass

    start = -1
    depth = 0
    for idx, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start >= 0:
                    snippet = text[start : idx + 1]
                    try:
                        return json.loads(snippet)
                    except Exception:
                        start = -1
    return {}


def _first_balanced_json_block(text: str) -> Optional[str]:
    """Return the first balanced JSON object found in ``text`` if possible."""

    if not isinstance(text, str):
        return None

    start_match = re.search(r"\{", text)
    if not start_match:
        return None

    start_idx = start_match.start()
    depth = 0
    in_string = False
    escape = False

    for idx in range(start_idx, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0:
                    return text[start_idx : idx + 1]

    return None


def _env_to_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _normalise_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        candidates = re.split(r"[\n,]+", value)
    elif isinstance(value, dict):
        candidates = list(value.values())
    else:
        try:
            candidates = list(value)
        except TypeError:
            candidates = [value]
    normalised: List[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        text = candidate.strip() if isinstance(candidate, str) else str(candidate).strip()
        if text:
            normalised.append(text)
    return normalised


def _normalise_string_list(value: Any) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for item in _as_list(value):
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _normalise_hashtags(value: Any) -> List[str]:
    hashtags = []
    for tag in _normalise_string_list(value):
        cleaned = tag.strip()
        if not cleaned:
            continue
        if " " in cleaned:
            cleaned = cleaned.replace(" ", "")
        if not cleaned.startswith("#"):
            cleaned = "#" + cleaned.lstrip("#")
        if cleaned not in hashtags:
            hashtags.append(cleaned)
    return hashtags


def _strip_accents(text: str) -> str:
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalise_provider_terms(
    values: Iterable[str],
    *,
    target_lang: Optional[str] = None,
) -> List[str]:
    language = (target_lang or "").strip().lower() or _target_language_default()
    language = (language or "").strip().lower()
    english_aliases = {
        "en",
        "en-us",
        "en-gb",
        "en-ca",
        "en-au",
        "en_us",
        "en_gb",
        "en_ca",
        "en_au",
        "english",
    }
    is_english = language in english_aliases

    normalised: List[str] = []
    seen: Set[str] = set()

    for raw in _normalise_string_list(values):
        if not raw:
            continue
        base = _strip_accents(raw)
        if not base:
            continue
        cleaned = re.sub(r"[^0-9A-Za-z]+", " ", base)
        tokens = [token for token in cleaned.lower().split() if token]
        if is_english:
            tokens = [token for token in tokens if len(token) >= 3 and token not in _STOPWORDS_EN]
        if not tokens:
            continue
        term = " ".join(tokens)
        if term not in seen:
            seen.add(term)
            normalised.append(term)

    return normalised


def _normalise_search_terms(
    terms: Sequence[str],
    *,
    target_lang: Optional[str] = None,
) -> List[str]:
    language = (target_lang or "").split("-")[0].strip().lower()
    stopwords = _BASIC_STOPWORDS.get(language, set())

    normalised: List[str] = []
    seen: Set[str] = set()

    for candidate in _normalise_string_list(terms):
        base = _strip_accents(candidate)
        if not base:
            continue
        cleaned = re.sub(r"[^0-9A-Za-z]+", " ", base)
        tokens: List[str] = []
        for token in cleaned.lower().split():
            if len(token) < 3:
                continue
            if stopwords and token in stopwords:
                continue
            tokens.append(token)
        if not tokens:
            continue
        phrase = " ".join(tokens)
        if phrase and phrase not in seen:
            seen.add(phrase)
            normalised.append(phrase)

    return normalised


def _default_metadata_payload() -> Dict[str, Any]:
    return {
        "title": "Auto-generated Clip Title",
        "description": "Auto-generated description from transcript.",
        "hashtags": [],
        "broll_keywords": [],
        "queries": [],
    }


def _empty_metadata_payload() -> Dict[str, Any]:
    return {
        "title": "",
        "description": "",
        "hashtags": [],
        "broll_keywords": [],
        "queries": [],
    }


def _hashtags_from_keywords(keywords: Sequence[str], *, limit: int = 5) -> List[str]:
    tags: List[str] = []
    seen: set[str] = set()
    for keyword in _as_list(keywords):
        keyword_text = str(keyword or '')
        cleaned = _filter_basic_latin(keyword_text)
        if not cleaned:
            continue
        hashtag = "#" + cleaned
        lower = hashtag.lower()
        if lower in seen:
            continue
        seen.add(lower)
        tags.append(hashtag)
        if len(tags) >= limit:
            break
    return tags


def _resolve_ollama_endpoint(endpoint: Optional[str]) -> str:
    base_url = endpoint or os.getenv("PIPELINE_LLM_ENDPOINT") or os.getenv("PIPELINE_LLM_BASE_URL") or os.getenv("OLLAMA_HOST")
    base_url = (base_url or _DEFAULT_OLLAMA_ENDPOINT).strip()
    return base_url.rstrip("/") or _DEFAULT_OLLAMA_ENDPOINT


def _resolve_ollama_model(model: Optional[str]) -> str:
    candidate = model or os.getenv("PIPELINE_LLM_MODEL") or os.getenv("OLLAMA_MODEL")
    candidate = (candidate or _DEFAULT_OLLAMA_MODEL).strip()
    return candidate or _DEFAULT_OLLAMA_MODEL


def _resolve_keep_alive(value: Optional[str]) -> Optional[str]:
    raw = value
    if raw is None:
        raw = os.getenv("PIPELINE_LLM_KEEP_ALIVE")
    if raw is None:
        raw = _DEFAULT_OLLAMA_KEEP_ALIVE
    cleaned = (raw or "").strip()
    return cleaned or None


def _metadata_transcript_limit() -> int:
    limit_env = os.getenv("PIPELINE_LLM_JSON_TRANSCRIPT_LIMIT")
    try:
        limit = int(limit_env) if limit_env is not None else 5500
    except (TypeError, ValueError):
        limit = 5500
    return max(500, limit)


_FALLBACK_STOPWORDS: set[str] = {
    "the",
    "a",
    "an",
    "to",
    "of",
    "in",
    "on",
    "for",
    "and",
    "or",
    "but",
    "with",
    "from",
    "by",
    "as",
    "is",
    "are",
    "be",
    "was",
    "were",
    "it",
    "its",
    "this",
    "that",
    "those",
    "these",
    "you",
    "your",
    "we",
    "they",
    "their",
    "there",
    "have",
    "has",
    "had",
    "will",
    "can",
    "should",
    "would",
    "could",
    "just",
    "really",
    "like",
    "about",
    "into",
    "over",
    "some",
    "very",
}

_DEFAULT_FALLBACK_PHRASES: List[str] = [
    "motivational speaker on stage",
    "business team brainstorming",
    "audience applauding event",
    "closeup writing notes",
    "hands typing on laptop",
    "city skyline at sunset",
    "coach guiding athlete",
    "doctor consulting patient",
    "scientist working laboratory",
    "students listening in class",
    "team celebrating success",
    "entrepreneur pitching investors",
]


_BASIC_STOPWORDS: Dict[str, Set[str]] = {
    "en": set(_FALLBACK_STOPWORDS),
    "fr": {
        "le",
        "la",
        "les",
        "des",
        "une",
        "un",
        "dans",
        "avec",
        "pour",
        "mais",
        "plus",
        "sans",
        "tout",
        "tous",
        "toute",
        "toutes",
        "cela",
        "cette",
        "ces",
        "comme",
        "elles",
        "ils",
        "nous",
        "vous",
        "sont",
        "est",
        "avoir",
        "etre",
    },
}

_PROVIDER_QUERY_TEMPLATES: Tuple[str, ...] = (
    "{kw} stock b-roll",
    "{kw} stock footage",
    "{kw} cinematic b-roll",
    "teamwork {kw}",
    "{kw} cinematic video",
    "{kw} background footage",
)


def _extract_terms_from_text(
    text: str,
    *,
    min_length: int = 4,
    language: Optional[str] = None,
) -> List[str]:
    cleaned_text = _strip_accents(text or "").lower()
    if not cleaned_text:
        return []

    pattern = rf"[a-z0-9]{{{min_length},}}"
    raw_tokens = re.findall(pattern, cleaned_text)
    if not raw_tokens:
        return []

    stopwords = _BASIC_STOPWORDS.get((language or "").split("-")[0].strip().lower(), set())

    terms: List[str] = []
    seen: Set[str] = set()
    for token in raw_tokens:
        if stopwords and token in stopwords:
            continue
        if token not in seen:
            seen.add(token)
            terms.append(token)
    return terms


def _extract_ngrams(
    text: str,
    *,
    sizes: Sequence[int] | None = None,
    limit: int = 24,
    language: Optional[str] = None,
) -> List[str]:
    """Return deduplicated n-grams extracted from text.

    Only tokens of length >=3 that are not stopwords are used to build n-grams
    for the requested window sizes.  Punctuation is stripped prior to token
    extraction and the resulting n-grams are returned in discovery order.
    """

    cleaned = _strip_accents(text or "").lower()
    if not cleaned:
        return []

    token_pattern = re.compile(r"[a-z0-9]+")
    tokens = token_pattern.findall(cleaned)
    if not tokens:
        return []

    lang = (language or "").split("-")[0].strip().lower()
    stopwords = _BASIC_STOPWORDS.get(lang, set())

    filtered_tokens = [
        token
        for token in tokens
        if len(token) >= 3 and (not stopwords or token not in stopwords)
    ]
    if not filtered_tokens:
        return []

    windows = [size for size in (sizes or (2, 3)) if isinstance(size, int) and size >= 2]
    if not windows:
        windows = [2, 3]

    seen: Set[str] = set()
    ngrams: List[str] = []
    for size in sorted(set(windows)):
        if len(filtered_tokens) < size:
            continue
        for idx in range(len(filtered_tokens) - size + 1):
            chunk = filtered_tokens[idx : idx + size]
            if not chunk:
                continue
            phrase = " ".join(chunk)
            if phrase in seen:
                continue
            seen.add(phrase)
            ngrams.append(phrase)
            if len(ngrams) >= limit:
                return ngrams[:limit]

    return ngrams[:limit]


def _build_provider_queries_from_terms(terms: Sequence[str]) -> List[str]:
    queries: List[str] = []
    seen: Set[str] = set()
    for term in terms:
        if not term:
            continue
        for template in _PROVIDER_QUERY_TEMPLATES:
            candidate = template.format(kw=term).strip()
            if not candidate:
                continue
            normalized = " ".join(candidate.split()).lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            queries.append(normalized)
            if len(queries) >= 12:
                return queries[:12]
    return queries[:12]


def _fallback_keywords_from_transcript(
    transcript: str,
    *,
    min_terms: int = 8,
    max_terms: int = 12,
    language: Optional[str] = None,
) -> List[str]:
    min_terms = max(1, min_terms)
    max_terms = max(min_terms, max_terms)
    text = (transcript or "").strip()
    if not text:
        return _DEFAULT_FALLBACK_PHRASES[:max_terms]

    tokens = _extract_terms_from_text(text, min_length=4, language=language)
    filtered = [tok for tok in tokens if tok and tok not in _FALLBACK_STOPWORDS]
    if not filtered:
        return _DEFAULT_FALLBACK_PHRASES[:max_terms]

    unigram_counts = Counter(filtered)
    ngram_counts: Counter[str] = Counter()
    for window in (3, 2):
        if len(filtered) < window:
            continue
        for idx in range(len(filtered) - window + 1):
            chunk = filtered[idx : idx + window]
            if any(tok in _FALLBACK_STOPWORDS for tok in chunk):
                continue
            phrase = " ".join(chunk)
            ngram_counts[phrase] += 1

    phrases: List[str] = []
    for phrase, _ in ngram_counts.most_common(max_terms * 2):
        if phrase not in phrases:
            phrases.append(phrase)
        if len(phrases) >= max_terms:
            break

    if len(phrases) < max_terms:
        for token, _ in unigram_counts.most_common(max_terms * 2):
            candidate = f"{token} visuals"
            if candidate not in phrases:
                phrases.append(candidate)
            if len(phrases) >= max_terms:
                break

    if len(phrases) < min_terms:
        for fallback in _DEFAULT_FALLBACK_PHRASES:
            if fallback not in phrases:
                phrases.append(fallback)
            if len(phrases) >= max_terms:
                break

    return phrases[:max_terms]


def _merge_with_fallback(
    primary: Sequence[str],
    fallback: Sequence[str],
    *,
    min_count: int = 8,
    max_count: int = 12,
) -> List[str]:
    min_count = max(1, min_count)
    max_count = max(min_count, max_count)

    merged: List[str] = []
    seen: set[str] = set()

    for item in primary or []:
        candidate = _normalise_string(item)
        if candidate and candidate not in seen:
            seen.add(candidate)
            merged.append(candidate)
        if len(merged) >= max_count:
            return merged[:max_count]

    for item in fallback or []:
        if len(merged) >= max_count:
            break
        candidate = _normalise_string(item)
        if candidate and candidate not in seen:
            seen.add(candidate)
            merged.append(candidate)

    if len(merged) < min_count:
        for candidate in _DEFAULT_FALLBACK_PHRASES:
            if len(merged) >= max_count:
                break
            normalised = _normalise_string(candidate)
            if normalised and normalised not in seen:
                seen.add(normalised)
                merged.append(normalised)
            if len(merged) >= max_count:
                break

    return merged[:max_count]


def _build_keywords_prompt(transcript_snippet: str, target_lang: str) -> str:
    snippet = (transcript_snippet or "").strip()
    language = (target_lang or "en").strip().lower() or "en"
    return (
        "You are a JSON API for segment-level B-roll planning.\n"
        f"Use {language} language for every field.\n"
        "Return ONLY one JSON object with keys: broll_keywords, queries. No prose, no markdown.\n"
        "broll_keywords: 8-12 visual noun phrases (2-3 words), concrete and shootable.\n"
        f"queries: 8-12 short, filmable search queries (2-4 words) in {language}, provider-friendly.\n"
        "Banned tokens: that, this, it, they, we, you, thing, stuff, very, just, really, stock, footage, roll, cinematic, timelapse, background.\n"
        f"Segment transcript:\n{snippet}"
    )


def _build_json_metadata_prompt(transcript: str, *, video_id: Optional[str] = None) -> str:
    override = os.getenv("PIPELINE_LLM_JSON_PROMPT")
    cleaned = (transcript or "").strip()
    limit = _metadata_transcript_limit()
    if len(cleaned) > limit:
        cleaned = cleaned[:limit]

    if override:
        try:
            return override.format(transcript=cleaned, video_id=video_id or "")
        except KeyError:
            return override.replace("{transcript}", cleaned)

    video_reference = f"Video ID: {video_id}\n" if video_id else ""
    return (
        "Tu es un expert des mÃ©tadonnÃ©es pour vidÃ©os courtes (TikTok, Reels, Shorts).\n"
        "Retourne STRICTEMENT un objet JSON unique avec les clÃ©s exactes suivantes :\n"
        "  \"title\": chaÃ®ne accrocheuse en langue source,\n"
        "  \"description\": texte synthÃ©tique en 1 Ã  2 phrases,\n"
        "  \"hashtags\": tableau de 5 hashtags pertinents sans doublons,\n"
        "  \"broll_keywords\": tableau de 6 Ã  10 mots-clÃ©s visuels concrets,\n"
        "  \"queries\": tableau de 4 Ã  8 requÃªtes de recherche prÃªtes pour des banques d'images/vidÃ©os.\n"
        "N'ajoute aucune explication hors JSON.\n\n"
        f"{video_reference}TRANSCRIPT:\n{cleaned}"
    )


def _ollama_generate_json(
    prompt: str,
    *,
    model: Optional[str] = None,
    endpoint: Optional[str] = None,
    timeout: Optional[int] = None,
    options: Optional[Dict[str, Any]] = None,
    keep_alive: Optional[str] = None,
    json_mode: Optional[bool] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[int]]:
    """Send a prompt to Ollama and parse the JSON payload when possible."""

    target_endpoint = _resolve_ollama_endpoint(endpoint)
    model_name = _resolve_ollama_model(model)
    keep_alive_value = _resolve_keep_alive(keep_alive)
    json_mode_env = json_mode
    if json_mode_env is None:
        json_mode_env = _env_to_bool(os.getenv("PIPELINE_LLM_JSON_MODE"))
    payload: Dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }

    opts: Dict[str, Any] = {}
    if options:
        for key, value in options.items():
            if value is None:
                continue
            if key == "stop":
                try:
                    stop_values = [str(token) for token in value if str(token)]
                except TypeError:
                    stop_values = []
                if stop_values:
                    opts[key] = stop_values
                continue
            opts[key] = value
    if opts:
        payload["options"] = opts

    json_flag = bool(json_mode_env)
    if json_flag:
        payload["format"] = "json"
    if keep_alive_value:
        payload["keep_alive"] = keep_alive_value

    request_timeout = timeout if timeout is not None else 60
    url = f"{target_endpoint}/api/generate"

    try:
        response = requests.post(url, json=payload, timeout=request_timeout)
        response.raise_for_status()
    except requests.Timeout as exc:
        raise TimeoutError("Ollama request timed out") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    try:
        raw_payload = response.json()
    except json.JSONDecodeError as exc:
        response_text = response.text or ""
        logger.warning(
            "[LLM] Ollama JSON decoding failed",
            extra={"error": str(exc), "endpoint": target_endpoint},
        )
        parsed_fallback = _safe_parse_json(response_text) or _extract_json_braces(response_text)
        raw_length = len(response_text) if response_text else None
        return parsed_fallback or {}, {"response": response_text}, raw_length
    except ValueError as exc:
        response_text = response.text or ""
        logger.warning(
            "[LLM] Ollama returned non-JSON payload",
            extra={"error": str(exc), "endpoint": target_endpoint},
        )
        parsed_fallback = _safe_parse_json(response_text) or _extract_json_braces(response_text)
        raw_length = len(response_text) if response_text else None
        return parsed_fallback or {}, {"response": response_text}, raw_length

    raw_response = raw_payload.get("response")
    raw_length: Optional[int] = len(raw_response) if isinstance(raw_response, str) else None

    if isinstance(raw_response, dict):
        parsed = raw_response
    elif isinstance(raw_response, str):
        parsed = _safe_parse_json(raw_response)
        if not parsed:
            parsed = _extract_json_braces(raw_response)
    else:
        parsed = {}

    if not parsed:
        if isinstance(raw_payload.get("message"), dict):
            parsed = raw_payload["message"]
        elif isinstance(raw_payload.get("data"), dict):
            parsed = raw_payload["data"]

    return parsed or {}, raw_payload, raw_length


def _ollama_generate_text(
    prompt: str,
    *,
    model: str,
    options: Optional[Dict[str, Any]] = None,
    timeout: float = 60.0,
) -> Tuple[str, str, int, int, int]:
    """Generate plain text from Ollama with retries and diagnostics."""

    cleaned_prompt = str(prompt or "").strip()
    if not cleaned_prompt:
        return "", "empty_payload", 0

    model_name = _resolve_ollama_model(model)
    endpoint = _resolve_ollama_endpoint(None)
    url = f"{endpoint}/api/generate"

    option_payload: Dict[str, Any] = {}
    if options:
        for key, value in options.items():
            if value is None:
                continue
            if key == "stop":
                try:
                    stops = [str(token) for token in value if str(token)]
                except TypeError:
                    stops = []
                if stops:
                    option_payload[key] = stops
                continue
            option_payload[key] = value

    payload: Dict[str, Any] = {
        "model": model_name,
        "prompt": cleaned_prompt,
        "stream": True,
    }
    keep_alive = _resolve_keep_alive(None)
    if keep_alive:
        payload["keep_alive"] = keep_alive
    if option_payload:
        payload["options"] = option_payload

    backoffs = (0.6, 1.2, 2.4)
    raw_text = ""
    reason = "empty_payload"
    chunk_count = 0
    attempts_used = 0
    request_timeout = max(float(timeout), 1.0)

    for attempt in range(3):
        attempts_used = attempt + 1
        chunk_count = 0
        text_parts: List[str] = []
        try:
            with requests.post(
                url,
                json=payload,
                stream=True,
                timeout=(10, request_timeout),
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    chunk_count += 1
                    try:
                        message = json.loads(line)
                    except json.JSONDecodeError:
                        message = {"response": line}
                    piece = message.get("response")
                    if piece:
                        text_parts.append(str(piece))
                raw_text = "".join(text_parts)
                if raw_text.strip():
                    reason = ""
                    break
                reason = "empty_payload"
        except requests.Timeout:
            reason = "timeout"
        except requests.RequestException:
            reason = "transport_error"
        except Exception:
            reason = "stream_err"

        if attempt < len(backoffs):
            time.sleep(backoffs[attempt])

    stripped_text = raw_text.strip()
    return stripped_text, (reason or ""), chunk_count, len(raw_text), attempts_used


_TERM_MIN_LEN = 3
# Leave human subject tokens like people/team/crowd out of the generic blocklist.
_GENERIC_TERMS = {
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
    "entrepreneur",
    "presenter",
    "runner",
    "speaker",
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

_SCENE_PROMPT_MAP = {
    "achievement": "celebrating achievement with confetti",
    "brain": "closeup brain scan animation",
    "coach": "coach mentoring athlete on field",
    "collaboration": "team collaboration around laptop",
    "conversion": "analyzing conversion metrics on laptop",
    "data": "data analysts reviewing dashboards",
    "discussion": "roundtable discussion in modern office",
    "focus": "focused typing keyboard",
    "goal": "writing goals on notebook",
    "growth": "chart showing business growth",
    "innovation": "engineer working on futuristic prototype",
    "learning": "student taking notes in classroom",
    "marketing": "marketing team reviewing charts",
    "motivation": "motivational speaker addressing audience",
    "planning": "planning strategy on whiteboard",
    "productivity": "busy professional working at desk",
    "research": "scientist examining sample in laboratory",
    "strategy": "team strategizing with sticky notes",
    "success": "team celebrating success with high five",
    "team": "team brainstorming around table",
}

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
    "cahier": "notebook",
    "cahiers": "notebooks",
    "controle": "control",
    "contrle": "control",
    "dans": "in",
    "des": "of",
    "ecrit": "write",
    "ecrire": "write",
    "ecrivent": "writing",
    "equipe": "team",
    "equipes": "teams",
    "lequipe": "team",
    "focus": "focus",
    "ils": "they",
    "processus": "process",
    "recompense": "reward",
    "recompenses": "rewards",
    "rcompense": "reward",
    "rcompenses": "rewards",
    "sur": "on",
    "dure": "duration",
    "duree": "duration",
    "durees": "durations",
    "objectif": "goal",
    "objectifs": "goals",
    "claire": "clear",
    "clair": "clear",
    "clairs": "clear",
    "claires": "clear",
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
        raw = _split_basic_latin_runs(text.lower(), keep={"'", "-"})
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


def _normalise_scene_queries(values: Iterable[str], *, limit: Optional[int] = None) -> List[str]:
    seen: set[str] = set()
    queries: List[str] = []
    for value in values or []:
        if not isinstance(value, str):
            continue
        term = _strip_diacritics(value)
        term = term.lower().strip()
        term = re.sub(r"[^a-z0-9\s-]", " ", term)
        term = re.sub(r"\s+", " ", term)
        if len(term) < _TERM_MIN_LEN:
            continue
        if term in seen:
            continue
        seen.add(term)
        queries.append(term)
        if limit is not None and len(queries) >= limit:
            break
    return queries


_SPACY_MODEL: Optional[Any] = None
_SPACY_FAILED = False


def _load_spacy_model() -> Optional[Any]:
    global _SPACY_MODEL, _SPACY_FAILED
    if _SPACY_FAILED:
        return None
    if _SPACY_MODEL is not None:
        return _SPACY_MODEL
    try:
        import spacy  # type: ignore
    except Exception:
        _SPACY_FAILED = True
        return None
    try:
        _SPACY_MODEL = spacy.load("en_core_web_sm")  # type: ignore[attr-defined]
    except Exception:
        _SPACY_FAILED = True
        return None
    return _SPACY_MODEL


def _extract_noun_phrases(text: str, *, limit: int = 12) -> List[str]:
    ascii_text = _strip_diacritics(text or "").lower()
    ascii_text = re.sub(r"\s+", " ", ascii_text)
    if not ascii_text:
        return []
    candidates: Counter[str] = Counter()
    nlp = _load_spacy_model()
    if nlp is not None:
        try:
            doc = nlp(ascii_text)
        except Exception:
            doc = None  # type: ignore[assignment]
        if doc is not None:
            for chunk in getattr(doc, "noun_chunks", []):
                phrase = re.sub(r"[^a-z0-9\s-]", " ", chunk.text.lower())
                phrase = re.sub(r"\s+", " ", phrase).strip()
                if len(phrase) >= _TERM_MIN_LEN and phrase not in _GENERIC_TERMS:
                    candidates[phrase] += 1
            for token in doc:
                if token.pos_ in {"NOUN", "PROPN"}:
                    word = re.sub(r"[^a-z0-9-]", "", token.text.lower())
                    if len(word) >= _TERM_MIN_LEN and word not in _STOPWORDS_EN and word not in _GENERIC_TERMS:
                        candidates[word] += 1
    if not candidates:
        tokens = re.findall(r"[a-z0-9]+", ascii_text)
        buffer: List[str] = []
        for token in tokens:
            if len(token) < _TERM_MIN_LEN or token in _STOPWORDS_EN or token in _GENERIC_TERMS:
                if buffer:
                    phrase = " ".join(buffer)
                    candidates[phrase] += 1
                    buffer.clear()
                continue
            candidates[token] += 1
            buffer.append(token)
        if buffer:
            phrase = " ".join(buffer)
            candidates[phrase] += 1
    ranked = [phrase for phrase, _ in candidates.most_common(limit * 2)]
    ordered: List[str] = []
    seen: set[str] = set()
    for phrase in ranked:
        if not phrase:
            continue
        if phrase in seen:
            continue
        seen.add(phrase)
        ordered.append(phrase)
        if len(ordered) >= limit:
            break
    return ordered


def _map_scene_prompts(candidates: Sequence[str], *, limit: int) -> List[str]:
    prompts: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        tokens = candidate.split()
        prompt: Optional[str] = None
        for token in tokens:
            mapped = _SCENE_PROMPT_MAP.get(token)
            if mapped:
                prompt = mapped
                break
        if prompt is None and tokens:
            mapped = _SCENE_PROMPT_MAP.get(tokens[0])
            if mapped:
                prompt = mapped
        if prompt is None:
            continue
        normalised_prompt = _strip_diacritics(prompt)
        normalised_prompt = re.sub(r"\s+", " ", normalised_prompt).strip().lower()
        if len(normalised_prompt) < _TERM_MIN_LEN:
            continue
        if normalised_prompt in seen:
            continue
        seen.add(normalised_prompt)
        prompts.append(normalised_prompt)
        if len(prompts) >= limit:
            break
    return prompts


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
        if not keywords and keywords_raw:
            keywords = keywords_raw
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
    keywords = [
        _strip_diacritics(term)
        for term, _ in sorted(tfidf.items(), key=lambda item: item[1], reverse=True)
        if term not in _GENERIC_TERMS
    ][:top_k]
    bigrams: Counter[str] = Counter()
    for tokens in segments:
        for first, second in zip(tokens, tokens[1:]):
            if first == second:
                continue
            if first in _GENERIC_TERMS or second in _GENERIC_TERMS:
                continue
            phrase = f'{first} {second}'
            bigrams[_strip_diacritics(phrase)] += 1
    noun_phrases_raw = _extract_noun_phrases(transcript, limit=top_k)
    noun_phrase_candidates = _normalise_scene_queries(noun_phrases_raw, limit=top_k)
    noun_phrases = enforce_fetch_language(noun_phrase_candidates, "en")
    keywords.extend(noun_phrases)
    queries_candidates: List[str] = []
    queries_candidates.extend(_map_scene_prompts(noun_phrases, limit=max(4, top_k // 2)))
    queries_candidates.extend(phrase for phrase, _ in bigrams.most_common(max(4, top_k // 2)))
    keywords_normalised = _normalise_terms(keywords, limit=top_k)
    queries_normalised = _normalise_scene_queries(queries_candidates, limit=max(4, top_k // 2))
    keywords_enforced = enforce_fetch_language(keywords_normalised, "en")
    keywords_final = _normalise_terms(keywords_enforced, limit=top_k)
    queries_enforced = enforce_fetch_language(queries_normalised, "en")
    queries_final = _normalise_scene_queries(queries_enforced, limit=max(4, top_k // 2))
    return keywords_final, queries_final


def _normalise_dynamic_payload(
    raw: Dict[str, Any],
    *,
    transcript: str,
    disable_tfidf: bool = False,
    fallback_reason: str = "empty_payload",
) -> Dict[str, Any]:
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
        if disable_tfidf:
            reason = (fallback_reason or "unknown").strip() or "unknown"
            raise TfidfFallbackDisabled(
                f"TF-IDF fallback disabled (fallback_reason={reason})"
            )
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
RÃ”LE
Tu es planificateur B-roll pour vidÃ©os verticales (TikTok/Shorts, 9:16).

OBJECTIF
Ã€ partir de la transcription, dÃ©tecte le(s) domaine(s) librement (pas de liste fixe), puis gÃ©nÃ¨re :
1) des mots-clÃ©s et phrases-clÃ©s visuelles (scÃ¨nes filmables) utiles aux banques vidÃ©os,
2) des synonymes/variantes/termes proches pour CHAQUE mot-clÃ© (2â€“4 max),
3) des requÃªtes de recherche (2â€“4 mots, provider-friendly),
4) des briefs segmentaires facultatifs.

CONTRAINTES
- ZÃ©ro domaine prÃ©dÃ©fini. DÃ©duis librement 1â€“3 â€œdetected_domainsâ€ + confidence (0â€“1).
- Ã‰vite les anti-termes gÃ©nÃ©riques : people, thing, nice, background, start, generic, template, stock.
- Priorise des requÃªtes concrÃ¨tes et filmables : Â« sujet_action_contexte Â», objets prÃ©cis, lieux identifiables.
- FenÃªtres visuelles recommandÃ©es : 3â€“6 secondes. Format vertical.
- Si la langue de la transcription nâ€™est pas lâ€™anglais, produis les requÃªtes en langue dâ€™origine + anglais.

RÃ‰PONDS UNIQUEMENT EN JSON:
{{
  "detected_domains": [{{"name": "...", "confidence": 0.0}}],
  "language": "fr|en|â€¦",
  "keywords": ["..."],
  "synonyms": {{ "keyword": ["variante1","variante2"] }},
  "search_queries": ["..."],
  "segment_briefs": [
    {{"segment_index": 0, "window_s": 4, "keywords": ["..."], "queries": ["..."]}}
  ],
  "notes": "piÃ¨ges, anti-termes, risques"
}}

TRANSCRIPT (tronquÃ© Ã  1500â€“2000 caractÃ¨res):
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
        self.last_metadata: Dict[str, Any] = {}

        def _coerce_disable_flag(value: Any) -> Optional[bool]:
            if value is None:
                return None
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                parsed = _env_to_bool(value)
                if parsed is not None:
                    return parsed
                return None
            if isinstance(value, (int, float)):
                return bool(value)
            return None

        def _extract_disable_flag(source: Any) -> Optional[bool]:
            if source is None:
                return None
            flag = None
            if isinstance(source, dict):
                if "disable_tfidf_fallback" in source:
                    flag = _coerce_disable_flag(source.get("disable_tfidf_fallback"))
                if flag is None and "llm" in source:
                    flag = _extract_disable_flag(source.get("llm"))
                return flag
            if hasattr(source, "disable_tfidf_fallback"):
                flag = _coerce_disable_flag(getattr(source, "disable_tfidf_fallback"))
            if flag is None and hasattr(source, "llm"):
                flag = _extract_disable_flag(getattr(source, "llm"))
            return flag

        disable_flag = _env_to_bool(os.getenv("PIPELINE_DISABLE_TFIDF_FALLBACK"))
        if disable_flag is None:
            disable_flag = _extract_disable_flag(config)
        self._disable_tfidf_fallback = bool(disable_flag) if disable_flag is not None else False

        timeout_env = os.getenv("PIPELINE_LLM_TIMEOUT_S")
        num_predict_env = os.getenv("PIPELINE_LLM_NUM_PREDICT")
        temperature_env = os.getenv("PIPELINE_LLM_TEMP")
        top_p_env = os.getenv("PIPELINE_LLM_TOP_P")
        repeat_penalty_env = os.getenv("PIPELINE_LLM_REPEAT_PENALTY")
        stop_tokens_env = os.getenv("PIPELINE_LLM_STOP_TOKENS")

        def _parse_int(value: Optional[str], *, default: int, minimum: int) -> int:
            try:
                parsed = int(value) if value is not None else default
            except (TypeError, ValueError):
                parsed = default
            return max(minimum, parsed)

        def _parse_float(
            value: Optional[str],
            *,
            default: float,
            minimum: float,
            maximum: Optional[float] = None,
        ) -> float:
            try:
                parsed = float(value) if value is not None else default
            except (TypeError, ValueError):
                parsed = default
            if maximum is not None:
                parsed = min(maximum, parsed)
            return max(minimum, parsed)

        self._llm_timeout = _parse_int(timeout_env, default=35, minimum=5)
        self._llm_num_predict = _parse_int(num_predict_env, default=256, minimum=1)
        self._llm_temperature = _parse_float(temperature_env, default=0.3, minimum=0.0)
        self._llm_top_p = _parse_float(top_p_env, default=0.9, minimum=0.0, maximum=1.0)
        self._llm_repeat_penalty = _parse_float(repeat_penalty_env, default=1.1, minimum=0.0)
        self._llm_stop_tokens: Sequence[str] = tuple(_parse_stop_tokens(stop_tokens_env))

        def _clean_model(value: Optional[str], fallback: str) -> str:
            candidate = (value or "").strip()
            return candidate or fallback

        base_model = _clean_model(os.getenv("PIPELINE_LLM_MODEL"), "qwen2.5:7b")
        self.model_default = base_model
        self.model_json = _clean_model(os.getenv("PIPELINE_LLM_MODEL_JSON"), base_model)
        configured_text_model = _clean_model(os.getenv("PIPELINE_LLM_MODEL_TEXT"), base_model)
        self.model_text = configured_text_model

        fallback_note: Optional[str] = None
        fallback_target: Optional[str] = None
        ready_path = PROJECT_ROOT / 'tools' / 'out' / 'llm_ready.json'
        if ready_path.exists():
            try:
                ready_payload = json.loads(ready_path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    "[LLM] failed to load readiness metadata from %s: %s",
                    ready_path,
                    exc,
                )
            else:
                broken_models = {
                    str(model).strip()
                    for model in ready_payload.get('broken', [])
                    if str(model).strip()
                }
                text_ready = [
                    str(model).strip()
                    for model in ready_payload.get('text_ready', [])
                    if str(model).strip()
                ]
                if configured_text_model in broken_models and text_ready:
                    fallback_target = text_ready[0]
                    logger.warning(
                        "[LLM] text model %s listed as broken in %s; falling back to %s",
                        configured_text_model,
                        ready_path,
                        fallback_target,
                    )
                    self.model_text = fallback_target
                    fallback_note = (
                        f"configured model {configured_text_model} listed as broken in {ready_path.name}"
                    )
                elif configured_text_model in broken_models and not text_ready:
                    logger.warning(
                        "[LLM] text model %s listed as broken in %s but no fallback available",
                        configured_text_model,
                        ready_path,
                    )
                    fallback_note = (
                        f"configured model {configured_text_model} listed as broken in {ready_path.name}"
                    )

        logger.info(
            "[LLM] using timeout=%ss num_predict=%s temp=%s top_p=%s repeat_penalty=%s",
            self._llm_timeout,
            self._llm_num_predict,
            self._llm_temperature,
            self._llm_top_p,
            self._llm_repeat_penalty,
        )

        if fallback_note:
            logger.info(
                "[LLM] text model selected: %s (fallback: %s%s)",
                self.model_text,
                fallback_note,
                f", target={fallback_target}" if fallback_target else "",
            )
        else:
            logger.info("[LLM] text model selected: %s", self.model_text)

    def _fallback_queries_from(
        self,
        transcript: str,
        *,
        metadata_queries: Optional[Sequence[str]] = None,
        metadata_keywords: Optional[Sequence[str]] = None,
        language: Optional[str] = None,
    ) -> List[str]:
        """Return provider-friendly fallback queries prioritising metadata seeds."""

        target_lang = (language or _target_language_default()).strip().lower() or _target_language_default()

        direct_queries = _normalise_provider_terms(metadata_queries or [], target_lang=target_lang)
        keyword_bases = _normalise_search_terms(metadata_keywords or [], target_lang=target_lang)[:12]

        if not keyword_bases:
            ngram_bases = _extract_ngrams(
                transcript,
                sizes=(3, 2),
                limit=18,
                language=target_lang,
            )
            keyword_bases = ngram_bases[:12]

        if not keyword_bases:
            keyword_bases = _normalise_search_terms(
                _DEFAULT_FALLBACK_PHRASES,
                target_lang=target_lang,
            )[:12]

        templates = (
            "stock footage {kw}",
            "b-roll {kw}",
            "cinematic {kw}",
            "timelapse {kw}",
            "teamwork {kw}",
            "office {kw}",
            "city {kw}",
        )

        queries: List[str] = []
        seen: Set[str] = set()

        def _append(candidate: str) -> bool:
            cleaned = " ".join((candidate or "").split()).strip()
            if not cleaned:
                return False
            key = cleaned.lower()
            if key in seen:
                return False
            seen.add(key)
            queries.append(cleaned)
            return len(queries) >= 12

        for candidate in direct_queries:
            if _append(candidate):
                return queries[:12]

        for base in keyword_bases:
            keyword = base.strip()
            if not keyword:
                continue
            for template in templates:
                if _append(template.format(kw=keyword)):
                    return queries[:12]

        if not queries:
            for fallback in _build_provider_queries_from_terms(_DEFAULT_FALLBACK_PHRASES):
                if _append(fallback):
                    break

        return queries[:12]

    def _call_llm(self, prompt: str, *, max_tokens: int = 256) -> str:
        """Proxy Ollama completion that normalises failures and responses."""

        cleaned_prompt = str(prompt or "").strip()
        if not cleaned_prompt:
            raise ValueError("empty prompt")

        try:
            requested_tokens = int(max_tokens)
        except (TypeError, ValueError):
            requested_tokens = self._llm_num_predict
        bounded_tokens = max(1, min(requested_tokens, self._llm_num_predict))

        try:
            result = self._complete_text(cleaned_prompt, max_tokens=bounded_tokens)
        except TimeoutError as exc:
            raise ValueError("timeout") from exc
        except Exception as exc:
            raise ValueError("request failed") from exc

        if isinstance(result, dict):
            try:
                result = json.dumps(result, ensure_ascii=False)
            except Exception as exc:
                raise ValueError("request failed") from exc

        text = str(result or "").strip()
        if not text:
            raise ValueError("empty response")

        return text

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
                integration = self._shared_integration
                self._apply_llm_config(integration)
                return integration
            if self._integration is None:
                logger.info('[LLM] FAST_TESTS stub integration (local) initialised')
                self._integration = create_pipeline_integration(config)
            integration = self._integration
            self._apply_llm_config(integration)
            return integration

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
            integration = self._shared_integration
            self._apply_llm_config(integration)
            return integration

        if self._integration is None:
            logger.info('[LLM] Initialising local pipeline integration')
            try:
                self._integration = create_pipeline_integration(config)
            except Exception:  # pragma: no cover - defensive logging
                logger.exception('[LLM] Failed to initialise local pipeline integration')
                raise
        integration = self._integration
        self._apply_llm_config(integration)
        return integration

    def _apply_llm_config(self, integration: Optional[Any]) -> None:
        if integration is None:
            return

        llm = getattr(integration, "llm", None)
        if llm is None:
            return

        stop_values = list(self._llm_stop_tokens)

        configure = getattr(llm, "configure_generation", None)
        if callable(configure):
            try:
                configure(
                    num_predict=self._llm_num_predict,
                    temperature=self._llm_temperature,
                    top_p=self._llm_top_p,
                    repeat_penalty=self._llm_repeat_penalty,
                    stop=stop_values,
                )
            except Exception:  # pragma: no cover - best effort configuration
                pass

        for attr, value in (
            ("timeout", self._llm_timeout),
            ("num_predict", self._llm_num_predict),
            ("temperature", self._llm_temperature),
            ("top_p", self._llm_top_p),
            ("repeat_penalty", self._llm_repeat_penalty),
        ):
            try:
                setattr(llm, attr, value)
            except Exception:  # pragma: no cover - best effort configuration
                continue

        for attr in ("stop", "stop_sequences", "stop_tokens", "stop_words"):
            if hasattr(llm, attr):
                try:
                    setattr(llm, attr, stop_values)
                except Exception:  # pragma: no cover - best effort configuration
                    pass

        set_stop = getattr(llm, "set_stop_sequences", None)
        if callable(set_stop):
            try:
                set_stop(stop_values)
            except Exception:  # pragma: no cover - best effort configuration
                pass

        target_model = getattr(self, "model_text", None)
        if isinstance(target_model, str) and target_model.strip():
            target_model = target_model.strip()
            for attr in ("model", "model_name"):
                if hasattr(llm, attr):
                    try:
                        setattr(llm, attr, target_model)
                    except Exception:  # pragma: no cover - best effort configuration
                        pass
            for attr in ("model", "model_name"):
                if hasattr(integration, attr):
                    try:
                        setattr(integration, attr, target_model)
                    except Exception:  # pragma: no cover - best effort configuration
                        pass

    # --- Compatibility layer for plain text completions ---------------------
    def _complete_text(self, prompt: str, *, max_tokens: int = 800, purpose: str = "generic") -> Any:
        """Attempt to invoke a completion method on the underlying integration.

        Tries common method names; falls back to calling the optimized LLM engine
        when available (integration.llm._call_llm). Returns raw text.
        """
        if purpose == "dynamic":
            model_candidates = (
                getattr(self, "model_text", None),
                getattr(self, "model_default", None),
                _DEFAULT_OLLAMA_MODEL,
            )
            resolved_model: Optional[str] = None
            for candidate in model_candidates:
                if isinstance(candidate, str):
                    cleaned = candidate.strip()
                    if cleaned:
                        resolved_model = cleaned
                        break
            if not resolved_model:
                resolved_model = _DEFAULT_OLLAMA_MODEL

            bounded_tokens = min(max_tokens, self._llm_num_predict)
            options = {
                "num_predict": bounded_tokens,
                "temperature": self._llm_temperature,
                "top_p": self._llm_top_p,
                "repeat_penalty": self._llm_repeat_penalty,
                "stop": list(self._llm_stop_tokens),
            }

            text, reason, chunk_count, raw_len, attempts = _ollama_generate_text(
                prompt,
                model=resolved_model,
                options=options,
                timeout=float(self._llm_timeout),
            )
            logger.info(
                "[LLM] dynamic text completion model=%s json_mode=False prompt_len=%s raw_len=%s chunks=%s reason_if_empty=%s attempts=%s",
                resolved_model,
                len(str(prompt or "")),
                raw_len,
                chunk_count,
                reason or "",
                attempts,
            )
            if not text:
                raise DynamicCompletionError(reason or "empty_payload")
            return text, reason, chunk_count

        integration = self._get_integration()

        target_model = getattr(self, "model_text", None) or getattr(self, "model_default", None)
        if isinstance(target_model, str):
            target_model = target_model.strip() or None

        # 1) Try common completion-shaped methods on integration directly
        last_error: Optional[BaseException] = None
        timed_out = False
        if purpose == "dynamic":
            method_order: Tuple[str, ...] = ("complete", "chat", "generate")
        else:
            method_order = ("complete_json", "complete", "chat", "generate")

        for attr in method_order:
            fn = getattr(integration, attr, None)
            if callable(fn):
                try:
                    return self._invoke_completion(fn, prompt, max_tokens=max_tokens, model=target_model)
                except Exception as exc:  # pragma: no cover - robustness
                    last_error = exc
                    if self._is_timeout_error(exc):
                        timed_out = True
                        break

        # 2) Try going through the optimized LLM engine if exposed
        if not timed_out:
            llm = getattr(integration, "llm", None)
            if llm is not None:
                if purpose == "dynamic":
                    llm_methods: Tuple[str, ...] = ("complete", "chat", "generate")
                else:
                    llm_methods = ("complete_json", "complete", "generate")
                for attr in llm_methods:
                    fn = getattr(llm, attr, None)
                    if callable(fn):
                        try:
                            return self._invoke_completion(fn, prompt, max_tokens=max_tokens, model=target_model)
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
                        bounded_max_tokens = min(max_tokens, self._llm_num_predict)
                        call_kwargs: Dict[str, Any] = {}
                        try:
                            call_signature = inspect.signature(call)
                        except (TypeError, ValueError):
                            call_signature = None

                        if call_signature is not None:
                            params = call_signature.parameters
                            if "temperature" in params:
                                call_kwargs["temperature"] = self._llm_temperature
                            if "max_tokens" in params:
                                call_kwargs["max_tokens"] = bounded_max_tokens
                            if "num_predict" in params:
                                call_kwargs["num_predict"] = self._llm_num_predict
                            if "timeout" in params:
                                call_kwargs["timeout"] = self._llm_timeout
                            for key in ("stop", "stop_sequences", "stop_tokens", "stop_words"):
                                if key in params:
                                    call_kwargs[key] = list(self._llm_stop_tokens)
                                    break
                            if target_model:
                                cleaned_model = target_model.strip()
                                if cleaned_model:
                                    if "model" in params and "model" not in call_kwargs:
                                        call_kwargs["model"] = cleaned_model
                                    elif "model_name" in params and "model_name" not in call_kwargs:
                                        call_kwargs["model_name"] = cleaned_model
                            if "json_mode" in params and purpose == "dynamic":
                                call_kwargs["json_mode"] = False
                        else:
                            call_kwargs = {
                                "temperature": self._llm_temperature,
                                "max_tokens": bounded_max_tokens,
                                "timeout": self._llm_timeout,
                            }
                            if target_model:
                                cleaned_model = target_model.strip() if isinstance(target_model, str) else None
                                if cleaned_model:
                                    call_kwargs["model"] = cleaned_model
                        if purpose == "dynamic" and "json_mode" not in call_kwargs:
                            call_kwargs["json_mode"] = False

                        ok, text, err = call(prompt, **call_kwargs)
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

    def _invoke_completion(self, fn, prompt: str, *, max_tokens: int, model: Optional[str] = None) -> Any:
        """Call a completion function with a bounded timeout when supported."""

        kwargs: Dict[str, Any] = {}
        bounded_max_tokens = min(max_tokens, self._llm_num_predict)
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):  # pragma: no cover - builtins
            signature = None

        if signature is not None:
            params = signature.parameters
            if "timeout" in params:
                kwargs["timeout"] = self._llm_timeout
            if "max_tokens" in params and "max_tokens" not in kwargs:
                kwargs["max_tokens"] = bounded_max_tokens
            if "num_predict" in params:
                kwargs["num_predict"] = self._llm_num_predict
            if "temperature" in params:
                kwargs.setdefault("temperature", self._llm_temperature)
            if "top_p" in params:
                kwargs["top_p"] = self._llm_top_p
            if model:
                cleaned_model = model.strip()
                if cleaned_model:
                    if "model" in params and "model" not in kwargs:
                        kwargs["model"] = cleaned_model
                    elif "model_name" in params and "model_name" not in kwargs:
                        kwargs["model_name"] = cleaned_model
            stop_keys = ("stop", "stop_sequences", "stop_tokens", "stop_words")
            for key in stop_keys:
                if key in params:
                    kwargs[key] = list(self._llm_stop_tokens)
                    break
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
        last_dynamic_error: Optional[DynamicCompletionError] = None
        last_reason: Optional[str] = None
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
                completion = self._complete_text(prompt, max_tokens=token_budget, purpose="dynamic")
            except TimeoutError:
                last_reason = "timeout"
                logger.warning(
                    '[LLM] dynamic context attempt %s timed out (limit=%s, tokens=%s)',
                    idx,
                    limit,
                    token_budget,
                )
                continue
            except DynamicCompletionError as exc:
                last_dynamic_error = exc
                last_reason = exc.reason
                logger.warning(
                    '[LLM] dynamic context attempt %s failed (limit=%s, tokens=%s, fallback_reason=%s)',
                    idx,
                    limit,
                    token_budget,
                    exc.reason,
                )
                continue
            except Exception:
                last_reason = "integration_error"
                logger.exception('[LLM] dynamic context attempt %s failed', idx)
                continue

            parsed_reason = ""
            if isinstance(completion, tuple):
                raw_text = completion[0]
                parsed_reason = completion[1] if len(completion) > 1 and isinstance(completion[1], str) else ""
            else:
                raw_text = completion

            parsed_payload: Dict[str, Any] = {}
            if isinstance(raw_text, dict):
                parsed_payload = raw_text
            else:
                parsed_payload = _safe_parse_json(raw_text)

            if parsed_payload:
                raw_payload = parsed_payload
                break

            if parsed_reason:
                last_reason = parsed_reason
            elif isinstance(raw_text, str) and raw_text.strip():
                last_reason = last_reason or "invalid_json"
            elif not raw_text:
                last_reason = last_reason or "empty_payload"
        if not raw_payload:
            fallback_reason = last_reason or (last_dynamic_error.reason if last_dynamic_error else "empty_payload")
            if self._disable_tfidf_fallback:
                reason = (fallback_reason or "unknown").strip() or "unknown"
                raise TfidfFallbackDisabled(
                    f"TF-IDF fallback disabled (fallback_reason={reason})"
                )
            logger.warning(
                '[LLM] dynamic context fell back to TF-IDF (no structured payload, fallback_reason=%s)',
                fallback_reason,
                extra={'fallback_reason': fallback_reason},
            )
            fallback_payload = _normalise_dynamic_payload(
                {},
                transcript=transcript_text or '',
                disable_tfidf=self._disable_tfidf_fallback,
                fallback_reason=fallback_reason,
            )
            if last_dynamic_error is not None:
                last_dynamic_error.payload = fallback_payload
                raise last_dynamic_error
            raise DynamicCompletionError(fallback_reason, payload=fallback_payload)

        return _normalise_dynamic_payload(
            raw_payload,
            transcript=transcript_text or '',
            disable_tfidf=self._disable_tfidf_fallback,
            fallback_reason=last_reason or "llm_missing_terms",
        )


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

        stored_keywords = _normalise_search_terms(
            broll_keywords or metadata.get("broll_keywords") or [],
            target_lang=_target_language_default(),
        )[:12]
        stored_queries = _normalise_search_terms(
            metadata.get("queries") or broll_data.get("queries") or [],
            target_lang=_target_language_default(),
        )[:12]
        self.last_metadata = {
            "queries": stored_queries,
            "broll_keywords": stored_keywords,
        }

        logger.info('[LLM] Metadata generated', extra={'hashtags': len(hashtags), 'broll_keywords': len(broll_keywords)})
        return LLMMetadata(
            title=title,
            description=description,
            hashtags=hashtags,
            broll_keywords=broll_keywords,
            raw_payload=result,
        )


    def _segment_llm_json(
        self,
        seg_text: str,
        *,
        timeout_s: Optional[float] = None,
        num_predict: Optional[int] = None,
    ) -> Optional[Dict[str, List[str]]]:
        snippet = (seg_text or "").strip()
        if not snippet:
            return None

        snippet = snippet[:1200]

        try:
            requested_predict = int(num_predict) if num_predict is not None else self._llm_num_predict
        except (TypeError, ValueError):
            requested_predict = self._llm_num_predict
        base_predict = max(64, requested_predict)

        timeout_value = int(timeout_s if timeout_s is not None else self._llm_timeout)

        base_options = {
            "num_predict": base_predict,
            "temperature": float(self._llm_temperature),
            "top_p": float(self._llm_top_p),
            "repeat_penalty": float(self._llm_repeat_penalty),
        }

        prompt = SEGMENT_JSON_PROMPT.format(segment_text=snippet)

        def _resolve_payload(parsed_payload: Any, raw_payload: Any) -> Dict[str, Any]:
            payload = parsed_payload if isinstance(parsed_payload, dict) else {}
            if payload:
                return payload
            if isinstance(raw_payload, dict):
                candidates: List[Any] = []
                for key in ("response", "content", "data", "message", "result"):
                    value = raw_payload.get(key)
                    if isinstance(value, dict) and value:
                        return value
                    if isinstance(value, str) and value.strip():
                        candidates.append(value)
                for candidate in candidates:
                    parsed = _coerce_ollama_json(candidate)
                    if isinstance(parsed, dict) and parsed:
                        return parsed
            return payload if isinstance(payload, dict) else {}

        attempts: List[int] = [base_predict]
        expanded_predict = min(base_predict + 64, 384)
        if expanded_predict > base_predict:
            attempts.append(expanded_predict)

        for attempt_index, predict_value in enumerate(attempts):
            options = dict(base_options)
            options["num_predict"] = predict_value
            try:
                parsed_payload, raw_payload, _ = _ollama_generate_json(
                    prompt,
                    model=self.model_json,
                    options=options,
                    timeout=timeout_value,
                    json_mode=True,
                )
            except TimeoutError:
                logger.warning(
                    "[LLM] Segment JSON request timed out",
                    extra={"timeout_s": timeout_value, "attempt": attempt_index + 1},
                )
                return None
            except Exception as exc:
                logger.warning(
                    "[LLM] Segment JSON request failed",
                    extra={"error": str(exc), "attempt": attempt_index + 1},
                )
                return None

            payload = _resolve_payload(parsed_payload, raw_payload)
            if not payload:
                if attempt_index == len(attempts) - 1:
                    return None
                continue

            keywords = _sanitize_queries(
                payload.get("broll_keywords") or payload.get("brollKeywords") or [],
                max_words=3,
                max_len=12,
            )
            queries = _sanitize_queries(_concretize_queries(payload.get("queries") or []), max_len=12)

            if keywords or queries:
                return {"broll_keywords": keywords, "queries": queries}

        return None

    def _tfidf_segment_fallback(
        self,
        segment_text: str,
        *,
        target_lang: Optional[str] = None,
        seed_queries: Optional[Sequence[str]] = None,
        seed_keywords: Optional[Sequence[str]] = None,
    ) -> Dict[str, List[str]]:
        snippet = (segment_text or "").strip()
        language = (target_lang or _target_language_default()).strip().lower() or _target_language_default()
        if self._disable_tfidf_fallback:
            raise TfidfFallbackDisabled('TF-IDF fallback disabled (fallback_reason=segment_generation)')
        base_keywords, base_queries = _tfidf_fallback(strip_banned(snippet))

        keyword_sources: List[str] = []
        for source in (seed_keywords, base_keywords, seed_queries, _DEFAULT_FALLBACK_PHRASES):
            if not source:
                continue
            keyword_sources = [str(item).strip() for item in source if str(item).strip()]
            if keyword_sources:
                break
        if not keyword_sources:
            keyword_sources = [str(item).strip() for item in _DEFAULT_FALLBACK_PHRASES]

        normalised_keywords = _normalise_search_terms(keyword_sources, target_lang=language)[:12]
        if not normalised_keywords:
            normalised_keywords = _normalise_search_terms(_DEFAULT_FALLBACK_PHRASES, target_lang=language)[:12]

        normalised_queries = _normalise_search_terms(base_queries or [], target_lang=language)[:12]
        if not normalised_queries:
            provider_queries = _build_provider_queries_from_terms(normalised_keywords or _DEFAULT_FALLBACK_PHRASES)
            normalised_queries = _normalise_search_terms(provider_queries, target_lang=language)[:12]

        keywords = _sanitize_queries(normalised_keywords, max_words=3, max_len=12)
        queries = _sanitize_queries(_concretize_queries(normalised_queries), max_len=12)

        if not queries:
            fallback_provider = _build_provider_queries_from_terms(normalised_keywords or _DEFAULT_FALLBACK_PHRASES)
            queries = _sanitize_queries(_concretize_queries(fallback_provider), max_len=12)

        if not keywords:
            keywords = _sanitize_queries(
                _normalise_search_terms(_DEFAULT_FALLBACK_PHRASES, target_lang=language),
                max_words=3,
                max_len=12,
            )

        return {"broll_keywords": keywords, "queries": queries}

    def generate_hints_for_segment(self, text: str, start: float, end: float) -> Dict:
        """Produce visual search hints for a transcript segment."""

        snippet = (text or "").strip()
        target_lang = _target_language_default()

        previous_metadata = getattr(self, "last_metadata", {}) or {}
        seed_queries = list(previous_metadata.get("queries") or [])
        seed_keywords = list(previous_metadata.get("broll_keywords") or [])

        for cached_query in _LAST_METADATA_QUERIES.get("values", []) or []:
            if cached_query not in seed_queries:
                seed_queries.append(cached_query)
        for cached_keyword in _LAST_METADATA_KEYWORDS.get("values", []) or []:
            if cached_keyword not in seed_keywords:
                seed_keywords.append(cached_keyword)

        seed_queries = seed_queries[:24]
        seed_keywords = seed_keywords[:24]

        if not snippet:
            fallback_queries_full = self._fallback_queries_from(
                "",
                metadata_queries=seed_queries,
                metadata_keywords=seed_keywords,
                language=target_lang,
            )
            if not fallback_queries_full:
                fallback_queries_full = _build_provider_queries_from_terms(_DEFAULT_FALLBACK_PHRASES)

            fallback_keywords_full = _normalise_search_terms(
                seed_keywords or seed_queries or _DEFAULT_FALLBACK_PHRASES,
                target_lang=target_lang,
            )[:12]
            if not fallback_keywords_full:
                fallback_keywords_full = _normalise_search_terms(
                    _DEFAULT_FALLBACK_PHRASES,
                    target_lang=target_lang,
                )[:12]

            fallback_queries = (fallback_queries_full or [])[:12]
            if not fallback_queries:
                fallback_queries = _build_provider_queries_from_terms(_DEFAULT_FALLBACK_PHRASES)[:12]

            source_label = (
                "metadata_keywords_fallback"
                if seed_keywords or seed_queries
                else "transcript_fallback"
            )

            result = {
                "title": "",
                "description": "",
                "queries": fallback_queries[:8],
                "broll_keywords": fallback_keywords_full[:8],
                "filters": {"min_duration_s": 3.0},
                "source": source_label,
            }

            self.last_metadata = {
                "queries": fallback_queries[:12],
                "broll_keywords": fallback_keywords_full[:12],
            }

            return result


        max_chars = min(_metadata_transcript_limit(), 1800)
        prompt_snippet = snippet[:max_chars]

        llm_payload = self._segment_llm_json(
            prompt_snippet,
            timeout_s=self._llm_timeout,
            num_predict=self._llm_num_predict,
        )

        if not llm_payload:
            retry_timeout = min(self._llm_timeout + 15, 90)
            retry_predict = self._llm_num_predict + 64
            llm_payload = self._segment_llm_json(
                prompt_snippet,
                timeout_s=retry_timeout,
                num_predict=retry_predict,
            )

        llm_failed = False
        if not llm_payload:
            fallback_reason = "segment_generation"
            if self._disable_tfidf_fallback:
                raise TfidfFallbackDisabled(
                    f"TF-IDF fallback disabled (fallback_reason={fallback_reason})"
                )
            logger.warning(
                "[LLM] Segment hint generation fell back to TF-IDF heuristics (fallback_reason=%s)",
                fallback_reason,
                extra={"segment_start": start, "segment_end": end, "fallback_reason": fallback_reason},
            )
            llm_payload = self._tfidf_segment_fallback(
                prompt_snippet,
                target_lang=target_lang,
                seed_queries=seed_queries,
                seed_keywords=seed_keywords,
            )
            llm_failed = True

        payload: Dict[str, Any] = dict(llm_payload or {})

        title = _normalise_string(payload.get("title"))
        description = _normalise_string(payload.get("description"))

        raw_keywords = (
            payload.get("broll_keywords")
            or payload.get("brollKeywords")
            or payload.get("keywords")
            or []
        )
        primary_keywords = _normalise_search_terms(raw_keywords, target_lang=target_lang)[:12]
        raw_llm_queries = list(payload.get("queries") or [])
        primary_queries = _normalise_search_terms(
            _concretize_queries(raw_llm_queries),
            target_lang=target_lang,
        )[:12]

        metadata_terms_source: List[str] = []
        if seed_keywords:
            metadata_terms_source.extend(seed_keywords)
        if not metadata_terms_source and seed_queries:
            metadata_terms_source.extend(seed_queries)
        if not metadata_terms_source:
            metadata_terms_source = list(_LAST_METADATA_KEYWORDS.get("values", []))
        metadata_terms = _normalise_search_terms(
            metadata_terms_source,
            target_lang=target_lang,
        )[:12]

        fallback_terms: List[str] = []
        used_metadata_fallback = False
        used_transcript_fallback = llm_failed

        if len(primary_keywords) < 6 or len(primary_queries) < 6:
            if metadata_terms:
                fallback_terms = metadata_terms
                used_metadata_fallback = True
            else:
                transcript_terms = _fallback_keywords_from_transcript(
                    snippet,
                    min_terms=8,
                    max_terms=12,
                    language=target_lang,
                )
                fallback_terms = _normalise_search_terms(transcript_terms, target_lang=target_lang)[:12]
                if not fallback_terms:
                    fallback_terms = _normalise_search_terms(
                        _DEFAULT_FALLBACK_PHRASES,
                        target_lang=target_lang,
                    )[:12]
                used_transcript_fallback = True

        combined_keywords = primary_keywords[:]
        for term in fallback_terms:
            if term not in combined_keywords:
                combined_keywords.append(term)
            if len(combined_keywords) >= 12:
                break
        if len(combined_keywords) < 6:
            for fallback in _normalise_search_terms(_DEFAULT_FALLBACK_PHRASES, target_lang=target_lang):
                if fallback not in combined_keywords:
                    combined_keywords.append(fallback)
                if len(combined_keywords) >= 12:
                    break
        broll_keywords = combined_keywords[:12]

        combined_queries = primary_queries[:]
        fallback_queries: List[str] = []
        metadata_used_for_queries = False
        if fallback_terms or len(combined_queries) < 6:
            metadata_for_keywords: Sequence[str] = (
                fallback_terms or seed_keywords or broll_keywords or primary_keywords
            )
            fallback_queries = self._fallback_queries_from(
                snippet,
                metadata_queries=seed_queries,
                metadata_keywords=metadata_for_keywords,
                language=target_lang,
            )
            if not fallback_queries:
                fallback_queries = _build_provider_queries_from_terms(metadata_for_keywords)
            if fallback_queries and (seed_queries or seed_keywords):
                metadata_used_for_queries = True

        for query in fallback_queries:
            if query not in combined_queries:
                combined_queries.append(query)
            if len(combined_queries) >= 12:
                break
        if fallback_queries:
            if metadata_used_for_queries:
                used_metadata_fallback = True
            elif not used_transcript_fallback:
                used_transcript_fallback = True
        if len(combined_queries) < 6:
            extra_sources = broll_keywords or _normalise_search_terms(
                _DEFAULT_FALLBACK_PHRASES,
                target_lang=target_lang,
            )
            extra_queries = _build_provider_queries_from_terms(extra_sources)
            for candidate in _normalise_search_terms(extra_queries, target_lang=target_lang):
                if candidate not in combined_queries:
                    combined_queries.append(candidate)
                if len(combined_queries) >= 12:
                    break
        queries = combined_queries[:12]

        if not queries:
            final_fallback_queries = self._fallback_queries_from(
                snippet,
                metadata_queries=seed_queries,
                metadata_keywords=seed_keywords or broll_keywords or primary_keywords,
                language=target_lang,
            )
            if not final_fallback_queries:
                final_fallback_queries = _build_provider_queries_from_terms(_DEFAULT_FALLBACK_PHRASES)
            queries = (final_fallback_queries or [])[:12]
            if queries and (seed_queries or seed_keywords):
                used_metadata_fallback = True
            elif queries:
                used_transcript_fallback = True
            if not queries:
                queries = _build_provider_queries_from_terms(_DEFAULT_FALLBACK_PHRASES)[:12]

        source = "llm_segment"
        if used_metadata_fallback:
            source = "metadata_keywords_fallback"
        elif used_transcript_fallback:
            source = "transcript_fallback"

        try:
            seg_duration = max(0.0, float(end) - float(start))
        except Exception:
            seg_duration = 0.0
        min_d = 3.0
        if seg_duration > 0:
            # target ~90% of the segment length, clamped to [2.5, 5.5]
            min_d = max(2.5, min(5.5, round(seg_duration * 0.9, 2)))
        # Enforce portrait orientation and dynamic duration bounds per segment
        # Also cap overly long clips and reject near-square portrait by min aspect ratio
        filters = {
            "orientation": "portrait",
            "min_duration_s": float(min_d),
            "max_duration_s": 8.0,
            # 9:16 ~= 1.78; allow a bit of slack to accept 4:7-ish while rejecting 4:5
            "min_aspect_ratio": 1.6,
        }

        seg_idx = f"{start:.2f}-{end:.2f}"
        queries = _sanitize_queries(_concretize_queries(queries), max_len=12)
        broll_keywords = _sanitize_queries(broll_keywords, max_words=3, max_len=12)

        logger.info(
            "[BROLL][LLM] segment=%s queries=%s (source=%s)",
            seg_idx,
            queries[:4],
            source,
        )

        self.last_metadata = {
            "title": title or "",
            "description": description or "",
            "queries": list(queries),
            "broll_keywords": list(broll_keywords),
            "filters": dict(filters),
            "source": source,
        }

        queries_with_synonyms = _augment_with_synonyms(queries, max_extra_per=1, limit=12)

        return {
            "title": title,
            "description": description,
            "queries": queries_with_synonyms,
            "broll_keywords": broll_keywords,
            "filters": filters,
            "source": source,
        }


    def provider_fallback_queries(
        self,
        transcript: str = "",
        *,
        max_items: int = 12,
        language: Optional[str] = None,
    ) -> Tuple[List[str], str]:
        """Return provider-friendly fallback queries and their origin label.

        The routine first reuses cached metadata queries (already normalised for
        providers), then derives queries from cached keywords, and finally
        falls back to transcript heuristics or default phrases.  The returned
        queries are deduplicated and normalised using the provider helper to
        ensure consistent downstream behaviour.
        """

        limit = max(1, int(max_items or 0))
        target_lang = (language or "").strip().lower() or _target_language_default()

        collected: List[str] = []
        seen: Set[str] = set()
        origin = "none"

        def _extend(values: Sequence[str], label: str) -> bool:
            nonlocal origin
            if not values:
                return False
            normalised = _normalise_provider_terms(values, target_lang=target_lang)
            added = False
            for candidate in normalised:
                if not candidate or candidate in seen:
                    continue
                seen.add(candidate)
                collected.append(candidate)
                added = True
                if len(collected) >= limit:
                    break
            if added and origin == "none":
                origin = label
            return len(collected) >= limit

        cached_queries = _LAST_METADATA_QUERIES.get("values", [])
        if _extend(cached_queries, "metadata_cached_queries"):
            return collected[:limit], origin

        cached_keywords = _LAST_METADATA_KEYWORDS.get("values", [])
        if cached_keywords:
            keyword_queries = _build_provider_queries_from_terms(cached_keywords)
            if _extend(keyword_queries, "metadata_keywords_fallback"):
                return collected[:limit], origin

        cleaned_transcript = (transcript or "").strip()
        if cleaned_transcript:
            transcript_terms = _fallback_keywords_from_transcript(
                cleaned_transcript,
                min_terms=8,
                max_terms=max(limit, 8),
                language=target_lang,
            )
            transcript_queries = _build_provider_queries_from_terms(transcript_terms)
            if _extend(transcript_queries, "transcript_fallback"):
                return collected[:limit], origin

        default_queries = _build_provider_queries_from_terms(_DEFAULT_FALLBACK_PHRASES)
        _extend(default_queries, "default_fallback")

        if origin == "none":
            origin = "default_fallback" if collected else "none"

        return collected[:limit], origin


def get_shared_llm_service() -> LLMMetadataGeneratorService:
    """Return a process-wide shared instance of :class:`LLMMetadataGeneratorService`."""

    global _SHARED
    if _SHARED is None:
        with _SHARED_LOCK:
            if _SHARED is None:
                _SHARED = LLMMetadataGeneratorService()
    return _SHARED


def generate_metadata_as_json(
    transcript: str,
    *,
    timeout_s: float | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Call Ollama directly, enforce JSON output and normalise the fields."""

    try:
        service = get_shared_llm_service()
    except Exception:
        logger.exception("[LLM] Unable to initialise shared metadata service")
        service = None

    def _remember_last_metadata(queries: Sequence[str], keywords: Sequence[str]) -> None:
        if service is None:
            return
        try:
            service.last_metadata = {
                "queries": list(queries),
                "broll_keywords": list(keywords),
            }
        except Exception:
            logger.debug("[LLM] Failed to persist last metadata on shared service", exc_info=True)

    cleaned_transcript = (transcript or "").strip()
    if not cleaned_transcript:
        logger.warning("[LLM] Empty transcript provided for metadata generation")
        failure = _empty_metadata_payload()
        failure["raw_response_length"] = 0
        _remember_last_metadata(failure.get("queries") or [], failure.get("broll_keywords") or [])
        return failure

    limit = _metadata_transcript_limit()
    if len(cleaned_transcript) > limit:
        cleaned_transcript = cleaned_transcript[:limit]

    video_id = kwargs.get("video_id")
    use_keywords_prompt = _keywords_first_enabled()
    target_lang = _target_language_default()
    prompt = (
        _build_keywords_prompt(cleaned_transcript, target_lang)
        if use_keywords_prompt
        else _build_json_metadata_prompt(cleaned_transcript, video_id=video_id)
    )

    model_default = (os.getenv("PIPELINE_LLM_MODEL") or "qwen2.5:7b").strip() or "qwen2.5:7b"
    model_name = (os.getenv("PIPELINE_LLM_MODEL_JSON") or model_default).strip() or model_default

    original_timeout = None
    if timeout_s is not None:
        original_timeout = os.getenv("PIPELINE_LLM_TIMEOUT_S")
        try:
            timeout_override = max(1.0, float(timeout_s))
        except (TypeError, ValueError):
            timeout_override = 1.0
        os.environ["PIPELINE_LLM_TIMEOUT_S"] = str(timeout_override)

    parsed_payload: Dict[str, Any] = {}
    raw_payload: Dict[str, Any] = {}
    raw_length: Optional[int] = None
    error: Optional[BaseException] = None
    started = time.perf_counter()
    try:
        if use_keywords_prompt:
            try:
                num_predict_env = os.getenv("PIPELINE_LLM_NUM_PREDICT")
                try:
                    configured_predict = int(num_predict_env) if num_predict_env is not None else 192
                except (TypeError, ValueError):
                    configured_predict = 192
                bounded_predict = max(64, min(192, configured_predict))

                temp_env = os.getenv("PIPELINE_LLM_TEMP")
                try:
                    configured_temp = float(temp_env) if temp_env is not None else 0.2
                except (TypeError, ValueError):
                    configured_temp = 0.2
                bounded_temp = max(0.0, min(0.4, configured_temp))

                parsed_payload, raw_payload, raw_length = _ollama_generate_json(
                    prompt,
                    model=model_name,
                    options={
                        "num_predict": bounded_predict,
                        "temperature": bounded_temp,
                        "top_p": 0.9,
                    },
                    json_mode=True,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                error = exc
        else:
            raw_payload = _ollama_json(prompt, model=model_name)
            if isinstance(raw_payload, dict):
                parsed_payload = raw_payload
            else:
                parsed_payload = _coerce_ollama_json(raw_payload)
                if not isinstance(parsed_payload, dict):
                    parsed_payload = {}
            try:
                raw_length = len(json.dumps(raw_payload, ensure_ascii=False)) if raw_payload else 0
            except Exception:
                raw_length = 0
    finally:
        if timeout_s is not None:
            if original_timeout is None:
                os.environ.pop("PIPELINE_LLM_TIMEOUT_S", None)
            else:
                os.environ["PIPELINE_LLM_TIMEOUT_S"] = original_timeout

    duration = time.perf_counter() - started

    if raw_length is None:
        try:
            raw_length = len(json.dumps(raw_payload, ensure_ascii=False)) if raw_payload else 0
        except Exception:
            raw_length = 0

    if error is not None:
        failure = _empty_metadata_payload()
        failure["raw_response_length"] = 0
        logger.warning(
            "[LLM] Metadata generation failed",
            extra={
                "model": model_name,
                "duration_s": round(duration, 3),
                "transcript_length": len(cleaned_transcript),
                "keywords_prompt": use_keywords_prompt,
                "error": str(error),
            },
        )
        _remember_last_metadata([], [])
        return failure

    metadata_section: Dict[str, Any] = parsed_payload if isinstance(parsed_payload, dict) else {}

    for key in ("metadata", "result", "data"):
        candidate = metadata_section.get(key) if isinstance(metadata_section, dict) else None
        if isinstance(candidate, dict):
            metadata_section = candidate

    if not metadata_section:
        logger.warning(
            "[LLM] JSON metadata payload missing",
            extra={
                "model": model_name,
                "duration_s": round(duration, 3),
                "transcript_length": len(cleaned_transcript),
                "keywords_prompt": use_keywords_prompt,
            },
        )
        metadata_section = {}

    if not isinstance(metadata_section, dict):
        logger.warning(
            "[LLM] Metadata payload is not a JSON object",
            extra={
                "model": model_name,
                "duration_s": round(duration, 3),
                "transcript_length": len(cleaned_transcript),
                "keywords_prompt": use_keywords_prompt,
            },
        )
        metadata_section = {}

    defaults = _default_metadata_payload()

    title = _normalise_string(metadata_section.get("title")) or defaults["title"]
    description = _normalise_string(metadata_section.get("description")) or defaults["description"]

    hashtags_disabled = _hashtags_disabled()
    if hashtags_disabled:
        metadata_section["hashtags"] = []

    hashtags: List[str] = []
    if not hashtags_disabled:
        hashtags = _normalise_hashtags(_as_list(metadata_section.get("hashtags")))[:5]

    raw_keyword_values = (
        metadata_section.get("broll_keywords")
        or metadata_section.get("brollKeywords")
        or metadata_section.get("keywords")
        or []
    )
    initial_broll = _normalise_search_terms(raw_keyword_values, target_lang=target_lang)[:12]

    queries_raw = metadata_section.get("queries")
    initial_queries = _normalise_search_terms(queries_raw, target_lang=target_lang)[:12]

    fallback_terms: List[str] = []
    keywords_fallback = len(initial_broll) < 6
    queries_fallback = len(initial_queries) < 6
    if keywords_fallback or queries_fallback:
        fallback_terms = _fallback_keywords_from_transcript(
            cleaned_transcript,
            min_terms=8,
            max_terms=12,
            language=target_lang,
        )

    if keywords_fallback:
        broll_keywords = _merge_with_fallback(initial_broll, fallback_terms, min_count=8, max_count=12)
    else:
        broll_keywords = initial_broll[:12]

    if queries_fallback:
        seed_terms = fallback_terms or broll_keywords
        queries = _merge_with_fallback(initial_queries, seed_terms, min_count=8, max_count=12)
    else:
        queries = initial_queries[:12]

    broll_keywords = _normalise_search_terms(broll_keywords, target_lang=target_lang)[:12]

    queries = _normalise_search_terms(queries, target_lang=target_lang)[:12]
    if len(queries) < 6 and broll_keywords:
        fallback_queries = _build_provider_queries_from_terms(broll_keywords)
        for candidate in _normalise_search_terms(fallback_queries, target_lang=target_lang):
            if candidate not in queries:
                queries.append(candidate)
            if len(queries) >= 12:
                break
    queries = queries[:12]

    provider_keywords = _normalise_provider_terms(broll_keywords, target_lang=target_lang)[:12]
    provider_queries = _normalise_provider_terms(queries, target_lang=target_lang)[:12]

    broll_keywords = provider_keywords
    queries = provider_queries

    # Final hardening at clip level: enforce anti-generic constraints
    queries = _sanitize_queries(_concretize_queries(queries), max_len=12)
    broll_keywords = _sanitize_queries(broll_keywords, max_words=3, max_len=12)

    if not hashtags and not hashtags_disabled:
        hashtags = _hashtags_from_keywords(broll_keywords, limit=5)

    now = time.time()
    global _LAST_METADATA_KEYWORDS, _LAST_METADATA_QUERIES
    _LAST_METADATA_KEYWORDS["values"] = list(broll_keywords)
    _LAST_METADATA_KEYWORDS["updated_at"] = now
    _LAST_METADATA_QUERIES["values"] = list(queries)
    _LAST_METADATA_QUERIES["updated_at"] = now

    result: Dict[str, Any] = {
        "title": title,
        "description": description,
        "hashtags": hashtags,
        "broll_keywords": broll_keywords,
        "queries": queries,
        "raw_response_length": raw_length if raw_length is not None else 0,
        "llm_status": "ok",
    }

    logger.info(
        "[LLM] JSON metadata generated",
        extra={
            "model": model_name,
            "duration_s": round(duration, 3),
            "transcript_length": len(cleaned_transcript),
            "hashtags": len(hashtags),
            "broll_keywords": len(broll_keywords),
            "queries": len(queries),
            "queries_fallback": queries_fallback,
            "keywords_fallback": keywords_fallback,
            "keywords_prompt": use_keywords_prompt,
        },
    )

    _remember_last_metadata(queries, broll_keywords)

    return result










