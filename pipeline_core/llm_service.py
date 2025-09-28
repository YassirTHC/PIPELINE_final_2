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


_DEFAULT_STOP_TOKENS: Tuple[str, ...] = ("```", "\n\n\n", "END_OF_CONTEXT", "</json>")

_DEFAULT_OLLAMA_ENDPOINT = "http://127.0.0.1:11434"
_DEFAULT_OLLAMA_MODEL = "mistral:7b-instruct"
_DEFAULT_OLLAMA_KEEP_ALIVE = "5m"


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


def _normalise_string_list(value: Any) -> List[str]:
    if not value:
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
    seen: set[str] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, str):
            cleaned = candidate.strip()
        else:
            cleaned = str(candidate).strip()
        if not cleaned:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        normalised.append(cleaned)
    return normalised


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
        "Tu es un expert des métadonnées pour vidéos courtes (TikTok, Reels, Shorts).\n"
        "Retourne STRICTEMENT un objet JSON unique avec les clés exactes suivantes :\n"
        "  \"title\": string accrocheur en français ou langue source,\n"
        "  \"description\": string synthétique (1-2 phrases),\n"
        "  \"hashtags\": array de 8-18 chaînes uniques (avec #),\n"
        "  \"broll_keywords\": array de mots-clés visuels,\n"
        "  \"queries\": array de requêtes de recherche de stock (2-5 mots).\n"
        "Aucune explication ni texte hors JSON.\n\n"
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
        raw_payload: Dict[str, Any] = response.json()
    except ValueError as exc:
        raise ValueError("Ollama did not return JSON payload") from exc

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

        logger.info(
            "[LLM] using timeout=%ss num_predict=%s temp=%s top_p=%s repeat_penalty=%s",
            self._llm_timeout,
            self._llm_num_predict,
            self._llm_temperature,
            self._llm_top_p,
            self._llm_repeat_penalty,
        )

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
                        else:
                            call_kwargs = {
                                "temperature": self._llm_temperature,
                                "max_tokens": bounded_max_tokens,
                                "timeout": self._llm_timeout,
                            }

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

    def _invoke_completion(self, fn, prompt: str, *, max_tokens: int) -> Any:
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


def generate_metadata_as_json(
    segments: Sequence[Dict[str, Any]] | str,
    *,
    video_id: Optional[str] = None,
    service: Optional[LLMMetadataGeneratorService] = None,
) -> Dict[str, Any]:
    """Generate metadata using the Ollama JSON endpoint."""

    if isinstance(segments, str):
        transcript_text = segments
    else:
        valid_segments = [
            SubtitleSegment.from_mapping(seg)
            for seg in segments
            if seg and seg.get("text")
        ]
        transcript_text = " ".join(seg.text for seg in valid_segments if seg.text)

    transcript = (transcript_text or "").strip()
    if not transcript:
        raise ValueError("No transcript provided for metadata generation")

    llm_service = service or LLMMetadataGeneratorService.get_shared()
    prompt = _build_json_metadata_prompt(transcript, video_id=video_id)

    options = {
        "num_predict": llm_service._llm_num_predict,
        "temperature": llm_service._llm_temperature,
        "top_p": llm_service._llm_top_p,
        "repeat_penalty": llm_service._llm_repeat_penalty,
        "stop": list(llm_service._llm_stop_tokens),
    }

    parsed_payload, raw_payload, raw_length = _ollama_generate_json(
        prompt,
        timeout=llm_service._llm_timeout,
        options=options,
    )

    if not parsed_payload:
        raise ValueError("Empty metadata payload returned by LLM")

    metadata_section: Dict[str, Any] = parsed_payload
    for key in ("metadata", "result", "data"):
        candidate = metadata_section.get(key) if isinstance(metadata_section, dict) else None
        if isinstance(candidate, dict):
            metadata_section = candidate

    title = _normalise_string(
        metadata_section.get("title")
        or parsed_payload.get("title")
    )
    description = _normalise_string(
        metadata_section.get("description")
        or parsed_payload.get("description")
    )

    hashtags_raw = (
        metadata_section.get("hashtags")
        or parsed_payload.get("hashtags")
    )
    hashtags = _normalise_hashtags(hashtags_raw)

    broll_raw = (
        metadata_section.get("broll_keywords")
        or metadata_section.get("keywords")
        or parsed_payload.get("broll_keywords")
        or parsed_payload.get("keywords")
    )
    if not broll_raw and isinstance(parsed_payload.get("broll_data"), dict):
        broll_raw = parsed_payload["broll_data"].get("keywords")
    broll_keywords = _normalise_string_list(broll_raw)

    queries_raw = (
        metadata_section.get("queries")
        or metadata_section.get("search_queries")
        or parsed_payload.get("queries")
        or parsed_payload.get("search_queries")
    )
    if not queries_raw and isinstance(parsed_payload.get("broll_data"), dict):
        queries_raw = parsed_payload["broll_data"].get("queries")
    queries = _normalise_string_list(queries_raw)

    if not any((title, description, hashtags, broll_keywords, queries)):
        raise ValueError("Metadata payload missing expected fields")

    logger.info(
        "[LLM] JSON metadata generated",
        extra={
            "hashtags": len(hashtags),
            "broll_keywords": len(broll_keywords),
            "queries": len(queries),
            "raw_response_length": raw_length,
        },
    )

    result: Dict[str, Any] = {
        "title": title,
        "description": description,
        "hashtags": hashtags,
        "broll_keywords": broll_keywords,
        "queries": queries,
        "raw_payload": raw_payload,
    }
    if raw_length is not None:
        result["raw_response_length"] = raw_length

    return result
