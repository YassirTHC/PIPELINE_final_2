"""Typed configuration loader and startup logging utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
from typing import Mapping, MutableMapping

TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}

_CONFIG_LOGGER = logging.getLogger("video_pipeline.config")

_SETTINGS_CACHE: "Settings | None" = None
_STARTUP_LOG_EMITTED = False


def _clean(value: object | None) -> str:
    if value is None:
        return ""
    return str(value).strip()


def to_bool(value: object | None, default: bool = False, *, name: str | None = None) -> bool:
    """Convert *value* to ``bool`` with a permissive parser."""

    if value is None:
        return bool(default)
    text = _clean(value).lower()
    if not text:
        return bool(default)
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    if name:
        _CONFIG_LOGGER.warning(
            "Invalid boolean for %s: %r – using default %s",
            name,
            value,
            default,
        )
    return bool(default)


def to_int(
    value: object | None,
    default: int,
    *,
    name: str | None = None,
    minimum: int | None = None,
) -> int:
    """Convert *value* to ``int`` or fall back to ``default``."""

    if value is None:
        parsed = default
    else:
        try:
            parsed = int(_clean(value))
        except (TypeError, ValueError):
            if name:
                _CONFIG_LOGGER.warning(
                    "Invalid integer for %s: %r – using default %s",
                    name,
                    value,
                    default,
                )
            parsed = default
    if minimum is not None and parsed < minimum:
        if name:
            _CONFIG_LOGGER.warning(
                "Integer for %s below minimum %s: %s – clamping",
                name,
                minimum,
                parsed,
            )
        parsed = max(minimum, default)
    return int(parsed)


def to_float(
    value: object | None,
    default: float,
    *,
    name: str | None = None,
    minimum: float | None = None,
) -> float:
    """Convert *value* to ``float`` or fall back to ``default``."""

    if value is None:
        parsed = default
    else:
        try:
            parsed = float(_clean(value))
        except (TypeError, ValueError):
            if name:
                _CONFIG_LOGGER.warning(
                    "Invalid float for %s: %r – using default %s",
                    name,
                    value,
                    default,
                )
            parsed = default
    if minimum is not None and parsed < minimum:
        if name:
            _CONFIG_LOGGER.warning(
                "Float for %s below minimum %s: %s – clamping",
                name,
                minimum,
                parsed,
            )
        parsed = max(minimum, default)
    return float(parsed)


def csv_list(value: object | None) -> list[str]:
    """Split a CSV string into a list, trimming empty tokens."""

    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        tokens = [str(item).strip() for item in value if str(item).strip()]
    else:
        tokens = [segment.strip() for segment in str(value).replace(";", ",").split(",")]
    seen: dict[str, None] = {}
    for token in tokens:
        if token:
            seen.setdefault(token, None)
    return list(seen.keys())


def mask(value: object | None) -> str | None:
    """Return a masked representation for secret values."""

    text = _clean(value)
    if not text:
        return None
    if len(text) <= 4:
        return f"****{text}"
    return f"****{text[-4:]}"


@dataclass(slots=True)
class LLMSettings:
    model: str = "qwen2.5:7b"
    model_json: str = ""
    model_text: str = ""
    endpoint: str = "http://localhost:11434"
    base_url: str = "http://localhost:11434"
    keep_alive: str = "30m"
    timeout_stream_s: float = 60.0
    timeout_fallback_s: float = 45.0
    min_chars: int = 8
    max_attempts: int = 3
    num_predict: int = 256
    temperature: float = 0.3
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    num_ctx: int = 4096
    fallback_trunc: int = 3500
    keywords_first: bool = True
    force_non_stream: bool = False
    disable_hashtags: bool = False
    json_prompt: str | None = None
    json_mode: bool = False
    json_transcript_limit: int | None = None
    target_lang: str = "en"

    @property
    def effective_json_model(self) -> str:
        return self.model_json or self.model

    @property
    def effective_text_model(self) -> str:
        return self.model_text or self.model

    def to_log_payload(self) -> dict[str, object]:
        return {
            "model": self.model,
            "model_json": self.model_json or None,
            "model_text": self.model_text or None,
            "effective_json_model": self.effective_json_model,
            "effective_text_model": self.effective_text_model,
            "endpoint": self.endpoint,
            "base_url": self.base_url,
            "keep_alive": self.keep_alive,
            "timeout_stream_s": self.timeout_stream_s,
            "timeout_fallback_s": self.timeout_fallback_s,
            "min_chars": self.min_chars,
            "max_attempts": self.max_attempts,
            "num_predict": self.num_predict,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
            "num_ctx": self.num_ctx,
            "fallback_trunc": self.fallback_trunc,
            "keywords_first": self.keywords_first,
            "force_non_stream": self.force_non_stream,
            "disable_hashtags": self.disable_hashtags,
            "json_prompt": self.json_prompt,
            "json_mode": self.json_mode,
            "json_transcript_limit": self.json_transcript_limit,
            "target_lang": self.target_lang,
        }


@dataclass(slots=True)
class BrollSettings:
    min_start_s: float = 2.0
    min_gap_s: float = 1.5
    no_repeat_s: float = 6.0

    def to_log_payload(self) -> dict[str, float]:
        return {
            "min_start_s": self.min_start_s,
            "min_gap_s": self.min_gap_s,
            "no_repeat_s": self.no_repeat_s,
        }


@dataclass(slots=True)
class FetchSettings:
    timeout_s: float = 8.0
    max_per_keyword: int = 8
    allow_images: bool = True
    allow_videos: bool = True
    providers: tuple[str, ...] = ("pixabay",)
    provider_limits: dict[str, int] = field(default_factory=dict)
    api_keys: dict[str, str] = field(default_factory=dict)

    def to_log_payload(self) -> dict[str, object]:
        masked_keys = {name: mask(value) for name, value in self.api_keys.items()}
        return {
            "timeout_s": self.timeout_s,
            "max_per_keyword": self.max_per_keyword,
            "allow_images": self.allow_images,
            "allow_videos": self.allow_videos,
            "providers": self.providers,
            "provider_limits": dict(self.provider_limits),
            "api_keys": masked_keys,
        }


@dataclass(slots=True)
class Settings:
    clips_dir: Path = Path("clips")
    output_dir: Path = Path("output")
    temp_dir: Path = Path("temp")
    llm: LLMSettings = field(default_factory=LLMSettings)
    broll: BrollSettings = field(default_factory=BrollSettings)
    fetch: FetchSettings = field(default_factory=FetchSettings)
    tfidf_fallback_disabled: bool = False
    llm_max_queries_per_segment: int = 3
    max_segments_in_flight: int = 1
    fast_tests: bool = False

    def to_log_payload(self) -> dict[str, object]:
        return {
            "clips_dir": str(self.clips_dir),
            "output_dir": str(self.output_dir),
            "temp_dir": str(self.temp_dir),
            "llm": self.llm.to_log_payload(),
            "broll": self.broll.to_log_payload(),
            "fetch": self.fetch.to_log_payload(),
            "tfidf_fallback_disabled": self.tfidf_fallback_disabled,
            "llm_max_queries_per_segment": self.llm_max_queries_per_segment,
            "max_segments_in_flight": self.max_segments_in_flight,
            "fast_tests": self.fast_tests,
        }


def _lookup(env: Mapping[str, object | None], key: str) -> object | None:
    value = env.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed if trimmed else None
    return value


def _provider_limits(env: Mapping[str, object | None]) -> dict[str, int]:
    limits: dict[str, int] = {}
    for key, value in env.items():
        if not key.startswith("BROLL_") or not key.endswith("_MAX_PER_KEYWORD"):
            continue
        provider = key[len("BROLL_") : -len("_MAX_PER_KEYWORD")].lower()
        if provider in {"fetch", "allow", "provider"}:
            continue
        limits[provider] = to_int(value, default=0, name=key, minimum=1)
    return limits


def load_settings(env: Mapping[str, object | None] | None = None) -> Settings:
    """Resolve the pipeline settings from the environment."""

    if env is None:
        env_mapping: MutableMapping[str, object | None] = dict(os.environ)
    else:
        env_mapping = dict(env)

    clips_dir = Path(_lookup(env_mapping, "PIPELINE_CLIPS_DIR") or "clips")
    output_dir = Path(_lookup(env_mapping, "PIPELINE_OUTPUT_DIR") or "output")
    temp_dir = Path(_lookup(env_mapping, "PIPELINE_TEMP_DIR") or "temp")

    model = _lookup(env_mapping, "PIPELINE_LLM_MODEL") or "qwen2.5:7b"
    llm = LLMSettings(
        model=model,
        model_json=_lookup(env_mapping, "PIPELINE_LLM_MODEL_JSON") or "",
        model_text=_lookup(env_mapping, "PIPELINE_LLM_MODEL_TEXT") or "",
        endpoint=_lookup(env_mapping, "PIPELINE_LLM_ENDPOINT")
        or _lookup(env_mapping, "PIPELINE_LLM_BASE_URL")
        or "http://localhost:11434",
        base_url=_lookup(env_mapping, "PIPELINE_LLM_BASE_URL")
        or _lookup(env_mapping, "PIPELINE_LLM_ENDPOINT")
        or "http://localhost:11434",
        keep_alive=_lookup(env_mapping, "PIPELINE_LLM_KEEP_ALIVE") or "30m",
        timeout_stream_s=to_float(
            env_mapping.get("PIPELINE_LLM_TIMEOUT_S"),
            default=60.0,
            name="PIPELINE_LLM_TIMEOUT_S",
            minimum=0.1,
        ),
        timeout_fallback_s=to_float(
            env_mapping.get("PIPELINE_LLM_FALLBACK_TIMEOUT_S"),
            default=45.0,
            name="PIPELINE_LLM_FALLBACK_TIMEOUT_S",
            minimum=0.1,
        ),
        min_chars=to_int(
            env_mapping.get("PIPELINE_LLM_MIN_CHARS"),
            default=8,
            name="PIPELINE_LLM_MIN_CHARS",
            minimum=0,
        ),
        max_attempts=to_int(
            env_mapping.get("PIPELINE_LLM_MAX_ATTEMPTS"),
            default=3,
            name="PIPELINE_LLM_MAX_ATTEMPTS",
            minimum=1,
        ),
        num_predict=to_int(
            env_mapping.get("PIPELINE_LLM_NUM_PREDICT"),
            default=256,
            name="PIPELINE_LLM_NUM_PREDICT",
            minimum=1,
        ),
        temperature=to_float(
            env_mapping.get("PIPELINE_LLM_TEMP"),
            default=0.3,
            name="PIPELINE_LLM_TEMP",
            minimum=0.0,
        ),
        top_p=to_float(
            env_mapping.get("PIPELINE_LLM_TOP_P"),
            default=0.9,
            name="PIPELINE_LLM_TOP_P",
            minimum=0.0,
        ),
        repeat_penalty=to_float(
            env_mapping.get("PIPELINE_LLM_REPEAT_PENALTY"),
            default=1.1,
            name="PIPELINE_LLM_REPEAT_PENALTY",
            minimum=0.0,
        ),
        num_ctx=to_int(
            env_mapping.get("PIPELINE_LLM_NUM_CTX"),
            default=4096,
            name="PIPELINE_LLM_NUM_CTX",
            minimum=1,
        ),
        fallback_trunc=to_int(
            env_mapping.get("PIPELINE_LLM_FALLBACK_TRUNC"),
            default=3500,
            name="PIPELINE_LLM_FALLBACK_TRUNC",
            minimum=1,
        ),
        keywords_first=to_bool(
            env_mapping.get("PIPELINE_LLM_KEYWORDS_FIRST"),
            default=True,
            name="PIPELINE_LLM_KEYWORDS_FIRST",
        ),
        force_non_stream=to_bool(
            env_mapping.get("PIPELINE_LLM_FORCE_NON_STREAM"),
            default=False,
            name="PIPELINE_LLM_FORCE_NON_STREAM",
        ),
        disable_hashtags=to_bool(
            env_mapping.get("PIPELINE_LLM_DISABLE_HASHTAGS"),
            default=False,
            name="PIPELINE_LLM_DISABLE_HASHTAGS",
        ),
        json_prompt=_lookup(env_mapping, "PIPELINE_LLM_JSON_PROMPT"),
        json_mode=to_bool(
            env_mapping.get("PIPELINE_LLM_JSON_MODE"),
            default=False,
            name="PIPELINE_LLM_JSON_MODE",
        ),
        json_transcript_limit=None
        if _lookup(env_mapping, "PIPELINE_LLM_JSON_TRANSCRIPT_LIMIT") is None
        else to_int(
            env_mapping.get("PIPELINE_LLM_JSON_TRANSCRIPT_LIMIT"),
            default=0,
            name="PIPELINE_LLM_JSON_TRANSCRIPT_LIMIT",
            minimum=1,
        ),
        target_lang=_lookup(env_mapping, "PIPELINE_LLM_TARGET_LANG") or "en",
    )

    broll = BrollSettings(
        min_start_s=to_float(
            env_mapping.get("PIPELINE_BROLL_MIN_START_SECONDS"),
            default=2.0,
            name="PIPELINE_BROLL_MIN_START_SECONDS",
            minimum=0.0,
        ),
        min_gap_s=to_float(
            env_mapping.get("PIPELINE_BROLL_MIN_GAP_SECONDS"),
            default=1.5,
            name="PIPELINE_BROLL_MIN_GAP_SECONDS",
            minimum=0.0,
        ),
        no_repeat_s=to_float(
            env_mapping.get("PIPELINE_BROLL_NO_REPEAT_SECONDS"),
            default=6.0,
            name="PIPELINE_BROLL_NO_REPEAT_SECONDS",
            minimum=0.0,
        ),
    )

    providers_source = _lookup(env_mapping, "BROLL_FETCH_PROVIDER") or _lookup(
        env_mapping, "AI_BROLL_FETCH_PROVIDER"
    )
    providers_list = csv_list(providers_source or "pixabay")
    providers = tuple(token.lower() for token in providers_list)

    fetch_timeout = to_float(
        env_mapping.get("PIPELINE_FETCH_TIMEOUT_S"),
        default=8.0,
        name="PIPELINE_FETCH_TIMEOUT_S",
        minimum=0.1,
    )
    max_default = to_int(
        env_mapping.get("FETCH_MAX"),
        default=8,
        name="FETCH_MAX",
        minimum=1,
    )
    max_per_keyword = to_int(
        env_mapping.get("BROLL_FETCH_MAX_PER_KEYWORD"),
        default=max_default,
        name="BROLL_FETCH_MAX_PER_KEYWORD",
        minimum=1,
    )

    provider_limits = _provider_limits(env_mapping)
    api_keys = {
        name: _lookup(env_mapping, name)
        for name in ("PEXELS_API_KEY", "PIXABAY_API_KEY", "UNSPLASH_ACCESS_KEY")
        if _lookup(env_mapping, name)
    }

    fetch = FetchSettings(
        timeout_s=fetch_timeout,
        max_per_keyword=max_per_keyword,
        allow_images=to_bool(
            env_mapping.get("BROLL_FETCH_ALLOW_IMAGES"),
            default=True,
            name="BROLL_FETCH_ALLOW_IMAGES",
        ),
        allow_videos=to_bool(
            env_mapping.get("BROLL_FETCH_ALLOW_VIDEOS"),
            default=True,
            name="BROLL_FETCH_ALLOW_VIDEOS",
        ),
        providers=providers,
        provider_limits=provider_limits,
        api_keys={name: str(value) for name, value in api_keys.items()},
    )

    tfidf_disabled = to_bool(
        env_mapping.get("PIPELINE_TFIDF_FALLBACK_DISABLED"),
        default=False,
        name="PIPELINE_TFIDF_FALLBACK_DISABLED",
    )
    alias_disabled = to_bool(
        env_mapping.get("PIPELINE_DISABLE_TFIDF_FALLBACK"),
        default=tfidf_disabled,
        name="PIPELINE_DISABLE_TFIDF_FALLBACK",
    )
    if alias_disabled and not tfidf_disabled:
        tfidf_disabled = True
        _CONFIG_LOGGER.warning(
            "PIPELINE_DISABLE_TFIDF_FALLBACK is deprecated; use PIPELINE_TFIDF_FALLBACK_DISABLED",
        )

    settings = Settings(
        clips_dir=clips_dir,
        output_dir=output_dir,
        temp_dir=temp_dir,
        llm=llm,
        broll=broll,
        fetch=fetch,
        tfidf_fallback_disabled=tfidf_disabled,
        llm_max_queries_per_segment=to_int(
            env_mapping.get("PIPELINE_LLM_MAX_QUERIES_PER_SEGMENT"),
            default=3,
            name="PIPELINE_LLM_MAX_QUERIES_PER_SEGMENT",
            minimum=1,
        ),
        max_segments_in_flight=to_int(
            env_mapping.get("PIPELINE_MAX_SEGMENTS_IN_FLIGHT"),
            default=1,
            name="PIPELINE_MAX_SEGMENTS_IN_FLIGHT",
            minimum=1,
        ),
        fast_tests=to_bool(
            env_mapping.get("PIPELINE_FAST_TESTS"),
            default=False,
            name="PIPELINE_FAST_TESTS",
        ),
    )

    return settings


def log_effective_settings(settings: Settings, *, logger: logging.Logger | None = None) -> None:
    """Emit a single `[CONFIG]` log with the resolved settings."""

    global _STARTUP_LOG_EMITTED
    if _STARTUP_LOG_EMITTED:
        return
    _STARTUP_LOG_EMITTED = True

    payload = settings.to_log_payload()
    message = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    (logger or _CONFIG_LOGGER).info("[CONFIG] effective=%s", message)


def get_settings() -> Settings:
    global _SETTINGS_CACHE
    if _SETTINGS_CACHE is None:
        _SETTINGS_CACHE = load_settings()
    return _SETTINGS_CACHE


def set_settings(settings: Settings) -> None:
    global _SETTINGS_CACHE
    _SETTINGS_CACHE = settings


def reset_settings_cache_for_tests() -> None:
    global _SETTINGS_CACHE
    _SETTINGS_CACHE = None


def reset_startup_log_for_tests() -> None:
    global _STARTUP_LOG_EMITTED
    _STARTUP_LOG_EMITTED = False
