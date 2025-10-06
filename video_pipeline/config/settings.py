"""Strongly typed configuration loader for the video pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
from typing import Mapping

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
    """Convert an environment value to ``bool`` with graceful fallback."""

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
    """Convert *value* to ``int`` when possible, otherwise ``default``."""

    if value is None:
        return int(default)
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
        return int(default)
    if minimum is not None and parsed < minimum:
        if name:
            _CONFIG_LOGGER.warning(
                "Integer for %s below minimum %s: %s – clamping",
                name,
                minimum,
                parsed,
            )
        return int(max(minimum, default))
    return parsed


def to_float(
    value: object | None,
    default: float,
    *,
    name: str | None = None,
    minimum: float | None = None,
) -> float:
    """Convert *value* to ``float`` when possible, otherwise ``default``."""

    if value is None:
        return float(default)
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
        return float(default)
    if minimum is not None and parsed < minimum:
        if name:
            _CONFIG_LOGGER.warning(
                "Float for %s below minimum %s: %s – clamping",
                name,
                minimum,
                parsed,
            )
        return float(max(minimum, default))
    return parsed


def csv_list(value: object | None) -> list[str]:
    """Return a normalized, deduplicated list from a comma separated value."""

    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        tokens = [str(item).strip() for item in value if str(item).strip()]
    else:
        tokens = [segment.strip() for segment in str(value).replace(";", ",").split(",")]
    deduped: dict[str, None] = {}
    for token in tokens:
        if token:
            deduped.setdefault(token, None)
    return list(deduped.keys())


def mask(value: object | None) -> str | None:
    """Return a masked representation for secret configuration entries."""

    if value is None:
        return None
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
    force_non_stream: bool = False
    keywords_first: bool = True
    disable_hashtags: bool = False
    target_lang: str = "en"
    json_prompt: str | None = None
    json_mode: bool = False
    json_transcript_limit: int | None = None

    @property
    def effective_json_model(self) -> str:
        return self.model_json or self.model

    @property
    def effective_text_model(self) -> str:
        return self.model_text or self.model


@dataclass(slots=True)
class BrollSettings:
    min_start_s: float = 2.0
    min_gap_s: float = 1.5
    no_repeat_s: float = 6.0


@dataclass(slots=True)
class FetchSettings:
    max_per_keyword: int = 8
    allow_images: bool = True
    allow_videos: bool = True
    providers: tuple[str, ...] = ("pixabay",)
    provider_limits: dict[str, int] = field(default_factory=dict)
    timeout_s: float = 8.0
    api_keys: dict[str, str | None] = field(default_factory=dict)


@dataclass(slots=True)
class LogSettings:
    logger_name: str = "video_pipeline.config"
    startup_label: str = "[CONFIG]"
    masked_env_keys: tuple[str, ...] = (
        "PEXELS_API_KEY",
        "PIXABAY_API_KEY",
        "UNSPLASH_ACCESS_KEY",
    )


@dataclass(slots=True)
class Settings:
    clips_dir: Path
    output_dir: Path
    temp_dir: Path
    llm: LLMSettings
    broll: BrollSettings
    fetch: FetchSettings
    max_segments_in_flight: int = 1
    llm_max_queries_per_segment: int = 3
    tfidf_fallback_disabled: bool = False
    fast_tests: bool = False
    log: LogSettings = field(default_factory=LogSettings)

    def to_log_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "paths": {
                "clips_dir": str(self.clips_dir),
                "output_dir": str(self.output_dir),
                "temp_dir": str(self.temp_dir),
            },
            "pipeline": {
                "max_segments_in_flight": self.max_segments_in_flight,
                "llm_max_queries_per_segment": self.llm_max_queries_per_segment,
                "tfidf_fallback_disabled": self.tfidf_fallback_disabled,
                "fast_tests": self.fast_tests,
            },
            "llm": {
                "model": self.llm.model,
                "model_json": self.llm.effective_json_model,
                "model_text": self.llm.effective_text_model,
                "endpoint": self.llm.endpoint,
                "keep_alive": self.llm.keep_alive,
                "timeout_stream_s": self.llm.timeout_stream_s,
                "timeout_fallback_s": self.llm.timeout_fallback_s,
                "min_chars": self.llm.min_chars,
                "max_attempts": self.llm.max_attempts,
            },
            "broll": {
                "min_start_s": self.broll.min_start_s,
                "min_gap_s": self.broll.min_gap_s,
                "no_repeat_s": self.broll.no_repeat_s,
            },
            "fetch": {
                "max_per_keyword": self.fetch.max_per_keyword,
                "allow_images": self.fetch.allow_images,
                "allow_videos": self.fetch.allow_videos,
                "providers": list(self.fetch.providers),
                "provider_limits": dict(self.fetch.provider_limits),
                "timeout_s": self.fetch.timeout_s,
            },
        }

        masked_keys: dict[str, str | None] = {}
        for key in self.log.masked_env_keys:
            masked_keys[key] = mask(self.fetch.api_keys.get(key))
        if masked_keys:
            payload["fetch"]["api_keys"] = masked_keys

        return payload


def _get_env(source: Mapping[str, str] | None, key: str, default: str | None = None) -> str | None:
    if source is None:
        if default is None:
            return os.getenv(key)
        return os.getenv(key, default)
    return source.get(key, default)


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Load :class:`Settings` from the provided mapping or ``os.environ``."""

    clips_dir = Path(_get_env(env, "PIPELINE_CLIPS_DIR", "clips") or "clips")
    output_dir = Path(_get_env(env, "PIPELINE_OUTPUT_DIR", "output") or "output")
    temp_dir = Path(_get_env(env, "PIPELINE_TEMP_DIR", "temp") or "temp")

    llm_model = _clean(_get_env(env, "PIPELINE_LLM_MODEL", "qwen2.5:7b") or "qwen2.5:7b")
    llm_model_json = _clean(_get_env(env, "PIPELINE_LLM_MODEL_JSON", "") or "")
    llm_model_text = _clean(_get_env(env, "PIPELINE_LLM_MODEL_TEXT", "") or "")
    llm_endpoint = _clean(
        _get_env(env, "PIPELINE_LLM_ENDPOINT")
        or _get_env(env, "PIPELINE_LLM_BASE_URL")
        or _get_env(env, "OLLAMA_HOST")
        or "http://localhost:11434"
    )
    llm_keep_alive = _clean(_get_env(env, "PIPELINE_LLM_KEEP_ALIVE", "30m") or "30m")
    llm_timeout_stream = to_float(
        _get_env(env, "PIPELINE_LLM_TIMEOUT_S", "60"),
        60.0,
        name="PIPELINE_LLM_TIMEOUT_S",
        minimum=0.1,
    )
    llm_timeout_fallback = to_float(
        _get_env(env, "PIPELINE_LLM_FALLBACK_TIMEOUT_S", "45"),
        45.0,
        name="PIPELINE_LLM_FALLBACK_TIMEOUT_S",
        minimum=0.1,
    )
    llm_min_chars = to_int(
        _get_env(env, "PIPELINE_LLM_MIN_CHARS", "8"),
        8,
        name="PIPELINE_LLM_MIN_CHARS",
        minimum=0,
    )
    llm_max_attempts = to_int(
        _get_env(env, "PIPELINE_LLM_MAX_ATTEMPTS", "3"),
        3,
        name="PIPELINE_LLM_MAX_ATTEMPTS",
        minimum=1,
    )
    llm_num_predict = to_int(
        _get_env(env, "PIPELINE_LLM_NUM_PREDICT", "256"),
        256,
        name="PIPELINE_LLM_NUM_PREDICT",
        minimum=1,
    )
    llm_temperature = to_float(
        _get_env(env, "PIPELINE_LLM_TEMP", "0.3"),
        0.3,
        name="PIPELINE_LLM_TEMP",
        minimum=0.0,
    )
    llm_top_p = to_float(
        _get_env(env, "PIPELINE_LLM_TOP_P", "0.9"),
        0.9,
        name="PIPELINE_LLM_TOP_P",
        minimum=0.0,
    )
    llm_repeat_penalty = to_float(
        _get_env(env, "PIPELINE_LLM_REPEAT_PENALTY", "1.1"),
        1.1,
        name="PIPELINE_LLM_REPEAT_PENALTY",
        minimum=0.0,
    )
    llm_num_ctx = to_int(
        _get_env(env, "PIPELINE_LLM_NUM_CTX", "4096"),
        4096,
        name="PIPELINE_LLM_NUM_CTX",
        minimum=1,
    )
    llm_fallback_trunc = to_int(
        _get_env(env, "PIPELINE_LLM_FALLBACK_TRUNC", "3500"),
        3500,
        name="PIPELINE_LLM_FALLBACK_TRUNC",
        minimum=1,
    )
    llm_force_non_stream = to_bool(
        _get_env(env, "PIPELINE_LLM_FORCE_NON_STREAM"),
        False,
        name="PIPELINE_LLM_FORCE_NON_STREAM",
    )
    llm_keywords_first = to_bool(
        _get_env(env, "PIPELINE_LLM_KEYWORDS_FIRST"),
        True,
        name="PIPELINE_LLM_KEYWORDS_FIRST",
    )
    llm_disable_hashtags = to_bool(
        _get_env(env, "PIPELINE_LLM_DISABLE_HASHTAGS"),
        False,
        name="PIPELINE_LLM_DISABLE_HASHTAGS",
    )
    llm_target_lang = _clean(_get_env(env, "PIPELINE_LLM_TARGET_LANG", "en") or "en")
    llm_json_prompt = _clean(_get_env(env, "PIPELINE_LLM_JSON_PROMPT") or "") or None
    llm_json_mode = to_bool(
        _get_env(env, "PIPELINE_LLM_JSON_MODE"), False, name="PIPELINE_LLM_JSON_MODE"
    )
    llm_json_transcript_limit_raw = _clean(
        _get_env(env, "PIPELINE_LLM_JSON_TRANSCRIPT_LIMIT") or ""
    )
    llm_json_transcript_limit = None
    if llm_json_transcript_limit_raw:
        llm_json_transcript_limit = to_int(
            llm_json_transcript_limit_raw,
            0,
            name="PIPELINE_LLM_JSON_TRANSCRIPT_LIMIT",
            minimum=0,
        )

    tfidf_disabled_source = _get_env(env, "PIPELINE_TFIDF_FALLBACK_DISABLED")
    if tfidf_disabled_source is None:
        tfidf_disabled_source = _get_env(env, "PIPELINE_DISABLE_TFIDF_FALLBACK")
    tfidf_fallback_disabled = to_bool(
        tfidf_disabled_source,
        False,
        name="PIPELINE_TFIDF_FALLBACK_DISABLED",
    )

    llm_max_queries_per_segment = to_int(
        _get_env(env, "PIPELINE_LLM_MAX_QUERIES_PER_SEGMENT", "3"),
        3,
        name="PIPELINE_LLM_MAX_QUERIES_PER_SEGMENT",
        minimum=1,
    )

    fast_tests = to_bool(
        _get_env(env, "PIPELINE_FAST_TESTS"), False, name="PIPELINE_FAST_TESTS"
    )

    llm_settings = LLMSettings(
        model=llm_model,
        model_json=llm_model_json,
        model_text=llm_model_text,
        endpoint=llm_endpoint,
        base_url=llm_endpoint,
        keep_alive=llm_keep_alive,
        timeout_stream_s=llm_timeout_stream,
        timeout_fallback_s=llm_timeout_fallback,
        min_chars=llm_min_chars,
        max_attempts=llm_max_attempts,
        num_predict=llm_num_predict,
        temperature=llm_temperature,
        top_p=llm_top_p,
        repeat_penalty=llm_repeat_penalty,
        num_ctx=llm_num_ctx,
        fallback_trunc=llm_fallback_trunc,
        force_non_stream=llm_force_non_stream,
        keywords_first=llm_keywords_first,
        disable_hashtags=llm_disable_hashtags,
        target_lang=llm_target_lang,
        json_prompt=llm_json_prompt,
        json_mode=llm_json_mode,
        json_transcript_limit=llm_json_transcript_limit,
    )

    broll_settings = BrollSettings(
        min_start_s=to_float(
            _get_env(env, "PIPELINE_BROLL_MIN_START_SECONDS", "2.0"),
            2.0,
            name="PIPELINE_BROLL_MIN_START_SECONDS",
            minimum=0.0,
        ),
        min_gap_s=to_float(
            _get_env(env, "PIPELINE_BROLL_MIN_GAP_SECONDS", "1.5"),
            1.5,
            name="PIPELINE_BROLL_MIN_GAP_SECONDS",
            minimum=0.0,
        ),
        no_repeat_s=to_float(
            _get_env(env, "PIPELINE_BROLL_NO_REPEAT_SECONDS", "6.0"),
            6.0,
            name="PIPELINE_BROLL_NO_REPEAT_SECONDS",
            minimum=0.0,
        ),
    )

    provider_values = csv_list(
        _get_env(env, "BROLL_FETCH_PROVIDER")
        or _get_env(env, "AI_BROLL_FETCH_PROVIDER")
    )
    if not provider_values:
        provider_values = ["pixabay"]

    provider_limits: dict[str, int] = {}
    for provider in provider_values:
        env_key = f"BROLL_{provider.upper()}_MAX_PER_KEYWORD"
        limit = _get_env(env, env_key)
        if limit is not None:
            provider_limits[provider.lower()] = to_int(
                limit,
                1,
                name=env_key,
                minimum=1,
            )

    fetch_timeout = to_float(
        _get_env(env, "PIPELINE_FETCH_TIMEOUT_S", "8"),
        8.0,
        name="PIPELINE_FETCH_TIMEOUT_S",
        minimum=0.1,
    )
    fetch_max_default = to_int(
        _get_env(env, "FETCH_MAX", "8"),
        8,
        name="FETCH_MAX",
        minimum=1,
    )
    fetch_max_per_keyword = to_int(
        _get_env(env, "BROLL_FETCH_MAX_PER_KEYWORD"),
        fetch_max_default,
        name="BROLL_FETCH_MAX_PER_KEYWORD",
        minimum=1,
    )

    fetch_settings = FetchSettings(
        max_per_keyword=fetch_max_per_keyword,
        allow_images=to_bool(
            _get_env(env, "BROLL_FETCH_ALLOW_IMAGES"),
            True,
            name="BROLL_FETCH_ALLOW_IMAGES",
        ),
        allow_videos=to_bool(
            _get_env(env, "BROLL_FETCH_ALLOW_VIDEOS"),
            True,
            name="BROLL_FETCH_ALLOW_VIDEOS",
        ),
        providers=tuple(p.lower() for p in provider_values),
        provider_limits=provider_limits,
        timeout_s=fetch_timeout,
        api_keys={
            "PEXELS_API_KEY": _get_env(env, "PEXELS_API_KEY"),
            "PIXABAY_API_KEY": _get_env(env, "PIXABAY_API_KEY"),
            "UNSPLASH_ACCESS_KEY": _get_env(env, "UNSPLASH_ACCESS_KEY"),
        },
    )

    log_settings = LogSettings()

    max_segments_in_flight = to_int(
        _get_env(env, "PIPELINE_MAX_SEGMENTS_IN_FLIGHT", "1"),
        1,
        name="PIPELINE_MAX_SEGMENTS_IN_FLIGHT",
        minimum=1,
    )

    settings = Settings(
        clips_dir=clips_dir,
        output_dir=output_dir,
        temp_dir=temp_dir,
        llm=llm_settings,
        broll=broll_settings,
        fetch=fetch_settings,
        max_segments_in_flight=max_segments_in_flight,
        llm_max_queries_per_segment=llm_max_queries_per_segment,
        tfidf_fallback_disabled=tfidf_fallback_disabled,
        fast_tests=fast_tests,
        log=log_settings,
    )

    return settings


def set_settings(settings: Settings) -> None:
    """Cache *settings* for global access."""

    global _SETTINGS_CACHE
    _SETTINGS_CACHE = settings


def get_settings() -> Settings:
    """Return the cached :class:`Settings` (loading them if needed)."""

    global _SETTINGS_CACHE
    if _SETTINGS_CACHE is None:
        _SETTINGS_CACHE = load_settings()
    return _SETTINGS_CACHE


def reset_settings_cache_for_tests() -> None:  # pragma: no cover - testing helper
    global _SETTINGS_CACHE
    _SETTINGS_CACHE = None


def reset_startup_log_for_tests() -> None:  # pragma: no cover - testing helper
    global _STARTUP_LOG_EMITTED
    _STARTUP_LOG_EMITTED = False


def log_effective_settings(settings: Settings, logger: logging.Logger | None = None) -> None:
    """Emit a single startup log describing the effective configuration."""

    global _STARTUP_LOG_EMITTED
    if _STARTUP_LOG_EMITTED:
        return

    logger = logger or logging.getLogger(settings.log.logger_name)
    payload = settings.to_log_payload()
    message = f"{settings.log.startup_label} effective={json.dumps(payload, sort_keys=True)}"
    logger.info(message)
    _STARTUP_LOG_EMITTED = True
