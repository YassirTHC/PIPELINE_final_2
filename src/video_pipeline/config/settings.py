"""Strongly typed configuration loader for the video pipeline."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import Dict, List, Mapping, Optional


_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}

_SETTINGS_CACHE: Optional["Settings"] = None
_CACHE_LOCK = Lock()
_STARTUP_LOG_EMITTED = False


def _clean_text(value: Optional[str]) -> str:
    return value.strip() if value else ""


def _coerce_bool(value: object | None, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = _clean_text(str(value)).lower()
    if not text:
        return bool(default)
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    return bool(default)


def _coerce_float(
    value: object | None,
    default: float,
    *,
    minimum: float | None = None,
) -> float:
    if value is None:
        result = float(default)
    else:
        try:
            result = float(str(value).strip())
        except (TypeError, ValueError):
            result = float(default)
    if minimum is not None and result < minimum:
        result = float(minimum)
    return result


def _coerce_int(
    value: object | None,
    default: int,
    *,
    minimum: int | None = None,
) -> int:
    if value is None:
        result = int(default)
    else:
        try:
            result = int(str(value).strip())
        except (TypeError, ValueError):
            result = int(default)
    if minimum is not None and result < minimum:
        result = int(minimum)
    return result


def _split_csv(value: object | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value]
    else:
        text = str(value).replace(";", ",")
        items = [chunk.strip() for chunk in text.split(",")]
    cleaned: List[str] = []
    seen = set()
    for item in items:
        if not item:
            continue
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned.append(lowered)
    return cleaned


def _mask_secret(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    trimmed = value.strip()
    if not trimmed:
        return None
    suffix = trimmed[-4:] if len(trimmed) >= 4 else trimmed
    return f"****{suffix}"


def _env(source: Optional[Mapping[str, str]], key: str, default: Optional[str] = None) -> Optional[str]:
    if source is None:
        return os.getenv(key, default) if default is not None else os.getenv(key)
    return source.get(key, default)


@dataclass(slots=True)
class LLMSettings:
    model: str
    model_json: str
    model_text: str
    provider: str
    endpoint: str
    base_url: str
    keep_alive: str
    timeout_stream_s: float
    timeout_fallback_s: float
    min_chars: int
    max_attempts: int
    num_predict: int
    temperature: float
    top_p: float
    repeat_penalty: float
    num_ctx: int
    fallback_trunc: int
    force_non_stream: bool
    keywords_first: bool = True
    disable_hashtags: bool = False
    target_lang: str = "en"
    json_prompt: Optional[str] = None
    json_mode: bool = False
    json_transcript_limit: Optional[int] = None
    disable_dynamic_segment: bool = False
    request_cooldown_s: float = 0.0
    request_cooldown_jitter_s: float = 0.0

    @property
    def effective_json_model(self) -> str:
        return self.model_json or self.model

    @property
    def effective_text_model(self) -> str:
        return self.model_text or self.model


@dataclass(slots=True)
class FetchSettings:
    timeout_s: float
    max_per_keyword: int
    allow_images: bool
    allow_videos: bool
    providers: List[str]
    provider_limits: Dict[str, int]
    api_keys: Dict[str, Optional[str]]


@dataclass(slots=True)
class BrollSettings:
    min_start_s: float = 0.5
    min_gap_s: float = 0.5
    no_repeat_s: float = 4.0


@dataclass(slots=True)
class SubtitleSettings:
    font_path: Optional[str]
    font: Optional[str] = None
    font_size: int = 96
    subtitle_safe_margin_px: int = 220
    keyword_background: bool = False
    stroke_px: int = 6
    shadow_opacity: float = 0.35
    shadow_offset: int = 3
    enable_emojis: bool = True
    emoji_target_per_10: int = 5
    emoji_min_gap_groups: int = 2
    emoji_max_per_segment: int = 3
    emoji_no_context_fallback: str = ""
    hero_emoji_enable: bool = True
    hero_emoji_max_per_segment: int = 1


@dataclass(slots=True)
class Settings:
    clips_dir: Path
    output_dir: Path
    temp_dir: Path
    tfidf_fallback_disabled: bool
    llm_max_queries_per_segment: int
    fast_tests: bool
    max_segments_in_flight: int
    llm: LLMSettings
    fetch: FetchSettings
    broll: BrollSettings
    subtitles: SubtitleSettings

    def to_log_payload(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "paths": {
                "clips_dir": str(self.clips_dir),
                "output_dir": str(self.output_dir),
                "temp_dir": str(self.temp_dir),
            },
            "llm": {
                "model": self.llm.model,
                "model_json": self.llm.effective_json_model,
                "model_text": self.llm.effective_text_model,
                "provider": self.llm.provider,
                "endpoint": self.llm.endpoint,
                "keep_alive": self.llm.keep_alive,
                "timeout_stream_s": self.llm.timeout_stream_s,
                "timeout_fallback_s": self.llm.timeout_fallback_s,
                "min_chars": self.llm.min_chars,
                "max_attempts": self.llm.max_attempts,
                "num_predict": self.llm.num_predict,
                "temperature": self.llm.temperature,
                "top_p": self.llm.top_p,
                "repeat_penalty": self.llm.repeat_penalty,
                "num_ctx": self.llm.num_ctx,
                "fallback_trunc": self.llm.fallback_trunc,
                "force_non_stream": self.llm.force_non_stream,
                "keywords_first": self.llm.keywords_first,
                "disable_hashtags": self.llm.disable_hashtags,
                "target_lang": self.llm.target_lang,
                "json_mode": self.llm.json_mode,
                "json_transcript_limit": self.llm.json_transcript_limit,
                "disable_dynamic_segment": self.llm.disable_dynamic_segment,
                "request_cooldown_s": self.llm.request_cooldown_s,
                "request_cooldown_jitter_s": self.llm.request_cooldown_jitter_s,
            },
            "fetch": {
                "timeout_s": self.fetch.timeout_s,
                "max_per_keyword": self.fetch.max_per_keyword,
                "allow_images": self.fetch.allow_images,
                "allow_videos": self.fetch.allow_videos,
                "providers": list(self.fetch.providers),
                "provider_limits": dict(self.fetch.provider_limits),
            },
            "broll": {
                "min_start_s": self.broll.min_start_s,
                "min_gap_s": self.broll.min_gap_s,
                "no_repeat_s": self.broll.no_repeat_s,
            },
            "subtitles": {
                "font_path": self.subtitles.font_path,
                "font": self.subtitles.font,
                "font_size": self.subtitles.font_size,
                "subtitle_safe_margin_px": self.subtitles.subtitle_safe_margin_px,
                "keyword_background": self.subtitles.keyword_background,
                "enable_emojis": self.subtitles.enable_emojis,
                "stroke_px": self.subtitles.stroke_px,
                "shadow_opacity": self.subtitles.shadow_opacity,
                "shadow_offset": self.subtitles.shadow_offset,
                "emoji_target_per_10": self.subtitles.emoji_target_per_10,
                "emoji_min_gap_groups": self.subtitles.emoji_min_gap_groups,
                "emoji_max_per_segment": self.subtitles.emoji_max_per_segment,
                "emoji_no_context_fallback": self.subtitles.emoji_no_context_fallback,
                "hero_emoji_enable": self.subtitles.hero_emoji_enable,
                "hero_emoji_max_per_segment": self.subtitles.hero_emoji_max_per_segment,
            },
            "flags": {
                "tfidf_fallback_disabled": self.tfidf_fallback_disabled,
                "llm_max_queries_per_segment": self.llm_max_queries_per_segment,
                "fast_tests": self.fast_tests,
                "max_segments_in_flight": self.max_segments_in_flight,
            },
        }

        masked: Dict[str, Optional[str]] = {}
        for key, raw_value in self.fetch.api_keys.items():
            masked[key] = _mask_secret(raw_value)
        if masked:
            payload["fetch"]["api_keys"] = masked

        return payload


def _resolve_bool_env(
    source: Optional[Mapping[str, str]],
    *keys: str,
    default: bool = False,
) -> bool:
    for key in keys:
        if not key:
            continue
        candidate = _env(source, key)
        if candidate is not None:
            return _coerce_bool(candidate, default=default)
    return bool(default)


def _resolve_optional_int(
    source: Optional[Mapping[str, str]],
    key: str,
    *,
    minimum: int | None = None,
) -> Optional[int]:
    value = _env(source, key)
    if value is None:
        return None
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    if minimum is not None and parsed < minimum:
        parsed = minimum
    return parsed


def _tfidf_disabled(source: Optional[Mapping[str, str]]) -> bool:
    flag = _resolve_bool_env(
        source,
        "PIPELINE_TFIDF_FALLBACK_DISABLED",
        "PIPELINE_DISABLE_TFIDF_FALLBACK",
        default=False,
    )
    return bool(flag)


def _llm_settings(env: Optional[Mapping[str, str]]) -> LLMSettings:
    model = _clean_text(_env(env, "PIPELINE_LLM_MODEL", "qwen2.5:7b") or "qwen2.5:7b")
    model_json = _clean_text(_env(env, "PIPELINE_LLM_MODEL_JSON", "") or "")
    model_text = _clean_text(_env(env, "PIPELINE_LLM_MODEL_TEXT", "") or "")
    provider = _clean_text(_env(env, "PIPELINE_LLM_PROVIDER", "ollama") or "ollama")
    endpoint = _clean_text(
        _env(env, "PIPELINE_LLM_ENDPOINT")
        or _env(env, "PIPELINE_LLM_BASE_URL")
        or "http://localhost:11434"
    )
    keep_alive = _clean_text(_env(env, "PIPELINE_LLM_KEEP_ALIVE", "30m") or "30m")
    timeout_stream_s = _coerce_float(
        _env(env, "PIPELINE_LLM_TIMEOUT_S", "60"),
        60.0,
        minimum=0.0,
    )
    timeout_fallback_s = _coerce_float(
        _env(env, "PIPELINE_LLM_FALLBACK_TIMEOUT_S", "45"),
        45.0,
        minimum=0.0,
    )
    min_chars = _coerce_int(
        _env(env, "PIPELINE_LLM_MIN_CHARS", "8"),
        8,
        minimum=0,
    )
    max_attempts = _coerce_int(
        _env(env, "PIPELINE_LLM_MAX_ATTEMPTS", "3"),
        3,
        minimum=1,
    )
    num_predict = _coerce_int(
        _env(env, "PIPELINE_LLM_NUM_PREDICT", "256"),
        256,
        minimum=1,
    )
    temperature = _coerce_float(
        _env(env, "PIPELINE_LLM_TEMP", "0.3"),
        0.3,
        minimum=0.0,
    )
    top_p = _coerce_float(
        _env(env, "PIPELINE_LLM_TOP_P", "0.9"),
        0.9,
        minimum=0.0,
    )
    repeat_penalty = _coerce_float(
        _env(env, "PIPELINE_LLM_REPEAT_PENALTY", "1.1"),
        1.1,
        minimum=0.0,
    )
    num_ctx = _coerce_int(
        _env(env, "PIPELINE_LLM_NUM_CTX", "4096"),
        4096,
        minimum=1,
    )
    fallback_trunc = _coerce_int(
        _env(env, "PIPELINE_LLM_FALLBACK_TRUNC", "3500"),
        3500,
        minimum=0,
    )
    force_non_stream = _resolve_bool_env(
        env,
        "PIPELINE_LLM_FORCE_NON_STREAM",
        default=False,
    )
    keywords_first = _resolve_bool_env(
        env,
        "PIPELINE_LLM_KEYWORDS_FIRST",
        default=True,
    )
    disable_hashtags = _resolve_bool_env(
        env,
        "PIPELINE_LLM_DISABLE_HASHTAGS",
        default=False,
    )
    target_lang = _clean_text(_env(env, "PIPELINE_LLM_TARGET_LANG", "en") or "en")
    json_prompt = _clean_text(_env(env, "PIPELINE_LLM_JSON_PROMPT") or "") or None
    json_mode = _resolve_bool_env(
        env,
        "PIPELINE_LLM_JSON_MODE",
        default=False,
    )
    json_transcript_limit_raw = _env(env, "PIPELINE_LLM_JSON_TRANSCRIPT_LIMIT")
    json_transcript_limit: Optional[int]
    if json_transcript_limit_raw:
        json_transcript_limit = _coerce_int(
            json_transcript_limit_raw,
            0,
            minimum=0,
        )
    else:
        json_transcript_limit = None

    disable_dynamic_segment = _resolve_bool_env(
        env,
        "PIPELINE_DISABLE_DYNAMIC_SEGMENT_LLM",
        default=False,
    )

    request_cooldown_s = _coerce_float(
        _env(env, "PIPELINE_LLM_REQUEST_COOLDOWN_S", "0"),
        0.0,
        minimum=0.0,
    )
    request_cooldown_jitter_s = _coerce_float(
        _env(env, "PIPELINE_LLM_REQUEST_COOLDOWN_JITTER_S", "0"),
        0.0,
        minimum=0.0,
    )

    return LLMSettings(
        model=model,
        model_json=model_json,
        model_text=model_text,
        provider=provider,
        endpoint=endpoint,
        base_url=endpoint,
        keep_alive=keep_alive,
        timeout_stream_s=timeout_stream_s,
        timeout_fallback_s=timeout_fallback_s,
        min_chars=min_chars,
        max_attempts=max_attempts,
        num_predict=num_predict,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        num_ctx=num_ctx,
        fallback_trunc=fallback_trunc,
        force_non_stream=force_non_stream,
        keywords_first=keywords_first,
        disable_hashtags=disable_hashtags,
        target_lang=target_lang,
        json_prompt=json_prompt,
        json_mode=json_mode,
        json_transcript_limit=json_transcript_limit,
        disable_dynamic_segment=disable_dynamic_segment,
        request_cooldown_s=request_cooldown_s,
        request_cooldown_jitter_s=request_cooldown_jitter_s,
    )


def _broll_settings(env: Optional[Mapping[str, str]]) -> BrollSettings:
    return BrollSettings(
        min_start_s=_coerce_float(
            _env(env, "PIPELINE_BROLL_MIN_START_SECONDS", "1.0"),
            1.0,
            minimum=0.0,
        ),
        min_gap_s=_coerce_float(
            _env(env, "PIPELINE_BROLL_MIN_GAP_S", "0.5"),
            0.5,
            minimum=0.0,
        ),
        no_repeat_s=_coerce_float(
            _env(env, "PIPELINE_BROLL_NO_REPEAT_SECONDS", "6.0"),
            6.0,
            minimum=0.0,
        ),
    )


def _subtitle_settings(env: Optional[Mapping[str, str]]) -> SubtitleSettings:
    repo_root = Path(__file__).resolve().parents[2]

    override = _env(env, "PIPELINE_SUB_FONT_PATH")
    if not override:
        override = _env(env, "PIPELINE_SUBTITLE_FONT_PATH")

    candidate_paths = []
    if override:
        candidate_paths.append(Path(str(override)).expanduser())

    assets_dir = repo_root / "assets" / "fonts"
    candidate_paths.extend(
        [
            assets_dir / "Montserrat-ExtraBold.ttf",
            assets_dir / "Montserrat-Bold.ttf",
        ]
    )

    windir = os.getenv("WINDIR")
    windows_fonts: List[Path] = []
    if windir:
        base_fonts = Path(windir) / "Fonts"
        windows_fonts.extend(
            [
                base_fonts / "Montserrat-ExtraBold.ttf",
                base_fonts / "Montserrat-Bold.ttf",
            ]
        )

    candidate_paths.extend(
        [
            Path("/System/Library/Fonts/Montserrat-ExtraBold.ttf"),
            Path("/System/Library/Fonts/Montserrat-Bold.ttf"),
            Path("/Library/Fonts/Montserrat-ExtraBold.ttf"),
            Path("/Library/Fonts/Montserrat-Bold.ttf"),
            *windows_fonts,
        ]
    )

    resolved_font: Optional[str] = None
    for candidate in candidate_paths:
        try:
            if candidate and candidate.exists():
                resolved_font = str(candidate.resolve())
                break
        except OSError:
            continue

    def _env_preferred(*keys: str) -> Optional[str]:
        for key in keys:
            value = _env(env, key)
            if value is not None:
                return value
        return None

    font_name = _env_preferred("PIPELINE_SUBTITLE_FONT", "PIPELINE_SUB_FONT")

    font_size = _coerce_int(
        _env_preferred("PIPELINE_SUBTITLE_FONT_SIZE", "PIPELINE_SUB_FONT_SIZE"),
        96,
        minimum=12,
    )
    safe_margin = _coerce_int(
        _env_preferred("PIPELINE_SUBTITLE_SAFE_MARGIN_PX", "PIPELINE_SUB_SAFE_MARGIN_PX"),
        220,
        minimum=0,
    )
    keyword_background = _resolve_bool_env(
        env,
        "PIPELINE_SUBTITLE_KEYWORD_BACKGROUND",
        "PIPELINE_SUB_KEYWORD_BACKGROUND",
        default=False,
    )
    enable_emojis = _resolve_bool_env(
        env,
        "PIPELINE_SUBTITLE_ENABLE_EMOJIS",
        "PIPELINE_SUB_ENABLE_EMOJIS",
        default=True,
    )
    stroke_px = _coerce_int(
        _env_preferred("PIPELINE_SUBTITLE_STROKE_PX", "PIPELINE_SUB_STROKE_PX"),
        6,
        minimum=0,
    )
    shadow_opacity = _coerce_float(
        _env_preferred("PIPELINE_SUBTITLE_SHADOW_OPACITY", "PIPELINE_SUB_SHADOW_OPACITY"),
        0.35,
        minimum=0.0,
    )
    shadow_offset = _coerce_int(
        _env_preferred("PIPELINE_SUBTITLE_SHADOW_OFFSET", "PIPELINE_SUB_SHADOW_OFFSET"),
        3,
        minimum=0,
    )
    emoji_target = _coerce_int(
        _env_preferred("PIPELINE_SUBTITLE_EMOJI_TARGET_PER_10", "PIPELINE_SUB_EMOJI_TARGET_PER_10"),
        5,
        minimum=0,
    )
    emoji_min_gap = _coerce_int(
        _env_preferred("PIPELINE_SUBTITLE_EMOJI_MIN_GAP_GROUPS", "PIPELINE_SUB_EMOJI_MIN_GAP_GROUPS"),
        2,
        minimum=0,
    )
    emoji_max_segment = _coerce_int(
        _env_preferred("PIPELINE_SUBTITLE_EMOJI_MAX_PER_SEGMENT", "PIPELINE_SUB_EMOJI_MAX_PER_SEGMENT"),
        3,
        minimum=0,
    )
    emoji_fallback = _env_preferred(
        "PIPELINE_SUBTITLE_EMOJI_NO_CONTEXT_FALLBACK",
        "PIPELINE_SUB_EMOJI_NO_CONTEXT_FALLBACK",
    ) or ""
    hero_enable = _resolve_bool_env(
        env,
        "PIPELINE_SUBTITLE_HERO_EMOJI_ENABLE",
        "PIPELINE_SUB_HERO_EMOJI_ENABLE",
        default=True,
    )
    hero_max = _coerce_int(
        _env_preferred("PIPELINE_SUBTITLE_HERO_EMOJI_MAX_PER_SEGMENT", "PIPELINE_SUB_HERO_EMOJI_MAX_PER_SEGMENT"),
        1,
        minimum=0,
    )

    return SubtitleSettings(
        font_path=resolved_font,
        font=font_name,
        font_size=font_size,
        subtitle_safe_margin_px=safe_margin,
        keyword_background=keyword_background,
        stroke_px=stroke_px,
        shadow_opacity=shadow_opacity,
        shadow_offset=shadow_offset,
        enable_emojis=enable_emojis,
        emoji_target_per_10=emoji_target,
        emoji_min_gap_groups=emoji_min_gap,
        emoji_max_per_segment=emoji_max_segment,
        emoji_no_context_fallback=emoji_fallback,
        hero_emoji_enable=hero_enable,
        hero_emoji_max_per_segment=hero_max,
    )


def _fetch_settings(env: Optional[Mapping[str, str]]) -> FetchSettings:
    providers = _split_csv(
        _env(env, "BROLL_FETCH_PROVIDER")
        or _env(env, "AI_BROLL_FETCH_PROVIDER")
    )
    if not providers:
        providers = ["pixabay"]

    provider_limits: Dict[str, int] = {}
    for provider in providers:
        key = f"BROLL_{provider.upper()}_MAX_PER_KEYWORD"
        limit = _resolve_optional_int(env, key, minimum=1)
        if limit is not None:
            provider_limits[provider] = limit

    timeout_s = _coerce_float(
        _env(env, "PIPELINE_FETCH_TIMEOUT_S", "8"),
        8.0,
        minimum=0.0,
    )
    default_max = _coerce_int(
        _env(env, "FETCH_MAX", "8"),
        8,
        minimum=1,
    )
    max_per_keyword = _coerce_int(
        _env(env, "BROLL_FETCH_MAX_PER_KEYWORD"),
        default_max,
        minimum=1,
    )
    allow_images = _resolve_bool_env(env, "BROLL_FETCH_ALLOW_IMAGES", default=True)
    allow_videos = _resolve_bool_env(env, "BROLL_FETCH_ALLOW_VIDEOS", default=True)

    api_keys: Dict[str, Optional[str]] = {
        "PEXELS_API_KEY": _env(env, "PEXELS_API_KEY"),
        "PIXABAY_API_KEY": _env(env, "PIXABAY_API_KEY"),
        "UNSPLASH_ACCESS_KEY": _env(env, "UNSPLASH_ACCESS_KEY"),
    }

    return FetchSettings(
        timeout_s=timeout_s,
        max_per_keyword=max_per_keyword,
        allow_images=allow_images,
        allow_videos=allow_videos,
        providers=providers,
        provider_limits=provider_limits,
        api_keys=api_keys,
    )


def load_settings(env: Optional[Mapping[str, str]] = None) -> Settings:
    clips_dir = Path(_env(env, "PIPELINE_CLIPS_DIR", "clips") or "clips")
    output_dir = Path(_env(env, "PIPELINE_OUTPUT_DIR", "output") or "output")
    temp_dir = Path(_env(env, "PIPELINE_TEMP_DIR", "temp") or "temp")

    llm = _llm_settings(env)
    fetch = _fetch_settings(env)
    broll = _broll_settings(env)
    subtitles = _subtitle_settings(env)

    tfidf_disabled = _tfidf_disabled(env)
    llm_queries = _coerce_int(
        _env(env, "PIPELINE_LLM_MAX_QUERIES_PER_SEGMENT"),
        3,
        minimum=1,
    )
    fast_tests = _resolve_bool_env(env, "PIPELINE_FAST_TESTS", default=False)
    max_segments_in_flight = _coerce_int(
        _env(env, "PIPELINE_MAX_SEGMENTS_IN_FLIGHT"),
        1,
        minimum=1,
    )

    return Settings(
        clips_dir=clips_dir,
        output_dir=output_dir,
        temp_dir=temp_dir,
        tfidf_fallback_disabled=tfidf_disabled,
        llm_max_queries_per_segment=llm_queries,
        fast_tests=fast_tests,
        max_segments_in_flight=max_segments_in_flight,
        llm=llm,
        fetch=fetch,
        broll=broll,
        subtitles=subtitles,
    )


def set_settings(settings: Settings) -> None:
    global _SETTINGS_CACHE
    with _CACHE_LOCK:
        _SETTINGS_CACHE = settings


def apply_llm_overrides(
    settings: Settings,
    *,
    provider: Optional[str] = None,
    model_text: Optional[str] = None,
    model_json: Optional[str] = None,
) -> Settings:
    updates: Dict[str, str] = {}
    if provider is not None:
        cleaned = provider.strip()
        if cleaned:
            updates["provider"] = cleaned
    if model_text is not None:
        cleaned = model_text.strip()
        if cleaned:
            updates["model_text"] = cleaned
    if model_json is not None:
        cleaned = model_json.strip()
        if cleaned:
            updates["model_json"] = cleaned
    if not updates:
        return settings
    new_llm = replace(settings.llm, **updates)
    return replace(settings, llm=new_llm)


def get_settings() -> Settings:
    global _SETTINGS_CACHE
    with _CACHE_LOCK:
        if _SETTINGS_CACHE is None:
            _SETTINGS_CACHE = load_settings()
        return _SETTINGS_CACHE


def log_effective_settings(settings: Settings, logger: Optional[logging.Logger] = None) -> None:
    global _STARTUP_LOG_EMITTED
    with _CACHE_LOCK:
        if _STARTUP_LOG_EMITTED:
            return
        _STARTUP_LOG_EMITTED = True
    logger = logger or logging.getLogger("video_pipeline.config")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    payload = settings.to_log_payload()
    message = f"[CONFIG] effective={json.dumps(payload, sort_keys=True, ensure_ascii=True)}"
    logger.info(message)


def reset_startup_log_for_tests() -> None:
    global _STARTUP_LOG_EMITTED
    with _CACHE_LOCK:
        _STARTUP_LOG_EMITTED = False

