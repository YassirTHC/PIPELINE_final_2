"""Strongly typed configuration loader for the video pipeline."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import Dict, List, Mapping, Optional, Tuple


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
    min_start_s: float = 0.7
    min_gap_s: float = 1.5
    max_gap_s: float = 4.0
    no_repeat_s: float = 4.0
    min_duration_s: float = 0.8
    max_duration_s: float = 2.0
    initial_lead_s: float = 0.7
    first_window_max_s: float = 1.5
    target_total: int = 12


@dataclass(slots=True)
class BrollSelectionSettings:
    enable_adaptive_topk: bool = False
    elbow_drop_pct: float = 0.15
    min_ratio_vs_best: float = 0.85
    k_max_per_query: int = 6
    k_max_per_query_generic: int = 4
    k_seg_max: int = 18
    generic_query_variants: Tuple[str, ...] = (
        "typing on laptop",
        "walking outdoors",
        "whiteboard sketch",
    )


@dataclass(slots=True)
class BrollDiversitySettings:
    enable_mmr: bool = False
    mmr_alpha: float = 0.7
    repeat_penalty: float = 0.25
    repeat_window: int = 2


@dataclass(slots=True)
class BrollEarlyStopSettings:
    enable: bool = False
    min_selected_before_stop: int = 1


@dataclass(slots=True)
class BrollBackfillSettings:
    enable: bool = False
    local_max_gap_multiplier: float = 1.2
    short_insert_min_s: float = 0.8
    short_insert_max_s: float = 1.2
    neutral_queries: Tuple[str, ...] = (
        "typing on laptop",
        "city street broll",
        "reaction close-up",
        "whiteboard sketch",
    )
    mini_topk: int = 3


@dataclass(slots=True)
class SchedulerTuningSettings:
    enable_local_relax: bool = False
    local_gap_multiplier: float = 1.2
    micro_insert_min_s: float = 0.8
    micro_insert_max_s: float = 1.2
    coverage_target: float = 0.8
    keyword_align_slack_s: float = 0.7


@dataclass(slots=True)
class SubtitleSettings:
    font_path: Optional[str]
    engine: str = "hormozi"
    font: Optional[str] = None
    font_size: int = 96
    theme: str = "hormozi"
    primary_color: str = "#FFFFFF"
    secondary_color: str = "#FBC531"
    stroke_color: str = "#000000"
    subtitle_safe_margin_px: int = 220
    keyword_background: bool = False
    stroke_px: int = 6
    shadow_opacity: float = 0.35
    shadow_offset: int = 3
    shadow_color: str = "#000000"
    background_color: str = "#000000"
    background_opacity: float = 0.35
    margin_bottom_pct: float = 0.12
    max_lines: int = 3
    max_chars_per_line: int = 24
    uppercase_keywords: bool = True
    uppercase_min_length: int = 6
    highlight_scale: float = 1.08
    enable_emojis: bool = True
    responsive_mode: bool = False
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
    broll_selection: BrollSelectionSettings
    broll_diversity: BrollDiversitySettings
    broll_early_stop: BrollEarlyStopSettings
    broll_backfill: BrollBackfillSettings
    scheduler_tuning: SchedulerTuningSettings
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
                "max_gap_s": getattr(self.broll, "max_gap_s", None),
                "no_repeat_s": self.broll.no_repeat_s,
                "min_duration_s": getattr(self.broll, "min_duration_s", None),
                "max_duration_s": getattr(self.broll, "max_duration_s", None),
                "initial_lead_s": getattr(self.broll, "initial_lead_s", None),
                "first_window_max_s": getattr(self.broll, "first_window_max_s", None),
                "target_total": getattr(self.broll, "target_total", None),
            },
            "broll_selection": {
                "enable_adaptive_topk": self.broll_selection.enable_adaptive_topk,
                "elbow_drop_pct": self.broll_selection.elbow_drop_pct,
                "min_ratio_vs_best": self.broll_selection.min_ratio_vs_best,
                "k_max_per_query": self.broll_selection.k_max_per_query,
                "k_max_per_query_generic": self.broll_selection.k_max_per_query_generic,
                "k_seg_max": self.broll_selection.k_seg_max,
                "generic_query_variants": list(self.broll_selection.generic_query_variants),
            },
            "broll_diversity": {
                "enable_mmr": self.broll_diversity.enable_mmr,
                "mmr_alpha": self.broll_diversity.mmr_alpha,
                "repeat_penalty": self.broll_diversity.repeat_penalty,
                "repeat_window": self.broll_diversity.repeat_window,
            },
            "broll_early_stop": {
                "enable": self.broll_early_stop.enable,
                "min_selected_before_stop": self.broll_early_stop.min_selected_before_stop,
            },
            "broll_backfill": {
                "enable": self.broll_backfill.enable,
                "local_max_gap_multiplier": self.broll_backfill.local_max_gap_multiplier,
                "short_insert_min_s": self.broll_backfill.short_insert_min_s,
                "short_insert_max_s": self.broll_backfill.short_insert_max_s,
                "mini_topk": self.broll_backfill.mini_topk,
                "neutral_queries": list(self.broll_backfill.neutral_queries),
            },
            "scheduler_tuning": {
                "enable_local_relax": self.scheduler_tuning.enable_local_relax,
                "local_gap_multiplier": self.scheduler_tuning.local_gap_multiplier,
                "micro_insert_min_s": self.scheduler_tuning.micro_insert_min_s,
                "micro_insert_max_s": self.scheduler_tuning.micro_insert_max_s,
                "coverage_target": self.scheduler_tuning.coverage_target,
                "keyword_align_slack_s": self.scheduler_tuning.keyword_align_slack_s,
            },
            "subtitles": {
                "font_path": self.subtitles.font_path,
                "engine": self.subtitles.engine,
                "font": self.subtitles.font,
                "font_size": self.subtitles.font_size,
                "theme": self.subtitles.theme,
                "primary_color": self.subtitles.primary_color,
                "secondary_color": self.subtitles.secondary_color,
                "stroke_color": self.subtitles.stroke_color,
                "subtitle_safe_margin_px": self.subtitles.subtitle_safe_margin_px,
                "keyword_background": self.subtitles.keyword_background,
                "enable_emojis": self.subtitles.enable_emojis,
                "stroke_px": self.subtitles.stroke_px,
                "shadow_opacity": self.subtitles.shadow_opacity,
                "shadow_offset": self.subtitles.shadow_offset,
                "shadow_color": self.subtitles.shadow_color,
                "background_color": self.subtitles.background_color,
                "background_opacity": self.subtitles.background_opacity,
                "margin_bottom_pct": self.subtitles.margin_bottom_pct,
                "max_lines": self.subtitles.max_lines,
                "max_chars_per_line": self.subtitles.max_chars_per_line,
                "uppercase_keywords": self.subtitles.uppercase_keywords,
                "uppercase_min_length": self.subtitles.uppercase_min_length,
                "highlight_scale": self.subtitles.highlight_scale,
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
    target_lang = _clean_text(
        _env(env, "VP_LLM_LANG")
        or _env(env, "PIPELINE_LLM_TARGET_LANG", "en")
        or "en"
    ) or "en"
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
    min_start = _coerce_float(
        _env(env, "PIPELINE_BROLL_MIN_START_SECONDS", "0.7"),
        0.7,
        minimum=0.0,
    )
    min_gap = _coerce_float(
        _env(env, "PIPELINE_BROLL_MIN_GAP_SECONDS", "1.5"),
        1.5,
        minimum=1.5,
    )
    max_gap = _coerce_float(
        _env(env, "PIPELINE_BROLL_MAX_GAP_SECONDS", "4.0"),
        4.0,
        minimum=0.0,
    )
    no_repeat = _coerce_float(
        _env(env, "PIPELINE_BROLL_NO_REPEAT_SECONDS", "4.0"),
        4.0,
        minimum=0.0,
    )
    min_duration = _coerce_float(
        _env(env, "PIPELINE_BROLL_MIN_DURATION_SECONDS", "0.8"),
        0.8,
        minimum=0.0,
    )
    max_duration = _coerce_float(
        _env(env, "PIPELINE_BROLL_MAX_DURATION_SECONDS", "2.0"),
        2.0,
        minimum=0.0,
    )
    if max_duration <= 0.0 or max_duration < min_duration:
        max_duration = max(min_duration, 0.1)
    initial_lead = _coerce_float(
        _env(env, "PIPELINE_BROLL_INITIAL_LEAD_SECONDS", "0.7"),
        0.7,
        minimum=0.0,
    )
    first_window_max = _coerce_float(
        _env(env, "PIPELINE_BROLL_FIRST_WINDOW_MAX_SECONDS", "1.5"),
        1.5,
        minimum=0.0,
    )
    if first_window_max < initial_lead:
        first_window_max = initial_lead
    target_total = _coerce_int(
        _env(env, "PIPELINE_BROLL_TARGET_TOTAL", "12"),
        12,
        minimum=0,
    )

    return BrollSettings(
        min_start_s=min_start,
        min_gap_s=min_gap,
        max_gap_s=max_gap,
        no_repeat_s=no_repeat,
        min_duration_s=min_duration,
        max_duration_s=max_duration,
        initial_lead_s=initial_lead,
        first_window_max_s=first_window_max,
        target_total=target_total,
    )


def _broll_selection_settings(env: Optional[Mapping[str, str]]) -> BrollSelectionSettings:
    defaults = BrollSelectionSettings()
    enable = _resolve_bool_env(
        env,
        "BROLL_SELECTION_ENABLE_ADAPTIVE_TOPK",
        "BROLL_ENABLE_ADAPTIVE_TOPK",
        default=defaults.enable_adaptive_topk,
    )
    elbow = _coerce_float(
        _env(env, "BROLL_SELECTION_ELBOW_DROP_PCT"),
        defaults.elbow_drop_pct,
        minimum=0.0,
    )
    ratio = _coerce_float(
        _env(env, "BROLL_SELECTION_MIN_RATIO_VS_BEST"),
        defaults.min_ratio_vs_best,
        minimum=0.0,
    )
    k_max = _coerce_int(
        _env(env, "BROLL_SELECTION_K_MAX_PER_QUERY"),
        defaults.k_max_per_query,
        minimum=1,
    )
    k_max_generic = _coerce_int(
        _env(env, "BROLL_SELECTION_K_MAX_GENERIC"),
        defaults.k_max_per_query_generic,
        minimum=1,
    )
    k_seg = _coerce_int(
        _env(env, "BROLL_SELECTION_K_SEG_MAX"),
        defaults.k_seg_max,
        minimum=1,
    )
    raw_variants = _split_csv(_env(env, "BROLL_SELECTION_GENERIC_VARIANTS"))
    variants = tuple(v.strip() for v in raw_variants if v.strip()) or defaults.generic_query_variants
    return BrollSelectionSettings(
        enable_adaptive_topk=enable,
        elbow_drop_pct=elbow,
        min_ratio_vs_best=ratio,
        k_max_per_query=k_max,
        k_max_per_query_generic=k_max_generic,
        k_seg_max=k_seg,
        generic_query_variants=variants,
    )


def _broll_diversity_settings(env: Optional[Mapping[str, str]]) -> BrollDiversitySettings:
    defaults = BrollDiversitySettings()
    enable = _resolve_bool_env(
        env,
        "BROLL_DIVERSITY_ENABLE_MMR",
        "BROLL_ENABLE_MMR",
        default=defaults.enable_mmr,
    )
    alpha = _coerce_float(
        _env(env, "BROLL_DIVERSITY_MMR_ALPHA"),
        defaults.mmr_alpha,
        minimum=0.0,
    )
    penalty = _coerce_float(
        _env(env, "BROLL_DIVERSITY_REPEAT_PENALTY"),
        defaults.repeat_penalty,
        minimum=0.0,
    )
    window = _coerce_int(
        _env(env, "BROLL_DIVERSITY_REPEAT_WINDOW"),
        defaults.repeat_window,
        minimum=0,
    )
    return BrollDiversitySettings(
        enable_mmr=enable,
        mmr_alpha=alpha,
        repeat_penalty=penalty,
        repeat_window=window,
    )


def _broll_early_stop_settings(env: Optional[Mapping[str, str]]) -> BrollEarlyStopSettings:
    defaults = BrollEarlyStopSettings()
    enable = _resolve_bool_env(
        env,
        "BROLL_EARLY_STOP_ENABLE",
        default=defaults.enable,
    )
    min_selected = _coerce_int(
        _env(env, "BROLL_EARLY_STOP_MIN_SELECTED"),
        defaults.min_selected_before_stop,
        minimum=0,
    )
    return BrollEarlyStopSettings(
        enable=enable,
        min_selected_before_stop=min_selected,
    )


def _broll_backfill_settings(env: Optional[Mapping[str, str]]) -> BrollBackfillSettings:
    defaults = BrollBackfillSettings()
    enable = _resolve_bool_env(
        env,
        "BROLL_BACKFILL_ENABLE",
        default=defaults.enable,
    )
    gap_multiplier = _coerce_float(
        _env(env, "BROLL_BACKFILL_LOCAL_MAX_GAP_MULTIPLIER"),
        defaults.local_max_gap_multiplier,
        minimum=1.0,
    )
    short_min = _coerce_float(
        _env(env, "BROLL_BACKFILL_SHORT_INSERT_MIN_S"),
        defaults.short_insert_min_s,
        minimum=0.0,
    )
    short_max = _coerce_float(
        _env(env, "BROLL_BACKFILL_SHORT_INSERT_MAX_S"),
        defaults.short_insert_max_s,
        minimum=short_min,
    )
    mini_topk = _coerce_int(
        _env(env, "BROLL_BACKFILL_MINI_TOPK"),
        defaults.mini_topk,
        minimum=1,
    )
    neutral_raw = _split_csv(_env(env, "BROLL_BACKFILL_NEUTRAL_QUERIES"))
    neutral_queries = tuple(q.strip() for q in neutral_raw if q.strip()) or defaults.neutral_queries
    return BrollBackfillSettings(
        enable=enable,
        local_max_gap_multiplier=gap_multiplier,
        short_insert_min_s=short_min,
        short_insert_max_s=short_max,
        neutral_queries=neutral_queries,
        mini_topk=mini_topk,
    )


def _scheduler_tuning_settings(env: Optional[Mapping[str, str]]) -> SchedulerTuningSettings:
    defaults = SchedulerTuningSettings()
    enable = _resolve_bool_env(
        env,
        "SCHEDULER_TUNING_ENABLE_LOCAL_RELAX",
        "BROLL_SCHEDULER_ENABLE_LOCAL_RELAX",
        default=defaults.enable_local_relax,
    )
    gap_multiplier = _coerce_float(
        _env(env, "SCHEDULER_TUNING_LOCAL_GAP_MULTIPLIER"),
        defaults.local_gap_multiplier,
        minimum=1.0,
    )
    micro_min = _coerce_float(
        _env(env, "SCHEDULER_TUNING_MICRO_INSERT_MIN_S"),
        defaults.micro_insert_min_s,
        minimum=0.0,
    )
    micro_max = _coerce_float(
        _env(env, "SCHEDULER_TUNING_MICRO_INSERT_MAX_S"),
        defaults.micro_insert_max_s,
        minimum=micro_min,
    )
    coverage_target = _coerce_float(
        _env(env, "SCHEDULER_TUNING_COVERAGE_TARGET"),
        defaults.coverage_target,
        minimum=0.0,
    )
    keyword_slack = _coerce_float(
        _env(env, "SCHEDULER_TUNING_KEYWORD_ALIGN_SLACK_S"),
        defaults.keyword_align_slack_s,
        minimum=0.0,
    )
    return SchedulerTuningSettings(
        enable_local_relax=enable,
        local_gap_multiplier=gap_multiplier,
        micro_insert_min_s=micro_min,
        micro_insert_max_s=micro_max,
        coverage_target=coverage_target,
        keyword_align_slack_s=keyword_slack,
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

    if resolved_font is None:
        fallback_candidate = assets_dir / "Montserrat-ExtraBold.ttf"
        try:
            resolved_font = str(fallback_candidate.resolve())
        except OSError:
            resolved_font = str(fallback_candidate)

    def _env_preferred(*keys: str) -> Optional[str]:
        for key in keys:
            value = _env(env, key)
            if value is not None:
                return value
        return None

    font_name = _env_preferred("PIPELINE_SUBTITLE_FONT", "PIPELINE_SUB_FONT")

    theme = _clean_text(
        _env_preferred(
            "VP_SUBTITLES_THEME",
            "PIPELINE_SUBTITLE_THEME",
            "PIPELINE_SUB_THEME",
        )
        or "hormozi"
    ).lower() or "hormozi"
    primary_color = _clean_text(
        _env_preferred("VP_SUBTITLES_PRIMARY_COLOR", "PIPELINE_SUBTITLE_PRIMARY_COLOR")
        or "#FFFFFF"
    ) or "#FFFFFF"
    secondary_color = _clean_text(
        _env_preferred("VP_SUBTITLES_SECONDARY_COLOR", "PIPELINE_SUBTITLE_SECONDARY_COLOR")
        or "#FBC531"
    ) or "#FBC531"
    stroke_color = _clean_text(
        _env_preferred("VP_SUBTITLES_STROKE_COLOR", "PIPELINE_SUBTITLE_STROKE_COLOR")
        or "#000000"
    ) or "#000000"

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
    vp_emojis = _env(env, "VP_SUBTITLES_EMOJIS")
    if vp_emojis is not None:
        enable_emojis = _coerce_bool(vp_emojis, default=enable_emojis)
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
    shadow_color = _clean_text(
        _env_preferred("VP_SUBTITLES_SHADOW_COLOR", "PIPELINE_SUBTITLE_SHADOW_COLOR")
        or "#000000"
    ) or "#000000"
    background_color = _clean_text(
        _env_preferred("VP_SUBTITLES_BG_COLOR", "PIPELINE_SUBTITLE_BG_COLOR")
        or "#000000"
    ) or "#000000"
    background_opacity = _coerce_float(
        _env_preferred("VP_SUBTITLES_BG_ALPHA", "PIPELINE_SUBTITLE_BG_ALPHA"),
        0.35,
        minimum=0.0,
    )
    margin_bottom_pct = _coerce_float(
        _env_preferred("VP_SUBTITLES_MARGIN_BOTTOM_PCT", "PIPELINE_SUBTITLE_MARGIN_BOTTOM_PCT"),
        0.12,
        minimum=0.0,
    )
    max_lines = _coerce_int(
        _env_preferred("VP_SUBTITLES_MAX_LINES", "PIPELINE_SUBTITLE_MAX_LINES"),
        3,
        minimum=1,
    )
    max_chars_per_line = _coerce_int(
        _env_preferred("VP_SUBTITLES_MAX_CHARS_PER_LINE", "PIPELINE_SUBTITLE_MAX_CHARS_PER_LINE"),
        24,
        minimum=8,
    )
    uppercase_keywords = _resolve_bool_env(
        env,
        "VP_SUBTITLES_UPPERCASE_KEYWORDS",
        "PIPELINE_SUBTITLE_UPPERCASE_KEYWORDS",
        default=True,
    )
    uppercase_min_length = _coerce_int(
        _env_preferred("VP_SUBTITLES_UPPERCASE_MIN_LEN", "PIPELINE_SUBTITLE_UPPERCASE_MIN_LEN"),
        6,
        minimum=2,
    )
    highlight_scale = _coerce_float(
        _env_preferred("VP_SUBTITLES_HIGHLIGHT_SCALE", "PIPELINE_SUBTITLE_HIGHLIGHT_SCALE"),
        1.08,
        minimum=1.0,
    )
    responsive_mode = _resolve_bool_env(
        env,
        "VP_SUBTITLES_RESPONSIVE",
        "PIPELINE_SUBTITLE_RESPONSIVE",
        default=False,
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

    raw_engine = _env(env, "VP_SUBTITLES_ENGINE")
    engine = _clean_text(raw_engine or "hormozi").lower() or "hormozi"
    if engine not in {"hormozi", "pycaps"}:
        engine = "hormozi"
    if engine == "pycaps" and vp_emojis is None:
        enable_emojis = False

    return SubtitleSettings(
        font_path=resolved_font,
        engine=engine,
        font=font_name,
        font_size=font_size,
        theme=theme,
        primary_color=primary_color,
        secondary_color=secondary_color,
        stroke_color=stroke_color,
        subtitle_safe_margin_px=safe_margin,
        keyword_background=keyword_background,
        stroke_px=stroke_px,
        shadow_opacity=shadow_opacity,
        shadow_offset=shadow_offset,
        shadow_color=shadow_color,
        background_color=background_color,
        background_opacity=background_opacity,
        margin_bottom_pct=margin_bottom_pct,
        max_lines=max_lines,
        max_chars_per_line=max_chars_per_line,
        uppercase_keywords=uppercase_keywords,
        uppercase_min_length=uppercase_min_length,
        highlight_scale=highlight_scale,
        enable_emojis=enable_emojis,
        emoji_target_per_10=emoji_target,
        emoji_min_gap_groups=emoji_min_gap,
        emoji_max_per_segment=emoji_max_segment,
        emoji_no_context_fallback=emoji_fallback,
        hero_emoji_enable=hero_enable,
        hero_emoji_max_per_segment=hero_max,
        responsive_mode=responsive_mode,
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
    broll_selection = _broll_selection_settings(env)
    broll_diversity = _broll_diversity_settings(env)
    broll_early_stop = _broll_early_stop_settings(env)
    broll_backfill = _broll_backfill_settings(env)
    scheduler_tuning = _scheduler_tuning_settings(env)
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
        broll_selection=broll_selection,
        broll_diversity=broll_diversity,
        broll_early_stop=broll_early_stop,
        broll_backfill=broll_backfill,
        scheduler_tuning=scheduler_tuning,
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
