"""Shared configuration dataclasses for the modular video pipeline.

These helpers centralise the defaults we want to reuse while we
progressively refactor the monolithic `VideoProcessor`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from config import Config


TRUE_SET = {"1", "true", "t", "yes", "y", "on"}
FALSE_SET = {"0", "false", "f", "no", "n", "off"}


def to_bool(v: object, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in TRUE_SET:
        return True
    if s in FALSE_SET:
        return False
    return default


def to_list(v: object, sep: str = ",") -> List[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return [str(x).strip() for x in v if str(x).strip()]
    return [x.strip() for x in str(v).split(sep) if x.strip()]


def _split_csv(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    items: list[str] = []
    for chunk in str(raw).replace(";", ",").split(","):
        value = chunk.strip()
        if value:
            items.append(value)
    return items


def _env_bool(*keys: str, default: Optional[bool] = None) -> Optional[bool]:
    for key in keys:
        if not key:
            continue
        flag = _env_to_bool(os.getenv(key))
        if flag is not None:
            return flag
    return default


def _env_int(*keys: str, default: Optional[int] = None, minimum: Optional[int] = None) -> Optional[int]:
    for key in keys:
        if not key:
            continue
        raw = os.getenv(key)
        if raw is None:
            continue
        try:
            parsed = int(str(raw).strip())
        except (TypeError, ValueError):
            continue
        if minimum is not None and parsed < minimum:
            parsed = minimum
        return parsed
    return default


def _coerce_positive_int(value: Optional[str | int], default: int) -> int:
    try:
        if isinstance(value, str):
            value = value.strip()
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _coerce_positive_float(value: Optional[str | float], default: float) -> float:
    try:
        if isinstance(value, str):
            value = value.strip()
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _default_per_segment_limit() -> int:
    fallback = _coerce_positive_int(getattr(Config, "BROLL_FETCH_MAX_PER_KEYWORD", 0), 6)
    env_value = _env_int("BROLL_FETCH_MAX_PER_KEYWORD", "FETCH_MAX", minimum=1)
    if env_value is not None:
        return max(1, env_value)
    return max(1, fallback)


def _default_allow_images() -> bool:
    override = _env_bool("BROLL_FETCH_ALLOW_IMAGES")
    if override is not None:
        return override
    config_value = getattr(Config, "BROLL_FETCH_ALLOW_IMAGES", None)
    if isinstance(config_value, bool):
        return config_value
    return True


def _default_allow_videos() -> bool:
    override = _env_bool("BROLL_FETCH_ALLOW_VIDEOS")
    if override is not None:
        return override
    config_value = getattr(Config, "BROLL_FETCH_ALLOW_VIDEOS", None)
    if isinstance(config_value, bool):
        return config_value
    return True


def _default_max_segments_in_flight() -> int:
    return _coerce_positive_int(os.getenv("PIPELINE_MAX_SEGMENTS_IN_FLIGHT"), 1)


def _default_llm_queries_per_segment() -> int:
    return _coerce_positive_int(os.getenv("PIPELINE_LLM_MAX_QUERIES_PER_SEGMENT"), 3)


def _parse_provider_list(raw: Optional[str]) -> Optional[set[str]]:
    if not raw:
        return None
    cleaned_parts = [item.lower() for item in _split_csv(raw)]
    if not cleaned_parts:
        return set()
    if {"all"} == set(cleaned_parts):
        return None
    return set(cleaned_parts)


def _provider_enabled_from_env(provider: str, default: bool) -> bool:
    normalized = provider.strip().upper()
    if not normalized:
        return default

    disable_keys = (
        f"BROLL_FETCH_DISABLE_{normalized}",
        f"AI_BROLL_DISABLE_{normalized}",
    )
    for key in disable_keys:
        flag = _env_bool(key)
        if flag is True:
            return False

    enable_keys = (
        f"BROLL_FETCH_ENABLE_{normalized}",
        f"AI_BROLL_ENABLE_{normalized}",
        f"ENABLE_{normalized}",
    )
    result = default
    for key in enable_keys:
        flag = _env_bool(key)
        if flag is not None:
            result = flag
    return result


def _provider_api_key(env_key: str) -> Optional[str]:
    value = os.getenv(env_key)
    if isinstance(value, str):
        value = value.strip()
    if value:
        return value
    fallback = getattr(Config, env_key, None)
    if isinstance(fallback, str):
        fallback = fallback.strip()
        if fallback:
            return fallback
    return None


def _build_provider_defaults(
    *,
    selected: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
) -> list[ProviderConfig]:
    resolved_limit = limit if limit is not None else _default_per_segment_limit()
    selection: Optional[set[str]]
    if selected is not None:
        selection = {item.lower() for item in selected}
    else:
        raw_selection = (
            os.getenv("BROLL_FETCH_PROVIDER")
            or os.getenv("AI_BROLL_FETCH_PROVIDER")
            or getattr(Config, "BROLL_FETCH_PROVIDER", None)
        )
        selection = _parse_provider_list(str(raw_selection) if raw_selection else None)

    provider_specs = [
        ("pexels", "PEXELS_API_KEY", 1.0),
        ("pixabay", "PIXABAY_API_KEY", 0.9),
    ]

    providers: list[ProviderConfig] = []
    for name, env_key, weight in provider_specs:
        api_key = _provider_api_key(env_key)
        if not api_key:
            continue

        normalized_name = name.lower()
        if selection is not None and normalized_name not in selection:
            continue

        enabled = _provider_enabled_from_env(name, True)

        provider = ProviderConfig(
            name=name,
            weight=weight,
            enabled=enabled,
            max_results=resolved_limit,
            supports_images=False,
            supports_videos=True,
        )
        providers.append(provider)
    return providers


@dataclass(slots=True)
class PipelinePaths:
    """Local directories used by the pipeline.

    The defaults are aligned with the existing ``Config`` class so that the
    new modules remain drop-in replacements for the current behaviour.
    """

    clips_dir: Path = field(default_factory=lambda: Config.CLIPS_FOLDER)
    output_dir: Path = field(default_factory=lambda: Config.OUTPUT_FOLDER)
    temp_dir: Path = field(default_factory=lambda: Config.TEMP_FOLDER)


@dataclass(slots=True)
class ProviderConfig:
    """Configuration for a single external media provider."""

    name: str
    weight: float = 1.0
    enabled: bool = True
    max_results: int = 6
    timeout_s: float = 8.0
    supports_images: bool = False
    supports_videos: bool = True


@dataclass(slots=True)
class FetcherOrchestratorConfig:
    """Controls how the orchestrator queries the external APIs."""

    providers: Sequence[ProviderConfig] = field(default_factory=lambda: tuple(_build_provider_defaults()))
    per_segment_limit: int = field(default_factory=_default_per_segment_limit)
    parallel_requests: int = 4
    allow_images: bool = field(default_factory=_default_allow_images)
    allow_videos: bool = field(default_factory=_default_allow_videos)
    retry_count: int = 2
    request_timeout_s: float = 8.0

    def __post_init__(self) -> None:
        limit = max(1, int(self.per_segment_limit or 1))
        self.per_segment_limit = limit
        self.allow_images = bool(self.allow_images)
        self.allow_videos = bool(self.allow_videos)

        providers: list[ProviderConfig] = []
        for provider in self.providers:
            provider.enabled = bool(getattr(provider, "enabled", True))
            provider.max_results = max(1, min(int(getattr(provider, "max_results", limit) or limit), limit))
            providers.append(provider)

        # ``providers`` is declared as a Sequence for flexibility but we keep an
        # immutable tuple internally to avoid accidental mutations from callers.
        self.providers = tuple(providers)

    @classmethod
    def from_environment(cls) -> "FetcherOrchestratorConfig":
        """Create a config instance honouring environment overrides."""

        timeout = _coerce_positive_float(
            os.getenv("PIPELINE_LLM_TIMEOUT_S")
            or getattr(Config, "PIPELINE_LLM_TIMEOUT_S", None),
            8.0,
        )

        fetch_max_raw = os.getenv("FETCH_MAX")
        try:
            per_segment_limit = int(str(fetch_max_raw).strip()) if fetch_max_raw is not None else 8
        except (TypeError, ValueError):
            per_segment_limit = 8
        if per_segment_limit <= 0:
            per_segment_limit = 8

        override_limit = os.getenv("BROLL_FETCH_MAX_PER_KEYWORD")
        if override_limit is not None:
            try:
                parsed = int(str(override_limit).strip())
            except (TypeError, ValueError):
                parsed = per_segment_limit
            if parsed > 0:
                per_segment_limit = parsed
        elif getattr(Config, "BROLL_FETCH_MAX_PER_KEYWORD", None):
            try:
                parsed = int(getattr(Config, "BROLL_FETCH_MAX_PER_KEYWORD"))
            except (TypeError, ValueError):
                parsed = per_segment_limit
            if parsed > 0:
                per_segment_limit = parsed

        provider_env = os.getenv("BROLL_FETCH_PROVIDER")
        if not provider_env:
            provider_env = os.getenv("AI_BROLL_FETCH_PROVIDER")
        provider_tokens = to_list(provider_env)
        if not provider_tokens:
            provider_tokens = ["pixabay"]

        normalized_selection = {token.lower() for token in provider_tokens}
        if normalized_selection == {"all"}:
            selection: Optional[Iterable[str]] = None
        else:
            selection = normalized_selection

        providers = []
        for provider in _build_provider_defaults(selected=selection, limit=per_segment_limit):
            if provider.enabled:
                provider.max_results = per_segment_limit
            providers.append(provider)

        allow_images = to_bool(os.getenv("BROLL_FETCH_ALLOW_IMAGES"), default=_default_allow_images())
        allow_videos = to_bool(os.getenv("BROLL_FETCH_ALLOW_VIDEOS"), default=_default_allow_videos())

        return cls(
            providers=tuple(providers),
            per_segment_limit=per_segment_limit,
            allow_images=allow_images,
            allow_videos=allow_videos,
            request_timeout_s=timeout,
        )


def resolved_providers(cfg: Optional[FetcherOrchestratorConfig]) -> list[str]:
    """Return configured provider names, falling back to pixabay."""

    if cfg is None:
        return ["pixabay"]

    names: list[str] = []
    for provider in getattr(cfg, "providers", ()):
        name = str(getattr(provider, "name", "") or "").strip()
        if name:
            names.append(name)

    if names:
        # Preserve order but de-duplicate while keeping first occurrence.
        seen: set[str] = set()
        unique: list[str] = []
        for name in names:
            lowered = name.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            unique.append(name)
        return unique

    return ["pixabay"]


def detected_provider_names(
    config: Optional[FetcherOrchestratorConfig] = None,
    *,
    only_enabled: bool = True,
) -> list[str]:
    if config is None:
        config = FetcherOrchestratorConfig.from_environment()
    providers = list(getattr(config, "providers", ()))
    names: list[str] = []
    for provider in providers:
        if only_enabled and not provider.enabled:
            continue
        names.append(provider.name)
    if names:
        return names
    fallback = resolved_providers(config)
    return list(fallback)


def _env_to_bool(value: Optional[str], *, default: Optional[bool] = None) -> Optional[bool]:
    """Parse boolean-like environment values.

    Returns ``True``/``False`` when the input can be interpreted as such, or
    ``default`` when the value is empty/unknown.
    """
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    if text in TRUE_SET:
        return True
    if text in FALSE_SET:
        return False
    return default


def _selection_config_from_environment() -> "SelectionConfig":
    """Factory reading environment overrides for selection guardrails."""

    config = SelectionConfig()

    env_min_score = os.getenv("BROLL_MIN_SCORE") or os.getenv("BROLL_SELECTION_MIN_SCORE")
    if env_min_score:
        try:
            config.min_score = max(0.0, float(env_min_score))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pass

    raw_budget = os.getenv("BROLL_FORCED_KEEP")
    if raw_budget is not None:
        try:
            parsed = int(raw_budget)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            parsed = None
        if parsed is not None:
            config.forced_keep_budget = max(0, parsed)

    enable_flag = None
    for key in ("BROLL_FORCED_KEEP_ENABLE", "BROLL_FORCED_KEEP_ENABLED"):
        flag = _env_to_bool(os.getenv(key))
        if flag is not None:
            enable_flag = flag

    disable_flag = None
    for key in ("BROLL_FORCED_KEEP_DISABLE", "BROLL_FORCED_KEEP_DISABLED"):
        flag = _env_to_bool(os.getenv(key))
        if flag is not None:
            disable_flag = flag

    if disable_flag is True:
        config.allow_forced_keep = False
    elif disable_flag is False and enable_flag is None:
        config.allow_forced_keep = True
    elif enable_flag is not None:
        config.allow_forced_keep = enable_flag

    return config


@dataclass(slots=True)
class SelectionConfig:
    """Selection guard-rails applied after ranking."""

    min_score: float = 0.54
    prefer_landscape: bool = True
    min_duration_s: float = 3.0
    require_license_ok: bool = True
    allow_forced_keep: bool = True
    forced_keep_budget: Optional[int] = None

    @classmethod
    def from_environment(cls) -> "SelectionConfig":
        """Build a configuration instance with environment overrides applied."""

        return _selection_config_from_environment()


@dataclass(slots=True)
class OrchestratorRuntimeConfig:
    """Runtime tuning for the orchestrator."""

    max_segments_in_flight: int = field(default_factory=_default_max_segments_in_flight)

    def __post_init__(self) -> None:
        self.max_segments_in_flight = max(1, int(self.max_segments_in_flight or 1))


@dataclass(slots=True)
class TimeboxingConfig:
    """Timeout thresholds (in milliseconds)."""

    fetch_rank_ms: int = 3000


@dataclass(slots=True)
class DedupePolicy:
    """Defines how we detect duplicates within a single video."""

    enable_url_tracking: bool = True
    enable_phash: bool = True
    phash_distance: int = 6
    title_similarity_threshold: float = 0.85


@dataclass(slots=True)
class LLMServiceConfig:
    """Runtime settings for the LLM keyword/metadata generator."""

    provider_name: str = "pipeline_integration"
    max_queries_per_segment: int = field(default_factory=_default_llm_queries_per_segment)
    include_bigrams: bool = True
    include_trigrams: bool = True
    cache_ttl_s: float = 15 * 60  # 15 minutes

    def __post_init__(self) -> None:
        self.max_queries_per_segment = _coerce_positive_int(self.max_queries_per_segment, 3)


@dataclass(slots=True)
class RendererConfig:
    """Configuration shared by the rendering subsystem."""

    crossfade_ms: int = 320
    pad_in_s: float = 0.25
    pad_out_s: float = 0.25
    min_duration_s: float = 2.0
    prefer_streaming: bool = True


@dataclass(slots=True)
class PipelineConfigBundle:
    """Aggregates the sub-configs for convenience."""

    paths: PipelinePaths = field(default_factory=PipelinePaths)
    fetcher: FetcherOrchestratorConfig = field(default_factory=FetcherOrchestratorConfig.from_environment)
    selection: SelectionConfig = field(default_factory=_selection_config_from_environment)
    orchestrator: OrchestratorRuntimeConfig = field(default_factory=OrchestratorRuntimeConfig)
    timeboxing: TimeboxingConfig = field(default_factory=TimeboxingConfig)
    dedupe: DedupePolicy = field(default_factory=DedupePolicy)
    llm: LLMServiceConfig = field(default_factory=LLMServiceConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)
