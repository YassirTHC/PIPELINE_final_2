"""Shared configuration dataclasses for the modular video pipeline.

These helpers centralise the defaults we want to reuse while we
progressively refactor the monolithic `VideoProcessor`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

try:  # Optional dependency: the typed settings live in the video_pipeline package.
    from video_pipeline.config import Settings, get_settings
except Exception:  # pragma: no cover - tests may stub the settings layer
    Settings = None  # type: ignore[assignment]
    get_settings = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


TRUE_SET = {"1", "true", "t", "yes", "y", "on"}
FALSE_SET = {"0", "false", "f", "no", "n", "off"}


_DEFAULT_CLIPS_DIR = Path("clips")
_DEFAULT_OUTPUT_DIR = Path("output")
_DEFAULT_TEMP_DIR = Path("temp")


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


def tfidf_fallback_disabled_from_env() -> Optional[bool]:
    """Return the TF-IDF disable flag from the typed settings layer."""

    settings = _current_settings()
    if settings is None:
        return None
    try:
        return bool(getattr(settings, "tfidf_fallback_disabled"))
    except Exception:  # pragma: no cover - defensive
        logger.debug("[CONFIG] tfidf fallback flag missing", exc_info=True)
        return None


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
    settings = _current_settings()
    if settings is not None:
        try:
            configured = int(getattr(settings.fetch, "max_per_keyword", 0))
        except Exception:  # pragma: no cover - defensive when fetch missing
            logger.debug("[CONFIG] fetch.max_per_keyword unavailable", exc_info=True)
            configured = 0
        if configured > 0:
            return configured

    return 6


def _default_allow_images() -> bool:
    settings = _current_settings()
    if settings is not None:
        try:
            return bool(getattr(settings.fetch, "allow_images"))
        except Exception:  # pragma: no cover - defensive
            logger.debug("[CONFIG] fetch.allow_images missing", exc_info=True)
    return True


def _default_allow_videos() -> bool:
    settings = _current_settings()
    if settings is not None:
        try:
            return bool(getattr(settings.fetch, "allow_videos"))
        except Exception:  # pragma: no cover - defensive
            logger.debug("[CONFIG] fetch.allow_videos missing", exc_info=True)
    return True


def _default_max_segments_in_flight() -> int:
    settings = _current_settings()
    if settings is not None:
        try:
            configured = int(getattr(settings, "max_segments_in_flight", 1))
        except Exception:  # pragma: no cover - defensive
            configured = 1
        if configured > 0:
            return configured
    return 1


def _default_llm_queries_per_segment() -> int:
    settings = _current_settings()
    if settings is not None:
        try:
            configured = int(getattr(settings, "llm_max_queries_per_segment", 3))
        except Exception:  # pragma: no cover - defensive
            configured = 3
        if configured > 0:
            return configured
    return 3


def _default_disable_tfidf_fallback() -> bool:
    flag = tfidf_fallback_disabled_from_env()
    return bool(flag) if flag is not None else False


def _provider_api_key(env_key: str) -> Optional[str]:
    settings = _current_settings()
    if settings is not None:
        try:
            api_keys = getattr(settings.fetch, "api_keys", {})
            typed = api_keys.get(env_key)
            if isinstance(typed, str):
                cleaned = typed.strip()
                if cleaned:
                    return cleaned
        except Exception:  # pragma: no cover - defensive
            logger.debug("[CONFIG] fetch.api_keys lookup failed", exc_info=True)
    return None


def _build_provider_defaults(
    *,
    selected: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
) -> list[ProviderConfig]:
    resolved_limit = limit if limit is not None else _default_per_segment_limit()
    settings = _current_settings()
    fetch_settings = getattr(settings, "fetch", None) if settings is not None else None

    typed_order: list[str] = []
    typed_limits: dict[str, int] = {}
    api_keys: dict[str, Optional[str]] = {}
    if fetch_settings is not None:
        try:
            typed_order = [
                str(item).strip().lower()
                for item in getattr(fetch_settings, "providers", [])
                if str(item).strip()
            ]
        except Exception:  # pragma: no cover - defensive
            logger.debug("[CONFIG] fetch.providers inspection failed", exc_info=True)
            typed_order = []
        try:
            for key, value in getattr(fetch_settings, "provider_limits", {}).items():
                try:
                    parsed = int(value)
                except (TypeError, ValueError):
                    continue
                if parsed > 0:
                    typed_limits[str(key).strip().lower()] = parsed
        except Exception:  # pragma: no cover - defensive
            logger.debug("[CONFIG] fetch.provider_limits inspection failed", exc_info=True)
        try:
            raw_keys = getattr(fetch_settings, "api_keys", {})
            api_keys = {str(k): v for k, v in raw_keys.items()}
        except Exception:  # pragma: no cover - defensive
            logger.debug("[CONFIG] fetch.api_keys inspection failed", exc_info=True)

    selection: Optional[set[str]]
    if selected is not None:
        selection = {str(item).strip().lower() for item in selected if str(item).strip()}
    elif typed_order:
        selection = set(typed_order)
    else:
        selection = {"pixabay"}

    if selection is not None and "all" in selection:
        selection = None

    provider_specs = [
        ("pexels", "PEXELS_API_KEY", 1.0),
        ("pixabay", "PIXABAY_API_KEY", 0.9),
    ]

    if typed_order:
        ordered_specs: list[tuple[str, str, float]] = []
        for provider_name in typed_order:
            for spec in provider_specs:
                if spec[0].lower() == provider_name and spec not in ordered_specs:
                    ordered_specs.append(spec)
        for spec in provider_specs:
            if spec not in ordered_specs:
                ordered_specs.append(spec)
        provider_specs = ordered_specs

    providers: list[ProviderConfig] = []
    for name, env_key, weight in provider_specs:
        normalized_name = name.lower()
        if selection is not None and normalized_name not in selection:
            continue

        api_key = api_keys.get(env_key) if api_keys else None
        if api_key is None:
            api_key = _provider_api_key(env_key)
        if not api_key:
            continue

        limit_override = typed_limits.get(normalized_name)
        max_results = limit_override if limit_override is not None else resolved_limit

        provider = ProviderConfig(
            name=name,
            weight=weight,
            enabled=True,
            max_results=max_results,
            supports_images=False,
            supports_videos=True,
        )
        providers.append(provider)
    return providers


def _path_from_settings(attr: str, fallback: Path) -> Path:
    settings = _current_settings()
    if settings is not None:
        try:
            value = getattr(settings, attr)
        except Exception:  # pragma: no cover - defensive
            logger.debug("[CONFIG] settings.%s unavailable", attr, exc_info=True)
        else:
            if isinstance(value, Path):
                return value
            if value is not None:
                try:
                    return Path(str(value))
                except Exception:  # pragma: no cover - defensive
                    logger.debug("[CONFIG] unable to normalise path for %s", attr, exc_info=True)
    return fallback


@dataclass(slots=True)
class PipelinePaths:
    """Local directories used by the pipeline.

    The defaults are aligned with the existing ``Config`` class so that the
    new modules remain drop-in replacements for the current behaviour.
    """

    clips_dir: Path = field(default_factory=lambda: _path_from_settings("clips_dir", _DEFAULT_CLIPS_DIR))
    output_dir: Path = field(default_factory=lambda: _path_from_settings("output_dir", _DEFAULT_OUTPUT_DIR))
    temp_dir: Path = field(default_factory=lambda: _path_from_settings("temp_dir", _DEFAULT_TEMP_DIR))


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
            raw_limit = getattr(provider, "max_results", limit)
            try:
                parsed_limit = int(raw_limit)
            except (TypeError, ValueError):
                parsed_limit = limit
            if parsed_limit <= 0:
                parsed_limit = limit
            provider.max_results = parsed_limit
            providers.append(provider)

        # ``providers`` is declared as a Sequence for flexibility but we keep an
        # immutable tuple internally to avoid accidental mutations from callers.
        self.providers = tuple(providers)

    @classmethod
    def from_environment(cls) -> "FetcherOrchestratorConfig":
        """Create a config instance backed purely by the typed settings."""

        settings = _current_settings()
        fetch_settings = getattr(settings, "fetch", None) if settings is not None else None

        per_segment_limit = _default_per_segment_limit()
        providers = _build_provider_defaults(limit=per_segment_limit)

        allow_images = _default_allow_images()
        allow_videos = _default_allow_videos()
        timeout = 8.0

        if fetch_settings is not None:
            try:
                timeout = float(getattr(fetch_settings, "timeout_s", timeout))
            except Exception:  # pragma: no cover - defensive
                logger.debug("[CONFIG] fetch.timeout_s unavailable", exc_info=True)
            try:
                typed_limit = int(getattr(fetch_settings, "max_per_keyword", per_segment_limit))
            except Exception:  # pragma: no cover - defensive
                logger.debug("[CONFIG] fetch.max_per_keyword unavailable", exc_info=True)
            else:
                if typed_limit > 0:
                    per_segment_limit = typed_limit
            try:
                allow_images = bool(getattr(fetch_settings, "allow_images", allow_images))
            except Exception:  # pragma: no cover - defensive
                logger.debug("[CONFIG] fetch.allow_images missing", exc_info=True)
            try:
                allow_videos = bool(getattr(fetch_settings, "allow_videos", allow_videos))
            except Exception:  # pragma: no cover - defensive
                logger.debug("[CONFIG] fetch.allow_videos missing", exc_info=True)

        effective_limits = [
            max(1, int(getattr(provider, "max_results", per_segment_limit) or per_segment_limit))
            for provider in providers
            if getattr(provider, "enabled", True)
        ]
        if effective_limits:
            per_segment_limit = min(per_segment_limit, min(effective_limits))

        return cls(
            providers=tuple(providers),
            per_segment_limit=per_segment_limit,
            allow_images=allow_images,
            allow_videos=allow_videos,
            request_timeout_s=max(0.1, float(timeout)),
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

def _selection_config_from_environment() -> "SelectionConfig":
    """Factory backed by typed settings only (legacy env removed)."""

    return SelectionConfig()


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
    disable_tfidf_fallback: bool = field(default_factory=_default_disable_tfidf_fallback)

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

def _current_settings() -> Optional["Settings"]:
    global get_settings, Settings
    if get_settings is None:  # pragma: no cover - optional dependency in tests
        try:
            from video_pipeline.config import Settings as _Settings, get_settings as _get_settings  # type: ignore
        except Exception:  # pragma: no cover - defensive guardrail when settings layer is unavailable
            return None
        else:
            Settings = _Settings  # type: ignore[assignment]
            get_settings = _get_settings  # type: ignore[assignment]
    try:
        return get_settings()
    except Exception:  # pragma: no cover - defensive guardrail
        logger.debug("[CONFIG] typed settings unavailable", exc_info=True)
        return None


