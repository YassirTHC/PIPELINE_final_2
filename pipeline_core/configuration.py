"""Shared configuration dataclasses for the modular video pipeline.

These helpers centralise the defaults we want to reuse while we
progressively refactor the monolithic `VideoProcessor`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from config import Config


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


@dataclass(slots=True)
class FetcherOrchestratorConfig:
    """Controls how the orchestrator queries the external APIs."""

    providers: Sequence[ProviderConfig] = field(
        default_factory=lambda: (
            ProviderConfig(name="pexels", weight=1.0, max_results=6),
            ProviderConfig(name="pixabay", weight=0.9, max_results=6),
        )
    )
    per_segment_limit: int = 6
    parallel_requests: int = 4
    allow_images: bool = False
    allow_videos: bool = True
    retry_count: int = 2
    request_timeout_s: float = 8.0


def _env_to_bool(value: Optional[str], *, default: Optional[bool] = None) -> Optional[bool]:
    """Parse boolean-like environment values.

    Returns ``True``/``False`` when the input can be interpreted as such, or
    ``default`` when the value is empty/unknown.
    """

    if value is None:
        return default
    if isinstance(value, bool):  # pragma: no cover - defensive (env always str)
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
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

    max_segments_in_flight: int = 2


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
    max_queries_per_segment: int = 6
    include_bigrams: bool = True
    include_trigrams: bool = True
    cache_ttl_s: float = 15 * 60  # 15 minutes


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
    fetcher: FetcherOrchestratorConfig = field(default_factory=FetcherOrchestratorConfig)
    selection: SelectionConfig = field(default_factory=_selection_config_from_environment)
    orchestrator: OrchestratorRuntimeConfig = field(default_factory=OrchestratorRuntimeConfig)
    timeboxing: TimeboxingConfig = field(default_factory=TimeboxingConfig)
    dedupe: DedupePolicy = field(default_factory=DedupePolicy)
    llm: LLMServiceConfig = field(default_factory=LLMServiceConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)
