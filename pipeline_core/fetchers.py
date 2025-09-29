"""Fetching orchestration helpers."""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pipeline_core.configuration import FetcherOrchestratorConfig

try:  # pragma: no cover - exercised indirectly via tests
    from src.pipeline.fetchers import (  # type: ignore
        build_search_query,
        pexels_search_videos,
        pixabay_search_videos,
        _best_vertical_video_file,
        _pixabay_best_video_url,
    )
except ModuleNotFoundError:  # pragma: no cover - unit tests provide stubs
    def build_search_query(keywords: Sequence[str]) -> str:
        return " ".join(str(kw).strip() for kw in keywords[:3] if kw).strip()

    def pexels_search_videos(*_args, **_kwargs):
        return []

    def pixabay_search_videos(*_args, **_kwargs):
        return []

    def _best_vertical_video_file(payload):
        return None

    def _pixabay_best_video_url(payload):
        return None


@dataclass(slots=True)
class RemoteAssetCandidate:
    provider: str
    url: str
    thumb_url: Optional[str]
    width: int
    height: int
    duration: Optional[float]
    title: str
    identifier: str
    tags: Sequence[str]


class FetcherOrchestrator:
    """Thin orchestration layer for remote B-roll providers."""

    def __init__(self, config: FetcherOrchestratorConfig | None = None, *, event_logger: Optional[Any] = None) -> None:
        self.config = config or FetcherOrchestratorConfig.from_environment()
        self._event_logger = event_logger
        self._logger = logging.getLogger(__name__)

    def fetch_candidates(
        self,
        keywords: Sequence[str],
        *,
        segment_index: Optional[int] = None,
        duration_hint: Optional[float] = None,
        filters: Optional[dict] = None,
        segment_timeout_s: Optional[float] = None,
    ) -> List[RemoteAssetCandidate]:
        """Fetch candidates in parallel while respecting simple guard rails."""

        queries = self._build_queries(keywords)
        if not queries:
            return []

        allow_images_active = bool(self.config.allow_images)
        providers = []
        for provider_conf in self.config.providers:
            if not getattr(provider_conf, 'enabled', True):
                continue
            supports_images = getattr(provider_conf, 'supports_images', False)
            supports_videos = getattr(provider_conf, 'supports_videos', True)
            if supports_videos and not self.config.allow_videos:
                continue
            if supports_images and not self.config.allow_images:
                continue
            providers.append(provider_conf)
        ready_providers = []
        attempted_names: set[str] = set()
        for provider_conf in providers:
            requires_key, has_key = self._provider_key_status(getattr(provider_conf, 'name', ''))
            provider_name = str(getattr(provider_conf, 'name', '') or '').strip()
            if requires_key and not has_key:
                self._log_event({
                    'event': 'provider_skipped_missing_key',
                    'provider': getattr(provider_conf, 'name', ''),
                })
                continue
            if provider_name:
                attempted_names.add(provider_name.lower())
            ready_providers.append(provider_conf)
        providers = ready_providers
        raw_results: List[RemoteAssetCandidate] = []

        if providers:
            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.parallel_requests or 1) as pool:
                futures = []
                for provider_conf in providers:
                    for query in queries:
                        futures.append(
                            pool.submit(self._run_provider_fetch, provider_conf, query, filters, segment_timeout_s)
                        )

                timeout_s = max(self.config.request_timeout_s, 0.1)
                deadline = start + timeout_s
                pending = set(futures)
                while pending and len(raw_results) < self.config.per_segment_limit:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    done, pending = concurrent.futures.wait(
                        pending,
                        timeout=remaining,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    if not done:
                        break
                    for fut in done:
                        try:
                            candidates = fut.result() or []
                        except concurrent.futures.TimeoutError:
                            continue
                        except Exception:
                            continue
                        if not candidates:
                            continue
                        for candidate in candidates:
                            raw_results.append(candidate)
                            if len(raw_results) >= self.config.per_segment_limit:
                                break
                for fut in pending:
                    fut.cancel()
        else:
            fallback_candidates = self._run_pixabay_fallback(
                queries,
                self.config.per_segment_limit,
                segment_index=segment_index,
            )
            if fallback_candidates:
                allow_images_active = True
                raw_results.extend(fallback_candidates)

        if not raw_results and "pixabay" not in attempted_names:
            fallback_candidates = self._run_pixabay_fallback(
                queries,
                self.config.per_segment_limit,
                segment_index=segment_index,
            )
            if fallback_candidates:
                allow_images_active = True
                raw_results.extend(fallback_candidates)

        processed: List[RemoteAssetCandidate] = []
        video_extensions = (
            ".mp4",
            ".mov",
            ".webm",
            ".mkv",
            ".m4v",
            ".avi",
        )

        for candidate in raw_results:
            url = str(getattr(candidate, "url", "") or "").strip()
            if not url:
                continue
            normalized_url = url.lower()
            has_video_extension = normalized_url.endswith(video_extensions)
            duration = getattr(candidate, "duration", None)
            has_positive_duration = False
            if isinstance(duration, (int, float)):
                try:
                    has_positive_duration = float(duration) > 0
                except Exception:
                    has_positive_duration = False

            is_video = has_video_extension or has_positive_duration
            if is_video and not self.config.allow_videos:
                continue
            if not is_video and not allow_images_active:
                continue
            processed.append(candidate)

            if len(processed) >= self.config.per_segment_limit:
                break

        provider_counts: Dict[str, int] = {}
        for provider_conf in providers:
            name = str(getattr(provider_conf, "name", "") or "").strip()
            if name:
                provider_counts.setdefault(name, 0)

        for candidate in processed:
            provider = str(getattr(candidate, "provider", "") or "").strip()
            if not provider:
                provider = "unknown"
            provider_counts[provider] = provider_counts.get(provider, 0) + 1

        for provider_name, count in provider_counts.items():
            if not provider_name:
                continue
            payload: Dict[str, Any] = {
                "event": "broll_candidate_evaluated",
                "provider": provider_name,
                "count": int(count),
            }
            if segment_index is not None:
                try:
                    segment_value = int(segment_index)
                except Exception:
                    segment_value = segment_index
                payload["segment_index"] = segment_value
                payload.setdefault("segment", segment_value)
            self._log_event(payload)

        return processed

    # ------------------------------------------------------------------
    # Provider fetch helpers
    # ------------------------------------------------------------------
    def _provider_key_status(self, provider_name: str) -> Tuple[bool, bool]:
        name = (provider_name or '').strip().lower()
        env_map = {
            'pexels': 'PEXELS_API_KEY',
            'pixabay': 'PIXABAY_API_KEY',
        }
        env_key = env_map.get(name)
        if not env_key:
            return False, True

        candidate = None
        try:
            config_cls = getattr(__import__('config', fromlist=['Config']), 'Config')
            candidate = getattr(config_cls, env_key, None)
        except Exception:
            candidate = None

        candidate = candidate or os.environ.get(env_key)
        if candidate is None:
            return True, False

        cleaned = str(candidate).strip()
        return True, bool(cleaned)

    def _run_provider_fetch(
        self,
        provider_conf,
        query: str,
        filters: Optional[dict],
        segment_timeout: Optional[float],
    ) -> List[RemoteAssetCandidate]:
        if not query or len(query.strip()) < 3:
            return []
        name = provider_conf.name.lower()
        limit = max(1, provider_conf.max_results or self.config.per_segment_limit)
        timeout = provider_conf.timeout_s or self.config.request_timeout_s
        timeout = max(timeout, 0.1)
        if segment_timeout and segment_timeout > 0:
            timeout = min(timeout, segment_timeout)
        attempts = max(1, self.config.retry_count)

        def _dispatch() -> List[RemoteAssetCandidate]:
            if name == 'pexels':
                return self._fetch_from_pexels(query, limit)
            if name == 'pixabay':
                return self._fetch_from_pixabay(query, limit)
            return []

        for attempt in range(attempts):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as gate:
                future = gate.submit(_dispatch)
                try:
                    return future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    self._log_event({
                        'event': 'fetch_timeout',
                        'provider': name,
                        'query': query,
                        'timeout_s': timeout,
                        'attempt': attempt + 1,
                    })
                    continue
                except Exception as exc:
                    if attempt == attempts - 1:
                        self._log_event({
                            'event': 'fetch_error',
                            'provider': name,
                            'query': query,
                            'error': str(exc),
                        })
                        return []
        return []

    def _run_pixabay_fallback(
        self,
        queries: Sequence[str],
        limit: int,
        *,
        segment_index: Optional[int] = None,
    ) -> List[RemoteAssetCandidate]:
        fallback_candidates: List[RemoteAssetCandidate] = []
        attempted_queries: List[str] = []
        for query in queries[:3]:
            clean_query = (query or "").strip()
            if not clean_query:
                continue
            attempted_queries.append(clean_query)
            try:
                candidates = self._fetch_from_pixabay(clean_query, limit)
            except Exception:
                candidates = []
            if candidates:
                fallback_candidates.extend(candidates)
            if len(fallback_candidates) >= limit:
                break

        payload: Dict[str, Any] = {
            'event': 'fallback_fetch',
            'provider': 'pixabay',
            'queries': attempted_queries,
            'added_candidates': len(fallback_candidates),
        }
        if segment_index is not None:
            try:
                payload['segment_index'] = int(segment_index)
            except Exception:
                payload['segment_index'] = segment_index
        self._log_event(payload)
        return fallback_candidates


    def _fetch_from_pexels(self, query: str, limit: int) -> List[RemoteAssetCandidate]:
        if not query or len(query) < 3:
            return []
        key = getattr(__import__('config', fromlist=['Config']).Config, 'PEXELS_API_KEY', None)
        if not key:
            return []
        start_time = time.perf_counter()
        try:
            print(f"[FETCH] provider=pexels query='{query}' per_page={limit * 2}")
            videos = pexels_search_videos(key, query, per_page=limit * 2)
        except Exception as exc:
            self._log_event(
                {
                    'event': 'fetch_error',
                    'provider': 'pexels',
                    'query': query,
                    'error': str(exc),
                }
            )
            return []
        candidates: List[RemoteAssetCandidate] = []
        for item in videos:
            best = _best_vertical_video_file(item)
            if not best:
                continue
            url = best.get('link') or best.get('file')
            if not url:
                continue
            candidates.append(
                RemoteAssetCandidate(
                    provider='pexels',
                    url=url,
                    thumb_url=(item.get('image') or None),
                    width=int(best.get('width', 0) or 0),
                    height=int(best.get('height', 0) or 0),
                    duration=float(item.get('duration', 0.0) or 0.0),
                    title=str(item.get('user', {}).get('name', '')),
                    identifier=str(item.get('id', '')),
                    tags=[t.get('title', '') for t in item.get('tags', []) if t.get('title')],
                )
            )
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        self._log_event(
            {
                'event': 'fetch_request',
                'provider': 'pexels',
                'query': query,
                'latency_ms': latency_ms,
                'count': len(candidates),
            }
        )
        return candidates


    def _fetch_from_pixabay(self, query: str, limit: int) -> List[RemoteAssetCandidate]:
        if not query or len(query) < 3:
            return []
        key = getattr(__import__('config', fromlist=['Config']).Config, 'PIXABAY_API_KEY', None)
        if not key:
            return []
        start_time = time.perf_counter()
        try:
            print(f"[FETCH] provider=pixabay query='{query}' per_page={limit * 2}")
            videos = pixabay_search_videos(key, query, per_page=limit * 2)
        except Exception as exc:
            self._log_event(
                {
                    'event': 'fetch_error',
                    'provider': 'pixabay',
                    'query': query,
                    'error': str(exc),
                }
            )
            return []
        candidates: List[RemoteAssetCandidate] = []
        for item in videos:
            url = _pixabay_best_video_url(item)
            if not url:
                continue
            videos_meta = item.get('videos', {}) or {}
            best_meta = videos_meta.get('large') or videos_meta.get('medium') or {}
            candidates.append(
                RemoteAssetCandidate(
                    provider='pixabay',
                    url=url,
                    thumb_url=item.get('picture_id'),
                    width=int(best_meta.get('width', 0) or 0),
                    height=int(best_meta.get('height', 0) or 0),
                    duration=None,
                    title=item.get('user', ''),
                    identifier=str(item.get('id', '')),
                    tags=[tag.strip() for tag in (item.get('tags', '') or '').split(',') if tag.strip()],
                )
            )
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        self._log_event(
            {
                'event': 'fetch_request',
                'provider': 'pixabay',
                'query': query,
                'latency_ms': latency_ms,
                'count': len(candidates),
            }
        )
        return candidates

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _build_queries(self, keywords: Sequence[str]) -> List[str]:
        normalized = [kw.strip() for kw in keywords if isinstance(kw, str) and kw.strip()]
        if not normalized:
            return []
        primary = build_search_query(normalized)
        queries = [primary] if primary else []
        # Add up to 3 secondary queries to diversify results
        for kw in normalized:
            if kw not in queries:
                queries.append(kw)
            if len(queries) >= 4:
                break
        return queries

    def evaluate_candidate_filters(
        self,
        candidate: RemoteAssetCandidate,
        filters: Optional[dict],
        duration_hint: Optional[float],
    ) -> Tuple[bool, Optional[str]]:
        if not getattr(candidate, 'url', None):
            return False, 'missing_url'

        filters = filters or {}
        orientation = filters.get('orientation') if isinstance(filters, dict) else None
        min_duration = filters.get('min_duration_s') if isinstance(filters, dict) else None

        width = getattr(candidate, 'width', 0) or 0
        height = getattr(candidate, 'height', 0) or 0
        if orientation == 'landscape' and width and height and width < height:
            return False, 'filter_orientation'
        if orientation == 'portrait' and width and height and height < width:
            return False, 'filter_orientation'

        target_min = None
        if isinstance(min_duration, (int, float)):
            target_min = float(min_duration)
        elif duration_hint is not None:
            try:
                target_min = max(0.0, float(duration_hint))
            except Exception:
                target_min = None

        duration = getattr(candidate, 'duration', None)
        if target_min and isinstance(duration, (int, float)) and float(duration) < target_min:
            return False, 'filter_duration'

        return True, None

    def _log_event(self, payload: Dict[str, Any]) -> None:
        _emit_event(payload)
        if self._event_logger:
            try:
                self._event_logger.log(payload)
            except Exception:
                self._logger.debug('[fetcher] failed to log event', exc_info=True)
        else:
            self._logger.debug('[fetcher] %s', payload)


_EVENT_LOG_LOCK = threading.Lock()


def _events_path() -> Path:
    base_dir: Optional[Path]
    try:
        config_module = __import__('config', fromlist=['Config'])
        raw = getattr(config_module.Config, 'OUTPUT_FOLDER', None)
        base_dir = Path(raw) if raw else None
    except Exception:
        base_dir = None

    if not base_dir:
        base_dir = Path('output')
    return base_dir / 'meta' / 'broll_pipeline_events.jsonl'


def _emit_event(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        record: Dict[str, Any] = {'event': 'unknown', 'payload': payload}
    else:
        record = dict(payload)

    try:
        destination = _events_path()
        destination.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(record, ensure_ascii=False)
    except Exception:
        logging.getLogger(__name__).debug('[fetcher] failed to prepare event payload', exc_info=True)
        return

    try:
        with _EVENT_LOG_LOCK:
            with destination.open('a', encoding='utf-8') as handle:
                handle.write(serialized + "\n")
    except Exception:
        logging.getLogger(__name__).debug('[fetcher] failed to emit event', exc_info=True)
