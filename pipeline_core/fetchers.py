"""Fetching orchestration helpers."""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import threading
import time
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pipeline_core.configuration import FetcherOrchestratorConfig

if os.environ.get('BROLL_FORCE_IPV4', '0') == '1':
    try:
        import socket
        import urllib3.util.connection as urllib3_cn

        def _ipv4_only() -> int:
            return socket.AF_INET

        urllib3_cn.allowed_gai_family = _ipv4_only
    except Exception:
        pass


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

    def _safe_request_json(url: str, *, headers: Optional[Dict[str, str]] = None, timeout: float = 10.0) -> Dict[str, Any]:
        if not url:
            return {}
        request_headers: Dict[str, str] = {}
        if headers:
            request_headers.update({str(k): str(v) for k, v in headers.items() if k and v})
        request_headers.setdefault('User-Agent', 'video-pipeline/1.0')
        try:
            request = urllib.request.Request(url, headers=request_headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, ValueError):
            return {}

    def pexels_search_videos(api_key: str, query: str, *, per_page: int = 6) -> List[Dict[str, Any]]:
        api_key = (api_key or "").strip()
        query = (query or "").strip()
        if not api_key or not query:
            return []
        try:
            limit = int(per_page)
        except (TypeError, ValueError):
            limit = 6
        limit = max(1, min(limit, 80))
        params = urllib.parse.urlencode({"query": query, "per_page": limit})
        data = _safe_request_json(
            f"https://api.pexels.com/videos/search?{params}",
            headers={"Authorization": api_key, "Accept": "application/json"},
        )
        videos = data.get("videos", []) if isinstance(data, dict) else []
        return videos if isinstance(videos, list) else []

    def pixabay_search_videos(api_key: str, query: str, *, per_page: int = 6) -> List[Dict[str, Any]]:
        api_key = (api_key or "").strip()
        query = (query or "").strip()
        if not api_key or not query:
            return []
        try:
            limit = int(per_page)
        except (TypeError, ValueError):
            limit = 6
        limit = max(3, min(limit, 200))
        params = urllib.parse.urlencode(
            {
                "key": api_key,
                "q": query,
                "per_page": limit,
                "safesearch": "true",
                "video_type": "all",
            }
        )
        data = _safe_request_json(f"https://pixabay.com/api/videos/?{params}")
        hits = data.get("hits", []) if isinstance(data, dict) else []
        return hits if isinstance(hits, list) else []

    def _best_vertical_video_file(payload: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return None
        files = payload.get("video_files") or []
        if not isinstance(files, list):
            return None
        best_candidate: Optional[Dict[str, Any]] = None
        best_score: Tuple[int, int, int] = (-1, -1, -1)
        for entry in files:
            if not isinstance(entry, dict):
                continue
            url = entry.get("link") or entry.get("file")
            if not url:
                continue
            try:
                width = int(entry.get("width", 0) or 0)
                height = int(entry.get("height", 0) or 0)
            except (TypeError, ValueError):
                width = 0
                height = 0
            vertical = 1 if height >= width and height > 0 and width > 0 else 0
            area = width * height
            score: Tuple[int, int, int] = (vertical, area, height)
            if score > best_score:
                best_score = score
                best_candidate = entry
        return best_candidate

    def _pixabay_best_video_url(payload: Any) -> Optional[str]:
        if not isinstance(payload, dict):
            return None
        videos_meta = payload.get("videos")
        if not isinstance(videos_meta, dict):
            return None
        best_url: Optional[str] = None
        best_score: Tuple[int, int, int] = (-1, -1, -1)
        for meta in videos_meta.values():
            if not isinstance(meta, dict):
                continue
            url = meta.get("url")
            if not url:
                continue
            try:
                width = int(meta.get("width", 0) or 0)
                height = int(meta.get("height", 0) or 0)
            except (TypeError, ValueError):
                width = 0
                height = 0
            vertical = 1 if height >= width and height > 0 and width > 0 else 0
            area = width * height
            score: Tuple[int, int, int] = (vertical, area, height)
            if score > best_score:
                best_score = score
                best_url = str(url)
        return best_url


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
    _phash: Optional[str] = field(default=None, init=False, repr=False)


def crop_viability_9_16(width: Any, height: Any) -> bool:
    """Return True when a vertical 9:16 crop is at least theoretically possible."""
    try:
        w = int(width or 0)
        h = int(height or 0)
    except (TypeError, ValueError):
        return False
    return w >= 2 and h >= 2


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
        want_images = bool(self.config.allow_images)
        want_videos = bool(self.config.allow_videos)
        requested_any = want_images or want_videos
        providers = []
        for provider_conf in self.config.providers:
            if not getattr(provider_conf, 'enabled', True):
                continue
            supports_images = getattr(provider_conf, 'supports_images', False)
            supports_videos = getattr(provider_conf, 'supports_videos', True)
            provider_name = str(getattr(provider_conf, 'name', '') or '').strip()
            provider_name_lc = provider_name.lower()
            if not want_videos and provider_name_lc in {'pexels', 'pixabay'}:
                continue

            supports_requested = False
            if want_images and supports_images:
                supports_requested = True
            if want_videos and supports_videos:
                supports_requested = True

            if not supports_requested:
                if requested_any:
                    self._log_event({
                        'event': 'provider_skipped_incapable',
                        'provider': getattr(provider_conf, 'name', ''),
                        'allow_images': self.config.allow_images,
                        'allow_videos': self.config.allow_videos,
                        'supports_images': supports_images,
                        'supports_videos': supports_videos,
                    })
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
                    if (
                        not self.config.allow_videos
                        and (
                            not getattr(provider_conf, 'supports_images', False)
                            or str(getattr(provider_conf, 'name', '') or '')
                            .strip()
                            .lower()
                            in {'pexels', 'pixabay'}
                        )
                    ):
                        continue
                    for query in queries:
                        futures.append(
                            pool.submit(self._run_provider_fetch, provider_conf, query, filters, segment_timeout_s)
                        )

                timeout_s = max(self.config.request_timeout_s, 0.1)
                deadline = start + timeout_s
                pending = set(futures)
                max_results_budget = self.config.per_segment_limit * max(1, len(providers))
                seen_providers: set[str] = set()
                while pending:
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
                            provider_token = str(getattr(candidate, "provider", "") or "").strip().lower()
                            if provider_token:
                                seen_providers.add(provider_token)
                        if len(raw_results) >= max_results_budget and (len(seen_providers) >= len(providers) or not pending):
                            break
                    if len(raw_results) >= max_results_budget and (len(seen_providers) >= len(providers) or not pending):
                        break
                for fut in pending:
                    fut.cancel()
        elif self.config.allow_videos:
            fallback_candidates = self._run_pixabay_fallback(
                queries,
                self.config.per_segment_limit,
                segment_index=segment_index,
            )
            if fallback_candidates:
                allow_images_active = True
                raw_results.extend(fallback_candidates)

        if self.config.allow_videos and not raw_results and "pixabay" not in attempted_names:
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
        # Compute provider counts BEFORE filtering for telemetry
        provider_counts_before: Dict[str, int] = {}
        for cand in raw_results:
            name = str(getattr(cand, "provider", "") or "").strip() or "unknown"
            provider_counts_before[name] = provider_counts_before.get(name, 0) + 1

        grouped_results: Dict[str, List[RemoteAssetCandidate]] = {}
        provider_order: List[str] = []

        # Apply per-candidate filters (orientation, duration, etc.) with a relaxed second pass if yield is poor
        strict_pass: List[RemoteAssetCandidate] = []
        relaxed_reasons: Dict[str, int] = {}

        def _relax_keep(c: RemoteAssetCandidate, why: Optional[str]) -> bool:
            # Allow near-portrait AR >= ~1.55 or short duration tolerance
            if why not in {"filter_orientation", "filter_duration"}:
                return False
            try:
                w = int(getattr(c, 'width', 0) or 0)
                h = int(getattr(c, 'height', 0) or 0)
                dur = float(getattr(c, 'duration', 0.0) or 0.0)
            except Exception:
                w = h = 0
                dur = 0.0
            if why == "filter_orientation":
                ar = (float(h) / float(w)) if w > 0 else 0.0
                return ar >= 1.55
            if why == "filter_duration":
                return False
            return False

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

            # Strict filter evaluation
            ok, why = self.evaluate_candidate_filters(candidate, filters, duration_hint)
            if ok:
                strict_pass.append(candidate)
            else:
                if why:
                    relaxed_reasons[why] = relaxed_reasons.get(why, 0) + 1

        # If too few after strict filters, do a relaxed pass on remaining items
        target_min = max(1, int(self.config.per_segment_limit // 2))
        selected: List[RemoteAssetCandidate] = list(strict_pass)
        if len(selected) < target_min:
            for candidate in raw_results:
                if candidate in selected:
                    continue
                ok, why = self.evaluate_candidate_filters(candidate, filters, duration_hint)
                if not ok and _relax_keep(candidate, why):
                    selected.append(candidate)
                if len(selected) >= self.config.per_segment_limit:
                    break

        # Telemetry for filter stats
        try:
            self._log_event({
                'event': 'broll_filter_stats',
                'raw_total': len(raw_results),
                'filtered_total': len(selected),
                'filter_reasons': relaxed_reasons,
                'provider_counts_before': provider_counts_before,
                'allow_videos': bool(self.config.allow_videos),
                'allow_images': bool(self.config.allow_images),
            })
        except Exception:
            pass

        # Group by provider then round-robin from the filtered selection
        for candidate in selected:
            provider_name = str(getattr(candidate, "provider", "") or "").strip() or "unknown"
            if provider_name not in provider_order:
                provider_order.append(provider_name)
            grouped_results.setdefault(provider_name, []).append(candidate)

        per_segment_limit = self.config.per_segment_limit
        active_providers = [name for name in provider_order if grouped_results.get(name)]

        while active_providers and len(processed) < per_segment_limit:
            next_cycle: List[str] = []
            for provider_name in active_providers:
                bucket = grouped_results.get(provider_name)
                if not bucket:
                    continue
                candidate = bucket.pop(0)
                processed.append(candidate)
                if bucket:
                    next_cycle.append(provider_name)
                if len(processed) >= per_segment_limit:
                    break
            if len(processed) >= per_segment_limit:
                break
            active_providers = next_cycle

        # Lightweight re-ranking favouring portrait framing and segment-friendly durations
        segment_length_s: Optional[float] = None
        if isinstance(duration_hint, (int, float)):
            try:
                segment_length_s = float(duration_hint)
            except (TypeError, ValueError):
                segment_length_s = None

        def _duration_acceptance_score(seg_len: Optional[float], cand_duration: Optional[float], target_ratio: float = 1.0) -> float:
            if not seg_len or not cand_duration or cand_duration <= 0:
                return 0.0
            diff = abs((cand_duration / float(seg_len)) - target_ratio)
            return max(0.0, 1.0 - (diff / 0.4))

        scored: List[Tuple[float, float, float, float, RemoteAssetCandidate]] = []
        for candidate in processed:
            try:
                cand_duration = float(getattr(candidate, 'duration', 0.0) or 0.0)
            except (TypeError, ValueError):
                cand_duration = 0.0
            duration_score = _duration_acceptance_score(segment_length_s, cand_duration)
            try:
                width = int(getattr(candidate, 'width', 0) or 0)
                height = int(getattr(candidate, 'height', 0) or 0)
            except (TypeError, ValueError):
                width = 0
                height = 0
            portrait_bonus = 1.0 if (height and width and height > width) else 0.0
            try:
                provider_score = float(getattr(candidate, 'provider_score', 0.0) or 0.0)
            except (TypeError, ValueError):
                provider_score = 0.0
            total_score = (duration_score * 0.7) + (portrait_bonus * 0.2) + (provider_score * 0.1)
            scored.append((total_score, duration_score, portrait_bonus, provider_score, candidate))

        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            processed = [entry[4] for entry in scored]

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
        name = str(getattr(provider_conf, 'name', '') or '').strip().lower()
        if not self.config.allow_videos and name in {'pexels', 'pixabay'}:
            return []
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
        key = os.getenv('PEXELS_API_KEY')
        if not key:
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
        banned = {"watermark", "logo", "overlay", "text", "tiktok", "repost", "subscribe", "text overlay", "text-overlay"}
        for item in videos:
            best = _best_vertical_video_file(item)
            if not best:
                continue
            url = best.get('link') or best.get('file')
            if not url:
                continue
            try:
                tag_list = [t.get('title','').strip().lower() for t in (item.get('tags') or []) if isinstance(t, dict)]
            except Exception:
                tag_list = []
            if any(tag in banned for tag in tag_list):
                continue
            text_sources = [
                item.get('description'),
                item.get('alt'),
                item.get('user', {}).get('name'),
            ]
            metadata_blob = " ".join(
                str(value).strip().lower()
                for value in text_sources
                if isinstance(value, str) and value.strip()
            )
            if metadata_blob:
                tokens = {token for token in metadata_blob.replace('-', ' ').replace('_', ' ').split() if token}
                if tokens.intersection({'watermark', 'subscribe', 'logo'}) or 'text overlay' in metadata_blob or 'text-overlay' in metadata_blob:
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
        key = os.getenv('PIXABAY_API_KEY')
        if not key:
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
        banned = {"watermark", "logo", "overlay", "text", "tiktok", "repost", "subscribe", "text overlay", "text-overlay"}
        for item in videos:
            url = _pixabay_best_video_url(item)
            if not url:
                continue
            videos_meta = item.get('videos', {}) or {}
            best_meta = videos_meta.get('large') or videos_meta.get('medium') or {}
            tags_raw = str(item.get('tags','') or '').lower()
            if any(b in tags_raw for b in banned):
                continue
            metadata_blob = " ".join(
                str(value).strip().lower()
                for value in (
                    item.get('user'),
                    item.get('pageURL'),
                    item.get('picture_id'),
                )
                if isinstance(value, str) and value.strip()
            )
            if metadata_blob:
                tokens = {token for token in metadata_blob.replace('-', ' ').replace('_', ' ').split() if token}
                if tokens.intersection({'watermark', 'subscribe', 'logo'}) or 'text overlay' in metadata_blob or 'text-overlay' in metadata_blob:
                    continue
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

        try:
            width = int(getattr(candidate, 'width', 0) or 0)
            height = int(getattr(candidate, 'height', 0) or 0)
        except (TypeError, ValueError):
            width = 0
            height = 0

        if orientation == 'landscape' and width and height and width < height:
            return False, 'filter_orientation'
        if orientation == 'portrait' and width and height and height <= width:
            return False, 'filter_orientation'

        duration_value = getattr(candidate, 'duration', None)
        if duration_value is None:
            return False, 'filter_duration'
        try:
            duration_f = float(duration_value)
        except (TypeError, ValueError):
            return False, 'filter_duration'

        if duration_f <= 0.0:
            return False, 'filter_duration'
        if duration_f < 0.8 or duration_f > 30.0:
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
















