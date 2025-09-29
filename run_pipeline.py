#!/usr/bin/env python3
"""Stable wrapper around video_processor with sane environment defaults."""
from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('PYTHONUTF8', '1')
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        try:
            stream.reconfigure(encoding='utf-8')
        except Exception:
            pass

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - keep working if optional dependency missing
    def load_dotenv(*_args, **_kwargs):
        return False

from config import Config
from pipeline_core.configuration import FetcherOrchestratorConfig, resolved_providers
from pipeline_core.fetchers import FetcherOrchestrator
from pipeline_core.logging import JsonlLogger
from pipeline_core.runtime import PipelineResult

# Expose the video_processor module at import time for compatibility with tests.
# When the heavy dependencies of video_processor (e.g., cv2) are unavailable at
# import time, fall back to a simple namespace so test suites can monkeypatch it.
try:  # pragma: no cover - depends on optional system libraries
    video_processor = importlib.import_module("video_processor")
except Exception:  # pragma: no cover - triggered in test environments
    from types import SimpleNamespace

    video_processor = SimpleNamespace(main=None)


def _compute_pythonpath(repo_root: Path) -> str:
    extra = [str(repo_root), str(repo_root / "AI-B-roll"), str(repo_root / "utils")]
    current = os.environ.get("PYTHONPATH")
    if current:
        extra.append(current)
    return os.pathsep.join(extra)


def _result_to_exit_code(result: PipelineResult) -> int:
    final_ok = bool(result.final_export_ok)
    has_errors = bool(result.errors)
    if final_ok:
        return 0 if not has_errors else 2
    return 1


_SANITIZE_KEYS = (
    'PEXELS_API_KEY',
    'PIXABAY_API_KEY',
    'UNSPLASH_ACCESS_KEY',
    'GIPHY_API_KEY',
    'BROLL_FETCH_PROVIDER',
    'BROLL_FETCH_ENABLE',
    'BROLL_FETCH_ALLOW_VIDEOS',
    'BROLL_FETCH_ALLOW_IMAGES',
    'BROLL_FETCH_MAX_PER_KEYWORD',
    'ENABLE_PIPELINE_CORE_FETCHER',
)


def _clean_env_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()
    return cleaned or None


def _sanitize_env_values(keys: Sequence[str]) -> dict[str, Optional[str]]:
    sanitized: dict[str, Optional[str]] = {}
    for key in keys:
        raw = os.environ.get(key)
        cleaned = _clean_env_value(raw)
        if raw is None and cleaned is None:
            continue
        if cleaned is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = cleaned
        sanitized[key] = cleaned
    return sanitized


def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def _mask_api_key(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    tail = cleaned[-4:] if len(cleaned) >= 4 else cleaned
    return f'****{tail}'


class _DiagEventLogger:
    """Proxy logger that tees events to disk and keeps them in memory."""

    def __init__(self, destination: Path) -> None:
        self._jsonl = JsonlLogger(destination)
        self.entries: List[Dict[str, Any]] = []

    @property
    def path(self) -> Path:
        return self._jsonl.path

    def log(self, payload: Dict[str, Any]) -> None:
        if isinstance(payload, dict):
            self.entries.append(dict(payload))
        else:
            self.entries.append({"event": "unknown", "payload": payload})
        self._jsonl.log(payload)


_DIAG_EVENT_LOGGER: Optional[_DiagEventLogger] = None


def _broll_events_path() -> Path:
    try:
        base_dir = Path(getattr(Config, "OUTPUT_FOLDER", Path("output")))
    except Exception:
        base_dir = Path("output")
    return base_dir / "meta" / "broll_pipeline_events.jsonl"


def _get_diag_event_logger() -> _DiagEventLogger:
    global _DIAG_EVENT_LOGGER
    if _DIAG_EVENT_LOGGER is None:
        _DIAG_EVENT_LOGGER = _DiagEventLogger(_broll_events_path())
    return _DIAG_EVENT_LOGGER


def _run_broll_diagnostic(repo_root: Path) -> int:
    try:
        import json
    except ImportError as exc:
        print(f"[DIAG] missing stdlib dependency: {exc}", file=sys.stderr)
        return 1

    provider_defs = [
        ('pexels', 'PEXELS_API_KEY'),
        ('pixabay', 'PIXABAY_API_KEY'),
    ]

    providers_meta = []
    for name, env_key in provider_defs:
        raw = os.environ.get(env_key)
        providers_meta.append({
            'name': name,
            'env_key': env_key,
            'key_present': bool(raw),
            'masked_key': _mask_api_key(raw),
        })

    event_logger = _get_diag_event_logger()
    config = FetcherOrchestratorConfig.from_environment()
    provider_configs = {provider.name.lower(): provider for provider in config.providers}
    resolved_names = resolved_providers(config)
    active_names = [provider.name for provider in config.providers if provider.enabled]
    per_segment_limit = int(config.per_segment_limit)
    allow_images = bool(config.allow_images)

    for meta in providers_meta:
        provider_cfg = provider_configs.get(meta['name'].lower())
        meta['selected'] = provider_cfg is not None
        meta['enabled'] = bool(provider_cfg.enabled) if provider_cfg is not None else False
        meta['max_results'] = int(provider_cfg.max_results) if provider_cfg is not None else None

    providers_display = ','.join(active_names) if active_names else ','.join(resolved_names)
    resolved_display = ','.join(resolved_names) if resolved_names else 'none'
    print(
        "[DIAG] providers="
        f"{providers_display} | resolved_providers={resolved_display} | allow_images={str(allow_images).lower()} | "
        f"per_segment_limit={per_segment_limit}"
    )

    orchestrator = FetcherOrchestrator(config, event_logger=event_logger)

    for meta in providers_meta:
        if not meta['key_present']:
            event_logger.log({'event': 'provider_skipped_missing_key', 'provider': meta['name']})

    base_index = len(event_logger.entries)
    try:
        candidates = orchestrator.fetch_candidates(['nature'], segment_index=0, duration_hint=6.0, segment_timeout_s=0.7)
    except Exception as exc:  # pragma: no cover - unexpected runtime faults
        print(f"[DIAG] fetch orchestrator failed: {exc}", file=sys.stderr)
        candidates = []

    new_events = event_logger.entries[base_index:]
    provider_counts: Dict[str, int] = defaultdict(int)
    for candidate in candidates:
        provider = str(getattr(candidate, 'provider', '') or '').strip() or 'unknown'
        provider_counts[provider] += 1

    latency_by_provider: Dict[str, Optional[int]] = {}
    error_by_provider: Dict[str, str] = {}

    for event in new_events:
        provider = str(event.get('provider', '') or '').strip()
        event_name = event.get('event')
        if not provider and event_name != 'broll_candidate_evaluated':
            continue
        if event_name == 'fetch_request':
            latency = event.get('latency_ms')
            if isinstance(latency, (int, float)):
                latency_by_provider[provider] = int(latency)
            count = event.get('count')
            if isinstance(count, int):
                provider_counts.setdefault(provider, 0)
                provider_counts[provider] = max(provider_counts[provider], int(count))
        elif event_name == 'fetch_timeout':
            error_by_provider[provider] = 'timeout'
        elif event_name == 'fetch_error':
            error_by_provider[provider] = str(event.get('error') or 'error')
        elif event_name == 'provider_skipped_missing_key':
            error_by_provider[provider] = 'missing_api_key'
        elif event_name == 'broll_candidate_evaluated':
            prov_name = str(event.get('provider', '') or '').strip()
            if not prov_name:
                continue
            count = event.get('count')
            if isinstance(count, int):
                provider_counts[prov_name] = int(count)
            if prov_name and provider not in latency_by_provider:
                latency_by_provider.setdefault(prov_name, None)

    results = []
    for provider in providers_meta:
        status = dict(provider)
        name = status['name']
        provider_cfg = provider_configs.get(name.lower())
        if not status['key_present']:
            status['success'] = False
            status['error'] = 'missing_api_key'
            print(f"[DIAG] provider={name} skipped (missing key)")
            results.append(status)
            continue

        if not status.get('selected'):
            status['success'] = False
            status['error'] = 'not_selected'
            print(f"[DIAG] provider={name} skipped (not selected)")
            results.append(status)
            continue

        if provider_cfg is not None and not provider_cfg.enabled:
            status['success'] = False
            status['error'] = 'disabled'
            print(f"[DIAG] provider={name} skipped (disabled)")
            results.append(status)
            continue

        count = provider_counts.get(name, 0)
        status['candidates'] = count
        status['latency_ms'] = latency_by_provider.get(name)
        status['success'] = count > 0
        if not status['success']:
            status['error'] = error_by_provider.get(name, 'no_results')

        latency_display = status.get('latency_ms')
        if latency_display is None:
            latency_display = 'n/a'
        print(
            f"[DIAG] provider={name} success={status.get('success')} latency_ms={latency_display} candidates={status.get('candidates', 0)}"
        )
        results.append(status)

    payload = {
        'timestamp': time.time(),
        'providers_actifs': active_names,
        'providers_resolved': resolved_names,
        'allow_images': allow_images,
        'per_segment_limit': per_segment_limit,
        'providers': results,
    }
    output_path = repo_root / 'diagnostic_broll.json'
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    print(f"[DIAG] report written to {output_path}")

    return 0 if any(item.get('success') for item in results) else 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parent

    load_dotenv(repo_root / '.env', override=False)
    load_dotenv(repo_root / '.env.local', override=True)
    _sanitize_env_values(_SANITIZE_KEYS)

    parser = argparse.ArgumentParser(
        description="Launch the video pipeline with stable environment defaults."
    )
    parser.add_argument("--video", help="Path to the source video (mp4, mov, etc.)")
    parser.add_argument("--diag-broll", action="store_true", help="Run a B-roll API diagnostic and exit.")
    parser.add_argument("--legacy", action="store_true", help="Disable the modern pipeline_core orchestrator.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose logs in the console.")
    parser.add_argument("--no-emoji", action="store_true", help="Disable emoji in console output.")
    args, passthrough = parser.parse_known_args(argv)

    if args.diag_broll:
        return _run_broll_diagnostic(repo_root)
    if not args.video:
        parser.error("--video is required unless --diag-broll is specified")

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ["ENABLE_PIPELINE_CORE_FETCHER"] = "false" if args.legacy else os.environ.get(
        "ENABLE_PIPELINE_CORE_FETCHER", "true"
    )
    os.environ["PYTHONPATH"] = _compute_pythonpath(repo_root)

    vp_args = ["--video", args.video]
    if args.verbose:
        vp_args.append("--verbose")
    if args.no_emoji:
        vp_args.append("--no-emoji")
    vp_args.extend(passthrough)

    global video_processor
    # Prefer an already imported/stubbed module so monkeypatched objects remain intact.
    module = globals().get("video_processor")
    if module is None:
        module = sys.modules.get("video_processor")
        if module is not None:
            globals()["video_processor"] = module

    main_attr = getattr(module, "main", None) if module is not None else None
    if callable(main_attr):
        video_processor_module = module  # type: ignore[assignment]
    else:
        needs_reload = module is not None and getattr(module, "__spec__", None) is not None
        if needs_reload:
            try:
                module = importlib.reload(module)  # type: ignore[arg-type]
            except Exception:
                module = importlib.import_module("video_processor")
        else:
            module = importlib.import_module("video_processor")
        globals()["video_processor"] = module
        video_processor_module = module

    # Bridge compatible : nouvelle API (retourne PipelineResult) ou legacy (retourne int)
    try:
        sig = inspect.signature(video_processor_module.main)
        supports_return = 'return_result' in sig.parameters
    except Exception:
        supports_return = False

    if supports_return:
        ret = video_processor_module.main(vp_args, return_result=True)
        if isinstance(ret, PipelineResult):
            return _result_to_exit_code(ret)
        if isinstance(ret, int):
            return ret
        return 0 if ret else 1
    else:
        code = video_processor_module.main(vp_args)
        return int(code) if code is not None else 0



if __name__ == "__main__":
    raise SystemExit(main())
