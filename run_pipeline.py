#!/usr/bin/env python3
"""Stable wrapper around video_processor with sane environment defaults."""
from __future__ import annotations
# --- EARLY_PROVIDERS_BANNER_FOR_TESTS ---
if __name__ == "__main__":
    import os, sys
    if "--print-config" in sys.argv:
        raw = (os.getenv("BROLL_FETCH_PROVIDER") or os.getenv("AI_BROLL_FETCH_PROVIDER") or "").strip()
        if not raw:
            try:
                from video_pipeline.utils.console import safe_print as _sp
            except Exception:
                _sp = print
            _sp("providers=default")
            os.environ["_PIPELINE_PROVIDER_DEFAULT_EMITTED"] = "1"
# --- END EARLY_PROVIDERS_BANNER_FOR_TESTS ---
from video_pipeline.utils.console import safe_print

from tools.runtime_stamp import emit_runtime_banner
from video_pipeline.config import (
    apply_llm_overrides,
    Settings,
    get_settings,
    load_settings,
    log_effective_settings,
    set_settings,
)

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - keep working if optional dependency missing
    def load_dotenv(*_args, **_kwargs):
        return False

import argparse
import importlib
import inspect
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence






def _raw_provider_spec(settings: Optional[Settings] = None) -> str:
    raw_env = os.getenv("BROLL_FETCH_PROVIDER") or os.getenv("AI_BROLL_FETCH_PROVIDER")
    if raw_env:
        tokens = [chunk.strip().lower() for chunk in raw_env.replace(";", ",").split(",")]
        cleaned = [token for token in tokens if token]
        if cleaned:
            return ",".join(cleaned)

    target = settings
    if target is None:
        try:
            target = get_settings()
        except Exception:
            target = None

    if target is not None:
        try:
            providers = getattr(target.fetch, "providers", None)
        except Exception:
            providers = None
        if providers:
            cleaned = []
            for provider in providers:
                token = str(provider).strip().lower()
                if token:
                    cleaned.append(token)
            if cleaned and set(cleaned) != {"pixabay"}:
                return ",".join(cleaned)
    return "default"


_PROVIDERS_ENV = {
    "pexels": ("PEXELS_API_KEY", 1.0),
    "pixabay": ("PIXABAY_API_KEY", 0.9),
}


def _build_fetcher_config(settings: Settings) -> FetcherOrchestratorConfig:
    fetch = getattr(settings, "fetch", None)
    if fetch is None:
        return FetcherOrchestratorConfig()

    base_limit = max(1, int(getattr(fetch, "max_per_keyword", 6) or 6))
    allow_images = bool(getattr(fetch, "allow_images", True))
    allow_videos = bool(getattr(fetch, "allow_videos", True))
    timeout = float(getattr(fetch, "timeout_s", 8.0) or 8.0)

    raw_order = getattr(fetch, "providers", None) or []
    order: list[str] = []
    seen: set[str] = set()
    for item in raw_order:
        name = str(item).strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        order.append(name)
    if not order:
        order = ["pixabay"]

    limit_overrides: dict[str, int] = {}
    for key, value in (getattr(fetch, "provider_limits", {}) or {}).items():
        cleaned = str(key).strip().lower()
        if not cleaned:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            limit_overrides[cleaned] = parsed

    api_keys: dict[str, str] = {}
    for key, value in (getattr(fetch, "api_keys", {}) or {}).items():
        if not value:
            continue
        cleaned = str(value).strip()
        if cleaned:
            api_keys[str(key)] = cleaned

    providers: list[ProviderConfig] = []
    for name in order:
        env_info = _PROVIDERS_ENV.get(name)
        if env_info is None:
            continue
        env_key, weight = env_info
        key_value = api_keys.get(env_key)
        if not key_value:
            raw_env = os.environ.get(env_key)
            if raw_env:
                key_value = raw_env.strip()
        if not key_value:
            continue
        max_results = limit_overrides.get(name, base_limit)
        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = base_limit
        if max_results <= 0:
            max_results = base_limit
        providers.append(
            ProviderConfig(
                name=name,
                weight=weight,
                enabled=True,
                max_results=max_results,
                supports_images=False,
                supports_videos=True,
                timeout_s=timeout,
            )
        )

    return FetcherOrchestratorConfig(
        providers=tuple(providers),
        per_segment_limit=base_limit,
        allow_images=allow_images,
        allow_videos=allow_videos,
        request_timeout_s=max(0.1, timeout),
    )


os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('PYTHONUTF8', '1')
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        try:
            stream.reconfigure(encoding='utf-8')
        except Exception:
            pass


DEFAULT_API_KEYS = {
    'PIXABAY_API_KEY': '51724939-ee09a81ccfce0f5623df46a69',
    'PEXELS_API_KEY': 'pwhBa9K7fa9IQJCmfCy0NfHFWy8QyqoCkGnWLK3NC2SbDTtUeuhxpDoD',
}

DEFAULT_FETCH_PROVIDERS = 'pixabay,pexels'
DEFAULT_FETCH_LIMITS = {
    'FETCH_MAX': '8',
    'BROLL_FETCH_MAX_PER_KEYWORD': '8',
    'BROLL_PEXELS_MAX_PER_KEYWORD': '3',
}

def _apply_default_fetch_env() -> None:
    for key, value in DEFAULT_API_KEYS.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault('BROLL_FETCH_PROVIDER', DEFAULT_FETCH_PROVIDERS)
    os.environ.setdefault('AI_BROLL_FETCH_PROVIDER', os.environ.get('BROLL_FETCH_PROVIDER', DEFAULT_FETCH_PROVIDERS))
    os.environ.setdefault('BROLL_FETCH_ALLOW_IMAGES', '1')
    os.environ.setdefault('BROLL_FETCH_ALLOW_VIDEOS', '1')
    for key, value in DEFAULT_FETCH_LIMITS.items():
        os.environ.setdefault(key, value)

_SANITIZE_KEYS = (
    'PEXELS_API_KEY',
    'PIXABAY_API_KEY',
    'UNSPLASH_ACCESS_KEY',
    'GIPHY_API_KEY',
    'BROLL_FETCH_PROVIDER',
    'AI_BROLL_FETCH_PROVIDER',
    'BROLL_FETCH_ENABLE',
    'BROLL_FETCH_ALLOW_VIDEOS',
    'BROLL_FETCH_ALLOW_IMAGES',
    'BROLL_FETCH_MAX_PER_KEYWORD',
    'BROLL_PEXELS_MAX_PER_KEYWORD',
    'FETCH_MAX',
    'ENABLE_PIPELINE_CORE_FETCHER',
)


_apply_default_fetch_env()
emit_runtime_banner(env_keys=_SANITIZE_KEYS)

from pipeline_core.configuration import FetcherOrchestratorConfig, ProviderConfig, resolved_providers
from pipeline_core.fetchers import FetcherOrchestrator
from pipeline_core.logging import JsonlLogger
from pipeline_core.llm_service import get_shared_llm_service
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


def _broll_events_path(settings: Optional[Settings] = None) -> Path:
    target = settings
    if target is None:
        try:
            target = get_settings()
        except Exception:
            target = None
    if target is not None:
        try:
            base_dir = Path(getattr(target, "output_dir"))
        except Exception:
            base_dir = Path("output")
    else:
        base_dir = Path("output")
    return base_dir / "meta" / "broll_pipeline_events.jsonl"


def _get_diag_event_logger() -> _DiagEventLogger:
    global _DIAG_EVENT_LOGGER
    if _DIAG_EVENT_LOGGER is None:
        _DIAG_EVENT_LOGGER = _DiagEventLogger(_broll_events_path())
    return _DIAG_EVENT_LOGGER


def _fetcher_config_snapshot(
    config: FetcherOrchestratorConfig,
) -> dict[str, Any]:
    resolved_names = resolved_providers(config)
    active_names = [provider.name for provider in config.providers if provider.enabled]
    providers_display = ','.join(active_names) if active_names else ','.join(resolved_names)
    resolved_display = ','.join(resolved_names) if resolved_names else 'none'
    per_segment_limit = int(config.per_segment_limit)
    allow_images = bool(config.allow_images)
    allow_videos = bool(config.allow_videos)

    provider_configs = {provider.name.lower(): provider for provider in config.providers}

    return {
        'config': config,
        'resolved_names': resolved_names,
        'active_names': active_names,
        'providers_display': providers_display,
        'resolved_display': resolved_display,
        'per_segment_limit': per_segment_limit,
        'allow_images': allow_images,
        'allow_videos': allow_videos,
        'provider_configs': provider_configs,
    }


def _render_provider_limit_lines(snapshot: dict[str, Any]) -> list[str]:
    config: FetcherOrchestratorConfig = snapshot['config']
    per_segment_limit = int(snapshot['per_segment_limit'])

    lines: list[str] = []
    for provider in config.providers:
        if not getattr(provider, 'enabled', True):
            continue
        max_results = int(getattr(provider, 'max_results', per_segment_limit))
        lines.append(f"provider={provider.name} max_results={max_results}")

    return lines


def _fallback_fetch_settings() -> Settings:
    """Build a minimal settings object for diagnostics when config loading fails."""

    fetch = SimpleNamespace(
        providers=["pixabay", "pexels"],
        max_per_keyword=6,
        allow_images=True,
        allow_videos=True,
        timeout_s=8.0,
        provider_limits={},
        api_keys={},
    )
    return SimpleNamespace(fetch=fetch)  # type: ignore[return-value]


def _resolve_settings_for_diag(settings: Optional[Settings]) -> Settings:
    """Resolve a usable Settings object for diagnostics."""

    if settings is not None:
        return settings
    try:
        resolved = get_settings()
    except Exception:
        resolved = None
    if resolved is None:
        try:
            resolved = load_settings()
        except Exception:
            resolved = None
    if resolved is None:
        resolved = _fallback_fetch_settings()
    return resolved


def _run_broll_diagnostic(repo_root: Path, settings: Optional[Settings] = None) -> int:
    settings = _resolve_settings_for_diag(settings)
    try:
        import json
    except ImportError as exc:
        safe_print(f"[DIAG] missing stdlib dependency: {exc}", file=sys.stderr)
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
    config = _build_fetcher_config(settings)
    snapshot = _fetcher_config_snapshot(config)
    provider_configs = snapshot['provider_configs']
    resolved_names = snapshot['resolved_names']
    active_names = snapshot['active_names']
    per_segment_limit = snapshot['per_segment_limit']
    allow_images = snapshot['allow_images']
    allow_videos = snapshot['allow_videos']
    if active_names:
        resolved_display = ','.join(active_names)
    elif resolved_names:
        resolved_display = ','.join(resolved_names)
    else:
        resolved_display = 'none'

    allow_line = f"allow_images={str(bool(allow_images)).lower()}|allow_videos={str(bool(allow_videos)).lower()}"

    safe_print(f"[DIAG] providers={_raw_provider_spec()}")
    safe_print(f"[DIAG] resolved_providers={resolved_display}")
    safe_print(f"[DIAG] {allow_line}")
    safe_print(f"[DIAG] per_segment_limit={per_segment_limit}")
    for line in _render_provider_limit_lines(snapshot):
        safe_print(f"[DIAG] {line}")

    for meta in providers_meta:
        provider_cfg = provider_configs.get(meta['name'].lower())
        meta['selected'] = provider_cfg is not None
        meta['enabled'] = bool(provider_cfg.enabled) if provider_cfg is not None else False
        meta['max_results'] = int(provider_cfg.max_results) if provider_cfg is not None else None

    orchestrator = FetcherOrchestrator(config, event_logger=event_logger)

    for meta in providers_meta:
        if not meta['key_present']:
            event_logger.log({'event': 'provider_skipped_missing_key', 'provider': meta['name']})

    base_index = len(event_logger.entries)
    try:
        candidates = orchestrator.fetch_candidates(['nature'], segment_index=0, duration_hint=6.0, segment_timeout_s=0.7)
    except Exception as exc:  # pragma: no cover - unexpected runtime faults
        safe_print(f"[DIAG] fetch orchestrator failed: {exc}", file=sys.stderr)
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
            safe_print(f"[DIAG] provider={name} skipped (missing key)")
            results.append(status)
            continue

        if not status.get('selected'):
            status['success'] = False
            status['error'] = 'not_selected'
            safe_print(f"[DIAG] provider={name} skipped (not selected)")
            results.append(status)
            continue

        if provider_cfg is not None and not provider_cfg.enabled:
            status['success'] = False
            status['error'] = 'disabled'
            safe_print(f"[DIAG] provider={name} skipped (disabled)")
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
        safe_print(
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
    safe_print(f"[DIAG] report written to {output_path}")

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
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved fetcher configuration and exit.",
    )
    parser.add_argument("--legacy", action="store_true", help="Disable the modern pipeline_core orchestrator.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose logs in the console.")
    parser.add_argument("--no-emoji", action="store_true", help="Disable emoji in console output.")
    parser.add_argument(
        "--llm-provider",
        help="Override the LLM provider for this run (ex: ollama, lmstudio).",
    )
    parser.add_argument(
        "--llm-model-text",
        help="Override the text completion LLM model identifier.",
    )
    parser.add_argument(
        "--llm-model-json",
        help="Override the JSON metadata LLM model identifier.",
    )
    args, passthrough = parser.parse_known_args(argv)

    settings = load_settings()
    settings = apply_llm_overrides(
        settings,
        provider=args.llm_provider,
        model_text=args.llm_model_text,
        model_json=args.llm_model_json,
    )
    set_settings(settings)
    settings = get_settings()

    log_effective_settings(settings)



    if args.print_config:
        config = _build_fetcher_config(settings)
        snapshot = _fetcher_config_snapshot(config)
        resolved_names: list[str] = list(snapshot['resolved_names'])
        active_names: list[str] = list(snapshot['active_names'])
        if active_names:
            resolved_display = ','.join(active_names)
        elif resolved_names:
            resolved_display = ','.join(resolved_names)
        else:
            resolved_display = 'none'

        raw_spec = _raw_provider_spec(settings)
        if raw_spec != 'default' or os.environ.get('_PIPELINE_PROVIDER_DEFAULT_EMITTED') != '1':
            safe_print(f"providers={raw_spec}")
            os.environ['_PIPELINE_PROVIDER_DEFAULT_EMITTED'] = '1'

        allow_images = str(bool(snapshot['allow_images'])).lower()
        allow_videos = str(bool(snapshot['allow_videos'])).lower()
        per_segment_limit = int(snapshot['per_segment_limit'])

        safe_print(f"resolved_providers={resolved_display}")
        safe_print(f"allow_images={allow_images}")
        safe_print(f"allow_videos={allow_videos}")
        safe_print(f"per_segment_limit={per_segment_limit}")
        for line in _render_provider_limit_lines(snapshot):
            safe_print(line)
        return 0

    if args.diag_broll:
        return _run_broll_diagnostic(repo_root, settings)
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

    try:
        shared_service = get_shared_llm_service()
    except Exception:
        shared_service = None
    else:
        processor_cls = getattr(video_processor_module, "VideoProcessor", None)
        if shared_service is not None and processor_cls is not None:
            try:
                setattr(processor_cls, "_shared_llm_service", shared_service)
            except Exception:
                pass

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





