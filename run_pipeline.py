#!/usr/bin/env python3
"""Stable wrapper around video_processor with sane environment defaults."""
from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

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


def _run_broll_diagnostic(repo_root: Path) -> int:
    try:
        import json
        import time
    except ImportError as exc:
        print(f"[DIAG] missing stdlib dependency: {exc}", file=sys.stderr)
        return 1
    try:
        import requests  # type: ignore[import-not-found]
    except ImportError:
        print('[DIAG] python-requests is required for --diag-broll.', file=sys.stderr)
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

    active_names = [item['name'] for item in providers_meta if item['key_present']]
    allow_images = _as_bool(os.environ.get('BROLL_FETCH_ALLOW_IMAGES'), default=False)
    fetch_max_raw = os.environ.get('BROLL_FETCH_MAX_PER_KEYWORD') or '0'
    try:
        fetch_max = int(fetch_max_raw)
    except ValueError:
        fetch_max = 0

    providers_display = ','.join(active_names) if active_names else 'none'
    print(f"[DIAG] providers={providers_display} | allow_images={str(allow_images).lower()} | fetch_max={fetch_max}")

    results = []
    for provider in providers_meta:
        status = dict(provider)
        name = status['name']
        if not status['key_present']:
            status['success'] = False
            status['error'] = 'missing_api_key'
            print(f"[DIAG] provider={name} skipped (missing key)")
            results.append(status)
            continue

        try:
            start = time.perf_counter()
            if name == 'pexels':
                resp = requests.get(
                    'https://api.pexels.com/videos/search',
                    headers={'Authorization': os.environ[status['env_key']]},
                    params={'query': 'nature', 'per_page': 1},
                    timeout=0.7,
                )
                latency_ms = int((time.perf_counter() - start) * 1000)
                status['latency_ms'] = latency_ms
                status['http_status'] = resp.status_code
                if resp.ok:
                    try:
                        payload = resp.json()
                    except Exception as exc:
                        status['success'] = False
                        status['error'] = f'json_error:{exc}'
                    else:
                        videos = payload.get('videos') or []
                        status['candidates'] = len(videos)
                        status['success'] = len(videos) > 0
                        if not status['success']:
                            status['error'] = 'no_videos'
                else:
                    status['success'] = False
                    status['error'] = f'http_{resp.status_code}'
            elif name == 'pixabay':
                resp = requests.get(
                    'https://pixabay.com/api/videos/',
                    params={'key': os.environ[status['env_key']], 'q': 'nature', 'per_page': 3},
                    timeout=0.7,
                )
                latency_ms = int((time.perf_counter() - start) * 1000)
                status['latency_ms'] = latency_ms
                status['http_status'] = resp.status_code
                if resp.ok:
                    try:
                        payload = resp.json()
                    except Exception as exc:
                        status['success'] = False
                        status['error'] = f'json_error:{exc}'
                    else:
                        hits = payload.get('hits') or []
                        status['candidates'] = len(hits)
                        status['success'] = len(hits) > 0
                        if not status['success']:
                            status['error'] = 'no_hits'
                else:
                    status['success'] = False
                    status['error'] = f'http_{resp.status_code}'
            else:
                status['success'] = False
                status['error'] = 'unsupported_provider'
        except requests.Timeout:
            status['success'] = False
            status['error'] = 'timeout'
        except Exception as exc:
            status['success'] = False
            status['error'] = str(exc)

        print(
            f"[DIAG] provider={name} success={status.get('success')} latency_ms={status.get('latency_ms', 'n/a')} candidates={status.get('candidates', 0)}"
        )
        results.append(status)

    payload = {
        'timestamp': time.time(),
        'providers_actifs': active_names,
        'allow_images': allow_images,
        'fetch_max': fetch_max,
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
    if getattr(video_processor, "__spec__", None) is None and getattr(video_processor, "main", None):
        video_processor_module = video_processor
    elif getattr(video_processor, "__spec__", None) is not None:
        try:
            video_processor_module = importlib.reload(video_processor)  # type: ignore[arg-type]
        except Exception:
            video_processor_module = importlib.import_module("video_processor")
            video_processor = video_processor_module
    else:
        video_processor_module = importlib.import_module("video_processor")
        video_processor = video_processor_module

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
