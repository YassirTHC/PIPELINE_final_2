"""Runtime diagnostics banner for early import tracing."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def _read_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(65536), b''):
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return '<unreadable>'
    return digest.hexdigest()


def _format_mtime(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
    except OSError:
        return '<unknown>'
    return datetime.fromtimestamp(ts).isoformat()


def _resolve_spec_path(module_name: str) -> Optional[Path]:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    origin = spec.origin
    if origin is None:
        locations = spec.submodule_search_locations or []
        for location in locations:
            candidate = Path(location) / '__init__.py'
            if candidate.exists():
                return candidate.resolve()
        return None
    candidate = Path(origin)
    if candidate.name == '__init__.py' and candidate.exists():
        return candidate.resolve()
    if candidate.exists():
        return candidate.resolve()
    return None


def _emit_module_stamp(module_name: str) -> None:
    module_obj = None
    try:
        module_obj = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - diagnostic logging only
        print(f"[module {module_name}] import_error={exc.__class__.__name__}: {exc}")
    path: Optional[Path] = None
    if module_obj is not None:
        file_attr = getattr(module_obj, '__file__', None)
        if file_attr:
            try:
                path = Path(file_attr).resolve()
            except OSError:
                path = Path(file_attr)
    if path is None:
        resolved = _resolve_spec_path(module_name)
        if resolved is not None:
            path = resolved
    if path is None:
        print(f"[module {module_name}] __file__=<missing> sha256=<n/a> mtime=<n/a>")
        return
    sha_value = _read_sha256(path)
    mtime_value = _format_mtime(path)
    print(f"[module {module_name}] __file__={path} sha256={sha_value} mtime={mtime_value}")


def emit_runtime_banner() -> None:
    """Print runtime diagnostics for imports and environment hygiene."""

    cwd = Path.cwd()
    print('=== RUNTIME BANNER ===')
    print(f"cwd={cwd}")
    print(f"sys.executable={sys.executable}")
    version_line = sys.version.replace(os.linesep, ' ')
    print(f"sys.version={version_line}")

    top_paths = list(sys.path)[:5]
    for idx, entry in enumerate(top_paths):
        print(f"sys.path[{idx}]={entry}")

    for needle in ('video_pipeline', 'pipeline_core'):
        matches = []
        for entry in sys.path:
            if not entry:
                continue
            if needle not in entry:
                continue
            try:
                resolved = str(Path(entry).resolve())
            except OSError:
                resolved = entry
            matches.append(resolved)
        unique = sorted(set(matches))
        if len(unique) > 1:
            print(f"ALERT: multiple sys.path entries contain '{needle}':")
            for item in unique:
                print(f"  -> {item}")

    modules = (
        'pipeline_core.configuration',
        'pipeline_core.llm_service',
        'video_processor',
        'utils.optimized_llm',
    )
    for module_name in modules:
        _emit_module_stamp(module_name)
    print('=== END RUNTIME BANNER ===')
