"""Runtime diagnostics banner for early import tracing."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence


_LOGGER = logging.getLogger("pipeline.startup")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _safe_resolve(entry: str) -> str:
    if not entry:
        return entry
    try:
        return str(Path(entry).resolve())
    except OSError:
        return entry


def _clean_env_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()
    return cleaned or None


def _mask_env_value(key: str, value: Optional[str]) -> Optional[str]:
    cleaned = _clean_env_value(value)
    if cleaned is None:
        return None
    upper_key = key.upper()
    sensitive_markers = ("KEY", "TOKEN", "SECRET", "PASSWORD")
    if any(marker in upper_key for marker in sensitive_markers):
        tail = cleaned[-4:] if len(cleaned) > 4 else cleaned
        return f"****{tail}"
    return cleaned


def _git_commit(repo_root: Path) -> Optional[str]:
    git_dir = repo_root / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return None
    try:
        head_contents = head_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if head_contents.startswith("ref:"):
        ref_name = head_contents.split(":", 1)[1].strip()
        ref_path = git_dir / ref_name
        try:
            commit = ref_path.read_text(encoding="utf-8").strip()
        except OSError:
            commit = None
    else:
        commit = head_contents
    if not commit:
        return None
    return commit[:12]


def _read_sha256(path: Path) -> Optional[str]:
    digest = hashlib.sha256()
    try:
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(65536), b''):
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _readiness_metadata(repo_root: Path) -> Dict[str, object]:
    readiness_path = repo_root / 'tools' / 'out' / 'llm_ready.json'
    exists = readiness_path.exists()
    info: Dict[str, object] = {
        'path': str(readiness_path),
        'sha256_12': None,
        'exists': exists,
    }
    if exists:
        digest = _read_sha256(readiness_path)
        if digest:
            info['sha256_12'] = digest[:12]
    return info


def _collect_env(keys: Iterable[str]) -> Dict[str, Optional[str]]:
    return {key: _mask_env_value(key, os.getenv(key)) for key in keys}


def emit_runtime_banner(*, env_keys: Optional[Sequence[str]] = None) -> None:
    """Log startup metadata for debugging and diagnostics."""

    repo_root = _repo_root()
    metadata = {
        'cwd': str(Path.cwd()),
        'python_executable': sys.executable,
        'python_version': sys.version.replace(os.linesep, ' '),
        'sys_path_head': [_safe_resolve(entry) for entry in list(sys.path)[:5]],
        'git_commit': _git_commit(repo_root),
        'llm_readiness': _readiness_metadata(repo_root),
    }
    if env_keys:
        metadata['env'] = _collect_env(env_keys)

    _LOGGER.info("startup metadata: %s", json.dumps(metadata, sort_keys=True, ensure_ascii=False))
