"""Repository diagnostics utility for environment and bytecode hygiene."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableSequence, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
READINESS_PATH = REPO_ROOT / "tools" / "out" / "llm_ready.json"
ENV_KEYS: Sequence[str] = (
    "PYTHONPATH",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONWARNINGS",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "PIP_INDEX_URL",
)
SKIP_SCAN_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
    "__pycache__",
}
SKIP_SCAN_PREFIXES = (".venv", "venv", "env")


def _normalise(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped


def _clean_env_value(key: str, value: Optional[str]) -> Optional[str]:
    cleaned = _normalise(value)
    if cleaned is None:
        return None
    markers = ("KEY", "TOKEN", "SECRET", "PASSWORD")
    upper_key = key.upper()
    if any(marker in upper_key for marker in markers):
        tail = cleaned[-4:] if len(cleaned) > 4 else cleaned
        return f"****{tail}"
    return cleaned


def _hash_readiness(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()[:12]


def _load_readiness(path: Path) -> Tuple[Optional[Mapping[str, object]], Optional[str]]:
    if not path.exists():
        return None, "readiness file not found"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"failed to load readiness metadata: {exc}"
    return data, None


def _collect_environment(keys: Iterable[str]) -> Dict[str, Optional[str]]:
    return {key: _clean_env_value(key, os.getenv(key)) for key in keys}


def _project_packages(repo_root: Path) -> Tuple[Sequence[str], Sequence[str]]:
    packages: MutableSequence[str] = []
    modules: MutableSequence[str] = []
    for child in repo_root.iterdir():
        if child.name.startswith('.'):
            continue
        if child.is_dir() and (child / "__init__.py").exists():
            packages.append(child.name)
        elif child.is_file() and child.suffix == ".py":
            modules.append(child.stem)
    return packages, modules


def _find_duplicates(packages: Sequence[str], modules: Sequence[str]) -> Dict[str, List[str]]:
    duplicates: Dict[str, List[str]] = {}
    seen_locations: Dict[str, set[str]] = defaultdict(set)
    for entry in sys.path:
        if not entry:
            continue
        try:
            path_entry = Path(entry).resolve()
        except OSError:
            continue
        if not path_entry.exists():
            continue
        if path_entry.is_file():
            candidate_name = path_entry.stem
            if candidate_name in modules:
                seen_locations[candidate_name].add(str(path_entry))
            continue
        for package in packages:
            package_path = path_entry / package
            if (package_path / "__init__.py").exists():
                seen_locations[package].add(str(package_path))
        for module in modules:
            module_path = path_entry / f"{module}.py"
            if module_path.exists():
                seen_locations[module].add(str(module_path))
    for name, locations in seen_locations.items():
        if len(locations) > 1:
            duplicates[name] = sorted(locations)
    return duplicates


def _should_skip_dir(directory: Path) -> bool:
    name = directory.name
    if name in SKIP_SCAN_DIRS and name != "__pycache__":
        return True
    return name.startswith(SKIP_SCAN_PREFIXES)


def _gather_bytecode(repo_root: Path) -> Tuple[List[Path], List[Path]]:
    pycache_dirs: List[Path] = []
    pyc_files: List[Path] = []
    for root, dirs, files in os.walk(repo_root):
        root_path = Path(root)
        filtered: List[str] = []
        for dirname in dirs:
            child = root_path / dirname
            if dirname == "__pycache__":
                pycache_dirs.append(child)
                for pyc in child.rglob("*.pyc"):
                    pyc_files.append(pyc)
                continue
            if _should_skip_dir(child):
                continue
            filtered.append(dirname)
        dirs[:] = filtered
        for filename in files:
            if filename.endswith(".pyc"):
                pyc_files.append(root_path / filename)
    pycache_dirs.sort()
    pyc_files = sorted(set(pyc_files))
    return pycache_dirs, pyc_files


def _format_paths(paths: Sequence[Path], limit: int = 5) -> str:
    if not paths:
        return "none"
    items = [str(path) for path in paths[:limit]]
    if len(paths) > limit:
        items.append(f"… and {len(paths) - limit} more")
    return "\n    " + "\n    ".join(items)


def _print_header() -> None:
    print("== pipeline doctor ==")
    print(f"[doctor] repo_root={REPO_ROOT}")
    print(f"[doctor] python_executable={sys.executable}")
    print(f"[doctor] python_version={sys.version.splitlines()[0]}")


def _print_readiness_section() -> None:
    print("\n-- readiness metadata --")
    digest = _hash_readiness(READINESS_PATH)
    payload, error = _load_readiness(READINESS_PATH)
    print(f"path: {READINESS_PATH}")
    print(f"exists: {READINESS_PATH.exists()}")
    print(f"sha256_12: {digest}")
    if error:
        print(f"status: {error}")
    elif payload is not None:
        print("status: loaded")
        try:
            keys = ", ".join(sorted(payload.keys()))
        except AttributeError:
            keys = "(payload not a mapping)"
        print(f"payload_keys: {keys}")


def _print_environment_section() -> None:
    print("\n-- environment variables --")
    env_data = _collect_environment(ENV_KEYS)
    for key in ENV_KEYS:
        value = env_data.get(key)
        print(f"{key}={value if value is not None else '<unset>'}")


def _print_duplicates_section() -> None:
    print("\n-- sys.path duplicate modules --")
    packages, modules = _project_packages(REPO_ROOT)
    duplicates = _find_duplicates(packages, modules)
    if not duplicates:
        print("no duplicate modules detected")
        return
    for name, locations in sorted(duplicates.items()):
        print(f"module '{name}' available at:")
        for location in locations:
            print(f"  - {location}")


def _print_bytecode_section() -> None:
    print("\n-- bytecode artefacts --")
    pycache_dirs, pyc_files = _gather_bytecode(REPO_ROOT)
    print(f"__pycache__ directories: {len(pycache_dirs)}")
    if pycache_dirs:
        print(_format_paths(pycache_dirs))
    print(f".pyc files: {len(pyc_files)}")
    if pyc_files:
        print(_format_paths(pyc_files))
    repo = REPO_ROOT
    print("suggested purge commands:")
    print(f"  find {repo} -name '__pycache__' -prune -exec rm -rf '{{}}' +")
    print(f"  find {repo} -name '*.pyc' -delete")
    print(
        "  Get-ChildItem -Path '{repo}' -Recurse -Force -Filter '__pycache__' -Directory | Remove-Item -Recurse -Force".format(
            repo=repo
        )
    )
    print(
        "  Get-ChildItem -Path '{repo}' -Recurse -Force -Include '*.pyc' | Remove-Item -Force".format(
            repo=repo
        )
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect repository environment and artefacts")
    parser.add_argument("--no-bytecode", action="store_true", help="skip bytecode artefact scan")
    parser.add_argument("--no-duplicates", action="store_true", help="skip sys.path duplicate inspection")
    args = parser.parse_args(argv)

    _print_header()
    _print_readiness_section()
    _print_environment_section()
    if not args.no_duplicates:
        _print_duplicates_section()
    if not args.no_bytecode:
        _print_bytecode_section()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
