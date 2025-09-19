#!/usr/bin/env python3
"""Stable wrapper around video_processor with sane environment defaults."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _compute_pythonpath(repo_root: Path) -> str:
    extra = [str(repo_root), str(repo_root / "AI-B-roll"), str(repo_root / "utils")]
    current = os.environ.get("PYTHONPATH")
    if current:
        extra.append(current)
    return os.pathsep.join(extra)


def main() -> int:
    repo_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Launch the video pipeline with stable environment defaults."
    )
    parser.add_argument("--video", required=True, help="Path to the source video (mp4, mov, etc.)")
    parser.add_argument("--legacy", action="store_true", help="Disable the modern pipeline_core orchestrator.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose logs in the console.")
    args, passthrough = parser.parse_known_args()

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ["ENABLE_PIPELINE_CORE_FETCHER"] = "false" if args.legacy else os.environ.get(
        "ENABLE_PIPELINE_CORE_FETCHER", "true"
    )
    os.environ["PYTHONPATH"] = _compute_pythonpath(repo_root)

    cmd = [sys.executable, str(repo_root / "video_processor.py"), "--video", args.video]
    if args.verbose:
        cmd.append("--verbose")
    cmd.extend(passthrough)

    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())