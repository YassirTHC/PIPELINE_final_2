"""Validate the PyCaps E2E run log to enforce acceptance criteria."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ALLOWED_FALLBACK_TOKENS = {
    "fallback_trunc",
    "timeout_fallback",
    "emoji_no_context_fallback",
    "tfidf_fallback_disabled",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path, help="Path to the captured pipeline log file.")
    args = parser.parse_args()

    if not args.log_path.exists():
        print(f"[validate] log file not found: {args.log_path}", file=sys.stderr)
        return 1

    has_engine_line = False
    has_pycaps_render = False
    has_title = False
    has_description = False
    has_hashtags = False

    failures: list[str] = []

    raw_bytes = args.log_path.read_bytes()
    decoded = None
    for encoding in ("utf-8", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            decoded = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        print(f"[validate] unable to decode log as utf-8/utf-16: {args.log_path}", file=sys.stderr)
        return 1

    for idx, raw_line in enumerate(decoded.splitlines(), start=1):
            line = raw_line.strip()
            lower = line.lower()

            if "fallback" in lower:
                if not any(token in lower for token in ALLOWED_FALLBACK_TOKENS):
                    failures.append(f"forbidden fallback trace at line {idx}: {line}")

            if "engine=pycaps" in lower:
                has_engine_line = True

            if "[pycaps] rendered subtitles using pycaps" in lower:
                has_pycaps_render = True

            if "title:" in lower and "none" not in lower:
                has_title = True

            if "description:" in lower and "none" not in lower:
                has_description = True

            if "hashtags:" in lower and "none" not in lower:
                has_hashtags = True

            if "title: none" in lower or "description: none" in lower:
                failures.append(f"metadata missing at line {idx}: {line}")

    if not has_engine_line:
        failures.append("missing 'Engine=pycaps' confirmation in log")
    if not has_pycaps_render:
        failures.append("missing PyCaps renderer completion log entry")
    if not has_title:
        failures.append("missing non-empty Title metadata log entry")
    if not has_description:
        failures.append("missing non-empty Description metadata log entry")
    if not has_hashtags:
        failures.append("missing hashtags log entry")

    if failures:
        print("[validate] log validation failed:", file=sys.stderr)
        for issue in failures:
            print(f"  - {issue}", file=sys.stderr)
        return 1

    print("[validate] log validation succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
