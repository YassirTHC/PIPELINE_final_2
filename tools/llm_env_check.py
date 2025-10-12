"""CLI to run the targeted pytest suite that validates the LLM environment."""
from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
from typing import Iterable, List, Sequence

_DEFAULT_TEST_TARGETS: tuple[str, ...] = (
    "tests/test_llm_optional_integration.py",
    "tests/test_llm_service_fallback.py",
    "tests/test_run_pipeline_env.py",
)


def build_pytest_command(extra_args: Sequence[str] | None = None) -> List[str]:
    """Return the pytest command used by the environment check."""

    command: List[str] = ["pytest", *_DEFAULT_TEST_TARGETS]
    if extra_args:
        command.extend(extra_args)
    return command


def _format_command(parts: Iterable[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the verification suite or print the pytest command."""

    parser = argparse.ArgumentParser(
        description=(
            "Execute the pytest suite that validates optional LLM integration and "
            "pipeline bootstrap dependencies. Use `--` to pass extra arguments to pytest."
        )
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only display the pytest command without executing it.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to pytest (prefix with -- before them).",
    )

    args = parser.parse_args(argv)
    extra_args = args.pytest_args or []
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    command = build_pytest_command(extra_args)

    if args.list:
        print(_format_command(command))
        return 0

    if shutil.which("pytest") is None:
        print(
            "pytest executable not found in PATH. Install project dependencies first.",
        )
        return 3

    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    raise SystemExit(main())
