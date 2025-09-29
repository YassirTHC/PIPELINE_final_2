"""Tests covering environment helpers and provider resolution."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from pipeline_core.configuration import (
    FetcherOrchestratorConfig,
    ProviderConfig,
    resolved_providers,
    to_bool,
)


@pytest.mark.parametrize(
    "value,default,expected",
    [
        (True, None, True),
        (False, None, False),
        ("ON", None, True),
        ("off", None, False),
        (" yes ", None, True),
        ("", True, True),
        (None, False, False),
        ("maybe", True, True),
    ],
)
def test_to_bool_interprets_common_tokens(value, default, expected):
    """Validate the normalisation logic for boolean environment values."""

    assert to_bool(value, default=default) is expected


def test_resolved_providers_default_to_pixabay():
    """The helper should fall back to pixabay when nothing is configured."""

    assert resolved_providers(None) == ["pixabay"]

    empty_cfg = FetcherOrchestratorConfig(providers=())
    assert resolved_providers(empty_cfg) == ["pixabay"]

    mixed_cfg = FetcherOrchestratorConfig(
        providers=(
            ProviderConfig(name="Pexels"),
            ProviderConfig(name="pixabay"),
            ProviderConfig(name="PEXELS"),
        )
    )
    assert resolved_providers(mixed_cfg) == ["Pexels", "pixabay"]


def test_print_config_reports_pexels_presence(monkeypatch, tmp_path):
    """Running ``--print-config`` should expose the API key presence flag."""

    script = Path(__file__).resolve().parents[1] / "run_pipeline.py"

    env = os.environ.copy()
    # Ensure the environment is deterministic for the subprocess invocation.
    env.pop("PEXELS_API_KEY", None)
    env["PIPELINE_FAST_TESTS"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("PYTHONPATH", str(script.parent))

    result = subprocess.run(
        [sys.executable, str(script), "--print-config"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "pexels_key_present=" in result.stdout
