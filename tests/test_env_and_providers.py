import os
import subprocess
import sys

import pytest

from pipeline_core.configuration import (
    FetcherOrchestratorConfig,
    resolved_providers,
    to_bool,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", True),
        ("0", False),
        ("true", True),
        ("false", False),
        ("yes", True),
        ("no", False),
        ("on", True),
        ("off", False),
    ],
)
def test_to_bool_normalises_variants(value, expected):
    assert to_bool(value, default=not expected) is expected


def test_resolved_providers_defaults_to_pixabay(monkeypatch):
    monkeypatch.delenv("BROLL_FETCH_PROVIDER", raising=False)
    monkeypatch.delenv("AI_BROLL_FETCH_PROVIDER", raising=False)
    monkeypatch.setenv("PIXABAY_API_KEY", "dummy-key")
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)

    config = FetcherOrchestratorConfig.from_environment()
    providers = resolved_providers(config)
    assert providers == ["pixabay"]


def test_print_config_reports_pexels_key(monkeypatch):
    monkeypatch.setenv("PIXABAY_API_KEY", "dummy-key")
    monkeypatch.setenv("PEXELS_API_KEY", "dummy-pexels")
    monkeypatch.setenv("FETCH_MAX", "5")
    monkeypatch.delenv("BROLL_FETCH_PROVIDER", raising=False)
    monkeypatch.delenv("AI_BROLL_FETCH_PROVIDER", raising=False)

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", os.getcwd())

    output = subprocess.check_output(
        [sys.executable, "run_pipeline.py", "--print-config"],
        cwd=os.getcwd(),
        env=env,
    ).decode("utf-8").strip()

    assert "pexels_key_present=" in output
    assert "resolved_providers=" in output
