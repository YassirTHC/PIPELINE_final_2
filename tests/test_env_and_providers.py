import os
import subprocess
import sys
from pathlib import Path

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


def test_print_config_reports_limits(monkeypatch):
    monkeypatch.setenv("PIXABAY_API_KEY", "dummy-key")
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    monkeypatch.setenv("FETCH_MAX", "5")
    monkeypatch.setenv("BROLL_FETCH_ALLOW_IMAGES", "0")
    monkeypatch.delenv("BROLL_FETCH_PROVIDER", raising=False)
    monkeypatch.delenv("AI_BROLL_FETCH_PROVIDER", raising=False)

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", os.getcwd())

    output = subprocess.check_output(
        [sys.executable, "run_pipeline.py", "--print-config"],
        cwd=os.getcwd(),
        env=env,
    ).decode("utf-8").strip().splitlines()

    assert output[0] == "providers=default"
    assert output[1] == "resolved_providers=pixabay"
    assert output[2] == "allow_images=false"
    assert output[3] == "allow_videos=true"
    assert output[4] == "per_segment_limit=5"
    assert "provider=pixabay max_results=5" in output


def test_diag_broll_reports_provider_limits(monkeypatch):
    monkeypatch.setenv("PIXABAY_API_KEY", "dummy-key")
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    monkeypatch.setenv("FETCH_MAX", "4")
    monkeypatch.setenv("BROLL_FETCH_ALLOW_IMAGES", "0")
    monkeypatch.delenv("BROLL_FETCH_PROVIDER", raising=False)
    monkeypatch.delenv("AI_BROLL_FETCH_PROVIDER", raising=False)

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", os.getcwd())

    proc = subprocess.run(
        [sys.executable, "run_pipeline.py", "--diag-broll"],
        cwd=os.getcwd(),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    output = proc.stdout.decode("utf-8")
    assert "[DIAG] providers=default" in output
    assert "[DIAG] resolved_providers=pixabay" in output
    assert "[DIAG] per_segment_limit=4" in output
    assert "[DIAG] provider=pixabay max_results=4" in output

    report_path = Path("diagnostic_broll.json")
    if report_path.exists():
        report_path.unlink()
    events_path = Path("output/meta/broll_pipeline_events.jsonl")
    if events_path.exists():
        events_path.unlink()
        try:
            events_path.parent.rmdir()
        except OSError:
            pass
        try:
            events_path.parent.parent.rmdir()
        except OSError:
            pass
