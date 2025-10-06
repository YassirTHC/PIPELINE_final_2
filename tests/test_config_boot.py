import json
import logging
from pathlib import Path

import pytest

from video_pipeline.config import load_settings, log_effective_settings


@pytest.fixture(autouse=True)
def _clear_custom_env(monkeypatch):
    keys = [
        "PIPELINE_LLM_TIMEOUT_S",
        "PIPELINE_LLM_FALLBACK_TIMEOUT_S",
        "PIPELINE_LLM_FORCE_NON_STREAM",
        "PIPELINE_LLM_KEYWORDS_FIRST",
        "PIPELINE_LLM_MODEL",
        "PIPELINE_LLM_MODEL_JSON",
        "PIPELINE_LLM_MODEL_TEXT",
        "PIPELINE_LLM_JSON_PROMPT",
        "PIPELINE_LLM_JSON_TRANSCRIPT_LIMIT",
        "PIPELINE_BROLL_MIN_START_SECONDS",
        "PIPELINE_BROLL_MIN_GAP_SECONDS",
        "PIPELINE_BROLL_NO_REPEAT_SECONDS",
        "PIPELINE_FETCH_TIMEOUT_S",
        "BROLL_FETCH_MAX_PER_KEYWORD",
        "BROLL_FETCH_ALLOW_IMAGES",
        "BROLL_FETCH_ALLOW_VIDEOS",
        "BROLL_FETCH_PROVIDER",
        "BROLL_PEXELS_MAX_PER_KEYWORD",
        "FETCH_MAX",
        "PEXELS_API_KEY",
        "PIXABAY_API_KEY",
        "UNSPLASH_ACCESS_KEY",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    yield


def test_config_boot_parses_types(monkeypatch):
    monkeypatch.setenv("PIPELINE_LLM_TIMEOUT_S", "42.5")
    monkeypatch.setenv("PIPELINE_LLM_FALLBACK_TIMEOUT_S", "33")
    monkeypatch.setenv("PIPELINE_LLM_FORCE_NON_STREAM", "yes")
    monkeypatch.setenv("PIPELINE_LLM_KEYWORDS_FIRST", "0")
    monkeypatch.setenv("PIPELINE_LLM_JSON_PROMPT", "  custom ")
    monkeypatch.setenv("PIPELINE_LLM_JSON_TRANSCRIPT_LIMIT", "128")
    monkeypatch.setenv("PIPELINE_BROLL_MIN_START_SECONDS", "2.75")
    monkeypatch.setenv("PIPELINE_BROLL_MIN_GAP_SECONDS", "1.25")
    monkeypatch.setenv("PIPELINE_BROLL_NO_REPEAT_SECONDS", "9.0")
    monkeypatch.setenv("PIPELINE_FETCH_TIMEOUT_S", "9.5")
    monkeypatch.setenv("BROLL_FETCH_MAX_PER_KEYWORD", "5")
    monkeypatch.setenv("BROLL_FETCH_ALLOW_IMAGES", "no")
    monkeypatch.setenv("BROLL_FETCH_ALLOW_VIDEOS", "1")
    monkeypatch.setenv("BROLL_FETCH_PROVIDER", " pixabay , Pexels ")
    monkeypatch.setenv("BROLL_PEXELS_MAX_PER_KEYWORD", "4")

    settings = load_settings()

    assert settings.llm.timeout_stream_s == pytest.approx(42.5)
    assert settings.llm.timeout_fallback_s == pytest.approx(33.0)
    assert settings.llm.force_non_stream is True
    assert settings.llm.keywords_first is False
    assert settings.llm.json_prompt == "custom"
    assert settings.llm.json_transcript_limit == 128

    assert settings.broll.min_start_s == pytest.approx(2.75)
    assert settings.broll.min_gap_s == pytest.approx(1.25)
    assert settings.broll.no_repeat_s == pytest.approx(9.0)

    assert settings.fetch.timeout_s == pytest.approx(9.5)
    assert settings.fetch.max_per_keyword == 5
    assert settings.fetch.allow_images is False
    assert settings.fetch.allow_videos is True
    assert settings.fetch.providers == ("pixabay", "pexels")
    assert settings.fetch.provider_limits["pexels"] == 4


def test_config_boot_log_masks_sensitive_keys(monkeypatch):
    monkeypatch.setenv("PIPELINE_LLM_MODEL", "llama")
    monkeypatch.setenv("PEXELS_API_KEY", "pexels-secret")
    monkeypatch.setenv("PIXABAY_API_KEY", "pixabay-secret")

    settings = load_settings()

    logger = logging.getLogger("video_pipeline.config")
    records: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
            records.append(record)

    handler = _Collector()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        log_effective_settings(settings)
    finally:
        logger.removeHandler(handler)

    assert records, "no startup log emitted"

    message = records[0].message
    assert message.startswith("[CONFIG] effective=")

    payload = json.loads(message.split("=", 1)[1])
    fetch_payload = payload["fetch"]
    assert fetch_payload["api_keys"]["PEXELS_API_KEY"] == "****cret"
    assert fetch_payload["api_keys"]["PIXABAY_API_KEY"] == "****cret"


def test_config_boot_effective_models_use_defaults(monkeypatch):
    monkeypatch.setenv("PIPELINE_LLM_MODEL", "qwen-main")
    monkeypatch.delenv("PIPELINE_LLM_MODEL_JSON", raising=False)
    monkeypatch.delenv("PIPELINE_LLM_MODEL_TEXT", raising=False)

    settings = load_settings()

    assert settings.llm.effective_json_model == "qwen-main"
    assert settings.llm.effective_text_model == "qwen-main"
    payload = settings.to_log_payload()
    assert payload["llm"]["model_json"] == "qwen-main"
    assert payload["llm"]["model_text"] == "qwen-main"
    assert Path(payload["paths"]["output_dir"]).name == "output"

