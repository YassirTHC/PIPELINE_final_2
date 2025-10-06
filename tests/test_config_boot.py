import json
import logging
from pathlib import Path

import pytest

from video_pipeline.config import (
    get_settings,
    load_settings,
    log_effective_settings,
    reset_startup_log_for_tests,
    set_settings,
)
from video_pipeline.config.settings import (
    csv_list,
    reset_settings_cache_for_tests,
)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
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
        "PIPELINE_LLM_JSON_MODE",
        "PIPELINE_LLM_TARGET_LANG",
        "PIPELINE_LLM_TEMP",
        "PIPELINE_LLM_TOP_P",
        "PIPELINE_LLM_REPEAT_PENALTY",
        "PIPELINE_LLM_NUM_CTX",
        "PIPELINE_LLM_FALLBACK_TRUNC",
        "PIPELINE_LLM_MAX_ATTEMPTS",
        "PIPELINE_LLM_MIN_CHARS",
        "PIPELINE_LLM_NUM_PREDICT",
        "PIPELINE_BROLL_MIN_START_SECONDS",
        "PIPELINE_BROLL_MIN_GAP_SECONDS",
        "PIPELINE_BROLL_NO_REPEAT_SECONDS",
        "PIPELINE_FETCH_TIMEOUT_S",
        "BROLL_FETCH_MAX_PER_KEYWORD",
        "BROLL_FETCH_ALLOW_IMAGES",
        "BROLL_FETCH_ALLOW_VIDEOS",
        "BROLL_FETCH_PROVIDER",
        "FETCH_MAX",
        "PEXELS_API_KEY",
        "PIXABAY_API_KEY",
        "UNSPLASH_ACCESS_KEY",
        "PIPELINE_TFIDF_FALLBACK_DISABLED",
        "PIPELINE_DISABLE_TFIDF_FALLBACK",
        "PIPELINE_MAX_SEGMENTS_IN_FLIGHT",
        "PIPELINE_LLM_MAX_QUERIES_PER_SEGMENT",
        "PIPELINE_FAST_TESTS",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    reset_settings_cache_for_tests()
    reset_startup_log_for_tests()
    yield


def test_config_boot_parses_types(monkeypatch):
    monkeypatch.setenv("PIPELINE_LLM_TIMEOUT_S", "42.5")
    monkeypatch.setenv("PIPELINE_LLM_FALLBACK_TIMEOUT_S", "33")
    monkeypatch.setenv("PIPELINE_LLM_FORCE_NON_STREAM", "yes")
    monkeypatch.setenv("PIPELINE_LLM_KEYWORDS_FIRST", "0")
    monkeypatch.setenv("PIPELINE_LLM_JSON_PROMPT", "  custom ")
    monkeypatch.setenv("PIPELINE_LLM_JSON_TRANSCRIPT_LIMIT", "128")
    monkeypatch.setenv("PIPELINE_LLM_JSON_MODE", "true")
    monkeypatch.setenv("PIPELINE_TFIDF_FALLBACK_DISABLED", "yes")
    monkeypatch.setenv("PIPELINE_BROLL_MIN_START_SECONDS", "2.75")
    monkeypatch.setenv("PIPELINE_BROLL_MIN_GAP_SECONDS", "1.25")
    monkeypatch.setenv("PIPELINE_BROLL_NO_REPEAT_SECONDS", "9.0")
    monkeypatch.setenv("PIPELINE_LLM_MAX_QUERIES_PER_SEGMENT", "4")
    monkeypatch.setenv("PIPELINE_MAX_SEGMENTS_IN_FLIGHT", "2")
    monkeypatch.setenv("PIPELINE_FAST_TESTS", "1")
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
    assert settings.llm.json_mode is True

    assert settings.broll.min_start_s == pytest.approx(2.75)
    assert settings.broll.min_gap_s == pytest.approx(1.25)
    assert settings.broll.no_repeat_s == pytest.approx(9.0)

    assert settings.fetch.timeout_s == pytest.approx(9.5)
    assert settings.fetch.max_per_keyword == 5
    assert settings.fetch.allow_images is False
    assert settings.fetch.allow_videos is True
    assert settings.fetch.providers == ("pixabay", "pexels")
    assert settings.fetch.provider_limits["pexels"] == 4
    assert settings.llm_max_queries_per_segment == 4
    assert settings.max_segments_in_flight == 2
    assert settings.tfidf_fallback_disabled is True
    assert settings.fast_tests is True


def test_config_boot_log_masks_sensitive_keys(monkeypatch, caplog):
    monkeypatch.setenv("PIPELINE_LLM_MODEL", "llama")
    monkeypatch.setenv("PEXELS_API_KEY", "pexels-secret")
    monkeypatch.setenv("PIXABAY_API_KEY", "pixabay-secret")

    settings = load_settings()

    caplog.set_level(logging.INFO, logger="video_pipeline.config")
    log_effective_settings(settings)

    assert caplog.records, "expected startup log"
    message = caplog.records[0].message
    assert message.startswith("[CONFIG] effective=")
    payload = json.loads(message.split("=", 1)[1])
    fetch_payload = payload["fetch"]
    assert fetch_payload["api_keys"]["PEXELS_API_KEY"] == "****cret"
    assert fetch_payload["api_keys"]["PIXABAY_API_KEY"] == "****cret"


def test_config_boot_invalid_numbers_fallback(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="video_pipeline.config")
    monkeypatch.setenv("PIPELINE_LLM_TIMEOUT_S", "not-a-number")
    monkeypatch.setenv("PIPELINE_LLM_MIN_CHARS", "-5")
    monkeypatch.setenv("PIPELINE_FETCH_TIMEOUT_S", "-2")

    settings = load_settings()

    assert settings.llm.timeout_stream_s == pytest.approx(60.0)
    assert settings.llm.min_chars == 8
    assert settings.fetch.timeout_s == pytest.approx(8.0)
    assert caplog.records, "expected warnings for invalid values"


def test_config_boot_alias_disables(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="video_pipeline.config")
    monkeypatch.setenv("PIPELINE_DISABLE_TFIDF_FALLBACK", "true")

    settings = load_settings()

    assert settings.tfidf_fallback_disabled is True
    assert any("deprecated" in record.message for record in caplog.records)


def test_config_boot_log_only_once(monkeypatch, caplog):
    monkeypatch.setenv("PIPELINE_LLM_MODEL", "llama")
    settings = load_settings()

    caplog.set_level(logging.INFO, logger="video_pipeline.config")
    log_effective_settings(settings)
    log_effective_settings(settings)

    assert len(caplog.records) == 1

    reset_startup_log_for_tests()
    caplog.clear()
    log_effective_settings(settings)
    assert caplog.records


def test_config_boot_cache_helpers(monkeypatch):
    monkeypatch.setenv("PIPELINE_LLM_MODEL", "qwen-main")
    reset_settings_cache_for_tests()
    reset_startup_log_for_tests()

    cached = get_settings()
    assert cached.llm.model == "qwen-main"

    override = load_settings({"PIPELINE_LLM_MODEL": "other"})
    set_settings(override)
    assert get_settings().llm.model == "other"


def test_csv_list_handles_duplicates():
    assert csv_list("a, b ,a ,c") == ["a", "b", "c"]
    assert csv_list(["x", "x", "y"]) == ["x", "y"]
    assert csv_list(None) == []


def test_settings_paths_are_paths(monkeypatch):
    monkeypatch.setenv("PIPELINE_CLIPS_DIR", " clips ")
    monkeypatch.setenv("PIPELINE_OUTPUT_DIR", "custom-output")
    monkeypatch.setenv("PIPELINE_TEMP_DIR", "tmp")

    settings = load_settings()

    assert settings.clips_dir == Path("clips")
    assert settings.output_dir == Path("custom-output")
    assert settings.temp_dir == Path("tmp")
