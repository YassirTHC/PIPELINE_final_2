import json
import logging
from pathlib import Path

import pytest

from video_pipeline.config import (
    load_settings,
    log_effective_settings,
    reset_startup_log_for_tests,
)


@pytest.fixture(autouse=True)
def _reset_config(monkeypatch):
    keys = [
        "PIPELINE_LLM_TIMEOUT_S",
        "PIPELINE_LLM_FALLBACK_TIMEOUT_S",
        "PIPELINE_LLM_FORCE_NON_STREAM",
        "PIPELINE_LLM_KEYWORDS_FIRST",
        "PIPELINE_LLM_DISABLE_HASHTAGS",
        "PIPELINE_LLM_MODEL",
        "PIPELINE_LLM_MODEL_JSON",
        "PIPELINE_LLM_MODEL_TEXT",
        "PIPELINE_LLM_JSON_PROMPT",
        "PIPELINE_LLM_JSON_MODE",
        "PIPELINE_LLM_JSON_TRANSCRIPT_LIMIT",
        "PIPELINE_LLM_TEMP",
        "PIPELINE_LLM_TOP_P",
        "PIPELINE_LLM_REPEAT_PENALTY",
        "PIPELINE_LLM_MIN_CHARS",
        "PIPELINE_LLM_MAX_ATTEMPTS",
        "PIPELINE_LLM_NUM_PREDICT",
        "PIPELINE_LLM_NUM_CTX",
        "PIPELINE_LLM_FALLBACK_TRUNC",
        "PIPELINE_LLM_TARGET_LANG",
        "PIPELINE_BROLL_MIN_START_SECONDS",
        "PIPELINE_BROLL_MIN_GAP_SECONDS",
        "PIPELINE_BROLL_NO_REPEAT_SECONDS",
        "PIPELINE_FETCH_TIMEOUT_S",
        "BROLL_FETCH_MAX_PER_KEYWORD",
        "FETCH_MAX",
        "BROLL_FETCH_ALLOW_IMAGES",
        "BROLL_FETCH_ALLOW_VIDEOS",
        "BROLL_FETCH_PROVIDER",
        "AI_BROLL_FETCH_PROVIDER",
        "BROLL_PEXELS_MAX_PER_KEYWORD",
        "PIPELINE_TFIDF_FALLBACK_DISABLED",
        "PIPELINE_DISABLE_TFIDF_FALLBACK",
        "PIPELINE_LLM_MAX_QUERIES_PER_SEGMENT",
        "PIPELINE_FAST_TESTS",
        "PIPELINE_MAX_SEGMENTS_IN_FLIGHT",
        "PIPELINE_CLIPS_DIR",
        "PIPELINE_OUTPUT_DIR",
        "PIPELINE_TEMP_DIR",
        "PEXELS_API_KEY",
        "PIXABAY_API_KEY",
        "UNSPLASH_ACCESS_KEY",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    reset_startup_log_for_tests()
    yield


def test_config_boot_parses_types_and_clamps(monkeypatch):
    monkeypatch.setenv("PIPELINE_LLM_TIMEOUT_S", "42.5")
    monkeypatch.setenv("PIPELINE_LLM_FALLBACK_TIMEOUT_S", "-3")  # clamp to 0
    monkeypatch.setenv("PIPELINE_LLM_FORCE_NON_STREAM", "yes")
    monkeypatch.setenv("PIPELINE_LLM_KEYWORDS_FIRST", "0")
    monkeypatch.setenv("PIPELINE_LLM_DISABLE_HASHTAGS", "TRUE")
    monkeypatch.setenv("PIPELINE_LLM_JSON_PROMPT", "  custom ")
    monkeypatch.setenv("PIPELINE_LLM_JSON_MODE", "1")
    monkeypatch.setenv("PIPELINE_LLM_JSON_TRANSCRIPT_LIMIT", "-5")
    monkeypatch.setenv("PIPELINE_LLM_TEMP", "-1")
    monkeypatch.setenv("PIPELINE_LLM_TOP_P", "0.95")
    monkeypatch.setenv("PIPELINE_LLM_REPEAT_PENALTY", "0.0")
    monkeypatch.setenv("PIPELINE_LLM_MIN_CHARS", "-4")
    monkeypatch.setenv("PIPELINE_LLM_MAX_ATTEMPTS", "0")
    monkeypatch.setenv("PIPELINE_LLM_NUM_PREDICT", "1024")
    monkeypatch.setenv("PIPELINE_LLM_NUM_CTX", "-1")
    monkeypatch.setenv("PIPELINE_LLM_FALLBACK_TRUNC", "100")
    monkeypatch.setenv("PIPELINE_LLM_TARGET_LANG", " FR ")
    monkeypatch.setenv("PIPELINE_BROLL_MIN_START_SECONDS", "2.75")
    monkeypatch.setenv("PIPELINE_BROLL_MIN_GAP_SECONDS", "-1")
    monkeypatch.setenv("PIPELINE_BROLL_NO_REPEAT_SECONDS", "9.0")
    monkeypatch.setenv("PIPELINE_FETCH_TIMEOUT_S", "-9.5")
    monkeypatch.setenv("BROLL_FETCH_MAX_PER_KEYWORD", "0")
    monkeypatch.setenv("FETCH_MAX", "12")
    monkeypatch.setenv("BROLL_FETCH_ALLOW_IMAGES", "no")
    monkeypatch.setenv("BROLL_FETCH_ALLOW_VIDEOS", "1")
    monkeypatch.setenv("BROLL_FETCH_PROVIDER", " pixabay , Pexels ,pixabay ")
    monkeypatch.setenv("BROLL_PEXELS_MAX_PER_KEYWORD", "-4")
    monkeypatch.setenv("PIPELINE_LLM_MAX_QUERIES_PER_SEGMENT", "0")
    monkeypatch.setenv("PIPELINE_FAST_TESTS", "true")
    monkeypatch.setenv("PIPELINE_MAX_SEGMENTS_IN_FLIGHT", "-2")

    settings = load_settings()

    assert settings.llm.timeout_stream_s == pytest.approx(42.5)
    assert settings.llm.timeout_fallback_s == pytest.approx(0.0)
    assert settings.llm.force_non_stream is True
    assert settings.llm.keywords_first is False
    assert settings.llm.disable_hashtags is True
    assert settings.llm.json_prompt == "custom"
    assert settings.llm.json_mode is True
    assert settings.llm.json_transcript_limit == 0
    assert settings.llm.temperature == pytest.approx(0.0)
    assert settings.llm.top_p == pytest.approx(0.95)
    assert settings.llm.repeat_penalty == pytest.approx(0.0)
    assert settings.llm.min_chars == 0
    assert settings.llm.max_attempts == 1
    assert settings.llm.num_predict == 1024
    assert settings.llm.num_ctx == 1
    assert settings.llm.fallback_trunc == 100
    assert settings.llm.target_lang == "FR"

    assert settings.broll.min_start_s == pytest.approx(2.75)
    assert settings.broll.min_gap_s == pytest.approx(0.0)
    assert settings.broll.no_repeat_s == pytest.approx(9.0)

    assert settings.fetch.timeout_s == pytest.approx(0.0)
    assert settings.fetch.max_per_keyword == 1
    assert settings.fetch.allow_images is False
    assert settings.fetch.allow_videos is True
    assert settings.fetch.providers == ["pixabay", "pexels"]
    assert settings.fetch.provider_limits["pexels"] == 1

    assert settings.llm_max_queries_per_segment == 1
    assert settings.fast_tests is True
    assert settings.max_segments_in_flight == 1


def test_config_boot_aliases_and_paths(monkeypatch, tmp_path):
    clips = tmp_path / "clips"
    output = tmp_path / "out"
    temp = tmp_path / "tmp"
    monkeypatch.setenv("PIPELINE_CLIPS_DIR", str(clips))
    monkeypatch.setenv("PIPELINE_OUTPUT_DIR", str(output))
    monkeypatch.setenv("PIPELINE_TEMP_DIR", str(temp))
    monkeypatch.setenv("PIPELINE_DISABLE_TFIDF_FALLBACK", "1")

    settings = load_settings()

    assert settings.clips_dir == clips
    assert settings.output_dir == output
    assert settings.temp_dir == temp
    assert settings.tfidf_fallback_disabled is True


def test_config_boot_log_masks_secrets_and_is_idempotent(monkeypatch):
    monkeypatch.setenv("PIPELINE_LLM_MODEL", "llama")
    monkeypatch.setenv("PEXELS_API_KEY", "pexels-secret")
    monkeypatch.setenv("PIXABAY_API_KEY", "pixabay-secret")
    monkeypatch.setenv("UNSPLASH_ACCESS_KEY", "ab")

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
        log_effective_settings(settings)
    finally:
        logger.removeHandler(handler)

    assert len(records) == 1
    message = records[0].getMessage()
    assert message.startswith("[CONFIG] effective=")

    payload = json.loads(message.split("=", 1)[1])
    api_keys = payload["fetch"]["api_keys"]
    assert api_keys["PEXELS_API_KEY"] == "****cret"
    assert api_keys["PIXABAY_API_KEY"] == "****cret"
    assert api_keys["UNSPLASH_ACCESS_KEY"] == "****ab"


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

