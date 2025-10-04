import json
import logging
from pathlib import Path
from typing import Any, Dict

import pytest

import pipeline_core.llm_service as llm_module
from pipeline_core.llm_service import (
    DynamicCompletionError,
    LLMMetadataGeneratorService,
    TfidfFallbackDisabled,
)


class DummyService(LLMMetadataGeneratorService):
    def __init__(self, payload: str):
        super().__init__(reuse_shared=False)
        self._payload = payload

    def _complete_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 800,
        purpose: str = "generic",
    ) -> str:  # type: ignore[override]
        return self._payload


def test_dynamic_context_normalisation():
    payload = json.dumps(
        {
            "detected_domains": [{"name": "Neuro_Science", "confidence": 0.87}],
            "language": "EN",
            "keywords": [
                "DOPAMINE_boost",
                "focus and clarity",
                "clarity when they concentrate",
            ],
            "synonyms": {"DOPAMINE_boost": ["dopamine surge", "chemical boost"]},
            "search_queries": ["brain focus when shot", "they and act"],
            "segment_briefs": [
                {"segment_index": "0", "keywords": ["Neural_Pathway"], "queries": ["brain first"]}
            ],
        }
    )
    service = DummyService(payload)
    result = service.generate_dynamic_context("Dopamine keeps you motivated")
    assert result["detected_domains"][0]["name"] == "neuro science"
    assert result["keywords"] == [
        "dopamine boost",
        "focus clarity",
        "clarity concentrate",
    ]
    assert result["search_queries"] == ["brain focus shot", "act"]
    assert result["segment_briefs"][0]["keywords"] == ["neural pathway"]
    assert result["segment_briefs"][0]["queries"] == []
    for term in result["keywords"] + result["search_queries"]:
        tokens = term.split()
        assert "and" not in tokens
        assert "when" not in tokens
        assert "they" not in tokens


def _initialise_with_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    *,
    payload: Dict[str, Any],
    base_model: str = "base-model",
    text_model: str = "text-model",
    json_model: str = "json-model",
):
    repo_root = tmp_path / "repo"
    ready_dir = repo_root / "tools" / "out"
    ready_dir.mkdir(parents=True, exist_ok=True)
    ready_path = ready_dir / "llm_ready.json"
    ready_path.write_text(json.dumps(payload))

    monkeypatch.setattr(llm_module, "PROJECT_ROOT", repo_root, raising=False)
    monkeypatch.setattr(llm_module, "_SHARED", None, raising=False)
    monkeypatch.setenv("PIPELINE_LLM_MODEL", base_model)
    monkeypatch.setenv("PIPELINE_LLM_MODEL_TEXT", text_model)
    monkeypatch.setenv("PIPELINE_LLM_MODEL_JSON", json_model)

    caplog.clear()
    with caplog.at_level("INFO", logger="pipeline_core.llm_service"):
        service = LLMMetadataGeneratorService(reuse_shared=False)

    readiness_entries = []
    for record in caplog.records:
        if record.name != "pipeline_core.llm_service":
            continue
        if record.msg != "[LLM] readiness routing decision: %s":
            continue
        if not record.args:
            continue
        if isinstance(record.args, dict):
            payload = record.args
        elif isinstance(record.args, tuple) and record.args:
            payload = record.args[0]
        else:
            payload = record.args
        if isinstance(payload, dict):
            readiness_entries.append(dict(payload))
    assert readiness_entries, "expected readiness routing log entry"
    return service, readiness_entries[-1]


def test_dynamic_context_fallback_keywords():
    class Failing(LLMMetadataGeneratorService):
        def _complete_text(
            self,
            prompt: str,
            *,
            max_tokens: int = 800,
            purpose: str = "generic",
        ) -> str:  # type: ignore[override]
            raise RuntimeError("boom")

    service = Failing(reuse_shared=False)
    transcript = "We discuss marketing funnels and conversion metrics for data driven teams."
    with pytest.raises(DynamicCompletionError) as excinfo:
        service.generate_dynamic_context(transcript)
    fallback = excinfo.value.payload or {}
    keywords = fallback.get("keywords", [])
    assert keywords, "fallback keywords missing"
    assert all(len(term) >= 3 for term in keywords)


def test_dynamic_context_respects_tfidf_disable(monkeypatch):
    class Failing(LLMMetadataGeneratorService):
        def _complete_text(
            self,
            prompt: str,
            *,
            max_tokens: int = 800,
            purpose: str = "generic",
        ) -> str:  # type: ignore[override]
            raise RuntimeError("boom")

    monkeypatch.setenv("PIPELINE_TFIDF_FALLBACK_DISABLED", "1")
    service = Failing(reuse_shared=False)

    transcript = "We discuss marketing funnels and conversion metrics for data driven teams."
    with pytest.raises(TfidfFallbackDisabled) as excinfo:
        service.generate_dynamic_context(transcript)
    assert "fallback_reason=integration_error" in str(excinfo.value)


def test_legacy_disable_env_emits_deprecation_warning(monkeypatch, caplog):
    class Failing(LLMMetadataGeneratorService):
        def _complete_text(
            self,
            prompt: str,
            *,
            max_tokens: int = 800,
            purpose: str = "generic",
        ) -> str:  # type: ignore[override]
            raise RuntimeError("boom")

    monkeypatch.delenv("PIPELINE_TFIDF_FALLBACK_DISABLED", raising=False)
    monkeypatch.setenv("PIPELINE_DISABLE_TFIDF_FALLBACK", "1")

    with caplog.at_level(logging.WARNING):
        service = Failing(reuse_shared=False)
        transcript = "We discuss marketing funnels and conversion metrics for data driven teams."
        with pytest.raises(TfidfFallbackDisabled):
            service.generate_dynamic_context(transcript)

    assert "PIPELINE_DISABLE_TFIDF_FALLBACK is deprecated" in caplog.text


def test_tfidf_fallback_scene_prompts_and_accent_normalisation():
    class FallbackOnly(LLMMetadataGeneratorService):
        def _complete_text(
            self,
            prompt: str,
            *,
            max_tokens: int = 800,
            purpose: str = "generic",
        ) -> str:  # type: ignore[override]
            raise RuntimeError("no llm")

    service = FallbackOnly(reuse_shared=False)
    transcript = (
        "Objectif et focus sur la réussite de l'équipe marketing. "
        "Ils écrivent des objectifs clairs dans un cahier."
    )
    with pytest.raises(DynamicCompletionError) as excinfo:
        service.generate_dynamic_context(transcript)
    fallback = excinfo.value.payload or {}
    keywords = fallback.get("keywords", [])
    assert "goal" in keywords
    assert "focus" in keywords
    assert "success" in keywords
    assert all(ord(ch) < 128 for term in keywords for ch in term)
    queries = fallback.get("search_queries", [])
    assert queries, "fallback queries missing"
    assert any(query == "writing goals on notebook" for query in queries)
    assert any(query == "focused typing keyboard" for query in queries)
    assert all(ord(ch) < 128 for query in queries for ch in query)


def test_segment_hint_generation_respects_tfidf_disable(monkeypatch):
    class SegmentFallback(LLMMetadataGeneratorService):
        def _segment_llm_json(
            self,
            prompt: str,
            *,
            timeout_s: float,
            num_predict: int,
        ) -> Dict[str, Any] | None:  # type: ignore[override]
            return None

    monkeypatch.setenv("PIPELINE_TFIDF_FALLBACK_DISABLED", "true")
    service = SegmentFallback(reuse_shared=False)

    with pytest.raises(TfidfFallbackDisabled) as excinfo:
        service.generate_hints_for_segment("Marketing focus drives success", 0.0, 3.0)
    assert "fallback_reason=segment_generation" in str(excinfo.value)


def test_force_english_terms_when_language_en():
    payload = json.dumps(
        {
            "language": "en",
            "keywords": ["récompense", "durée"],
            "search_queries": ["plan de récompense", "augmentation de durée"],
        }
    )
    service = DummyService(payload)
    result = service.generate_dynamic_context("Une transcription en anglais implicite")
    assert "reward" in result["keywords"]
    assert "duration" in result["keywords"]
    assert all("récompense" not in q and "durée" not in q for q in result["search_queries"])


def test_force_english_terms_handles_accented_sequences():
    payload = json.dumps(
        {
            "language": "en",
            "keywords": ["adrénaline récompense contrôle"],
            "search_queries": ["adrénaline récompense contrôle processus"],
        }
    )
    service = DummyService(payload)
    result = service.generate_dynamic_context("Adrénaline récompense contrôle")
    assert result["keywords"] == ["adrenaline reward control"]
    assert result["search_queries"] == ["adrenaline reward control process"]


def test_text_model_marked_ready_logs_reason(monkeypatch, tmp_path, caplog):
    payload = {"text_ready": ["text-model"], "broken": []}
    _, readiness_log = _initialise_with_ready(
        monkeypatch,
        tmp_path,
        caplog,
        payload=payload,
        base_model="base-model",
        text_model="text-model",
        json_model="json-model",
    )

    assert readiness_log["readiness_reason"] == "configured model text-model marked ready"
    assert readiness_log["chosen_text_model"] == "text-model"
    assert readiness_log["chosen_json_model"] == "json-model"
    assert readiness_log["readiness_filename"] == "llm_ready.json"


def test_text_model_broken_with_fallback_logs_reason(monkeypatch, tmp_path, caplog):
    payload = {"text_ready": ["fallback-model"], "broken": ["text-model"]}
    service, readiness_log = _initialise_with_ready(
        monkeypatch,
        tmp_path,
        caplog,
        payload=payload,
        base_model="base-model",
        text_model="text-model",
        json_model="json-model",
    )

    assert service.model_text == "fallback-model"
    assert readiness_log["readiness_reason"] == (
        "configured model text-model listed as broken; fallback to fallback-model"
    )
    assert readiness_log["fallback_target"] == "fallback-model"
    assert readiness_log["fallback_note"].endswith("listed as broken in llm_ready.json")


def test_text_model_broken_without_fallback_logs_reason(monkeypatch, tmp_path, caplog):
    payload = {"text_ready": [], "broken": ["text-model"]}
    service, readiness_log = _initialise_with_ready(
        monkeypatch,
        tmp_path,
        caplog,
        payload=payload,
        base_model="base-model",
        text_model="text-model",
        json_model="json-model",
    )

    assert service.model_text == "text-model"
    assert readiness_log["readiness_reason"] == (
        "configured model text-model listed as broken; no fallback available"
    )
    assert "fallback_target" not in readiness_log
    assert readiness_log["fallback_note"].endswith("listed as broken in llm_ready.json")


def test_readiness_log_includes_chosen_models(monkeypatch, tmp_path, caplog):
    payload = {"text_ready": ["alt-text"], "broken": []}
    _, readiness_log = _initialise_with_ready(
        monkeypatch,
        tmp_path,
        caplog,
        payload=payload,
        base_model="base-base",
        text_model="alt-text",
        json_model="json-variant",
    )

    assert readiness_log["chosen_text_model"] == "alt-text"
    assert readiness_log["chosen_json_model"] == "json-variant"
