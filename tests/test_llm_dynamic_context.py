import json

import pytest

from pipeline_core.llm_service import LLMMetadataGeneratorService


class DummyService(LLMMetadataGeneratorService):
    def __init__(self, payload: str):
        super().__init__(reuse_shared=False)
        self._payload = payload

    def _complete_text(self, prompt: str) -> str:  # type: ignore[override]
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


def test_dynamic_context_fallback_keywords():
    class Failing(LLMMetadataGeneratorService):
        def _complete_text(self, prompt: str) -> str:  # type: ignore[override]
            raise RuntimeError("boom")

    service = Failing(reuse_shared=False)
    transcript = "We discuss marketing funnels and conversion metrics for data driven teams."
    result = service.generate_dynamic_context(transcript)
    assert result["keywords"], "fallback keywords missing"
    assert all(len(term) >= 3 for term in result["keywords"])


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
