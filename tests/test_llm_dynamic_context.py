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
            "keywords": ["DOPAMINE_boost", "focus", "focus"],
            "synonyms": {"DOPAMINE_boost": ["dopamine surge", "chemical boost"]},
            "search_queries": ["brain focus shot", "nice people"],
            "segment_briefs": [
                {"segment_index": "0", "keywords": ["Neural_Pathway"], "queries": ["brain first"]}
            ],
        }
    )
    service = DummyService(payload)
    result = service.generate_dynamic_context("Dopamine keeps you motivated")
    assert result["detected_domains"][0]["name"] == "neuro science"
    assert result["keywords"] == ["dopamine boost", "focus"]
    assert result["search_queries"] == ["brain focus shot"]
    assert result["segment_briefs"][0]["keywords"] == ["neural pathway"]
    assert result["segment_briefs"][0]["queries"] == []


def test_dynamic_context_fallback_keywords():
    class Failing(LLMMetadataGeneratorService):
        def _complete_text(self, prompt: str) -> str:  # type: ignore[override]
            raise RuntimeError("boom")

    service = Failing(reuse_shared=False)
    transcript = "We discuss marketing funnels and conversion metrics for data driven teams."
    result = service.generate_dynamic_context(transcript)
    assert result["keywords"], "fallback keywords missing"
    assert all(len(term) >= 3 for term in result["keywords"])
