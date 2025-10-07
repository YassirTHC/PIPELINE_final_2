import json

import pytest

from pipeline_core import llm_service


class _FallbackClient:
    provider_name = "groq"

    def __init__(self):
        self.json_calls = 0

    def complete_json(self, prompt: str, schema, timeout_s: float) -> str:
        self.json_calls += 1
        return "not-json"  # force parse failure

    def complete_text(self, prompt: str, timeout_s: float) -> str:
        payload = {
            "title": "Recovered Title",
            "description": "Recovered description from text mode.",
            "hashtags": ["#One", "#Two", "#Three", "#Four", "#Five"],
            "broll_keywords": [
                "forest trail",
                "mountain sunrise",
                "drone landscape",
                "river valley",
                "adventure hiking",
                "travel couple",
                "campfire night",
                "stargazing sky",
            ],
            "queries": [
                "forest adventure",
                "mountain sunrise",
                "landscape drone",
                "river valley",
                "hiking couple",
                "campfire night",
                "stargazing sky",
                "travel nature",
                "morning hike",
                "outdoor journey",
                "sunrise vista",
                "nature escape",
            ],
        }
        return f"Here you go: {json.dumps(payload)}"


@pytest.fixture(autouse=True)
def _patch_client(monkeypatch):
    client = _FallbackClient()
    monkeypatch.setenv("PIPELINE_LLM_PROVIDER", "groq")
    monkeypatch.setattr(llm_service, "get_llm_client", lambda settings=None: client)
    monkeypatch.setattr(llm_service, "tfidf_fallback_disabled_from_env", lambda: False)
    yield client


def test_json_invalid_then_text_extract(_patch_client):
    transcript = "Beautiful travel vlog exploring nature trails and sunrise."

    result = llm_service.generate_metadata_json(transcript)

    assert result["title"] == "Recovered Title"
    assert result["hashtags"] == ["#One", "#Two", "#Three", "#Four", "#Five"]
    assert result["broll_keywords"][0] == "forest trail"
    assert len(result["queries"]) == 12
