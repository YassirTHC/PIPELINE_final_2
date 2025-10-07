import json

import pytest

from pipeline_core import llm_service


class _HappyClient:
    provider_name = "openai"

    def complete_json(self, prompt: str, schema, timeout_s: float) -> str:
        payload = {
            "title": "Clip Title",
            "description": "Concise description of the clip.",
            "hashtags": ["#Story", "#Focus", "#Energy", "#Growth", "#Success"],
            "broll_keywords": [
                "office teamwork",
                "brainstorm meeting",
                "whiteboard planning",
                "startup launch",
                "team celebration",
                "hands typing",
                "closeup notebook",
                "city skyline",
            ],
            "queries": [
                "team brainstorming",
                "office collaboration",
                "startup planning",
                "business strategy",
                "creative meeting",
                "whiteboard ideas",
                "celebration success",
                "typing hands",
                "entrepreneur focus",
                "goal setting",
                "motivated founders",
                "city skyline",
            ],
        }
        return json.dumps(payload)

    def complete_text(self, prompt: str, timeout_s: float) -> str:  # pragma: no cover - not expected
        raise AssertionError("text completion should not be called in happy path")


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    monkeypatch.setenv("PIPELINE_LLM_PROVIDER", "openai")
    monkeypatch.setattr(llm_service, "get_llm_client", lambda settings=None: _HappyClient())
    monkeypatch.setattr(llm_service, "tfidf_fallback_disabled_from_env", lambda: False)
    yield


def test_generate_metadata_json_happy_path():
    transcript = "Team discusses strategy for upcoming product launch."

    result = llm_service.generate_metadata_json(transcript)

    assert result["title"] == "Clip Title"
    assert result["description"].startswith("Concise description")
    assert result["hashtags"] == ["#Story", "#Focus", "#Energy", "#Growth", "#Success"]
    assert result["broll_keywords"][0] == "office teamwork"
    assert len(result["broll_keywords"]) == 8
    assert result["queries"][0] == "team brainstorming"
    assert len(result["queries"]) == 12
