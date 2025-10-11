import json
import types

import pipeline_core.llm_service as ls


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload


def test_metadata_json_ranges_kept(monkeypatch):
    # sample valid payload inside new ranges
    llm_json = {
        "title": "7 Habits To Escape Rat Race",
        "description": "A punchy summary of the core ideas.",
        "hashtags": [
            "#money",
            "#sidehustle",
            "#freelance",
            "#ai",
            "#tiktoktips",
            "#growth",
        ],
        "broll_keywords": [
            "home desk setup",
            "typing hands",
            "budget spreadsheet",
            "whiteboard planning",
            "late night coding",
            "city timelapse",
            "phone notifications",
            "coffee mug",
        ],
        "queries": [
            "typing on laptop",
            "budget spreadsheet screen",
            "whiteboard planning office",
            "night city timelapse",
            "phone notification closeup",
            "coffee mug steam",
            "freelancer home office",
            "ai workflow broll",
        ],
    }

    def fake_generate_json(*_args, **_kwargs):
        return llm_json, {"response": json.dumps(llm_json)}, len(json.dumps(llm_json))

    monkeypatch.setattr(ls, "_keywords_first_enabled", lambda: True, raising=True)
    monkeypatch.setattr(ls, "_ollama_generate_json", fake_generate_json, raising=True)
    monkeypatch.setattr(ls, "_ollama_json", lambda *args, **kwargs: llm_json, raising=True)

    result = ls.generate_metadata_as_json(
        transcript="dummy text",
        model="qwen3:8b",
        endpoint="http://localhost:11434",
    )

    assert result["title"] == llm_json["title"]
    assert result["description"] == llm_json["description"]
    assert result["hashtags"] == llm_json["hashtags"]
    assert len(result["broll_keywords"]) == len(llm_json["broll_keywords"])
    assert set(result["broll_keywords"]) == set(llm_json["broll_keywords"])
    assert len(result["queries"]) == len(llm_json["queries"])
    assert set(result["queries"]) == set(llm_json["queries"])
