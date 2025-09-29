import json

import pytest

from pipeline_core import llm_service


def test_keywords_prompt_schema():
    prompt = llm_service._build_keywords_prompt("Example transcript", "en")
    assert "Return ONLY a JSON object with keys: title (string), description (string), queries (array of 8-12 short search queries in en), broll_keywords (array of 8-12 short visual noun phrases in en)." in prompt
    assert "broll_keywords" in prompt
    assert "queries" in prompt
    assert "hashtags" not in prompt.lower()


def test_non_empty_keywords(monkeypatch):
    monkeypatch.setenv("PIPELINE_LLM_KEYWORDS_FIRST", "1")
    monkeypatch.setenv("PIPELINE_LLM_DISABLE_HASHTAGS", "1")

    payload = {
        "title": "Short Title",
        "description": "Short description",
        "queries": ["studio lighting"],
        "broll_keywords": ["modern office"],
    }

    def fake_generate(prompt: str, **kwargs):
        return payload, {"response": json.dumps(payload)}, len(json.dumps(payload))

    monkeypatch.setattr(llm_service, "_ollama_generate_json", fake_generate)

    transcript = "Marketing teams discuss growth strategies, customer journeys, analytics dashboards, and product demos."
    result = llm_service.generate_metadata_as_json(transcript)

    assert result["title"] == "Short Title"
    assert result["description"] == "Short description"
    assert len(result["queries"]) >= 8
    assert len(result["broll_keywords"]) >= 8
    assert result["queries"][0] == "studio lighting"
    assert result["broll_keywords"][0] == "modern office"


def test_generate_hints_for_segment_integrates(monkeypatch):
    monkeypatch.setenv("PIPELINE_LLM_KEYWORDS_FIRST", "1")

    expected_queries = [
        "marketing analytics dashboard",
        "team planning session",
        "customer success meeting",
        "digital strategy workshop",
        "startup founders collaboration",
        "audience engagement presentation",
        "business growth chart",
        "product demo recording",
    ]
    expected_keywords = [
        "marketing team collaboration",
        "analytics dashboard closeup",
        "customer success handshake",
        "digital workshop whiteboard",
        "startup founders meeting",
        "audience engagement speaker",
        "business growth presentation",
        "product demo studio",
    ]

    captured = {}

    def fake_call_llm(self, prompt: str, max_tokens: int = 192):
        captured["prompt"] = prompt
        return json.dumps(
            {
                "title": "Segment Title",
                "description": "Segment Description",
                "queries": expected_queries,
                "broll_keywords": expected_keywords,
            }
        )

    monkeypatch.setattr(llm_service.LLMMetadataGeneratorService, "_call_llm", fake_call_llm, raising=False)

    service = llm_service.LLMMetadataGeneratorService(reuse_shared=False)
    result = service.generate_hints_for_segment(
        "The host explains marketing analytics and customer growth strategies with vivid examples.",
        0.0,
        10.0,
    )

    assert captured["prompt"].count("title (string)") == 1
    assert "hashtags" not in captured["prompt"].lower()
    assert result["title"] == "Segment Title"
    assert result["description"] == "Segment Description"
    assert result["queries"] == expected_queries
    assert result["broll_keywords"] == expected_keywords
    assert isinstance(result.get("filters"), dict)


def test_service_exposes_call_llm_and_fallbacks(monkeypatch):
    monkeypatch.setattr(
        llm_service,
        "_LAST_METADATA_KEYWORDS",
        {"values": [], "updated_at": 0.0},
        raising=False,
    )
    monkeypatch.setattr(
        llm_service,
        "_LAST_METADATA_QUERIES",
        {"values": [], "updated_at": 0.0},
        raising=False,
    )

    def fail_call_llm(self, prompt: str, max_tokens: int = 192):
        raise ValueError("timeout")

    monkeypatch.setattr(
        llm_service.LLMMetadataGeneratorService,
        "_call_llm",
        fail_call_llm,
        raising=False,
    )

    service = llm_service.LLMMetadataGeneratorService(reuse_shared=False)
    result = service.generate_hints_for_segment(
        "Marketing teams review analytics dashboards, share growth metrics, and plan new campaigns.",
        0.0,
        10.0,
    )

    queries = [query for query in result.get("queries", []) if query]
    assert len(queries) >= 4


def test_disable_hashtags_env_clears_result(monkeypatch):
    monkeypatch.setenv("PIPELINE_LLM_DISABLE_HASHTAGS", "1")
    monkeypatch.setattr(
        llm_service,
        "_LAST_METADATA_KEYWORDS",
        {"values": [], "updated_at": 0.0},
        raising=False,
    )
    monkeypatch.setattr(
        llm_service,
        "_LAST_METADATA_QUERIES",
        {"values": [], "updated_at": 0.0},
        raising=False,
    )

    payload = {
        "title": "Campaign Strategy",
        "description": "A quick summary of the marketing campaign roadmap.",
        "hashtags": ["#GrowthMarketing", "#CampaignLaunch"],
        "broll_keywords": [
            "marketing team meeting",
            "analytics dashboard review",
            "digital strategy planning",
            "startup founders collaboration",
            "customer journey mapping",
            "social media scheduling",
        ],
        "queries": [
            "marketing team collaboration",
            "analytics dashboard presentation",
            "digital strategy meeting",
            "startup founders planning",
            "customer journey workshop",
            "social media strategy session",
        ],
    }

    def fake_generate_json(*args, **kwargs):
        serialized = json.dumps(payload)
        return payload, {"response": serialized}, len(serialized)

    monkeypatch.setattr(llm_service, "_ollama_generate_json", fake_generate_json)

    result = llm_service.generate_metadata_as_json(
        "The host outlines a marketing campaign, reviews analytics, and discusses customer journeys."
    )

    assert result["hashtags"] == []

