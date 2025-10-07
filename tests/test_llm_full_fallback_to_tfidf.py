import pytest

from pipeline_core import llm_service


class _FailingClient:
    provider_name = "together"

    def complete_json(self, prompt: str, schema, timeout_s: float) -> str:
        raise RuntimeError("json failed")

    def complete_text(self, prompt: str, timeout_s: float) -> str:
        raise RuntimeError("text failed")


@pytest.fixture(autouse=True)
def _patch_env(monkeypatch):
    monkeypatch.setenv("PIPELINE_LLM_PROVIDER", "together")
    client = _FailingClient()
    monkeypatch.setattr(llm_service, "get_llm_client", lambda settings=None: client)
    monkeypatch.setattr(llm_service, "tfidf_fallback_disabled_from_env", lambda: False)
    calls = {"count": 0}

    def _fake_tfidf(transcript: str, *, top_k: int = 12):
        calls["count"] += 1
        keywords = [
            "city skyline",
            "night traffic",
            "aerial view",
            "urban sunset",
            "neon lights",
            "river bridge",
            "city crowd",
            "street market",
            "evening commute",
            "office towers",
            "harbor lights",
            "downtown view",
        ]
        queries = [
            "city skyline",
            "night traffic",
            "aerial city",
            "urban sunset",
            "neon downtown",
            "river bridge",
            "busy street",
            "market crowd",
            "evening commute",
            "office towers",
            "harbor lights",
            "downtown view",
        ]
        return keywords[:top_k], queries[:top_k]

    monkeypatch.setattr(llm_service, "_tfidf_fallback", _fake_tfidf)
    yield calls


def test_full_fallback_to_tfidf(_patch_env):
    transcript = "City montage describing lights, skyline and nightlife."

    result = llm_service.generate_metadata_json(transcript)

    assert _patch_env["count"] == 1
    assert len(result["hashtags"]) == 5
    assert len(result["broll_keywords"]) == 8
    assert len(result["queries"]) == 12
    assert result["broll_keywords"][0] == "city skyline"
