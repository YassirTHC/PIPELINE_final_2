import pytest

from pipeline_core.configuration import FetcherOrchestratorConfig, ProviderConfig
from pipeline_core.fetchers import FetcherOrchestrator


@pytest.mark.parametrize(
    "keywords",
    [
        ["  Hello   world  ", "HELLO WORLD", "hello world"],
        ["FOCUS   mode", "focus mode", "focus   mode  "],
    ],
)
def test_fetch_candidates_normalize_and_deduplicate(monkeypatch, keywords):
    provider = ProviderConfig(name="pixabay", enabled=True, max_results=6, supports_videos=True)
    config = FetcherOrchestratorConfig(
        providers=(provider,),
        allow_videos=True,
        allow_images=False,
        per_segment_limit=6,
    )
    orchestrator = FetcherOrchestrator(config)

    monkeypatch.setenv("PIXABAY_API_KEY", "token")

    calls = []

    def _fake_pixabay_search(api_key, query, *, per_page=50):
        calls.append({"api_key": api_key, "query": query, "per_page": per_page})
        return []

    monkeypatch.setattr("pipeline_core.fetchers.pixabay_search_videos", _fake_pixabay_search)
    monkeypatch.setattr("pipeline_core.fetchers._pixabay_best_video_url", lambda payload: None)

    orchestrator.fetch_candidates(keywords, filters=None)

    assert calls, "expected a provider call"
    assert len(calls) == 1
    assert calls[0]["query"] == " ".join(keywords[0].split())


def test_pixabay_per_page_limits(monkeypatch):
    monkeypatch.setenv("PIXABAY_API_KEY", "token")

    calls = []

    def _fake_pixabay_search(api_key, query, *, per_page=50):
        calls.append({"query": query, "per_page": per_page})
        return []

    monkeypatch.setattr("pipeline_core.fetchers.pixabay_search_videos", _fake_pixabay_search)
    monkeypatch.setattr("pipeline_core.fetchers._pixabay_best_video_url", lambda payload: None)

    high_provider = ProviderConfig(name="pixabay", enabled=True, max_results=300)
    high_config = FetcherOrchestratorConfig(
        providers=(high_provider,),
        allow_videos=True,
        allow_images=False,
        per_segment_limit=12,
    )
    high_orchestrator = FetcherOrchestrator(high_config)
    high_orchestrator.fetch_candidates(["wide limit"], filters=None)

    assert calls[0]["per_page"] == 200

    calls.clear()

    small_provider = ProviderConfig(name="pixabay", enabled=True, max_results=1)
    small_config = FetcherOrchestratorConfig(
        providers=(small_provider,),
        allow_videos=True,
        allow_images=False,
        per_segment_limit=1,
    )
    small_orchestrator = FetcherOrchestrator(small_config)
    small_orchestrator.fetch_candidates(["tight limit"], filters=None)

    assert calls[0]["per_page"] == 50
    assert calls[0]["query"] == "tight limit"
