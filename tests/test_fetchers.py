import pytest

from pipeline_core.configuration import FetcherOrchestratorConfig, ProviderConfig
from pipeline_core.fetchers import FetcherOrchestrator, RemoteAssetCandidate


def test_fetch_candidates_returns_unfiltered_results(monkeypatch):
    config = FetcherOrchestratorConfig(
        providers=(ProviderConfig(name="dummy", enabled=True, max_results=3),)
    )
    orchestrator = FetcherOrchestrator(config)

    candidate = RemoteAssetCandidate(
        provider="dummy",
        url="http://example.com/video.mp4",
        thumb_url=None,
        width=720,
        height=1280,
        duration=2.0,
        title="demo",
        identifier="vid-1",
        tags=(),
    )

    monkeypatch.setattr(
        FetcherOrchestrator,
        "_build_queries",
        lambda self, keywords: ["demo"],
    )
    monkeypatch.setattr(
        FetcherOrchestrator,
        "_run_provider_fetch",
        lambda self, provider_conf, query, filters, segment_timeout: [candidate],
    )

    results = orchestrator.fetch_candidates(["demo"], filters={"orientation": "landscape"})

    assert results == [candidate]


def test_fetch_candidates_skip_video_providers_when_videos_disabled(monkeypatch):
    provider = ProviderConfig(name="pexels", enabled=True, max_results=3)
    config = FetcherOrchestratorConfig(
        providers=(provider,),
        allow_images=True,
        allow_videos=False,
        per_segment_limit=3,
    )
    orchestrator = FetcherOrchestrator(config)

    monkeypatch.setattr(
        FetcherOrchestrator,
        "_build_queries",
        lambda self, keywords: ["demo"],
    )

    run_calls = {"provider": 0, "fallback": 0}

    def _fake_run_provider_fetch(self, provider_conf, query, filters, segment_timeout):
        run_calls["provider"] += 1
        return []

    def _fake_run_pixabay_fallback(self, queries, limit, *, segment_index=None):
        run_calls["fallback"] += 1
        return []

    monkeypatch.setattr(FetcherOrchestrator, "_run_provider_fetch", _fake_run_provider_fetch)
    monkeypatch.setattr(FetcherOrchestrator, "_run_pixabay_fallback", _fake_run_pixabay_fallback)

    results = orchestrator.fetch_candidates(["demo"], filters=None)

    assert results == []
    assert run_calls == {"provider": 0, "fallback": 0}


@pytest.mark.parametrize("provider_name", ["pexels", "pixabay"])
def test_video_providers_allowed_when_images_disabled(monkeypatch, provider_name):
    provider = ProviderConfig(
        name=provider_name,
        enabled=True,
        max_results=3,
        supports_images=False,
        supports_videos=True,
    )
    config = FetcherOrchestratorConfig(
        providers=(provider,),
        allow_images=False,
        allow_videos=True,
        per_segment_limit=3,
    )
    orchestrator = FetcherOrchestrator(config)

    monkeypatch.setenv("PEXELS_API_KEY", "token")
    monkeypatch.setenv("PIXABAY_API_KEY", "token")

    monkeypatch.setattr(
        FetcherOrchestrator,
        "_build_queries",
        lambda self, keywords: ["demo"],
    )

    provider_calls = []

    def _fake_run_provider_fetch(self, provider_conf, query, filters, segment_timeout):
        provider_calls.append((provider_conf.name, query))
        return []

    def _fake_run_pixabay_fallback(self, queries, limit, *, segment_index=None):
        return []

    monkeypatch.setattr(FetcherOrchestrator, "_run_provider_fetch", _fake_run_provider_fetch)
    monkeypatch.setattr(FetcherOrchestrator, "_run_pixabay_fallback", _fake_run_pixabay_fallback)

    results = orchestrator.fetch_candidates(["demo"], filters=None)

    assert results == []
    assert provider_calls == [(provider_name, "demo")]
