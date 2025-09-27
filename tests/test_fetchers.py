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
