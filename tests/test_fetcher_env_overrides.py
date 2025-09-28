import pytest

from pipeline_core.configuration import FetcherOrchestratorConfig
from pipeline_core.fetchers import FetcherOrchestrator, RemoteAssetCandidate


@pytest.fixture(autouse=True)
def _restore_env(monkeypatch):
    # Ensure provider API keys exist so orchestrator does not skip providers.
    monkeypatch.setenv("PEXELS_API_KEY", "test-key")
    monkeypatch.setenv("PIXABAY_API_KEY", "test-key")
    yield


def test_config_reads_environment_defaults(monkeypatch):
    monkeypatch.setenv("FETCH_MAX", "4")
    monkeypatch.setenv("BROLL_FETCH_PROVIDER", "pixabay")
    monkeypatch.setenv("BROLL_FETCH_ALLOW_IMAGES", "1")
    monkeypatch.setenv("BROLL_FETCH_ALLOW_VIDEOS", "0")

    config = FetcherOrchestratorConfig.from_environment()

    enabled = [provider for provider in config.providers if provider.enabled]
    assert [provider.name for provider in enabled] == ["pixabay"]
    assert all(provider.max_results == 4 for provider in enabled)
    assert config.per_segment_limit == 4
    assert config.allow_images is True
    assert config.allow_videos is False


def test_fetcher_respects_allow_flags(monkeypatch):
    monkeypatch.setenv("BROLL_FETCH_MAX_PER_KEYWORD", "3")
    monkeypatch.setenv("BROLL_FETCH_PROVIDER", "pexels")
    monkeypatch.setenv("BROLL_FETCH_ALLOW_VIDEOS", "0")

    config = FetcherOrchestratorConfig.from_environment()
    orchestrator = FetcherOrchestrator(config)

    calls = []

    def _boom(self, *_args, **_kwargs):
        calls.append(True)
        return [RemoteAssetCandidate(
            provider="pexels",
            url="http://example.com/video.mp4",
            thumb_url=None,
            width=720,
            height=1280,
            duration=2.0,
            title="demo",
            identifier="vid-1",
            tags=(),
        )]

    monkeypatch.setattr(FetcherOrchestrator, "_run_provider_fetch", _boom)
    monkeypatch.setattr(FetcherOrchestrator, "_build_queries", lambda self, keywords: ["demo"])

    assert orchestrator.fetch_candidates(["demo"]) == []
    assert calls == []


def test_fetcher_applies_segment_limit(monkeypatch):
    monkeypatch.setenv("BROLL_FETCH_MAX_PER_KEYWORD", "2")
    monkeypatch.setenv("BROLL_FETCH_PROVIDER", "pexels")
    monkeypatch.setenv("BROLL_FETCH_ALLOW_VIDEOS", "1")

    config = FetcherOrchestratorConfig.from_environment()
    orchestrator = FetcherOrchestrator(config)

    candidate = RemoteAssetCandidate(
        provider="pexels",
        url="http://example.com/video.mp4",
        thumb_url=None,
        width=720,
        height=1280,
        duration=2.0,
        title="demo",
        identifier="vid-",
        tags=(),
    )

    def _capped(self, *_args, **_kwargs):
        return [candidate, candidate, candidate]

    monkeypatch.setattr(FetcherOrchestrator, "_run_provider_fetch", _capped)
    monkeypatch.setattr(FetcherOrchestrator, "_build_queries", lambda self, keywords: ["demo"])

    results = orchestrator.fetch_candidates(["demo"])
    assert len(results) == 2
