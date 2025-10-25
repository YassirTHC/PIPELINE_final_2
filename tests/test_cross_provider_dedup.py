# -*- coding: utf-8 -*-
import pytest

from pipeline_core.configuration import FetcherOrchestratorConfig, ProviderConfig
from pipeline_core.fetchers import FetcherOrchestrator, RemoteAssetCandidate


@pytest.fixture()
def orchestrator(monkeypatch):
    monkeypatch.setenv('PEXELS_API_KEY', 'dedup')
    monkeypatch.setenv('PIXABAY_API_KEY', 'dedup')
    monkeypatch.setenv('COVERR_API_KEY', 'dedup')
    monkeypatch.setenv('BROLL_MAX_URL_REUSE', '1')

    config = FetcherOrchestratorConfig(
        providers=(
            ProviderConfig(name='pexels', weight=0.4, max_results=2, timeout_s=2.0),
            ProviderConfig(name='coverr', weight=0.3, max_results=2, timeout_s=2.0),
            ProviderConfig(name='pixabay', weight=0.3, max_results=2, timeout_s=2.0),
        ),
        per_segment_limit=5,
        allow_videos=True,
        allow_images=False,
        parallel_requests=1,
    )

    orch = FetcherOrchestrator(config=config)

    def fake_run(self, provider_conf, query, filters, segment_timeout, limit):
        provider_name = provider_conf.name.strip().lower()
        candidate = RemoteAssetCandidate(
            provider=provider_name,
            url="https://example.com/shared.mp4",
            thumb_url=None,
            width=1920,
            height=1080,
            duration=5.0,
            title="Shared Clip",
            identifier="shared-asset",
            tags=(),
        )
        return [candidate], 25, None

    monkeypatch.setattr(FetcherOrchestrator, "_run_provider_fetch", fake_run, raising=False)
    return orch


def test_cross_provider_dedup_respects_cap(orchestrator):
    results = orchestrator.fetch_candidates(["discipline"])
    assert len(results) == 1
    snapshot = orchestrator.get_runtime_snapshot()
    provider_states = snapshot.get('provider_states', {})
    assert any(state['reuse_dropped'] >= 1 for state in provider_states.values())

