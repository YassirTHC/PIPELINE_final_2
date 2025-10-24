from typing import List

import pytest

from pipeline_core.configuration import FetcherOrchestratorConfig, ProviderConfig
from pipeline_core.fetchers import FetcherOrchestrator, RemoteAssetCandidate


@pytest.fixture()
def orchestrator(monkeypatch):
    # Ensure API keys are present so providers are considered enabled.
    monkeypatch.setenv('PEXELS_API_KEY', 'test-pexels')
    monkeypatch.setenv('PIXABAY_API_KEY', 'test-pixabay')
    monkeypatch.setenv('COVERR_API_KEY', 'test-coverr')

    config = FetcherOrchestratorConfig(
        providers=(
            ProviderConfig(name='pexels', weight=0.5, max_results=3, timeout_s=2.0),
            ProviderConfig(name='coverr', weight=0.3, max_results=3, timeout_s=2.0),
            ProviderConfig(name='pixabay', weight=0.2, max_results=3, timeout_s=2.0),
        ),
        per_segment_limit=6,
        allow_videos=True,
        allow_images=False,
        parallel_requests=1,
    )

    orch = FetcherOrchestrator(config=config)

    def fake_run(self, provider_conf, query, filters, segment_timeout, limit):
        provider_name = provider_conf.name.strip().lower()
        results: List[RemoteAssetCandidate] = []
        for idx in range(limit):
            results.append(
                RemoteAssetCandidate(
                    provider=provider_name,
                    url=f"https://example.com/{provider_name}/{idx}.mp4",
                    thumb_url=None,
                    width=1920,
                    height=1080,
                    duration=6.0,
                    title=f"{provider_conf.name}-{idx}",
                    identifier=f"{provider_conf.name}-{idx}",
                    tags=(),
                )
            )
        return results, 40, None

    monkeypatch.setattr(FetcherOrchestrator, "_run_provider_fetch", fake_run, raising=False)
    return orch


def test_provider_mix_respected(orchestrator):
    candidates = orchestrator.fetch_candidates(["motivation", "focus"])
    snapshot = orchestrator.get_runtime_snapshot()
    provider_mix = snapshot.get('provider_mix', {})

    assert len(candidates) == orchestrator.config.per_segment_limit
    assert set(provider_mix.keys()) == {'pexels', 'coverr', 'pixabay'}
    assert provider_mix['pexels'] >= provider_mix['coverr'] >= provider_mix['pixabay']
    # Ensure diversity greater than single provider dominance.
    assert provider_mix['pexels'] < len(candidates)
