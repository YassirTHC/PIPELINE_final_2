import sys
import types

import pytest

# Stub src.pipeline.fetchers before importing
src_module = types.ModuleType('src')
pipeline_module = types.ModuleType('src.pipeline')
fetchers_module = types.ModuleType('src.pipeline.fetchers')
fetchers_module.build_search_query = lambda keywords: keywords[0] if keywords else ''
fetchers_module.pexels_search_videos = lambda *args, **kwargs: []
fetchers_module.pixabay_search_videos = lambda *args, **kwargs: []
fetchers_module._best_vertical_video_file = lambda item: {}
fetchers_module._pixabay_best_video_url = lambda item: ''

sys.modules.setdefault('src', src_module)
sys.modules.setdefault('src.pipeline', pipeline_module)
sys.modules['src.pipeline.fetchers'] = fetchers_module

from pipeline_core.configuration import FetcherOrchestratorConfig, ProviderConfig
from pipeline_core.fetchers import FetcherOrchestrator


class DummyLogger:
    def __init__(self):
        self.events = []

    def log(self, payload):
        self.events.append(payload)


@pytest.fixture(autouse=True)
def clear_provider_env(monkeypatch):
    monkeypatch.delenv('PEXELS_API_KEY', raising=False)
    monkeypatch.delenv('PIXABAY_API_KEY', raising=False)


def test_providers_skipped_without_keys(monkeypatch):
    logger = DummyLogger()
    config = FetcherOrchestratorConfig(
        providers=(
            ProviderConfig(name='pexels'),
            ProviderConfig(name='pixabay'),
        ),
        per_segment_limit=2,
    )
    orchestrator = FetcherOrchestrator(config, event_logger=logger)
    results = orchestrator.fetch_candidates(['marketing strategy'], segment_timeout_s=0.5)
    assert results == []
    skipped = [evt['provider'] for evt in logger.events if evt.get('event') == 'provider_skipped_missing_key']
    assert skipped.count('pexels') == 1
    assert skipped.count('pixabay') == 1
