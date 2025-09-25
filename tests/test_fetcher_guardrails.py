import sys
from types import ModuleType

import os


def test_fetcher_skips_provider_without_key(monkeypatch):
    monkeypatch.delenv('PEXELS_API_KEY', raising=False)
    import config as project_config
    monkeypatch.setattr(project_config.Config, 'PEXELS_API_KEY', None, raising=False)

    called = {'value': False}

    def fail_if_called(*_args, **_kwargs):
        called['value'] = True
        raise AssertionError('pexels_search_videos should not be invoked when key is missing')

    dummy_src = ModuleType('src')
    dummy_pipeline = ModuleType('src.pipeline')
    dummy_fetchers = ModuleType('src.pipeline.fetchers')
    dummy_fetchers.build_search_query = lambda keywords, **_kwargs: list(keywords)
    dummy_fetchers.pexels_search_videos = fail_if_called
    dummy_fetchers.pixabay_search_videos = lambda *_args, **_kwargs: []
    dummy_fetchers._best_vertical_video_file = lambda *_args, **_kwargs: None
    dummy_fetchers._pixabay_best_video_url = lambda *_args, **_kwargs: None

    sys.modules['src'] = dummy_src
    sys.modules['src.pipeline'] = dummy_pipeline
    sys.modules['src.pipeline.fetchers'] = dummy_fetchers

    from pipeline_core.configuration import FetcherOrchestratorConfig, ProviderConfig
    from pipeline_core import fetchers as pc_fetchers

    config = FetcherOrchestratorConfig(
        providers=(ProviderConfig(name='pexels', enabled=True, max_results=2),)
    )

    orchestrator = pc_fetchers.FetcherOrchestrator(config)
    candidates = orchestrator.fetch_candidates(['sample'])

    assert candidates == []
    assert called['value'] is False
