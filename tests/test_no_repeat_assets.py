from types import SimpleNamespace, ModuleType
from pathlib import Path
import sys
import importlib
import importlib.machinery as machinery


def _load_video_processor():
    if 'video_processor' in sys.modules:
        return importlib.reload(sys.modules['video_processor'])

    stubs = {
        'temp_function': SimpleNamespace(_llm_generate_caption_hashtags_fixed=lambda *args, **kwargs: ""),
        'whisper': SimpleNamespace(load_model=lambda *args, **kwargs: SimpleNamespace(transcribe=lambda *a, **k: {})),
    }

    for name, module in stubs.items():
        sys.modules.setdefault(name, module)

    dotenv_module = ModuleType('dotenv')
    dotenv_module.load_dotenv = lambda *args, **kwargs: None
    sys.modules.setdefault('dotenv', dotenv_module)

    cv2_module = ModuleType('cv2')
    cv2_module.__spec__ = machinery.ModuleSpec('cv2', loader=None)
    sys.modules['cv2'] = cv2_module

    moviepy_module = ModuleType('moviepy')
    editor_module = ModuleType('moviepy.editor')

    class _FakeClip:
        def __init__(self, *args, **kwargs):
            self.duration = 0.0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write_videofile(self, *args, **kwargs):
            return None

    class _FakeComposite(_FakeClip):
        pass

    editor_module.VideoFileClip = _FakeClip
    editor_module.TextClip = _FakeClip
    editor_module.CompositeVideoClip = _FakeComposite
    moviepy_module.editor = editor_module
    sys.modules['moviepy'] = moviepy_module
    sys.modules['moviepy.editor'] = editor_module
    moviepy_config = ModuleType('moviepy.config')
    moviepy_config.IMAGEMAGICK_BINARY = None
    sys.modules['moviepy.config'] = moviepy_config

    broll_module = ModuleType('broll_selector')

    class _Dummy:
        def __init__(self, *args, **kwargs):
            pass

    broll_module.BrollSelector = _Dummy
    broll_module.Asset = _Dummy
    broll_module.ScoringFeatures = _Dummy
    broll_module.BrollCandidate = _Dummy
    sys.modules['broll_selector'] = broll_module

    pipeline_fetchers = ModuleType('pipeline_core.fetchers')

    class _FakeFetcherOrchestrator:
        def __init__(self, *args, **kwargs):
            self._args = args
            self._kwargs = kwargs

        def fetch_candidates(self, *args, **kwargs):
            return []

    pipeline_fetchers.FetcherOrchestrator = _FakeFetcherOrchestrator
    sys.modules['pipeline_core.fetchers'] = pipeline_fetchers

    pipeline_config = ModuleType('pipeline_core.configuration')

    class _FakePipelineConfigBundle:
        def __init__(self):
            self.fetcher = SimpleNamespace(providers=["stub"])
            self.selection = SimpleNamespace(min_score=0.5, prefer_landscape=True, min_duration_s=0.0)
            self.timeboxing = SimpleNamespace(fetch_rank_ms=0, request_timeout_s=1)

    pipeline_config.PipelineConfigBundle = _FakePipelineConfigBundle
    sys.modules['pipeline_core.configuration'] = pipeline_config

    pipeline_dedupe = ModuleType('pipeline_core.dedupe')
    pipeline_dedupe.compute_phash = lambda *args, **kwargs: None
    pipeline_dedupe.hamming_distance = lambda a, b: 64
    sys.modules['pipeline_core.dedupe'] = pipeline_dedupe

    pipeline_logging = ModuleType('pipeline_core.logging')

    class _DummyJsonlLogger:
        def __init__(self, path):
            self.path = Path(path)
            self.records = []

        def log(self, payload):
            self.records.append(payload)

    def _log_broll_decision(logger, *, segment_idx, start, end, query_count, candidate_count, unique_candidates,
                            url_dedup_hits, phash_dedup_hits, selected_url, selected_score, provider,
                            latency_ms, llm_healthy, reject_reasons, queries=None, provider_status=None,
                            best_score=None):
        event = 'broll_segment_decision' if segment_idx >= 0 else 'broll_session_summary'
        payload = {
            'event': event,
            'segment': segment_idx,
            't0': start,
            't1': end,
            'q_count': query_count,
            'candidates': candidate_count,
            'unique_candidates': unique_candidates,
            'dedup_url_hits': url_dedup_hits,
            'dedup_phash_hits': phash_dedup_hits,
            'selected_url': selected_url,
            'selected_score': selected_score,
            'provider': provider,
            'latency_ms': latency_ms,
            'llm_healthy': llm_healthy,
            'reject_reasons': sorted(set(reject_reasons or [])),
        }
        if queries is not None:
            payload['queries'] = list(queries)
        if provider_status is not None:
            payload['providers'] = provider_status
        if best_score is not None:
            payload['best_score'] = best_score
        logger.log(payload)

    pipeline_logging.JsonlLogger = _DummyJsonlLogger
    pipeline_logging.log_broll_decision = _log_broll_decision
    sys.modules['pipeline_core.logging'] = pipeline_logging

    pipeline_llm = ModuleType('pipeline_core.llm_service')

    class _DummyLLMService:
        def generate_hints_for_segment(self, *args, **kwargs):
            return {}

    pipeline_llm.LLMMetadataGeneratorService = _DummyLLMService
    pipeline_llm.enforce_fetch_language = lambda queries, language=None: list(queries or [])
    sys.modules['pipeline_core.llm_service'] = pipeline_llm

    return importlib.import_module('video_processor')


def test_dedupe_by_url_prevents_reuse():
    vp = _load_video_processor()
    vp.SEEN_URLS.clear()
    candidates = [
        SimpleNamespace(url="https://example.com/a.mp4"),
        SimpleNamespace(url="https://example.com/a.mp4"),
        SimpleNamespace(url="https://example.com/b.mp4"),
    ]
    unique, hits = vp.dedupe_by_url(candidates)
    assert len(unique) == 2
    assert hits == 1

    vp.SEEN_URLS.add("https://example.com/a.mp4")
    unique, hits = vp.dedupe_by_url([SimpleNamespace(url="https://example.com/a.mp4")])
    assert not unique
    assert hits == 1


def test_dedupe_by_identifier_prevents_reuse():
    vp = _load_video_processor()
    vp.SEEN_IDENTIFIERS.clear()
    vp.SEEN_URLS.clear()
    candidates = [
        SimpleNamespace(url="https://cdn/1.mp4", identifier="pexels-42"),
        SimpleNamespace(url="https://cdn/alt.mp4", identifier="pexels-42"),
    ]
    unique, hits = vp.dedupe_by_url(candidates)
    assert len(unique) == 1
    assert hits == 1


def test_fallback_selects_candidate_when_min_score_too_high():
    vp = _load_video_processor()
    vp.SEEN_URLS.clear()
    vp.SEEN_PHASHES.clear()
    vp.SEEN_IDENTIFIERS.clear()

    events = []

    class DummyLogger:
        def log(self, payload):
            events.append(payload)

    processor = vp.VideoProcessor.__new__(vp.VideoProcessor)
    processor._pipeline_config = SimpleNamespace(
        fetcher=SimpleNamespace(providers=["stub"]),
        selection=SimpleNamespace(min_score=0.95, prefer_landscape=True, min_duration_s=0.0),
        timeboxing=SimpleNamespace(fetch_rank_ms=0, request_timeout_s=1),
    )
    processor._core_last_run_used = False
    processor._llm_service = None
    processor._dyn_context = {}
    processor._selector_keywords = []
    processor._fetch_keywords = []
    dummy_logger = DummyLogger()
    processor._get_broll_event_logger = lambda: dummy_logger
    processor._derive_segment_keywords = lambda segment, keywords: ["keyword"]

    candidate = SimpleNamespace(
        url="https://cdn/fallback.mp4",
        identifier="asset-1",
        width=1920,
        height=1080,
        duration=4.0,
        provider="stub",
    )

    def fake_fetch_candidates(*args, **kwargs):
        return [candidate]

    original_fetcher = vp.FetcherOrchestrator
    original_dedupe_by_phash = vp.dedupe_by_phash
    try:
        vp.FetcherOrchestrator = lambda cfg: SimpleNamespace(fetch_candidates=fake_fetch_candidates)
        vp.dedupe_by_phash = lambda candidates: (candidates, 0)

        processor._insert_brolls_pipeline_core(
            segments=[SimpleNamespace(start=0.0, end=4.0, text="hello world")],
            broll_keywords=["keyword"],
            subtitles=None,
            input_path=Path("video.mp4"),
        )
    finally:
        vp.FetcherOrchestrator = original_fetcher
        vp.dedupe_by_phash = original_dedupe_by_phash

    assert "https://cdn/fallback.mp4" in vp.SEEN_URLS
    assert "asset-1" in vp.SEEN_IDENTIFIERS
    decision_events = [event for event in events if event.get("event") == "broll_segment_decision"]
    assert decision_events, "expected a broll decision event"
    selected = decision_events[0]
    assert selected.get("selected_url") == "https://cdn/fallback.mp4"
    assert "fallback_low_score" in selected.get("reject_reasons", [])
