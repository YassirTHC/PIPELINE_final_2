from types import SimpleNamespace, MethodType, ModuleType

import pytest

import sys


if "cv2" not in sys.modules:
    sys.modules["cv2"] = SimpleNamespace(
        cvtColor=lambda *args, **kwargs: None,
        Canny=lambda *args, **kwargs: [],
        findContours=lambda *args, **kwargs: ([], []),
        moments=lambda *args, **kwargs: [],
        contourArea=lambda *args, **kwargs: 0,
        resize=lambda *args, **kwargs: None,
        COLOR_RGB2GRAY=0,
        RETR_EXTERNAL=0,
        CHAIN_APPROX_SIMPLE=0,
        INTER_LANCZOS4=0,
    )

if "src.pipeline.fetchers" not in sys.modules:
    fetchers_stub = ModuleType("src.pipeline.fetchers")
    fetchers_stub.build_search_query = lambda *args, **kwargs: []
    fetchers_stub.pexels_search_videos = lambda *args, **kwargs: []
    fetchers_stub.pixabay_search_videos = lambda *args, **kwargs: []
    fetchers_stub._best_vertical_video_file = lambda *args, **kwargs: None
    fetchers_stub._pixabay_best_video_url = lambda *args, **kwargs: None

    pipeline_module = ModuleType("src.pipeline")
    pipeline_module.fetchers = fetchers_stub

    src_module = sys.modules.setdefault("src", ModuleType("src"))
    src_module.pipeline = pipeline_module
    sys.modules["src.pipeline"] = pipeline_module
    sys.modules["src.pipeline.fetchers"] = fetchers_stub

if "moviepy.editor" not in sys.modules:
    moviepy_module = ModuleType("moviepy")
    editor_stub = ModuleType("moviepy.editor")

    class _ClipStub:  # pragma: no cover - simple placeholder
        def __init__(self, *args, **kwargs):
            pass

    editor_stub.VideoFileClip = _ClipStub
    editor_stub.TextClip = _ClipStub
    editor_stub.CompositeVideoClip = _ClipStub

    moviepy_module.editor = editor_stub
    sys.modules["moviepy"] = moviepy_module
    sys.modules["moviepy.editor"] = editor_stub

import video_processor


class MemoryLogger:
    def __init__(self):
        self.events = []

    def log(self, payload):
        self.events.append(payload)


class DummyLLM:
    def generate_hints_for_segment(self, *args, **kwargs):
        raise AssertionError("generate_hints_for_segment should not be called when briefs are present")


class DummyOrchestrator:
    def __init__(self, *_args, **_kwargs):
        self.fetch_calls = []

    def fetch_candidates(self, queries, *, duration_hint, filters):  # noqa: D401 - simple stub
        self.fetch_calls.append((list(queries), duration_hint, dict(filters or {})))
        return []


@pytest.mark.parametrize(
    "brief_terms",
    [
        ("dopamine reward", "brain scan lab"),
    ],
)
def test_segment_briefs_drive_queries(monkeypatch, brief_terms):
    memory_logger = MemoryLogger()
    decisions = []

    monkeypatch.setattr(video_processor, "FetcherOrchestrator", DummyOrchestrator)

    def fake_log(logger, **payload):
        payload_copy = dict(payload)
        decisions.append(payload_copy)

    monkeypatch.setattr(video_processor, "log_broll_decision", fake_log)

    processor = video_processor.VideoProcessor.__new__(video_processor.VideoProcessor)
    processor._pipeline_config = SimpleNamespace(
        fetcher=SimpleNamespace(),
        selection=SimpleNamespace(min_score=-1.0),
        timeboxing=SimpleNamespace(fetch_rank_ms=0, request_timeout_s=0),
    )
    processor._dyn_context = {
        "language": "en",
        "segment_briefs": [
            {
                "segment_index": 0,
                "keywords": [brief_terms[0]],
                "queries": [brief_terms[1]],
            }
        ],
    }
    processor._llm_service = DummyLLM()
    processor._core_last_run_used = False

    def fake_event_logger(self):
        return memory_logger

    processor._broll_event_logger = memory_logger
    processor._get_broll_event_logger = MethodType(fake_event_logger, processor)
    processor._derive_segment_keywords = MethodType(
        lambda self, _segment, _keywords: ["doctor", "stethoscope", "person discussing"],
        processor,
    )
    processor._rank_candidate = MethodType(lambda self, *_args, **_kwargs: 0.0, processor)

    segment = SimpleNamespace(start=0.0, end=1.0, text="Brain science dopamine focus")
    processor._insert_brolls_pipeline_core([segment], ["doctor"], subtitles=None, input_path=SimpleNamespace(name="clip.mp4"))

    logged_queries = [
        event
        for event in memory_logger.events
        if event.get("event") == "broll_segment_queries" and event.get("segment") == 0
    ]
    assert logged_queries, "expected queries event"
    queries_event = logged_queries[0]

    assert queries_event["source"] == "segment_brief"
    assert queries_event["queries"] == list(brief_terms)
    banned_tokens = {"person discussing", "doctor", "stethoscope"}
    assert not banned_tokens.intersection(queries_event["queries"])

    decision_events = [payload for payload in decisions if payload.get("segment_idx") == 0]
    assert decision_events, "expected per-segment decision"
    assert decision_events[0]["queries"] == list(brief_terms)
