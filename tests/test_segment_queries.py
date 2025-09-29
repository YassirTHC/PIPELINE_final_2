from types import SimpleNamespace, MethodType, ModuleType

import importlib

import json
import os
import re

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
    sys.modules["cv2"].__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)

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

os.environ.setdefault("PIPELINE_FAST_TESTS", "1")

import video_processor


class MemoryLogger:
    def __init__(self):
        self.events = []

    def log(self, payload):
        self.events.append(payload)


class DummyLLM:
    def __init__(self):
        self.calls = 0

    def generate_hints_for_segment(self, *args, **kwargs):
        self.calls += 1
        return {"queries": []}


class DummyOrchestrator:
    def __init__(self, *_args, **_kwargs):
        self.fetch_calls = []
        instances = getattr(DummyOrchestrator, "instances", None)
        if instances is None:
            DummyOrchestrator.instances = []
            instances = DummyOrchestrator.instances
        instances.append(self)

    def fetch_candidates(self, queries, *, duration_hint, filters):  # noqa: D401 - simple stub
        self.fetch_calls.append((list(queries), duration_hint, dict(filters or {})))
        return []

    def evaluate_candidate_filters(self, *_args, **_kwargs):
        return True, None


@pytest.mark.parametrize(
    "brief_terms,expected_phrases",
    [
        (("dopamine reward", "brain scan lab"), ["dopamine showing reward", "brain scan lab"]),
    ],
)
def test_segment_briefs_drive_queries(monkeypatch, brief_terms, expected_phrases):
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
    llm = DummyLLM()
    processor._llm_service = llm
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

    assert llm.calls == 1

    logged_queries = [
        event
        for event in memory_logger.events
        if event.get("event") == "broll_segment_queries" and event.get("segment") == 0
    ]
    assert logged_queries, "expected queries event"
    queries_event = logged_queries[0]

    assert queries_event["source"] == "segment_brief"
    assert queries_event["queries"] == expected_phrases
    banned_tokens = {"person discussing", "doctor", "stethoscope", "professional", "buffer", "signal"}
    for phrase in queries_event["queries"]:
        assert not any(banned in phrase for banned in banned_tokens)
        assert re.match(r"^[a-z]+(?: [a-z]+){1,4}$", phrase)

    decision_events = [payload for payload in decisions if payload.get("segment_idx") == 0]
    assert decision_events, "expected per-segment decision"
    assert decision_events[0]["queries"] == expected_phrases


def test_dedupe_queries_drop_abstract_tokens():
    raw_terms = [
        "professional buffer signal",
        "doctor analyzing data center",
        "scientist planning experiment",
        "doctor analyzing data center",  # duplicate to test dedupe
    ]
    cleaned = video_processor._dedupe_queries(raw_terms, cap=5)
    assert cleaned == [
        "doctor analyzing data center",
        "scientist planning experiment",
    ]
    for phrase in cleaned:
        assert re.match(r"^[a-z]+(?: [a-z]+){1,4}$", phrase)
        assert "professional" not in phrase
        assert "buffer" not in phrase
        assert "signal" not in phrase


def test_selector_and_seed_queries_used_when_llm_empty(monkeypatch, tmp_path):
    memory_logger = MemoryLogger()

    monkeypatch.setattr(video_processor, "FetcherOrchestrator", DummyOrchestrator)
    DummyOrchestrator.instances = []

    seed_path = tmp_path / "seed_queries.json"
    seed_path.write_text(json.dumps(["team collaborating project"]), encoding="utf-8")
    monkeypatch.setenv("BROLL_SEED_QUERIES", str(seed_path))
    monkeypatch.setattr(video_processor, "_SEED_QUERY_CACHE", None, raising=False)

    processor = video_processor.VideoProcessor.__new__(video_processor.VideoProcessor)
    processor._pipeline_config = SimpleNamespace(
        fetcher=SimpleNamespace(),
        selection=SimpleNamespace(min_score=-1.0),
        timeboxing=SimpleNamespace(fetch_rank_ms=0, request_timeout_s=0),
    )
    processor._dyn_context = {"language": "en", "segment_briefs": []}

    class QuietLLM:
        def generate_hints_for_segment(self, *_args, **_kwargs):
            return {"queries": []}

    processor._llm_service = QuietLLM()
    processor._selector_keywords = ["motivation"]
    processor._core_last_run_used = False

    def fake_event_logger(self):
        return memory_logger

    processor._broll_event_logger = memory_logger
    processor._get_broll_event_logger = MethodType(fake_event_logger, processor)
    processor._derive_segment_keywords = MethodType(lambda self, *_: [], processor)
    processor._rank_candidate = MethodType(lambda self, *_args, **_kwargs: 0.0, processor)

    segment = SimpleNamespace(start=0.0, end=1.0, text="Inspiring motivation talk for teams")
    processor._insert_brolls_pipeline_core([segment], [], subtitles=None, input_path=SimpleNamespace(name="clip.mp4"))

    assert DummyOrchestrator.instances, "expected orchestrator to be constructed"
    fetch_calls = DummyOrchestrator.instances[0].fetch_calls
    assert fetch_calls, "expected orchestrator fetch to be invoked"
    queries_used, _, _ = fetch_calls[0]
    assert queries_used, "expected non-empty queries"

    logged_queries = [
        event
        for event in memory_logger.events
        if event.get("event") == "broll_segment_queries" and event.get("segment") == 0
    ]
    assert logged_queries, "expected queries event"
    queries_event = logged_queries[0]
    assert queries_event["source"] == "seed_queries"
    assert queries_event["queries"], "expected resolved queries"


def test_llm_hint_queries_bypass_merge(monkeypatch):
    memory_logger = MemoryLogger()

    monkeypatch.setattr(video_processor, "FetcherOrchestrator", DummyOrchestrator)
    DummyOrchestrator.instances = []

    processor = video_processor.VideoProcessor.__new__(video_processor.VideoProcessor)
    processor._pipeline_config = SimpleNamespace(
        fetcher=SimpleNamespace(),
        selection=SimpleNamespace(min_score=-1.0),
        timeboxing=SimpleNamespace(fetch_rank_ms=0, request_timeout_s=0),
    )
    processor._dyn_context = {"language": "en", "segment_briefs": []}
    processor._selector_keywords = []
    processor._core_last_run_used = False

    class HintLLM:
        def __init__(self):
            self.calls = 0

        def generate_hints_for_segment(self, *_args, **_kwargs):
            self.calls += 1
            return {"queries": ["Cinematic Skyline", "Happy Team Meeting"], "source": "llm_direct"}

    llm = HintLLM()
    processor._llm_service = llm

    def fake_event_logger(self):
        return memory_logger

    processor._broll_event_logger = memory_logger
    processor._get_broll_event_logger = MethodType(fake_event_logger, processor)
    processor._derive_segment_keywords = MethodType(lambda self, *_: [], processor)
    processor._rank_candidate = MethodType(lambda self, *_args, **_kwargs: 0.0, processor)

    segment = SimpleNamespace(start=0.0, end=1.0, text="Inspiring marketing speech")
    processor._insert_brolls_pipeline_core([segment], [], subtitles=None, input_path=SimpleNamespace(name="clip.mp4"))

    assert llm.calls == 1
    assert DummyOrchestrator.instances, "expected orchestrator to be constructed"
    fetch_calls = DummyOrchestrator.instances[0].fetch_calls
    assert fetch_calls, "expected orchestrator fetch to be invoked"
    queries_used, _, _ = fetch_calls[0]
    expected_queries = video_processor._dedupe_queries(
        ["Cinematic Skyline", "Happy Team Meeting"],
        cap=video_processor.SEGMENT_REFINEMENT_MAX_TERMS,
    )
    assert queries_used == expected_queries

    logged_queries = [
        event
        for event in memory_logger.events
        if event.get("event") == "broll_segment_queries" and event.get("segment") == 0
    ]
    assert logged_queries, "expected queries event"
    queries_event = logged_queries[0]
    assert queries_event["source"] == "llm_direct"
    assert queries_event["queries"] == expected_queries
