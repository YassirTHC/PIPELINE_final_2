from types import SimpleNamespace, MethodType, ModuleType

import importlib

import json
import os
import re
import logging

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

    def fetch_candidates(self, queries, *, segment_index=None, duration_hint, filters):  # noqa: D401 - simple stub
        self.fetch_calls.append((list(queries), segment_index, duration_hint, dict(filters or {})))
        return []

    def evaluate_candidate_filters(self, *_args, **_kwargs):
        return True, None


@pytest.fixture
def core_event_log(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    monkeypatch.setattr(video_processor.Config, "OUTPUT_FOLDER", output_dir, raising=False)
    monkeypatch.setattr(video_processor.Config, "TEMP_FOLDER", tmp_path / "temp", raising=False)
    monkeypatch.setattr(video_processor, "_GLOBAL_BROLL_EVENTS_LOGGER", None, raising=False)

    events_path = output_dir / "meta" / "broll_pipeline_events.jsonl"
    if events_path.exists():
        events_path.unlink()
    return events_path


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


def test_metadata_queries_leave_room_for_contextual_terms():
    llm_terms = [
        "self reward process",
        "intrinsic motivation focus",
        "positive outcome path",
        "personal achievement drive",
    ]
    segment_keywords = [
        "runner tying shoes",
        "starting blocks preparation",
    ]
    selector_keywords = [
        "athlete preparing",
    ]

    merged, source = video_processor._merge_segment_query_sources(
        segment_text="Runner ties shoes before sprint",
        llm_queries=llm_terms,
        brief_queries=[],
        brief_keywords=[],
        segment_keywords=segment_keywords,
        selector_keywords=selector_keywords,
        cap=4,
    )

    assert len(merged) == 4
    assert source in {"llm_hint", "segment_brief", "segment_keywords", "selector_keywords"}

    llm_term_set = set(llm_terms)
    contextual_terms = [term for term in merged if term not in llm_term_set]
    assert contextual_terms, "expected contextual keywords to fill reserved slots"
    assert any("runner" in term or "athlete" in term for term in contextual_terms)


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
    queries_used, seg_idx, *_ = fetch_calls[0]
    assert seg_idx == 0
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
    queries_used, seg_idx, *_ = fetch_calls[0]
    assert seg_idx == 0
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


def test_metadata_fallback_logging_and_events(monkeypatch, caplog, core_event_log, tmp_path):
    class FallbackLLM:
        def __init__(self):
            self.calls = 0

        def generate_hints_for_segment(self, *_args, **_kwargs):
            self.calls += 1
            return {
                "queries": ["sunset beach", "calm ocean"],
                "source": "metadata_keywords_fallback",
            }

    class FetcherStub:
        def __init__(self, config):
            self.config = config

        def fetch_candidates(self, queries, *, segment_index=None, duration_hint=None, filters=None):
            self.last_queries = list(queries)
            candidates = []
            for idx in range(3):
                candidates.append(
                    SimpleNamespace(
                        url=f"http://example.com/{idx}.mp4",
                        provider="pexels",
                        width=1080,
                        height=1920,
                        duration=3.5 + idx,
                        title=f"Scene {idx}",
                        tags=("beach",),
                        order=idx,
                    )
                )
            return candidates

        def evaluate_candidate_filters(self, *_args, **_kwargs):
            return True, None

    def fake_download(self, _candidate, download_dir, order, _segment=None):
        if download_dir is None:
            return None
        download_dir.mkdir(parents=True, exist_ok=True)
        path = download_dir / f"candidate-{order}.mp4"
        path.write_text("data", encoding="utf-8")
        return path

    llm = FallbackLLM()
    monkeypatch.setattr(video_processor, "FetcherOrchestrator", FetcherStub)
    monkeypatch.setattr(video_processor.VideoProcessor, "_download_core_candidate", fake_download)
    monkeypatch.setattr(video_processor.Config, "ENABLE_PIPELINE_CORE_FETCHER", True, raising=False)

    processor = video_processor.VideoProcessor.__new__(video_processor.VideoProcessor)
    processor._pipeline_config = SimpleNamespace(
        fetcher=SimpleNamespace(
            providers=[SimpleNamespace(name="pexels", enabled=True, max_results=3)],
            per_segment_limit=3,
            allow_images=True,
            allow_videos=True,
            request_timeout_s=0,
        ),
        selection=SimpleNamespace(
            min_score=0.0,
            prefer_landscape=False,
            min_duration_s=0.0,
            allow_forced_keep=False,
            forced_keep_budget=0,
        ),
        timeboxing=SimpleNamespace(fetch_rank_ms=0, request_timeout_s=0),
    )
    processor._dyn_context = {"language": "en", "segment_briefs": []}
    processor._selector_keywords = []
    processor._fetch_keywords = []
    processor._latest_metadata = {}
    processor._llm_service = llm
    processor._broll_event_logger = None
    processor._broll_env_logged = False
    processor._core_last_run_used = False
    processor._derive_segment_keywords = MethodType(lambda self, *_args, **_kwargs: [], processor)
    processor._rank_candidate = MethodType(
        lambda self, _text, candidate, _cfg, _duration: 0.9 - 0.1 * getattr(candidate, "order", 0),
        processor,
    )

    input_path = tmp_path / "clip.mp4"
    input_path.write_text("source", encoding="utf-8")
    segment = SimpleNamespace(start=0.0, end=5.0, text="Relaxing beach talk about sunsets")

    caplog.set_level(logging.INFO)
    caplog.clear()
    processor._insert_brolls_pipeline_core([segment], ["beach"], subtitles=None, input_path=input_path)

    assert llm.calls == 1
    assert "[BROLL][LLM] segment=0.00-5.00 queries=['sunset beach', 'calm ocean'] (source=metadata_keywords_fallback)" in caplog.text

    assert core_event_log.exists()
    events = [json.loads(line) for line in core_event_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    candidate_events = [event for event in events if event.get("event") == "broll_candidate_evaluated"]
    assert candidate_events, "expected at least one candidate evaluation event in JSONL log"
