import json
import os
import sys
import importlib
import importlib.machinery
import types
from pathlib import Path
from typing import Any, Dict

import pytest


@pytest.fixture
def video_processor_module(monkeypatch):
    os.environ["FAST_TESTS"] = "1"

    dummy_cv2 = types.ModuleType("cv2")
    dummy_cv2.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)
    dummy_whisper = types.ModuleType("whisper")
    setattr(dummy_whisper, "load_model", lambda *_: object())
    monkeypatch.setitem(sys.modules, "cv2", dummy_cv2)
    monkeypatch.setitem(sys.modules, "whisper", dummy_whisper)

    dummy_moviepy = types.ModuleType("moviepy.editor")
    dummy_moviepy.VideoFileClip = lambda *args, **kwargs: None
    dummy_moviepy.TextClip = lambda *args, **kwargs: None
    dummy_moviepy.CompositeVideoClip = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "moviepy", types.ModuleType("moviepy"))
    monkeypatch.setitem(sys.modules, "moviepy.editor", dummy_moviepy)

    dummy_fetchers = types.ModuleType("pipeline_core.fetchers")

    class DummyFetcherOrchestrator:
        def __init__(self, *args, **kwargs):
            pass

        def fetch_candidates(self, *args, **kwargs):
            return []

        def evaluate_candidate_filters(self, *_args, **_kwargs):
            return True, None

    setattr(dummy_fetchers, "FetcherOrchestrator", DummyFetcherOrchestrator)
    monkeypatch.setitem(sys.modules, "pipeline_core.fetchers", dummy_fetchers)

    dummy_config_module = types.ModuleType("pipeline_core.configuration")

    class DummyPipelineConfigBundle:
        def __init__(self):
            self.fetcher = DummyFetcherOrchestrator()
            self.selection = types.SimpleNamespace(min_score=0.0, prefer_landscape=False, min_duration_s=0.0)
            self.timeboxing = types.SimpleNamespace(fetch_rank_ms=0)

    setattr(dummy_config_module, "PipelineConfigBundle", DummyPipelineConfigBundle)
    monkeypatch.setitem(sys.modules, "pipeline_core.configuration", dummy_config_module)

    dummy_logging_module = types.ModuleType("pipeline_core.logging")

    class DummyLogger:
        def __init__(self, *args, **kwargs):
            self.entries = []

        def log(self, entry):
            self.entries.append(entry)

    setattr(dummy_logging_module, "JsonlLogger", DummyLogger)

    def dummy_log_decision(*args, **kwargs):
        pass

    setattr(dummy_logging_module, "log_broll_decision", dummy_log_decision)
    setattr(dummy_logging_module, "log_pipeline_summary", dummy_log_decision)
    setattr(dummy_logging_module, "log_pipeline_error", dummy_log_decision)
    setattr(dummy_logging_module, "log_stage_start", lambda *args, **kwargs: 0.0)
    setattr(dummy_logging_module, "log_stage_end", dummy_log_decision)
    monkeypatch.setitem(sys.modules, "pipeline_core.logging", dummy_logging_module)

    dummy_llm_module = types.ModuleType("pipeline_core.llm_service")

    class DummyLLMService:
        def generate_hints_for_segment(self, *args, **kwargs):
            return {"queries": ["stub"], "filters": {}}

    setattr(dummy_llm_module, "LLMMetadataGeneratorService", DummyLLMService)
    monkeypatch.setitem(sys.modules, "pipeline_core.llm_service", dummy_llm_module)

    dummy_dedupe_module = types.ModuleType("pipeline_core.dedupe")
    setattr(dummy_dedupe_module, "compute_phash", lambda *args, **kwargs: None)
    setattr(dummy_dedupe_module, "hamming_distance", lambda *args, **kwargs: 0)
    monkeypatch.setitem(sys.modules, "pipeline_core.dedupe", dummy_dedupe_module)

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    return importlib.import_module("video_processor")


def test_process_single_clip_smoke(tmp_path, monkeypatch, video_processor_module):
    video_processor = video_processor_module

    clips_dir = tmp_path / "clips"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    for folder in (clips_dir, output_dir, temp_dir):
        folder.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(video_processor.Config, "CLIPS_FOLDER", clips_dir)
    monkeypatch.setattr(video_processor.Config, "OUTPUT_FOLDER", output_dir)
    monkeypatch.setattr(video_processor.Config, "TEMP_FOLDER", temp_dir)

    monkeypatch.setattr(video_processor.whisper, "load_model", lambda *_: object())

    class DummyLLM:
        def generate_hints_for_segment(self, *args, **kwargs):
            return {"queries": ["sample"], "filters": {}}

    monkeypatch.setattr(video_processor, "LLMMetadataGeneratorService", lambda: DummyLLM())

    class DummySelection:
        min_score = 0.0
        prefer_landscape = False
        min_duration_s = 0.0

    class DummyTimeboxing:
        fetch_rank_ms = 0

    class DummyFetcher:
        def fetch_candidates(self, *args, **kwargs):
            return []

    class DummyConfig:
        def __init__(self):
            self.fetcher = DummyFetcher()
            self.selection = DummySelection()
            self.timeboxing = DummyTimeboxing()

    monkeypatch.setattr(video_processor, "PipelineConfigBundle", lambda: DummyConfig())

    def fake_reframe(self, clip_path):
        out = temp_dir / "reframed.mp4"
        out.write_bytes(b"data")
        return out

    monkeypatch.setattr(video_processor.VideoProcessor, "reframe_to_vertical", fake_reframe)

    def fake_transcribe(self, clip_path):
        return [
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": 1.0, "end": 2.0, "text": "world"},
        ]

    monkeypatch.setattr(video_processor.VideoProcessor, "transcribe_segments", fake_transcribe)

    def fake_generate(self, subtitles):
        return (
            "Test Title",
            "Test Description",
            ["#test"],
            ["keyword1", "keyword2"],
        )

    monkeypatch.setattr(video_processor.VideoProcessor, "generate_caption_and_hashtags", fake_generate)

    monkeypatch.setattr(
        video_processor.VideoProcessor,
        "insert_brolls_if_enabled",
        lambda self, path, subtitles, keywords: path,
    )

    def fake_copy(self, src: Path, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"data")

    monkeypatch.setattr(video_processor.VideoProcessor, "_safe_copy", fake_copy)
    monkeypatch.setattr(video_processor.VideoProcessor, "_hardlink_or_copy", fake_copy)

    def fake_unique_path(self, base_dir: Path, base_name: str, suffix: str) -> Path:
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir / f"{base_name}{suffix}"

    monkeypatch.setattr(video_processor.VideoProcessor, "_unique_path", fake_unique_path)

    def fake_add_subtitles(input_path, subtitles, output_path, **kwargs):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"data")

    monkeypatch.setattr(video_processor, "add_hormozi_subtitles", fake_add_subtitles)

    class DummyVideoClip:
        duration = 1.0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr(video_processor, "VideoFileClip", lambda *_: DummyVideoClip())

    src_clip = clips_dir / "sample.mp4"
    src_clip.write_bytes(b"data")

    processor = video_processor.VideoProcessor()
    processor.process_single_clip(src_clip)


def test_legacy_fallback_disabled_skips_src_pipeline(tmp_path, monkeypatch, video_processor_module):
    import builtins

    video_processor = video_processor_module
    monkeypatch.chdir(tmp_path)

    (Path("AI-B-roll") / "broll_library").mkdir(parents=True, exist_ok=True)
    temp_dir = Path("temp")
    output_dir = Path("output")
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(video_processor.Config, "ENABLE_BROLL", True)
    monkeypatch.setattr(video_processor.Config, "TEMP_FOLDER", temp_dir)
    monkeypatch.setattr(video_processor.Config, "OUTPUT_FOLDER", output_dir)
    monkeypatch.setattr(video_processor.Config, "ENABLE_LEGACY_PIPELINE_FALLBACK", False)
    monkeypatch.delenv("ENABLE_LEGACY_PIPELINE_FALLBACK", raising=False)
    assert video_processor._legacy_pipeline_fallback_enabled() is False

    class DummyLogger:
        def __init__(self):
            self.events = []

        def log(self, payload):
            self.events.append(dict(payload))

    event_logger = DummyLogger()

    monkeypatch.setattr(
        video_processor.VideoProcessor,
        "_maybe_use_pipeline_core",
        lambda self, segments, broll_keywords, *, subtitles, input_path: (0, None),
    )
    monkeypatch.setattr(video_processor.VideoProcessor, "_get_broll_event_logger", lambda self: event_logger)

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("src.pipeline"):
            raise AssertionError("legacy pipeline import attempted")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    processor = video_processor.VideoProcessor()
    clip_path = Path("clip.mp4")
    clip_path.write_bytes(b"data")

    result = processor.insert_brolls_if_enabled(
        clip_path,
        subtitles=[{"start": 0.0, "end": 1.0, "text": "hello"}],
        broll_keywords=["focus"],
    )

    assert result == clip_path
    assert any(evt.get("event") == "legacy_skipped" for evt in event_logger.events)
    for evt in event_logger.events:
        serialized = json.dumps(evt, ensure_ascii=False)
        assert "archive" not in serialized.lower()
        assert "giphy" not in serialized.lower()


def test_insert_brolls_initialises_library_and_calls_core(tmp_path, monkeypatch, video_processor_module):
    module = video_processor_module
    monkeypatch.chdir(tmp_path)

    (tmp_path / "AI-B-roll").mkdir(parents=True, exist_ok=True)

    clips_dir = tmp_path / "clips"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    for folder in (clips_dir, output_dir, temp_dir):
        folder.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(module.Config, "ENABLE_BROLL", True)
    monkeypatch.setattr(module.Config, "CLIPS_FOLDER", clips_dir)
    monkeypatch.setattr(module.Config, "OUTPUT_FOLDER", output_dir)
    monkeypatch.setattr(module.Config, "TEMP_FOLDER", temp_dir)
    monkeypatch.setattr(module.Config, "ENABLE_LEGACY_PIPELINE_FALLBACK", False)
    monkeypatch.delenv("ENABLE_LEGACY_PIPELINE_FALLBACK", raising=False)
    monkeypatch.setenv("ENABLE_PIPELINE_CORE_FETCHER", "1")

    calls: Dict[str, Any] = {}

    def fake_core(self, segments, broll_keywords, *, subtitles, input_path):
        calls["segments"] = list(segments)
        calls["keywords"] = list(broll_keywords)
        calls["subtitles"] = list(subtitles)
        calls["input_path"] = Path(input_path)
        return (0, None)

    monkeypatch.setattr(module.VideoProcessor, "_maybe_use_pipeline_core", fake_core, raising=False)

    class DummyLogger:
        def __init__(self):
            self.entries = []

        def log(self, payload):
            self.entries.append(dict(payload))

    dummy_logger = DummyLogger()
    monkeypatch.setattr(module.VideoProcessor, "_get_broll_event_logger", lambda self: dummy_logger, raising=False)

    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"data")

    processor = module.VideoProcessor()

    result = processor.insert_brolls_if_enabled(
        clip_path,
        subtitles=[{"start": 0.0, "end": 1.0, "text": "hello world"}],
        broll_keywords=["focus"],
    )

    assert result == clip_path
    assert "segments" in calls and calls["segments"]
    assert calls["input_path"] == clip_path
    assert (tmp_path / "AI-B-roll" / "broll_library").exists()


def test_pipeline_core_single_provider_warns(monkeypatch, caplog, tmp_path, video_processor_module):
    video_processor = video_processor_module

    processor = video_processor.VideoProcessor.__new__(video_processor.VideoProcessor)
    fetcher_cfg = types.SimpleNamespace(providers=types.SimpleNamespace(name="solo", enabled=True))
    processor._pipeline_config = types.SimpleNamespace(fetcher=fetcher_cfg)
    processor._llm_service = None

    monkeypatch.setattr(video_processor, "_pipeline_core_fetcher_enabled", lambda: True)

    called = False

    def fake_insert(*args, **kwargs):
        nonlocal called
        called = True

    processor._insert_brolls_pipeline_core = fake_insert  # type: ignore[attr-defined]

    caplog.set_level("WARNING")
    used = processor._maybe_use_pipeline_core(
        segments=[types.SimpleNamespace(start=0.0, end=1.0, text="hello")],
        broll_keywords=["kw"],
        subtitles=[{"start": 0.0, "end": 1.0, "text": "hello"}],
        input_path=tmp_path / "video.mp4",
    )

    assert used is None
    assert called is False
    assert any("pipeline_core fetcher misconfigured" in record.message for record in caplog.records)


def test_core_pipeline_materializes_and_renders(monkeypatch, tmp_path, video_processor_module):
    module = video_processor_module

    temp_dir = tmp_path / "temp"
    output_dir = tmp_path / "out"
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(module.Config, "TEMP_FOLDER", temp_dir)
    monkeypatch.setattr(module.Config, "OUTPUT_FOLDER", output_dir)

    processor = module.VideoProcessor.__new__(module.VideoProcessor)
    processor._pipeline_config = types.SimpleNamespace(
        fetcher=types.SimpleNamespace(providers=[types.SimpleNamespace(name="stub", enabled=True)]),
        selection=types.SimpleNamespace(min_score=0.0, prefer_landscape=False, min_duration_s=0.0),
        timeboxing=types.SimpleNamespace(fetch_rank_ms=0),
    )
    processor._llm_service = None

    class DummyLogger:
        def __init__(self):
            self.entries = []

        def log(self, payload):
            self.entries.append(dict(payload))

    event_logger = DummyLogger()
    processor._broll_event_logger = event_logger
    processor._get_broll_event_logger = lambda: event_logger

    candidate = types.SimpleNamespace(
        url="http://example.com/video.mp4",
        provider="stub",
        duration=1.5,
        width=720,
        height=1280,
        title="Sample",
        tags=["sample"],
        thumb_url=None,
        identifier="stub-1",
    )

    class StubOrchestrator:
        def __init__(self, *_args, **_kwargs):
            pass

        def fetch_candidates(self, *_args, **_kwargs):
            return [candidate]

        def evaluate_candidate_filters(self, *_args, **_kwargs):
            return True, None

    monkeypatch.setattr(module, "FetcherOrchestrator", StubOrchestrator)

    processor._derive_segment_keywords = types.MethodType(
        lambda _self, _segment, _keywords: ["sample"],
        processor,
    )
    processor._rank_candidate = types.MethodType(
        lambda _self, *_args, **_kwargs: 1.0,
        processor,
    )

    class DummyResponse:
        def __init__(self, payload: bytes):
            self._payload = payload

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield self._payload

    def fake_get(_url, stream=True, timeout=15):
        assert stream is True
        assert timeout == 15
        return DummyResponse(b"downloaded")

    monkeypatch.setattr(module.requests, "get", fake_get)

    class DummyClip:
        def __init__(self, path):
            self.path = Path(path)
            self.h = 720
            self.duration = 1.5

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def resize(self, *args, **kwargs):
            return self

        def without_audio(self):
            return self

        def set_audio(self, *_args, **_kwargs):
            return self

        def subclip(self, *_args, **_kwargs):
            return self

        def set_start(self, *_args, **_kwargs):
            return self

        def set_duration(self, *_args, **_kwargs):
            return self

        def set_position(self, *_args, **_kwargs):
            return self

    class DummyComposite:
        def __init__(self, layers):
            self.layers = layers

        def write_videofile(self, path, **_kwargs):
            Path(path).write_bytes(b"render")

        def close(self):
            return None

    monkeypatch.setattr(module, "VideoFileClip", DummyClip)
    monkeypatch.setattr(module, "CompositeVideoClip", DummyComposite)

    input_clip = tmp_path / "input.mp4"
    input_clip.write_bytes(b"base")

    inserted_count, rendered_path, meta = processor._insert_brolls_pipeline_core(
        [types.SimpleNamespace(start=0.0, end=2.0, text="hello world")],
        ["sample"],
        subtitles=[],
        input_path=input_clip,
    )

    assert inserted_count == 1
    assert rendered_path is not None
    assert rendered_path != input_clip
    assert meta.get("render_ok") is True
    assert rendered_path.exists()

    timeline_entries = [
        entry
        for entry in processor._core_last_timeline
    ]
    assert timeline_entries
    first_entry = timeline_entries[0]
    assert isinstance(first_entry, module.CoreTimelineEntry)
    assert first_entry.path.exists()
    assert pytest.approx(first_entry.start, rel=1e-6) == 0.0
    assert pytest.approx(first_entry.end, rel=1e-6) == 1.5

    events = event_logger.entries
    download_events = [entry for entry in events if entry.get("event") == "broll_asset_downloaded"]
    assert download_events
    download_event = download_events[-1]
    assert download_event["url"] == candidate.url
    assert download_event["provider"] == candidate.provider
    assert Path(download_event["path"]).exists()

    timeline_events = [entry for entry in events if entry.get("event") == "broll_timeline_rendered"]
    assert timeline_events
    timeline_event = timeline_events[-1]
    assert timeline_event["clips"] == 1
    assert timeline_event["output"] == str(rendered_path)

    summary_events = [entry for entry in events if entry.get("event") == "broll_summary"]
    assert summary_events
    summary_event = summary_events[-1]
    assert summary_event["render_ok"] is True
    assert summary_event["inserted"] == inserted_count
    assert events.index(timeline_event) < events.index(summary_event)

    sys.modules.pop("video_processor", None)

def test_to_bool_accepts_common_values(video_processor_module):
    module = video_processor_module
    assert module._to_bool('1') is True
    assert module._to_bool('yes') is True
    assert module._to_bool('0') is False
    assert module._to_bool('false') is False
