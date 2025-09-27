import os
import sys
import importlib
import importlib.machinery
import types
from pathlib import Path

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

    events = getattr(processor._broll_event_logger, "entries", [])
    assert any(event.get("event") == "broll_env_ready" for event in events)

    meta_path = output_dir / "clips" / "sample" / "meta.txt"
    assert meta_path.exists()
    assert (output_dir / "final" ).exists()


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

    monkeypatch.setattr(module, "FetcherOrchestrator", StubOrchestrator)

    processor._derive_segment_keywords = types.MethodType(
        lambda _self, _segment, _keywords: ["sample"],
        processor,
    )
    processor._rank_candidate = types.MethodType(
        lambda _self, *_args, **_kwargs: 1.0,
        processor,
    )

    downloaded_asset = temp_dir / "core" / "asset.mp4"
    downloaded_asset.parent.mkdir(parents=True, exist_ok=True)
    downloaded_asset.write_bytes(b"data")

    captured_timeline = {}

    def fake_download(self, _candidate, directory, order):
        captured_timeline.setdefault("download_calls", []).append((directory, order))
        return downloaded_asset

    def fake_render(self, base_path, timeline):
        captured_timeline["timeline"] = timeline
        rendered_path = temp_dir / "core" / "rendered.mp4"
        rendered_path.parent.mkdir(parents=True, exist_ok=True)
        rendered_path.write_bytes(b"render")
        return rendered_path

    monkeypatch.setattr(module.VideoProcessor, "_download_core_candidate", fake_download, raising=False)
    monkeypatch.setattr(module.VideoProcessor, "_render_core_broll_timeline", fake_render, raising=False)

    input_clip = tmp_path / "input.mp4"
    input_clip.write_bytes(b"base")

    inserted_count, rendered_path = processor._insert_brolls_pipeline_core(
        [types.SimpleNamespace(start=0.0, end=2.0, text="hello world")],
        ["sample"],
        subtitles=[],
        input_path=input_clip,
    )

    assert inserted_count == 1
    assert rendered_path is not None
    assert rendered_path != input_clip
    assert captured_timeline.get("timeline")
    assert captured_timeline["timeline"][0]["path"] == downloaded_asset

    sys.modules.pop("video_processor", None)

def test_to_bool_accepts_common_values(video_processor_module):
    module = video_processor_module
    assert module._to_bool('1') is True
    assert module._to_bool('yes') is True
    assert module._to_bool('0') is False
    assert module._to_bool('false') is False
