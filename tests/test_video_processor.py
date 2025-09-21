import os
import sys
import importlib
import types
from pathlib import Path

import pytest


@pytest.fixture
def video_processor_module(monkeypatch):
    os.environ["FAST_TESTS"] = "1"

    dummy_cv2 = types.ModuleType("cv2")
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
