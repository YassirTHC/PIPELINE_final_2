import json
import importlib.machinery
import sys
import types
from pathlib import Path


def _ensure_stub(module_name: str, stub: types.ModuleType) -> None:
    if module_name not in sys.modules:
        sys.modules[module_name] = stub


_ensure_stub("cv2", types.ModuleType("cv2"))
sys.modules["cv2"].__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)

dummy_whisper = types.ModuleType("whisper")
dummy_whisper.__spec__ = importlib.machinery.ModuleSpec("whisper", loader=None)
setattr(dummy_whisper, "load_model", lambda *_: object())
_ensure_stub("whisper", dummy_whisper)

dummy_moviepy = types.ModuleType("moviepy")
dummy_moviepy_editor = types.ModuleType("moviepy.editor")
dummy_moviepy_editor.VideoFileClip = lambda *args, **kwargs: None
dummy_moviepy_editor.TextClip = lambda *args, **kwargs: None
dummy_moviepy_editor.CompositeVideoClip = lambda *args, **kwargs: None
_ensure_stub("moviepy", dummy_moviepy)
_ensure_stub("moviepy.editor", dummy_moviepy_editor)

dummy_fetchers = types.ModuleType("pipeline_core.fetchers")


class _StubFetcherOrchestrator:
    def __init__(self, *args, **kwargs):
        pass


setattr(dummy_fetchers, "FetcherOrchestrator", _StubFetcherOrchestrator)
_ensure_stub("pipeline_core.fetchers", dummy_fetchers)

dummy_config = types.ModuleType("pipeline_core.configuration")


class _StubPipelineConfigBundle:
    def __init__(self):
        self.fetcher = _StubFetcherOrchestrator()
        self.selection = types.SimpleNamespace(min_score=0.0, prefer_landscape=False, min_duration_s=0.0)
        self.timeboxing = types.SimpleNamespace(fetch_rank_ms=0)


setattr(dummy_config, "PipelineConfigBundle", _StubPipelineConfigBundle)
_ensure_stub("pipeline_core.configuration", dummy_config)

dummy_logging = types.ModuleType("pipeline_core.logging")


class _StubJsonlLogger:
    def __init__(self, destination):
        self._path = Path(destination)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload):
        event = dict(payload)
        event.setdefault("ts", 0.0)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def write_jsonl(self, payload):
        self.log(payload)

    @property
    def path(self):
        return self._path


def _stub_log_pipeline_summary(logger, result, extra=None):
    summary = result.to_dict()
    duration = summary.pop("duration_s", None)
    payload = {
        "event": "pipeline_summary",
        "stage": "pipeline",
        "ok": bool(summary.get("final_export_ok")),
    }
    if duration is not None:
        payload["duration_ms"] = int(max(0.0, duration) * 1000)
    payload.update(summary)
    if extra:
        payload.update(extra)
    logger.log(payload)


setattr(dummy_logging, "JsonlLogger", _StubJsonlLogger)
setattr(dummy_logging, "log_broll_decision", lambda *args, **kwargs: None)
setattr(dummy_logging, "log_pipeline_summary", _stub_log_pipeline_summary)
_ensure_stub("pipeline_core.logging", dummy_logging)

dummy_llm_service = types.ModuleType("pipeline_core.llm_service")
setattr(dummy_llm_service, "LLMMetadataGeneratorService", lambda *args, **kwargs: None)
setattr(dummy_llm_service, "enforce_fetch_language", lambda queries, language=None: queries)
_ensure_stub("pipeline_core.llm_service", dummy_llm_service)

dummy_dedupe = types.ModuleType("pipeline_core.dedupe")
setattr(dummy_dedupe, "compute_phash", lambda *args, **kwargs: None)
setattr(dummy_dedupe, "hamming_distance", lambda *args, **kwargs: 0)
_ensure_stub("pipeline_core.dedupe", dummy_dedupe)

from pipeline_core.logging import JsonlLogger, log_pipeline_summary
from pipeline_core.runtime import PipelineResult
from video_processor import format_broll_completion_banner


def test_pipeline_summary_event_contains_flags(tmp_path):
    log_path = tmp_path / "events.jsonl"
    logger = JsonlLogger(log_path)

    result = PipelineResult()
    result.final_export_ok = True
    result.broll_inserted_count = 3
    result.finish()

    log_pipeline_summary(logger, result, extra={"effective_domain": "generic", "queries_count": 5})

    content = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert content, "expected summary event to be written"
    payload = json.loads(content[-1])

    assert payload["event"] == "pipeline_summary"
    assert payload["stage"] == "pipeline"
    assert payload["final_export_ok"] is True
    assert payload["effective_domain"] == "generic"
    assert payload["broll_inserted_count"] == 3
    assert "duration_ms" in payload


def test_broll_summary_matches_console(tmp_path):
    log_path = tmp_path / "events.jsonl"
    logger = JsonlLogger(log_path)
    logger.log({
        "event": "broll_summary",
        "segments": 3,
        "inserted": 2,
        "providers_used": ["pexels"],
    })

    content = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert content, "expected broll summary event to be written"
    payload = json.loads(content[-1])
    assert payload["event"] == "broll_summary"
    assert payload["inserted"] == 2
    assert payload["segments"] == 3
    assert payload["providers_used"] == ["pexels"]

    fake_console_line = "    📊 B-roll sélectionnés: 2/3"
    import re

    match = re.search(r"B-roll sélectionnés:\s*(\d+)\s*/\s*(\d+)", fake_console_line)
    assert match, "expected to parse console summary"
    assert int(match.group(1)) == payload["inserted"]
    assert int(match.group(2)) == payload["segments"]


def test_format_broll_banner_warns_on_zero_insertions():
    success, banner = format_broll_completion_banner(0, origin="legacy")
    assert success is False
    assert "⚠️" in banner
    assert "Aucun B-roll" in banner


def test_format_broll_banner_keeps_success_icon():
    success, banner = format_broll_completion_banner(3, origin="pipeline_core")
    assert success is True
    assert "✅" in banner
    assert "3" in banner
    assert "B-roll" in banner
