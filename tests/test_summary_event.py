import json
import importlib.machinery
import os
import sys
import types
from collections import Counter
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


class _StubSelectionConfig:
    def __init__(self):
        self.min_score = 0.6
        self.prefer_landscape = False
        self.min_duration_s = 0.0
        self.forced_keep_budget = 0
        self.allow_forced_keep = False

    @classmethod
    def from_environment(cls):
        instance = cls()
        raw_min_score = os.getenv("BROLL_MIN_SCORE")
        if raw_min_score is not None:
            try:
                instance.min_score = float(raw_min_score)
            except (TypeError, ValueError):
                pass
        raw_forced_keep = os.getenv("BROLL_FORCED_KEEP")
        if raw_forced_keep is not None:
            try:
                instance.forced_keep_budget = max(0, int(float(raw_forced_keep)))
            except (TypeError, ValueError):
                instance.forced_keep_budget = 0
        if instance.forced_keep_budget > 0:
            instance.allow_forced_keep = True
        raw_allow = os.getenv("BROLL_ALLOW_FORCED_KEEP")
        if raw_allow is not None:
            instance.allow_forced_keep = raw_allow.strip().lower() not in {"0", "false", "no", "off"}
        return instance


setattr(dummy_config, "SelectionConfig", _StubSelectionConfig)
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


def _stub_log_broll_decision(logger, *, segment_idx, start, end, query_count, candidate_count,
                             unique_candidates, url_dedup_hits, phash_dedup_hits, selected_url,
                             selected_score, provider, latency_ms, llm_healthy, reject_reasons,
                             queries=None, provider_status=None, best_score=None, reject_summary=None):
    event_name = "broll_segment_decision" if segment_idx >= 0 else "broll_session_summary"
    counts = Counter(reject_reasons or [])
    payload = {
        "event": event_name,
        "segment": segment_idx,
        "t0": start,
        "t1": end,
        "q_count": query_count,
        "candidates": candidate_count,
        "unique_candidates": unique_candidates,
        "dedup_url_hits": url_dedup_hits,
        "dedup_phash_hits": phash_dedup_hits,
        "selected_url": selected_url,
        "selected_score": selected_score,
        "provider": provider,
        "latency_ms": latency_ms,
        "llm_healthy": llm_healthy,
        "reject_reasons": dict(counts),
    }
    if queries is not None:
        payload["queries"] = list(queries)
    if provider_status is not None:
        payload["providers"] = provider_status
    if best_score is not None:
        payload["best_score"] = best_score
    if reject_summary is not None:
        payload["reject_summary"] = reject_summary
    elif counts:
        payload["reject_summary"] = {"counts": dict(counts)}
    logger.log(payload)


setattr(dummy_logging, "log_broll_decision", _stub_log_broll_decision)
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


def teardown_module(module):
    for name in (
        "pipeline_core.fetchers",
        "pipeline_core.configuration",
        "pipeline_core.logging",
        "pipeline_core.llm_service",
        "pipeline_core.dedupe",
    ):
        sys.modules.pop(name, None)

from pipeline_core.logging import JsonlLogger, log_pipeline_summary
from pipeline_core.runtime import PipelineResult
import video_processor
from video_processor import VideoProcessor, format_broll_completion_banner


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
        "selection_rate": round(2 / 3, 4),
        "selected_segments": [0, 2],
        "avg_broll_duration": 2.5,
        "broll_per_min": 1.2,
        "avg_latency_ms": 420.0,
        "refined_ratio": round(1 / 3, 4),
        "provider_mix": {"pexels": 2},
        "providers_used": ["pexels"],
        "query_source_counts": {"segment_brief": 2, "fallback_keywords": 1},
        "total_url_dedup_hits": 1,
        "total_phash_dedup_hits": 0,
        "dedupe_counts": {"url": 1, "phash": 0},
        "forced_keep_segments": 1,
        "forced_keep_count": 1,
        "total_candidates": 9,
        "total_unique_candidates": 4,
        "video_duration_s": 75.0,
        "render_ok": True,
    })

    content = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert content, "expected broll summary event to be written"
    payload = json.loads(content[-1])
    assert payload["event"] == "broll_summary"
    assert payload["inserted"] == 2
    assert payload["segments"] == 3
    assert payload["provider_mix"] == {"pexels": 2}
    assert payload["providers_used"] == ["pexels"]
    assert payload["selection_rate"] == round(2 / 3, 4)
    assert payload["query_source_counts"] == {"segment_brief": 2, "fallback_keywords": 1}
    assert payload["forced_keep_segments"] == 1
    assert payload["forced_keep_count"] == 1
    assert payload["dedupe_counts"] == {"url": 1, "phash": 0}
    assert payload["render_ok"] is True

    fake_console_line = "    📊 B-roll sélectionnés: 2/3 (66.7%); providers=pexels:2"
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


def test_format_broll_banner_warns_on_failed_render():
    success, banner = format_broll_completion_banner(2, origin="pipeline_core", render_ok=False)
    assert success is False
    assert "⚠️" in banner
    assert "rendu" in banner


def test_pipeline_core_download_failure_zero_count(monkeypatch, tmp_path):
    monkeypatch.setenv("FAST_TESTS", "1")

    monkeypatch.setattr(video_processor.Config, "CLIPS_FOLDER", tmp_path / "clips", raising=False)
    monkeypatch.setattr(video_processor.Config, "OUTPUT_FOLDER", tmp_path / "output", raising=False)
    monkeypatch.setattr(video_processor.Config, "TEMP_FOLDER", tmp_path / "temp", raising=False)

    processor = VideoProcessor()

    events = []

    class _DummyLogger:
        def log(self, payload):
            events.append(dict(payload))

    dummy_logger = _DummyLogger()
    monkeypatch.setattr(processor, "_get_broll_event_logger", lambda: dummy_logger)

    class _DummyOrchestrator:
        def __init__(self, *args, **kwargs):
            pass

        def fetch_candidates(self, *args, **kwargs):
            return [
                types.SimpleNamespace(
                    url="http://example.com/a.mp4",
                    provider="pexels",
                    duration=2.5,
                    title="hello world",
                )
            ]

        def evaluate_candidate_filters(self, *_args, **_kwargs):
            return True, None

    monkeypatch.setattr(video_processor, "FetcherOrchestrator", _DummyOrchestrator)
    monkeypatch.setattr(processor, "_download_core_candidate", lambda *args, **kwargs: None)
    monkeypatch.setattr(processor, "_rank_candidate", lambda *args, **kwargs: 1.0)

    segment = types.SimpleNamespace(start=0.0, end=5.0, text="hello world")
    input_path = tmp_path / "input.mp4"
    input_path.write_text("dummy", encoding="utf-8")

    count, render_path, meta = processor._insert_brolls_pipeline_core(
        [segment],
        ["hello"],
        subtitles=None,
        input_path=input_path,
    )

    assert count == 0
    assert render_path is None
    assert meta.get("render_ok") is False

    candidate_events = [event for event in events if event.get("event") == "broll_candidate_evaluated"]
    assert candidate_events, "expected per-candidate telemetry"
    candidate_payload = candidate_events[-1]
    assert candidate_payload["provider"] == "pexels"
    assert candidate_payload["selected"] is True
    assert candidate_payload["score"] == 1.0
    assert candidate_payload["reject_reason"] is None

    decision_events = [event for event in events if event.get("event") == "broll_segment_decision"]
    assert decision_events, "expected a broll decision event"
    decision_payload = decision_events[-1]
    assert decision_payload.get("reject_reasons") == {}
    summary = decision_payload.get("reject_summary", {})
    assert summary.get("counts") == {}
    assert summary.get("candidates")
    assert summary["candidates"][0]["selected"] is True
    assert summary["candidates"][0]["reject_reason"] is None

    summary_events = [event for event in events if event.get("event") == "broll_summary"]
    assert summary_events, "expected a summary event"
    assert summary_events[-1]["inserted"] == 0
    assert summary_events[-1]["render_ok"] is False

    banner_success, banner_text = format_broll_completion_banner(count, origin="pipeline_core")
    assert banner_success is False
    assert "⚠️" in banner_text

    report_path = tmp_path / "output" / "meta" / "selection_report_input.json"
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_payload["selection_rate"] == 1.0
    assert len(report_payload["segments"]) == 1
    segment_entry = report_payload["segments"][0]
    assert segment_entry["segment"] == 0
    assert segment_entry["candidates"]
    assert segment_entry["selected"]
    assert len(segment_entry["candidates"]) == 1
    assert len(segment_entry["selected"]) == 1
    assert segment_entry["candidates"][0] == {
        "provider": "pexels",
        "url": "http://example.com/a.mp4",
        "score": 1.0,
        "reject_reason": None,
        "selected": True,
    }
    assert segment_entry["selected"][0] == {
        "provider": "pexels",
        "url": "http://example.com/a.mp4",
        "score": 1.0,
    }


def test_selection_report_captures_candidates_success(monkeypatch, tmp_path):
    monkeypatch.setenv("FAST_TESTS", "1")

    monkeypatch.setattr(video_processor.Config, "CLIPS_FOLDER", tmp_path / "clips", raising=False)
    monkeypatch.setattr(video_processor.Config, "OUTPUT_FOLDER", tmp_path / "output", raising=False)
    monkeypatch.setattr(video_processor.Config, "TEMP_FOLDER", tmp_path / "temp", raising=False)

    processor = VideoProcessor()
    processor._pipeline_config.selection.min_score = 0.0  # type: ignore[attr-defined]

    events = []

    class _DummyLogger:
        def log(self, payload):
            events.append(dict(payload))

    monkeypatch.setattr(processor, "_get_broll_event_logger", lambda: _DummyLogger())

    class _DummyOrchestrator:
        def __init__(self, *args, **kwargs):
            pass

        def fetch_candidates(self, *args, **kwargs):
            return [
                types.SimpleNamespace(
                    url="http://example.com/a.mp4",
                    provider="pexels",
                    duration=2.0,
                    width=1920,
                    height=1080,
                ),
                types.SimpleNamespace(
                    url="http://example.com/b.mp4",
                    provider="pixabay",
                    duration=2.0,
                    width=1080,
                    height=1080,
                ),
            ]

        def evaluate_candidate_filters(self, *_args, **_kwargs):
            return True, None

    monkeypatch.setattr(video_processor, "FetcherOrchestrator", _DummyOrchestrator)

    def _fake_rank(_text, candidate, *_args, **_kwargs):
        return 0.9 if getattr(candidate, "url", "").endswith("a.mp4") else 0.3

    monkeypatch.setattr(processor, "_rank_candidate", _fake_rank)

    def _fake_download(_self, candidate, download_dir, order):
        download_dir.mkdir(parents=True, exist_ok=True)
        path = download_dir / f"asset_{order}.mp4"
        path.write_text("data", encoding="utf-8")
        return path

    monkeypatch.setattr(processor, "_download_core_candidate", _fake_download.__get__(processor, VideoProcessor))

    render_output = tmp_path / "output" / "render.mp4"

    def _fake_render(_self, _input_path, timeline):
        render_output.parent.mkdir(parents=True, exist_ok=True)
        render_output.write_text("render", encoding="utf-8")
        return render_output

    monkeypatch.setattr(processor, "_render_core_broll_timeline", _fake_render.__get__(processor, VideoProcessor))

    segment = types.SimpleNamespace(start=0.0, end=5.0, text="greetings earthlings")
    input_path = tmp_path / "clip.mp4"
    input_path.write_text("dummy", encoding="utf-8")

    count, render_path, meta = processor._insert_brolls_pipeline_core(
        [segment],
        ["greetings"],
        subtitles=None,
        input_path=input_path,
    )

    assert count == 1
    assert render_path == render_output
    assert meta.get("render_ok") is True

    report_path = tmp_path / "output" / "meta" / "selection_report_clip.json"
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_payload["selection_rate"] == 1.0
    assert len(report_payload["segments"]) == 1
    segment_entry = report_payload["segments"][0]
    assert segment_entry["segment"] == 0
    assert segment_entry["queries"]
    assert len(segment_entry["candidates"]) == 2
    assert sum(1 for candidate in segment_entry["candidates"] if candidate["selected"]) == 1
    assert segment_entry["selected"] == [
        {
            "provider": "pexels",
            "url": "http://example.com/a.mp4",
            "score": 0.9,
        }
    ]
    providers = {candidate["provider"] for candidate in segment_entry["candidates"]}
    assert providers == {"pexels", "pixabay"}
