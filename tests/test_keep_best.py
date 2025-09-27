from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline_core.configuration import SelectionConfig
from tests.test_no_repeat_assets import _load_video_processor


def test_forced_keep_budget_is_enforced(monkeypatch):
    monkeypatch.setenv("BROLL_MIN_SCORE", "0.95")
    monkeypatch.setenv("BROLL_FORCED_KEEP", "1")

    vp = _load_video_processor()
    vp.SEEN_URLS.clear()
    vp.SEEN_PHASHES.clear()
    vp.SEEN_IDENTIFIERS.clear()

    selection = SelectionConfig.from_environment()
    assert selection.min_score == pytest.approx(0.95)
    assert selection.forced_keep_budget == 1
    assert selection.allow_forced_keep is True

    events = []

    class DummyLogger:
        def log(self, payload):
            events.append(payload)

    dummy_logger = DummyLogger()

    processor = vp.VideoProcessor.__new__(vp.VideoProcessor)
    processor._pipeline_config = SimpleNamespace(
        fetcher=SimpleNamespace(providers=["stub"]),
        selection=selection,
        timeboxing=SimpleNamespace(fetch_rank_ms=0, request_timeout_s=1),
    )
    processor._core_last_run_used = False
    processor._llm_service = None
    processor._dyn_context = {}
    processor._selector_keywords = []
    processor._fetch_keywords = []
    processor._get_broll_event_logger = lambda: dummy_logger
    processor._derive_segment_keywords = lambda segment, keywords: ["keyword"]

    candidate_one = SimpleNamespace(
        url="https://cdn/fallback-one.mp4",
        identifier="asset-1",
        width=1920,
        height=1080,
        duration=4.0,
        provider="stub",
    )
    candidate_two = SimpleNamespace(
        url="https://cdn/fallback-two.mp4",
        identifier="asset-2",
        width=1920,
        height=1080,
        duration=4.0,
        provider="stub",
    )

    candidates_iter = iter([[candidate_one], [candidate_two]])

    def fake_fetch_candidates(*_args, **_kwargs):
        try:
            return next(candidates_iter)
        except StopIteration:
            return []

    original_fetcher = vp.FetcherOrchestrator
    original_dedupe_by_phash = vp.dedupe_by_phash
    original_download = vp.VideoProcessor._download_core_candidate
    original_render = vp.VideoProcessor._render_core_broll_timeline
    original_rank = vp.VideoProcessor._rank_candidate
    try:
        vp.FetcherOrchestrator = lambda cfg: SimpleNamespace(
            fetch_candidates=fake_fetch_candidates,
            evaluate_candidate_filters=lambda *args, **kwargs: (True, None),
        )
        vp.dedupe_by_phash = lambda candidates: (candidates, 0)
        vp.VideoProcessor._download_core_candidate = (
            lambda self, *_args, **_kwargs: Path("core_asset.mp4")
        )
        vp.VideoProcessor._render_core_broll_timeline = (
            lambda self, *_args, **_kwargs: Path("rendered.mp4")
        )
        vp.VideoProcessor._rank_candidate = lambda *args, **kwargs: 0.1

        inserted, _ = processor._insert_brolls_pipeline_core(
            segments=[
                SimpleNamespace(start=0.0, end=4.0, text="hello world"),
                SimpleNamespace(start=5.0, end=9.0, text="another"),
            ],
            broll_keywords=["keyword"],
            subtitles=None,
            input_path=Path("video.mp4"),
        )
    finally:
        vp.FetcherOrchestrator = original_fetcher
        vp.dedupe_by_phash = original_dedupe_by_phash
        vp.VideoProcessor._download_core_candidate = original_download
        vp.VideoProcessor._render_core_broll_timeline = original_render
        vp.VideoProcessor._rank_candidate = original_rank

    assert inserted == 1

    forced_events = [event for event in events if event.get("event") == "forced_keep_consumed"]
    assert len(forced_events) == 1
    forced_event = forced_events[0]
    assert forced_event["provider"] == "stub"
    assert forced_event["url"] == "https://cdn/fallback-one.mp4"
    assert forced_event["remaining_budget"] == 0

    skipped_events = [event for event in events if event.get("event") == "forced_keep_skipped"]
    assert skipped_events and skipped_events[0]["reason"] == "exhausted"

    summary_events = [event for event in events if event.get("event") == "broll_summary"]
    assert summary_events, "expected summary event"
    summary = summary_events[-1]
    assert summary["forced_keep_count"] == 1
    assert summary["forced_keep_segments"] == 1
    assert summary["selected_segments"] == [0]

    decision_events = [event for event in events if event.get("event") == "broll_segment_decision"]
    assert decision_events, "expected decision events"
    selected_urls = [event.get("selected_url") for event in decision_events if event.get("selected_url")]
    assert selected_urls == ["https://cdn/fallback-one.mp4"]
    assert "https://cdn/fallback-two.mp4" not in selected_urls
