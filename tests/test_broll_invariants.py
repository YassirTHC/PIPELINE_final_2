from pathlib import Path
from types import SimpleNamespace

from test_duration_gap_rules import _load_video_processor
from video_pipeline.broll_rules import BrollClip, enforce_broll_schedule_rules


def _c(t0, t1, aid, seg):
    return BrollClip(start_s=t0, end_s=t1, asset_id=aid, segment_index=seg)


def test_filters_min_start_and_gap():
    clips = [
        _c(0.5, 1.5, "pexels:1", 0),
        _c(2.0, 3.0, "pexels:2", 1),
        _c(3.2, 4.0, "pexels:3", 2),
        _c(4.6, 5.4, "pexels:4", 3),
    ]
    out = enforce_broll_schedule_rules(
        clips,
        min_start_s=2.0,
        min_gap_s=1.5,
        no_repeat_s=6.0,
    )
    assert [(c.asset_id, c.start_s) for c in out] == [("pexels:2", 2.0), ("pexels:4", 4.6)]


def test_anti_repeat_window():
    clips = [
        _c(2.0, 3.0, "pixabay:42", 0),
        _c(6.9, 7.5, "pixabay:42", 1),
        _c(8.1, 9.0, "pixabay:99", 2),
        _c(9.0, 9.8, "pixabay:42", 3),
    ]
    out = enforce_broll_schedule_rules(
        clips,
        min_start_s=0.0,
        min_gap_s=0.0,
        no_repeat_s=6.0,
    )
    assert [c.asset_id for c in out] == ["pixabay:42", "pixabay:99", "pixabay:42"]


def test_stable_sort_and_duration_guard():
    clips = [
        _c(2.0, 4.0, "a:long", 0),
        _c(2.0, 2.5, "b:short", 1),
        _c(4.6, 5.0, "c:ok", 2),
    ]
    out = enforce_broll_schedule_rules(
        clips,
        min_start_s=0.0,
        min_gap_s=1.5,
        no_repeat_s=0.0,
    )
    assert [c.asset_id for c in out] == ["b:short", "c:ok"]


def test_core_pipeline_integration_applies_invariants(monkeypatch):
    vp = _load_video_processor()

    dummy_settings = SimpleNamespace(
        broll=SimpleNamespace(min_start_s=2.0, min_gap_s=1.5, no_repeat_s=6.0)
    )
    monkeypatch.setattr(vp, "get_settings", lambda: dummy_settings, raising=False)

    entries = [
        vp.CoreTimelineEntry(
            path=Path("clip_hook.mp4"),
            start=0.5,
            end=1.4,
            segment_index=0,
            url="asset:hook",
        ),
        vp.CoreTimelineEntry(
            path=Path("clip_one.mp4"),
            start=2.0,
            end=3.0,
            segment_index=1,
            url="asset:repeat",
        ),
        vp.CoreTimelineEntry(
            path=Path("clip_gap.mp4"),
            start=3.1,
            end=4.0,
            segment_index=2,
            url="asset:gap",
        ),
        vp.CoreTimelineEntry(
            path=Path("clip_two.mp4"),
            start=4.7,
            end=5.4,
            segment_index=3,
            url="asset:unique",
        ),
        vp.CoreTimelineEntry(
            path=Path("clip_three.mp4"),
            start=11.2,
            end=12.0,
            segment_index=4,
            url="asset:repeat",
        ),
    ]

    updates = [{"url": entry.url} for entry in entries]

    filtered_entries, filtered_updates = vp._apply_broll_invariants_to_core_entries(
        entries,
        seen_updates=updates,
    )

    assert [entry.url for entry in filtered_entries] == [
        "asset:repeat",
        "asset:unique",
        "asset:repeat",
    ]
    assert filtered_updates is not None
    assert [update["url"] for update in filtered_updates] == [
        "asset:repeat",
        "asset:unique",
        "asset:repeat",
    ]
