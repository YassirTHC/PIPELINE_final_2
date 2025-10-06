import math
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
