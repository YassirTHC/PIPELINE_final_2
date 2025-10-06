from __future__ import annotations

from tests.factories import plan_from_tuples
from video_pipeline.broll_rules import enforce_broll_schedule_rules


def test_broll_no_repeat_window():
    clips = plan_from_tuples(
        [
            (2.0, 3.2, "asset:loop"),
            (6.5, 7.4, "asset:loop"),
            (8.0, 8.9, "asset:unique"),
            (11.9, 12.8, "asset:loop"),
        ]
    )

    kept = enforce_broll_schedule_rules(
        clips,
        min_start_s=0.0,
        min_gap_s=0.0,
        no_repeat_s=6.0,
    )

    assert [clip.asset_id for clip in kept] == [
        "asset:loop",
        "asset:unique",
        "asset:loop",
    ]
    assert kept[1].start_s - kept[0].start_s >= 6.0
