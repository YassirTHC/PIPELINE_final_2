from __future__ import annotations

from tests.factories import plan_from_tuples
from video_pipeline.broll_rules import enforce_broll_schedule_rules


def test_broll_min_gap_respected():
    clips = plan_from_tuples(
        [
            (2.0, 3.0, "segment:a"),
            (3.2, 4.0, "segment:b"),
            (4.3, 5.0, "segment:c"),
            (5.6, 6.4, "segment:d"),
        ]
    )

    kept = enforce_broll_schedule_rules(
        clips,
        min_start_s=0.0,
        min_gap_s=1.5,
        no_repeat_s=0.0,
    )

    assert [clip.asset_id for clip in kept] == ["segment:a", "segment:d"]
    for first, second in zip(kept, kept[1:]):
        assert (second.start_s - first.end_s) >= 1.5 - 1e-6
