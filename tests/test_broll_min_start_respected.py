from __future__ import annotations

from tests.factories import plan_from_tuples
from video_pipeline.broll_rules import enforce_broll_schedule_rules


def test_broll_min_start_respected():
    clips = plan_from_tuples(
        [
            (0.5, 1.6, "hook:ignore"),
            (1.9, 3.0, "hook:still-too-soon"),
            (2.1, 3.4, "segment:a"),
            (4.4, 5.2, "segment:b"),
        ]
    )

    kept = enforce_broll_schedule_rules(
        clips,
        min_start_s=2.0,
        min_gap_s=0.0,
        no_repeat_s=0.0,
    )

    assert [clip.asset_id for clip in kept] == ["segment:a", "segment:b"]
    assert all(clip.start_s >= 2.0 for clip in kept)
