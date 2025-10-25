# -*- coding: utf-8 -*-
from types import SimpleNamespace

import pytest

from video_processor import _enforce_broll_runtime_constraints


def _make_event(start, end, media, transition=0.1):
    return SimpleNamespace(
        start_s=start,
        end_s=end,
        media_path=media,
        transition_duration=transition,
    )


def test_runtime_constraints_adjust_duration_and_transition():
    events = [
        _make_event(0.0, 1.0, "asset:a.mp4"),
        _make_event(5.0, 9.5, "asset:b.mp4", transition=0.9),
        {'start': 12.0, 'end': 13.0, 'media_path': 'asset:c.mp4'},
    ]

    adjusted = _enforce_broll_runtime_constraints(
        events,
        min_duration=2.0,
        max_duration=3.2,
        min_transition=0.35,
        max_transition=0.45,
        cooldown_s=4.0,
    )

    assert len(adjusted) == 3

    first = adjusted[0]
    assert pytest.approx(first.end_s - first.start_s, rel=1e-6) == 2.0
    assert 0.35 <= first.transition_duration <= 0.45

    second = adjusted[1]
    assert pytest.approx(second.end_s - second.start_s, rel=1e-6) == 3.2
    assert 0.35 <= second.transition_duration <= 0.45

    third = adjusted[2]
    assert isinstance(third, dict)
    assert pytest.approx(third['end'] - third['start'], rel=1e-6) == 2.0
    assert third['duration_s'] == pytest.approx(2.0, rel=1e-6)
    assert 0.35 <= third['transition_duration'] <= 0.45


def test_runtime_constraints_cooldown_and_no_consecutive():
    events = [
        _make_event(0.0, 3.0, "asset:dup.mp4"),
        _make_event(2.5, 5.5, "asset:dup.mp4"),
        _make_event(6.0, 8.0, "asset:alt.mp4"),
        _make_event(9.0, 11.5, "asset:dup.mp4"),
    ]

    adjusted = _enforce_broll_runtime_constraints(
        events,
        min_duration=2.0,
        max_duration=3.2,
        min_transition=0.35,
        max_transition=0.45,
        cooldown_s=3.0,
    )

    media_sequence = [evt.media_path if hasattr(evt, "media_path") else evt.get("media_path") for evt in adjusted]
    assert media_sequence == ["asset:dup.mp4", "asset:alt.mp4", "asset:dup.mp4"]

    # Ensure the cooldown rule kept the final event because enough time elapsed.
    last_dup = adjusted[-1]
    first_dup = adjusted[0]
    gap = last_dup.start_s - first_dup.end_s
    assert gap >= 3.0 - 1e-6

