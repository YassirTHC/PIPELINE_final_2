"""Test helpers for constructing deterministic B-roll clips."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from video_pipeline.broll_rules import BrollClip


def make_clip(
    start_s: float,
    end_s: float,
    asset_id: str,
    *,
    segment_index: int = 0,
) -> BrollClip:
    """Create a ``BrollClip`` with explicit typing for tests."""

    return BrollClip(
        start_s=float(start_s),
        end_s=float(end_s),
        asset_id=str(asset_id),
        segment_index=int(segment_index),
    )


def plan_from_tuples(
    items: Sequence[Tuple[float, float, str]],
    *,
    segment_offset: int = 0,
) -> List[BrollClip]:
    """Convert ``(start, end, asset)`` tuples into ``BrollClip`` instances."""

    clips: List[BrollClip] = []
    for idx, (start_s, end_s, asset_id) in enumerate(items, start=segment_offset):
        clips.append(
            make_clip(
                start_s=start_s,
                end_s=end_s,
                asset_id=asset_id,
                segment_index=idx,
            )
        )
    return clips
