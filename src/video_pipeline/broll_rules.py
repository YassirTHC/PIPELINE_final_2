from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict


@dataclass(frozen=True)
class BrollClip:
    start_s: float
    end_s: float
    asset_id: str
    segment_index: int


def _duration(c: BrollClip) -> float:
    return max(0.0, c.end_s - c.start_s)


def enforce_broll_schedule_rules(
    clips: List[BrollClip],
    *,
    min_start_s: float,
    min_gap_s: float,
    no_repeat_s: float
) -> List[BrollClip]:
    """
    Applique 3 invariants:
      1) Aucun B-roll avant `min_start_s`
      2) Espace minimal `min_gap_s` entre la fin du précédent et le début du suivant
      3) Anti-répétition: même asset interdit dans une fenêtre `no_repeat_s`
    Stratégie: tri par (start, duration), filtrage en un passage.
    """
    if not clips:
        return []

    # Normalisation/tri déterministe
    clips = [
        BrollClip(
            start_s=max(0.0, float(c.start_s)),
            end_s=max(float(c.start_s), float(c.end_s)),
            asset_id=str(c.asset_id),
            segment_index=int(c.segment_index),
        )
        for c in clips
    ]
    clips.sort(key=lambda c: (c.start_s, _duration(c)))

    kept: List[BrollClip] = []
    last_end: float = float("-inf")
    last_by_asset: Dict[str, float] = {}

    for c in clips:
        # (1) Hook initial
        if c.start_s < float(min_start_s):
            continue

        # (2) Gap minimal
        if kept and (c.start_s - last_end) < float(min_gap_s):
            continue

        # (3) Anti-repeat sur fenêtre temporelle
        prev_t = last_by_asset.get(c.asset_id)
        if prev_t is not None and (c.start_s - prev_t) < float(no_repeat_s):
            continue

        kept.append(c)
        last_end = c.end_s
        last_by_asset[c.asset_id] = c.start_s

    return kept
