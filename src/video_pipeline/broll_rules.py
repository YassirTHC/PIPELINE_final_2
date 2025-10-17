from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


def _log(message: str) -> None:
    """Emit scheduling logs via print to match pipeline expectations."""

    print(message)


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
    no_repeat_s: float,
    max_gap_s: Optional[float] = None,
    target_total: Optional[int] = None,
) -> List[BrollClip]:
    """
    Apply scheduling invariants on the proposed B-roll clips.
    """
    if not clips:
        return []

    normalized = [
        BrollClip(
            start_s=max(0.0, float(c.start_s)),
            end_s=max(float(c.start_s), float(c.end_s)),
            asset_id=str(c.asset_id),
            segment_index=int(c.segment_index),
        )
        for c in clips
    ]
    normalized.sort(key=lambda c: (c.start_s, _duration(c)))

    kept: List[BrollClip] = []
    last_end: float = float("-inf")
    last_by_asset: Dict[str, float] = {}

    min_start = float(min_start_s)
    min_gap = float(min_gap_s)
    no_repeat = float(no_repeat_s)
    max_gap = float(max_gap_s) if max_gap_s is not None else None
    target_cap = int(target_total) if target_total not in (None, 0) else None

    for clip in normalized:
        if clip.start_s < min_start:
            _log(
                f"[BROLL] skip: too-early (<min_start) start={clip.start_s:.2f}s asset={clip.asset_id}"
            )
            continue

        gap_from_prev_end = float("inf") if not kept else clip.start_s - last_end
        if kept and gap_from_prev_end < min_gap:
            _log(
                f"[BROLL] skip: too-close (<min_gap) start={clip.start_s:.2f}s asset={clip.asset_id}"
            )
            continue

        prev_usage = last_by_asset.get(clip.asset_id)
        repeated = prev_usage is not None and (clip.start_s - prev_usage) < no_repeat
        if repeated:
            if target_cap is not None and len(kept) < target_cap:
                _log(
                    f"[BROLL] keep: repeated asset allowed to reach target start={clip.start_s:.2f}s asset={clip.asset_id}"
                )
            else:
                _log(
                    f"[BROLL] skip: repeated (<no_repeat) start={clip.start_s:.2f}s asset={clip.asset_id}"
                )
                continue

        kept.append(clip)
        if (
            max_gap is not None
            and kept
            and last_end != float("-inf")
            and gap_from_prev_end > max_gap
        ):
            _log(
                f"[BROLL] gap-warning (>max_gap) gap={gap_from_prev_end:.2f}s asset={clip.asset_id}"
            )

        last_end = clip.end_s
        last_by_asset[clip.asset_id] = clip.start_s

    return kept


# --- SAFE PRINT OVERRIDE (appended for pytest capture robustness) ---
try:
    from video_pipeline.utils.console import safe_print as __safe_print

    def _log(message: str) -> None:
        try:
            __safe_print(message)
        except Exception:
            pass

except Exception:

    def _log(message: str) -> None:
        try:
            import sys

            stream = getattr(sys, "__stdout__", None) or getattr(sys, "stdout", None)
            if stream:
                text = str(message) if message is not None else ""
                if text and not text.endswith("\n"):
                    text += "\n"
                stream.write(text)
                try:
                    stream.flush()
                except Exception:
                    pass
        except Exception:
            pass

# --- END OVERRIDE ---
