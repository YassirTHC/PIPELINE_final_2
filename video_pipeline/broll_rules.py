from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List


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
    no_repeat_s: float
) -> List[BrollClip]:
    """
    Applique 3 invariants:
      1) Aucun B-roll avant `min_start_s`
      2) Espace minimal `min_gap_s` entre la fin du prÃ©cÃ©dent et le dÃ©but du suivant
      3) Anti-rÃ©pÃ©tition: mÃªme asset interdit dans une fenÃªtre `no_repeat_s`
    StratÃ©gie: tri par (start, duration), filtrage en un passage.
    """
    if not clips:
        return []

    # Normalisation/tri dÃ©terministe
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

    min_start_s = float(min_start_s)
    min_gap_s = float(min_gap_s)
    no_repeat_s = float(no_repeat_s)

    for c in clips:
        # (1) Hook initial
        if c.start_s < min_start_s:
            _log(
                f"[BROLL] skip: too-early (<min_start) start={c.start_s:.2f}s asset={c.asset_id}"
            )
            continue

        # (2) Gap minimal
        if kept and (c.start_s - last_end) < min_gap_s:
            _log(
                f"[BROLL] skip: too-close (<min_gap) start={c.start_s:.2f}s asset={c.asset_id}"
            )
            continue

        # (3) Anti-repeat sur fenÃªtre temporelle
        prev_t = last_by_asset.get(c.asset_id)
        if prev_t is not None and (c.start_s - prev_t) < no_repeat_s:
            _log(
                f"[BROLL] skip: repeated (<no_repeat) start={c.start_s:.2f}s asset={c.asset_id}"
            )
            continue

        kept.append(c)
        last_end = c.end_s
        last_by_asset[c.asset_id] = c.start_s

    return kept



# --- SAFE PRINT OVERRIDE (appended for pytest capture robustness) ---
try:
    from video_pipeline.utils.console import safe_print as __safe_print
    def _log(message: str) -> None:
        try:
            __safe_print(message)
        except Exception:
            # never raise from logging
            pass
except Exception:
    # last resort: best-effort write to a real stream if present, swallow otherwise
    def _log(message: str) -> None:
        try:
            import sys
            stream = getattr(sys, "__stdout__", None) or getattr(sys, "stdout", None)
            if stream:
                text = str(message) if message is not None else ""
                if text and not text.endswith("\n"):
                    text += "\n"
                stream.write(text)
                try: stream.flush()
                except Exception: pass
        except Exception:
            pass
# --- END OVERRIDE ---
