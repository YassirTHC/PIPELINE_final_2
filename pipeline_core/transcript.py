"""Transcript analysis helpers for the modular pipeline."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence


@dataclass(slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


class TranscriptAnalyzer:
    """Adapts existing transcription functions to the new pipeline."""

    def __init__(self, transcribe_segments: Callable[..., Sequence[Dict]]):
        self._transcribe_segments = transcribe_segments

    def analyse(self, input_path, /, **kwargs) -> List[TranscriptSegment]:
        segments = self._transcribe_segments(input_path, **kwargs) or []
        result: List[TranscriptSegment] = []
        for payload in segments:
            text = str(payload.get("text", "")).strip() if isinstance(payload, dict) else ""
            if not text:
                continue
            start_raw = payload.get("start", 0.0) if isinstance(payload, dict) else 0.0
            end_raw = payload.get("end", start_raw) if isinstance(payload, dict) else start_raw
            try:
                start = float(start_raw)
            except (TypeError, ValueError):
                start = 0.0
            try:
                end = float(end_raw)
            except (TypeError, ValueError):
                end = start
            if not math.isfinite(start) or start < 0:
                start = 0.0
            if not math.isfinite(end):
                end = start
            if end < start:
                end = start
            result.append(TranscriptSegment(start=start, end=end, text=text))
        return result
