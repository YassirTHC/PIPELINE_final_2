"""Transcript analysis helpers for the modular pipeline."""
from __future__ import annotations

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
            start = float(payload.get("start", 0.0) or 0.0)
            end = float(payload.get("end", start) or start)
            if end < start:
                end = start
            result.append(TranscriptSegment(start=start, end=end, text=text))
        return result
