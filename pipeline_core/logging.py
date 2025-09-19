"""JSONL logging utilities for the modular pipeline."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class JsonlLogger:
    """Appends structured events to a JSONL file in a threadsafe way."""

    def __init__(self, destination: Path) -> None:
        self._destination = Path(destination)
        self._destination.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, payload: Dict[str, Any]) -> None:
        event = dict(payload)
        event.setdefault("ts", time.time())
        with self._lock:
            with self._destination.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def write_jsonl(self, payload: Dict[str, Any]) -> None:
        """Backward compatible alias for callers expecting write_jsonl."""
        self.log(payload)

    @property
    def path(self) -> Path:
        return self._destination


def log_broll_decision(
    logger: JsonlLogger,
    *,
    segment_idx: int,
    start: float,
    end: float,
    query_count: int,
    candidate_count: int,
    unique_candidates: int,
    url_dedup_hits: int,
    phash_dedup_hits: int,
    selected_url: Optional[str],
    selected_score: Optional[float],
    provider: Optional[str],
    latency_ms: int,
    llm_healthy: bool,
    reject_reasons: List[str],
) -> None:
    logger.log(
        {
            "segment": segment_idx,
            "t0": start,
            "t1": end,
            "q_count": query_count,
            "candidates": candidate_count,
            "unique_candidates": unique_candidates,
            "dedup_url_hits": url_dedup_hits,
            "dedup_phash_hits": phash_dedup_hits,
            "selected_url": selected_url,
            "selected_score": selected_score,
            "provider": provider,
            "latency_ms": latency_ms,
            "llm_healthy": llm_healthy,
            "reject_reasons": reject_reasons,
        }
    )

JSONLLogger = JsonlLogger
