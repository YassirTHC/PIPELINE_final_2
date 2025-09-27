"""JSONL logging utilities for the modular pipeline."""
from __future__ import annotations

import json
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pipeline_core.runtime import PipelineResult


def _coerce_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _coerce_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_coerce_jsonable(v) for v in value]
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, 'tolist'):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


class JsonlLogger:
    """Appends structured events to a JSONL file in a threadsafe way."""

    def __init__(self, destination: Path) -> None:
        self._destination = Path(destination)
        self._destination.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, payload: Dict[str, Any]) -> None:
        event = _coerce_jsonable(dict(payload))
        if isinstance(event, dict):
            event.setdefault('ts', time.time())
        else:
            event = {
                'event': str(payload.get('event', 'unknown')),
                'ts': time.time(),
                'payload': event,
            }
        with self._lock:
            with self._destination.open('a', encoding='utf-8', newline='') as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + '\n')

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
    queries: Optional[List[str]] = None,
    provider_status: Optional[Dict[str, Any]] = None,
    best_score: Optional[float] = None,
    reject_summary: Optional[Dict[str, Any]] = None,
) -> None:
    event_name = 'broll_segment_decision' if segment_idx >= 0 else 'broll_session_summary'
    reason_counts = Counter(reject_reasons or [])

    payload = {
        'event': event_name,
        'segment': segment_idx,
        't0': start,
        't1': end,
        'q_count': query_count,
        'candidates': candidate_count,
        'unique_candidates': unique_candidates,
        'dedup_url_hits': url_dedup_hits,
        'dedup_phash_hits': phash_dedup_hits,
        'selected_url': selected_url,
        'selected_score': selected_score,
        'provider': provider,
        'latency_ms': latency_ms,
        'llm_healthy': llm_healthy,
        'reject_reasons': dict(reason_counts),
        'best_score': best_score,
    }
    if queries is not None:
        payload['queries'] = list(queries)
    if provider_status:
        payload['providers'] = provider_status
    if reject_summary is not None:
        payload['reject_summary'] = reject_summary
    elif reason_counts:
        payload['reject_summary'] = {'counts': dict(reason_counts)}
    logger.log(payload)


JSONLLogger = JsonlLogger


class StageEventLogger:
    def __init__(self, jsonl_logger: JsonlLogger):
        self._jsonl = jsonl_logger

    def log_stage_start(self, stage: str, details: Optional[Dict[str, Any]] = None) -> float:
        started_at = time.time()
        payload = {"event": "pipeline_stage_start", "stage": stage}
        if details:
            payload["details"] = details
        payload["ts_start"] = started_at
        self._jsonl.log(payload)
        return started_at

    def log_stage_end(self, stage: str, *, started_at: float, ok: bool, details: Optional[Dict[str, Any]] = None) -> None:
        latency_ms = int(max(0.0, (time.time() - started_at)) * 1000)
        payload = {"event": "pipeline_stage_end", "stage": stage, "ok": bool(ok), "latency_ms": latency_ms}
        if details:
            payload["details"] = details
        self._jsonl.log(payload)

    def log_error(self, stage: str, *, error_type: str, message: str, trace_hint: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> None:
        payload = {"event": "pipeline_error", "stage": stage, "error_type": error_type, "message": message}
        if trace_hint:
            payload["trace_hint"] = trace_hint
        if details:
            payload["details"] = details
        self._jsonl.log(payload)

    def log_summary(self, summary: Dict[str, Any]) -> None:
        payload = {"event": "pipeline_summary"}
        payload.update(summary)
        self._jsonl.log(payload)


def log_stage_start(logger: JsonlLogger, stage: str, details: Optional[Dict[str, Any]] = None) -> float:
    """Emit a start event for a pipeline stage and return the start timestamp."""
    started_at = time.time()
    payload: Dict[str, Any] = {
        "event": "pipeline_stage_start",
        "stage": str(stage),
    }
    if details:
        payload["details"] = details
    payload["ts_start"] = started_at
    logger.log(payload)
    return started_at


def log_stage_end(
    logger: JsonlLogger,
    stage: str,
    *,
    started_at: float,
    ok: bool,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a completion event for a pipeline stage."""
    latency_ms = int(max(0.0, time.time() - started_at) * 1000)
    payload: Dict[str, Any] = {
        "event": "pipeline_stage_end",
        "stage": str(stage),
        "ok": bool(ok),
        "latency_ms": latency_ms,
    }
    if details:
        payload["details"] = details
    logger.log(payload)


def log_pipeline_error(
    logger: JsonlLogger,
    stage: str,
    *,
    error_type: str,
    message: str,
    trace_hint: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a structured error event for troubleshooting."""
    payload: Dict[str, Any] = {
        "event": "pipeline_error",
        "stage": str(stage),
        "error_type": error_type,
        "message": message,
    }
    if trace_hint:
        payload["trace_hint"] = trace_hint
    if details:
        payload["details"] = details
    logger.log(payload)


def log_pipeline_summary(
    logger: JsonlLogger,
    result: PipelineResult,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit the final summary event aggregating pipeline execution flags."""
    summary = result.to_dict()
    duration = summary.pop("duration_s", None)
    payload: Dict[str, Any] = {
        "event": "pipeline_summary",
        "stage": "pipeline",
        "ok": bool(summary.get("final_export_ok")),
    }
    if duration is not None:
        payload["duration_ms"] = int(max(0.0, duration) * 1000)
    payload.update(summary)
    if extra:
        payload.update(extra)
    logger.log(payload)

