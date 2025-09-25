"""Runtime helpers for pipeline execution state."""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict


class Stage(str, Enum):
    REFRAME = "reframe"
    ASR = "asr"
    LLM_META = "llm_meta"
    DYN_CTX = "dyn_ctx"
    BROLL_CORE = "broll_core"
    BROLL_LEGACY = "broll_legacy"
    SUBTITLES = "subtitles"
    EXPORT = "export"
    PIPELINE = "pipeline"


@dataclass
class PipelineResult:
    reframe_ok: Optional[bool] = None
    asr_ok: Optional[bool] = None
    llm_meta_ok: Optional[bool] = None
    dyn_context_ok: Optional[bool] = None
    broll_core_ok: Optional[bool] = None
    broll_legacy_ok: Optional[bool] = None
    broll_inserted_count: int = 0
    schedule_drop_counts: Dict[str, int] = field(default_factory=dict)
    subtitles_ok: Optional[bool] = None
    final_export_ok: Optional[bool] = None
    final_export_path: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    start_ts: float = field(default_factory=time.time)
    end_ts: Optional[float] = None

    def finish(self) -> None:
        self.end_ts = time.time()

    def duration_s(self) -> Optional[float]:
        if self.end_ts is None:
            return None
        return self.end_ts - self.start_ts

    def to_dict(self) -> dict:
        data = asdict(self)
        data["duration_s"] = self.duration_s()
        return data
