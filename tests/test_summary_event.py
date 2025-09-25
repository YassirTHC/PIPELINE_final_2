import json

from pipeline_core.logging import JsonlLogger, log_pipeline_summary
from pipeline_core.runtime import PipelineResult


def test_pipeline_summary_event_contains_flags(tmp_path):
    log_path = tmp_path / "events.jsonl"
    logger = JsonlLogger(log_path)

    result = PipelineResult()
    result.final_export_ok = True
    result.broll_inserted_count = 3
    result.finish()

    log_pipeline_summary(logger, result, extra={"effective_domain": "generic", "queries_count": 5})

    content = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert content, "expected summary event to be written"
    payload = json.loads(content[-1])

    assert payload["event"] == "pipeline_summary"
    assert payload["stage"] == "pipeline"
    assert payload["final_export_ok"] is True
    assert payload["effective_domain"] == "generic"
    assert payload["broll_inserted_count"] == 3
    assert "duration_ms" in payload
