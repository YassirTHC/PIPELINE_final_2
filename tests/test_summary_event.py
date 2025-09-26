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


def test_broll_summary_matches_console(tmp_path):
    log_path = tmp_path / "events.jsonl"
    logger = JsonlLogger(log_path)
    logger.log({
        "event": "broll_summary",
        "segments": 3,
        "inserted": 2,
        "providers_used": ["pexels"],
    })

    content = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert content, "expected broll summary event to be written"
    payload = json.loads(content[-1])
    assert payload["event"] == "broll_summary"
    assert payload["inserted"] == 2
    assert payload["segments"] == 3
    assert payload["providers_used"] == ["pexels"]

    fake_console_line = "    📊 B-roll sélectionnés: 2/3"
    import re

    match = re.search(r"B-roll sélectionnés:\s*(\d+)\s*/\s*(\d+)", fake_console_line)
    assert match, "expected to parse console summary"
    assert int(match.group(1)) == payload["inserted"]
    assert int(match.group(2)) == payload["segments"]
