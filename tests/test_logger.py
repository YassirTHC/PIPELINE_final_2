from pathlib import Path
import json

from pipeline_core.logging import JSONLLogger


def test_jsonl_logger_writes(tmp_path: Path):
    log_file = tmp_path / "events.jsonl"
    logger = JSONLLogger(log_file)
    logger.write_jsonl({"hello": "world", "n": 1})
    logger.write_jsonl({"ok": True})

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["hello"] == "world"
