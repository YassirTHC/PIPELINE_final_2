import json
import numpy as np

from pipeline_core.logging import JsonlLogger


def test_json_logger_coerces_numpy(tmp_path):
    path = tmp_path / "events.jsonl"
    logger = JsonlLogger(path)
    logger.log({"event": "test", "value": np.float32(0.5)})
    data = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert isinstance(data["value"], float)
    assert data["value"] == 0.5
