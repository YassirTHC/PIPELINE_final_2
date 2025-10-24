from pathlib import Path

from smart_reframe import SmartReframeConfig, smart_reframe_broll


def test_smart_reframe_returns_original_when_processing_unavailable(tmp_path):
    src = tmp_path / "clip.mp4"
    out = tmp_path / "clip_smart.mp4"
    src.write_bytes(b"\x00\x00")

    cfg = SmartReframeConfig()
    result = smart_reframe_broll(str(src), str(out), cfg)

    assert Path(result) == src
    assert not out.exists()
