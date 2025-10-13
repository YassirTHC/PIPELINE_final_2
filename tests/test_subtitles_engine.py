from __future__ import annotations

from pathlib import Path
import types

import pytest

import video_processor
from subtitle_engines import pycaps_engine


def test_router_uses_pycaps_when_engine_selected(monkeypatch, tmp_path):
    segments = [{"text": "hello", "start": 0.0, "end": 1.0}]
    template_dir = tmp_path / "tmpl"
    template_dir.mkdir()
    (template_dir / "pycaps.template.json").write_text("{}", encoding="utf-8")

    settings = types.SimpleNamespace(subtitles=types.SimpleNamespace(engine="pycaps", enable_emojis=True))
    monkeypatch.setattr(video_processor, "get_settings", lambda: settings)

    recorded: dict[str, str] = {}

    def fake_render(subs, output_video_path, template_path, *, input_video_path):
        recorded["output"] = output_video_path
        recorded["template"] = template_path
        recorded["input"] = input_video_path

    monkeypatch.setattr(video_processor, "ensure_template_assets", lambda *_: None)
    monkeypatch.setattr(video_processor, "render_with_pycaps", fake_render)

    def fail_hormozi(*args, **kwargs):
        raise AssertionError("Hormozi disabled")

    monkeypatch.setattr(video_processor, "_render_subtitles_with_hormozi", fail_hormozi)

    video_processor.render_subtitles_router(
        tmp_path / "input.mp4",
        segments,
        tmp_path / "output.mp4",
        template_dir=template_dir,
    )

    assert recorded["input"].endswith("input.mp4")
    assert recorded["output"].endswith("output.mp4")
    assert settings.subtitles.enable_emojis is False


def test_render_with_pycaps_missing_dependency(monkeypatch, tmp_path):
    template_dir = tmp_path / "pycaps"
    template_dir.mkdir()
    (template_dir / "pycaps.template.json").write_text("{}", encoding="utf-8")

    def fail_loader():
        raise ModuleNotFoundError("pycaps not installed")

    monkeypatch.setattr(pycaps_engine, "_load_pycaps_loader", fail_loader)

    with pytest.raises(RuntimeError) as excinfo:
        pycaps_engine.render_with_pycaps(
            [],
            tmp_path / "out.mp4",
            template_dir,
            input_video_path=tmp_path / "src.mp4",
        )

    assert "pip install pycaps" in str(excinfo.value)
