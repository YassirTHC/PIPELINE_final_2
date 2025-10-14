from __future__ import annotations

from pathlib import Path
import types
import sys
import importlib
import pkgutil

import pytest

import video_processor
from subtitle_engines import pycaps_engine


def _clear_pycaps_modules(monkeypatch):
    for name in list(sys.modules):
        if name == "pycaps" or name.startswith("pycaps."):
            monkeypatch.delitem(sys.modules, name, raising=False)


def test_load_pycaps_loader_prefers_pipeline_layout(monkeypatch):
    _clear_pycaps_modules(monkeypatch)

    pipeline_module = types.ModuleType("pycaps.pipeline")

    class Loader:
        pass

    pipeline_module.JsonConfigLoader = Loader

    package = types.ModuleType("pycaps")
    package.__path__ = []  # mark as package for import machinery

    monkeypatch.setitem(sys.modules, "pycaps", package)
    monkeypatch.setitem(sys.modules, "pycaps.pipeline", pipeline_module)

    assert pycaps_engine._load_pycaps_loader() is Loader


def test_load_pycaps_loader_supports_root_layout(monkeypatch):
    _clear_pycaps_modules(monkeypatch)

    class Loader:
        pass

    package = types.ModuleType("pycaps")
    package.JsonConfigLoader = Loader
    package.__path__ = []

    monkeypatch.setitem(sys.modules, "pycaps", package)

    assert pycaps_engine._load_pycaps_loader() is Loader


def test_load_pycaps_loader_discovers_nested_modules(monkeypatch):
    _clear_pycaps_modules(monkeypatch)

    package = types.ModuleType("pycaps")
    package.__path__ = ["<pycaps>"]

    class Loader:
        pass

    def fake_iter_modules(path):
        assert list(path) == ["<pycaps>"]
        yield pkgutil.ModuleInfo(None, "alt_layout", False)

    def fake_import(name):
        if name == "pycaps.alt_layout":
            module = types.ModuleType(name)
            module.JsonConfigLoader = Loader
            return module
        raise ModuleNotFoundError(name)

    monkeypatch.setitem(sys.modules, "pycaps", package)
    monkeypatch.setattr(pkgutil, "iter_modules", fake_iter_modules)
    monkeypatch.setattr(importlib, "import_module", fake_import)

    assert pycaps_engine._load_pycaps_loader() is Loader


def test_load_pycaps_loader_raises_informative_error(monkeypatch):
    _clear_pycaps_modules(monkeypatch)

    package = types.ModuleType("pycaps")
    package.__path__ = []

    monkeypatch.setitem(sys.modules, "pycaps", package)

    with pytest.raises(RuntimeError) as excinfo:
        pycaps_engine._load_pycaps_loader()

    message = str(excinfo.value)
    assert "JsonConfigLoader" in message
    assert "pip install --no-cache-dir git+https://github.com/francozanardi/pycaps" in message


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
