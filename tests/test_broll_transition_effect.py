# -*- coding: utf-8 -*-
import random
import sys
import types
from contextlib import ExitStack
from pathlib import Path

import pytest


def _install_pipeline_core_stubs():
    pkg = types.ModuleType("pipeline_core")
    sys.modules.setdefault("pipeline_core", pkg)

    configuration = types.ModuleType("pipeline_core.configuration")
    configuration.PipelineConfigBundle = object  # minimal placeholder
    sys.modules["pipeline_core.configuration"] = configuration

    fetchers = types.ModuleType("pipeline_core.fetchers")

    class _FetcherOrchestrator:
        def __init__(self, *_a, **_kw):
            pass

    fetchers.FetcherOrchestrator = _FetcherOrchestrator
    sys.modules["pipeline_core.fetchers"] = fetchers

    selection = types.ModuleType("pipeline_core.selection")

    def _mmr_rerank(*_a, **_kw):
        return []

    selection.mmr_rerank = _mmr_rerank
    sys.modules["pipeline_core.selection"] = selection

    dedupe = types.ModuleType("pipeline_core.dedupe")

    def _compute_phash(*_a, **_kw):
        return 0

    def _hamming_distance(*_a, **_kw):
        return 0

    dedupe.compute_phash = _compute_phash
    dedupe.hamming_distance = _hamming_distance
    sys.modules["pipeline_core.dedupe"] = dedupe

    logging_mod = types.ModuleType("pipeline_core.logging")

    class _JsonlLogger:
        def __init__(self, *_a, **_kw):
            pass

        def log(self, *_a, **_kw):
            pass

    def _log_broll_decision(*_a, **_kw):
        pass

    logging_mod.JsonlLogger = _JsonlLogger
    logging_mod.log_broll_decision = _log_broll_decision
    sys.modules["pipeline_core.logging"] = logging_mod

    llm_service = types.ModuleType("pipeline_core.llm_service")
    llm_service.generate_metadata_as_json = None
    llm_service.DynamicCompletionError = type("DynamicCompletionError", (Exception,), {})

    class _LLMMetadataGeneratorService:
        def __init__(self, *_a, **_kw):
            pass

    llm_service.LLMMetadataGeneratorService = _LLMMetadataGeneratorService
    llm_service._CONCRETE_SUBJECTS = ()

    def _identity_queries(*args, **kwargs):
        return []

    def _noop(*_a, **_kw):
        return None

    llm_service._concretize_queries = _identity_queries
    llm_service.build_visual_phrases = _identity_queries
    llm_service.enforce_fetch_language = _identity_queries
    llm_service.get_shared_llm_service = _noop
    llm_service.has_concrete_subject = lambda *_a, **_kw: False
    llm_service.generate_segment_queries = _identity_queries

    sys.modules["pipeline_core.llm_service"] = llm_service


_install_pipeline_core_stubs()

import video_processor as vp
from video_processor import TransitionConfig, VideoProcessor, apply_broll_transition_with_leak


class DummyClip:
    def __init__(self, *, w=1920, h=1080, duration=1.0):
        self.w = w
        self.h = h
        self.duration = duration
        self._start = 0.0
        self.position_calls = []
        self.resize_height = None
        self.opacity = None
        self.audio = None
        self.subclip_args = None
        self.fl_image_applied = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def without_audio(self):
        return self

    def set_audio(self, audio):
        self.audio = audio
        return self

    def set_position(self, position):
        self.position_calls.append(position)
        return self

    def set_start(self, value):
        self._start = value
        return self

    def set_duration(self, value):
        self.duration = value
        return self

    def resize(self, *, height=None):
        self.resize_height = height
        return self

    def subclip(self, start, end):
        clip = DummyClip(w=self.w, h=self.h, duration=max(0.0, end - start))
        clip.subclip_args = (start, end)
        return clip

    def set_opacity(self, value):
        self.opacity = value
        return self

    def fl_image(self, func):
        # We only need to know the effect was requested.
        self.fl_image_applied = True
        return self


class DummyAudioClip:
    def __init__(self):
        self.start = 0.0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def set_start(self, value):
        self.start = value
        return self

    def close(self):
        pass


class DummyCompositeAudio:
    def __init__(self, clips):
        self.clips = list(clips)

    def close(self):
        pass


@pytest.fixture(autouse=True)
def patch_moviepy(monkeypatch):
    monkeypatch.setattr(vp, "VideoFileClip", lambda *_a, **_kw: DummyClip(), raising=True)
    monkeypatch.setattr(vp, "AudioFileClip", lambda *_a, **_kw: DummyAudioClip(), raising=True)
    monkeypatch.setattr(vp, "CompositeAudioClip", lambda clips: DummyCompositeAudio(clips), raising=True)
    yield


def test_apply_transition_creates_leak_and_audio(tmp_path: Path):
    leak_path = tmp_path / "leak1.mp4"
    leak_path.write_bytes(b"0")
    shutter_path = tmp_path / "shutter.m4a"
    shutter_path.write_bytes(b"0")

    config = TransitionConfig(
        enabled=True,
        trans_dur=0.4,
        leak_in=0.15,
        leak_out=0.55,
        leak_opacity=0.7,
        shutter_offset=0.15,
        alternate_direction=True,
        light_leak_paths=[leak_path],
        shutter_path=shutter_path,
    )

    prev_clip = DummyClip()
    broll_clip = DummyClip()

    with ExitStack() as stack:
        artifacts = apply_broll_transition_with_leak(
            prev_clip,
            broll_clip,
            config,
            stack=stack,
            direction="right",
            leak_cycle=[],
            rng=random.Random(0),
        )

    assert artifacts.position_applied
    assert artifacts.leak_clip is not None
    assert artifacts.shutter_clip is not None
    assert artifacts.metadata.get("leak") == leak_path.name
    assert artifacts.metadata.get("audio") is True
    assert artifacts.metadata.get("status") == "whip"


def test_apply_transition_fallback_when_geometry_missing(tmp_path: Path):
    config = TransitionConfig(
        enabled=True,
        trans_dur=0.4,
        light_leak_paths=[tmp_path / "missing.mp4"],
        shutter_path=tmp_path / "missing.m4a",
    )
    prev_clip = DummyClip(w=0)
    broll_clip = DummyClip(w=0)

    with ExitStack() as stack:
        artifacts = apply_broll_transition_with_leak(
            prev_clip,
            broll_clip,
            config,
            stack=stack,
            direction="left",
            rng=random.Random(0),
        )

    assert not artifacts.position_applied
    assert artifacts.leak_clip is None
    assert artifacts.shutter_clip is None
    assert artifacts.metadata.get("status") == "skipped_geometry"


def test_resolve_transition_direction_alternates():
    processor = VideoProcessor.__new__(VideoProcessor)
    processor._transition_config = TransitionConfig(alternate_direction=True)
    processor._transition_direction_last = "left"

    first = VideoProcessor._resolve_transition_direction(processor)
    second = VideoProcessor._resolve_transition_direction(processor)

    assert first == "right"
    assert second == "left"

    processor._transition_config = TransitionConfig(alternate_direction=False)
    fixed = VideoProcessor._resolve_transition_direction(processor)
    assert fixed == "right"

