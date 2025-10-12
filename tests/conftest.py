import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(1, str(SRC_DIR))


def _install_cv2_stub() -> None:
    """Install a minimal cv2 stub so tests can run without native OpenCV."""

    if "cv2" in sys.modules:
        return
    try:  # pragma: no cover - optional dependency
        import cv2  # type: ignore  # noqa: F401
        return
    except Exception:  # pragma: no cover - executed in CI without libGL/moviepy
        pass

    stub = types.ModuleType("cv2")
    stub.COLOR_RGB2BGR = 0
    stub.COLOR_BGR2RGB = 1
    stub.COLOR_BGR2GRAY = 2
    stub.COLOR_RGB2GRAY = 3
    stub.INTER_LANCZOS4 = 4
    stub.THRESH_BINARY = 0
    stub.CV_64F = 64.0
    stub.CAP_PROP_FPS = 5
    stub.CAP_PROP_FRAME_COUNT = 7
    stub.CAP_PROP_FRAME_WIDTH = 3
    stub.CAP_PROP_FRAME_HEIGHT = 4
    stub.CAP_PROP_POS_FRAMES = 1

    def _flip_channels(arr):
        array = np.asarray(arr)
        if array.ndim >= 3 and array.shape[-1] >= 3:
            return array[..., ::-1].copy()
        return array.copy()

    def cvtColor(arr, code):  # pragma: no cover - simple shim
        return _flip_channels(arr)

    stub.cvtColor = cvtColor

    def resize(arr, size, interpolation=None):  # pragma: no cover - shim
        array = np.asarray(arr)
        w, h = size
        channels = array.shape[2] if array.ndim == 3 else 1
        return np.zeros((h, w, channels), dtype=array.dtype)

    stub.resize = resize

    class _VideoCapture:  # pragma: no cover - shim
        def __init__(self, *args, **kwargs):
            self.opened = False

        def isOpened(self):
            return False

        def read(self):
            return False, None

        def release(self):
            return None

        def get(self, *args, **kwargs):
            return 0.0

        def set(self, *args, **kwargs):
            return False

    stub.VideoCapture = _VideoCapture

    class _VideoWriter:  # pragma: no cover - shim
        def __init__(self, *args, **kwargs):
            pass

        def write(self, *args, **kwargs):
            return None

        def release(self):
            return None

    stub.VideoWriter = _VideoWriter

    def VideoWriter_fourcc(*args, **kwargs):  # pragma: no cover - shim
        return 0

    stub.VideoWriter_fourcc = VideoWriter_fourcc

    def absdiff(a, b):  # pragma: no cover - shim
        return np.abs(np.asarray(a) - np.asarray(b))

    stub.absdiff = absdiff

    def Canny(*args, **kwargs):  # pragma: no cover - shim
        src = np.asarray(args[0]) if args else np.zeros((1, 1), dtype=np.uint8)
        return np.zeros_like(src, dtype=np.uint8)

    stub.Canny = Canny

    def findContours(*args, **kwargs):  # pragma: no cover - shim
        return [], None

    stub.findContours = findContours

    def contourArea(*args, **kwargs):  # pragma: no cover - shim
        return 0.0

    stub.contourArea = contourArea

    def moments(*args, **kwargs):  # pragma: no cover - shim
        return {}

    stub.moments = moments

    def Laplacian(arr, ddepth):  # pragma: no cover - shim
        return np.zeros_like(np.asarray(arr), dtype=np.float64)

    stub.Laplacian = Laplacian

    def threshold(src, thresh, maxval, type):  # pragma: no cover - shim
        arr = np.asarray(src)
        return thresh, np.zeros_like(arr, dtype=np.uint8)

    stub.threshold = threshold

    class _Cascade:  # pragma: no cover - shim
        def __init__(self, *args, **kwargs):
            pass

        def detectMultiScale(self, *args, **kwargs):
            return []

    stub.CascadeClassifier = _Cascade

    class _Data:  # pragma: no cover - shim
        haarcascades = ""

    stub.data = _Data()

    sys.modules["cv2"] = stub


def _install_moviepy_stub() -> None:
    """Install a light moviepy stub used during unit tests."""

    if "moviepy" in sys.modules:
        return
    try:  # pragma: no cover - optional dependency
        import moviepy.editor  # type: ignore  # noqa: F401
        return
    except Exception:  # pragma: no cover - executed in CI without moviepy
        pass

    moviepy = types.ModuleType("moviepy")
    editor = types.ModuleType("moviepy.editor")

    class _DummyClip:  # pragma: no cover - shim
        def __init__(self, *args, **kwargs):
            pass

        def write_videofile(self, *args, **kwargs):
            return None

        def set_audio(self, *args, **kwargs):
            return self

        def set_duration(self, *args, **kwargs):
            return self

        def set_start(self, *args, **kwargs):
            return self

        def fx(self, *args, **kwargs):
            return self

        def crossfadein(self, *args, **kwargs):
            return self

        def crossfadeout(self, *args, **kwargs):
            return self

        def set_position(self, *args, **kwargs):
            return self

        def resize(self, *args, **kwargs):
            return self

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

    editor.VideoFileClip = _DummyClip
    editor.AudioFileClip = _DummyClip
    editor.TextClip = _DummyClip
    editor.CompositeVideoClip = lambda clips, **kwargs: _DummyClip()
    editor.ColorClip = lambda size, color=(0, 0, 0): _DummyClip()

    moviepy.editor = editor

    video_module = types.ModuleType("moviepy.video")
    fx_module = types.ModuleType("moviepy.video.fx")
    fx_all_module = types.ModuleType("moviepy.video.fx.all")

    def _passthrough_clip(clip, *args, **kwargs):  # pragma: no cover - shim
        return clip

    fx_all_module.crop = _passthrough_clip

    fx_module.all = fx_all_module
    video_module.fx = fx_module
    moviepy.video = video_module

    sys.modules["moviepy"] = moviepy
    sys.modules["moviepy.editor"] = editor
    sys.modules["moviepy.video"] = video_module
    sys.modules["moviepy.video.fx"] = fx_module
    sys.modules["moviepy.video.fx.all"] = fx_all_module


_install_cv2_stub()
_install_moviepy_stub()

os.environ.setdefault("PIPELINE_FAST_TESTS", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


@pytest.fixture
def reset_settings_cache():
    from video_pipeline.config import settings as settings_module

    with settings_module._CACHE_LOCK:  # type: ignore[attr-defined]
        settings_module._SETTINGS_CACHE = None  # type: ignore[attr-defined]
    settings_module.reset_startup_log_for_tests()
    try:
        yield
    finally:
        with settings_module._CACHE_LOCK:  # type: ignore[attr-defined]
            settings_module._SETTINGS_CACHE = None  # type: ignore[attr-defined]
        settings_module.reset_startup_log_for_tests()


@pytest.fixture(autouse=True)
def _stub_segment_json(monkeypatch):
    import video_processor as vp

    monkeypatch.setattr(
        vp,
        "generate_segment_queries",
        lambda *_args, **_kwargs: [],
        raising=False,
    )
    yield


