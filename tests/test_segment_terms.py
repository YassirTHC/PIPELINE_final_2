import importlib
import sys
import types


def _stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.SimpleNamespace()
    sys.modules["cv2"].__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)

if "moviepy.editor" not in sys.modules:
    moviepy_editor = _stub_module(
        "moviepy.editor",
        VideoFileClip=object,
        TextClip=object,
        CompositeVideoClip=object,
    )
    moviepy_pkg = sys.modules.get("moviepy") or _stub_module("moviepy")
    moviepy_pkg.editor = moviepy_editor
    sys.modules["moviepy"] = moviepy_pkg


def test_segment_terms_from_briefs_picks_and_limits():
    from video_processor import _segment_terms_from_briefs
    dyn = {"segment_briefs": [
        {"segment_index": 0, "keywords": ["deep_work", "focus"], "queries": ["laptop focus", "office desk"]},
        {"segment_index": 1, "keywords": ["sleep_hygiene", "people"], "queries": ["dark bedroom", "nice background"]},
    ]}
    out0 = _segment_terms_from_briefs(dyn, 0, 3)
    assert out0 == ["laptop focus", "office desk", "deep work"]

    out1 = _segment_terms_from_briefs(dyn, 1, 4)
    assert out1 == ["dark bedroom", "nice background", "sleep hygiene", "people"]


def test_segment_terms_dedupes_and_orders_matches():
    from video_processor import _segment_terms_from_briefs

    dyn = {"segment_briefs": [
        {"segment_index": 2, "queries": ["alpha", "beta"], "keywords": ["gamma", "alpha"]},
        {"segment_index": 2, "queries": ["beta", "delta"], "keywords": ["epsilon"]},
        {"segment_index": 3, "queries": ["should ignore"], "keywords": ["also ignore"]},
    ]}

    out = _segment_terms_from_briefs(dyn, 2, 10)
    assert out == ["alpha", "beta", "delta", "gamma", "epsilon"]

