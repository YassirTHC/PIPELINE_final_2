from types import SimpleNamespace, ModuleType
import sys
import importlib
import importlib.machinery as machinery


def _load_video_processor():
    if 'video_processor' in sys.modules:
        return importlib.reload(sys.modules['video_processor'])

    stubs = {
        'temp_function': SimpleNamespace(_llm_generate_caption_hashtags_fixed=lambda *args, **kwargs: ""),
        'whisper': SimpleNamespace(load_model=lambda *args, **kwargs: SimpleNamespace(transcribe=lambda *a, **k: {})),
    }

    for name, module in stubs.items():
        sys.modules.setdefault(name, module)

    dotenv_module = ModuleType('dotenv')
    dotenv_module.load_dotenv = lambda *args, **kwargs: None
    sys.modules.setdefault('dotenv', dotenv_module)

    cv2_module = ModuleType('cv2')
    cv2_module.__spec__ = machinery.ModuleSpec('cv2', loader=None)
    sys.modules['cv2'] = cv2_module

    moviepy_module = ModuleType('moviepy')
    editor_module = ModuleType('moviepy.editor')

    class _FakeClip:
        def __init__(self, *args, **kwargs):
            self.duration = 0.0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write_videofile(self, *args, **kwargs):
            return None

    class _FakeComposite(_FakeClip):
        pass

    editor_module.VideoFileClip = _FakeClip
    editor_module.TextClip = _FakeClip
    editor_module.CompositeVideoClip = _FakeComposite
    moviepy_module.editor = editor_module
    sys.modules['moviepy'] = moviepy_module
    sys.modules['moviepy.editor'] = editor_module
    moviepy_config = ModuleType('moviepy.config')
    moviepy_config.IMAGEMAGICK_BINARY = None
    sys.modules['moviepy.config'] = moviepy_config

    broll_module = ModuleType('broll_selector')

    class _Dummy:
        def __init__(self, *args, **kwargs):
            pass

    broll_module.BrollSelector = _Dummy
    broll_module.Asset = _Dummy
    broll_module.ScoringFeatures = _Dummy
    broll_module.BrollCandidate = _Dummy
    sys.modules['broll_selector'] = broll_module

    return importlib.import_module('video_processor')


def _make_item(start, end):
    return SimpleNamespace(start=start, end=end)


def test_enforce_broll_schedule_rules_filters_short_and_gap():
    vp = _load_video_processor()
    plan = [
        _make_item(0.0, 2.0),      # duration ok
        _make_item(2.2, 4.0),      # duration ok but gap 0.2s
        _make_item(4.5, 10.5),     # duration capped
        _make_item(12.0, 12.8),    # duration too short
    ]
    filtered, drops = vp.enforce_broll_schedule_rules(plan)
    assert len(filtered) == 2
    reasons = {drop['reason'] for drop in drops}
    assert 'duration_short' in reasons
    assert 'gap_violation' in reasons
