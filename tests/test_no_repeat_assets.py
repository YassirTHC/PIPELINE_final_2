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


def test_dedupe_by_url_prevents_reuse():
    vp = _load_video_processor()
    vp.SEEN_URLS.clear()
    candidates = [
        SimpleNamespace(url="https://example.com/a.mp4"),
        SimpleNamespace(url="https://example.com/a.mp4"),
        SimpleNamespace(url="https://example.com/b.mp4"),
    ]
    unique, hits = vp.dedupe_by_url(candidates)
    assert len(unique) == 2
    assert hits == 1

    vp.SEEN_URLS.add("https://example.com/a.mp4")
    unique, hits = vp.dedupe_by_url([SimpleNamespace(url="https://example.com/a.mp4")])
    assert not unique
    assert hits == 1
