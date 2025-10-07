import numpy as np

from hormozi_subtitles import HormoziSubtitles


def _build_test_words():
    return [
        {
            "text": "MONEY MOVE",
            "start": 0.0,
            "end": 1.0,
            "animation_progress": 1.0,
            "tokens": [
                {
                    "text": "MONEY",
                    "normalized": "MONEY",
                    "is_keyword": True,
                    "color": "#FFD54F",
                },
                {
                    "text": "MOVE",
                    "normalized": "MOVE",
                    "is_keyword": False,
                    "color": "#FFFFFF",
                },
            ],
            "emojis": [],
        }
    ]


def test_keyword_background_enabled_sets_metadata():
    proc = HormoziSubtitles()
    proc.config['keyword_background'] = True
    proc.config['enable_emojis'] = False
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    words = _build_test_words()

    _ = proc.create_subtitle_frame(frame, words, current_time=0.0)
    metadata = getattr(proc, '_last_render_metadata', {})
    items = metadata.get('items', [])
    assert any(item.get('bg_rgb') for item in items if item.get('type') == 'word')


def test_keyword_background_disabled_has_no_rectangles():
    proc = HormoziSubtitles()
    proc.config['keyword_background'] = False
    proc.config['enable_emojis'] = False
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    words = _build_test_words()

    _ = proc.create_subtitle_frame(frame, words, current_time=0.0)
    metadata = getattr(proc, '_last_render_metadata', {})
    items = metadata.get('items', [])
    assert all(not item.get('bg_rgb') for item in items if item.get('type') == 'word')


def test_palette_covers_primary_categories():
    proc = HormoziSubtitles()
    required = ['finance', 'sales', 'content', 'mobile', 'sports']
    for category in required:
        assert category in proc.category_colors
        assert proc.category_colors[category] != '#FFFFFF'
