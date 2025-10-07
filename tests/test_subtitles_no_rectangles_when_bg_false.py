from unittest.mock import patch

import numpy as np
import pytest

from hormozi_subtitles import HormoziSubtitles
from video_pipeline.config.settings import load_settings


def _build_words():
    return [
        {
            "animation_progress": 1.0,
            "tokens": [
                {"text": "GROWTH", "color": "#FFAA00", "is_keyword": True},
                {"text": "BOOST", "color": "#FFFFFF", "is_keyword": False},
            ],
            "emojis": [],
        }
    ]


@pytest.mark.usefixtures("reset_settings_cache")
def test_no_rectangles_when_keyword_background_disabled(monkeypatch):
    monkeypatch.setenv("PIPELINE_SUBTITLE_KEYWORD_BACKGROUND", "0")
    settings = load_settings()
    settings.subtitles.keyword_background = False

    subtitles = HormoziSubtitles(subtitle_settings=settings.subtitles)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    words = _build_words()

    with patch("PIL.ImageDraw.ImageDraw.rounded_rectangle") as mock_rect:
        subtitles.create_subtitle_frame(frame, words, current_time=0.0)

    assert mock_rect.call_count == 0

    keyword_items = [
        item
        for item in subtitles._last_render_metadata.get("items", [])
        if item.get("type") == "word" and item.get("keyword")
    ]
    assert keyword_items, "Expected at least one keyword word item"
    for item in keyword_items:
        assert item.get("bg_rgb") is None
        assert item.get("rgb") == subtitles.hex_to_rgb("#FFAA00")
