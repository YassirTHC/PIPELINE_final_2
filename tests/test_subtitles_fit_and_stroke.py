from pathlib import Path

import numpy as np
import pytest

from hormozi_subtitles import HormoziSubtitles
from video_pipeline.config.settings import load_settings


def _generate_long_words():
    return [
        {
            "animation_progress": 1.0,
            "tokens": [
                {"text": word, "color": "#FF6B00", "is_keyword": True if idx % 2 == 0 else False}
                for idx, word in enumerate(
                    [
                        "ULTRA",
                        "GROWTH",
                        "STRATEGY",
                        "DOMINATION",
                        "PLAYBOOK",
                        "FORMULA",
                    ]
                )
            ],
            "emojis": [],
        }
    ]


@pytest.mark.usefixtures("reset_settings_cache")
def test_subtitles_fit_and_stroke(monkeypatch):
    font_path = Path("assets/fonts/Montserrat-ExtraBold.ttf").resolve()
    monkeypatch.setenv("PIPELINE_SUBTITLE_FONT_PATH", str(font_path))

    settings = load_settings()
    settings.subtitles.font_size = 140
    settings.subtitles.keyword_background = False

    subtitles = HormoziSubtitles(subtitle_settings=settings.subtitles)

    width = 1280
    height = 720
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    words = _generate_long_words()

    output = subtitles.create_subtitle_frame(frame.copy(), words, current_time=0.0)

    nonzero = np.argwhere(output.sum(axis=2) > 0)
    assert nonzero.size > 0
    min_x = int(nonzero[:, 1].min())
    max_x = int(nonzero[:, 1].max())
    rendered_width = max_x - min_x + 1

    stroke_px = subtitles.config.get("stroke_px", 0)
    shadow_offset = subtitles.config.get("shadow_offset", 0)
    max_allowed = int(width * 0.92) + 2 * int(stroke_px) + int(shadow_offset)

    assert rendered_width <= max_allowed
    assert subtitles._last_render_metadata.get("stroke_px") == stroke_px

    dark_pixels = (
        (output[:, :, 0] < 25)
        & (output[:, :, 1] < 25)
        & (output[:, :, 2] < 25)
    ).sum()
    assert dark_pixels > 0
