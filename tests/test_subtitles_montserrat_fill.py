import numpy as np

from hormozi_subtitles import HormoziSubtitles


def _dummy_frame() -> np.ndarray:
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def test_montserrat_fill_and_stroke_application():
    proc = HormoziSubtitles()

    transcription = [
        {
            "text": "profit energy offer",
            "start": 0.0,
            "end": 1.2,
            "words": [
                {"word": "profit", "start": 0.0, "end": 0.4},
                {"word": "energy", "start": 0.4, "end": 0.8},
                {"word": "offer", "start": 0.8, "end": 1.2},
            ],
        }
    ]

    groups = proc.parse_transcription_to_word_groups(transcription, group_size=2)
    assert groups, "expected at least one subtitle group"
    colored_tokens = [tok for tok in groups[0]["tokens"] if tok.get("is_keyword")]
    assert colored_tokens, "expected keywords to be detected"
    assert all(tok["color"] != "#FFFFFF" for tok in colored_tokens)
    assert proc.config["keyword_background"] is False

    font_path = proc.get_font_path()
    assert font_path is not None and "montserrat" in font_path.lower()

    active = [dict(groups[0], animation_progress=1.0)]
    frame = proc.create_subtitle_frame(_dummy_frame(), active, current_time=0.2)
    assert frame.shape == (720, 1280, 3)
    metadata = proc._last_render_metadata
    assert metadata["stroke_px"] == proc.config["stroke_px"]
    keyword_items = [item for item in metadata["items"] if item.get("keyword")]
    assert keyword_items, "render metadata should flag keyword items"
    assert all(item.get("bg_rgb") is None for item in keyword_items)
