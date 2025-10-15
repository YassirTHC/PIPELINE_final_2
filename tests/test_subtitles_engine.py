from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from moviepy import ColorClip

from subtitle_engines import pycaps_engine


def test_to_pycaps_input_normalises_segments():
    segments = [
        {
            "text": "Hello world",
            "start": 0.0,
            "end": 1.5,
            "words": [
                {"text": "Hello", "start": 0.0, "end": 0.6},
                {"text": "world", "start": 0.6, "end": 1.2},
            ],
        }
    ]

    payload = pycaps_engine.to_pycaps_input(segments)
    assert "segments" in payload
    assert len(payload["segments"]) == 1

    segment = payload["segments"][0]
    assert segment["text"] == "Hello world"
    assert segment["start"] == 0.0
    assert segment["end"] > segment["start"]
    assert len(segment["words"]) == 2
    assert segment["words"][0]["text"] == "Hello"
    assert segment["words"][0]["end"] > segment["words"][0]["start"]


def _dummy_settings() -> SimpleNamespace:
    subtitles = SimpleNamespace(
        font_path=None,
        engine="pycaps",
        font="Arial-Bold",
        font_size=64,
        theme="hormozi",
        primary_color="#FFFFFF",
        secondary_color="#FFAA33",
        stroke_color="#000000",
        subtitle_safe_margin_px=200,
        keyword_background=False,
        stroke_px=4,
        shadow_opacity=0.4,
        shadow_offset=2,
        shadow_color="#000000",
        background_color="#000000",
        background_opacity=0.25,
        margin_bottom_pct=0.12,
        max_lines=3,
        max_chars_per_line=26,
        uppercase_keywords=True,
        uppercase_min_length=5,
        highlight_scale=1.08,
        enable_emojis=False,
        emoji_target_per_10=5,
        emoji_min_gap_groups=2,
        emoji_max_per_segment=3,
        emoji_no_context_fallback="",
        hero_emoji_enable=True,
        hero_emoji_max_per_segment=1,
    )
    return SimpleNamespace(subtitles=subtitles)


def test_render_with_pycaps_creates_video(monkeypatch, tmp_path):
    input_path = tmp_path / "input.mp4"
    clip = ColorClip(size=(320, 568), color=(12, 34, 56), duration=1.6)
    clip.write_videofile(
        str(input_path),
        fps=24,
        codec="libx264",
        audio=False,
        logger=None,
    )
    clip.close()

    output_path = tmp_path / "output.mp4"
    template_dir = tmp_path / "templates"
    template_dir.mkdir()

    segments = [
        {
            "text": "Build momentum fast",
            "start": 0.2,
            "end": 1.0,
            "words": [
                {"text": "Build", "start": 0.2, "end": 0.45},
                {"text": "momentum", "start": 0.45, "end": 0.8},
                {"text": "fast", "start": 0.8, "end": 1.0},
            ],
        },
        {
            "text": "Consistency beats intensity every single time when you are launching.",
            "start": 1.0,
            "end": 1.55,
        },
    ]

    monkeypatch.setattr(pycaps_engine, "get_settings", lambda: _dummy_settings())

    recorded = {}

    def fake_render(input_video: str, segments, output_path: str, style, **options):
        Path(output_path).write_bytes(b"ok")
        recorded["segments"] = segments
        recorded["style"] = style
        recorded["options"] = options
        return output_path

    monkeypatch.setattr(pycaps_engine, "render_subtitles_over_video", fake_render)

    result_path = pycaps_engine.render_with_pycaps(
        segments,
        output_video_path=output_path,
        template_dir=template_dir,
        input_video_path=input_path,
    )

    assert Path(result_path).exists()
    assert Path(result_path).stat().st_size > 0
    assert recorded["segments"][0]["text"] == "Build momentum fast"
