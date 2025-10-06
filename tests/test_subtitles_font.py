import os

from PIL import ImageFont

from hormozi_subtitles import HormoziSubtitles


def test_montserrat_font_resolves():
    proc = HormoziSubtitles()
    font_path = proc._resolve_font_path()
    assert font_path is not None
    assert "Montserrat" in os.path.basename(font_path)


def test_font_fallback_to_default(tmp_path):
    missing = tmp_path / "missing.ttf"
    proc = HormoziSubtitles(font_candidates=[str(missing)])
    proc._font_candidates = [str(missing)]
    proc._font_primary = None
    assert proc._resolve_font_path() is None
    font = proc._load_font(42)
    assert isinstance(font, (ImageFont.ImageFont, ImageFont.FreeTypeFont))
