import os

from hormozi_subtitles import HormoziSubtitles


def test_montserrat_font_resolves_and_logs(caplog):
    caplog.set_level("INFO")
    proc = HormoziSubtitles()
    font_path = proc.get_font_path()
    assert font_path is not None
    assert "Montserrat" in os.path.basename(font_path)
    joined = "\n".join(record.getMessage() for record in caplog.records)
    assert "Impact" not in joined
    assert "Montserrat" in joined


def test_font_fallback_to_packaged_montserrat(tmp_path):
    missing = tmp_path / "missing.ttf"
    proc = HormoziSubtitles(font_candidates=[str(missing)])
    font_path = proc.get_font_path()
    assert font_path is not None
    assert "Montserrat" in os.path.basename(font_path)
