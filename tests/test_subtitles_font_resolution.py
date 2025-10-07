import logging
from pathlib import Path

import pytest

from hormozi_subtitles import HormoziSubtitles
from video_pipeline.config.settings import load_settings


@pytest.mark.usefixtures("reset_settings_cache")
def test_font_resolution_prefers_montserrat(monkeypatch, caplog):
    font_path = Path("assets/fonts/Montserrat-ExtraBold.ttf").resolve()
    monkeypatch.setenv("PIPELINE_SUBTITLE_FONT_PATH", str(font_path))

    settings = load_settings()

    with caplog.at_level(logging.INFO):
        subtitles = HormoziSubtitles(subtitle_settings=settings.subtitles)

    using_logs = [msg for msg in caplog.messages if "[Subtitles] Using font:" in msg]
    assert len(using_logs) == 1
    assert "montserrat" in using_logs[0].lower()

    joined = " ".join(caplog.messages).lower()
    assert "impact" not in joined

    resolved_path = subtitles.get_font_path()
    assert resolved_path is not None
    assert "montserrat" in Path(resolved_path).name.lower()
