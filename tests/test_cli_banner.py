from __future__ import annotations

import sys
import types
from pathlib import Path


def test_cli_warns_when_no_broll_inserted(monkeypatch, tmp_path, capsys):
    # Ensure optional runtime dependencies resolve during the CLI boot path.
    monkeypatch.setitem(sys.modules, "whisper", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "moviepy", types.SimpleNamespace())

    def _fake_add_hormozi_subtitles(src: str, subtitles, dst: str) -> None:  # pragma: no cover - trivial stub
        Path(dst).touch()

    monkeypatch.setitem(
        sys.modules,
        "hormozi_subtitles",
        types.SimpleNamespace(add_hormozi_subtitles=_fake_add_hormozi_subtitles),
    )

    import main as cli

    output_root = tmp_path / "output"

    class DummyConfig:
        CLIPS_FOLDER = tmp_path / "clips"
        OUTPUT_FOLDER = output_root
        TEMP_FOLDER = tmp_path / "temp"

    class DummyProcessor:
        def __init__(self):
            self._count = 0

        def reframe_to_vertical(self, video_path):
            return tmp_path / "reframed.mp4"

        def transcribe_segments(self, clip_path):
            return ["segment"]

        def generate_caption_and_hashtags(self, subtitles):
            return "title", "description", ["#tag"], ["kw1", "kw2"]

        def insert_brolls_if_enabled(self, clip_path, subtitles, keywords):
            return tmp_path / "with_broll.mp4"

        def get_last_broll_insert_count(self):
            return self._count

    def _fake_banner(count, *, origin="pipeline"):
        if int(count) > 0:
            return True, f"    ✅ B-roll insérés avec succès ({int(count)})"
        if origin == "pipeline_core":
            return False, "    ⚠️ Pipeline core: aucun B-roll sélectionné; retour à la vidéo d'origine"
        return False, "    ⚠️ Aucun B-roll inséré; retour à la vidéo d'origine"

    monkeypatch.setitem(
        sys.modules,
        "video_processor",
        types.SimpleNamespace(
            VideoProcessor=DummyProcessor,
            Config=DummyConfig,
            format_broll_completion_banner=_fake_banner,
        ),
    )

    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"test")

    monkeypatch.setattr(sys, "argv", ["main.py", "--cli", "--video", str(video_path)])

    cli.main()

    captured = capsys.readouterr().out

    assert "⚠️ Pipeline core: aucun B-roll sélectionné; retour à la vidéo d'origine" in captured
    assert "B-roll insérés avec succès" not in captured
