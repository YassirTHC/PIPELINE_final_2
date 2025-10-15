"""Integration helpers for the local kinetic caption renderer."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .pycaps_renderer import CaptionStyle, THEME_PRESETS, render_subtitles_over_video

try:  # pragma: no cover - optional dependency during tests
    from video_pipeline.config import get_settings
except Exception:  # pragma: no cover - defensive import
    get_settings = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

_PYCAPS_TEMPLATE_FILENAME = "pycaps.template.json"


def ensure_template_assets(template_dir: str | Path) -> None:
    """Ensure the template directory contains the bundled fonts."""

    base_dir = Path(template_dir)
    resources_dir = base_dir / "resources"
    try:
        resources_dir.mkdir(parents=True, exist_ok=True)
    except Exception:  # pragma: no cover - filesystem best effort
        return

    repo_root = Path(__file__).resolve().parents[1]
    bundled_font = repo_root / "assets" / "fonts" / "Montserrat-ExtraBold.ttf"
    target_font = resources_dir / "Montserrat-ExtraBold.ttf"
    if bundled_font.exists() and not target_font.exists():
        try:
            shutil.copyfile(bundled_font, target_font)
        except Exception:  # pragma: no cover - filesystem best effort
            logger.debug("[PyCaps] Unable to copy bundled font", exc_info=True)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def to_pycaps_input(segments: Sequence[Mapping[str, Any]] | None) -> Dict[str, Any]:
    """Normalise the pipeline subtitle payload for the local renderer."""

    if not segments:
        return {"segments": []}

    normalised: list[dict[str, Any]] = []
    for segment in segments:
        start = _coerce_float(segment.get("start"), 0.0)
        end = _coerce_float(segment.get("end"), start)
        if end <= start:
            end = start + 0.08
        text = _coerce_text(segment.get("text"))

        words_payload: list[dict[str, Any]] = []
        words = segment.get("words")
        if isinstance(words, Sequence):
            for word in words:
                word_text = _coerce_text(getattr(word, "text", None) if not isinstance(word, Mapping) else word.get("text"))
                if not word_text:
                    continue
                word_start = _coerce_float(
                    getattr(word, "start", None) if not isinstance(word, Mapping) else word.get("start"),
                    start,
                )
                word_end = _coerce_float(
                    getattr(word, "end", None) if not isinstance(word, Mapping) else word.get("end"),
                    word_start,
                )
                if word_end <= word_start:
                    word_end = word_start + max(0.02, (end - start) / 8)
                words_payload.append({"text": word_text, "start": word_start, "end": word_end})

        normalised.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "words": words_payload,
            }
        )

    return {"segments": normalised}


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def _resolve_subtitle_settings() -> Any:
    if get_settings is None:  # pragma: no cover - optional dependency
        return None
    try:
        settings = get_settings()
    except Exception:  # pragma: no cover - defensive logging
        logger.debug("[PyCaps] Unable to load settings", exc_info=True)
        return None
    return getattr(settings, "subtitles", None)


def _style_from_settings(sub_settings: Any) -> tuple[CaptionStyle, Dict[str, Any]]:
    theme = "hormozi"
    overrides: Dict[str, Any] = {}

    if sub_settings is not None:
        theme = (_coerce_text(getattr(sub_settings, "theme", "")) or "hormozi").lower()
        overrides = {
            "font": getattr(sub_settings, "font_path", None) or getattr(sub_settings, "font", None) or "Arial-Bold",
            "fontsize": int(getattr(sub_settings, "font_size", 74) or 74),
            "primary_color": _coerce_text(getattr(sub_settings, "primary_color", "")) or "#FFFFFF",
            "secondary_color": _coerce_text(getattr(sub_settings, "secondary_color", "")) or "#FBC531",
            "stroke_color": _coerce_text(getattr(sub_settings, "stroke_color", "")) or "#000000",
            "stroke_width": int(getattr(sub_settings, "stroke_px", 4) or 4),
            "shadow_color": _coerce_text(getattr(sub_settings, "shadow_color", "")) or "#000000",
            "shadow_opacity": float(getattr(sub_settings, "shadow_opacity", 0.45) or 0.45),
            "background_color": _coerce_text(getattr(sub_settings, "background_color", "")) or "#000000",
            "background_opacity": float(getattr(sub_settings, "background_opacity", 0.35) or 0.35),
            "margin_bottom_pct": float(getattr(sub_settings, "margin_bottom_pct", 0.12) or 0.12),
            "max_lines": int(getattr(sub_settings, "max_lines", 3) or 3),
            "max_chars_per_line": int(getattr(sub_settings, "max_chars_per_line", 24) or 24),
            "uppercase_keywords": bool(getattr(sub_settings, "uppercase_keywords", True)),
            "uppercase_min_length": int(getattr(sub_settings, "uppercase_min_length", 6) or 6),
            "highlight_scale": float(getattr(sub_settings, "highlight_scale", 1.08) or 1.08),
        }
        allow_emojis = bool(getattr(sub_settings, "enable_emojis", False))
    else:
        allow_emojis = False

    preset = THEME_PRESETS.get(theme, THEME_PRESETS["hormozi"])
    style = preset.with_overrides(**overrides)
    allow_emojis = _env_bool("VP_SUBTITLES_EMOJIS", allow_emojis)
    style = style.with_overrides(allow_emojis=allow_emojis)

    options: Dict[str, Any] = {
        "theme": theme,
        "codec": "libx264",
        "audio_codec": "aac",
        "threads": int(os.getenv("VP_SUBTITLES_THREADS", "4") or 4),
    }
    return style, options


def render_with_pycaps(
    segments: Sequence[Mapping[str, Any]] | None,
    output_video_path: str | Path,
    template_dir: str | Path,
    *,
    input_video_path: str | Path,
) -> str:
    """Render subtitles using the bundled kinetic caption renderer."""

    template_base = Path(template_dir)
    template_base.mkdir(parents=True, exist_ok=True)
    ensure_template_assets(template_base)

    template_path = template_base / _PYCAPS_TEMPLATE_FILENAME
    if not template_path.exists():
        try:
            template_path.write_text(json.dumps({"version": 1}), encoding="utf-8")
        except Exception:  # pragma: no cover - best effort placeholder
            logger.debug("[PyCaps] Unable to create template placeholder", exc_info=True)

    payload = to_pycaps_input(segments)
    subtitle_segments = payload["segments"]

    subtitle_settings = _resolve_subtitle_settings()
    style, options = _style_from_settings(subtitle_settings)

    logger.info("[PyCaps] Using pycaps renderer (local)")
    try:
        return render_subtitles_over_video(
            input_video=str(input_video_path),
            segments=subtitle_segments,
            output_path=str(output_video_path),
            style=style,
            **options,
        )
    except Exception as exc:
        raise RuntimeError(f"Local PyCaps rendering failed: {exc}") from exc


__all__ = ["ensure_template_assets", "render_with_pycaps", "to_pycaps_input"]

