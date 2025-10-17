"""Local kinetic caption renderer used by the PyCaps engine.

This module provides a small renderer inspired by the public PyCaps project in
order to keep the pipeline self contained.  It focuses on recreating the
Hormozi-style kinetic captions with sensible defaults while exposing enough
configuration hooks for the pipeline configuration loader to override colours,
fonts and layout behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Import MoviePy depuis l'API stable
try:
    from moviepy.editor import ColorClip, CompositeVideoClip, TextClip, VideoFileClip
except Exception:  # compat éventuelle avec environnements atypiques
    # Dégradé minimal : certains environnements n’exposent pas TextClip si ImageMagick n’est pas configuré.
    # On importe au moins ce dont on a besoin pour PyCaps; TextClip sera importé à l’usage si nécessaire.
    from moviepy.editor import ColorClip, CompositeVideoClip, VideoFileClip
    try:
        from moviepy.editor import TextClip  # lazy compat : ne crashe pas si indisponible
    except Exception:
        TextClip = None


logger = logging.getLogger(__name__)


EMPHASIS_WORDS = {
    "success",
    "winning",
    "profit",
    "growth",
    "viral",
    "breakthrough",
    "guaranteed",
    "secrets",
}


def _clean_color(value: str, default: str) -> str:
    text = (value or "").strip()
    if not text:
        return default
    if text.startswith("#") and len(text) in {4, 7, 9}:
        return text
    if re.match(r"^[0-9A-Fa-f]{6}$", text):
        return f"#{text}"
    return default


def _color_to_rgb(color: str) -> Tuple[int, int, int]:
    cleaned = _clean_color(color, "#000000")[1:]
    if len(cleaned) == 3:
        cleaned = "".join(ch * 2 for ch in cleaned)
    try:
        r = int(cleaned[0:2], 16)
        g = int(cleaned[2:4], 16)
        b = int(cleaned[4:6], 16)
    except ValueError:
        return (0, 0, 0)
    return (r, g, b)


_EMOJI_PATTERN = re.compile(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF]")


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _estimate_char_limit(font_size: float, ratio: float, video_width: int) -> int:
    if video_width <= 0:
        return max(8, int(font_size * 1.6))
    effective_width = max(32.0, video_width * _clamp(ratio, 0.2, 0.95))
    approx_char_width = max(1.0, font_size * 0.55)
    return max(8, int(effective_width / approx_char_width))


@dataclass(slots=True)
class CaptionStyle:
    """Kinetic caption styling hints consumed by :func:`render_subtitles_over_video`."""

    font: str = "Arial-Bold"
    fontsize: int = 74
    primary_color: str = "#FFFFFF"
    secondary_color: str = "#FFD15C"
    stroke_color: str = "#000000"
    stroke_width: int = 4
    shadow_color: str = "#000000"
    shadow_opacity: float = 0.45
    shadow_offset: Tuple[int, int] = (2, 2)
    background_color: str = "#000000"
    background_opacity: float = 0.35
    align: str = "center"
    margin_bottom_pct: float = 0.12
    max_lines: int = 3
    max_chars_per_line: int = 24
    uppercase_keywords: bool = True
    uppercase_min_length: int = 6
    highlight_scale: float = 1.08
    allow_emojis: bool = False
    line_height: float = 1.1
    max_line_width_ratio: float = 0.9
    box_padding_px: Optional[int] = None
    min_font_size: int = 24
    max_font_size: Optional[int] = None

    def with_overrides(self, **overrides: Any) -> "CaptionStyle":
        """Return a new style with the provided overrides applied."""

        return replace(self, **overrides)


THEME_PRESETS: Dict[str, CaptionStyle] = {
    "clean": CaptionStyle(
        font="Arial-Bold",
        fontsize=70,
        primary_color="#FFFFFF",
        secondary_color="#5B8CFF",
        stroke_color="#000000",
        stroke_width=3,
        shadow_opacity=0.25,
        background_opacity=0.18,
        margin_bottom_pct=0.10,
        max_chars_per_line=28,
    ),
    "bold": CaptionStyle(
        font="Montserrat-ExtraBold",
        fontsize=78,
        primary_color="#FFFFFF",
        secondary_color="#FF4F5E",
        stroke_color="#141414",
        stroke_width=5,
        shadow_opacity=0.55,
        background_opacity=0.32,
        margin_bottom_pct=0.11,
        max_chars_per_line=26,
    ),
    "hormozi": CaptionStyle(
        font="Montserrat-ExtraBold",
        fontsize=76,
        primary_color="#FFFFFF",
        secondary_color="#FBC531",
        stroke_color="#050505",
        stroke_width=6,
        shadow_opacity=0.5,
        background_opacity=0.38,
        margin_bottom_pct=0.12,
        max_chars_per_line=24,
    ),
}


def _responsive_base_style(style: CaptionStyle, video_size: Tuple[int, int]) -> CaptionStyle:
    width, height = video_size
    if height <= 0:
        return style

    min_font = max(20, getattr(style, "min_font_size", 24))
    max_font = getattr(style, "max_font_size", None)
    baseline = int(round(height * 0.035))
    if max_font is not None:
        target_font = int(_clamp(baseline, min_font, max_font))
    else:
        target_font = int(_clamp(baseline, min_font, 42))

    box_padding = int(round(target_font * 0.60))
    ratio = 0.78
    margin_bottom_pct = _clamp(0.06, 0.03, 0.18)
    char_limit = max(style.max_chars_per_line, _estimate_char_limit(target_font, ratio, width))

    return style.with_overrides(
        fontsize=target_font,
        max_lines=2,
        line_height=1.15,
        max_line_width_ratio=ratio,
        margin_bottom_pct=margin_bottom_pct,
        box_padding_px=box_padding,
        stroke_width=2,
        shadow_offset=(0, 2),
        shadow_opacity=0.65,
        background_opacity=0.35,
        max_chars_per_line=char_limit,
        min_font_size=min_font,
    )


@dataclass(slots=True)
class WordTiming:
    text: str
    start: float
    end: float
    display_text: str


def _strip_emojis(text: str) -> str:
    return _EMOJI_PATTERN.sub("", text)


def _normalise_word(word: str, style: CaptionStyle) -> str:
    token = word.strip()
    if not token:
        return ""
    if not style.allow_emojis:
        token = _strip_emojis(token)
    if style.uppercase_keywords:
        bare = re.sub(r"[^A-Za-z]", "", token)
        if len(bare) >= style.uppercase_min_length or bare.lower() in EMPHASIS_WORDS:
            token = token.upper()
    return token


def _ensure_duration(start: float, end: float) -> Tuple[float, float]:
    s = max(0.0, float(start))
    e = max(s + 0.04, float(end))
    return s, e


def _build_word_timings(
    segment: Mapping[str, Any],
    *,
    style: CaptionStyle,
) -> List[WordTiming]:
    start = float(segment.get("start", 0.0) or 0.0)
    end = float(segment.get("end", start) or start)
    text = str(segment.get("text", "") or "").strip()

    raw_words = segment.get("words")
    extracted: List[WordTiming] = []

    if isinstance(raw_words, Iterable):
        for item in raw_words:
            try:
                token = item.get("text") if isinstance(item, Mapping) else getattr(item, "text", None)
                token = str(token or "").strip()
            except Exception:
                token = ""
            cleaned = _normalise_word(token, style)
            if not cleaned:
                continue
            try:
                w_start = float(item.get("start", start)) if isinstance(item, Mapping) else float(getattr(item, "start", start))
                w_end = float(item.get("end", w_start)) if isinstance(item, Mapping) else float(getattr(item, "end", w_start))
            except Exception:
                w_start, w_end = start, end
            w_start, w_end = _ensure_duration(w_start, w_end)
            extracted.append(WordTiming(token, w_start, w_end, cleaned))

    if extracted:
        return extracted

    words = [w for w in text.split() if w.strip()]
    if not words:
        return []

    duration = max(end - start, 0.08)
    per_word = duration / len(words)
    timings: List[WordTiming] = []
    for index, token in enumerate(words):
        w_start = start + index * per_word
        w_end = w_start + per_word
        cleaned = _normalise_word(token, style)
        if cleaned:
            timings.append(WordTiming(token, w_start, w_end, cleaned))
    return timings


def _wrap_words(words: Sequence[WordTiming], style: CaptionStyle, video_width: int) -> List[List[WordTiming]]:
    if not words:
        return []

    max_lines = max(1, style.max_lines)
    ratio = max(0.2, min(0.95, getattr(style, "max_line_width_ratio", 0.9)))
    approx_char_width = max(1.0, style.fontsize * 0.55)
    dynamic_limit = int((video_width * ratio) / approx_char_width) if video_width > 0 else 0
    if dynamic_limit <= 0:
        dynamic_limit = style.max_chars_per_line
    max_chars = max(8, min(style.max_chars_per_line, dynamic_limit))

    lines: List[List[WordTiming]] = []
    current: List[WordTiming] = []
    current_len = 0

    for word in words:
        display = word.display_text
        length = len(display)
        projected = length if not current else current_len + 1 + length
        if current and projected > max_chars:
            lines.append(current)
            current = [word]
            current_len = length
            continue

        current.append(word)
        current_len = projected

    if current:
        lines.append(current)

    return lines


def _fit_segment_style(
    words: Sequence[WordTiming],
    base_style: CaptionStyle,
    video_width: int,
) -> Tuple[CaptionStyle, List[List[WordTiming]]]:
    ratio_candidates = []
    initial_ratio = getattr(base_style, "max_line_width_ratio", 0.78)
    ratio_candidates.append(_clamp(initial_ratio, 0.2, 0.9))
    for fallback_ratio in (0.72, 0.68):
        if fallback_ratio not in ratio_candidates:
            ratio_candidates.append(fallback_ratio)

    min_font = max(12, getattr(base_style, "min_font_size", 24))
    max_font = max(min_font, getattr(base_style, "fontsize", 64))

    for ratio in ratio_candidates:
        font_size = max_font
        while font_size >= min_font:
            char_limit = _estimate_char_limit(font_size, ratio, video_width)
            trial = base_style.with_overrides(
                fontsize=int(font_size),
                max_line_width_ratio=ratio,
                max_chars_per_line=max(base_style.max_chars_per_line, char_limit),
            )
            wrapped = _wrap_words(words, trial, video_width)
            if len(wrapped) <= trial.max_lines:
                return trial, wrapped
            font_size -= 2

    # Final fallback: clamp to minimum font and merge overflow into last line
    fallback_ratio = ratio_candidates[-1]
    fallback_font = int(min_font)
    fallback_char_limit = _estimate_char_limit(fallback_font, fallback_ratio, video_width)
    fallback_style = base_style.with_overrides(
        fontsize=fallback_font,
        max_line_width_ratio=fallback_ratio,
        max_chars_per_line=max(base_style.max_chars_per_line, fallback_char_limit),
    )
    wrapped_words = _wrap_words(words, fallback_style, video_width)
    max_lines = max(1, fallback_style.max_lines)
    if len(wrapped_words) > max_lines:
        head = list(wrapped_words[: max_lines - 1]) if max_lines > 1 else []
        tail_words: List[WordTiming] = []
        for line in wrapped_words[max_lines - 1 :]:
            tail_words.extend(line)
        if tail_words:
            if max_lines > 1:
                head.append(tail_words)
            else:
                head = [tail_words]
        wrapped_words = head
    return fallback_style, wrapped_words


def _text_clip(
    text: str,
    *,
    style: CaptionStyle,
    color: Optional[str] = None,
) -> TextClip:
    clip = TextClip(
        text or " ",
        font=style.font,
        fontsize=style.fontsize,
        color=color or style.primary_color,
        stroke_color=style.stroke_color,
        stroke_width=max(0, style.stroke_width),
        method="label",
    )
    return clip


def _space_width(style: CaptionStyle) -> float:
    try:
        clip = _text_clip(" ", style=style)
        width = float(clip.w or 0)
        clip.close()
        if width > 0:
            return width
    except Exception:
        logger.debug("[PyCapsRenderer] Unable to measure space width", exc_info=True)
    return style.fontsize * 0.33


def _compose_segment_clips(
    *,
    video_size: Tuple[int, int],
    words_by_line: Sequence[Sequence[WordTiming]],
    segment_start: float,
    segment_end: float,
    style: CaptionStyle,
) -> List[TextClip]:
    width, height = video_size
    duration = max(segment_end - segment_start, 0.08)
    line_spacing = style.fontsize * (style.line_height if style.line_height else 1.1)
    bottom_margin = max(0.02, min(0.3, style.margin_bottom_pct)) * height
    total_height = len(words_by_line) * line_spacing
    base_y = height - bottom_margin - total_height
    space_width = _space_width(style)
    max_line_width = width * _clamp(getattr(style, "max_line_width_ratio", 0.9), 0.2, 0.95)

    overlays: List[TextClip] = []

    for line_index, line_words in enumerate(words_by_line):
        line_text = " ".join(word.display_text for word in line_words)
        if not line_text:
            continue

        base_clip = _text_clip(line_text, style=style, color=style.primary_color)
        line_width = float(base_clip.w or width * 0.8)
        line_height = float(base_clip.h or style.fontsize)
        scale_factor = 1.0
        if max_line_width > 0 and line_width > max_line_width:
            scale_factor = max_line_width / line_width
            base_clip = base_clip.resize(scale_factor)
            line_width = float(base_clip.w or max_line_width)
            line_height = float(base_clip.h or style.fontsize * scale_factor)
        x_start = (width - line_width) / 2
        y_pos = base_y + line_index * line_spacing

        layer_clips: List[Any] = []

        if style.background_opacity > 0:
            padding = style.box_padding_px if style.box_padding_px is not None else None
            padding_w = (padding if padding is not None else max(20, style.fontsize * 0.4)) * scale_factor
            padding_h = (padding if padding is not None else max(12, style.fontsize * 0.25)) * scale_factor
            bg_size = (
                int(line_width + padding_w * 2),
                int(line_height + padding_h * 2),
            )
            bg_clip = ColorClip(bg_size, color=_color_to_rgb(style.background_color))
            bg_clip = bg_clip.set_start(segment_start).set_duration(duration)
            bg_clip = bg_clip.set_opacity(max(0.0, min(1.0, style.background_opacity)))
            bg_clip = bg_clip.set_position(
                (
                    (width - bg_size[0]) / 2,
                    y_pos - padding_h / 2,
                )
            )
            layer_clips.append(bg_clip)

        if style.shadow_opacity > 0:
            shadow_clip = _text_clip(line_text, style=style, color=style.shadow_color)
            shadow_clip = shadow_clip.set_start(segment_start).set_duration(duration)
            shadow_clip = shadow_clip.set_position(
                (
                    x_start + style.shadow_offset[0],
                    y_pos + style.shadow_offset[1],
                )
            )
            shadow_clip = shadow_clip.set_opacity(max(0.0, min(1.0, style.shadow_opacity)))
            layer_clips.append(shadow_clip)

        base_layer = base_clip.set_start(segment_start).set_duration(duration)
        base_layer = base_layer.set_position((x_start, y_pos))
        layer_clips.append(base_layer)

        overlays.extend(layer_clips)

        metrics: List[Tuple[WordTiming, float]] = []
        total_word_width = 0.0
        for word in line_words:
            metric_clip = _text_clip(word.display_text, style=style)
            width_value = float(metric_clip.w or 0.0) * scale_factor
            metrics.append((word, width_value))
            total_word_width += width_value
            metric_clip.close()

        spaced_gap = space_width * scale_factor
        total_word_width += spaced_gap * max(0, len(line_words) - 1)
        scale = 1.0
        if total_word_width > 0:
            scale = line_width / total_word_width

        cursor = x_start
        for idx, (word, measured_width) in enumerate(metrics):
            width_word = measured_width * scale if measured_width > 0 else style.fontsize * scale
            highlight_clip = _text_clip(word.display_text, style=style, color=style.secondary_color)
            highlight_duration = max(0.05, word.end - word.start)
            highlight_clip = highlight_clip.resize(style.highlight_scale * scale_factor)
            highlight_clip = highlight_clip.set_start(word.start).set_duration(highlight_duration)
            highlight_clip = highlight_clip.set_position(
                (
                    cursor - (highlight_clip.w - width_word) / 2,
                    y_pos - (highlight_clip.h - line_height) / 2,
                )
            )
            overlays.append(highlight_clip)

            cursor += width_word
            if idx < len(line_words) - 1:
                cursor += spaced_gap * scale

    return overlays


def render_subtitles_over_video(
    input_video: str | Path,
    segments: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    style: Optional[CaptionStyle] = None,
    **options: Any,
) -> str:
    """Render kinetic captions for ``segments`` and return the output path."""

    video_path = Path(input_video)
    output = Path(output_path)
    responsive_enabled = bool(options.pop("responsive", False))
    if style is None:
        theme = str(options.get("theme", "hormozi")).lower() or "hormozi"
        style = THEME_PRESETS.get(theme, THEME_PRESETS["hormozi"])

    style_overrides: Dict[str, Any] = {}
    for key in (
        "font",
        "fontsize",
        "primary_color",
        "secondary_color",
        "stroke_color",
        "stroke_width",
        "shadow_color",
        "shadow_opacity",
        "shadow_offset",
        "background_color",
        "background_opacity",
        "margin_bottom_pct",
        "max_lines",
        "max_chars_per_line",
        "uppercase_keywords",
        "uppercase_min_length",
        "highlight_scale",
        "allow_emojis",
        "line_height",
        "max_line_width_ratio",
        "box_padding_px",
        "min_font_size",
        "max_font_size",
    ):
        if key in options and options[key] is not None:
            style_overrides[key] = options[key]

    if style_overrides:
        style = style.with_overrides(**style_overrides)

    overlays: List[TextClip] = []
    try:
        video = VideoFileClip(str(video_path))
    except Exception as exc:  # pragma: no cover - defensive logging
        raise RuntimeError(f"Unable to open video for subtitle rendering: {exc}") from exc

    try:
        base_style = style
        if responsive_enabled:
            base_style = _responsive_base_style(style, video.size)

        video_width = int(video.size[0] if isinstance(video.size, tuple) and video.size else 0)

        for segment in segments:
            word_timings = _build_word_timings(segment, style=base_style)
            if not word_timings:
                continue
            seg_start, seg_end = _ensure_duration(
                segment.get("start", 0.0), segment.get("end", 0.0)
            )
            if responsive_enabled:
                segment_style, lines = _fit_segment_style(word_timings, base_style, video_width)
            else:
                segment_style = base_style
                lines = _wrap_words(word_timings, segment_style, video_width)
                if len(lines) > segment_style.max_lines:
                    max_lines = max(1, segment_style.max_lines)
                    merged_tail: List[WordTiming] = []
                    for line in lines[max_lines - 1 :]:
                        merged_tail.extend(line)
                    lines = lines[: max_lines - 1]
                    if merged_tail:
                        if max_lines > 1:
                            lines.append(merged_tail)
                        else:
                            lines = [merged_tail]
            overlays.extend(
                _compose_segment_clips(
                    video_size=video.size,
                    words_by_line=lines,
                    segment_start=seg_start,
                    segment_end=seg_end,
                    style=segment_style,
                )
            )

        if not overlays:
            logger.warning("[PyCapsRenderer] No subtitle overlays generated")
            video.write_videofile(
                str(output),
                codec=options.get("codec", "libx264"),
                audio_codec=options.get("audio_codec", "aac"),
                fps=video.fps or 30,
                threads=int(options.get("threads", 4)),
                preset=options.get("preset", "medium"),
                ffmpeg_params=["-movflags", "+faststart"],
                logger=None,
            )
            return str(output)

        composite = CompositeVideoClip([video] + overlays, size=video.size)
        composite.write_videofile(
            str(output),
            codec=options.get("codec", "libx264"),
            audio_codec=options.get("audio_codec", "aac"),
            fps=video.fps or 30,
            threads=int(options.get("threads", 4)),
            temp_audiofile=str(output.with_suffix(".temp-audio.m4a")),
            remove_temp=True,
            ffmpeg_params=["-movflags", "+faststart"],
            preset=options.get("preset", "medium"),
            logger=None,
        )
        composite.close()
    finally:
        for clip in overlays:
            try:
                clip.close()
            except Exception:
                pass
        video.close()

    return str(output)


__all__ = ["CaptionStyle", "render_subtitles_over_video", "THEME_PRESETS"]
