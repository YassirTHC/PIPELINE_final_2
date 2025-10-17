"""Integration helpers for the PyCaps-based subtitle renderer."""

from __future__ import annotations

import html
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
_VENDOR_CANDIDATES = [
    _REPO_ROOT / "vendor" / "pycaps",
    _REPO_ROOT / "vendor" / "pycaps_new" / "src",
]
for _candidate in _VENDOR_CANDIDATES:
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

try:  # pragma: no cover - optional dependency checked at runtime
    from pycaps import __version__ as _PYCAPS_VERSION  # type: ignore[attr-defined]
    from pycaps.common import CacheStrategy, Document, Line, Segment, TimeFragment, Word
    from pycaps.pipeline import JsonConfigLoader
    from pycaps.renderer import PictexSubtitleRenderer
except Exception:  # pragma: no cover - gracefully handle missing dependency
    _PYCAPS_VERSION = None
    CacheStrategy = Document = Line = Segment = TimeFragment = Word = None  # type: ignore[assignment]
    JsonConfigLoader = PictexSubtitleRenderer = None  # type: ignore[assignment]

try:  # pragma: no cover - optional config module
    from video_pipeline.config import get_settings
except Exception:  # pragma: no cover - defensive import
    get_settings = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

from .pycaps_renderer import CaptionStyle, THEME_PRESETS, render_subtitles_over_video  # noqa: E402

_PYCAPS_TEMPLATE_FILENAME = "pycaps.template.json"
_PYCAPS_CSS_FILENAME = "pycaps.css"
_PYCAPS_RESOURCES_DIRNAME = "resources"

_FALLBACK_TEMPLATE_JSON = """{
    "css": "pycaps.css",
    "resources": "resources",
    "layout": {
        "max_width_ratio": 0.88,
        "max_number_of_lines": 2,
        "min_number_of_lines": 1,
        "vertical_align": {
            "align": "bottom",
            "offset": -0.12
        }
    },
    "splitters": [
        {
            "type": "split_into_sentences"
        },
        {
            "type": "limit_by_chars",
            "min_chars": 14,
            "max_chars": 24,
            "avoid_finishing_segment_with_word_shorter_than": 4
        }
    ],
    "effects": [
        {
            "type": "remove_punctuation_marks",
            "punctuation_marks": [
                ".",
                "?",
                "!"
            ],
            "exception_marks": [
                "...",
                "?!"
            ]
        },
        {
            "type": "typewriting",
            "tag_condition": ""
        }
    ],
    "animations": [
        {
            "type": "zoom_in_primitive",
            "when": "narration-starts",
            "what": "word",
            "duration": 0.18,
            "delay": 0.0,
            "overshoot": {
                "amount": 0.06,
                "peak_at": 0.65
            }
        },
        {
            "type": "fade_in",
            "when": "narration-starts",
            "what": "segment",
            "duration": 0.16,
            "delay": 0.0
        },
        {
            "type": "fade_out",
            "when": "narration-ends",
            "what": "segment",
            "duration": 0.24,
            "delay": 0.0
        }
    ],
    "cache_strategy": "css-classes-aware"
}
"""

_FALLBACK_CSS = """@font-face {
    font-family: 'Montserrat ExtraBold';
    src: url('Montserrat-ExtraBold.ttf') format('truetype');
    font-style: normal;
    font-weight: 800;
}

@font-face {
    font-family: 'Montserrat Bold';
    src: url('Montserrat-Bold.ttf') format('truetype');
    font-style: normal;
    font-weight: 700;
}

:root {
    --pycaps-primary: #ffffff;
    --pycaps-accent: #f7aa2d;
    --pycaps-shadow: rgba(0, 0, 0, 0.82);
    --pycaps-muted: rgba(210, 220, 240, 0.82);
    --pycaps-line-bg: linear-gradient(135deg, rgba(14, 14, 19, 0.92) 0%, rgba(31, 32, 45, 0.86) 100%);
    --pycaps-line-border: rgba(255, 255, 255, 0.18);
    --pycaps-line-glow: rgba(255, 193, 90, 0.35);
}

.line {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 16px;
    padding: 20px 36px;
    border-radius: 36px;
    background: var(--pycaps-line-bg);
    border: 2px solid var(--pycaps-line-border);
    box-shadow:
        0 28px 60px rgba(9, 8, 15, 0.55),
        0 12px 30px rgba(0, 0, 0, 0.35);
    backdrop-filter: blur(5px);
    transform: translateZ(0);
}

.line::before {
    content: '';
    position: absolute;
    inset: -12px;
    border-radius: 44px;
    background: radial-gradient(circle at 70% 20%, rgba(255, 210, 120, 0.32), transparent 65%);
    opacity: 0;
    transition: opacity 0.35s ease;
    pointer-events: none;
}

.line-being-narrated::before {
    opacity: 1;
}

.line-not-narrated-yet {
    opacity: 0.88;
}

.line-already-narrated {
    opacity: 0.94;
}

.word {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-family: 'Montserrat ExtraBold', 'Montserrat Bold', sans-serif;
    font-size: 64px;
    line-height: 1.06;
    letter-spacing: 0.35px;
    color: var(--pycaps-muted);
    padding: 4px 12px;
    transition:
        transform 0.28s cubic-bezier(0.34, 1.56, 0.64, 1),
        color 0.24s ease-out,
        text-shadow 0.24s ease-out,
        opacity 0.24s ease-out;
}

.word + .word {
    margin-left: 10px;
}

.word::after {
    content: '';
    position: absolute;
    inset: -10px;
    border-radius: 26px;
    opacity: 0;
    background: radial-gradient(circle at center, rgba(255, 195, 90, 0.38), rgba(255, 195, 90, 0));
    transition: opacity 0.24s ease;
    z-index: -1;
}

.word span {
    display: block;
    transform-origin: center;
}

.word-not-narrated-yet {
    opacity: 0.58;
}

.word-being-narrated {
    color: var(--pycaps-primary);
    transform: scale(1.12);
    text-shadow:
        0 4px 28px rgba(247, 170, 45, 0.78),
        0 2px 16px rgba(255, 206, 120, 0.96),
        0 0 3px rgba(255, 255, 255, 0.7);
}

.word-being-narrated::after {
    opacity: 1;
}

.word-already-narrated {
    color: var(--pycaps-primary);
    opacity: 0.86;
    text-shadow:
        0 2px 18px rgba(0, 0, 0, 0.52),
        0 0 3px rgba(255, 255, 255, 0.4);
}
"""


def _render_pycaps_debug_html(payload: Mapping[str, Any]) -> str:
    segments = payload.get("segments") if isinstance(payload, Mapping) else None
    segment_blocks: list[str] = []
    for index, segment in enumerate(segments or []):
        line_blocks: list[str] = []
        for line in segment.get("lines", []):
            word_spans: list[str] = []
            for word in line.get("words", []):
                text = html.escape(str(word.get("text", "")))
                word_spans.append(f'<span class="word word-not-narrated-yet">{text}</span>')
            line_blocks.append(f'<div class="line line-not-narrated-yet">{" ".join(word_spans)}</div>')
        segment_blocks.append(
            f'<section class="segment" data-index="{index}">{"".join(line_blocks)}</section>'
        )

    body = "".join(segment_blocks)
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"/>"
        "<title>PyCaps Debug Preview</title>"
        "<link rel=\"stylesheet\" href=\"pycaps.css\" />"
        "<style>"
        "body{background:#101020;color:#fff;font-family:'Montserrat',Arial,sans-serif;padding:24px;}"
        ".segment{margin-bottom:24px;} .line{margin-bottom:6px;} .word{display:inline-block;margin-right:8px;}"
        "</style></head><body>"
        + body
        + "</body></html>"
    )


def _export_subtitle_debug_assets(output_video_path: Path, template_dir: Path) -> None:
    json_path = Path(output_video_path).with_suffix(".json")
    if not json_path.exists():
        return

    debug_root = Path("temp") / "pycaps_debug"
    try:
        debug_root.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.debug("[PyCaps] Unable to create debug directory", exc_info=True)
        return

    try:
        shutil.copy2(json_path, debug_root / json_path.name)
    except Exception:
        logger.debug("[PyCaps] Unable to copy subtitle JSON", exc_info=True)

    css_source = template_dir / _PYCAPS_CSS_FILENAME
    css_target = debug_root / "pycaps.css"
    if css_source.exists():
        try:
            shutil.copy2(css_source, css_target)
        except Exception:
            logger.debug("[PyCaps] Unable to copy debug CSS", exc_info=True)

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("[PyCaps] Unable to parse subtitle JSON", exc_info=True)
        return

    html_path = debug_root / f"{json_path.stem}.html"
    try:
        html_path.write_text(_render_pycaps_debug_html(payload), encoding="utf-8")
    except Exception:
        logger.debug("[PyCaps] Unable to export debug HTML", exc_info=True)


_EMOJI_PATTERN = re.compile(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF]")


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


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


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except Exception:
        return default


def _build_local_renderer_style(allow_emojis: bool) -> Tuple[CaptionStyle | None, Dict[str, Any]]:
    """Construct a :class:`CaptionStyle` instance and render options for the fallback renderer."""

    style: CaptionStyle | None = None
    renderer_options: Dict[str, Any] = {}
    base_theme = "hormozi"
    style_overrides: Dict[str, Any] = {"allow_emojis": allow_emojis}

    subtitles = None
    if get_settings is not None:
        try:
            settings = get_settings()
            subtitles = getattr(settings, "subtitles", None)
        except Exception:
            logger.debug("[PyCaps] Unable to load subtitles settings for fallback renderer", exc_info=True)

    if subtitles is not None:
        theme_value = getattr(subtitles, "theme", None)
        if isinstance(theme_value, str) and theme_value.strip():
            base_theme = theme_value.strip().lower()

        # Prefer explicit font name, fall back to path if provided.
        font_override = getattr(subtitles, "font", None)
        font_path = getattr(subtitles, "font_path", None)
        if isinstance(font_override, str) and font_override.strip():
            style_overrides["font"] = font_override.strip()
        elif isinstance(font_path, str) and font_path.strip():
            style_overrides["font"] = font_path.strip()

        attr_mapping = {
            "font_size": "fontsize",
            "primary_color": "primary_color",
            "secondary_color": "secondary_color",
            "stroke_color": "stroke_color",
            "shadow_color": "shadow_color",
            "shadow_opacity": "shadow_opacity",
            "background_color": "background_color",
            "background_opacity": "background_opacity",
            "margin_bottom_pct": "margin_bottom_pct",
            "max_lines": "max_lines",
            "max_chars_per_line": "max_chars_per_line",
            "uppercase_keywords": "uppercase_keywords",
            "uppercase_min_length": "uppercase_min_length",
            "highlight_scale": "highlight_scale",
        }
        for attr, target in attr_mapping.items():
            value = getattr(subtitles, attr, None)
            if value is not None:
                style_overrides[target] = value

        stroke_px = getattr(subtitles, "stroke_px", None)
        if stroke_px is not None:
            style_overrides["stroke_width"] = stroke_px

        shadow_offset = getattr(subtitles, "shadow_offset", None)
        if isinstance(shadow_offset, (int, float)):
            style_overrides["shadow_offset"] = (int(shadow_offset), int(shadow_offset))
        elif isinstance(shadow_offset, (list, tuple)) and len(shadow_offset) == 2:
            try:
                style_overrides["shadow_offset"] = (int(shadow_offset[0]), int(shadow_offset[1]))
            except Exception:
                pass

    base_style = THEME_PRESETS.get(base_theme, THEME_PRESETS["hormozi"])
    filtered_overrides = {key: value for key, value in style_overrides.items() if value is not None}
    style = base_style.with_overrides(**filtered_overrides)

    renderer_options["threads"] = max(1, _env_int("VP_SUBTITLES_THREADS", 4))

    return style, renderer_options


def _render_with_local_renderer(
    *,
    normalised_payload: Mapping[str, Any],
    allow_emojis: bool,
    input_video_path: str | Path,
    output_path: Path,
    template_dir: Path,
    responsive: bool,
) -> str:
    """Render subtitles via the built-in renderer and export debug artefacts."""

    style, renderer_options = _build_local_renderer_style(allow_emojis)
    if responsive:
        renderer_options["responsive"] = True
    rendered_path = Path(
        render_subtitles_over_video(
            input_video=str(input_video_path),
            segments=normalised_payload["segments"],
            output_path=str(output_path),
            style=style,
            **renderer_options,
        )
    )
    try:
        debug_payload_path = rendered_path.with_suffix(".json")
        debug_payload_path.write_text(
            json.dumps(normalised_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        logger.debug("[PyCaps] Unable to persist fallback subtitle JSON", exc_info=True)
    try:
        _export_subtitle_debug_assets(rendered_path, template_dir)
    except Exception:
        logger.debug("[PyCaps] Unable to export fallback debug artefacts", exc_info=True)

    logger.info("[PyCaps] Rendered subtitles using local fallback renderer")
    return str(rendered_path)


def ensure_template_assets(template_dir: str | Path) -> None:
    """Ensure the PyCaps template directory contains required assets."""

    base_dir = Path(template_dir)
    resources_dir = base_dir / _PYCAPS_RESOURCES_DIRNAME
    try:
        resources_dir.mkdir(parents=True, exist_ok=True)
    except Exception:  # pragma: no cover - filesystem best effort
        return

    fonts_dir = _REPO_ROOT / "assets" / "fonts"
    for font_name in ("Montserrat-ExtraBold.ttf", "Montserrat-Bold.ttf"):
        bundled_font = fonts_dir / font_name
        target_font = resources_dir / font_name
        if bundled_font.exists() and not target_font.exists():
            try:
                shutil.copyfile(bundled_font, target_font)
            except Exception:  # pragma: no cover - best effort on Windows locks
                logger.debug("[PyCaps] Unable to copy bundled font %s", font_name, exc_info=True)

    css_path = base_dir / _PYCAPS_CSS_FILENAME
    if not css_path.exists():
        css_path.write_text(_FALLBACK_CSS, encoding="utf-8")

    template_path = base_dir / _PYCAPS_TEMPLATE_FILENAME
    if not template_path.exists():
        template_path.write_text(_FALLBACK_TEMPLATE_JSON, encoding="utf-8")


def _prepare_runtime_template(source_template_dir: Path) -> Path:
    """Copy template assets into a writable runtime directory."""

    source_template_dir = source_template_dir.resolve()
    if not source_template_dir.exists():
        raise FileNotFoundError(f"PyCaps template directory missing: {source_template_dir}")

    runtime_dir = Path("temp") / "pycaps_template"
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir, ignore_errors=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    template_src = source_template_dir / _PYCAPS_TEMPLATE_FILENAME
    css_src = source_template_dir / _PYCAPS_CSS_FILENAME
    if not template_src.exists():
        raise FileNotFoundError(f"Missing PyCaps template: {template_src}")
    if not css_src.exists():
        raise FileNotFoundError(f"Missing PyCaps CSS: {css_src}")

    shutil.copy2(template_src, runtime_dir / _PYCAPS_TEMPLATE_FILENAME)
    shutil.copy2(css_src, runtime_dir / _PYCAPS_CSS_FILENAME)

    resources_src = source_template_dir / _PYCAPS_RESOURCES_DIRNAME
    resources_dst = runtime_dir / _PYCAPS_RESOURCES_DIRNAME
    if resources_src.exists():
        shutil.copytree(resources_src, resources_dst, dirs_exist_ok=True)
    else:
        resources_dst.mkdir(parents=True, exist_ok=True)

    ensure_template_assets(runtime_dir)
    return runtime_dir


def to_pycaps_input(segments: Sequence[Mapping[str, Any]] | None) -> Dict[str, Any]:
    """Normalise the pipeline subtitle payload for the PyCaps renderer."""

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
        if isinstance(words, Sequence) and not isinstance(words, (str, bytes)):
            for word in words:
                if isinstance(word, Mapping):
                    raw_text = word.get("text")
                    raw_start = word.get("start")
                    raw_end = word.get("end")
                else:
                    raw_text = getattr(word, "text", None)
                    raw_start = getattr(word, "start", None)
                    raw_end = getattr(word, "end", None)

                word_text = _coerce_text(raw_text)
                if not word_text:
                    continue
                word_start = _coerce_float(raw_start, start)
                word_end = _coerce_float(raw_end, word_start)
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


def _strip_emojis(text: str) -> str:
    return _EMOJI_PATTERN.sub("", text)


def _fabricate_words(text: str, start: float, end: float) -> list[dict[str, Any]]:
    clean_text = _coerce_text(text)
    tokens = [token for token in clean_text.split() if token.strip()]
    if not tokens:
        return []

    duration = max(end - start, 0.08)
    step = duration / max(len(tokens), 1)
    cursor = start
    generated: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        token_start = cursor
        if index == len(tokens) - 1:
            token_end = end
        else:
            token_end = cursor + max(step, 0.05)
        generated.append({"text": token, "start": token_start, "end": token_end})
        cursor = token_end
    return generated


def _payload_to_document(payload: Sequence[Mapping[str, Any]], *, allow_emojis: bool) -> Document:
    if Document is None:
        raise RuntimeError("PyCaps is not installed. Install the GitHub version to enable this renderer.")

    document = Document()
    for segment_payload in payload:
        seg_start = _coerce_float(segment_payload.get("start"), 0.0)
        seg_end = _coerce_float(segment_payload.get("end"), seg_start + 0.08)
        if seg_end <= seg_start:
            seg_end = seg_start + 0.08

        segment_time = TimeFragment(start=seg_start, end=seg_end)
        segment = Segment(time=segment_time)
        line = Line(time=segment_time)
        segment.lines.add(line)

        raw_words = segment_payload.get("words") or []
        if not raw_words:
            raw_words = _fabricate_words(segment_payload.get("text", ""), seg_start, seg_end)

        inserted = 0
        total_words = max(len(raw_words), 1)
        for word_data in raw_words:
            if not isinstance(word_data, Mapping):
                # Already normalised data structure, guard anyway
                continue
            text = _coerce_text(word_data.get("text"))
            if not allow_emojis:
                text = _strip_emojis(text)
            if not text:
                continue
            word_start = _coerce_float(word_data.get("start"), seg_start)
            word_end = _coerce_float(word_data.get("end"), word_start)
            if word_end <= word_start:
                word_end = word_start + max(0.02, (seg_end - seg_start) / total_words)
            word_time = TimeFragment(start=word_start, end=word_end)
            line.words.add(Word(text=text, time=word_time))
            inserted += 1

        if inserted == 0:
            # Drop segments that resolved to no words to avoid renderer crashes.
            continue

        document.segments.add(segment)

    return document


def _should_allow_emojis(default: bool = False) -> bool:
    allow = default
    if get_settings is not None:
        try:
            settings = get_settings()
            subtitles = getattr(settings, "subtitles", None)
            if subtitles is not None:
                allow = bool(getattr(subtitles, "enable_emojis", allow))
        except Exception:  # pragma: no cover - configuration is optional
            logger.debug("[PyCaps] Unable to load subtitles settings", exc_info=True)
    return _env_bool("VP_SUBTITLES_EMOJIS", allow)


def _should_use_responsive_style(default: bool = False) -> bool:
    enabled = default
    if get_settings is not None:
        try:
            settings = get_settings()
            subtitles = getattr(settings, "subtitles", None)
            if subtitles is not None:
                enabled = bool(getattr(subtitles, "responsive_mode", enabled))
        except Exception:  # pragma: no cover - configuration is optional
            logger.debug("[PyCaps] Unable to inspect responsive subtitle flag", exc_info=True)
    return _env_bool("VP_SUBTITLES_RESPONSIVE", enabled)


def render_with_pycaps(
    segments: Sequence[Mapping[str, Any]] | None,
    output_video_path: str | Path,
    template_dir: str | Path,
    *,
    input_video_path: str | Path,
) -> str:
    """Render subtitles using the installed PyCaps renderer or the local fallback."""

    normalised = to_pycaps_input(segments)
    allow_emojis = _should_allow_emojis(default=False)
    responsive_enabled = _should_use_responsive_style(default=False)

    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template_path = Path(template_dir)

    if JsonConfigLoader is not None and PictexSubtitleRenderer is not None and Document is not None:
        try:
            runtime_template_dir = _prepare_runtime_template(template_path)
        except FileNotFoundError:
            logger.info(
                "[PyCaps] Template assets missing, using local fallback renderer",
                extra={"template_dir": str(template_path)},
            )
        else:
            document = _payload_to_document(normalised["segments"], allow_emojis=allow_emojis)
            if not document.segments:
                raise RuntimeError("PyCaps rendering received no subtitle segments.")

            config_path = runtime_template_dir / _PYCAPS_TEMPLATE_FILENAME
            loader = JsonConfigLoader(str(config_path))
            builder = loader.load(should_build_pipeline=False)
            builder.should_save_subtitle_data(True)
            builder.with_cache_strategy(CacheStrategy.CSS_CLASSES_AWARE)
            builder.with_custom_subtitle_renderer(PictexSubtitleRenderer())
            resources_dir = runtime_template_dir / _PYCAPS_RESOURCES_DIRNAME
            if resources_dir.exists():
                try:
                    builder.with_resources(str(resources_dir))
                except Exception:
                    logger.debug("[PyCaps] Unable to register resources directory", exc_info=True)
            try:
                builder.add_css(str(runtime_template_dir / _PYCAPS_CSS_FILENAME))
            except Exception:
                logger.debug("[PyCaps] Unable to append template CSS explicitly", exc_info=True)
            builder.with_input_video(str(input_video_path))

            if output_path.exists():
                output_path.unlink()
            builder.with_output_video(str(output_path))

            pipeline = builder.build()

            cleanup_needed = False
            try:
                pipeline.prepare()
                cleanup_needed = True
                processed_document = pipeline.process_document(document)
                pipeline.render(processed_document)
                try:
                    _export_subtitle_debug_assets(output_path, runtime_template_dir)
                except Exception:
                    logger.debug("[PyCaps] Unable to export debug artefacts", exc_info=True)
                cleanup_needed = False  # render() triggers cleanup internally
            except Exception as exc:
                logger.error("[PyCaps] Rendering failed", exc_info=True)
                raise RuntimeError(f"PyCaps rendering failed: {exc}") from exc
            finally:
                if cleanup_needed:
                    try:
                        pipeline.close()
                    except Exception:  # pragma: no cover - defensive cleanup
                        logger.debug("[PyCaps] Pipeline cleanup failed", exc_info=True)

            logger.info(
                "[PyCaps] Rendered subtitles using PyCaps %s",
                _PYCAPS_VERSION or "unknown-version",
            )
            return str(output_path)

    return _render_with_local_renderer(
        normalised_payload=normalised,
        allow_emojis=allow_emojis,
        input_video_path=input_video_path,
        output_path=output_path,
        template_dir=template_path,
        responsive=responsive_enabled,
    )


__all__ = ["ensure_template_assets", "render_with_pycaps", "to_pycaps_input"]
