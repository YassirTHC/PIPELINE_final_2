"""Integration helpers for the optional PyCaps subtitle engine."""
from __future__ import annotations

import importlib
import json
import logging
import pkgutil
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

logger = logging.getLogger(__name__)

_PYCAPS_TEMPLATE_FILENAME = "pycaps.template.json"


def ensure_template_assets(template_dir: str | Path) -> None:
    """Ensure runtime resources expected by the PyCaps template are available.

    Parameters
    ----------
    template_dir:
        Directory that contains the PyCaps template JSON and CSS files.
    """

    base_dir = Path(template_dir)
    resources_dir = base_dir / "resources"
    try:
        resources_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If we cannot create the directory we simply skip the copy. The CSS will
        # fall back to system fonts.
        return

    repo_root = Path(__file__).resolve().parents[1]
    source_font = repo_root / "assets" / "fonts" / "Montserrat-ExtraBold.ttf"
    target_font = resources_dir / "Montserrat-ExtraBold.ttf"

    if source_font.exists() and not target_font.exists():
        try:
            shutil.copyfile(source_font, target_font)
            logger.debug("Copied Montserrat font to %s", target_font)
        except Exception as exc:  # pragma: no cover - best effort copy
            logger.debug("Unable to copy Montserrat font: %s", exc)


def _time_fragment(start: float, end: float) -> Dict[str, float]:
    return {"start": float(start), "end": float(end)}


def _empty_layout() -> Dict[str, Dict[str, int]]:
    return {"position": {"x": 0, "y": 0}, "size": {"width": 0, "height": 0}}


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def to_pycaps_input(segments: Sequence[Mapping[str, Any]] | None) -> Dict[str, Any]:
    """Transform pipeline subtitle segments into the JSON format expected by PyCaps."""

    if not segments:
        return {"segments": []}

    document_segments: List[Dict[str, Any]] = []

    for segment in segments:
        seg_start = _coerce_float(segment.get("start"), 0.0)
        seg_end = _coerce_float(segment.get("end"), seg_start)
        seg_text = str(segment.get("text", "")).strip()

        words_payload: List[Dict[str, Any]] = []
        words = segment.get("words")
        if isinstance(words, Iterable):
            for word in words:
                text = str(getattr(word, "get", lambda k, d=None: word[k])("text", "") or "").strip()
                if not text:
                    continue
                start = _coerce_float(word.get("start", seg_start), seg_start)
                end = _coerce_float(word.get("end", start), start)
                words_payload.append(
                    {
                        "clips": [],
                        "text": text,
                        "semantic_tags": [],
                        "structure_tags": [],
                        "max_layout": _empty_layout(),
                        "time": _time_fragment(start, end),
                    }
                )
        if not words_payload and seg_text:
            words_payload.append(
                {
                    "clips": [],
                    "text": seg_text,
                    "semantic_tags": [],
                    "structure_tags": [],
                    "max_layout": _empty_layout(),
                    "time": _time_fragment(seg_start, seg_end),
                }
            )

        if not words_payload:
            continue

        line_start = min(item["time"]["start"] for item in words_payload)
        line_end = max(item["time"]["end"] for item in words_payload)

        line = {
            "words": words_payload,
            "structure_tags": [],
            "max_layout": _empty_layout(),
            "time": _time_fragment(line_start, line_end),
        }

        document_segments.append(
            {
                "lines": [line],
                "structure_tags": [],
                "max_layout": _empty_layout(),
                "time": _time_fragment(seg_start, seg_end),
            }
        )

    return {"segments": document_segments}


def _load_pycaps_loader():  # pragma: no cover - exercised via render_with_pycaps
    """Locate PyCaps JsonConfigLoader across possible layouts."""

    try:
        from pycaps.pipeline import JsonConfigLoader  # type: ignore

        logger.info("[PyCaps] Using layout A: pycaps.pipeline.JsonConfigLoader")
        return JsonConfigLoader
    except ModuleNotFoundError:
        pass

    try:
        import pycaps  # type: ignore  # noqa: F401

        JsonConfigLoader = getattr(pycaps, "JsonConfigLoader")
        logger.info("[PyCaps] Using layout B: pycaps.JsonConfigLoader")
        return JsonConfigLoader  # type: ignore
    except Exception:
        pass

    import pycaps  # type: ignore

    for module in pkgutil.iter_modules(getattr(pycaps, "__path__", [])):
        try:
            mod = importlib.import_module(f"pycaps.{module.name}")
        except Exception:
            continue

        if hasattr(mod, "JsonConfigLoader"):
            JsonConfigLoader = getattr(mod, "JsonConfigLoader")
            logger.info(
                "[PyCaps] Using layout C: pycaps.%s.JsonConfigLoader", module.name
            )
            return JsonConfigLoader  # type: ignore

    raise RuntimeError(
        "PyCaps import error: 'JsonConfigLoader' introuvable dans pycaps. "
        "Installe la version GitHub (ou mets à jour la lib) : "
        "pip install --no-cache-dir git+https://github.com/francozanardi/pycaps"
    )


def render_with_pycaps(
    segments: Sequence[Mapping[str, Any]] | None,
    output_video_path: str | Path,
    template_dir: str | Path,
    *,
    input_video_path: str | Path,
) -> None:
    """Render subtitles using the PyCaps engine."""

    template_base = Path(template_dir)
    template_path = template_base / _PYCAPS_TEMPLATE_FILENAME
    if not template_path.exists():
        raise FileNotFoundError(f"PyCaps template not found: {template_path}")

    ensure_template_assets(template_base)

    try:
        JsonConfigLoader = _load_pycaps_loader()
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(
            "PyCaps is not installed in this interpreter. "
            "Active venv311 puis `pip install pycaps`."
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Unable to import/use PyCaps: {exc}") from exc

    builder = JsonConfigLoader(str(template_path)).load(False)
    builder.with_input_video(str(input_video_path))

    output_path = Path(output_video_path)
    if output_path.exists():
        try:
            output_path.unlink()
        except Exception:
            pass
    builder.with_output_video(str(output_path))

    subtitle_payload = to_pycaps_input(segments)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "subtitles.json"
        data_path.write_text(json.dumps(subtitle_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        builder.with_subtitle_data_path(str(data_path))
        builder.should_save_subtitle_data(False)

        pipeline = builder.build()
        try:
            pipeline.run()
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional deps
            raise RuntimeError(
                "PyCaps is not installed in this interpreter. "
                "Active venv311 puis `pip install pycaps`."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"PyCaps rendering failed: {exc}") from exc
        finally:
            try:
                pipeline.close()
            except Exception:
                pass
