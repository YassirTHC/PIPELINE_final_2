"""Integration helpers for the optional PyCaps subtitle engine.

This module contains utilities that bridge the pipeline subtitle payload with the
`pycaps` package. Field reports highlighted that some released wheels expose the
``JsonConfigLoader`` symbol under different module layouts (``pycaps`` root vs
``pycaps.pipeline``), while the original integration only attempted a single
import path and re-raised ``ModuleNotFoundError`` as a misleading
"PyCaps is not installed" message. The helpers below keep the more permissive
import strategy and add runtime logging so that mismatches between the active
interpreter and the installed package are easy to diagnose.
"""
from __future__ import annotations

import importlib
import json
import logging
import pkgutil
import shutil
import tempfile
import sys
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


def _log_runtime_metadata(source_module: Any) -> None:
    """Emit diagnostic information about the active PyCaps installation."""

    pycaps_module = sys.modules.get("pycaps")
    module_for_metadata = pycaps_module or source_module
    version = getattr(pycaps_module, "__version__", None) if pycaps_module else None
    location = getattr(module_for_metadata, "__file__", "<unknown>")

    logger.info(
        "[PyCaps] Runtime info: interpreter=%s, version=%s, module_file=%s",
        sys.executable,
        version or "unknown",
        location,
    )


def _load_pycaps_loader():
    """Locate :class:`JsonConfigLoader` across the various PyCaps layouts."""

    attempt_errors: list[str] = []

    def _record_failure(label: str, exc: BaseException | str) -> None:
        if isinstance(exc, BaseException):
            attempt_errors.append(f"{label}: {exc}")
        else:
            attempt_errors.append(f"{label}: {exc}")

    def _extract_loader(module: Any, label: str):
        loader = getattr(module, "JsonConfigLoader", None)
        if loader is not None:
            logger.info("[PyCaps] Using layout %s", label)
            _log_runtime_metadata(module)
            return loader
        _record_failure(label, "JsonConfigLoader attribute missing")
        return None

    # Layout A: historical pycaps.pipeline export
    try:
        pipeline_module = importlib.import_module("pycaps.pipeline")
    except ModuleNotFoundError as exc:
        _record_failure("pycaps.pipeline", exc)
    except Exception as exc:  # pragma: no cover - defensive
        _record_failure("pycaps.pipeline", exc)
    else:
        loader = _extract_loader(pipeline_module, "A: pycaps.pipeline.JsonConfigLoader")
        if loader:
            return loader

    # Layout B: loader on the root package
    imported_package: Any | None = None
    pycaps_package: Any | None = None
    try:
        imported_package = importlib.import_module("pycaps")
    except ModuleNotFoundError as exc:
        _record_failure("pycaps", exc)
    except Exception as exc:  # pragma: no cover - defensive
        _record_failure("pycaps", exc)
    else:
        pycaps_package = imported_package

    if pycaps_package is None:
        pycaps_package = sys.modules.get("pycaps")

    if pycaps_package is not None:
        loader = _extract_loader(pycaps_package, "B: pycaps.JsonConfigLoader")
        if loader:
            return loader

        # Layout C: scan nested modules inside the distribution
        package_path = getattr(pycaps_package, "__path__", [])
        for module in pkgutil.iter_modules(package_path):
            module_name = f"pycaps.{module.name}"
            try:
                candidate = importlib.import_module(module_name)
            except Exception as exc:  # pragma: no cover - defensive
                _record_failure(module_name, exc)
                continue

            loader = _extract_loader(
                candidate, f"C: {module_name}.JsonConfigLoader"
            )
            if loader:
                return loader

    interpreter = sys.executable or "<unknown>"
    version = getattr(pycaps_package, "__version__", "unavailable") if pycaps_package else "unavailable"
    module_file = getattr(pycaps_package, "__file__", "unavailable") if pycaps_package else "unavailable"
    attempts = "; ".join(attempt_errors) if attempt_errors else "none"

    raise RuntimeError(
        "PyCaps import error: 'JsonConfigLoader' introuvable dans pycaps. "
        f"Tentatives: {attempts}. "
        f"Interpreter: {interpreter}. Version: {version}. Module: {module_file}. "
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

    pycaps_module = None
    try:
        pycaps_module = importlib.import_module("pycaps")
    except ModuleNotFoundError:
        logger.warning(
            "[PyCaps] Module introuvable dans l'environnement courant (%s)",
            sys.executable,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("[PyCaps] Impossible d'inspecter le module pycaps: %s", exc)

    try:
        JsonConfigLoader = _load_pycaps_loader()
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(
            "PyCaps is not installed in this interpreter. "
            "Active venv311 puis `pip install pycaps`."
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Unable to import/use PyCaps: {exc}") from exc

    if pycaps_module is not None:
        version = getattr(pycaps_module, "__version__", "unknown")
        module_file = getattr(pycaps_module, "__file__", "<unknown>")
        logger.info(
            "[PyCaps] Runtime: exe=%s version=%s file=%s",
            sys.executable,
            version,
            module_file,
        )

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
