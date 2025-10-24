from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:  # optional dependency, only logged for observability
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover - optional dependency absent in tests
    mp = None  # type: ignore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class SmartReframeConfig:
    """Configuration bundle for smart B-roll reframing."""

    sample_fps: int = 8
    padding: float = 0.10
    max_zoom: float = 1.8
    score_th: float = 0.50
    ema_alpha: float = 0.80

    def validated(self) -> "SmartReframeConfig":
        """Return a clamped version of the config."""
        sample = max(1, int(self.sample_fps))
        padding = max(0.0, float(self.padding))
        max_zoom = max(1.0, float(self.max_zoom))
        score = float(self.score_th)
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0
        alpha = float(self.ema_alpha)
        if alpha <= 0.0:
            alpha = 0.5
        if alpha > 1.0:
            alpha = 1.0
        return SmartReframeConfig(
            sample_fps=sample,
            padding=padding,
            max_zoom=max_zoom,
            score_th=score,
            ema_alpha=alpha,
        )


def smart_reframe_broll(input_path: str, output_path: str, cfg: SmartReframeConfig) -> str:
    """Generate a reframed version of a B-roll clip when possible."""
    config = cfg.validated()
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"B-roll source not found: {src}")
    out = Path(output_path)

    logger.info(
        "[BROLL][reframe] start mediapipe=%s fps=%s pad=%.2f max_zoom=%.2f score_th=%.2f src=%s",
        bool(mp),
        config.sample_fps,
        config.padding,
        config.max_zoom,
        config.score_th,
        src.name,
    )

    if out.exists():
        try:
            src_mtime = src.stat().st_mtime
            out_stat = out.stat()
            if out_stat.st_mtime >= src_mtime and out_stat.st_size > 0:
                return str(out)
        except OSError:
            pass

    try:
        import cv2  # type: ignore
    except Exception:
        logger.info("[BROLL][reframe] aborted reason=opencv_missing src=%s", src.name)
        return str(src)

    cap = cv2.VideoCapture(str(src))
    if not cap or not cap.isOpened():
        if cap:
            cap.release()
        logger.debug("cv2.VideoCapture failed for %s", src)
        return str(src)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or config.sample_fps or 24)

    if width <= 0 or height <= 0:
        cap.release()
        logger.debug("Invalid source dimensions for %s", src)
        return str(src)

    aspect = width / float(height)
    sample_interval = max(1, int(round(fps / max(1, config.sample_fps))))

    out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (width, height))
    if hasattr(writer, "isOpened") and not writer.isOpened():  # pragma: no cover - defensive
        writer.release()
        cap.release()
        logger.debug("cv2.VideoWriter failed to open for %s", out)
        return str(src)

    prev_gray: Optional[np.ndarray] = None
    ema_box: Optional[Tuple[float, float, float, float]] = None
    frame_index = 0
    any_crop = False
    processed_frames = 0
    last_detector = "none"
    last_box: Optional[Tuple[float, float, float, float]] = None
    last_crop: Optional[Tuple[int, int, int, int]] = None
    last_zoom: Optional[float] = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            processed_frames += 1
            frame_index += 1

            gray = _to_gray(frame, cv2)
            score = 0.0
            method = "saliency"
            new_box: Optional[Tuple[float, float, float, float]] = None

            if prev_gray is not None and frame_index % sample_interval == 0:
                new_box, score, method = _detect_region(gray, prev_gray, width, height, cv2)
            elif prev_gray is None:
                # First meaningful frame.
                new_box, score, method = _detect_region(gray, None, width, height, cv2)

            prev_gray = gray

            if new_box and score >= config.score_th:
                ema_box = _smooth_box(ema_box, new_box, config.ema_alpha)
            if new_box:
                last_detector = method
                last_box = new_box
            x1, y1, x2, y2 = _compute_crop(ema_box, width, height, aspect, config)
            if x1 <= 0 and y1 <= 0 and x2 >= width and y2 >= height:
                writer.write(frame)
                continue

            any_crop = True
            cropped = frame[y1:y2, x1:x2]
            if cropped.shape[1] != width or cropped.shape[0] != height:
                resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
            else:
                resized = cropped
            writer.write(resized)
            zoom = width / float(max(1, x2 - x1))
            last_crop = (x1, y1, x2, y2)
            last_zoom = zoom
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Smart reframe failed for %s: %s", src, exc, exc_info=True)
        processed_frames = 0
    finally:
        cap.release()
        writer.release()

    if processed_frames == 0 or not out.exists():
        _safe_unlink(out)
        logger.info("[BROLL][reframe] fallback_to_original reason=processing_failed src=%s", src.name)
        return str(src)

    if not any_crop:
        logger.info(
            "[BROLL][reframe] passthrough mediapipe=%s src=%s out=%s",
            bool(mp),
            src.name,
            out.name,
        )
        return str(out)

    raw_box = last_box if last_box is not None else (0.0, 0.0, float(width), float(height))
    crop_box = last_crop if last_crop is not None else (0, 0, width, height)
    zoom_value = last_zoom if last_zoom is not None else 1.0
    logger.info(
        "[BROLL][reframe] mediapipe=%s detector=%s raw=(%.1f,%.1f,%.1f,%.1f) crop916=(%d,%d,%d,%d) zoom=%.2f -> %s",
        bool(mp),
        last_detector,
        raw_box[0],
        raw_box[1],
        raw_box[2],
        raw_box[3],
        crop_box[0],
        crop_box[1],
        crop_box[2],
        crop_box[3],
        zoom_value,
        out.name,
    )
    return str(out)


def _to_gray(frame: np.ndarray, cv2_module) -> np.ndarray:
    try:
        gray = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2GRAY)
    except Exception:
        gray = np.asarray(frame)
        if gray.ndim == 3:
            gray = gray[..., 0]
        gray = gray.astype(np.uint8, copy=False)
    return gray


def _detect_region(
    gray: np.ndarray,
    prev_gray: Optional[np.ndarray],
    width: int,
    height: int,
    cv2_module,
) -> Tuple[Optional[Tuple[float, float, float, float]], float, str]:
    h, w = gray.shape[:2]
    if h != height or w != width:
        scale_y = height / float(h)
        scale_x = width / float(w)
    else:
        scale_y = scale_x = 1.0

    motion_score = 0.0
    combined = None

    if prev_gray is not None and prev_gray.shape == gray.shape:
        diff = np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16)).astype(np.uint8)
        mask = diff > 15
        motion_score = float(mask.sum()) / float(mask.size) if mask.size else 0.0
        combined = np.maximum(diff, 0)
    else:
        combined = gray

    edges = _canny_fallback(gray, cv2_module)
    if combined is None:
        combined = edges
    else:
        combined = np.maximum(edges, combined)

    detector = "motion_saliency"

    if not np.any(combined):
        return None, motion_score, detector

    coords = np.argwhere(combined > 0)
    if coords.size == 0:
        return None, motion_score, detector

    y_min = int(np.min(coords[:, 0]))
    y_max = int(np.max(coords[:, 0]))
    x_min = int(np.min(coords[:, 1]))
    x_max = int(np.max(coords[:, 1]))

    bw = x_max - x_min + 1
    bh = y_max - y_min + 1
    if bw <= 0 or bh <= 0:
        return None, motion_score, detector

    cx = (x_min + x_max) / 2.0 + 0.5
    cy = (y_min + y_max) / 2.0 + 0.5

    cx *= scale_x
    cy *= scale_y
    bw_abs = bw * scale_x
    bh_abs = bh * scale_y

    total_pixels = float(width * height) if width and height else 1.0
    norm_area = min(1.0, (bw_abs * bh_abs) / total_pixels)
    mask_area_ratio = min(1.0, coords.shape[0] / total_pixels)
    framing_score = max(0.0, min(1.0, 1.0 - norm_area))
    score = max(framing_score, motion_score, mask_area_ratio)

    return (cx, cy, bw_abs, bh_abs), score, detector


def _canny_fallback(gray: np.ndarray, cv2_module) -> np.ndarray:
    if cv2_module is None:
        return np.zeros_like(gray, dtype=np.uint8)

    try:
        edges = cv2_module.Canny(gray, 50, 150)
    except Exception:
        edges = np.zeros_like(gray, dtype=np.uint8)
    return edges


def _smooth_box(
    previous: Optional[Tuple[float, float, float, float]],
    current: Tuple[float, float, float, float],
    alpha: float,
) -> Tuple[float, float, float, float]:
    if previous is None:
        return current
    beta = 1.0 - alpha
    return (
        alpha * current[0] + beta * previous[0],
        alpha * current[1] + beta * previous[1],
        alpha * current[2] + beta * previous[2],
        alpha * current[3] + beta * previous[3],
    )


def _compute_crop(
    ema_box: Optional[Tuple[float, float, float, float]],
    width: int,
    height: int,
    aspect: float,
    cfg: SmartReframeConfig,
) -> Tuple[int, int, int, int]:
    if not ema_box:
        return 0, 0, width, height

    cx, cy, bw, bh = ema_box
    pad_factor = 1.0 + cfg.padding * 2.0

    target_w = min(float(width), bw * pad_factor)
    target_h = min(float(height), bh * pad_factor)

    min_w = float(width) / cfg.max_zoom
    min_h = float(height) / cfg.max_zoom

    target_w = max(min_w, min(float(width), max(1.0, target_w)))
    target_h = max(min_h, min(float(height), max(1.0, target_h)))

    desired_w = max(target_w, target_h * aspect)
    desired_h = desired_w / aspect

    if desired_h < target_h:
        desired_h = target_h
        desired_w = desired_h * aspect

    desired_w = min(float(width), max(min_w, desired_w))
    desired_h = min(float(height), max(min_h, desired_h))

    half_w = desired_w / 2.0
    half_h = desired_h / 2.0

    cx = float(np.clip(cx, half_w, width - half_w))
    cy = float(np.clip(cy, half_h, height - half_h))

    x1 = int(math.floor(cx - half_w))
    y1 = int(math.floor(cy - half_h))
    x2 = int(math.ceil(cx + half_w))
    y2 = int(math.ceil(cy + half_h))

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > width:
        shift = x2 - width
        x1 -= shift
        x2 = width
    if y2 > height:
        shift = y2 - height
        y1 -= shift
        y2 = height

    if x2 <= x1 or y2 <= y1:
        return 0, 0, width, height

    final_w = x2 - x1
    final_h = y2 - y1
    current_aspect = final_w / float(final_h)
    if abs(current_aspect - aspect) > 0.01:
        if current_aspect > aspect:
            needed_w = int(round(final_h * aspect))
            delta = final_w - needed_w
            x1 += delta // 2
            x2 = x1 + needed_w
        else:
            needed_h = int(round(final_w / aspect))
            delta = final_h - needed_h
            y1 += delta // 2
            y2 = y1 + needed_h
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return 0, 0, width, height

    return x1, y1, x2, y2


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        logger.debug("Unable to remove temporary smart reframe file %s", path)


__all__ = ["SmartReframeConfig", "smart_reframe_broll"]
