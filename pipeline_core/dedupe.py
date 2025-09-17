"""Deduplication helpers (URL + perceptual hash)."""
from __future__ import annotations

from io import BytesIO
import math
import subprocess
import shlex
from typing import Optional

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None


def _dct_2d(pixels):
    size = len(pixels)
    result = [[0.0] * size for _ in range(size)]
    factor = math.pi / (2.0 * size)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for u in range(size):
        for v in range(size):
            accum = 0.0
            for x in range(size):
                cos_x = math.cos((2 * x + 1) * u * factor)
                row = pixels[x]
                for y in range(size):
                    accum += row[y] * cos_x * math.cos((2 * y + 1) * v * factor)
            cu = inv_sqrt2 if u == 0 else 1.0
            cv = inv_sqrt2 if v == 0 else 1.0
            result[u][v] = 0.25 * cu * cv * accum
    return result


def _phash_from_image(img) -> int:
    img = img.convert("L").resize((32, 32))
    data = list(img.getdata())
    matrix = [data[i * 32:(i + 1) * 32] for i in range(32)]
    dct = _dct_2d(matrix)
    block = [dct[u][v] for u in range(8) for v in range(8)]
    median = sorted(block)[len(block) // 2]
    bits = 0
    for idx, coeff in enumerate(block):
        if coeff > median:
            bits |= (1 << idx)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def _fetch_first_frame_bytes_from_video(url: str, ss: float = 0.5, timeout_s: int = 6) -> Optional[bytes]:
    cmd = f'ffmpeg -hide_banner -loglevel error -ss {ss} -i "{url}" -frames:v 1 -f image2pipe -vcodec mjpeg -'
    try:
        proc = subprocess.run(shlex.split(cmd), capture_output=True, timeout=timeout_s)
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout
    except Exception:
        return None
    return None


def _fetch_image_bytes(url: str, timeout_s: int = 6) -> Optional[bytes]:
    try:
        import requests  # type: ignore
        resp = requests.get(url, timeout=timeout_s)
        if resp.ok:
            return resp.content
    except Exception:
        pass
    try:
        from urllib.request import urlopen  # type: ignore
        with urlopen(url, timeout=timeout_s) as resp:  # nosec - controlled usage
            return resp.read()
    except Exception:
        return None


def compute_phash(preview_or_stream, media_url: Optional[str] = None, timeout_s: int = 6) -> Optional[int]:
    """Return a 64-bit perceptual hash for an image/stream, or None."""
    if Image is None:
        return None

    data: Optional[bytes] = None
    if isinstance(preview_or_stream, (bytes, bytearray)):
        data = bytes(preview_or_stream)
    elif isinstance(preview_or_stream, str) and preview_or_stream:
        data = _fetch_image_bytes(preview_or_stream, timeout_s=timeout_s)

    if data is None and media_url:
        data = _fetch_first_frame_bytes_from_video(media_url, timeout_s=timeout_s)

    if not data:
        return None

    try:
        img = Image.open(BytesIO(data)).convert("RGB")
        return _phash_from_image(img)
    except Exception:
        return None
