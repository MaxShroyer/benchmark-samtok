import io
import json
from typing import Any, Dict

import numpy as np
from pycocotools import mask as mask_utils
from PIL import Image


def _decode_svg(svg: str, height: int, width: int) -> np.ndarray:
    """
    Rasterize an SVG string to a binary mask at (height, width).

    `moondream/lvis_segmentation` provides per-instance masks as SVG paths.
    We rasterize them so downstream metrics can use the same mask pipeline as RLE.
    """
    try:
        import cairosvg  # type: ignore
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "SVG masks require `cairosvg`.\n"
            "Install: `pip install cairosvg` (may also require OS cairo libs)."
        ) from exc

    png_bytes: bytes = cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        output_width=int(width),
        output_height=int(height),
        background_color="transparent",
    )
    img = Image.open(io.BytesIO(png_bytes))
    arr = np.array(img)
    if arr.ndim == 3 and arr.shape[2] >= 4:
        # Prefer alpha channel when present (transparent background).
        mask = arr[:, :, 3] > 0
    else:
        # Fallback: threshold luminance.
        gray = np.array(img.convert("L"))
        mask = gray < 128
    return mask.astype(np.uint8)


def decode_rle(rle: Any, height: int, width: int) -> np.ndarray:
    if rle is None:
        return np.zeros((height, width), dtype=np.uint8)

    # Some datasets store the RLE object JSON-encoded as a string.
    if isinstance(rle, str):
        s = rle.strip()
        if s.startswith("<svg"):
            return _decode_svg(s, height, width)
        if s.startswith("{") or s.startswith("["):
            try:
                rle = json.loads(s)
            except Exception:
                # If parsing fails, fall through to best-effort decoding below.
                pass

    # Support polygon segmentations (COCO style) which may arrive as:
    # - a flat list of numbers: [x1, y1, x2, y2, ...]
    # - a list of polygons: [[...], [...]]
    if isinstance(rle, list):
        if not rle:
            return np.zeros((height, width), dtype=np.uint8)
        if all(isinstance(x, (int, float)) for x in rle):
            rle_obj = mask_utils.frPyObjects([rle], height, width)
            mask = mask_utils.decode(rle_obj)
            if mask.ndim == 3:
                mask = np.any(mask, axis=2)
            return mask.astype(np.uint8)
        if all(isinstance(poly, list) for poly in rle) and all(
            all(isinstance(x, (int, float)) for x in poly) for poly in rle
        ):
            rle_obj = mask_utils.frPyObjects(rle, height, width)
            mask = mask_utils.decode(rle_obj)
            if mask.ndim == 3:
                mask = np.any(mask, axis=2)
            return mask.astype(np.uint8)
        # Some datasets store multiple RLE dicts in a list; pick the first.
        if isinstance(rle[0], dict):
            rle = rle[0]

    if not isinstance(rle, dict):
        return np.zeros((height, width), dtype=np.uint8)

    # SVG-style masks (e.g. moondream/lvis_segmentation)
    svg = rle.get("svg")
    if isinstance(svg, str) and svg.strip().startswith("<svg"):
        return _decode_svg(svg, height, width)
    svg_data = rle.get("svg_data")
    if isinstance(svg_data, dict):
        svg_str = svg_data.get("svg_str")
        paths = svg_data.get("paths")
        if isinstance(svg_str, str) and isinstance(paths, list) and paths:
            # Reconstruct the full svg by substituting |PATH| with a combined path string.
            # SVG allows multiple subpaths in a single `d` attribute.
            combined_d = " ".join(p for p in paths if isinstance(p, str))
            reconstructed = svg_str.replace("|PATH|", combined_d)
            if reconstructed.strip().startswith("<svg"):
                return _decode_svg(reconstructed, height, width)

    counts = rle.get("counts")
    if isinstance(counts, list):
        rle_obj = mask_utils.frPyObjects(rle, height, width)
        mask = mask_utils.decode(rle_obj)
    else:
        mask = mask_utils.decode(rle)

    if mask.ndim == 3:
        # If multiple components are present, merge them.
        mask = np.any(mask, axis=2)
    return mask.astype(np.uint8)


def encode_rle(mask: np.ndarray) -> Dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle
