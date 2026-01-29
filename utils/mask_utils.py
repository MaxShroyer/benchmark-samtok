from typing import Any, Dict

import numpy as np
from pycocotools import mask as mask_utils


def decode_rle(rle: Any, height: int, width: int) -> np.ndarray:
    if rle is None:
        return np.zeros((height, width), dtype=np.uint8)

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
