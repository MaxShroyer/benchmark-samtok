from typing import Any, Dict

import numpy as np
from pycocotools import mask as mask_utils


def decode_rle(rle: Dict[str, Any], height: int, width: int) -> np.ndarray:
    if rle is None:
        return np.zeros((height, width), dtype=np.uint8)
    if isinstance(rle, list):
        rle = rle[0]

    counts = rle.get("counts")
    if isinstance(counts, list):
        rle_obj = mask_utils.frPyObjects(rle, height, width)
        mask = mask_utils.decode(rle_obj)
    else:
        mask = mask_utils.decode(rle)

    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.uint8)


def encode_rle(mask: np.ndarray) -> Dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle
