from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hf_path: str
    family: str  # "qwen3" or "qwen25"


MODELS: List[ModelConfig] = [
    ModelConfig(
        name="Qwen3-VL-8B-SAMTok",
        hf_path="zhouyik/Qwen3-VL-8B-SAMTok",
        family="qwen3",
    ),
    ModelConfig(
        name="Qwen3-VL-4B-SAMTok",
        hf_path="zhouyik/Qwen3-VL-4B-SAMTok",
        family="qwen3",
    ),
    ModelConfig(
        name="Qwen2.5-VL-7B-SAMTok-co",
        hf_path="zhouyik/Qwen2.5-VL-7B-SAMTok-co",
        family="qwen25",
    ),
    ModelConfig(
        name="Qwen2.5-VL-3B-SAMTok-co",
        hf_path="zhouyik/Qwen2.5-VL-3B-SAMTok-co",
        family="qwen25",
    ),
    ModelConfig(
        name="Qwen3-VL-4B-SAMTok-co",
        hf_path="zhouyik/Qwen3-VL-4B-SAMTok-co",
        family="qwen3",
    ),
]

DATASET_NAME = "moondream/refcoco-m"
DATASET_SPLIT = "refcoco_val"
DATASET_DEFAULT_SPLITS = {
    "lmms-lab/RefCOCOplus": "val",
    "moondream/refcoco_plus_rle_val": "validation",
    "moondream/refcoco_rle": "validation",
    "moondream/lvis_segmentation": "validation",
    "moondream/refcoco-m": "validation",

}

PROMPT_TEMPLATE = "Please segment {referring_expression} in this image."

CODEBOOK_SIZE = 256
CODEBOOK_DEPTH = 2
SAM2_IMAGE_SIZE = 1024
