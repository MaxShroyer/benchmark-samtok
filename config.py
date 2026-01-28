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

PROMPT_TEMPLATE = (
    "Please segment the object described by: \"{referring_expression}\". "
    "Respond with the segmentation mask tokens."
)

CODEBOOK_SIZE = 256
CODEBOOK_DEPTH = 2
SAM2_IMAGE_SIZE = 1024
