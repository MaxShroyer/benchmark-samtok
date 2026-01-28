import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from config import CODEBOOK_DEPTH, CODEBOOK_SIZE, DATASET_NAME, DATASET_SPLIT, PROMPT_TEMPLATE
from samtok_imports import ensure_samtok_imports
from utils.mask_utils import decode_rle
from utils.metrics import MetricTracker, compute_sample_metrics
from utils.token_utils import parse_quant_codes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SAMTok on RefCOCO-M")
    parser.add_argument("--model", required=True, help="Hugging Face model path")
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--split", default=DATASET_SPLIT)
    parser.add_argument("--max-samples", type=int, default=0, help="Limit total expressions")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--save-per-sample", action="store_true")
    return parser.parse_args()


def resolve_asset(model_path: str, filename: str) -> str:
    if os.path.isdir(model_path):
        candidate = os.path.join(model_path, filename)
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Missing asset {candidate}")
        return candidate
    return hf_hub_download(repo_id=model_path, filename=filename)


def load_vlm(model_path: str) -> torch.nn.Module:
    if "qwen3" in model_path.lower():
        from transformers import Qwen3VLForConditionalGeneration

        model_cls = Qwen3VLForConditionalGeneration
    elif "qwen2.5" in model_path.lower() or "qwen25" in model_path.lower():
        try:
            from transformers import Qwen2_5VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Qwen2.5-VL model class not found. "
                "Please install a transformers version that includes Qwen2_5VLForConditionalGeneration."
            ) from exc

        model_cls = Qwen2_5VLForConditionalGeneration
    else:
        raise ValueError(f"Unknown model family for {model_path}")

    return model_cls.from_pretrained(model_path, torch_dtype="auto")


def build_samtok(model_path: str, device: torch.device):
    samtok_path = ensure_samtok_imports()
    if not samtok_path:
        raise RuntimeError(
            "Sa2VA repo not found. Set SAMTOK_SA2VA_PATH to the Sa2VA repository root."
        )

    from projects.samtok.models import DirectResize, VQ_SAM2, VQ_SAM2Config, SAM2Config

    sam2_ckpt = resolve_asset(model_path, "sam2.1_hiera_large.pt")
    tokenizer_ckpt = resolve_asset(model_path, "mask_tokenizer_256x2.pth")

    sam2_config = SAM2Config(ckpt_path=sam2_ckpt)
    vq_sam2_config = VQ_SAM2Config(
        sam2_config=sam2_config,
        codebook_size=CODEBOOK_SIZE,
        codebook_depth=CODEBOOK_DEPTH,
        shared_codebook=False,
        latent_dim=256,
    )
    vq_sam2 = VQ_SAM2(vq_sam2_config).to(device).eval()
    state = torch.load(tokenizer_ckpt, map_location="cpu")
    vq_sam2.load_state_dict(state)
    sam2_image_processor = DirectResize(1024)

    return vq_sam2, sam2_image_processor


def build_prompt(expression: str) -> str:
    return PROMPT_TEMPLATE.format(referring_expression=expression)


def resolve_hf_token(dataset_name: str) -> str | None:
    gated_datasets = {
        "moondream/lvis_segmentation",
        "moondream/refcoco_plus_rle_val",
        "moondream/refcoco_rle",
    }
    if dataset_name not in gated_datasets:
        print(f"Dataset {dataset_name} is not gated, skipping token resolution")
        return None

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing Hugging Face token for gated datasets. "
            "Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN in .env."
        )
    return token


def iter_expressions(sample: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    samples = sample.get("samples", [])
    output = []
    for inst in samples:
        sentences = inst.get("sentences") or inst.get("sentence")
        if isinstance(sentences, str):
            sentences = [sentences]
        for sentence in sentences or []:
            output.append((sentence, inst))
    return output


@torch.no_grad()
def predict_mask(
    model: torch.nn.Module,
    processor: AutoProcessor,
    vq_sam2: torch.nn.Module,
    sam2_image_processor,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
) -> np.ndarray:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        top_p=1.0,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    quant_codes = parse_quant_codes(output_text, CODEBOOK_SIZE, CODEBOOK_DEPTH)
    if not quant_codes:
        return None

    codes = quant_codes[0][:CODEBOOK_DEPTH]
    normalized = []
    for code in codes:
        if 0 <= code < CODEBOOK_SIZE:
            normalized.append(code)
        else:
            normalized.append(-1)

    if all(code == -1 for code in normalized):
        return None

    sam2_image = np.array(image)
    sam2_image = sam2_image_processor.apply_image(sam2_image)
    sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
    sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

    quant_ids = torch.LongTensor(normalized).unsqueeze(0).to(vq_sam2.device)

    pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_ids)
    pred_masks = pred_masks.float()
    pred_masks = F.interpolate(
        pred_masks, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    pred_masks = (pred_masks > 0.5).cpu().numpy().astype(np.uint8)
    return pred_masks[0, 0]


def main() -> None:
    args = parse_args()
    load_dotenv()

    model = load_vlm(args.model)
    model = model.cuda().eval()
    processor = AutoProcessor.from_pretrained(args.model)
    vq_sam2, sam2_image_processor = build_samtok(args.model, model.device)

    hf_token = resolve_hf_token(args.dataset)
    dataset = load_dataset(args.dataset, split=args.split, token=hf_token)
    tracker = MetricTracker()
    per_sample_results = []

    total_expr = 0
    for sample in tqdm(dataset, desc="Evaluating"):
        image = sample["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        width, height = image.size

        for expression, inst in iter_expressions(sample):
            if args.max_samples and total_expr >= args.max_samples:
                break

            prompt = build_prompt(expression)
            pred_mask = predict_mask(
                model,
                processor,
                vq_sam2,
                sam2_image_processor,
                image,
                prompt,
                args.max_new_tokens,
            )

            gt_mask = decode_rle(inst.get("mask"), height, width)
            if pred_mask is None:
                pred_mask = np.zeros_like(gt_mask)

            sample_metrics = compute_sample_metrics(pred_mask, gt_mask)
            tracker.update(sample_metrics)

            if args.save_per_sample:
                per_sample_results.append(
                    {
                        "image_id": sample.get("image_id"),
                        "expression": expression,
                        "metrics": {
                            "iou": sample_metrics["iou"],
                            "precision": sample_metrics["precision"],
                            "recall": sample_metrics["recall"],
                            "f1": sample_metrics["f1"],
                        },
                    }
                )

            total_expr += 1
        if args.max_samples and total_expr >= args.max_samples:
            break

    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "metrics": tracker.summary(),
    }
    if args.save_per_sample:
        output["per_sample_results"] = per_sample_results

    output_path = os.path.join(
        args.output_dir, args.model.replace("/", "_") + ".json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
