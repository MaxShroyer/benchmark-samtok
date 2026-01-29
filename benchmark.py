import argparse
import io
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import IterableDataset, get_dataset_split_names, load_dataset
from datasets.exceptions import DatasetGenerationError
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from config import (
    CODEBOOK_DEPTH,
    CODEBOOK_SIZE,
    DATASET_DEFAULT_SPLITS,
    DATASET_NAME,
    DATASET_SPLIT,
    PROMPT_TEMPLATE,
    SAM2_IMAGE_SIZE,
)
from samtok_imports import ensure_samtok_imports
from utils.mask_utils import decode_rle
from utils.metrics import MetricTracker, compute_sample_metrics, compute_set_metrics
from utils.token_utils import parse_quant_codes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SAMTok on RefCOCO-M")
    parser.add_argument("--model", required=True, help="Hugging Face model path")
    parser.add_argument(
        "--dataset",
        default=DATASET_NAME,
        help="Dataset name or a comma-separated list for comparison",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split or comma-separated list (optional)",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Limit total expressions")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--save-per-sample", action="store_true")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="samtok-benchmark")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--wandb-log-every", type=int, default=1)
    return parser.parse_args()


def resolve_asset(model_path: str, filename: str) -> str:
    if os.path.isdir(model_path):
        candidate = os.path.join(model_path, filename)
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Missing asset {candidate}")
        return candidate
    return hf_hub_download(repo_id=model_path, filename=filename)


def load_vlm(model_path: str) -> torch.nn.Module:
    model_kwargs: Dict[str, Any] = {"torch_dtype": "auto"}

    if "qwen3" in model_path.lower():
        from transformers import Qwen3VLForConditionalGeneration

        model_cls = Qwen3VLForConditionalGeneration
    elif "qwen2.5" in model_path.lower() or "qwen25" in model_path.lower():
        try:
            from transformers import Qwen2_5VLForConditionalGeneration
        except ImportError as exc:
            try:
                from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (  # type: ignore
                    Qwen2_5VLForConditionalGeneration,
                )
            except ImportError:
                try:
                    import transformers

                    auto_causal = getattr(transformers, "AutoModelForCausalLM", None)
                    auto_model = getattr(transformers, "AutoModel", None)
                    if auto_causal is None and auto_model is None:
                        raise ImportError(
                            "AutoModelForCausalLM not available in transformers."
                        )
                except Exception as inner_exc:
                    raise ImportError(
                        "Qwen2.5-VL model class not found and auto model fallback "
                        "is unavailable in this transformers build. "
                        "Please install a standard transformers wheel (e.g. "
                        "`pip install -U \"transformers>=4.49.0,<5.0.0\"`)."
                    ) from inner_exc

                try:
                    return auto_causal.from_pretrained(
                        model_path, torch_dtype="auto", trust_remote_code=True
                    )
                except Exception:
                    if auto_model is None:
                        raise
                    return auto_model.from_pretrained(
                        model_path, torch_dtype="auto", trust_remote_code=True
                    )

        model_cls = Qwen2_5VLForConditionalGeneration
    else:
        raise ValueError(f"Unknown model family for {model_path}")

    try:
        import transformers
        from transformers import AutoModel, AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "Loaded a model without `generate`, and auto model classes "
            "are unavailable. Please upgrade transformers."
        ) from exc
    auto_vision2seq = getattr(transformers, "AutoModelForVision2Seq", None)

    def try_load(cls, trust_remote_code: bool) -> torch.nn.Module | None:
        try:
            return cls.from_pretrained(
                model_path,
                torch_dtype="auto",
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            return None

    # Qwen model repos sometimes ship custom modeling code that returns a bare
    # torch.nn.Module without GenerationMixin (no `.generate`). Prefer the
    # official Transformers implementations first.
    trust_order = [False, True] if "qwen" in model_path.lower() else [False]

    candidates: list[type] = []
    if auto_vision2seq is not None:
        candidates.append(auto_vision2seq)
    candidates.append(model_cls)
    candidates.append(AutoModelForCausalLM)
    candidates.append(AutoModel)

    for cls in candidates:
        for use_remote in trust_order:
            fallback_model = try_load(cls, use_remote)
            if fallback_model is None:
                continue
            if hasattr(fallback_model, "generate"):
                return fallback_model

    raise RuntimeError(
        "Model backend does not provide `generate`. Install a compatible "
        "transformers build or use a Qwen* VL model class that supports generation."
    )


def load_processor(model_path: str) -> AutoProcessor:
    base_kwargs: Dict[str, Any] = {}
    # Qwen VL processors frequently rely on custom code in the repo.
    if "qwen" in model_path.lower():
        base_kwargs["trust_remote_code"] = True
        # Suppress fast/slow warning and match legacy processor behavior.
        base_kwargs["use_fast"] = False

    # If torchvision isn't available, "fast" image processing can't be used anyway.
    # Avoid the noisy warning + fallback by explicitly disabling it.
    try:
        import torchvision  # noqa: F401
    except Exception:
        base_kwargs["use_fast"] = False

    try:
        return AutoProcessor.from_pretrained(model_path, **base_kwargs)
    except TypeError:
        tb = traceback.format_exc()
        # Transformers may crash inside video_processing_auto when a model declares a
        # video processor but provides no class name (e.g. `None`).
        if "video_processing_auto" not in tb and "video_processor" not in tb:
            raise

        # Work around models that declare a video processor without a class.
        fallback_kwargs_list = [
            {**base_kwargs, "video_processor": None},
            {**base_kwargs, "use_fast": False, "video_processor": None},
            {**base_kwargs, "trust_remote_code": True, "video_processor": None},
            {
                **base_kwargs,
                "trust_remote_code": True,
                "use_fast": False,
                "video_processor": None,
            },
        ]
        last_exc: Exception | None = None
        for kwargs in fallback_kwargs_list:
            try:
                return AutoProcessor.from_pretrained(model_path, **kwargs)
            except Exception as exc:  # pragma: no cover - best-effort compatibility
                last_exc = exc
                continue
        assert last_exc is not None
        raise last_exc


class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("DirectResize expects a numpy array input.")
        pil_image = Image.fromarray(image).convert("RGB")
        resized = pil_image.resize((self.target_length, self.target_length))
        return np.array(resized)


def _load_module_directly(name: str, filepath: str):
    """Load a Python module directly from file, bypassing package __init__.py."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_package_directly(name: str, package_dir: str):
    """Load a Python package from <package_dir>/__init__.py without parents' __init__.py."""
    import importlib.util

    init_py = os.path.join(package_dir, "__init__.py")
    spec = importlib.util.spec_from_file_location(name, init_py, submodule_search_locations=[package_dir])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_samtok(model_path: str, device: torch.device):
    samtok_path = ensure_samtok_imports()
    if not samtok_path:
        raise RuntimeError(
            "Sa2VA repo not found.\n"
            "- Clone it: `git clone https://github.com/bytedance/Sa2VA.git`\n"
            "- Then either:\n"
            "  - export SAMTOK_SA2VA_PATH=/path/to/Sa2VA\n"
            "  - or place the clone at `../Sa2VA` relative to this benchmark repo.\n"
        )

    # Import modules directly to avoid the __init__.py which pulls in xtuner.
    # We need to pre-load dependencies and register them in sys.modules so that
    # when sam2.py imports from projects.samtok.models.losses, Python finds our
    # pre-loaded module instead of triggering the package __init__.py.
    models_dir = os.path.join(samtok_path, "projects", "samtok", "models")

    # Create stub parent packages to prevent __init__.py from running
    import types

    for pkg_name in ["projects", "projects.samtok", "projects.samtok.models"]:
        if pkg_name not in sys.modules:
            sys.modules[pkg_name] = types.ModuleType(pkg_name)

    # Load losses first (dependency of sam2.py). In Sa2VA, this is a package: models/losses/
    losses_dir = os.path.join(models_dir, "losses")
    if os.path.isdir(losses_dir):
        _load_package_directly("projects.samtok.models.losses", losses_dir)
    else:
        # Older layouts may have a single losses.py file
        _load_module_directly("projects.samtok.models.losses", os.path.join(models_dir, "losses.py"))

    # Load sam2.py
    sam2_module = _load_module_directly(
        "projects.samtok.models.sam2",
        os.path.join(models_dir, "sam2.py"),
    )

    VQ_SAM2 = sam2_module.VQ_SAM2
    VQ_SAM2Config = sam2_module.VQ_SAM2Config
    SAM2Config = sam2_module.SAM2Config

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
    sam2_image_processor = DirectResize(SAM2_IMAGE_SIZE)

    return vq_sam2, sam2_image_processor


def build_prompt(expression: str) -> str:
    return PROMPT_TEMPLATE.format(referring_expression=expression)


def normalize_split_name(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "")


def resolve_split_candidates(dataset_name: str, token: str | None) -> List[str]:
    try:
        return get_dataset_split_names(dataset_name, token=token)
    except Exception:
        return []


def resolve_splits(dataset_name: str, split_arg: str | None, token: str | None) -> List[str]:
    requested = split_arg or DATASET_DEFAULT_SPLITS.get(dataset_name) or DATASET_SPLIT
    splits = [chunk.strip() for chunk in requested.split(",") if chunk.strip()]

    split_aliases = {}
    if dataset_name == "lmms-lab/RefCOCOplus":
        split_aliases = {
            "val": "validation",
            "valid": "validation",
            "testa": "testA",
            "testb": "testB",
        }

    available = resolve_split_candidates(dataset_name, token)
    normalized_available = {normalize_split_name(name): name for name in available}

    resolved = []
    for split in splits:
        normalized = normalize_split_name(split)
        mapped = split_aliases.get(normalized, split)
        if normalized_available:
            mapped_normalized = normalize_split_name(mapped)
            mapped = normalized_available.get(mapped_normalized, mapped)
        resolved.append(mapped)
    return resolved


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


def load_parquet_fallback(dataset_name: str, split: str, token: str | None):
    """
    Fallback loader that bypasses dataset info/feature casting by reading parquet
    files directly from the dataset repo.
    """
    api = HfApi(token=token)
    repo_files = api.list_repo_files(dataset_name, repo_type="dataset")
    parquet_files = [path for path in repo_files if path.endswith(".parquet")]
    if not parquet_files:
        raise RuntimeError(f"No parquet files found in dataset repo {dataset_name}.")

    split_key = normalize_split_name(split)
    matched = []
    for path in parquet_files:
        name = normalize_split_name(os.path.basename(path))
        if split_key and split_key in name:
            matched.append(path)

    data_files = matched or parquet_files
    hf_paths = [f"hf://datasets/{dataset_name}/{path}" for path in data_files]
    return load_dataset(
        "parquet",
        data_files=hf_paths,
        split="train",
        streaming=True,
        token=token,
    )


def load_dataset_robust(dataset_name: str, split: str, token: str | None):
    """
    Load a dataset split, with a fallback for parquet schema mismatches.

    Some HF-hosted parquet datasets can drift between the repo metadata schema and
    the parquet files' actual Arrow schema, causing `download_and_prepare()` to
    fail with a cast error. In those cases, streaming mode often succeeds because
    it avoids the prepare step.
    """
    def _exception_chain_messages(exc: BaseException) -> List[str]:
        msgs: List[str] = []
        seen: set[int] = set()
        cur: BaseException | None = exc
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            msgs.append(f"{type(cur).__name__}: {cur}")
            cur = cur.__cause__ or cur.__context__
        return msgs

    try:
        return load_dataset(dataset_name, split=split, token=token)
    except DatasetGenerationError as exc:
        chain_msgs = _exception_chain_messages(exc)
        if not any("Couldn't cast array of type" in m for m in chain_msgs):
            raise

        print(
            f"Warning: failed to build {dataset_name}:{split} due to a parquet schema mismatch. "
            "Retrying in streaming mode."
        )
        dataset = load_dataset(dataset_name, split=split, token=token, streaming=True)
        if isinstance(dataset, IterableDataset):
            try:
                next(iter(dataset))
            except TypeError as inner_exc:
                if "Couldn't cast array of type" in str(inner_exc):
                    print(
                        f"Warning: streaming cast failed for {dataset_name}:{split}. "
                        "Falling back to direct parquet reader."
                    )
                    return load_parquet_fallback(dataset_name, split, token)
                raise
            except StopIteration:
                pass
            # Re-create to avoid skipping the first sample from the check above.
            dataset = load_dataset(dataset_name, split=split, token=token, streaming=True)
        return dataset


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
def select_primary_mask(pred_masks: List[np.ndarray]) -> np.ndarray | None:
    if not pred_masks:
        return None
    for mask in pred_masks:
        if mask is not None and mask.any():
            return mask
    return pred_masks[0]


def decode_masks_from_codes(
    vq_sam2: torch.nn.Module,
    sam2_image_processor,
    image: Image.Image,
    codes_list: List[List[int]],
) -> List[np.ndarray]:
    if not codes_list:
        return []

    sam2_image = np.array(image)
    sam2_image = sam2_image_processor.apply_image(sam2_image)
    sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
    sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
    sam2_pixel_values = sam2_pixel_values.repeat(len(codes_list), 1, 1, 1)

    quant_ids = torch.LongTensor(codes_list).to(vq_sam2.device)

    pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_ids)
    pred_masks = pred_masks.float()
    pred_masks = F.interpolate(
        pred_masks, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    pred_masks = (pred_masks > 0.5).cpu().numpy().astype(np.uint8)
    return [pred_masks[idx, 0] for idx in range(pred_masks.shape[0])]


def predict_masks(
    model: torch.nn.Module,
    processor: AutoProcessor,
    vq_sam2: torch.nn.Module,
    sam2_image_processor,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
) -> List[np.ndarray]:
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
        return []

    normalized_codes = []
    for codes in quant_codes:
        codes = codes[:CODEBOOK_DEPTH]
        normalized = []
        for code in codes:
            if 0 <= code < CODEBOOK_SIZE:
                normalized.append(code)
            else:
                normalized.append(-1)
        if all(code == -1 for code in normalized):
            continue
        normalized_codes.append(normalized)

    return decode_masks_from_codes(vq_sam2, sam2_image_processor, image, normalized_codes)


def init_wandb(args: argparse.Namespace):
    if args.no_wandb:
        return None, None
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        return None, None
    try:
        import wandb
    except ImportError:
        print("Warning: WANDB_API_KEY set but wandb is not installed.")
        return None, None

    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=tags or None,
    )
    return wandb, run


def write_results(args: argparse.Namespace, results: List[Dict[str, Any]]) -> str:
    os.makedirs(args.output_dir, exist_ok=True)
    if len(results) == 1:
        output = results[0]
        if not args.save_per_sample:
            output.pop("per_sample_results", None)
    else:
        if not args.save_per_sample:
            for result in results:
                result.pop("per_sample_results", None)
        output = {"model": args.model, "runs": results}

    output_path = os.path.join(
        args.output_dir, args.model.replace("/", "_") + ".json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    return output_path


def main() -> None:
    args = parse_args()
    load_dotenv()

    model = load_vlm(args.model)
    model = model.cuda().eval()
    processor = load_processor(args.model)
    vq_sam2, sam2_image_processor = build_samtok(args.model, model.device)

    dataset_names = [name.strip() for name in args.dataset.split(",") if name.strip()]
    results = []
    wandb, wandb_run = init_wandb(args)
    run_start = time.perf_counter()

    if wandb_run is not None:
        wandb.config.update(
            {
                "model": args.model,
                "datasets": dataset_names,
                "split_arg": args.split,
                "max_samples": args.max_samples,
                "max_new_tokens": args.max_new_tokens,
                "save_per_sample": args.save_per_sample,
            }
        )
        wandb.define_metric("expr_step")
        wandb.define_metric("sample/*", step_metric="expr_step")
        wandb.define_metric("perf/*", step_metric="expr_step")
        wandb.define_metric("summary/*")

    for dataset_name in dataset_names:
        hf_token = resolve_hf_token(dataset_name)
        splits = resolve_splits(dataset_name, args.split, hf_token)

        for split in splits:
            dataset = load_dataset_robust(dataset_name, split=split, token=hf_token)
            if isinstance(dataset, IterableDataset):
                print(
                    f"Loaded {dataset_name}:{split} in streaming mode; "
                    "progress has no total and may start at 0it."
                )
            tracker = MetricTracker()
            per_sample_results = []

            total_expr = 0
            for sample in tqdm(dataset, desc=f"Evaluating {dataset_name}:{split}"):
                image = sample["image"]
                if not isinstance(image, Image.Image):
                    if isinstance(image, dict):
                        if image.get("path"):
                            image = Image.open(image["path"])
                        elif image.get("bytes"):
                            image = Image.open(io.BytesIO(image["bytes"]))
                        else:
                            raise TypeError(f"Unsupported image dict format: keys={list(image.keys())}")
                    else:
                        image = Image.open(image)
                    image = image.convert("RGB")
                width, height = image.size

                for expression, inst in iter_expressions(sample):
                    if args.max_samples and total_expr >= args.max_samples:
                        break

                    prompt = build_prompt(expression)
                    pred_start = time.perf_counter()
                    pred_masks = predict_masks(
                        model,
                        processor,
                        vq_sam2,
                        sam2_image_processor,
                        image,
                        prompt,
                        args.max_new_tokens,
                    )
                    pred_end = time.perf_counter()

                    gt_mask = decode_rle(inst.get("mask"), height, width)
                    primary_mask = select_primary_mask(pred_masks)
                    if primary_mask is None:
                        primary_mask = np.zeros_like(gt_mask)
                        pred_masks = []

                    sample_metrics = compute_sample_metrics(primary_mask, gt_mask)
                    set_metrics = compute_set_metrics(pred_masks, gt_mask)
                    sample_metrics.update(set_metrics)
                    tracker.update(sample_metrics)
                    expr_end = time.perf_counter()

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
                                    "gIoU": sample_metrics["gIoU"],
                                    "cIoU": sample_metrics["cIoU"],
                                    "n_acc": sample_metrics["n_acc"],
                                    "pred_count": sample_metrics["pred_count"],
                                    "gt_count": sample_metrics["gt_count"],
                                },
                                "timing": {
                                    "predict_latency_s": pred_end - pred_start,
                                    "expr_latency_s": expr_end - pred_start,
                                },
                            }
                        )

                    total_expr += 1
                    if wandb_run is not None and args.wandb_log_every > 0:
                        if total_expr % args.wandb_log_every == 0:
                            throughput = total_expr / max(time.perf_counter() - run_start, 1e-6)
                            wandb.log(
                                {
                                    "expr_step": total_expr,
                                    "sample/iou": sample_metrics["iou"],
                                    "sample/precision": sample_metrics["precision"],
                                    "sample/recall": sample_metrics["recall"],
                                    "sample/f1": sample_metrics["f1"],
                                    "sample/gIoU": sample_metrics["gIoU"],
                                    "sample/cIoU": sample_metrics["cIoU"],
                                    "sample/n_acc": sample_metrics["n_acc"],
                                    "sample/pred_count": sample_metrics["pred_count"],
                                    "sample/gt_count": sample_metrics["gt_count"],
                                    "sample/dataset": dataset_name,
                                    "sample/split": split,
                                    "sample/image_id": sample.get("image_id"),
                                    "sample/expression": expression,
                                    "perf/predict_latency_s": pred_end - pred_start,
                                    "perf/expr_latency_s": expr_end - pred_start,
                                    "perf/throughput_expr_s": throughput,
                                }
                            )
                if args.max_samples and total_expr >= args.max_samples:
                    break

            if total_expr == 0:
                print(
                    f"Warning: no expressions processed for {dataset_name}:{split}. "
                    "Check the split name or dataset access."
                )
            results.append(
                {
                    "model": args.model,
                    "dataset": dataset_name,
                    "split": split,
                    "metrics": tracker.summary(),
                    "per_sample_results": per_sample_results if args.save_per_sample else None,
                }
            )
            output_path = write_results(args, results)
            if wandb_run is not None:
                summary = tracker.summary()
                wandb.log(
                    {
                        "summary/mIoU": summary["mIoU"],
                        "summary/precision": summary["precision"],
                        "summary/recall": summary["recall"],
                        "summary/f1": summary["f1"],
                        "summary/gIoU": summary["gIoU"],
                        "summary/cIoU": summary["cIoU"],
                        "summary/n_acc": summary["n_acc"],
                        "summary/count": summary["count"],
                        "summary/dataset": dataset_name,
                        "summary/split": split,
                    }
                )

    output_path = write_results(args, results)

    if wandb_run is not None:
        artifact = wandb.Artifact("benchmark-results", type="results")
        artifact.add_file(output_path)
        wandb_run.log_artifact(artifact)
        wandb_run.finish()


if __name__ == "__main__":
    main()
