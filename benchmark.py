import argparse
import io
import json
import os
import re
import sys
import time
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


def _extract_mt_codes_from_token_ids(tokenizer, token_ids: List[int]) -> List[int]:
    """
    Extract SAMTok mt codes directly from generated token IDs.

    This is more robust than text decoding because:
    - Some tokenizers may treat mt tokens as "special" and strip them in decode.
    - Some decoders may alter spacing/formatting around the raw token strings.
    """
    # Convert to token strings (no decoding heuristics).
    if tokenizer is None:
        return []
    try:
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
    except Exception:
        return []

    # Accept both zero-padded (<|mt_0000|>) and non-padded (<|mt_1|>) forms.
    pattern = re.compile(r"^<\|mt_(\d{1,4})\|>$")
    out: List[int] = []
    for tok in tokens:
        m = pattern.match(tok)
        if m:
            out.append(int(m.group(1)))
    return out


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
    debug = os.getenv("SAMTOK_DEBUG_MODEL", "").strip().lower() in {"1", "true", "yes", "y"}

    def _debug(msg: str) -> None:
        if debug:
            # Use stdout + flush so it shows up even with progress bars.
            print(f"[samtok-benchmark] {msg}", flush=True)

    try:
        import transformers
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "transformers is required for model loading. "
            "Install with `pip install -U \"transformers>=4.49.0,<5.0.0\"`."
        ) from exc

    model_path_l = model_path.lower()
    if "qwen3" in model_path_l:
        from transformers import Qwen3VLForConditionalGeneration

        model_cls = Qwen3VLForConditionalGeneration
    elif "qwen2.5" in model_path_l or "qwen25" in model_path_l:
        model_cls = None
        # Keep a few import fallbacks for older Transformers builds.
        try:
            from transformers import Qwen2_5VLForConditionalGeneration

            model_cls = Qwen2_5VLForConditionalGeneration
        except Exception as exc:
            _debug(f"Qwen2_5VLForConditionalGeneration import failed: {type(exc).__name__}: {exc}")
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore

                model_cls = Qwen2_5_VLForConditionalGeneration
            except Exception as exc2:
                _debug(f"Qwen2_5_VLForConditionalGeneration import failed: {type(exc2).__name__}: {exc2}")
                try:
                    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (  # type: ignore
                        Qwen2_5_VLForConditionalGeneration,
                    )

                    model_cls = Qwen2_5_VLForConditionalGeneration
                except Exception as exc3:
                    _debug(
                        "Direct import from transformers.models.qwen2_5_vl failed: "
                        f"{type(exc3).__name__}: {exc3}"
                    )
        if model_cls is None:
            raise RuntimeError(
                "Could not import a Qwen2.5-VL model class from Transformers.\n"
                f"- transformers={getattr(transformers, '__version__', 'unknown')}\n"
                "Fix:\n"
                "- Upgrade Transformers: `pip install -U \"transformers>=4.49.0,<5.0.0\"`"
            )
    else:
        raise ValueError(f"Unknown model family for {model_path}")

    # Prefer official implementations (no remote code) but allow it as a fallback.
    last_exc: Exception | None = None
    for trust_remote_code in (False, True):
        for ignore_mismatched_sizes in (False, True):
            try:
                _debug(
                    f"loading {model_cls.__name__} "
                    f"(trust_remote_code={trust_remote_code}, ignore_mismatched_sizes={ignore_mismatched_sizes})"
                )
                model = model_cls.from_pretrained(
                    model_path,
                    torch_dtype="auto",
                    trust_remote_code=trust_remote_code,
                    ignore_mismatched_sizes=ignore_mismatched_sizes,
                )
                if hasattr(model, "generate"):
                    return model
                last_exc = RuntimeError(
                    f"Loaded {type(model).__name__} but it has no `.generate`."
                )
                _debug(str(last_exc))
            except Exception as exc:
                last_exc = exc
                _debug(
                    f"{model_cls.__name__} from_pretrained failed "
                    f"(trust_remote_code={trust_remote_code}, ignore_mismatched_sizes={ignore_mismatched_sizes}): "
                    f"{type(exc).__name__}: {exc}"
                )

    assert last_exc is not None
    raise RuntimeError(
        "Could not load a model with `.generate`.\n"
        f"- transformers={getattr(transformers, '__version__', 'unknown')}\n"
        f"- model_class={getattr(model_cls, '__name__', None)}\n"
        "Debug:\n"
        "- Re-run with `SAMTOK_DEBUG_MODEL=1` to print loader attempts.\n"
        "Fix:\n"
        "- Install official Transformers: `pip install -U \"transformers>=4.49.0,<5.0.0\"`\n"
        "- If your `transformers.__version__` looks unusual, uninstall/reinstall:\n"
        "  `pip uninstall -y transformers && pip install -U \"transformers>=4.49.0,<5.0.0\"`"
    ) from last_exc


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
    except Exception as exc:
        raise RuntimeError(
            "Could not load a processor for this model.\n"
            f"- model={model_path}\n"
            f"- kwargs={base_kwargs}"
        ) from exc


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
            "validation": "val",
            "valid": "val",
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
    def resolve_text(payload: Any) -> str | None:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            for key in ("sent", "sentence", "text", "expression", "expr", "caption", "query"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        return None

    def ensure_mask(inst: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
        if inst.get("mask") is not None:
            return inst
        for key in ("mask", "segmentation", "segmentation_rle", "rle"):
            value = inst.get(key)
            if value is None:
                value = fallback.get(key)
            if value is not None:
                merged = dict(inst)
                merged["mask"] = value
                return merged
        return inst

    def collect_from_container(container: Dict[str, Any], fallback: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        entries: List[Tuple[str, Dict[str, Any]]] = []
        sentences = (
            container.get("sentences")
            or container.get("sentence")
            or container.get("expressions")
            or container.get("expression")
            or container.get("refexp")
        )
        if isinstance(sentences, str):
            entries.append((sentences, ensure_mask(container, fallback)))
            return entries
        if isinstance(sentences, list):
            for item in sentences:
                if isinstance(item, str):
                    entries.append((item, ensure_mask(container, fallback)))
                    continue
                text = resolve_text(item)
                if text is None:
                    continue
                inst = ensure_mask({**container, **item} if isinstance(item, dict) else container, fallback)
                entries.append((text, inst))
        return entries

    output: List[Tuple[str, Dict[str, Any]]] = []

    # lmms-lab/RefCOCOplus schema (lmms-eval formatted):
    # - question: str (generic prompt)
    # - answer: List[str] (referring expressions)
    # - segmentation: polygon coords (COCO-style)
    answers = sample.get("answer")
    if isinstance(answers, str) and answers.strip():
        answers = [answers]
    if isinstance(answers, list) and any(isinstance(a, str) and a.strip() for a in answers):
        inst = ensure_mask({"mask": sample.get("segmentation")}, sample)
        for a in answers:
            if isinstance(a, str) and a.strip():
                output.append((a.strip(), inst))
        if output:
            return output

    samples = sample.get("samples")
    if isinstance(samples, list) and samples:
        for inst in samples:
            if isinstance(inst, dict):
                output.extend(collect_from_container(inst, inst))
        return output

    output.extend(collect_from_container(sample, sample))

    refs = sample.get("refs") or sample.get("ref")
    if isinstance(refs, dict):
        output.extend(collect_from_container(refs, refs))
    elif isinstance(refs, list):
        for ref in refs:
            if isinstance(ref, dict):
                output.extend(collect_from_container(ref, ref))

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
    # Build a single chat-formatted string, then let the processor encode multimodal inputs.
    # This is the most compatible path across Qwen2.5-VL and Qwen3-VL processors.
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(model.device)
    debug_decode = os.getenv("SAMTOK_DEBUG_DECODE", "").strip().lower() in {"1", "true", "yes", "y"}

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        top_p=1.0,
    )
    # Trim the prompt tokens to keep only new tokens.
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode both ways: some tokenizers may (mis)classify mt tokens as special.
    output_text_clean = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    output_text_raw = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0]

    # First try parsing from decoded text, then fall back to token-id based extraction.
    quant_codes = parse_quant_codes(output_text_clean, CODEBOOK_SIZE, CODEBOOK_DEPTH)
    if not quant_codes and output_text_raw != output_text_clean:
        quant_codes = parse_quant_codes(output_text_raw, CODEBOOK_SIZE, CODEBOOK_DEPTH)
    if not quant_codes:
        # Token-id based recovery: convert mt token strings directly from generated IDs.
        mt_ids = _extract_mt_codes_from_token_ids(
            getattr(processor, "tokenizer", None),
            generated_ids_trimmed[0].tolist(),
        )
        if mt_ids:
            # Reconstruct the "text-like" form expected by parse_quant_codes.
            # (We keep the same remapping logic centralized in parse_quant_codes.)
            mt_text = "".join(f"<|mt_{i:04d}|>" for i in mt_ids)
            quant_codes = parse_quant_codes(mt_text, CODEBOOK_SIZE, CODEBOOK_DEPTH)
        elif debug_decode:
            # Helpful when a model returns "mask:N" instead of mt tokens.
            print(
                "[samtok-benchmark] decode debug:"
                f"\n- inputs_keys={sorted(list(inputs.keys()))}"
                f"\n- output_text_clean={output_text_clean!r}"
                f"\n- output_text_raw={output_text_raw!r}",
                flush=True,
            )
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
    if not hasattr(model, "generate"):
        raise RuntimeError(
            f"Loaded model {type(model).__name__} without `generate`. "
            "Set SAMTOK_DEBUG_MODEL=1 to see loader attempts."
        )
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
                                    # Box metrics computed from mask-derived bboxes.
                                    "box_iou": sample_metrics.get("box_iou", 0.0),
                                    "box_giou": sample_metrics.get("box_giou", sample_metrics["gIoU"]),
                                    "box_ciou": sample_metrics.get("box_ciou", sample_metrics["cIoU"]),
                                    "gIoU": sample_metrics["gIoU"],
                                    "cIoU": sample_metrics["cIoU"],
                                    # Legacy set-level mask metrics (previously misnamed gIoU/cIoU).
                                    "union_iou": sample_metrics.get("union_iou", 0.0),
                                    "best_match_iou": sample_metrics.get("best_match_iou", 0.0),
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
                                    "sample/box_iou": sample_metrics.get("box_iou", 0.0),
                                    "sample/box_giou": sample_metrics.get("box_giou", sample_metrics["gIoU"]),
                                    "sample/box_ciou": sample_metrics.get("box_ciou", sample_metrics["cIoU"]),
                                    "sample/gIoU": sample_metrics["gIoU"],
                                    "sample/cIoU": sample_metrics["cIoU"],
                                    "sample/union_iou": sample_metrics.get("union_iou", 0.0),
                                    "sample/best_match_iou": sample_metrics.get("best_match_iou", 0.0),
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
                        "summary/box_iou": summary.get("box_iou", 0.0),
                        "summary/box_giou": summary.get("box_giou", summary["gIoU"]),
                        "summary/box_ciou": summary.get("box_ciou", summary["cIoU"]),
                        "summary/gIoU": summary["gIoU"],
                        "summary/cIoU": summary["cIoU"],
                        "summary/union_iou": summary.get("union_iou", 0.0),
                        "summary/best_match_iou": summary.get("best_match_iou", 0.0),
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
