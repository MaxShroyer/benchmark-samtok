"""
Download + summarize SAMTok benchmark metrics from Weights & Biases.

This repo logs:
- step:            expr_step
- per-expression:  sample/*, perf/*
- per-split:       summary/*

Example:
  python analyze_wandb_runs.py \
    --entity maxshroyer49-na \
    --project samtok-benchmark \
    --runs ljkbcg2y,ux1urfnk \
    --outdir wandb_reports
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _safe_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    try:
        s = str(x)
        return s
    except Exception:
        return None


@dataclass
class RunningStats:
    values: List[float]

    def __init__(self) -> None:
        self.values = []

    def add(self, v: Optional[float]) -> None:
        if v is None:
            return
        if not np.isfinite(v):
            return
        self.values.append(float(v))

    def summary(self) -> Dict[str, Optional[float]]:
        if not self.values:
            return {"mean": None, "p50": None, "p90": None, "p95": None}
        arr = np.asarray(self.values, dtype=float)
        return {
            "mean": float(arr.mean()),
            "p50": float(np.quantile(arr, 0.50)),
            "p90": float(np.quantile(arr, 0.90)),
            "p95": float(np.quantile(arr, 0.95)),
        }


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _artifact_download_best_effort(run, outdir: str) -> List[str]:
    """
    Try to download any `benchmark-results` artifacts logged by this benchmark.
    Returns a list of downloaded file paths (may be empty).
    """
    downloaded: List[str] = []
    try:
        artifacts = list(run.logged_artifacts())
    except Exception:
        return downloaded

    for art in artifacts:
        try:
            if getattr(art, "type", None) != "results":
                continue
            name = getattr(art, "name", "") or ""
            if not name.startswith("benchmark-results"):
                continue
            target = os.path.join(outdir, "artifacts", name.replace(":", "_"))
            _ensure_dir(target)
            art_dir = art.download(root=target)
            # Collect json files in the artifact dir.
            for root, _, files in os.walk(art_dir):
                for fn in files:
                    if fn.endswith(".json"):
                        downloaded.append(os.path.join(root, fn))
        except Exception:
            continue

    return downloaded


def _iter_history(run, keys: List[str]) -> Iterable[Dict[str, Any]]:
    # scan_history avoids forcing pandas as a hard dependency.
    for row in run.scan_history(keys=keys, page_size=1000):
        if not isinstance(row, dict):
            continue
        yield row


def _pick(d: Dict[str, Any], key: str) -> Any:
    # wandb sometimes uses keys with slashes; keep literal keys.
    return d.get(key)


def _summarize_run(run, outdir: str) -> Dict[str, Any]:
    run_dir = os.path.join(outdir, run.id)
    _ensure_dir(run_dir)

    # Basic run metadata
    meta = {
        "id": run.id,
        "name": getattr(run, "name", None),
        "state": getattr(run, "state", None),
        "url": getattr(run, "url", None),
        "created_at": str(getattr(run, "created_at", "")),
    }

    config = dict(getattr(run, "config", {}) or {})
    # W&B adds internal keys; keep a small subset + the originals as-is in file output.
    config_compact = {
        k: config.get(k)
        for k in ["model", "datasets", "split_arg", "max_samples", "max_new_tokens", "save_per_sample"]
        if k in config
    }

    summary = dict(getattr(run, "summary", {}) or {})

    # Keys we care about from streaming logs
    hist_keys = [
        "expr_step",
        "sample/iou",
        "sample/precision",
        "sample/recall",
        "sample/f1",
        "sample/box_iou",
        "sample/box_giou",
        "sample/box_ciou",
        "sample/gIoU",
        "sample/cIoU",
        "sample/union_iou",
        "sample/best_match_iou",
        "sample/n_acc",
        "sample/pred_count",
        "sample/gt_count",
        "sample/dataset",
        "sample/split",
        "perf/predict_latency_s",
        "perf/expr_latency_s",
        "perf/throughput_expr_s",
        # per-split summaries (last write wins in W&B summary; but also shows up in history)
        "summary/mIoU",
        "summary/precision",
        "summary/recall",
        "summary/f1",
        "summary/box_iou",
        "summary/box_giou",
        "summary/box_ciou",
        "summary/gIoU",
        "summary/cIoU",
        "summary/union_iou",
        "summary/best_match_iou",
        "summary/n_acc",
        "summary/count",
        "summary/dataset",
        "summary/split",
    ]

    # Aggregate streaming stats
    max_step = 0
    dataset_split_counts: Dict[Tuple[str, str], int] = {}
    iou_stats = RunningStats()
    f1_stats = RunningStats()
    box_iou_stats = RunningStats()
    box_giou_stats = RunningStats()
    box_ciou_stats = RunningStats()
    union_iou_stats = RunningStats()
    best_match_iou_stats = RunningStats()
    pred_latency = RunningStats()
    expr_latency = RunningStats()
    throughput_last: Optional[float] = None
    throughput_ts: List[Tuple[int, float]] = []

    # Also write a compact CSV (step-level) for later plotting
    csv_path = os.path.join(run_dir, "history_sample_perf.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "expr_step",
                "sample/dataset",
                "sample/split",
                "sample/iou",
                "sample/f1",
                "sample/box_iou",
                "sample/gIoU",
                "sample/cIoU",
                "sample/union_iou",
                "sample/best_match_iou",
                "perf/predict_latency_s",
                "perf/expr_latency_s",
                "perf/throughput_expr_s",
            ]
        )
        for row in _iter_history(run, keys=hist_keys):
            step = int(_pick(row, "expr_step") or 0)
            if step > max_step:
                max_step = step

            ds = _safe_str(_pick(row, "sample/dataset")) or ""
            sp = _safe_str(_pick(row, "sample/split")) or ""
            if ds or sp:
                dataset_split_counts[(ds, sp)] = dataset_split_counts.get((ds, sp), 0) + 1

            iou = _safe_float(_pick(row, "sample/iou"))
            f1 = _safe_float(_pick(row, "sample/f1"))
            box_iou = _safe_float(_pick(row, "sample/box_iou"))
            giou = _safe_float(_pick(row, "sample/gIoU"))
            ciou = _safe_float(_pick(row, "sample/cIoU"))
            union_iou = _safe_float(_pick(row, "sample/union_iou"))
            best_match_iou = _safe_float(_pick(row, "sample/best_match_iou"))
            pl = _safe_float(_pick(row, "perf/predict_latency_s"))
            el = _safe_float(_pick(row, "perf/expr_latency_s"))
            thr = _safe_float(_pick(row, "perf/throughput_expr_s"))

            iou_stats.add(iou)
            f1_stats.add(f1)
            box_iou_stats.add(box_iou)
            box_giou_stats.add(_safe_float(_pick(row, "sample/box_giou")) or giou)
            box_ciou_stats.add(_safe_float(_pick(row, "sample/box_ciou")) or ciou)
            union_iou_stats.add(union_iou)
            best_match_iou_stats.add(best_match_iou)
            pred_latency.add(pl)
            expr_latency.add(el)
            if thr is not None:
                throughput_last = thr
                throughput_ts.append((step, thr))

            # Write row if it looks like a sample step (has step + any metric).
            if step and (
                iou is not None
                or f1 is not None
                or box_iou is not None
                or giou is not None
                or ciou is not None
                or union_iou is not None
                or best_match_iou is not None
                or pl is not None
                or el is not None
                or thr is not None
            ):
                w.writerow(
                    [
                        step,
                        ds,
                        sp,
                        iou,
                        f1,
                        box_iou,
                        giou,
                        ciou,
                        union_iou,
                        best_match_iou,
                        pl,
                        el,
                        thr,
                    ]
                )

    # Download benchmark-results artifact (if present)
    artifact_jsons = _artifact_download_best_effort(run, run_dir)

    # Build report object
    report: Dict[str, Any] = {
        "meta": meta,
        "config_compact": config_compact,
        "processed_expr_step_max": int(max_step),
        "throughput_last_expr_s": throughput_last,
        "sample_iou": iou_stats.summary(),
        "sample_f1": f1_stats.summary(),
        "sample_box_iou": box_iou_stats.summary(),
        "sample_box_giou": box_giou_stats.summary(),
        "sample_box_ciou": box_ciou_stats.summary(),
        "sample_union_iou": union_iou_stats.summary(),
        "sample_best_match_iou": best_match_iou_stats.summary(),
        "predict_latency_s": pred_latency.summary(),
        "expr_latency_s": expr_latency.summary(),
        "dataset_split_counts": {
            f"{ds}::{sp}": int(n) for (ds, sp), n in sorted(dataset_split_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        },
        "wandb_summary_keys": sorted(list(summary.keys())),
        "wandb_summary_selected": {k: summary.get(k) for k in sorted(summary.keys()) if k.startswith("summary/")},
        "downloaded_artifact_jsons": artifact_jsons,
        "files": {"history_csv": csv_path},
    }

    # Save JSON report
    with open(os.path.join(run_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save a friendly markdown summary too
    md_path = os.path.join(run_dir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# W&B run report: {run.id}\n\n")
        f.write(f"- name: {meta.get('name')}\n")
        f.write(f"- state: {meta.get('state')}\n")
        f.write(f"- url: {meta.get('url')}\n")
        f.write(f"- processed expr (max expr_step): {max_step}\n")
        if throughput_last is not None:
            f.write(f"- last throughput (expr/s): {throughput_last:.4f}\n")
        f.write("\n## Config (compact)\n\n")
        f.write("```json\n")
        f.write(json.dumps(config_compact, indent=2))
        f.write("\n```\n")
        f.write("\n## Sample metrics (from streamed history)\n\n")
        f.write("```json\n")
        f.write(
            json.dumps(
                {
                    "sample/iou": iou_stats.summary(),
                    "sample/f1": f1_stats.summary(),
                    "sample/box_iou": box_iou_stats.summary(),
                    "sample/gIoU": box_giou_stats.summary(),
                    "sample/cIoU": box_ciou_stats.summary(),
                    "sample/union_iou": union_iou_stats.summary(),
                    "sample/best_match_iou": best_match_iou_stats.summary(),
                    "perf/predict_latency_s": pred_latency.summary(),
                    "perf/expr_latency_s": expr_latency.summary(),
                },
                indent=2,
            )
        )
        f.write("\n```\n")
        f.write("\n## Dataset/split counts (from streamed history)\n\n")
        for k, n in report["dataset_split_counts"].items():
            f.write(f"- {k}: {n}\n")
        f.write("\n## Logged split summaries (W&B summary/*)\n\n")
        for k, v in report["wandb_summary_selected"].items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Outputs\n\n")
        f.write(f"- history CSV: `{os.path.basename(csv_path)}`\n")
        f.write(f"- report JSON: `report.json`\n")
        if artifact_jsons:
            f.write("- downloaded artifacts:\n")
            for p in artifact_jsons:
                f.write(f"  - `{os.path.relpath(p, run_dir)}`\n")

    report["files"]["report_md"] = md_path
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--runs", required=True, help="Comma-separated run IDs, e.g. ljkbcg2y,ux1urfnk")
    ap.add_argument("--outdir", default="wandb_reports")
    args = ap.parse_args()

    try:
        import wandb
    except Exception as exc:
        raise SystemExit(f"wandb import failed: {type(exc).__name__}: {exc}")

    _ensure_dir(args.outdir)
    api = wandb.Api()

    run_ids = [r.strip() for r in args.runs.split(",") if r.strip()]
    reports: List[Dict[str, Any]] = []
    for run_id in run_ids:
        path = f"{args.entity}/{args.project}/{run_id}"
        run = api.run(path)
        report = _summarize_run(run, args.outdir)
        reports.append(report)

    # Write combined index
    index_path = os.path.join(args.outdir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)

    # Console summary
    for rep in reports:
        meta = rep["meta"]
        print(f"\n=== {meta.get('id')} ({meta.get('name')}) ===")
        print(f"url: {meta.get('url')}")
        print(f"processed expr (max expr_step): {rep.get('processed_expr_step_max')}")
        if rep.get("throughput_last_expr_s") is not None:
            print(f"last throughput (expr/s): {rep['throughput_last_expr_s']:.4f}")
        print(f"sample/iou: {rep['sample_iou']}")
        print(f"sample/f1:  {rep['sample_f1']}")
        print(f"predict latency (s): {rep['predict_latency_s']}")
        print(f"expr latency (s):    {rep['expr_latency_s']}")
        # show top 5 dataset/split counts
        counts = list(rep.get("dataset_split_counts", {}).items())[:5]
        if counts:
            print("top dataset/split counts:")
            for k, n in counts:
                print(f"  - {k}: {n}")

    print(f"\nWrote reports to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()

