# W&B run report: ux1urfnk

- name: glorious-vortex-12
- state: running
- url: https://wandb.ai/maxshroyer49-na/samtok-benchmark/runs/ux1urfnk
- processed expr (max expr_step): 0

## Config (compact)

```json
{
  "model": "zhouyik/Qwen3-VL-4B-SAMTok",
  "datasets": [
    "moondream/refcoco-m",
    "moondream/refcoco_rle",
    "moondream/refcoco_plus_rle_val",
    "moondream/lvis_segmentation",
    "lmms-lab/RefCOCOplus"
  ],
  "split_arg": null,
  "max_samples": 0,
  "max_new_tokens": 256,
  "save_per_sample": false
}
```

## Sample metrics (from streamed history)

```json
{
  "sample/iou": {
    "mean": null,
    "p50": null,
    "p90": null,
    "p95": null
  },
  "sample/f1": {
    "mean": null,
    "p50": null,
    "p90": null,
    "p95": null
  },
  "perf/predict_latency_s": {
    "mean": null,
    "p50": null,
    "p90": null,
    "p95": null
  },
  "perf/expr_latency_s": {
    "mean": null,
    "p50": null,
    "p90": null,
    "p95": null
  }
}
```

## Dataset/split counts (from streamed history)


## Logged split summaries (W&B summary/*)

- summary/cIoU: 0.6268458828341418
- summary/count: 5598
- summary/dataset: moondream/refcoco-m
- summary/f1: 0.6557242131996223
- summary/gIoU: 0.6268458828341418
- summary/mIoU: 0.6268458828341418
- summary/n_acc: 0.9841014648088604
- summary/precision: 0.7005480290564317
- summary/recall: 0.6462919735104425
- summary/split: validation

## Outputs

- history CSV: `history_sample_perf.csv`
- report JSON: `report.json`
