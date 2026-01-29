# W&B run report: ljkbcg2y

- name: gentle-sea-11
- state: running
- url: https://wandb.ai/maxshroyer49-na/samtok-benchmark/runs/ljkbcg2y
- processed expr (max expr_step): 0

## Config (compact)

```json
{
  "model": "zhouyik/Qwen3-VL-8B-SAMTok",
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

- summary/cIoU: 0.7109649528553494
- summary/count: 5598
- summary/dataset: moondream/refcoco-m
- summary/f1: 0.7396282296685235
- summary/gIoU: 0.7109649528553494
- summary/mIoU: 0.7109649528553494
- summary/n_acc: 0.8208288674526617
- summary/precision: 0.7605731418504927
- summary/recall: 0.734023627155952
- summary/split: validation

## Outputs

- history CSV: `history_sample_perf.csv`
- report JSON: `report.json`
