from dataclasses import dataclass
from typing import Dict

import numpy as np


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def compute_sample_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()

    iou = _safe_div(tp, tp + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


@dataclass
class MetricTracker:
    total_iou: float = 0.0
    total_precision: float = 0.0
    total_recall: float = 0.0
    total_f1: float = 0.0
    count: int = 0

    def update(self, sample_metrics: Dict[str, float]) -> None:
        self.total_iou += sample_metrics["iou"]
        self.total_precision += sample_metrics["precision"]
        self.total_recall += sample_metrics["recall"]
        self.total_f1 += sample_metrics["f1"]
        self.count += 1

    def summary(self) -> Dict[str, float]:
        return {
            "mIoU": _safe_div(self.total_iou, self.count),
            "precision": _safe_div(self.total_precision, self.count),
            "recall": _safe_div(self.total_recall, self.count),
            "f1": _safe_div(self.total_f1, self.count),
            "count": int(self.count),
        }
