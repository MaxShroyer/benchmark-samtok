from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    return _safe_div(tp, tp + fp + fn)


def _normalize_mask_list(masks: Iterable[np.ndarray]) -> List[np.ndarray]:
    normalized = []
    for mask in masks:
        if mask is None:
            continue
        normalized.append(mask.astype(bool))
    return normalized


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


def compute_set_metrics(
    pred_masks: Iterable[np.ndarray], gt_mask: np.ndarray
) -> Dict[str, float]:
    pred_list = _normalize_mask_list(pred_masks)
    gt_list = _normalize_mask_list([gt_mask])

    if pred_list:
        pred_union = np.logical_or.reduce(pred_list)
    else:
        pred_union = np.zeros_like(gt_mask, dtype=bool)

    if gt_list:
        gt_union = np.logical_or.reduce(gt_list)
    else:
        gt_union = np.zeros_like(gt_mask, dtype=bool)

    g_iou = _mask_iou(pred_union, gt_union)

    if pred_list and gt_list:
        best_ious = []
        for gt in gt_list:
            best_ious.append(max(_mask_iou(pred, gt) for pred in pred_list))
        c_iou = _safe_div(sum(best_ious), max(len(pred_list), len(gt_list)))
    else:
        c_iou = 0.0

    pred_count = sum(int(mask.any()) for mask in pred_list)
    gt_count = sum(int(mask.any()) for mask in gt_list)
    n_acc = 1.0 if pred_count == gt_count else 0.0

    return {
        "gIoU": float(g_iou),
        "cIoU": float(c_iou),
        "n_acc": float(n_acc),
        "pred_count": int(pred_count),
        "gt_count": int(gt_count),
    }


@dataclass
class MetricTracker:
    total_iou: float = 0.0
    total_precision: float = 0.0
    total_recall: float = 0.0
    total_f1: float = 0.0
    total_giou: float = 0.0
    total_ciou: float = 0.0
    total_nacc: float = 0.0
    count: int = 0

    def update(self, sample_metrics: Dict[str, float]) -> None:
        self.total_iou += sample_metrics["iou"]
        self.total_precision += sample_metrics["precision"]
        self.total_recall += sample_metrics["recall"]
        self.total_f1 += sample_metrics["f1"]
        self.total_giou += sample_metrics["gIoU"]
        self.total_ciou += sample_metrics["cIoU"]
        self.total_nacc += sample_metrics["n_acc"]
        self.count += 1

    def summary(self) -> Dict[str, float]:
        return {
            "mIoU": _safe_div(self.total_iou, self.count),
            "precision": _safe_div(self.total_precision, self.count),
            "recall": _safe_div(self.total_recall, self.count),
            "f1": _safe_div(self.total_f1, self.count),
            "gIoU": _safe_div(self.total_giou, self.count),
            "cIoU": _safe_div(self.total_ciou, self.count),
            "n_acc": _safe_div(self.total_nacc, self.count),
            "count": int(self.count),
        }
