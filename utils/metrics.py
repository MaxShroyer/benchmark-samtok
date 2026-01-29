from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Optional, Tuple

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


BoxXYXY = Tuple[float, float, float, float]  # (x1, y1, x2, y2) with x2/y2 exclusive


def mask_to_bbox(mask: np.ndarray) -> Optional[BoxXYXY]:
    """
    Convert a boolean mask to an XYXY bbox in pixel coordinates.

    Convention: (x1, y1, x2, y2) where x2/y2 are *exclusive* (half-open box),
    so area = (x2-x1)*(y2-y1). Returns None for empty masks.
    """
    m = mask.astype(bool)
    ys, xs = np.where(m)
    if xs.size == 0 or ys.size == 0:
        return None
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)
    return (x1, y1, x2, y2)


def _box_area_xyxy(box: BoxXYXY) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_intersection_area_xyxy(a: BoxXYXY, b: BoxXYXY) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def box_iou(a: BoxXYXY, b: BoxXYXY) -> float:
    inter = _box_intersection_area_xyxy(a, b)
    union = _box_area_xyxy(a) + _box_area_xyxy(b) - inter
    return _safe_div(inter, union)


def box_giou(a: BoxXYXY, b: BoxXYXY) -> float:
    """
    Generalized IoU for boxes (Rezatofighi et al., 2019).
    """
    iou = box_iou(a, b)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    cx1 = min(ax1, bx1)
    cy1 = min(ay1, by1)
    cx2 = max(ax2, bx2)
    cy2 = max(ay2, by2)
    c_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    if c_area == 0.0:
        return float(iou)
    inter = _box_intersection_area_xyxy(a, b)
    union = _box_area_xyxy(a) + _box_area_xyxy(b) - inter
    return float(iou - _safe_div(c_area - union, c_area))


def box_ciou(a: BoxXYXY, b: BoxXYXY) -> float:
    """
    Complete IoU for boxes (Zheng et al., 2020).
    """
    iou = box_iou(a, b)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    aw = max(0.0, ax2 - ax1)
    ah = max(0.0, ay2 - ay1)
    bw = max(0.0, bx2 - bx1)
    bh = max(0.0, by2 - by1)

    acx = (ax1 + ax2) / 2.0
    acy = (ay1 + ay2) / 2.0
    bcx = (bx1 + bx2) / 2.0
    bcy = (by1 + by2) / 2.0

    rho2 = (acx - bcx) ** 2 + (acy - bcy) ** 2

    cx1 = min(ax1, bx1)
    cy1 = min(ay1, by1)
    cx2 = max(ax2, bx2)
    cy2 = max(ay2, by2)
    c2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2

    eps = 1e-9
    diou_term = _safe_div(rho2, c2 + eps)

    v = (4.0 / (math.pi**2)) * (math.atan2(bw, bh + eps) - math.atan2(aw, ah + eps)) ** 2
    alpha = 0.0 if v == 0.0 else float(v / (1.0 - iou + v + eps))
    return float(iou - diou_term - alpha * v)


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

    pred_box = mask_to_bbox(pred)
    gt_box = mask_to_bbox(gt)
    if pred_box is not None and gt_box is not None:
        box_iou_v = box_iou(pred_box, gt_box)
        box_giou_v = box_giou(pred_box, gt_box)
        box_ciou_v = box_ciou(pred_box, gt_box)
    else:
        box_iou_v = 0.0
        box_giou_v = 0.0
        box_ciou_v = 0.0

    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        # True *box* metrics computed from mask-derived bounding boxes.
        # Keep gIoU/cIoU names for compatibility with existing logs.
        "box_iou": float(box_iou_v),
        "box_giou": float(box_giou_v),
        "box_ciou": float(box_ciou_v),
        "gIoU": float(box_giou_v),
        "cIoU": float(box_ciou_v),
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

    union_iou = _mask_iou(pred_union, gt_union)

    if pred_list and gt_list:
        best_ious = []
        for gt in gt_list:
            best_ious.append(max(_mask_iou(pred, gt) for pred in pred_list))
        best_match_iou = _safe_div(sum(best_ious), max(len(pred_list), len(gt_list)))
    else:
        best_match_iou = 0.0

    pred_count = sum(int(mask.any()) for mask in pred_list)
    gt_count = sum(int(mask.any()) for mask in gt_list)
    n_acc = 1.0 if pred_count == gt_count else 0.0

    return {
        # Legacy set-level mask metrics (previously misnamed gIoU/cIoU).
        "union_iou": float(union_iou),
        "best_match_iou": float(best_match_iou),
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
    total_box_iou: float = 0.0
    total_giou: float = 0.0
    total_ciou: float = 0.0
    total_union_iou: float = 0.0
    total_best_match_iou: float = 0.0
    total_nacc: float = 0.0
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    count: int = 0

    def update(self, sample_metrics: Dict[str, float]) -> None:
        self.total_iou += sample_metrics["iou"]
        self.total_precision += sample_metrics["precision"]
        self.total_recall += sample_metrics["recall"]
        self.total_f1 += sample_metrics["f1"]
        self.total_box_iou += sample_metrics.get("box_iou", 0.0)
        self.total_giou += sample_metrics["gIoU"]
        self.total_ciou += sample_metrics["cIoU"]
        self.total_union_iou += sample_metrics.get("union_iou", 0.0)
        self.total_best_match_iou += sample_metrics.get("best_match_iou", 0.0)
        self.total_nacc += sample_metrics["n_acc"]
        self.total_tp += sample_metrics["tp"]
        self.total_fp += sample_metrics["fp"]
        self.total_fn += sample_metrics["fn"]
        self.count += 1

    def summary(self) -> Dict[str, float]:
        cumulative_iou = _safe_div(
            self.total_tp, self.total_tp + self.total_fp + self.total_fn
        )
        return {
            "mIoU": _safe_div(self.total_iou, self.count),
            "precision": _safe_div(self.total_precision, self.count),
            "recall": _safe_div(self.total_recall, self.count),
            "f1": _safe_div(self.total_f1, self.count),
            "box_iou": _safe_div(self.total_box_iou, self.count),
            "box_giou": _safe_div(self.total_giou, self.count),
            "box_ciou": _safe_div(self.total_ciou, self.count),
            # Keep these keys for existing dashboards/scripts.
            "gIoU": _safe_div(self.total_giou, self.count),
            "cIoU": float(cumulative_iou),
            "union_iou": _safe_div(self.total_union_iou, self.count),
            "best_match_iou": _safe_div(self.total_best_match_iou, self.count),
            "n_acc": _safe_div(self.total_nacc, self.count),
            "count": int(self.count),
        }
