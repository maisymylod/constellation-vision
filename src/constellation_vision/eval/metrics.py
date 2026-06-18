"""Segmentation metrics, pure-numpy so they are testable without torch.

The headline is **mean IoU** over the defect classes (background excluded from
the mean, since a background-only model would otherwise score deceptively well).
We also report per-class IoU and a confusion matrix for the eval report.

IoU for a class c is  TP / (TP + FP + FN)  computed over all pixels of the split.
A class absent from both prediction and ground truth contributes no denominator
and is reported as None (skipped from the mean), which is the standard convention
and keeps the mean honest when a rare class never appears in a split.
"""
from __future__ import annotations

import numpy as np

from .. import schema


def confusion_matrix(pred: np.ndarray, true: np.ndarray, num_classes: int = schema.NUM_CLASSES) -> np.ndarray:
    """Rows = ground truth, cols = prediction. Counts over all pixels."""
    pred = pred.reshape(-1).astype(np.int64)
    true = true.reshape(-1).astype(np.int64)
    k = (true >= 0) & (true < num_classes)
    idx = num_classes * true[k] + pred[k]
    cm = np.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def per_class_iou(cm: np.ndarray) -> list[float | None]:
    """IoU per class from a confusion matrix. None where the class is absent."""
    ious: list[float | None] = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        ious.append(float(tp / denom) if denom > 0 else None)
    return ious


def mean_iou(cm: np.ndarray, include_background: bool = False) -> float:
    """Mean IoU over classes that appear in the split. Background excluded by
    default so the headline reflects defect-segmentation quality."""
    ious = per_class_iou(cm)
    start = 0 if include_background else 1
    vals = [v for v in ious[start:] if v is not None]
    return float(np.mean(vals)) if vals else 0.0


def pixel_accuracy(cm: np.ndarray) -> float:
    total = cm.sum()
    return float(np.trace(cm) / total) if total > 0 else 0.0


def summarize(pred: np.ndarray, true: np.ndarray) -> dict:
    """Full metric bundle for an evaluated split."""
    cm = confusion_matrix(pred, true)
    ious = per_class_iou(cm)
    per_class = {schema.INDEX_TO_NAME[i]: ious[i] for i in range(schema.NUM_CLASSES)}
    return {
        "mean_iou": mean_iou(cm),
        "mean_iou_with_background": mean_iou(cm, include_background=True),
        "pixel_accuracy": pixel_accuracy(cm),
        "per_class_iou": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": list(schema.CLASS_NAMES),
    }
