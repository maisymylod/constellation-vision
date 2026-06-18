"""Calibration tests for the segmentation metrics.

These are the quality gate's correctness anchors: a perfect prediction must score
mean IoU 1.0, and a degenerate background-only prediction must score 0.0 on the
defect classes. That the metric reads 1.0 only for a correct mask and 0.0 for a
trivial one is the proof it measures segmentation quality, not bookkeeping.
"""
from __future__ import annotations

import numpy as np

from constellation_vision import schema
from constellation_vision.eval import metrics as M


def _frame_with_all_classes():
    m = np.zeros((10, 10), np.int64)
    m[0:2, 0:2] = schema.NAME_TO_INDEX[schema.CRACK]
    m[0:2, 5:7] = schema.NAME_TO_INDEX[schema.HOT_SPOT]
    m[5:7, 0:2] = schema.NAME_TO_INDEX[schema.DELAMINATION]
    m[5:7, 5:7] = schema.NAME_TO_INDEX[schema.MISSING_FASTENER]
    return m


def test_perfect_prediction_scores_one():
    true = _frame_with_all_classes()
    out = M.summarize(true.copy(), true)
    assert out["mean_iou"] == 1.0
    assert out["pixel_accuracy"] == 1.0
    for name, v in out["per_class_iou"].items():
        assert v == 1.0


def test_background_only_prediction_scores_zero_on_defects():
    true = _frame_with_all_classes()
    pred = np.zeros_like(true)  # predict everything as background
    out = M.summarize(pred, true)
    assert out["mean_iou"] == 0.0
    # Background IoU is non-trivial but excluded from the headline mean.
    assert out["per_class_iou"][schema.BACKGROUND] is not None


def test_absent_class_is_skipped_from_mean():
    # Only crack present in both pred and true; other defect classes absent.
    true = np.zeros((8, 8), np.int64)
    true[0:2, 0:2] = schema.NAME_TO_INDEX[schema.CRACK]
    out = M.summarize(true.copy(), true)
    assert out["per_class_iou"][schema.CRACK] == 1.0
    assert out["per_class_iou"][schema.HOT_SPOT] is None
    assert out["mean_iou"] == 1.0  # mean over present defect classes only


def test_partial_overlap_iou_value():
    # 2x2 gt crack, predicted shifted by one column -> overlap 2, union 6.
    true = np.zeros((6, 6), np.int64)
    pred = np.zeros((6, 6), np.int64)
    c = schema.NAME_TO_INDEX[schema.CRACK]
    true[1:3, 1:3] = c
    pred[1:3, 2:4] = c
    out = M.summarize(pred, true)
    assert abs(out["per_class_iou"][schema.CRACK] - (2 / 6)) < 1e-9


def test_confusion_matrix_shape_and_total():
    true = _frame_with_all_classes()
    cm = np.array(M.summarize(true.copy(), true)["confusion_matrix"])
    assert cm.shape == (schema.NUM_CLASSES, schema.NUM_CLASSES)
    assert cm.sum() == true.size
