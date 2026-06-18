"""Tests for the eval report renderer."""
from __future__ import annotations

from constellation_vision import schema
from constellation_vision.eval import report


def _metrics():
    return {
        "mean_iou": 0.7321,
        "mean_iou_with_background": 0.8012,
        "pixel_accuracy": 0.9510,
        "per_class_iou": {n: (0.7 if n != schema.BACKGROUND else 0.99) for n in schema.CLASS_NAMES},
        "confusion_matrix": [[1] * schema.NUM_CLASSES for _ in range(schema.NUM_CLASSES)],
        "class_names": list(schema.CLASS_NAMES),
    }


def _meta():
    return {
        "n_train": 480, "n_test": 160, "frame_size": 96, "seed": 42,
        "epochs": 12, "batch_size": 16, "lr": 1e-3, "num_parameters": 123456,
        "generated_at": "2026-06-18T00:00:00Z",
    }


def test_render_contains_headline_and_classes():
    md = report.render_markdown(_metrics(), _meta())
    assert "Mean IoU (defect classes)" in md
    assert "0.7321" in md
    for name in schema.CLASS_NAMES:
        assert name in md
    assert "Confusion matrix" in md
    # No em dashes anywhere in generated docs.
    assert "—" not in md
