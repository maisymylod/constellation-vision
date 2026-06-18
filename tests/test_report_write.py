"""Tests for the report write path and committed-metric loader."""
from __future__ import annotations

import json

from constellation_vision import config, schema
from constellation_vision.eval import report


def _metrics():
    return {
        "mean_iou": 0.66,
        "mean_iou_with_background": 0.72,
        "pixel_accuracy": 0.93,
        "per_class_iou": {n: 0.6 for n in schema.CLASS_NAMES},
        "confusion_matrix": [[1] * schema.NUM_CLASSES for _ in range(schema.NUM_CLASSES)],
        "class_names": list(schema.CLASS_NAMES),
    }


def _meta():
    return {"n_train": 10, "n_test": 5, "frame_size": 96, "seed": 42,
            "epochs": 1, "batch_size": 4, "lr": 1e-3, "num_parameters": 100}


def test_write_reports_creates_files(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(config, "EVAL_MD", tmp_path / "EVAL.md")
    monkeypatch.setattr(config, "METRICS_JSON", tmp_path / "metrics.json")
    monkeypatch.setattr(config, "HISTORY", tmp_path / "history.jsonl")

    report.write_reports(_metrics(), _meta(), append_history=True)
    assert (tmp_path / "EVAL.md").exists()
    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert payload["mean_iou"] == 0.66
    assert (tmp_path / "history.jsonl").read_text().count("\n") == 1

    # load_committed_mean_iou reads it back.
    assert report.load_committed_mean_iou() == 0.66


def test_load_committed_returns_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "METRICS_JSON", tmp_path / "missing.json")
    assert report.load_committed_mean_iou() is None
