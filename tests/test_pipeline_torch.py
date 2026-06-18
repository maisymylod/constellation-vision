"""End-to-end pipeline test: train tiny, export ONNX, evaluate, enrich.

Skipped automatically if torch/onnx are not installed (the core CI job runs
without the ML stack); the train CI job and `make train` exercise it for real.
This keeps a single check that the model builds, learns something above chance,
round-trips through ONNX, and feeds the argus enrichment step.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")


def test_model_builds_and_shapes():
    from constellation_vision import config, schema
    from constellation_vision.model.unet import UNetLite

    model = UNetLite()
    x = torch.zeros(2, config.IN_CHANNELS, config.FRAME_SIZE, config.FRAME_SIZE)
    out = model(x)
    assert out.shape == (2, schema.NUM_CLASSES, config.FRAME_SIZE, config.FRAME_SIZE)
    assert model.num_parameters() > 0


def test_train_export_eval_enrich(tmp_path, monkeypatch):
    from constellation_vision import config, schema
    from constellation_vision.data import generate
    from constellation_vision.enrich import argus_enrich
    from constellation_vision.eval import harness, metrics
    from constellation_vision.model.export_onnx import export
    from constellation_vision import train as train_mod

    # Tiny, fast run into a temp dir.
    monkeypatch.setattr(config, "N_TRAIN", 24)
    monkeypatch.setattr(config, "N_TEST", 12)
    monkeypatch.setattr(config, "EPOCHS", 2)
    monkeypatch.setattr(config, "BATCH_SIZE", 8)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "CHECKPOINT", tmp_path / "segmenter.pt")
    monkeypatch.setattr(config, "ONNX_MODEL", tmp_path / "segmenter.onnx")
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(config, "EVAL_MD", tmp_path / "EVAL.md")
    monkeypatch.setattr(config, "METRICS_JSON", tmp_path / "metrics.json")
    monkeypatch.setattr(config, "HISTORY", tmp_path / "history.jsonl")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)

    metrics_out = train_mod.train(epochs=2)
    assert 0.0 <= metrics_out["mean_iou"] <= 1.0
    # Two epochs on 24 frames should already beat a background-only baseline (0).
    assert metrics_out["mean_iou"] > 0.0
    assert (tmp_path / "segmenter.onnx").exists()
    assert (tmp_path / "EVAL.md").exists()

    # ONNX enrichment round-trip on a held-out test frame.
    frame, true_mask = generate.generate_frame(config.TEST_SEED_BASE)
    mask = argus_enrich.segment_frame(frame, tmp_path / "segmenter.onnx")
    assert mask.shape == (config.FRAME_SIZE, config.FRAME_SIZE)
    assert mask.max() < schema.NUM_CLASSES
    summary = argus_enrich.enrich_frame(frame, tmp_path / "segmenter.onnx")
    assert "defects_found" in summary

    # ONNX argmax must agree with the torch model on the same frame.
    model = harness.load_model(tmp_path / "segmenter.pt")
    torch_pred = harness.predict(model, frame.astype(np.float32)[None, None])[0]
    agree = (torch_pred == mask).mean()
    assert agree > 0.99


def test_enrich_load_frame_npy_and_npz(tmp_path):
    from constellation_vision import config
    from constellation_vision.data import generate
    from constellation_vision.enrich import argus_enrich

    frame, _ = generate.generate_frame(1)
    npy = tmp_path / "f.npy"
    np.save(npy, frame)
    assert argus_enrich._load_frame(npy).shape == (config.FRAME_SIZE, config.FRAME_SIZE)

    npz = tmp_path / "f.npz"
    np.savez(npz, frame=frame)
    assert np.array_equal(argus_enrich._load_frame(npz), frame)

    npz2 = tmp_path / "f2.npz"
    np.savez(npz2, images=frame[None, None])
    assert argus_enrich._load_frame(npz2).shape == (config.FRAME_SIZE, config.FRAME_SIZE)
