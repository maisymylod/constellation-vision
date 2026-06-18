"""Tests for the synthetic-data drift check."""
from __future__ import annotations

import json

from constellation_vision import schema
from constellation_vision.data import drift


def test_compute_stats_shape():
    stats = drift.compute_stats()
    assert set(stats["class_pixel_fraction"]) == set(schema.CLASS_NAMES)
    assert stats["n_train"] > 0 and stats["n_test"] > 0
    total = sum(stats["class_pixel_fraction"].values())
    assert abs(total - 1.0) < 1e-3


def test_check_matches_committed_reference():
    # The committed reference must reproduce from the deterministic generator.
    ok, stats = drift.check(write_reference=False)
    assert ok, "synthetic data drifted from the committed reference"
    assert stats["frame_size"] == drift.config.FRAME_SIZE


def test_check_writes_reference_when_requested(tmp_path, monkeypatch):
    ref = tmp_path / "data_stats.json"
    monkeypatch.setattr(drift, "REFERENCE", ref)
    ok, stats = drift.check(write_reference=True)
    assert ok
    assert ref.exists()
    written = json.loads(ref.read_text())
    assert written == stats
