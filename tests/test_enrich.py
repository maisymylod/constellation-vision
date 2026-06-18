"""Tests for the argus enrichment postprocessing (no ONNX needed)."""
from __future__ import annotations

import numpy as np

from constellation_vision import schema
from constellation_vision.enrich import argus_enrich


def test_summarize_empty_mask():
    mask = np.zeros((20, 20), np.uint8)
    out = argus_enrich.summarize_mask(mask)
    assert out["defects_found"] == 0
    assert out["defects"] == []
    assert out["defect_area_fraction"] == 0.0


def test_summarize_single_defect():
    mask = np.zeros((20, 20), np.uint8)
    cls = schema.NAME_TO_INDEX[schema.HOT_SPOT]
    mask[5:10, 5:10] = cls  # 25 pixels
    out = argus_enrich.summarize_mask(mask)
    assert out["defects_found"] == 1
    d = out["defects"][0]
    assert d["defect"] == schema.HOT_SPOT
    assert d["pixel_count"] == 25
    assert d["bbox"] == [5, 5, 9, 9]
    assert abs(d["area_fraction"] - 25 / 400) < 1e-6


def test_summarize_orders_by_pixel_count():
    mask = np.zeros((30, 30), np.uint8)
    mask[0:2, 0:2] = schema.NAME_TO_INDEX[schema.CRACK]            # 4 px
    mask[10:20, 10:20] = schema.NAME_TO_INDEX[schema.DELAMINATION] # 100 px
    out = argus_enrich.summarize_mask(mask)
    assert out["defects_found"] == 2
    assert out["defects"][0]["defect"] == schema.DELAMINATION
    assert out["defects"][1]["defect"] == schema.CRACK
