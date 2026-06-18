"""Tests for the deterministic synthetic frame generator."""
from __future__ import annotations

import numpy as np

from constellation_vision import config, schema
from constellation_vision.data import generate


def test_frame_shapes_and_ranges():
    img, mask = generate.generate_frame(123)
    assert img.shape == (config.FRAME_SIZE, config.FRAME_SIZE)
    assert mask.shape == (config.FRAME_SIZE, config.FRAME_SIZE)
    assert img.dtype == np.float32 and mask.dtype == np.int64
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert mask.min() >= 0 and mask.max() < schema.NUM_CLASSES


def test_frame_is_deterministic():
    a_img, a_mask = generate.generate_frame(777)
    b_img, b_mask = generate.generate_frame(777)
    assert np.array_equal(a_img, b_img)
    assert np.array_equal(a_mask, b_mask)


def test_different_seeds_differ():
    a_img, _ = generate.generate_frame(1)
    b_img, _ = generate.generate_frame(2)
    assert not np.array_equal(a_img, b_img)


def test_train_test_seed_ranges_are_disjoint():
    # No frame seed may be shared across the split barrier.
    train_seeds = set(range(config.TRAIN_SEED_BASE, config.TRAIN_SEED_BASE + config.N_TRAIN))
    test_seeds = set(range(config.TEST_SEED_BASE, config.TEST_SEED_BASE + config.N_TEST))
    assert train_seeds.isdisjoint(test_seeds)


def test_split_frames_do_not_leak():
    # Build both splits and confirm no test frame is byte-identical to a train
    # frame (a real leak would show up here even if the seed ranges were wrong).
    (train_x, _), (test_x, _) = generate.build_dataset()
    train_hashes = {x.tobytes() for x in train_x}
    assert all(x.tobytes() not in train_hashes for x in test_x)


def test_all_defect_classes_appear_in_dataset():
    (_, train_y), (_, test_y) = generate.build_dataset()
    present = set(np.unique(train_y)) | set(np.unique(test_y))
    for cls in schema.CLASSES:
        assert cls.index in present, f"class {cls.name} missing from dataset"


def test_some_frames_are_clean_background_only():
    # The empty-defect case must exist so the background distribution is real.
    masks = [generate.generate_frame(config.TRAIN_SEED_BASE + i)[1] for i in range(60)]
    assert any((m == 0).all() for m in masks)
