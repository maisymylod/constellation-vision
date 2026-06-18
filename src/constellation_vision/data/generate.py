"""Deterministic synthetic-frame generator for orbital-hardware defect segmentation.

Mirrors how the Heliosnet `groundstation` telemetry synthesiser works: everything
is procedural and seeded, so the dataset reproduces byte-for-byte from a clean
clone. Each sample is a small single-band optical frame of a panel surface plus a
per-pixel ground-truth mask drawn from `schema.CLASSES`.

A frame is built in layers:

1. A panel surface: a smooth brightness gradient (simulated illumination) plus
   a faint regular fastener grid and low-amplitude sensor noise.
2. Zero or more injected defects, each painted into both the image and the mask:
   - crack            a thin dark poly-line
   - hot_spot         a bright Gaussian blob
   - delamination     a mottled mid-bright blob with a ragged edge
   - missing_fastener a dark circular bore where a fastener should be

Determinism and the split barrier
----------------------------------
Every frame is generated from its own integer seed. Train frames draw seeds from
`[TRAIN_SEED_BASE, TRAIN_SEED_BASE + N_TRAIN)` and test frames from
`[TEST_SEED_BASE, ...)`; the two ranges are disjoint, so no frame seed is shared
across splits and the held-out test set cannot leak through the synthesiser.
"""
from __future__ import annotations

import numpy as np

from .. import config, schema


def _panel_surface(rng: np.random.Generator, size: int) -> np.ndarray:
    """A plausible quiet panel surface in [0, 1]: illumination gradient, a faint
    fastener grid, and low-amplitude noise."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    # Illumination gradient from a random direction.
    ang = rng.uniform(0, 2 * np.pi)
    grad = np.cos(ang) * xx + np.sin(ang) * yy
    grad = (grad - grad.min()) / (np.ptp(grad) + 1e-6)
    base = 0.45 + 0.25 * grad

    # Faint regular fastener grid (intact fasteners read slightly bright).
    grid = np.zeros((size, size), np.float32)
    step = size // 4
    off = step // 2
    for cy in range(off, size, step):
        for cx in range(off, size, step):
            grid += 0.06 * _disk(size, cy, cx, 1.6)
    surface = base + grid + rng.normal(0, 0.02, (size, size)).astype(np.float32)
    return np.clip(surface, 0.0, 1.0)


def _disk(size: int, cy: float, cx: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return np.exp(-d2 / (2.0 * r * r)).astype(np.float32)


def _hard_disk(size: int, cy: float, cx: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r)


def _paint_crack(img, mask, rng, size):
    """A thin dark poly-line of a few connected segments."""
    cls = schema.NAME_TO_INDEX[schema.CRACK]
    n_seg = int(rng.integers(2, 4))
    y, x = rng.uniform(0.15, 0.85, 2) * size
    pts = [(y, x)]
    for _ in range(n_seg):
        ang = rng.uniform(0, 2 * np.pi)
        length = rng.uniform(0.2, 0.4) * size
        y = np.clip(y + np.sin(ang) * length, 1, size - 2)
        x = np.clip(x + np.cos(ang) * length, 1, size - 2)
        pts.append((y, x))
    half = rng.uniform(0.6, 1.1)
    for (y0, x0), (y1, x1) in zip(pts, pts[1:]):
        steps = int(max(abs(y1 - y0), abs(x1 - x0)) * 2) + 1
        for t in np.linspace(0, 1, steps):
            cy, cx = y0 + t * (y1 - y0), x0 + t * (x1 - x0)
            region = _hard_disk(size, cy, cx, half)
            mask[region] = cls
            img[region] += schema.INTENSITY_DELTA[cls]


def _paint_hot_spot(img, mask, rng, size):
    cls = schema.NAME_TO_INDEX[schema.HOT_SPOT]
    cy, cx = rng.uniform(0.2, 0.8, 2) * size
    r = rng.uniform(0.07, 0.14) * size
    blob = _disk(size, cy, cx, r)
    img += schema.INTENSITY_DELTA[cls] * blob
    mask[blob > 0.4] = cls


def _paint_delamination(img, mask, rng, size):
    cls = schema.NAME_TO_INDEX[schema.DELAMINATION]
    cy, cx = rng.uniform(0.25, 0.75, 2) * size
    r = rng.uniform(0.10, 0.18) * size
    base = _disk(size, cy, cx, r)
    mottle = rng.normal(1.0, 0.4, base.shape).astype(np.float32)
    region = (base * np.clip(mottle, 0, None)) > 0.4
    img[region] += schema.INTENSITY_DELTA[cls] * np.clip(mottle, 0.3, 1.6)[region]
    mask[region] = cls


def _paint_missing_fastener(img, mask, rng, size):
    """A dark circular bore placed on a fastener grid node."""
    cls = schema.NAME_TO_INDEX[schema.MISSING_FASTENER]
    step = size // 4
    off = step // 2
    nodes = [(cy, cx) for cy in range(off, size, step) for cx in range(off, size, step)]
    cy, cx = nodes[int(rng.integers(0, len(nodes)))]
    r = rng.uniform(2.0, 3.2)
    region = _hard_disk(size, cy, cx, r)
    img[region] += schema.INTENSITY_DELTA[cls]
    mask[region] = cls


_PAINTERS = {
    schema.CRACK: _paint_crack,
    schema.HOT_SPOT: _paint_hot_spot,
    schema.DELAMINATION: _paint_delamination,
    schema.MISSING_FASTENER: _paint_missing_fastener,
}


def generate_frame(seed: int, size: int = config.FRAME_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Generate one (image, mask) pair deterministically from ``seed``.

    Returns
    -------
    image : float32 [H, W] in [0, 1]
    mask  : int64   [H, W] class indices from ``schema.CLASSES``
    """
    rng = np.random.default_rng(seed)
    img = _panel_surface(rng, size)
    mask = np.zeros((size, size), np.int64)

    # Each frame gets 0..3 defects; the empty case keeps a real "clean panel"
    # background distribution in the data.
    n_defects = int(rng.integers(0, 4))
    defect_names = list(_PAINTERS.keys())
    rng.shuffle(defect_names)
    for name in defect_names[:n_defects]:
        _PAINTERS[name](img, mask, rng, size)

    np.clip(img, 0.0, 1.0, out=img)
    return img.astype(np.float32), mask


def build_split(n: int, seed_base: int, size: int = config.FRAME_SIZE):
    """Build a stacked split: images [N,1,H,W] float32, masks [N,H,W] int64."""
    images = np.empty((n, 1, size, size), np.float32)
    masks = np.empty((n, size, size), np.int64)
    for i in range(n):
        img, mask = generate_frame(seed_base + i, size)
        images[i, 0] = img
        masks[i] = mask
    return images, masks


def build_dataset(size: int = config.FRAME_SIZE):
    """The full train/test dataset from disjoint generation-seed ranges."""
    train_x, train_y = build_split(config.N_TRAIN, config.TRAIN_SEED_BASE, size)
    test_x, test_y = build_split(config.N_TEST, config.TEST_SEED_BASE, size)
    return (train_x, train_y), (test_x, test_y)


def main() -> None:
    config.data_dir()
    (train_x, train_y), (test_x, test_y) = build_dataset()
    np.savez_compressed(config.DATA_DIR / "train.npz", images=train_x, masks=train_y)
    np.savez_compressed(config.DATA_DIR / "test.npz", images=test_x, masks=test_y)
    px_train = train_y.size
    defect_px = int((train_y > 0).sum())
    print(f"wrote {len(train_x)} train + {len(test_x)} test frames "
          f"({config.FRAME_SIZE}x{config.FRAME_SIZE}) -> data/")
    print(f"train defect-pixel fraction: {defect_px / px_train:.4f}")


if __name__ == "__main__":
    main()
