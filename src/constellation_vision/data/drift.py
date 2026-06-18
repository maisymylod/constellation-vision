"""Drift-check on the synthetic frame distribution.

The continuous-improvement loop regenerates data each run. Because the generator
is deterministic, the per-class pixel mix and frame-intensity statistics should
reproduce exactly. This module computes those summary statistics and compares
them against a committed reference (`artifacts/data_stats.json`). A mismatch
means the synthesiser (or numpy) drifted, which would silently change what the
model trains and is graded on, so the loop should stop and surface it rather than
quietly retrain on shifted data.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .. import config, schema
from . import generate

REFERENCE = config.ARTIFACTS_DIR / "data_stats.json"


def compute_stats() -> dict:
    (train_x, train_y), (test_x, test_y) = generate.build_dataset()
    stats: dict = {
        "frame_size": config.FRAME_SIZE,
        "n_train": int(len(train_x)),
        "n_test": int(len(test_x)),
        "train_mean_intensity": round(float(train_x.mean()), 6),
        "train_std_intensity": round(float(train_x.std()), 6),
        "class_pixel_fraction": {},
    }
    total = train_y.size
    for cls in schema.CLASSES:
        frac = float((train_y == cls.index).sum()) / total
        stats["class_pixel_fraction"][cls.name] = round(frac, 6)
    return stats


def check(write_reference: bool = False) -> tuple[bool, dict]:
    """Return (ok, stats). ok is True if stats match the committed reference."""
    stats = compute_stats()
    if write_reference or not REFERENCE.exists():
        config.artifacts_dir()
        REFERENCE.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        return True, stats
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    return (reference == stats), stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Drift-check the synthetic frame distribution.")
    ap.add_argument("--write", action="store_true", help="(re)write the committed reference stats")
    args = ap.parse_args()
    ok, stats = check(write_reference=args.write)
    print(json.dumps(stats, indent=2))
    if not ok:
        raise SystemExit("DRIFT: synthetic data statistics no longer match the committed reference")
    print("ok: synthetic data distribution matches the committed reference")


if __name__ == "__main__":
    main()
