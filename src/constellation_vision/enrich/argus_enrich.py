"""Argus enrichment: run the ONNX defect segmenter over a frame and summarize.

This is the thin module the `argus` sibling can call as an extra enrichment step.
`argus` already has an ONNX inference wrapper (`services/ml/infer.py`) that loads
a model with a single named input and a mask-like output, and its own model card
notes that its built-in `veg_segmenter` is a deterministic threshold model to be
swapped for a real trained network. This repo is that real, trained network for
orbital-hardware frames: load `segmenter.onnx` here, get a per-pixel defect mask,
and return a structured per-defect summary (and optional overlay) the same way
argus turns masks into downstream detections.

Pure-numpy postprocessing so the summary is testable without onnxruntime; the
ONNX session is loaded lazily only when a frame is actually segmented.
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import config, schema


@lru_cache(maxsize=2)
def _session(model_path: str):
    import onnxruntime as ort

    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def segment_frame(frame: np.ndarray, model_path: str | Path = config.ONNX_MODEL) -> np.ndarray:
    """Run the ONNX segmenter on a single-band frame [H, W], return mask [H, W].

    Mirrors the argus `infer.segment` contract: float32 input, argmax over the
    class-logit channels gives the per-pixel class index.
    """
    session = _session(str(model_path))
    inp = frame.astype(np.float32)[None, None, :, :]
    logits = session.run(None, {"frame": inp})[0]
    return logits[0].argmax(axis=0).astype(np.uint8)


def summarize_mask(mask: np.ndarray) -> dict:
    """Per-defect summary from a predicted mask. No ONNX needed (testable)."""
    total = int(mask.size)
    defects = []
    for cls in schema.DEFECT_CLASSES:
        sel = mask == cls.index
        count = int(sel.sum())
        if count == 0:
            continue
        ys, xs = np.nonzero(sel)
        defects.append({
            "defect": cls.name,
            "pixel_count": count,
            "area_fraction": round(count / total, 4),
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "centroid": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
        })
    defects.sort(key=lambda d: d["pixel_count"], reverse=True)
    return {
        "defects_found": len(defects),
        "defect_area_fraction": round(sum(d["area_fraction"] for d in defects), 4),
        "defects": defects,
    }


def enrich_frame(frame: np.ndarray, model_path: str | Path = config.ONNX_MODEL) -> dict:
    """Full enrichment: segment a frame and return its per-defect summary."""
    mask = segment_frame(frame, model_path)
    return summarize_mask(mask)


def _load_frame(path: Path) -> np.ndarray:
    """Load a frame from .npy (float32 [H,W]) or .npz (key 'frame' or 'images')."""
    if path.suffix == ".npz":
        d = np.load(path)
        if "frame" in d:
            return d["frame"]
        arr = d["images"]
        return arr[0, 0] if arr.ndim == 4 else arr
    return np.load(path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Argus enrichment: segment a panel frame and summarize defects."
    )
    ap.add_argument("--frame", help="path to a .npy/.npz frame; omit to use a synthetic demo frame")
    ap.add_argument("--seed", type=int, default=config.TEST_SEED_BASE,
                    help="seed for the synthetic demo frame when --frame is omitted")
    ap.add_argument("--model", default=str(config.ONNX_MODEL), help="path to segmenter.onnx")
    args = ap.parse_args()

    if args.frame:
        frame = _load_frame(Path(args.frame))
    else:
        from ..data.generate import generate_frame

        frame, _ = generate_frame(args.seed)

    summary = enrich_frame(frame, args.model)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
