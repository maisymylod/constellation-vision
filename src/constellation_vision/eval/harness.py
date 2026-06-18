"""Evaluate a trained checkpoint on the held-out test split and write the report.

`make eval` runs this. It loads `data/test.npz` (regenerating it if needed),
runs the model in batches, and computes mean IoU + per-class IoU + a confusion
matrix via `eval.metrics`. Real numbers, regenerated from a clean checkout.
"""
from __future__ import annotations

import argparse

import numpy as np

from .. import config
from ..data import generate
from . import metrics as M
from . import report


def _load_test_split():
    path = config.DATA_DIR / "test.npz"
    if not path.exists():
        generate.main()
    d = np.load(path)
    return d["images"], d["masks"]


def predict(model, images: np.ndarray, batch_size: int = config.BATCH_SIZE) -> np.ndarray:
    """Run the torch model over images [N,1,H,W], return predicted masks [N,H,W]."""
    import torch

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            x = torch.from_numpy(images[i : i + batch_size])
            logits = model(x)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def load_model(checkpoint=None):
    import torch

    from ..model.unet import UNetLite

    if checkpoint is None:
        checkpoint = config.CHECKPOINT
    model = UNetLite()
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    return model


def evaluate(model=None, append_history: bool = True) -> dict:
    """Evaluate ``model`` (or the committed checkpoint) and write the reports."""
    images, masks = _load_test_split()
    if model is None:
        model = load_model()
    pred = predict(model, images)
    metrics = M.summarize(pred, masks)
    meta = {
        "n_train": config.N_TRAIN,
        "n_test": len(images),
        "frame_size": config.FRAME_SIZE,
        "seed": config.SEED,
        "epochs": config.EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "lr": config.LR,
        "num_parameters": model.num_parameters(),
    }
    report.write_reports(metrics, meta, append_history=append_history)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the defect segmenter on the held-out split.")
    ap.add_argument("--no-history", action="store_true", help="do not append to history.jsonl")
    args = ap.parse_args()
    metrics = evaluate(append_history=not args.no_history)
    print(f"mean IoU (defect classes): {metrics['mean_iou']:.4f}")
    for name, v in metrics["per_class_iou"].items():
        print(f"  {name:18s}: {'n/a' if v is None else f'{v:.4f}'}")
    print(f"wrote {config.rel(config.EVAL_MD)}")


if __name__ == "__main__":
    main()
