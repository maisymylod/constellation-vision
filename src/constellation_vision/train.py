"""Train the UNet-lite defect segmenter from scratch, then evaluate it.

`make train` runs this end to end: regenerate data if needed, fit the model with
fixed seeds and deterministic kernels, save the checkpoint, export to ONNX, then
run the held-out eval and write artifacts/EVAL.md. The mean IoU printed and
written is the real measured number on the held-out split.

Loss is a combination of cross-entropy and a soft multi-class Dice term. Dice
counteracts the heavy background imbalance (defects are a small pixel fraction),
which plain cross-entropy alone tends to ignore.
"""
from __future__ import annotations

import numpy as np

from . import config, schema
from .data import generate
from .eval import harness


def set_seed(seed: int = config.SEED) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _load_or_build_split(name: str, n: int, seed_base: int):
    path = config.DATA_DIR / f"{name}.npz"
    if not path.exists():
        generate.main()
    d = np.load(path)
    return d["images"], d["masks"]


def dice_loss(logits, target, num_classes: int = schema.NUM_CLASSES, eps: float = 1.0):
    """Soft multi-class Dice loss. logits [N,C,H,W], target [N,H,W]."""
    import torch
    import torch.nn.functional as F

    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    union = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def train(epochs: int = config.EPOCHS) -> dict:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from .model.unet import UNetLite

    set_seed()

    train_x, train_y = _load_or_build_split("train", config.N_TRAIN, config.TRAIN_SEED_BASE)
    ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    g = torch.Generator()
    g.manual_seed(config.SEED)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True, generator=g)

    model = UNetLite()
    opt = torch.optim.Adam(model.parameters(), lr=config.LR)

    # Inverse-frequency class weights for cross-entropy. Defects are a small
    # pixel fraction and the rarest (a missing-fastener bore is a handful of
    # pixels), so unweighted CE collapses to predicting background. Weights are
    # derived from the train masks only, capped to keep the loss well-scaled.
    counts = np.bincount(train_y.reshape(-1), minlength=schema.NUM_CLASSES).astype(np.float64)
    inv = counts.sum() / (counts * schema.NUM_CLASSES + 1.0)
    weights = torch.tensor(np.clip(inv, 0.5, 25.0), dtype=torch.float32)
    ce = torch.nn.CrossEntropyLoss(weight=weights)

    print(f"training UNet-lite ({model.num_parameters():,} params) "
          f"on {len(train_x)} frames for {epochs} epochs (CPU-friendly)")
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = ce(logits, yb) + dice_loss(logits, yb)
            loss.backward()
            opt.step()
            total += loss.detach().item() * len(xb)
        print(f"  epoch {epoch + 1:2d}/{epochs}  loss {total / len(train_x):.4f}")

    config.models_dir()
    torch.save({"model_state": model.state_dict(),
                "seed": config.SEED, "epochs": epochs}, config.CHECKPOINT)
    print(f"saved checkpoint -> {config.rel(config.CHECKPOINT)}")

    # Export to ONNX for the argus enrichment step.
    from .model.export_onnx import export

    export(model)

    # Evaluate the exported ONNX model (not the in-memory torch model) on the
    # held-out split and write artifacts/EVAL.md. Evaluating what actually ships
    # makes `make train` and `make eval` report identical numbers, so the headline
    # regenerates from a clean clone via `make eval` alone.
    metrics = harness.evaluate(model=None, append_history=True)
    print(f"\nmean IoU (defect classes): {metrics['mean_iou']:.4f}")
    for name, v in metrics["per_class_iou"].items():
        print(f"  {name:18s}: {'n/a' if v is None else f'{v:.4f}'}")
    return metrics


def main() -> None:
    metrics = train()
    floor = config.MIN_MEAN_IOU
    if metrics["mean_iou"] < floor:
        raise SystemExit(
            f"mean IoU {metrics['mean_iou']:.4f} is below the quality floor {floor}"
        )


if __name__ == "__main__":
    main()
