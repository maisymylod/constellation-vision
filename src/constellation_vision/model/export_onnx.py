"""Export the trained UNet-lite to ONNX for the argus enrichment step.

The exported graph takes a single-band frame `[1, 1, H, W]` named ``frame`` and
returns class logits `[1, NUM_CLASSES, H, W]` named ``logits``; argsmax over the
channel axis gives the per-pixel defect mask. Mirrors the ONNX-session contract
the `argus` sibling already uses for its inference wrapper, so the enrichment
module can load this model with the same onnxruntime pattern.
"""
from __future__ import annotations

from pathlib import Path

from .. import config


def export(model=None, path: str | Path | None = None) -> Path:
    import torch

    from .unet import UNetLite

    if path is None:
        path = config.ONNX_MODEL
    if model is None:
        model = UNetLite()
        state = torch.load(config.CHECKPOINT, map_location="cpu")
        model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.eval()

    config.models_dir()
    path = Path(path)
    dummy = torch.zeros(1, config.IN_CHANNELS, config.FRAME_SIZE, config.FRAME_SIZE)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["frame"],
        output_names=["logits"],
        dynamic_axes={"frame": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        # Use the stable TorchScript exporter so the only ONNX deps are onnx +
        # onnxruntime (no onnxscript), keeping the install lean and CPU-only.
        dynamo=False,
    )
    print(f"exported ONNX model -> {config.rel(path)}")
    return path


def main() -> None:
    export()


if __name__ == "__main__":
    main()
