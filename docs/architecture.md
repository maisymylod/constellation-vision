# Architecture

constellation-vision is a single, self-contained pipeline: a deterministic
synthetic-frame generator, a from-scratch compact UNet, a held-out IoU eval, an
ONNX export, and a thin enrichment module that the `argus` sibling can call.

```
data/generate.py            model/unet.py           eval/harness.py
(synthetic panel frames  -> (UNet-lite, trained  -> (predict held-out split,
 + ground-truth masks)       from scratch)            mean IoU + per-class IoU)
       |                          |                          |
   schema.py (defect taxonomy: the single source of pixel-class ground truth)
       |                          |                          |
       v                          v                          v
  drift.py                 model/export_onnx.py       eval/report.py
 (distribution guard)      (segmenter.onnx)          (EVAL.md + history.jsonl)
                                  |
                                  v
                        enrich/argus_enrich.py
                 (ONNX session -> per-pixel mask -> per-defect summary)
```

## Components

### `schema.py` (ground truth)
Defines the per-pixel class taxonomy: `background` (index 0) plus four defect
classes (`crack`, `hot_spot`, `delamination`, `missing_fastener`). The same
taxonomy sizes the model's output head, drives the synthesiser's mask painting,
labels the per-class IoU table, and names the defects in the enrichment summary.
One definition, consumed everywhere.

### `data/generate.py` (synthetic frames)
A deterministic procedural renderer, in the spirit of the `groundstation`
telemetry synthesiser. Each frame is built from a panel surface (illumination
gradient, faint fastener grid, sensor noise) plus zero to three injected defects,
each painted into both the image and the mask. Every frame is produced from its
own integer seed. Train and test frames draw from disjoint seed ranges, so the
held-out split cannot leak through the generator (a property enforced by a test
that also checks no test frame is byte-identical to a train frame).

### `model/unet.py` (from-scratch UNet-lite)
A compact two-level UNet (encoder, bottleneck, decoder with skip connections),
roughly 117k parameters at the default width. No pretrained weights and no model
download. Small enough to train to a real held-out IoU in a few minutes on CPU.

### `train.py` (training)
Fixed seeds, deterministic kernels. Loss is cross-entropy (with inverse-frequency
class weights, since defects are a small pixel fraction and a missing-fastener
bore is only a handful of pixels) plus a soft multi-class Dice term. After
fitting, it saves the checkpoint, exports ONNX, evaluates on the held-out split,
writes the report, and exits non-zero if mean IoU is below the configured floor.

### `eval/metrics.py` and `eval/harness.py` (evaluation)
Pure-numpy IoU, per-class IoU, and a confusion matrix from predicted vs ground
truth masks. The headline is mean IoU over the defect classes (background
excluded so a background-only model cannot look good). The harness runs the model
over `data/test.npz` and hands the metrics to the report writer.

### `eval/report.py` (artifacts)
Writes `artifacts/EVAL.md` (headline, per-class table, confusion matrix),
`artifacts/metrics.json` (machine-readable), and appends to
`artifacts/history.jsonl` (metric over time). Nothing is hand-entered.

### `model/export_onnx.py` and `enrich/argus_enrich.py` (ONNX + argus)
The trained network is exported to ONNX with a single `frame` input and a
`logits` output. The enrichment module loads that ONNX model with the same
onnxruntime session pattern `argus` already uses, runs argmax to a per-pixel
defect mask, and returns a structured per-defect summary (counts, area fraction,
bounding boxes, centroids) for downstream use.

## How it fits the rest of Heliosnet

```
constellation        argus                          constellation-vision (this repo)
(telemetry plane)    (NL exploration of imagery,    from-scratch defect segmenter
                      ONNX inference -> detections)  exported to ONNX, called as an
                                  ^                   enrichment step on hardware frames
                                  |                                |
                                  +--------- segmenter.onnx -------+
```

`argus` ships a deliberately simple built-in ONNX segmenter (a deterministic
threshold model) and its model card says to swap it for a real trained network.
constellation-vision is that network for orbital-hardware frames: same ONNX
session contract, real held-out metrics, reproducible from a clean clone.
