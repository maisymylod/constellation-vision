# Model Card: Orbital-Hardware Defect Segmenter (UNet-lite, from scratch)

## Overview

A compact UNet-lite trained from scratch for semantic segmentation of surface
defects on synthetic optical frames of orbital-hardware panels. Given a
single-band 96x96 frame it returns a per-pixel class map over five classes:
`background`, `crack`, `hot_spot`, `delamination`, `missing_fastener`. The
trained model is exported to ONNX and serves as an enrichment step for the
Heliosnet `argus` imagery system.

- **Architecture:** two-level UNet (encoder, bottleneck, decoder with skip
  connections), about 117k parameters at the default width. No pretrained
  weights, no model download.
- **Loss:** cross-entropy with inverse-frequency class weights plus a soft
  multi-class Dice term. Both counteract the heavy background imbalance.
- **Determinism:** fixed seeds for data, model init, and shuffles; deterministic
  kernels. Reproducible from a clean clone.

## Data

- **Source:** programmatically synthesised by `data/generate.py`, deterministic
  per frame seed. Panel surface (illumination gradient, fastener grid, noise)
  plus zero to three injected defects, each painted into the image and the mask.
- **Ground truth:** the per-pixel mask is the synthesiser's own paint, under the
  `schema.CLASSES` taxonomy (the single source of truth shared by the model,
  eval, and enrichment).
- **Split:** held out by generation seed. Train and test seeds come from disjoint
  ranges, so no frame seed leaks across the boundary (enforced by a test that
  also checks no test frame is byte-identical to a train frame).
- **Sizes:** 480 train frames, 160 held-out test frames, 96x96, single-band.
  Train defect-pixel fraction about 6.8% (background dominates, as in real
  inspection).

## Evaluation

Headline is **mean IoU over the defect classes** (background excluded so a
background-only model cannot score well), on the held-out split. Regenerate with
`make eval` (or `make train`, which trains then evaluates). The numbers below are
the real measured output of the committed run, copied from `artifacts/EVAL.md`;
this card is not hand-tuned.

### Calibration anchors (enforced in CI)

These are not model results. They prove the metric measures segmentation quality:

| Prediction | Mean IoU (defect classes) |
|---|---|
| Perfect mask (equals ground truth) | 1.0 |
| Background-only mask | 0.0 |

### Committed model results (held-out split, 160 frames)

| Metric | Value |
|---|---|
| Mean IoU (defect classes) | **0.8774** |
| Pixel accuracy | 0.9920 |

| Class | IoU |
|---|---|
| background | 0.9943 |
| crack | 0.9132 |
| hot_spot | 0.7503 |
| delamination | 0.9083 |
| missing_fastener | 0.9378 |

Run: seed 42, 16 epochs, batch 16, lr 1e-3, base width 16. See
[artifacts/EVAL.md](artifacts/EVAL.md) for the confusion matrix.

## Intended use and limitations

- **Intended use:** an ONNX enrichment step over synthetic orbital-hardware
  frames inside Heliosnet `argus`. It proposes a per-defect summary; it is not a
  control system.
- **Synthetic only.** Trained and evaluated on procedurally generated frames.
  Real optical imagery would shift the distribution and require retraining; do
  not expect transfer without real data.
- **`missing_fastener` is the hardest class.** The bores are only a few pixels
  each; the class weighting is what lets the model find them, and its IoU is the
  most hyperparameter-sensitive.
- **Not a benchmarked land/scene classifier.** It is a small task-specific
  segmenter that demonstrates from-scratch training with an honest held-out eval.

## Reproduce

```bash
make install-train   # CPU torch + onnx + onnxruntime
make train           # train from scratch, export ONNX, eval, write EVAL.md
```

The acceptance gate (`CV_MIN_MEAN_IOU`, default 0.75) fails the run if the
held-out mean IoU drops below the floor.
