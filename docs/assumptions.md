# Assumptions and "what's real vs what a run produces"

## What is real in this repo

- **The dataset is real and reproducible.** `make data` regenerates every frame
  deterministically. The generator is pure and seeded; the drift check
  (`make drift`) confirms the committed distribution statistics reproduce
  byte-for-byte from a clean clone.
- **The model is trained from scratch.** No pretrained weights, no download. The
  committed checkpoint is the output of `make train` on CPU.
- **The metrics are real and regenerable.** Every number in the README, in
  `MODEL_CARD.md`, and in `artifacts/EVAL.md` comes from evaluating the trained
  model on the held-out split via a single command. The split is held out by
  generation seed and the no-leak property is enforced by a test.
- **The ONNX export and enrichment are real.** `segmenter.onnx` is the exported
  trained model; the enrichment CLI loads it and segments a real (synthetic)
  held-out frame. A test asserts the ONNX argmax agrees with the torch model.

## What is simulated (and why)

- **The imagery is synthetic.** Frames are procedurally rendered panel surfaces
  with injected defects, not photographs of real hardware. This keeps the repo
  self-contained, license-clean, offline, and CI-friendly: no proprietary or
  restricted imagery is fetched, stored, or required. Real optical inspection
  frames would shift the distribution and require retraining.
- **The defect appearances are priors, not labels.** The synthesiser paints each
  defect with a characteristic intensity signature, but the model is not handed
  those rules; it learns the spatial signature from pixels and is graded against
  held-out masks it never saw.

## Honest limitations

- **`missing_fastener` is a hard, tiny class.** A bore is only a few pixels
  (about 0.07% of all pixels). Plain cross-entropy ignores it entirely; the
  inverse-frequency class weighting is what lets the model find it. Its IoU is
  the most sensitive to hyperparameters of any class.
- **Domain gap.** A model trained on synthetic frames will not transfer to real
  imagery without retraining on real data. The synthetic task demonstrates the
  method (from-scratch segmentation, honest held-out eval, ONNX serving), not a
  fielded inspection system.
- **Small by design.** The default model and dataset are sized to train in
  minutes on a CPU. `configs/train.large.env` documents a wider/longer run; its
  headline must be whatever that run actually measures, not a claim made here.

## The acceptance gate

`make train` exits non-zero if the held-out mean IoU falls below
`CV_MIN_MEAN_IOU`. CI runs that same path, so a regression that drops the model
below the floor fails the build rather than landing silently.
