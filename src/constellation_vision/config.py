"""Central paths, seeds, and shapes. Everything that affects determinism lives
here so a clean checkout reproduces the dataset, the training run, and the eval.

The two seeds are deliberately separate. ``SEED`` drives model init and training;
``SPLIT_SEED`` drives the per-frame generation seeds. The train and test splits
draw from disjoint generation-seed ranges (see ``data/generate.py``), so no frame
seed is shared across splits and the held-out set cannot leak through the
synthesiser.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODELS_DIR = REPO_ROOT / "src" / "constellation_vision" / "model" / "weights"

# Frame geometry. Small on purpose: a compact UNet trains to a real IoU in a few
# minutes on CPU at this resolution.
FRAME_SIZE = 96  # square frames, FRAME_SIZE x FRAME_SIZE
IN_CHANNELS = 1  # single-band optical (grayscale) panel frames

# Dataset sizes. Kept modest so `make train` finishes in a few minutes on CPU.
N_TRAIN = 480
N_TEST = 160

# Determinism.
SEED = 42          # model init + training shuffles
SPLIT_SEED = 7     # base offset for per-frame generation seeds

# Disjoint generation-seed ranges so no frame seed crosses the split boundary.
TRAIN_SEED_BASE = SPLIT_SEED * 1_000_000
TEST_SEED_BASE = SPLIT_SEED * 1_000_000 + 500_000

# Training hyperparameters (env-overridable for the documented longer run).
EPOCHS = int(os.environ.get("CV_EPOCHS", "16"))
BATCH_SIZE = int(os.environ.get("CV_BATCH_SIZE", "16"))
LR = float(os.environ.get("CV_LR", "1e-3"))
BASE_WIDTH = int(os.environ.get("CV_BASE_WIDTH", "16"))  # UNet first-stage width

# Artifact paths.
CHECKPOINT = MODELS_DIR / "segmenter.pt"
ONNX_MODEL = MODELS_DIR / "segmenter.onnx"
METRICS_JSON = ARTIFACTS_DIR / "metrics.json"
EVAL_MD = ARTIFACTS_DIR / "EVAL.md"
HISTORY = ARTIFACTS_DIR / "history.jsonl"

# Quality gate floor enforced in CI and tests. The committed run must clear this.
MIN_MEAN_IOU = float(os.environ.get("CV_MIN_MEAN_IOU", "0.75"))


def data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def artifacts_dir() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


def models_dir() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def rel(path: Path) -> str:
    """Path relative to the repo root for tidy logging, robust to temp dirs."""
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
