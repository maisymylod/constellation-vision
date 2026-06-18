"""Defect taxonomy: the single source of ground truth for the segmentation task.

Each pixel of a frame belongs to exactly one class. Class 0 is the panel
background; classes 1..N are surface-defect types injected by the synthesiser.
The same taxonomy is consumed by the generator (to paint masks), the trainer (to
size the output head), the eval (to report per-class IoU), and the argus
enrichment step (to name detected defects).
"""
from __future__ import annotations

from dataclasses import dataclass

BACKGROUND = "background"
CRACK = "crack"
HOT_SPOT = "hot_spot"
DELAMINATION = "delamination"
MISSING_FASTENER = "missing_fastener"


@dataclass(frozen=True)
class DefectClass:
    index: int
    name: str
    # A representative grayscale intensity delta the synthesiser applies inside
    # the defect region, relative to the panel surface. Cracks are dark, hot
    # spots are bright, delamination is a mottled mid-bright patch, a missing
    # fastener exposes a dark bore. These are appearance priors, not labels:
    # the model still has to learn the spatial signature from pixels.
    intensity_delta: float


# Order defines the class indices. Background must be index 0.
CLASSES: tuple[DefectClass, ...] = (
    DefectClass(0, BACKGROUND, 0.0),
    DefectClass(1, CRACK, -0.55),
    DefectClass(2, HOT_SPOT, +0.60),
    DefectClass(3, DELAMINATION, +0.30),
    DefectClass(4, MISSING_FASTENER, -0.75),
)

NUM_CLASSES = len(CLASSES)
DEFECT_CLASSES: tuple[DefectClass, ...] = CLASSES[1:]
CLASS_NAMES: tuple[str, ...] = tuple(c.name for c in CLASSES)
NAME_TO_INDEX = {c.name: c.index for c in CLASSES}
INDEX_TO_NAME = {c.index: c.name for c in CLASSES}
INTENSITY_DELTA = {c.index: c.intensity_delta for c in CLASSES}
