"""Map-quality metrics.

The headline number is `semantic_miou`, computed over *every* ground-truth
occupied voxel with unobserved voxels counted as errors. The observed-only
variant is reported alongside because it is the one most papers quote, and the
gap between the two is exactly the amount of credit a mapper gets for the parts
of the scene it never looked at.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from .scene import CLASS_NAMES, N_CLASSES, Scene


class Metrics(BaseModel):
    """Everything one run measures. Serialises straight to the results JSON."""

    semantic_miou: float = Field(description="mIoU over all GT-occupied voxels")
    semantic_miou_observed: float = Field(description="mIoU restricted to observed voxels")
    per_class_iou: dict[str, float | None] = Field(
        description="per-class IoU; null for classes absent from both truth and prediction"
    )
    occupancy_iou: float
    coverage: float = Field(description="fraction of GT-occupied voxels observed at least once")
    ece: float = Field(description="expected calibration error of the semantic posterior")
    mean_confidence: float
    frame_accuracy: float = Field(description="per-observation detector accuracy")
    n_observations: int
    n_frames: int


def iou_per_class(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Per-class IoU over flattened label arrays. Absent classes yield NaN."""
    out: dict[str, float] = {}
    for i, name in enumerate(CLASS_NAMES, start=1):
        t, p = truth == i, pred == i
        union = int((t | p).sum())
        out[name] = float("nan") if union == 0 else float((t & p).sum()) / union
    return out


def mean_iou(per_class: dict[str, float]) -> float:
    """Mean over classes present in the scene; NaN classes are skipped, not zeroed."""
    values = [v for v in per_class.values() if not np.isnan(v)]
    return float(np.mean(values)) if values else 0.0


def expected_calibration_error(
    correct: np.ndarray, confidence: np.ndarray, n_bins: int = 10
) -> float:
    """Standard equal-width ECE. Empty bins contribute nothing."""
    if correct.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # `right=True` so the top bin is closed and confidence == 1.0 is not dropped.
    bins = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)

    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        n = int(mask.sum())
        if n == 0:
            continue
        total += n * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total / correct.size)


def evaluate_map(
    scene: Scene,
    labels: np.ndarray,
    occupied: np.ndarray,
    confidence: np.ndarray,
    n_obs: np.ndarray,
    frame_accuracy: float,
    n_observations: int,
    n_frames: int,
) -> Metrics:
    truth = scene.labels
    gt_occupied = scene.occupancy

    per_class = iou_per_class(truth[gt_occupied | occupied], labels[gt_occupied | occupied])

    observed = n_obs > 0
    obs_mask = gt_occupied & observed
    per_class_observed = iou_per_class(truth[obs_mask], labels[obs_mask])

    occ_union = int((gt_occupied | occupied).sum())
    occupancy_iou = float((gt_occupied & occupied).sum()) / occ_union if occ_union else 0.0

    n_gt = int(gt_occupied.sum())
    coverage = float(obs_mask.sum()) / n_gt if n_gt else 0.0

    # Calibration is only meaningful where the map actually committed to a label.
    conf_mask = observed & occupied & gt_occupied
    correct = (labels[conf_mask] == truth[conf_mask]).astype(float)
    conf_values = confidence[conf_mask]

    return Metrics(
        semantic_miou=mean_iou(per_class),
        semantic_miou_observed=mean_iou(per_class_observed),
        per_class_iou={k: (None if np.isnan(v) else v) for k, v in per_class.items()},
        occupancy_iou=occupancy_iou,
        coverage=coverage,
        ece=expected_calibration_error(correct, conf_values),
        mean_confidence=float(conf_values.mean()) if conf_values.size else 0.0,
        frame_accuracy=frame_accuracy,
        n_observations=n_observations,
        n_frames=n_frames,
    )


CHANCE_MIOU = 1.0 / N_CLASSES
