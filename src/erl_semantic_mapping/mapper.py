"""Voxel mapping: log-odds occupancy plus per-voxel semantic fusion.

Occupancy and semantics are kept separate on purpose. Occupancy is a two-class
problem with a well-behaved recursive Bayes update; semantics is a K-class
problem whose update is only Bayes *if* observations are conditionally
independent given the voxel's class. This module implements that assumption
faithfully, which is exactly why `viewbias` breaks it in an instructive way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scene import N_CLASSES

FUSION_STRATEGIES: tuple[str, ...] = ("bayes", "majority", "first", "last", "none")


@dataclass(frozen=True)
class MapperConfig:
    fusion: str = "bayes"
    assumed_accuracy: float = 0.7  # the mapper's model of its detector
    l_occ: float = 0.85  # log-odds added on a hit
    l_free: float = -0.4  # log-odds added on a pass-through
    l_clamp: float = 4.0


@dataclass(frozen=True)
class MapResult:
    labels: np.ndarray  # (X, Y, Z) int8, 0 = predicted empty
    occupied: np.ndarray  # (X, Y, Z) bool
    confidence: np.ndarray  # (X, Y, Z) float, posterior of the predicted class
    n_obs: np.ndarray  # (X, Y, Z) int32, semantic observations per voxel


class VoxelMap:
    """Incrementally fused semantic occupancy grid."""

    def __init__(self, shape: tuple[int, int, int], config: MapperConfig | None = None):
        self.shape = shape
        self.config = config or MapperConfig()
        if self.config.fusion not in FUSION_STRATEGIES:
            raise ValueError(
                f"unknown fusion '{self.config.fusion}'. Expected one of {FUSION_STRATEGIES}"
            )

        n_voxels = int(np.prod(shape))
        self.log_odds = np.zeros(n_voxels, dtype=np.float32)
        self.counts = np.zeros((n_voxels, N_CLASSES), dtype=np.int32)
        self.sem_logl = np.zeros((n_voxels, N_CLASSES), dtype=np.float64)
        self.first_label = np.zeros(n_voxels, dtype=np.int8)
        self.last_label = np.zeros(n_voxels, dtype=np.int8)
        self.n_obs = np.zeros(n_voxels, dtype=np.int32)

        # Symmetric confusion model: the mapper believes its detector is correct
        # with probability `assumed_accuracy` and otherwise uniformly wrong.
        q = self.config.assumed_accuracy
        off = (1.0 - q) / (N_CLASSES - 1)
        self._log_conf = np.full((N_CLASSES, N_CLASSES), np.log(off))
        np.fill_diagonal(self._log_conf, np.log(q))

    def integrate(
        self, hit_voxels: np.ndarray, observed_labels: np.ndarray, free_voxels: np.ndarray
    ) -> None:
        """Fold one frame into the map."""
        cfg = self.config

        np.add.at(self.log_odds, hit_voxels, cfg.l_occ)
        if free_voxels.size:
            np.add.at(self.log_odds, free_voxels, cfg.l_free)
        np.clip(self.log_odds, -cfg.l_clamp, cfg.l_clamp, out=self.log_odds)

        if hit_voxels.size == 0:
            return

        z = observed_labels.astype(np.int64) - 1  # to 0-based class index
        if cfg.fusion == "bayes":
            np.add.at(self.sem_logl, hit_voxels, self._log_conf[:, z].T)
        elif cfg.fusion == "majority":
            np.add.at(self.counts, (hit_voxels, z), 1)
        elif cfg.fusion == "first":
            unseen = self.n_obs[hit_voxels] == 0
            self.first_label[hit_voxels[unseen]] = observed_labels[unseen]
        elif cfg.fusion == "last":
            self.last_label[hit_voxels] = observed_labels

        # `none` falls through deliberately: the voxel *was* observed and the
        # label *was* thrown away. Reporting it as unobserved would let a mapper
        # that discards all semantics hide behind a coverage of zero instead of
        # showing up as full coverage at zero mIoU.
        np.add.at(self.n_obs, hit_voxels, 1)

    def result(self) -> MapResult:
        """Collapse the accumulators into a labelled map."""
        cfg = self.config
        n_voxels = self.log_odds.size
        labels = np.zeros(n_voxels, dtype=np.int8)
        confidence = np.zeros(n_voxels, dtype=np.float64)
        seen = self.n_obs > 0

        if cfg.fusion == "bayes":
            posterior = _softmax(self.sem_logl[seen])
            labels[seen] = (posterior.argmax(axis=1) + 1).astype(np.int8)
            confidence[seen] = posterior.max(axis=1)
        elif cfg.fusion == "majority":
            votes = self.counts[seen]
            labels[seen] = (votes.argmax(axis=1) + 1).astype(np.int8)
            confidence[seen] = votes.max(axis=1) / votes.sum(axis=1)
        elif cfg.fusion in {"first", "last"}:
            single = self.first_label if cfg.fusion == "first" else self.last_label
            labels[seen] = single[seen]
            # A single observation carries exactly the detector's own accuracy.
            confidence[seen] = cfg.assumed_accuracy

        occupied = self.log_odds > 0.0
        labels[~occupied] = 0
        confidence[~occupied] = 0.0

        return MapResult(
            labels=labels.reshape(self.shape),
            occupied=occupied.reshape(self.shape),
            confidence=confidence.reshape(self.shape),
            n_obs=self.n_obs.reshape(self.shape),
        )


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)
