"""Semantic detectors: models of *how* per-frame labels are wrong.

Every detector here is defined by a single per-frame accuracy, so two detectors
can be given the same accuracy and still differ in the only thing that turns out
to matter downstream — whether their errors are independent across views.

- `independent:p`  errors are i.i.d. per observation. Fusion averages them away.
- `viewbias:p`     errors are a deterministic function of (true class, viewing
                   sector). A surface seen repeatedly from one sector is
                   mislabelled *the same way* every time, so fusion cannot
                   average anything away and instead accumulates confidence in
                   the wrong class.

`viewbias` is calibrated against the actual observation distribution of a run so
that its realised accuracy matches the requested one. Without that step the
comparison would be confounded: corrupting a (class, sector) pair that happens
to carry 30% of all rays is a very different detector from one that corrupts a
pair carrying 0.3%.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scene import N_CLASSES
from .sensor import N_SECTORS


class Detector:
    """Base class. Subclasses map true labels to observed labels."""

    name: str = "detector"

    def calibrate(self, counts: np.ndarray) -> None:
        """Optionally adapt to the observation histogram, shape (N_CLASSES+1, N_SECTORS)."""

    def observe(
        self, true_labels: np.ndarray, sectors: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        raise NotImplementedError


class OracleDetector(Detector):
    """Perfect labels. Establishes the ceiling the metric can actually reach."""

    name = "oracle"

    def observe(self, true_labels, sectors, rng):
        return true_labels.copy()


class RandomDetector(Detector):
    """Uniform labels. Establishes the floor: a mapper fed this must score at chance."""

    name = "random"

    def observe(self, true_labels, sectors, rng):
        return rng.integers(1, N_CLASSES + 1, size=true_labels.shape).astype(np.int8)


@dataclass
class IndependentDetector(Detector):
    """Correct with probability `accuracy`; otherwise a uniformly wrong class."""

    accuracy: float

    def __post_init__(self):
        self.name = f"independent:{self.accuracy:g}"

    def observe(self, true_labels, sectors, rng):
        out = true_labels.copy()
        wrong = rng.random(true_labels.shape) >= self.accuracy
        n_wrong = int(wrong.sum())
        if n_wrong:
            offset = rng.integers(1, N_CLASSES, size=n_wrong)
            out[wrong] = ((true_labels[wrong] - 1 + offset) % N_CLASSES + 1).astype(np.int8)
        return out


@dataclass
class ViewBiasDetector(Detector):
    """Deterministically wrong on a calibrated subset of (class, sector) pairs.

    Same headline accuracy as `IndependentDetector`, radically different
    behaviour under fusion.
    """

    accuracy: float
    seed: int = 0
    corrupt: np.ndarray = field(default_factory=lambda: np.zeros((N_CLASSES + 1, N_SECTORS), bool))

    def __post_init__(self):
        self.name = f"viewbias:{self.accuracy:g}"

    def calibrate(self, counts: np.ndarray) -> None:
        total = counts.sum()
        if total == 0:
            return
        shares = counts / total
        target = 1.0 - self.accuracy

        # Greedy subset-sum over a deterministically shuffled pair order: take a
        # pair whenever it does not overshoot the error budget. Many small pairs
        # means the realised rate lands within a fraction of a point of target.
        pairs = [(c, s) for c in range(1, N_CLASSES + 1) for s in range(N_SECTORS)]
        order = np.random.default_rng(self.seed).permutation(len(pairs))

        corrupt = np.zeros_like(self.corrupt)
        budget = 0.0
        for i in order:
            c, s = pairs[i]
            share = shares[c, s]
            if share == 0.0 or budget + share > target:
                continue
            corrupt[c, s] = True
            budget += share
        self.corrupt = corrupt

    def observe(self, true_labels, sectors, rng):
        out = true_labels.copy()
        wrong = self.corrupt[true_labels, sectors]
        if wrong.any():
            # The confusion depends on the sector, so the bias is not a single
            # global relabelling that a downstream consumer could undo.
            offset = 1 + (sectors[wrong].astype(np.int64) % (N_CLASSES - 1))
            out[wrong] = ((true_labels[wrong] - 1 + offset) % N_CLASSES + 1).astype(np.int8)
        return out


def parse_detector(spec: str, seed: int = 0) -> Detector:
    """Build a detector from a CLI spec such as `viewbias:0.85`."""
    name, _, arg = spec.partition(":")
    name = name.strip().lower()

    if name == "oracle":
        return OracleDetector()
    if name == "random":
        return RandomDetector()
    if name in {"independent", "viewbias"}:
        if not arg:
            raise ValueError(f"detector '{name}' needs an accuracy, e.g. '{name}:0.85'")
        accuracy = float(arg)
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError(f"accuracy must be in [0, 1], got {accuracy}")
        if name == "independent":
            return IndependentDetector(accuracy)
        return ViewBiasDetector(accuracy, seed=seed)

    raise ValueError(
        f"unknown detector '{spec}'. Expected one of: oracle, random, independent:<p>, viewbias:<p>"
    )
