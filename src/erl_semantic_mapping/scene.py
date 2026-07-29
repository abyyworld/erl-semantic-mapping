"""Deterministic synthetic scenes with a known semantic ground truth.

A mapping benchmark is only as trustworthy as its ground truth. Real datasets
give you noisy human annotations and no way to separate mapper error from label
error; here the ground truth is exact by construction, so every point of lost
mIoU is attributable to the pipeline under test.

The scene is a voxel grid. `occupancy` is a boolean array, `labels` is an int8
array where 0 means empty and 1..n_classes index `CLASS_NAMES`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Index 0 is reserved for "empty"; semantic classes are 1-based throughout.
CLASS_NAMES: tuple[str, ...] = ("floor", "wall", "table", "chair", "shelf", "box")
N_CLASSES = len(CLASS_NAMES)

FLOOR, WALL, TABLE, CHAIR, SHELF, BOX = range(1, N_CLASSES + 1)

# Classes that appear as free-standing furniture, in the order objects cycle
# through them. Kept deterministic so a scene is fully described by its seed.
OBJECT_CLASSES: tuple[int, ...] = (TABLE, CHAIR, SHELF, BOX)


@dataclass(frozen=True)
class Scene:
    """A voxelised room with exact per-voxel semantics."""

    occupancy: np.ndarray  # (X, Y, Z) bool
    labels: np.ndarray  # (X, Y, Z) int8, 0 = empty
    voxel_size: float
    seed: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.occupancy.shape  # type: ignore[return-value]

    @property
    def extent(self) -> np.ndarray:
        """Physical size of the grid in metres."""
        return np.asarray(self.shape, dtype=float) * self.voxel_size

    @property
    def center(self) -> np.ndarray:
        return self.extent / 2.0

    def class_counts(self) -> dict[str, int]:
        counts = np.bincount(self.labels.ravel(), minlength=N_CLASSES + 1)
        return {name: int(counts[i + 1]) for i, name in enumerate(CLASS_NAMES)}


def build_scene(
    shape: tuple[int, int, int] = (64, 64, 24),
    voxel_size: float = 0.1,
    n_objects: int = 10,
    seed: int = 0,
) -> Scene:
    """Construct a room: floor, four walls, and `n_objects` box-shaped objects.

    Objects are placed in an annulus around the room centre so that the orbiting
    sensor trajectory sees each one from a restricted range of directions. That
    restriction is the point: it is what makes viewpoint-correlated detector
    error accumulate instead of averaging out.
    """
    rng = np.random.default_rng(seed)
    nx, ny, nz = shape
    labels = np.zeros(shape, dtype=np.int8)

    # Floor slab and perimeter walls.
    labels[:, :, 0] = FLOOR
    labels[0, :, :] = WALL
    labels[-1, :, :] = WALL
    labels[:, 0, :] = WALL
    labels[:, -1, :] = WALL

    cx, cy = nx / 2.0, ny / 2.0
    r_min, r_max = 0.28 * min(nx, ny), 0.40 * min(nx, ny)

    for i in range(n_objects):
        cls = OBJECT_CLASSES[i % len(OBJECT_CLASSES)]
        angle = 2.0 * np.pi * (i + rng.uniform(-0.2, 0.2)) / n_objects
        radius = rng.uniform(r_min, r_max)
        ox = int(cx + radius * np.cos(angle))
        oy = int(cy + radius * np.sin(angle))

        sx, sy = int(rng.integers(3, 7)), int(rng.integers(3, 7))
        height = int(rng.integers(4, 13))

        x0, x1 = np.clip([ox - sx // 2, ox + sx // 2 + 1], 2, nx - 2)
        y0, y1 = np.clip([oy - sy // 2, oy + sy // 2 + 1], 2, ny - 2)
        labels[x0:x1, y0:y1, 1 : 1 + height] = cls

    return Scene(
        occupancy=labels > 0,
        labels=labels,
        voxel_size=voxel_size,
        seed=seed,
    )
