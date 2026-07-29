"""Run a mapping experiment end to end and measure it.

One run = one scene, one trajectory, one detector, one fusion strategy. The
scene and trajectory are functions of the seed alone, so two runs that differ
only in detector are compared on *identical* geometry and *identical* viewing
directions. That pairing is what makes the detector comparison meaningful.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, Field, model_validator

from .detectors import parse_detector
from .mapper import MapperConfig, VoxelMap
from .metrics import Metrics, evaluate_map
from .scene import N_CLASSES, build_scene
from .sensor import N_SECTORS, CameraConfig, cast, orbit_trajectory


class RunConfig(BaseModel):
    """Everything needed to reproduce a run, byte for byte."""

    detector: str = "independent:0.7"
    fusion: str = "bayes"
    assumed_accuracy: float = 0.7

    seed: int = 0
    grid: tuple[int, int, int] = (64, 64, 24)
    voxel_size: float = 0.1
    n_objects: int = 10

    n_waypoints: int = 12
    yaws_per_waypoint: int = 4
    orbit_radius: float = 2.0
    camera_height: float = 1.2

    image_width: int = 48
    image_height: int = 36
    hfov_deg: float = 90.0
    max_range: float = 8.0

    @model_validator(mode="after")
    def _trajectory_must_fit_inside_the_room(self) -> RunConfig:
        """Reject configurations that place the sensor in or beyond a wall.

        A camera outside the room still produces a full set of rays and a
        plausible-looking map, so this failure is invisible in the metrics — it
        just quietly makes every number meaningless. Caught here instead.
        """
        nx, ny, nz = self.grid
        half_x = nx * self.voxel_size / 2.0
        half_y = ny * self.voxel_size / 2.0
        margin = 2 * self.voxel_size  # keep clear of the wall voxels themselves

        if self.orbit_radius >= min(half_x, half_y) - margin:
            raise ValueError(
                f"orbit_radius {self.orbit_radius} m does not fit in a "
                f"{2 * half_x:.1f} x {2 * half_y:.1f} m room; "
                f"use less than {min(half_x, half_y) - margin:.2f}"
            )
        if not margin < self.camera_height < nz * self.voxel_size - margin:
            raise ValueError(
                f"camera_height {self.camera_height} m is outside the room's "
                f"{nz * self.voxel_size:.1f} m of headroom"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str, **overrides: Any) -> RunConfig:
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)


class RunResult(BaseModel):
    tag: str
    config: RunConfig
    metrics: Metrics
    wall_seconds: float = Field(description="end-to-end run time, for cost accounting")

    @property
    def label(self) -> str:
        return f"{self.config.detector} / {self.config.fusion}"


def run(config: RunConfig, tag: str = "run") -> RunResult:
    started = time.perf_counter()

    scene = build_scene(
        shape=tuple(config.grid),  # type: ignore[arg-type]
        voxel_size=config.voxel_size,
        n_objects=config.n_objects,
        seed=config.seed,
    )
    cam = CameraConfig(
        width=config.image_width,
        height=config.image_height,
        hfov_deg=config.hfov_deg,
        max_range=config.max_range,
    )
    poses = orbit_trajectory(
        scene,
        n_waypoints=config.n_waypoints,
        radius=config.orbit_radius,
        height=config.camera_height,
        yaws_per_waypoint=config.yaws_per_waypoint,
    )

    # Ray casting is the expensive part and does not depend on the detector, so
    # it happens once and both the calibration pass and the mapping pass reuse it.
    frames = [cast(scene, pose, cam) for pose in poses]

    counts = np.zeros((N_CLASSES + 1, N_SECTORS), dtype=np.int64)
    for frame in frames:
        np.add.at(counts, (frame.hit_labels.astype(np.int64), frame.hit_octants), 1)

    detector = parse_detector(config.detector, seed=config.seed)
    detector.calibrate(counts)

    vmap = VoxelMap(
        scene.shape,
        MapperConfig(fusion=config.fusion, assumed_accuracy=config.assumed_accuracy),
    )
    rng = np.random.default_rng(config.seed + 1)

    n_correct = 0
    n_observations = 0
    for frame in frames:
        observed = detector.observe(frame.hit_labels, frame.hit_octants, rng)
        n_correct += int((observed == frame.hit_labels).sum())
        n_observations += int(observed.size)
        vmap.integrate(frame.hit_voxels, observed, frame.free_voxels)

    result = vmap.result()
    metrics = evaluate_map(
        scene,
        labels=result.labels,
        occupied=result.occupied,
        confidence=result.confidence,
        n_obs=result.n_obs,
        frame_accuracy=n_correct / n_observations if n_observations else 0.0,
        n_observations=n_observations,
        n_frames=len(frames),
    )

    return RunResult(
        tag=tag,
        config=config,
        metrics=metrics,
        wall_seconds=time.perf_counter() - started,
    )
