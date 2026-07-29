"""Posed depth sensor: trajectory generation and vectorised ray casting.

Each frame yields, for every ray, the voxel it terminated on and the voxels it
passed through on the way. Both matter: hits are semantic evidence, and the
free-space carving is what keeps occupancy IoU from being trivially gamed by a
mapper that marks everything occupied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scene import Scene


@dataclass(frozen=True)
class Pose:
    position: np.ndarray  # (3,) metres
    yaw: float  # radians, rotation about +z
    pitch: float  # radians, negative looks down


@dataclass(frozen=True)
class CameraConfig:
    width: int = 48
    height: int = 36
    hfov_deg: float = 90.0
    max_range: float = 8.0
    step: float = 0.05  # ray-march increment, metres


@dataclass(frozen=True)
class Frame:
    """One sensor observation, already resolved against the ground truth."""

    hit_voxels: np.ndarray  # (H,) int64 flat voxel indices that rays terminated on
    hit_labels: np.ndarray  # (H,) int8 true class of each hit voxel
    hit_octants: np.ndarray  # (H,) int8 viewing-direction octant, 0..7
    free_voxels: np.ndarray  # (F,) int64 flat voxel indices observed as free


def orbit_trajectory(
    scene: Scene,
    n_waypoints: int = 12,
    radius: float = 2.0,
    height: float = 1.2,
    yaws_per_waypoint: int = 4,
    pitch_deg: float = -10.0,
) -> list[Pose]:
    """A circular fly-through, panning through `yaws_per_waypoint` headings.

    Panning at each waypoint is what gives the walls and floor coverage; the
    orbit is what gives the central objects parallax.
    """
    center = scene.center
    poses: list[Pose] = []
    for i in range(n_waypoints):
        theta = 2.0 * np.pi * i / n_waypoints
        position = np.array(
            [center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta), height]
        )
        for k in range(yaws_per_waypoint):
            poses.append(
                Pose(
                    position=position,
                    yaw=2.0 * np.pi * k / yaws_per_waypoint,
                    pitch=np.deg2rad(pitch_deg),
                )
            )
    return poses


def _ray_directions(pose: Pose, cam: CameraConfig) -> np.ndarray:
    """Unit ray directions in world frame, shape (width * height, 3)."""
    hfov = np.deg2rad(cam.hfov_deg)
    vfov = hfov * cam.height / cam.width

    # Pixel-centre sampling, so the frustum is symmetric about the optical axis.
    yaw_offsets = (np.arange(cam.width) + 0.5) / cam.width - 0.5
    pitch_offsets = (np.arange(cam.height) + 0.5) / cam.height - 0.5
    yaws = pose.yaw + yaw_offsets * hfov
    pitches = pose.pitch - pitch_offsets * vfov

    yy, pp = np.meshgrid(yaws, pitches, indexing="xy")
    dirs = np.stack(
        [np.cos(pp) * np.cos(yy), np.cos(pp) * np.sin(yy), np.sin(pp)], axis=-1
    ).reshape(-1, 3)
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


N_SECTORS = 16  # 8 in azimuth x 2 in elevation


def _octants(dirs: np.ndarray) -> np.ndarray:
    """Bucket ray directions into `N_SECTORS` viewing sectors.

    Coarse enough that a voxel surface is almost always observed from one or two
    sectors, which is precisely the correlation structure a viewpoint-dependent
    detector exploits.
    """
    azimuth = np.arctan2(dirs[:, 1], dirs[:, 0]) % (2.0 * np.pi)
    octant = (azimuth / (np.pi / 4.0)).astype(np.int8) % 8
    return (octant * 2 + (dirs[:, 2] >= 0).astype(np.int8)).astype(np.int8)


def cast(scene: Scene, pose: Pose, cam: CameraConfig) -> Frame:
    """Ray-march one frame against the scene's ground-truth occupancy."""
    nx, ny, nz = scene.shape
    dirs = _ray_directions(pose, cam)
    n_rays = dirs.shape[0]

    n_steps = int(cam.max_range / cam.step)
    t = (np.arange(1, n_steps + 1) * cam.step)[None, :, None]
    points = pose.position[None, None, :] + dirs[:, None, :] * t  # (R, S, 3)

    idx = np.floor(points / scene.voxel_size).astype(np.int64)
    inside = (
        (idx[..., 0] >= 0)
        & (idx[..., 0] < nx)
        & (idx[..., 1] >= 0)
        & (idx[..., 1] < ny)
        & (idx[..., 2] >= 0)
        & (idx[..., 2] < nz)
    )
    np.clip(idx[..., 0], 0, nx - 1, out=idx[..., 0])
    np.clip(idx[..., 1], 0, ny - 1, out=idx[..., 1])
    np.clip(idx[..., 2], 0, nz - 1, out=idx[..., 2])
    flat = (idx[..., 0] * ny + idx[..., 1]) * nz + idx[..., 2]  # (R, S)

    solid = scene.occupancy.ravel()[flat] & inside
    has_hit = solid.any(axis=1)
    first_hit = solid.argmax(axis=1)  # 0 where has_hit is False; masked out below

    step_no = np.arange(n_steps)[None, :]
    # A voxel counts as free only if the ray reached it *and* continued past it.
    free_mask = inside & (step_no < first_hit[:, None]) & has_hit[:, None]

    rays = np.arange(n_rays)
    hit_flat = flat[rays[has_hit], first_hit[has_hit]]

    return Frame(
        hit_voxels=hit_flat,
        hit_labels=scene.labels.ravel()[hit_flat],
        hit_octants=_octants(dirs)[has_hit],
        free_voxels=np.unique(flat[free_mask]),
    )


def scan(scene: Scene, poses: list[Pose], cam: CameraConfig):
    """Yield one `Frame` per pose."""
    for pose in poses:
        yield cast(scene, pose, cam)
