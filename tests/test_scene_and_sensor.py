import numpy as np
import pytest

from erl_semantic_mapping.scene import CLASS_NAMES, FLOOR, WALL, build_scene
from erl_semantic_mapping.sensor import CameraConfig, cast, orbit_trajectory

CAM = CameraConfig()


@pytest.fixture(scope="module")
def scene():
    return build_scene(seed=0)


def test_scene_is_a_function_of_its_seed(scene):
    assert np.array_equal(build_scene(seed=0).labels, scene.labels)
    assert not np.array_equal(build_scene(seed=1).labels, scene.labels)


def test_every_class_is_present(scene):
    counts = scene.class_counts()
    assert set(counts) == set(CLASS_NAMES)
    assert all(n > 0 for n in counts.values()), counts


def test_occupancy_and_labels_agree(scene):
    assert np.array_equal(scene.occupancy, scene.labels > 0)


def test_room_is_enclosed(scene):
    # Walls are laid down over the floor slab, so the floor is only exposed
    # strictly inside the perimeter.
    assert (scene.labels[1:-1, 1:-1, 0] == FLOOR).all()
    for face in (scene.labels[0], scene.labels[-1], scene.labels[:, 0], scene.labels[:, -1]):
        assert (face == WALL).all()


def test_rays_terminate_on_occupied_voxels(scene):
    pose = orbit_trajectory(scene)[0]
    frame = cast(scene, pose, CAM)

    assert frame.hit_voxels.size > 0
    assert (scene.occupancy.ravel()[frame.hit_voxels]).all()
    assert np.array_equal(frame.hit_labels, scene.labels.ravel()[frame.hit_voxels])


def test_free_space_is_actually_free(scene):
    """The carved volume must not intersect the ground-truth surface.

    If it does, occupancy IoU stops meaning anything and a mapper can erase real
    geometry for free.
    """
    pose = orbit_trajectory(scene)[0]
    frame = cast(scene, pose, CAM)
    assert not scene.occupancy.ravel()[frame.free_voxels].any()


def test_a_closed_room_leaves_no_ray_unterminated(scene):
    """Every ray from inside a sealed room must hit something within max range."""
    pose = orbit_trajectory(scene)[0]
    frame = cast(scene, pose, CAM)
    assert frame.hit_voxels.size == CAM.width * CAM.height


def test_trajectory_stays_inside_the_room(scene):
    for pose in orbit_trajectory(scene):
        assert (pose.position > 0).all()
        assert (pose.position < scene.extent).all()


def test_sectors_are_in_range(scene):
    frame = cast(scene, orbit_trajectory(scene)[0], CAM)
    assert frame.hit_octants.min() >= 0
    assert frame.hit_octants.max() < 16
