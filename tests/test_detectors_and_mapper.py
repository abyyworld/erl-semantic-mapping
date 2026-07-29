import numpy as np
import pytest

from erl_semantic_mapping.detectors import (
    IndependentDetector,
    ViewBiasDetector,
    parse_detector,
)
from erl_semantic_mapping.mapper import MapperConfig, VoxelMap
from erl_semantic_mapping.scene import N_CLASSES
from erl_semantic_mapping.sensor import N_SECTORS

SHAPE = (4, 4, 4)


def _labels_and_sectors(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(1, N_CLASSES + 1, size=n).astype(np.int8)
    sectors = rng.integers(0, N_SECTORS, size=n).astype(np.int8)
    return labels, sectors


def _uniform_counts():
    counts = np.zeros((N_CLASSES + 1, N_SECTORS), dtype=np.int64)
    counts[1:, :] = 1000
    return counts


@pytest.mark.parametrize("spec", ["oracle", "random", "independent:0.7", "viewbias:0.85"])
def test_parse_roundtrips(spec):
    assert parse_detector(spec) is not None


@pytest.mark.parametrize("spec", ["nonsense", "independent", "independent:1.5"])
def test_parse_rejects_bad_specs(spec):
    with pytest.raises(ValueError):
        parse_detector(spec)


def test_detectors_emit_valid_classes():
    labels, sectors = _labels_and_sectors()
    rng = np.random.default_rng(0)
    for spec in ("oracle", "random", "independent:0.7", "viewbias:0.85"):
        det = parse_detector(spec)
        det.calibrate(_uniform_counts())
        out = det.observe(labels, sectors, rng)
        assert out.min() >= 1 and out.max() <= N_CLASSES


def test_independent_detector_hits_its_nominal_accuracy():
    labels, sectors = _labels_and_sectors(n=200_000)
    out = IndependentDetector(0.7).observe(labels, sectors, np.random.default_rng(0))
    assert (out == labels).mean() == pytest.approx(0.7, abs=0.01)


def test_viewbias_calibrates_to_its_nominal_accuracy():
    labels, sectors = _labels_and_sectors(n=200_000)
    det = ViewBiasDetector(0.85, seed=0)
    det.calibrate(_uniform_counts())
    out = det.observe(labels, sectors, np.random.default_rng(0))
    assert (out == labels).mean() == pytest.approx(0.85, abs=0.02)


def test_viewbias_is_deterministic_given_class_and_sector():
    """The whole mechanism: the same surface seen from the same sector is
    mislabelled identically every time, so fusion has nothing to average."""
    labels, sectors = _labels_and_sectors()
    det = ViewBiasDetector(0.85, seed=0)
    det.calibrate(_uniform_counts())

    first = det.observe(labels, sectors, np.random.default_rng(0))
    second = det.observe(labels, sectors, np.random.default_rng(999))
    assert np.array_equal(first, second)


def test_independent_is_not_deterministic():
    labels, sectors = _labels_and_sectors()
    det = IndependentDetector(0.7)
    first = det.observe(labels, sectors, np.random.default_rng(0))
    second = det.observe(labels, sectors, np.random.default_rng(999))
    assert not np.array_equal(first, second)


def test_unknown_fusion_is_rejected():
    with pytest.raises(ValueError):
        VoxelMap(SHAPE, MapperConfig(fusion="telepathy"))


def test_unobserved_voxels_stay_empty():
    vmap = VoxelMap(SHAPE)
    result = vmap.result()
    assert not result.occupied.any()
    assert (result.labels == 0).all()


def test_free_space_carving_wins_against_a_single_hit():
    """A voxel seen through many times and hit once must end up free."""
    vmap = VoxelMap(SHAPE)
    hit = np.array([5], dtype=np.int64)
    vmap.integrate(hit, np.array([1], dtype=np.int8), np.empty(0, dtype=np.int64))
    for _ in range(5):
        vmap.integrate(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int8), hit)
    assert not vmap.result().occupied.ravel()[5]


def test_bayes_and_majority_agree_under_a_symmetric_noise_model():
    """Not a coincidence and worth pinning down.

    With a symmetric confusion matrix, every observation contributes the same
    constant to every class plus a fixed bonus to the observed one, so the
    log-likelihood argmax *is* the vote count argmax. Bayes fusion buys a
    calibrated posterior here, not a different decision — and this test fails
    loudly the day someone introduces a class-dependent confusion model and
    assumes the two are still interchangeable.

    The equivalence is asserted where the vote has a unique winner. Under a tie
    the two disagree only on which of the tied classes wins, which is
    floating-point summation order rather than a modelling difference.
    """
    rng = np.random.default_rng(0)
    voxels = rng.integers(0, np.prod(SHAPE), size=4000).astype(np.int64)
    labels = rng.integers(1, N_CLASSES + 1, size=4000).astype(np.int8)
    free = np.empty(0, dtype=np.int64)

    bayes = VoxelMap(SHAPE, MapperConfig(fusion="bayes"))
    majority = VoxelMap(SHAPE, MapperConfig(fusion="majority"))
    for m in (bayes, majority):
        m.integrate(voxels, labels, free)

    votes = majority.counts
    top = np.sort(votes, axis=1)
    decisive = (top[:, -1] > top[:, -2]) & (majority.n_obs > 0)
    assert decisive.sum() > 0

    a = bayes.result().labels.ravel()[decisive]
    b = majority.result().labels.ravel()[decisive]
    assert np.array_equal(a, b)


def test_none_fusion_records_occupancy_but_no_semantics():
    rng = np.random.default_rng(0)
    voxels = rng.integers(0, np.prod(SHAPE), size=500).astype(np.int64)
    labels = rng.integers(1, N_CLASSES + 1, size=500).astype(np.int8)

    vmap = VoxelMap(SHAPE, MapperConfig(fusion="none"))
    vmap.integrate(voxels, labels, np.empty(0, dtype=np.int64))
    result = vmap.result()

    assert result.occupied.any()
    assert (result.labels == 0).all()
