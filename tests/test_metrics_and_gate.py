import numpy as np
import pytest

from erl_semantic_mapping.evaluate import RunConfig, run
from erl_semantic_mapping.gate import all_checks, run_sweep, suite_sanity
from erl_semantic_mapping.metrics import (
    expected_calibration_error,
    iou_per_class,
    mean_iou,
)
from erl_semantic_mapping.scene import CLASS_NAMES

# A deliberately small grid and sparse trajectory: the gate tests below need a
# full sweep, and the properties they assert are scale-free. The orbit and
# camera height are scaled with the room — `RunConfig` rejects them otherwise.
FAST = RunConfig(
    grid=(32, 32, 16),
    n_objects=6,
    n_waypoints=8,
    orbit_radius=0.9,
    camera_height=0.8,
    image_width=32,
    image_height=24,
)


def test_iou_is_one_for_a_perfect_prediction():
    truth = np.array([1, 1, 2, 3])
    per_class = iou_per_class(truth, truth.copy())
    assert per_class[CLASS_NAMES[0]] == 1.0
    assert mean_iou(per_class) == 1.0


def test_absent_classes_are_skipped_not_scored_as_zero():
    """Averaging a missing class in as 0.0 silently penalises a correct map."""
    truth = np.array([1, 1])
    per_class = iou_per_class(truth, truth.copy())
    assert np.isnan(per_class[CLASS_NAMES[-1]])
    assert mean_iou(per_class) == 1.0


def test_iou_is_zero_when_nothing_matches():
    per_class = iou_per_class(np.array([1, 1, 1]), np.array([2, 2, 2]))
    assert per_class[CLASS_NAMES[0]] == 0.0
    assert per_class[CLASS_NAMES[1]] == 0.0


def test_ece_is_zero_for_a_perfectly_calibrated_predictor():
    confidence = np.full(1000, 0.7)
    correct = np.zeros(1000)
    correct[:700] = 1.0
    assert expected_calibration_error(correct, confidence) == pytest.approx(0.0, abs=1e-9)


def test_ece_catches_confident_wrongness():
    confidence = np.full(1000, 0.99)
    correct = np.zeros(1000)
    correct[:500] = 1.0
    assert expected_calibration_error(correct, confidence) == pytest.approx(0.49, abs=0.01)


def test_ece_of_an_empty_set_is_zero():
    assert expected_calibration_error(np.array([]), np.array([])) == 0.0


def test_runs_are_reproducible():
    a = run(FAST, tag="a")
    b = run(FAST, tag="b")
    assert a.metrics.semantic_miou == b.metrics.semantic_miou
    assert a.metrics.ece == b.metrics.ece


def test_a_run_serialises_and_reloads():
    from erl_semantic_mapping.evaluate import RunResult

    original = run(FAST, tag="rt")
    restored = RunResult.model_validate_json(original.model_dump_json())
    assert restored.metrics.semantic_miou == original.metrics.semantic_miou
    assert restored.config == original.config


def test_coverage_bounds_the_headline_metric():
    """mIoU over all voxels can never exceed what coverage allows."""
    result = run(FAST.model_copy(update={"detector": "oracle"}), tag="oracle")
    assert result.metrics.semantic_miou <= result.metrics.semantic_miou_observed
    assert 0.0 <= result.metrics.coverage <= 1.0


@pytest.fixture(scope="module")
def results():
    """The full matrix, run once and shared. This is the expensive fixture."""
    return run_sweep(FAST)


def test_the_config_rejects_a_sensor_outside_the_room():
    with pytest.raises(ValueError, match="does not fit"):
        RunConfig(grid=(28, 28, 12), orbit_radius=2.0)
    with pytest.raises(ValueError, match="headroom"):
        RunConfig(grid=(28, 28, 12), orbit_radius=0.8, camera_height=5.0)


class TestSweep:
    def test_suite_sanity_holds_on_a_small_scene(self, results):
        failed = [c for c in suite_sanity(results) if not c.passed]
        assert not failed, "\n".join(str(c) for c in failed)

    def test_all_checks_are_reported(self, results):
        checks = all_checks(results)
        assert len(checks) == 7
        assert len({c.name for c in checks}) == len(checks)

    def test_the_finding_reproduces_at_reduced_scale(self, results):
        """The claim is a property of correlated error, not of one scene size."""
        biased = results["viewbias85"].metrics
        weaker = results["independent70"].metrics
        assert biased.frame_accuracy > weaker.frame_accuracy
        assert biased.semantic_miou < weaker.semantic_miou

    def test_fusion_beats_a_single_observation(self, results):
        assert (
            results["independent70"].metrics.semantic_miou
            > results["single_shot"].metrics.semantic_miou
        )

    def test_every_run_saw_the_same_geometry(self, results):
        """Paired comparison: identical rays, identical hits, only labels differ."""
        counts = {r.metrics.n_observations for r in results.values()}
        assert len(counts) == 1
