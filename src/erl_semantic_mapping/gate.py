"""The properties this benchmark must keep having to be worth reporting.

A mapping benchmark where a random detector scores respectably, or where a
perfect detector cannot reach the metric's ceiling, is measuring the scene
generator rather than the pipeline. These checks run in CI on every push and
fail the build, so no number this repo publishes outlives the assumptions that
make it meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

from .evaluate import RunConfig, RunResult, run
from .metrics import CHANCE_MIOU

#: The standard matrix. `tag -> (detector, fusion)`.
SWEEP: dict[str, tuple[str, str]] = {
    "oracle": ("oracle", "bayes"),
    "random": ("random", "bayes"),
    "independent70": ("independent:0.7", "bayes"),
    "independent85": ("independent:0.85", "bayes"),
    "viewbias85": ("viewbias:0.85", "bayes"),
    "single_shot": ("independent:0.7", "first"),
    "majority": ("independent:0.7", "majority"),
    "no_semantics": ("independent:0.7", "none"),
}


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


def run_sweep(base: RunConfig | None = None) -> dict[str, RunResult]:
    """Run every entry in `SWEEP` on identical geometry."""
    base = base or RunConfig()
    results: dict[str, RunResult] = {}
    for tag, (detector, fusion) in SWEEP.items():
        config = base.model_copy(update={"detector": detector, "fusion": fusion})
        results[tag] = run(config, tag=tag)
    return results


def suite_sanity(results: dict[str, RunResult]) -> list[Check]:
    """Four properties, each of which invalidates the suite if it stops holding."""
    m = {tag: r.metrics for tag, r in results.items()}
    checks: list[Check] = []

    ceiling = m["oracle"].semantic_miou_observed
    checks.append(
        Check(
            "the metric is reachable",
            ceiling >= 0.99,
            f"a perfect detector scores {ceiling:.3f} mIoU on observed voxels "
            f"(need >= 0.990, else the mapper or the metric is lossy on its own)",
        )
    )

    floor = m["random"].semantic_miou
    checks.append(
        Check(
            "chance is not rewarded",
            floor <= CHANCE_MIOU,
            f"a uniform-random detector scores {floor:.3f} mIoU "
            f"(need <= {CHANCE_MIOU:.3f}, else labels can be guessed)",
        )
    )

    gain = m["independent70"].semantic_miou - m["single_shot"].semantic_miou
    checks.append(
        Check(
            "fusion earns its keep",
            gain >= 0.15,
            f"fusing {m['independent70'].n_observations} observations beats using the "
            f"first one by {gain:+.3f} mIoU (need >= 0.150)",
        )
    )

    coverage = m["oracle"].coverage
    checks.append(
        Check(
            "the trajectory sees the scene",
            coverage >= 0.80,
            f"the trajectory observes {coverage:.1%} of occupied voxels "
            f"(need >= 80%, else the map is being graded on a corner of the room)",
        )
    )

    return checks


def finding_holds(results: dict[str, RunResult]) -> list[Check]:
    """The headline claim, re-derived on every push rather than quoted from a README."""
    m = {tag: r.metrics for tag, r in results.items()}
    checks: list[Check] = []

    biased, weaker = m["viewbias85"], m["independent70"]
    checks.append(
        Check(
            "accuracy does not order map quality",
            biased.semantic_miou < weaker.semantic_miou,
            f"viewbias at {biased.frame_accuracy:.1%} frame accuracy maps to "
            f"{biased.semantic_miou:.3f} mIoU, worse than independent at "
            f"{weaker.frame_accuracy:.1%} which maps to {weaker.semantic_miou:.3f}",
        )
    )

    matched = m["independent85"]
    delta = matched.semantic_miou - biased.semantic_miou
    checks.append(
        Check(
            "correlation is the cause, not accuracy",
            delta >= 0.05,
            f"at matched {matched.frame_accuracy:.1%} accuracy, independent errors give "
            f"{matched.semantic_miou:.3f} mIoU and correlated errors {biased.semantic_miou:.3f} "
            f"({delta:+.3f}, need >= 0.050)",
        )
    )

    checks.append(
        Check(
            "correlated error is confidently wrong",
            biased.ece >= 2.0 * matched.ece,
            f"ECE is {biased.ece:.3f} under correlated error vs {matched.ece:.3f} under "
            f"independent error at the same accuracy (need >= 2x)",
        )
    )

    return checks


def all_checks(results: dict[str, RunResult]) -> list[Check]:
    return suite_sanity(results) + finding_holds(results)
