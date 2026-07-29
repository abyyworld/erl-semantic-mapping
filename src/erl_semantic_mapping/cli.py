"""`erl-map` — command line entry point."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from .evaluate import RunConfig, RunResult, run
from .gate import SWEEP, all_checks, run_sweep
from .report import render_markdown, render_per_class_markdown, render_table

app = typer.Typer(
    add_completion=False,
    help="Semantic occupancy mapping with an evaluation harness that gates on its own assumptions.",
)
console = Console()

RESULTS = Path("results")

# Shared option types. Declaring them once keeps the five commands consistent
# and keeps `typer.Option(...)` out of the function defaults.
SceneOpt = Annotated[Path | None, typer.Option(help="Scene YAML to load")]
SeedOpt = Annotated[int | None, typer.Option(help="Scene and noise seed")]


def _load_config(scene: Path | None, **overrides) -> RunConfig:
    if scene is not None:
        return RunConfig.from_yaml(str(scene), **overrides)
    return RunConfig(**{k: v for k, v in overrides.items() if v is not None})


def _write(path: Path, payload: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
    return path


def _append_step_summary(markdown: str) -> None:
    """Post metrics to the GitHub Actions job summary when running under CI."""
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a") as fh:
            fh.write(markdown + "\n")


@app.command("run")
def run_cmd(
    detector: Annotated[
        str, typer.Argument(help="oracle | random | independent:<p> | viewbias:<p>")
    ] = "independent:0.7",
    fusion: Annotated[str, typer.Option(help="bayes | majority | first | last | none")] = "bayes",
    tag: Annotated[str, typer.Option(help="Name for the results file")] = "run",
    scene: SceneOpt = None,
    seed: SeedOpt = None,
) -> None:
    """Build a map with one detector and one fusion strategy, and measure it."""
    config = _load_config(scene, detector=detector, fusion=fusion, seed=seed)
    result = run(config, tag=tag)

    render_table([result], title=f"{result.label}")
    path = _write(RESULTS / f"{tag}.json", result.model_dump_json(indent=2))
    console.print(f"[dim]wrote {path}[/dim]")


@app.command("sweep")
def sweep_cmd(
    scene: SceneOpt = None,
    seed: SeedOpt = None,
) -> None:
    """Run the standard detector x fusion matrix on identical geometry."""
    base = _load_config(scene, seed=seed)
    results = run_sweep(base)
    ordered = list(results.values())

    render_table(ordered, title="Sweep")
    markdown = render_markdown(ordered, "Sweep") + "\n" + render_per_class_markdown(ordered)

    _write(
        RESULTS / "sweep.json",
        json.dumps({k: v.model_dump() for k, v in results.items()}, indent=2),
    )
    path = _write(RESULTS / "sweep.md", markdown)
    _append_step_summary(markdown)
    console.print(f"[dim]wrote {path} and results/sweep.json[/dim]")


@app.command("check")
def check_cmd(
    scene: SceneOpt = None,
    seed: SeedOpt = None,
) -> None:
    """Re-derive the suite's assumptions and the headline finding. Exits non-zero on failure."""
    base = _load_config(scene, seed=seed)
    results = run_sweep(base)
    checks = all_checks(results)

    lines = ["### Suite checks", ""]
    for check in checks:
        style = "green" if check.passed else "bold red"
        console.print(f"[{style}]{check}[/{style}]")
        lines.append(f"- {'✅' if check.passed else '❌'} **{check.name}** — {check.detail}")

    verdicts = "\n".join(lines) + "\n"
    # The file keeps the matrix for anyone reading the artifact on its own; the
    # step summary gets verdicts only, because `sweep` has already posted the
    # matrix and a job summary that repeats itself is a job summary nobody reads.
    _write(
        RESULTS / "checks.md", verdicts + "\n" + render_markdown(list(results.values()), "Sweep")
    )
    _append_step_summary(verdicts)

    failed = [c for c in checks if not c.passed]
    if failed:
        console.print(f"\n[bold red]{len(failed)}/{len(checks)} checks failed[/bold red]")
        raise typer.Exit(code=1)
    console.print(f"\n[bold green]all {len(checks)} checks passed[/bold green]")


@app.command("compare")
def compare_cmd(
    baseline: Annotated[Path, typer.Argument(help="Baseline results JSON")],
    candidate: Annotated[Path, typer.Argument(help="Candidate results JSON")],
    tolerance: Annotated[
        float, typer.Option(help="mIoU the candidate may lose before this fails")
    ] = 0.02,
) -> None:
    """Refuse a candidate that loses more than `tolerance` mIoU against a baseline."""
    a = RunResult.model_validate_json(baseline.read_text())
    b = RunResult.model_validate_json(candidate.read_text())

    render_table([a, b], title="Comparison")
    delta = b.metrics.semantic_miou - a.metrics.semantic_miou
    verdict = "PASS" if delta >= -tolerance else "FAIL"

    markdown = (
        f"### Comparison\n\n"
        f"- baseline `{a.label}` — {a.metrics.semantic_miou:.3f} mIoU\n"
        f"- candidate `{b.label}` — {b.metrics.semantic_miou:.3f} mIoU\n"
        f"- **delta {delta:+.3f}** against a tolerance of {tolerance:.3f} → **{verdict}**\n"
    )
    _write(RESULTS / "comparison.md", markdown)
    _append_step_summary(markdown)

    style = "green" if verdict == "PASS" else "bold red"
    console.print(
        f"[{style}]{verdict}: mIoU delta {delta:+.3f} (tolerance {tolerance:.3f})[/{style}]"
    )
    if verdict == "FAIL":
        raise typer.Exit(code=1)


@app.command("scenarios")
def scenarios_cmd() -> None:
    """List the tags the sweep runs."""
    for tag, (detector, fusion) in SWEEP.items():
        console.print(f"  [bold]{tag:16s}[/bold] {detector:18s} {fusion}")


if __name__ == "__main__":  # pragma: no cover
    app()
