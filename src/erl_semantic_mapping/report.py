"""Rendering: rich tables for humans, markdown for CI step summaries."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from .evaluate import RunResult
from .scene import CLASS_NAMES

COLUMNS = (
    ("detector", "{d}"),
    ("fusion", "{f}"),
    ("frame acc", "{m.frame_accuracy:.1%}"),
    ("mIoU", "{m.semantic_miou:.3f}"),
    ("mIoU (obs)", "{m.semantic_miou_observed:.3f}"),
    ("occ IoU", "{m.occupancy_iou:.3f}"),
    ("coverage", "{m.coverage:.1%}"),
    ("ECE", "{m.ece:.3f}"),
)


def _row(result: RunResult) -> list[str]:
    return [
        fmt.format(d=result.config.detector, f=result.config.fusion, m=result.metrics)
        for _, fmt in COLUMNS
    ]


def render_table(results: list[RunResult], title: str = "Mapping runs") -> None:
    """Print a summary table to the terminal."""
    table = Table(title=title, header_style="bold")
    for name, _ in COLUMNS:
        is_label = name in {"detector", "fusion"}
        table.add_column(
            name,
            justify="left" if is_label else "right",
            # Detector specs carry the accuracy that makes the row meaningful;
            # a truncated "indepen…" loses exactly the information you came for.
            no_wrap=is_label,
            min_width=18 if name == "detector" else None,
        )
    for result in results:
        table.add_row(*_row(result))
    Console().print(table)


def render_markdown(results: list[RunResult], title: str = "Mapping runs") -> str:
    """Render the same table as markdown, for `$GITHUB_STEP_SUMMARY`."""
    header = [name for name, _ in COLUMNS]
    align = ["---" if name in {"detector", "fusion"} else "---:" for name in header]
    lines = [f"### {title}", "", "| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(_row(r)) + " |" for r in results]
    return "\n".join(lines) + "\n"


def render_per_class_markdown(results: list[RunResult]) -> str:
    """Per-class IoU, which is where a viewpoint-correlated detector shows itself."""
    header = ["detector / fusion", *CLASS_NAMES]
    lines = [
        "### Per-class IoU",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] + ["---:"] * len(CLASS_NAMES)) + " |",
    ]
    for r in results:
        cells = [
            "—" if (v := r.metrics.per_class_iou.get(name)) is None else f"{v:.2f}"
            for name in CLASS_NAMES
        ]
        lines.append("| " + " | ".join([r.label, *cells]) + " |")
    return "\n".join(lines) + "\n"
