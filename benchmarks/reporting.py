"""Validate and report closed-system benchmark results."""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "quchip-matplotlib"))

METRICS = {
    "cold_build_s": "Cold build",
    "build_s": "Repeated build",
    "first_solve_s": "First solve",
    "warm_solve_s": "Warm solve",
}
FAMILIES = ("qutip", "dynamiqs")
PATHS = ("head", "base", "native")

NAVY = "#1D3557"
INK = "#3D405B"
RED = "#E63946"
SAGE = "#81B29A"
TERRA = "#E07A5F"
PAPER = "#FAFBFC"


def load_result(path: Path) -> dict[str, Any]:
    """Load a complete successful benchmark result."""
    import json

    document = json.loads(path.read_text())
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("unsupported benchmark schema")
    if document.get("complete") is not True:
        raise ValueError("benchmark result is incomplete")

    provenance = document.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("benchmark provenance is missing")
    for key in ("head_commit", "base_commit"):
        value = provenance.get(key)
        if not isinstance(value, str) or len(value) != 40:
            raise ValueError(f"benchmark provenance has invalid {key}")

    rows = document.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("benchmark rows are missing")
    failures: list[str] = []
    cells: set[tuple[str, str, int]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("benchmark row is not an object")
        family = row.get("family")
        path_name = row.get("path")
        n = row.get("N")
        status = row.get("status")
        cell = f"{family}/{path_name}/N={n}"
        if status != "ok":
            failures.append(f"{cell}: {status}")
            continue
        if family not in FAMILIES or path_name not in PATHS or not isinstance(n, int):
            raise ValueError(f"invalid benchmark cell {cell}")
        if not isinstance(row.get("dim"), int) or row["dim"] < 1:
            raise ValueError(f"{cell} has invalid dimension")
        for metric in METRICS:
            value = row.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{cell} has invalid {metric}")
        cells.add((family, path_name, n))
    if failures:
        raise ValueError("; ".join(failures))

    rungs = {row["N"] for row in rows}
    missing = [
        f"{family}/{path_name}/N={n}"
        for family in FAMILIES
        for path_name in PATHS
        for n in rungs
        if (family, path_name, n) not in cells
    ]
    if missing:
        raise ValueError(f"benchmark cells are missing: {', '.join(missing)}")
    return document


def _rows(document: Mapping[str, Any], family: str, path_name: str) -> list[dict[str, Any]]:
    return sorted(
        (
            row
            for row in document["rows"]
            if row["family"] == family and row["path"] == path_name
        ),
        key=lambda row: row["dim"],
    )


def _paired(document: Mapping[str, Any], family: str, metric: str) -> tuple[list[int], list[float]]:
    head = _rows(document, family, "head")
    base = _rows(document, family, "base")
    if [row["dim"] for row in head] != [row["dim"] for row in base]:
        raise ValueError(f"head/base dimensions differ for {family}")
    deltas = [(head_row[metric] / base_row[metric] - 1.0) * 100.0 for head_row, base_row in zip(head, base)]
    return [row["dim"] for row in head], deltas


def render_markdown(document: Mapping[str, Any]) -> str:
    """Render a compact head-versus-base job summary."""
    provenance = document["provenance"]
    lines = [
        "## Closed-system performance",
        "",
        f"Head `{provenance['head_commit'][:12]}` compared with base `{provenance['base_commit'][:12]}` in one job.",
        "Timing changes are informational; execution and physics-parity failures are not.",
        "",
        "| Backend | Regime | Largest space | Worst observed |",
        "| --- | --- | ---: | ---: |",
    ]
    for family in FAMILIES:
        for metric, label in METRICS.items():
            _, deltas = _paired(document, family, metric)
            largest = deltas[-1]
            worst = max(deltas, key=abs)
            backend = "QuTiP" if family == "qutip" else "dynamiqs"
            lines.append(f"| {backend} | {label} | {largest:+.1f}% | {worst:+.1f}% |")
    lines.extend(
        ("", "Lower is faster. Download the run bundle for absolute timings, samples, environment, and plots.", "")
    )
    return "\n".join(lines)


def _time_label(value: float, _: float) -> str:
    if value >= 1.0:
        return f"{value:.3g} s"
    if value >= 1e-3:
        return f"{value * 1e3:.3g} ms"
    return f"{value * 1e6:.3g} μs"


def _style_axis(axis: Any, dims: list[int], n_by_dim: Mapping[int, int], *, xlabel: bool) -> None:
    from matplotlib.ticker import NullLocator

    axis.set_xscale("log")
    axis.set_xticks(dims)
    axis.set_xticklabels([f"{dim}\nN={n_by_dim[dim]}" for dim in dims])
    axis.xaxis.set_minor_locator(NullLocator())
    axis.set_xlim(dims[0] / 1.35, dims[-1] * 1.7)
    axis.spines[["top", "right"]].set_visible(False)
    if xlabel:
        axis.set_xlabel("Hilbert dimension")


def render_plots(document: Mapping[str, Any], output_dir: Path) -> list[Path]:
    """Render the dashboard and regression plots from validated data."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    output_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path = output_dir / "closed-system-dashboard.png"
    regression_path = output_dir / "closed-system-regression.png"
    n_by_dim = {row["dim"]: row["N"] for row in document["rows"]}
    styles: dict[str, dict[str, Any]] = {
        "head": dict(color=NAVY, marker="o", linestyle="-"),
        "base": dict(color=TERRA, marker="s", linestyle="-."),
        "native": dict(color=SAGE, marker="o", linestyle="--", markerfacecolor="white"),
    }
    labels = {"head": "PR head", "base": "stacked base", "native": "direct framework"}
    rc: dict[str, Any] = {
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "axes.grid": True,
        "grid.color": INK,
        "grid.alpha": 0.13,
        "axes.titlelocation": "left",
        "axes.titlesize": 9.5,
        "font.size": 8.5,
        "legend.frameon": False,
    }

    with plt.rc_context(rc):  # type: ignore[arg-type]
        figure, axes = plt.subplots(2, 3, figsize=(11.5, 6.2), squeeze=False)
        for row_index, family in enumerate(FAMILIES):
            dims = [row["dim"] for row in _rows(document, family, "head")]
            absolute, delta, builds = axes[row_index]
            for path_name in PATHS:
                rows = _rows(document, family, path_name)
                absolute.plot(dims, [row["warm_solve_s"] for row in rows], **styles[path_name])
            absolute.set_yscale("log")
            absolute.yaxis.set_major_formatter(FuncFormatter(_time_label))
            absolute.set_title(f"{'QuTiP' if family == 'qutip' else 'dynamiqs'} · warm solve")
            _style_axis(absolute, dims, n_by_dim, xlabel=row_index == 1)

            _, warm_delta = _paired(document, family, "warm_solve_s")
            delta.axhspan(-5, 5, color=SAGE, alpha=0.18, linewidth=0)
            delta.axhline(0, color=INK, alpha=0.65, linestyle="--", linewidth=0.9)
            delta.plot(dims, warm_delta, color=RED, marker="o", linewidth=1.6)
            delta.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:+.0f}%"))
            delta.set_title(f"{'QuTiP' if family == 'qutip' else 'dynamiqs'} · warm delta")
            _style_axis(delta, dims, n_by_dim, xlabel=row_index == 1)

            for metric, marker, linestyle in (("cold_build_s", "s", "-"), ("build_s", "o", "--")):
                for path_name in PATHS:
                    rows = _rows(document, family, path_name)
                    builds.plot(
                        dims,
                        [row[metric] for row in rows],
                        color=styles[path_name]["color"],
                        marker=marker,
                        linestyle=linestyle,
                        linewidth=1.4,
                    )
            builds.set_yscale("log")
            builds.yaxis.set_major_formatter(FuncFormatter(_time_label))
            builds.set_title(f"{'QuTiP' if family == 'qutip' else 'dynamiqs'} · build regimes")
            _style_axis(builds, dims, n_by_dim, xlabel=row_index == 1)
        axes[0, 0].set_ylabel("time")
        axes[1, 0].set_ylabel("time")
        axes[0, 1].set_ylabel("head vs base")
        axes[1, 1].set_ylabel("head vs base")
        handles = [Line2D([0], [0], label=labels[path], **styles[path]) for path in PATHS]
        handles.extend(
            (
                Line2D([0], [0], label="cold build", color=INK, marker="s", linestyle="-"),
                Line2D([0], [0], label="repeated build", color=INK, marker="o", linestyle="--"),
            )
        )
        figure.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.93))
        figure.suptitle("Closed-system benchmark · CI dashboard", fontsize=14, y=0.99)
        figure.tight_layout(rect=(0, 0, 1, 0.88), h_pad=1.8, w_pad=1.6)
        figure.savefig(dashboard_path, dpi=220, bbox_inches="tight", facecolor=PAPER)
        plt.close(figure)

        figure, axes = plt.subplots(4, 2, figsize=(9.2, 9.2), sharex="col", squeeze=False)
        for row_index, (metric, label) in enumerate(METRICS.items()):
            for column, family in enumerate(FAMILIES):
                axis = axes[row_index, column]
                dims, deltas = _paired(document, family, metric)
                axis.axhspan(-5, 5, color=SAGE, alpha=0.18, linewidth=0)
                axis.axhline(0, color=INK, alpha=0.65, linestyle="--", linewidth=0.9)
                axis.plot(dims, deltas, color=RED, marker="o", linewidth=1.6)
                axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:+.0f}%"))
                axis.set_title(f"{'QuTiP' if family == 'qutip' else 'dynamiqs'} · {label}")
                _style_axis(axis, dims, n_by_dim, xlabel=row_index == len(METRICS) - 1)
            axes[row_index, 0].set_ylabel("head vs base")
        figure.suptitle("Performance regression view · lower is faster", fontsize=14, y=1.0)
        figure.tight_layout(h_pad=1.5, w_pad=1.6)
        figure.savefig(regression_path, dpi=220, bbox_inches="tight", facecolor=PAPER)
        plt.close(figure)

    return [dashboard_path, regression_path]
