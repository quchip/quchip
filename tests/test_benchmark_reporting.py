"""Contracts for benchmark validation and reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.reporting import load_result, render_markdown, render_plots


METRICS = {
    "cold_build_s": 0.4,
    "build_s": 0.3,
    "first_solve_s": 0.2,
    "warm_solve_s": 0.1,
}


def _document() -> dict[str, object]:
    rows = []
    for family in ("qutip", "dynamiqs"):
        for n in (1, 2):
            for path, scale in (("head", 1.1), ("main", 1.0), ("native", 0.8)):
                row = {
                    "N": n,
                    "levels": 3,
                    "dim": 3**n,
                    "family": family,
                    "path": path,
                    "status": "ok",
                    "parity": 0.0,
                }
                row.update({name: value * scale * n for name, value in METRICS.items()})
                rows.append(row)
    return {
        "schema_version": 2,
        "complete": True,
        "provenance": {
            "head_commit": "a" * 40,
            "main_commit": "b" * 40,
            "parity_tol": 1e-5,
        },
        "rows": rows,
    }


def test_load_result_rejects_failed_measurements(tmp_path: Path) -> None:
    """Reject incomplete or failed measurements before reporting them."""
    document = _document()
    document["rows"][0]["status"] = "parity-fail"  # type: ignore[index]
    path = tmp_path / "result.json"
    path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="qutip/head/N=1: parity-fail"):
        load_result(path)


def test_render_markdown_reports_exact_revisions_and_all_regimes(tmp_path: Path) -> None:
    """Show exact revisions and all timing regimes in the CI summary."""
    path = tmp_path / "result.json"
    path.write_text(json.dumps(_document()))

    markdown = render_markdown(load_result(path))

    assert "aaaaaaaaaaaa" in markdown
    assert "bbbbbbbbbbbb" in markdown
    assert "Cold build" in markdown
    assert "Repeated build" in markdown
    assert "First solve" in markdown
    assert "Warm solve" in markdown
    assert "first model construction in a fresh worker" in markdown
    assert "median of later model constructions" in markdown
    assert "includes backend setup and JAX compilation where applicable" in markdown
    assert "repeated execution after solver warm-up" in markdown
    assert "+10.0%" in markdown


def test_render_plots_writes_one_simple_comparison(tmp_path: Path) -> None:
    """Render one PR-versus-main-versus-hand-built comparison."""
    path = tmp_path / "result.json"
    path.write_text(json.dumps(_document()))

    outputs = render_plots(load_result(path), tmp_path)

    assert [output.name for output in outputs] == ["closed-system-comparison.png"]
    assert all(output.stat().st_size > 1_000 for output in outputs)
