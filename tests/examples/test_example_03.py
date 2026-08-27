"""Contract coverage for the public differentiability guide."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import jupytext
import matplotlib.image as mpimg
import nbformat
from jupytext.compare import compare_notebooks


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MD = ROOT / "examples" / "03_differentiate_a_driven_chip.md"
EXAMPLE_IPYNB = ROOT / "examples" / "03_differentiate_a_driven_chip.ipynb"
RESULT_RE = re.compile(r"^RESULT gradient=(\{.*\})$", re.MULTILINE)


def _code(notebook: dict) -> str:
    return "\n\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")


def _stream_output(notebook: dict) -> str:
    return "".join(
        "".join(output.get("text", []))
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )


def test_guide_is_a_strict_executed_jupytext_pair() -> None:
    """The canonical Markdown and executed notebook have identical code cells."""
    authored = jupytext.read(EXAMPLE_MD)
    executed = nbformat.read(EXAMPLE_IPYNB, as_version=4)
    nbformat.validate(executed)
    compare_notebooks(authored, executed, fmt="md", compare_outputs=False, compare_ids=False)
    strict = subprocess.run(
        [sys.executable, "-m", "jupytext", "--to", "md", "--test-strict", str(EXAMPLE_IPYNB)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert strict.returncode == 0, strict.stdout + strict.stderr


def test_source_has_all_three_loss_stages() -> None:
    """The guide covers statics, one driven sequence, and shared multi-sequence parameters."""
    source = EXAMPLE_MD.read_text(encoding="utf-8")
    code = _code(jupytext.read(EXAMPLE_MD))
    for heading in (
        "## Losses through statics",
        "## Losses through simple dynamics",
        "## Losses through multi-sequence analysis",
    ):
        assert heading in source
    for required in (
        "jax.grad(loss)",
        "jax.jacrev(residual)",
        "static_finite_difference",
        "dynamic_loss",
        "central_differences",
        "experiment_outputs",
        "multi_residual",
        "multi_loss",
        "multi_jacobian",
        "multi_loss_gradient",
    ):
        assert required in code
    for forbidden in ("from quchip.engine", "from quchip.chip", "import dynamiqs", "jax.hessian"):
        assert forbidden not in code


def test_executed_receipt_records_checked_and_multi_sequence_gradients() -> None:
    """The receipt keeps the checked pulse derivatives and joint-loss shape."""
    executed = nbformat.read(EXAMPLE_IPYNB, as_version=4)
    matches = RESULT_RE.findall(_stream_output(executed))
    assert len(matches) == 1
    receipt = json.loads(matches[0])
    assert receipt["backend"] == "dynamiqs"
    assert receipt["solver"] == "sesolve"
    assert receipt["maximum_relative_error_across_steps"] < 0.01
    assert receipt["multi_sequence_count"] == 3
    assert receipt["multi_sequence_jacobian_shape"] == [3, 3]
    assert len(receipt["multi_sequence_loss_gradient"]) == 3
    assert all(math.isfinite(value) for value in receipt["multi_sequence_loss_gradient"])
    gradients = receipt["gradient_per_reference_perturbation"]
    assert gradients["pulse.0.amplitude"] > 0.0
    assert gradients["pulse.0.sigmas"] < 0.0
    assert gradients["pulse.0.freq"] > 0.0


def test_figure_and_docs_route_are_valid() -> None:
    """The derivative figure and canonical guide route are present."""
    image = mpimg.imread(ROOT / "docs" / "images" / "differentiate_a_driven_chip.png")
    assert image.shape[0] >= 1000 and float(image.std()) > 0.02
    page = (ROOT / "docs" / "guides" / "differentiability.md").read_text(encoding="utf-8")
    sqa = (ROOT / "docs" / "guides" / "from-sqa-2026.md").read_text(encoding="utf-8")
    assert "03_differentiate_a_driven_chip.md" in page
    assert "https://docs.quchip.org/guides/differentiability" in sqa
