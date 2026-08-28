"""Contract coverage for the public differentiability guide."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import jupytext
import matplotlib.image as mpimg
import nbformat


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MD = ROOT / "examples" / "03_differentiate_a_driven_chip.md"
EXAMPLE_IPYNB = ROOT / "examples" / "03_differentiate_a_driven_chip.ipynb"
RESULT_RE = re.compile(r"^RESULT gradient=(\{.*\})$", re.MULTILINE)
EXPERIMENTAL_RESULT_RE = re.compile(
    r"^RESULT experimental_statics=(\{.*\})$",
    re.MULTILINE,
)


def _code(notebook: dict) -> str:
    return "\n\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")


def _stream_output(notebook: dict) -> str:
    return "".join(
        "".join(output.get("text", []))
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )


def test_guide_code_matches_the_executed_notebook() -> None:
    """The reader-facing Markdown contains exactly the executed code."""
    authored = jupytext.read(EXAMPLE_MD)
    executed = nbformat.read(EXAMPLE_IPYNB, as_version=4)
    nbformat.validate(executed)
    assert _code(authored) == _code(executed)
    assert all(cell.execution_count is not None for cell in executed.cells if cell.cell_type == "code")


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
        "experimental_loss",
        "jax.value_and_grad(experimental_loss)",
        "training_indices = np.arange(0, measured_flux.size, 8)",
        "model_flux = jnp.linspace(0.5, 0.85, 351)",
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


def test_experimental_static_fit_recovers_fluxonium_parameters_on_holdout_data() -> None:
    """The experimental fit uses sparse training data and reports held-out agreement."""
    executed = nbformat.read(EXAMPLE_IPYNB, as_version=4)
    matches = EXPERIMENTAL_RESULT_RE.findall(_stream_output(executed))
    assert len(matches) == 1
    receipt = json.loads(matches[0])
    assert receipt["fit_success"] is True
    assert receipt["training_points"] == 20
    assert receipt["holdout_points"] == 133
    assert max(abs(value) for value in receipt["relative_parameter_error"]) < 0.01
    assert receipt["holdout_median_absolute_error_mhz"] < 1.0


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
    fit_image = mpimg.imread(ROOT / "docs" / "images" / "differentiate_fluxonium_fit.png")
    assert fit_image.shape[0] >= 700 and float(fit_image.std()) > 0.02
    static_image = mpimg.imread(ROOT / "docs" / "images" / "differentiate_static_slope.png")
    assert static_image.shape[0] >= 500 and float(static_image.std()) > 0.02
    page = (ROOT / "docs" / "guides" / "differentiability.md").read_text(encoding="utf-8")
    sqa = (ROOT / "docs" / "guides" / "from-sqa-2026.md").read_text(encoding="utf-8")
    assert "03_differentiate_a_driven_chip.md" in page
    assert "https://docs.quchip.org/guides/differentiability" in sqa
