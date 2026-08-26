"""Contract coverage for the public differentiability example."""

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
FIGURE = ROOT / "docs" / "images" / "differentiate_a_driven_chip.png"
RESULT_RE = re.compile(r"^RESULT gradient=(\{.*\})$", re.MULTILINE)


def _code_cells(notebook: dict) -> list[str]:
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _stream_output(notebook: dict) -> str:
    chunks: list[str] = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                chunks.append("".join(output.get("text", [])))
    return "".join(chunks)


def test_example_is_a_strict_executed_jupytext_pair() -> None:
    authored = jupytext.read(EXAMPLE_MD)
    executed = nbformat.read(EXAMPLE_IPYNB, as_version=4)
    nbformat.validate(executed)
    compare_notebooks(authored, executed, fmt="md", compare_outputs=False, compare_ids=False)
    assert all(cell.cell_type != "raw" for cell in executed.cells)

    strict = subprocess.run(
        [sys.executable, "-m", "jupytext", "--to", "md", "--test-strict", str(EXAMPLE_IPYNB)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert strict.returncode == 0, strict.stdout + strict.stderr


def test_source_uses_only_the_public_differentiable_surface() -> None:
    source = EXAMPLE_MD.read_text(encoding="utf-8")
    code = "\n\n".join(_code_cells(jupytext.read(EXAMPLE_MD)))

    assert "# Gradient and Jacobian" in source
    assert "## Start small" in source
    for required in (
        "from quchip import",
        "DynamiqsBackend",
        "jax.grad(loss)",
        "jax.jacrev(residual)",
        'backend="dynamiqs"',
        "sequence.with_params(",
        '"pulse.0.amplitude"',
        '"pulse.0.sigmas"',
        '"pulse.0.freq"',
        "jax.value_and_grad(final_population)",
        "result.population(\"q\", level=1)",
        "central_differences",
    ):
        assert required in code
    for excluded in (
        "from quchip.engine",
        "from quchip.chip",
        "import dynamiqs",
        "._",
        "jax.hessian",
    ):
        assert excluded not in code
    assert code.count(".schedule(") == 1


def test_executed_receipt_records_three_checked_gradients() -> None:
    executed = nbformat.read(EXAMPLE_IPYNB, as_version=4)
    matches = RESULT_RE.findall(_stream_output(executed))
    assert len(matches) == 1
    receipt = json.loads(matches[0])

    assert receipt.keys() == {
        "backend",
        "base_population",
        "figure",
        "first_order_only",
        "fixed_structure_during_trace",
        "gradient_per_reference_perturbation",
        "maximum_relative_error_across_steps",
        "original_sequence_unchanged",
        "parameter_paths",
        "solver",
    }
    assert receipt["backend"] == "dynamiqs"
    assert receipt["solver"] == "sesolve"
    assert receipt["parameter_paths"] == [
        "pulse.0.amplitude",
        "pulse.0.sigmas",
        "pulse.0.freq",
    ]
    assert receipt["first_order_only"] is True
    assert receipt["fixed_structure_during_trace"] is True
    assert receipt["original_sequence_unchanged"] is True
    assert receipt["figure"] == "../docs/images/differentiate_a_driven_chip.png"
    assert 0.99 < receipt["base_population"] < 1.0
    assert receipt["maximum_relative_error_across_steps"] < 0.01

    gradients = receipt["gradient_per_reference_perturbation"]
    assert gradients.keys() == {"pulse.0.amplitude", "pulse.0.sigmas", "pulse.0.freq"}
    assert gradients["pulse.0.amplitude"] > 0.0
    assert gradients["pulse.0.sigmas"] < 0.0
    assert gradients["pulse.0.freq"] > 0.0
    assert math.isclose(gradients["pulse.0.amplitude"], 0.001576, rel_tol=0.02)
    assert math.isclose(gradients["pulse.0.sigmas"], -0.001567, rel_tol=0.02)
    assert math.isclose(gradients["pulse.0.freq"], 0.003378, rel_tol=0.02)
    assert all(math.isfinite(value) and abs(value) > 1.0e-6 for value in gradients.values())


def test_executed_figure_is_valid() -> None:
    image = mpimg.imread(FIGURE)
    assert image.shape[0] >= 1000
    assert image.shape[1] >= 1200
    assert image.shape[2] in (3, 4)
    assert float(image.std()) > 0.02


def test_documentation_links_the_example_and_notebook() -> None:
    page = (ROOT / "docs" / "examples" / "differentiate-a-driven-chip.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "guides" / "from-sqa-2026.md").read_text(encoding="utf-8")
    index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")

    assert "../../examples/03_differentiate_a_driven_chip.md" in page
    assert "../../examples/03_differentiate_a_driven_chip.ipynb" in page
    assert "https://github.com/quchip/quchip/blob/main/examples/03_differentiate_a_driven_chip.md" in page
    assert "differentiate_a_driven_chip.png" in page
    assert "../examples/differentiate-a-driven-chip" in guide
    assert "examples/differentiate-a-driven-chip" in index
