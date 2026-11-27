"""The guides, studies and front page show output that a fresh execution reproduces."""

from __future__ import annotations

import contextlib
import io
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import jupytext
import matplotlib
import nbformat
import numpy as np
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

from tools.execute_markdown import execute_page
from tools.sync_example_outputs import DEFAULT_PAIRS, GUIDE_PAIRS, render_cell_outputs


matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
# An imaginary unit after a number stays in the template, so real and imaginary values stay distinct.
NUMBER_RE = re.compile(
    r"(?<![A-Za-z_])[-+]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))"
    r"(?:[eE][-+]?\d+)?(?=j(?![A-Za-z_])|(?![A-Za-z_]))"
)
GUIDE_OUTPUT_RTOL = 1e-10
# Two hertz for GHz-valued receipts, below the guide's displayed precision.
GUIDE_OUTPUT_ATOL = 2e-9
DYNAMIQS_NOTEBOOKS = {"gradients-and-fitting", "fluxonium-spectroscopy"}
LOCAL_PATH_RE = re.compile(r"/Users/|/home/|/private/|/tmp/|[A-Za-z]:\\|site-packages")
README_EXAMPLE_RE = re.compile(r"```python\n(.*?)\n```\n\n```text\n(.*?)\n```", re.DOTALL)
README_FIGURE_RE = re.compile(r"https://docs\.quchip\.org/_static/readme/([\w.-]+\.png)")


@pytest.fixture
def guide_workspace(tmp_path):
    """Keep rendered guide figures outside the checkout with the same relative paths."""
    for directory in ("examples", "docs/images", "docs/_static"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "docs/_static/quchip.mplstyle", tmp_path / "docs/_static/quchip.mplstyle")
    return tmp_path


def _assert_output_matches(actual: str, expected: str | None, where: str = "") -> None:
    """Compare guide output exactly except for numerical solver roundoff."""
    actual = actual.strip()
    expected = (expected or "").strip()
    assert NUMBER_RE.sub("<number>", actual) == NUMBER_RE.sub("<number>", expected), where

    actual_numbers = np.array([float(value) for value in NUMBER_RE.findall(actual)])
    expected_numbers = np.array([float(value) for value in NUMBER_RE.findall(expected)])
    np.testing.assert_allclose(
        actual_numbers,
        expected_numbers,
        rtol=GUIDE_OUTPUT_RTOL,
        atol=GUIDE_OUTPUT_ATOL,
        err_msg=where,
    )


def _run_opening_example(path: str, workspace: Path) -> dict[str, object]:
    notebook = jupytext.read(ROOT / path)
    namespace: dict[str, object] = {"__name__": "__guide_example__"}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
        with contextlib.chdir(workspace / "examples"):
            started = False
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    started = True
                    exec(compile(cell.source, str(ROOT / path), "exec"), namespace)
                elif started and ("\n## " in "\n" + cell.source or "```{figure}" in cell.source):
                    break
    return namespace


@pytest.mark.examples
def test_statics_guide_starts_with_a_small_declaration(guide_workspace) -> None:
    """The opening statics example reads one declared chip before sweeping it."""
    example = _run_opening_example("examples/spectra-and-sweeps.md", guide_workspace)
    assert tuple(device.label for device in example["chip"].devices) == ("q1", "q2", "bus")
    assert float(example["chip"].freq("q1")) > 5.0


@pytest.mark.examples
def test_measurement_guide_matches_the_linear_cavity_response(guide_workspace) -> None:
    """The declared readout reproduces the coherent-state occupation and measurement dephasing."""
    example = _run_opening_example("examples/continuous-measurement.md", guide_workspace)
    kappa, detuning = example["kappa"], 2 * np.pi * 0.002
    denominator = (kappa / 2) ** 2 + detuning ** 2
    photons = (np.pi * example["amplitude"]) ** 2 / denominator
    dephasing = 2 * kappa * detuning ** 2 * photons / denominator
    # Ten times the default relative integration tolerance; cavity truncation is negligible at nbar < 0.2.
    np.testing.assert_allclose([example["nbar"], example["gamma"]], [photons, dephasing], rtol=1e-5)


def test_committed_markdown_contains_current_notebook_outputs() -> None:
    """Every canonical Markdown guide contains outputs from its executed pair."""
    check = subprocess.run(
        [sys.executable, "tools/sync_example_outputs.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, check.stdout + check.stderr
    for stem in DEFAULT_PAIRS:
        assert "<!-- executed-output:start -->" in (ROOT / "examples" / f"{stem}.md").read_text(encoding="utf-8")


def test_committed_notebooks_show_no_warnings_or_local_paths() -> None:
    """A warning prints the path of its source file, so its output differs between machines."""
    for stem in DEFAULT_PAIRS:
        notebook = nbformat.read(ROOT / "examples" / f"{stem}.ipynb", as_version=4)
        for index, cell in enumerate((cell for cell in notebook.cells if cell.cell_type == "code"), start=1):
            stderr = "".join(
                "".join(output.get("text", "")) for output in cell.outputs if output.get("name") == "stderr"
            )
            assert not stderr, f"{stem} code cell {index} writes to stderr:\n{stderr}"
            rendered = render_cell_outputs(cell)
            assert LOCAL_PATH_RE.search(rendered) is None, f"{stem} code cell {index} shows a local path"


def test_rendered_output_does_not_depend_on_stream_flushes() -> None:
    """The kernel can split one printed text into several stream outputs."""
    text = ["K = 1: 8.9e-04\n", "K = 2: 2.7e-06\n"]
    split = nbformat.v4.new_code_cell(outputs=[nbformat.v4.new_output("stream", text=line) for line in text])
    whole = nbformat.v4.new_code_cell(outputs=[nbformat.v4.new_output("stream", text="".join(text))])
    assert render_cell_outputs(split) == render_cell_outputs(whole)


def test_readme_example_reproduces_its_output() -> None:
    """The README chip prints the frequencies that the README shows."""
    examples = README_EXAMPLE_RE.findall((ROOT / "README.md").read_text(encoding="utf-8"))
    assert len(examples) == 1
    code, expected = examples[0]
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(compile(code, str(ROOT / "README.md"), "exec"), {"__name__": "__readme__"})
    _assert_output_matches(captured.getvalue(), expected)


def test_readme_figures_are_current_copies_of_the_docs_figures() -> None:
    """The README links to copies in docs/_static/readme, which tools/theme_figures.py refreshes."""
    published = ROOT / "docs" / "_static" / "readme"
    linked = README_FIGURE_RE.findall((ROOT / "README.md").read_text(encoding="utf-8"))
    assert linked
    for name in linked:
        assert (published / name).exists(), f"the README links to a missing figure: {name}"
    for copy in sorted(published.glob("*.png")):
        source = ROOT / "docs" / "images" / copy.name
        if source.exists():
            assert copy.read_bytes() == source.read_bytes(), f"{copy.name} differs from docs/images"


PAGES = [
    pytest.param("index", marks=pytest.mark.e2e),
    pytest.param("get-started/first-simulation", marks=pytest.mark.e2e),
    *(
        pytest.param(page, marks=pytest.mark.validation)
        for page in (
            "get-started/first-chip",
            "guides/steady-state-and-vna",
            "guides/gate-sets",
            "guides/resonance-fluorescence",
            "guides/scqubits",
            "studies/purcell-filter",
        )
    ),
]


@pytest.mark.examples
@pytest.mark.parametrize("page", PAGES)
def test_page_outputs_match_a_fresh_execution(page: str, guide_workspace) -> None:
    """Pages without a notebook execute their checks and reproduce their shown output."""
    if page == "guides/scqubits":
        pytest.importorskip("scqubits")
    executed = execute_page(ROOT / "docs" / f"{page}.md", guide_workspace / "docs" / "images")
    assert executed, "the page must contain executable examples"
    for index, (block, printed) in enumerate(executed, start=1):
        _assert_output_matches(printed, block["text"], f"{page}, Python block {index}")


@pytest.mark.examples
@pytest.mark.parametrize(
    "stem",
    [
        pytest.param(stem, marks=[
            pytest.mark.validation if stem in GUIDE_PAIRS else pytest.mark.release,
            *([pytest.mark.optional_backend] if stem in DYNAMIQS_NOTEBOOKS else []),
        ])
        for stem in DEFAULT_PAIRS
    ],
)
def test_notebook_outputs_match_a_fresh_execution(stem: str, guide_workspace) -> None:
    """Each notebook executes again and reproduces the textual outputs that its page shows.

    The guide notebooks run in validation. The studies download data or run long solves, so
    they run in the release suite.
    """
    if stem in DYNAMIQS_NOTEBOOKS:
        pytest.importorskip("dynamiqs")
    path = ROOT / "examples" / f"{stem}.ipynb"
    committed = nbformat.read(path, as_version=4)
    fresh = nbformat.read(path, as_version=4)
    # The same execution as the documented `jupyter nbconvert --execute` command.
    ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(
        fresh, {"metadata": {"path": str(guide_workspace / "examples")}},
    )

    committed_cells = [cell for cell in committed.cells if cell.cell_type == "code"]
    fresh_cells = [cell for cell in fresh.cells if cell.cell_type == "code"]
    for index, (committed_cell, fresh_cell) in enumerate(zip(committed_cells, fresh_cells, strict=True), start=1):
        _assert_output_matches(
            render_cell_outputs(fresh_cell), render_cell_outputs(committed_cell), f"{stem}, code cell {index}"
        )
