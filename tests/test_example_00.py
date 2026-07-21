"""Contract coverage for the first reader-facing quchip example."""

from __future__ import annotations

import ast
import json
import math
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import jupytext
import matplotlib.image as mpimg
import nbformat
import numpy as np
import pytest
from jupytext.compare import compare_notebooks


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_MD = ROOT / "examples" / "00_hello_chip.md"
EXAMPLE_IPYNB = ROOT / "examples" / "00_hello_chip.ipynb"
DRIVE_IMAGE = ROOT / "docs" / "images" / "hello_qubit_drive_leakage.png"
IQ_IMAGE = ROOT / "docs" / "images" / "hello_dispersive_readout_iq.png"
RESULT_RE = re.compile(r"^RESULT (drive|readout)=(\{.*\})$", re.MULTILINE)


def _markdown_code_cells(source: str) -> list[str]:
    """Return standard fenced Python cells from Jupytext Markdown."""
    return [cell.rstrip() for cell in re.findall(r"```python\n(.*?)\n```", source, re.DOTALL)]


def _notebook_code_cells(notebook: dict) -> list[str]:
    return ["".join(cell["source"]).rstrip() for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _stream_output(notebook: dict) -> str:
    chunks: list[str] = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                chunks.append("".join(output.get("text", [])))
    return "".join(chunks)


def _parse_receipts(output: str) -> dict[str, dict]:
    lines = output.splitlines()
    assert len(lines) == 2
    assert [match.group(1) for line in lines if (match := RESULT_RE.fullmatch(line))] == ["drive", "readout"]
    return {name: json.loads(payload) for name, payload in RESULT_RE.findall(output)}


def _assert_receipts_close(actual: object, expected: object, *, path: str = "receipt") -> None:
    """Compare JSON receipts with small numeric tolerance and exact structure.

    The ``1e-7`` relative and ``1e-9`` absolute tolerances absorb harmless
    NumPy/SciPy/QuTiP platform drift. Physics acceptance remains covered by
    the separate derived bounds below; strings, booleans, keys, and sequence
    lengths stay exact.
    """
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected mapping"
        assert actual.keys() == expected.keys(), f"{path}: receipt keys differ"
        for key in expected:
            _assert_receipts_close(actual[key], expected[key], path=f"{path}.{key}")
        return
    if isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected list"
        assert len(actual) == len(expected), f"{path}: list lengths differ"
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_receipts_close(actual_item, expected_item, path=f"{path}[{index}]")
        return
    if isinstance(expected, bool):
        assert isinstance(actual, bool) and actual is expected, f"{path}: booleans differ"
        return
    if isinstance(expected, (int, float)):
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool), f"{path}: expected number"
        assert math.isclose(actual, expected, rel_tol=1.0e-7, abs_tol=1.0e-9), f"{path}: numbers differ"
        return
    assert actual == expected, f"{path}: categorical values differ"


def test_receipt_comparison_allows_numeric_drift_but_keeps_structure_exact() -> None:
    """Reproduced receipts tolerate solver noise without weakening categorical evidence."""
    reference = {
        "solver": "mesolve",
        "finite": True,
        "values": [1.0, {"small": 1.0e-8}],
    }
    slightly_perturbed = {
        "solver": "mesolve",
        "finite": True,
        "values": [1.0 + 2.0e-8, {"small": 1.01e-8}],
    }

    _assert_receipts_close(slightly_perturbed, reference)

    with pytest.raises(AssertionError):
        _assert_receipts_close({**reference, "solver": "sesolve"}, reference)
    with pytest.raises(AssertionError):
        _assert_receipts_close({**reference, "finite": False}, reference)
    with pytest.raises(AssertionError):
        _assert_receipts_close({**reference, "extra": 1.0}, reference)
    with pytest.raises(AssertionError):
        _assert_receipts_close({**reference, "values": [1.01, {"small": 1.0e-8}]}, reference)


def test_hello_chip_is_an_operational_strict_jupytext_pair() -> None:
    """Jupytext recognizes the pair, and its full authored structure matches the notebook."""
    paired = subprocess.run(
        [sys.executable, "-m", "jupytext", "--paired-paths", str(EXAMPLE_MD)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    # Jupytext 1.19.x intentionally returns 1 after printing paired paths.
    assert paired.returncode == 1, paired.stderr
    paired_paths = {Path(line).resolve() for line in paired.stdout.splitlines() if line}
    assert paired_paths == {EXAMPLE_IPYNB.resolve()}

    authored = jupytext.read(EXAMPLE_MD)
    executed = nbformat.read(EXAMPLE_IPYNB, as_version=4)
    nbformat.validate(executed)
    compare_notebooks(authored, executed, fmt="md", compare_outputs=False, compare_ids=False)
    assert all(cell.cell_type != "raw" for cell in executed.cells)
    assert all("execution" not in cell.metadata for cell in executed.cells)

    strict = subprocess.run(
        [sys.executable, "-m", "jupytext", "--to", "md", "--test-strict", str(EXAMPLE_IPYNB)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert strict.returncode == 0, strict.stdout + strict.stderr


def test_hello_chip_source_encodes_the_locked_two_part_experiment() -> None:
    """The canonical source declares real multilevel drive/leakage and readout experiments."""
    markdown = EXAMPLE_MD.read_text(encoding="utf-8")
    code = "\n\n".join(_markdown_code_cells(markdown))
    tree = ast.parse(code)

    assert "# Hello, drive and readout" in markdown
    assert len(markdown.splitlines()) < 350
    assert "this example couples a duffing transmon to a lossy resonator" in markdown.lower()
    assert "one chip now shows both effects" in markdown.lower()
    for stale_prose in (
        "explicitly labeled",
        "ordinary frequencies",
        "normal `simulationresult`",
        "the dictionary only labels",
        "visible input rather than a fitted value",
        "equal aspect ratio preserves iq geometry",
        "selective control follows from",
    ):
        assert stale_prose not in markdown.lower()
    assert "## Part 1: Qubit drive and leakage" in markdown
    assert "## Part 2: Dispersive readout" in markdown

    charge_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ChargeDrive"
    ]
    assert len(charge_calls) == 2
    assert all(all(keyword.arg != "rwa" for keyword in call.keywords) for call in charge_calls)
    assert "frame=\"rotating\"" in code
    assert "rwa=True" in code
    assert "rwa=False" not in code

    assert "chip.freq(qubit)" in code
    assert "chip.freq(qubit, when={qubit: 1})" in code
    assert code.count("Gaussian(") == 2
    assert "def pi_gaussian(duration" in code
    assert "drive_pulses = tuple(pi_gaussian(duration) for duration in drive_durations)" in code
    assert "integrate.trapezoid" in code
    assert "amplitude=0.5 / unit_area" in code
    for repeated_detail in (
        "unit_short",
        "unit_long",
        "short_integration_times",
        "long_integration_times",
        "short_unit_integral",
        "long_unit_integral",
        "short_nominal_area",
        "long_nominal_area",
    ):
        assert repeated_detail not in code
    assert "chip.state({qubit: 0, readout: 0})" in code
    assert "for level in range(3)" in code
    assert "drive_batch.population(qubit, level)" in code
    assert "## Inspect the batch with Quchip" in markdown
    assert "## Customize the comparison" in markdown
    population_plot_cells = [
        cell.strip() for cell in _markdown_code_cells(markdown) if ".plot_populations(" in cell
    ]
    assert population_plot_cells == [
        "drive_batch[0].plot_populations(trace_out=readout)\nplt.show()",
        "drive_batch[1].plot_populations(trace_out=readout)\nplt.show()",
    ]
    assert "drive_times = np.linspace(0.0, drive_durations[-1], 601)" in code
    assert 'drive_handle.vary("start_time"' not in code
    assert "pulse_starts" not in code
    assert 'drive_batch.population(qubit, 1, reduce="last")' in code
    assert 'drive_batch.population(qubit, 2, reduce="max")' in code
    assert "np.where(" in code
    assert "drive_times <= duration" in code
    assert "envelope.sample(drive_times, real=True)" in code
    assert "envelope_limit =" not in code
    assert "envelope_axis.set_ylim(" in code
    assert "max(pulse.amplitude for pulse in drive_pulses)" in code
    assert "local_times" not in code
    assert "active =" not in code
    assert code.count(".twinx(") == 1
    assert "sharex=True, sharey=True" in code
    assert "xlim=(0.0, drive_durations[-1])" in code
    drive_plot_cell = next(cell for cell in _markdown_code_cells(markdown) if "drive_figure" in cell)
    assert len(drive_plot_cell.splitlines()) < 55

    assert "ChargeDrive(readout" in code
    assert "chip.freq(readout, when={qubit: 0})" in code
    assert "chip.freq(readout, when={qubit: 1})" in code
    assert "GaussianEdge(" in code
    assert "5.0 / (2.0 * np.pi * resonator_linewidth)" in code
    assert "1.0 / (2.0 * abs(readout_frequencies[1] - readout_frequencies[0]))" in code
    assert "readout_sequence.vary(" in code
    assert '"initial_state",' in code
    assert code.count(".simulate_batch(") == 2
    assert 'chip.e_ops(readout="a")' in code
    assert 'readout_batch.expect("readout")' in code

    forbidden = (
        "analyze_dispersive_readout",
        "eliminate(",
        "effective_hamiltonian",
        "steady_state",
        "assignment_error",
        "classifier",
        "synthetic",
        "shot",
        "snr",
        "chip.dress(",
        "bare_state(",
        "ellipse",
        "lowering_squared",
        "lowering_operator",
        "number_operator",
        "second_moment",
        "covariance",
        "max_photons",
        "qubit_retention",
        "top_level_population",
    )
    assert not any(term in code.lower() for term in forbidden)
    assert code.count("plt.subplots(") == 2
    assert code.count(".plot_populations(") == 2
    assert "Ellipse" not in code
    assert "drive_figure.legend(" in code
    assert "bbox_to_anchor=(0.5, 0.02)" in code
    assert "pulse_amplitudes" not in code
    assert "np.unique" not in code
    assert "np.concatenate" not in code
    assert "np.argmin" not in code
    assert "drive_end_indices" not in code
    assert "end_index =" not in code
    assert "selected_times" not in code
    assert "hello_qubit_drive_leakage.png" in code
    assert "hello_dispersive_readout_iq.png" in code


def test_hello_chip_separates_calculations_from_quchip_workflow() -> None:
    """Support calculations do not obscure the core quchip simulation cells."""
    markdown = EXAMPLE_MD.read_text(encoding="utf-8")
    cells = _markdown_code_cells(markdown)
    code = "\n\n".join(cells)

    def cell_with(fragment: str) -> str:
        matches = [cell for cell in cells if fragment in cell]
        assert len(matches) == 1, fragment
        return matches[0]

    transition_cell = cell_with("f01 = float(chip.freq(qubit))")
    pulse_cell = cell_with("def pi_gaussian")
    drive_cell = cell_with("drive_batch = drive_sequence.simulate_batch")
    frequency_cell = cell_with("readout_frequencies = (")
    timing_cell = cell_with("readout_duration =")
    readout_cell = cell_with("readout_batch = readout_sequence.simulate_batch")
    iq_cell = cell_with("alpha = np.asarray(readout_batch.expect")

    assert "Gaussian(" not in transition_cell and "np.linspace" not in transition_cell
    assert "QuantumSequence" not in pulse_cell and "simulate_batch" not in pulse_cell
    assert "integrate." not in drive_cell and "plt." not in drive_cell
    assert "GaussianEdge" not in frequency_cell and "np.ceil" not in frequency_cell
    assert "QuantumSequence" not in timing_cell and "chip." not in timing_cell
    assert "np.ceil" not in readout_cell and "plt." not in readout_cell
    assert "plt." not in iq_cell

    for redundant_name in (
        "qubit_frequency",
        "qubit_anharmonicity",
        "readout_frequency",
        "coupling_strength",
        "resonator_quality_factor",
        "qubit_levels",
        "readout_levels",
        "gaussian_sigmas",
        "short_duration",
        "long_duration",
        "short_pulse",
        "long_pulse",
        "pulse_names",
        "drive_handle",
        "drive_axis",
        "conditional_frequency_0",
        "conditional_frequency_1",
        "readout_carrier",
        "kappa",
        "duration_floor_linewidth",
        "duration_floor_pull",
        "prepared_states",
        "prepared_qubit",
        "readout_observable",
        "mean_i",
        "mean_q",
        "final_iq_separation",
    ):
        assert re.search(rf"\b{redundant_name}\s*=", code) is None


def test_hello_chip_pair_executes_and_records_physical_receipts(tmp_path: Path) -> None:
    """A clean run reproduces both figures and the physics-derived receipts."""
    markdown = EXAMPLE_MD.read_text(encoding="utf-8")
    notebook = json.loads(EXAMPLE_IPYNB.read_text(encoding="utf-8"))
    markdown_cells = _markdown_code_cells(markdown)
    notebook_cells = _notebook_code_cells(notebook)
    notebook_code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert markdown_cells
    assert markdown_cells == notebook_cells
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    assert all(cell.get("execution_count") is not None for cell in notebook_code_cells)
    assert all(
        output.get("output_type") != "execute_result"
        for cell in notebook_code_cells
        for output in cell.get("outputs", [])
    )
    stream_output = _stream_output(notebook)
    notebook_receipts = _parse_receipts(stream_output)
    embedded_figures = [
        output
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    ]
    assert len(embedded_figures) == 4
    assert sum(bool(cell.get("outputs")) for cell in notebook_code_cells) == 4
    assert sum(
        output["output_type"] == "stream"
        for cell in notebook_code_cells
        for output in cell.get("outputs", [])
    ) == 2
    for cell in notebook_code_cells:
        if cell.get("outputs"):
            display_outputs = [
                output
                for output in cell["outputs"]
                if output["output_type"] in {"display_data", "execute_result"}
                and "image/png" in output.get("data", {})
            ]
            assert len(display_outputs) == 1
            assert set(display_outputs[0]["data"]) <= {"image/png", "text/plain"}

    run_root = tmp_path / "clean-example"
    run_examples = run_root / "examples"
    (run_root / "docs" / "images").mkdir(parents=True)
    run_examples.mkdir()
    script = run_examples / "00_hello_chip.py"
    script.write_text("\n\n".join(markdown_cells), encoding="utf-8")
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    completed = subprocess.run(
        [sys.executable, script.name],
        cwd=run_examples,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    receipts = _parse_receipts(completed.stdout)
    _assert_receipts_close(receipts, notebook_receipts)

    drive = receipts["drive"]
    assert set(drive) == {
        "drive_plot",
        "durations_ns",
        "f01_ghz",
        "f12_ghz",
        "final_p1",
        "peak_p2",
    }
    separation = abs(drive["f12_ghz"] - drive["f01_ghz"])
    short_duration = drive["durations_ns"]["short"]
    long_duration = drive["durations_ns"]["long"]
    assert set(drive["durations_ns"]) == {"short", "long"}
    assert math.isclose(short_duration, 3.0 / (math.pi * separation), rel_tol=2e-4)
    assert math.isclose(long_duration, 4.0 * short_duration, rel_tol=2e-4)
    assert set(drive["final_p1"]) == {"short", "long"}
    assert set(drive["peak_p2"]) == {"short", "long"}
    for values in (drive["final_p1"], drive["peak_p2"]):
        assert all(0.0 <= value <= 1.0 for value in values.values())
    # A four-times-longer Gaussian has one quarter the spectral width. Require
    # clear adjacent-line excitation from the short pulse and a broad 4x
    # reduction in peak leakage from the selective pulse.
    assert drive["final_p1"]["long"] > 0.90
    assert drive["peak_p2"]["short"] > 0.02
    assert drive["peak_p2"]["long"] < 0.25 * drive["peak_p2"]["short"]

    readout = receipts["readout"]
    assert set(readout) == {
        "conditional_resonator_frequencies_ghz",
        "final_iq_separation",
        "iq_plot",
        "readout_carrier_ghz",
        "readout_duration_ns",
        "solver",
    }
    f0, f1 = readout["conditional_resonator_frequencies_ghz"]
    assert math.isclose(readout["readout_carrier_ghz"], 0.5 * (f0 + f1), abs_tol=1e-9)
    kappa = 2.0 * math.pi * 0.001
    linewidth_floor = 5.0 / kappa
    pull_floor = 1.0 / (2.0 * abs(f1 - f0))
    assert readout["readout_duration_ns"] >= max(linewidth_floor, pull_floor)
    assert readout["readout_duration_ns"] - max(linewidth_floor, pull_floor) < 5.0 + 1e-9
    assert math.isclose(readout["readout_duration_ns"], 880.0, abs_tol=1e-9)
    assert readout["final_iq_separation"] > 0.01
    assert readout["solver"] == "mesolve"

    for receipt, field, committed in (
        (drive, "drive_plot", DRIVE_IMAGE),
        (readout, "iq_plot", IQ_IMAGE),
    ):
        image_path = (run_examples / receipt[field]).resolve()
        assert image_path.is_relative_to(run_root)
        image = mpimg.imread(image_path)
        committed_image = mpimg.imread(committed)
        assert image.ndim in (2, 3)
        assert image.shape == committed_image.shape
        assert np.isfinite(image).all()
        assert np.isfinite(committed_image).all()
        pixel_delta = np.abs(committed_image.astype(float) - image.astype(float))
        assert float(np.mean(pixel_delta)) < 0.01
        assert float(np.quantile(pixel_delta, 0.99)) < 0.10


def test_example_toolchain_and_reproducible_commands_are_documented() -> None:
    """Installed extras and author guidance cover the real pairing/execution workflow."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    required = {"jupytext", "nbconvert", "ipykernel"}
    for extra in ("test", "dev"):
        names = {
            dependency.split("[")[0].split("<")[0].split(">")[0].split("=")[0]
            for dependency in project["optional-dependencies"][extra]
        }
        assert required <= names

    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert not (ROOT / "examples" / "AUTHORING.md").exists()
    assert "## Examples and notebooks" in contributing
    assert "jupytext --sync examples/<name>.md" in contributing
    assert "--ExecutePreprocessor.record_timing=False" in contributing
    assert "jupytext --diff --diff-format md" in contributing
    assert "jupytext --to md --test-strict examples/<name>.ipynb" in contributing
    assert "Notebook outputs remain in the executed `.ipynb`" in contributing
    assert "docs/examples/<name>.md" in contributing


def test_hello_chip_is_discoverable_with_durable_cookbook_guidance() -> None:
    """README and docs navigation expose Unit 00 and both physical parts."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs_index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    docs_example = (ROOT / "docs" / "examples" / "hello-chip.md").read_text(encoding="utf-8")
    cookbook = (ROOT / "docs" / "cookbook.md").read_text(encoding="utf-8")
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")

    assert "examples/00_hello_chip.md" in readme
    assert "docs/images/hello_qubit_drive_leakage.png" in readme
    assert "docs/images/hello_dispersive_readout_iq.png" in readme
    assert "cookbook" in docs_index
    assert "examples/hello-chip" in docs_index
    assert "hello_qubit_drive_leakage.png" in docs_index
    assert "hello_dispersive_readout_iq.png" in docs_index
    assert "include" in docs_example and "00_hello_chip.md" in docs_example
    assert "hello_qubit_drive_leakage.png" in docs_example
    assert "hello_dispersive_readout_iq.png" in docs_example
    assert not (ROOT / "docs" / "images" / "hello_chip_populations.png").exists()

    for heading in ("Purpose", "Assumptions", "Minimal usage", "Expected receipt", "Common mistake", "Full notebook"):
        assert heading in cookbook
    assert "chip.state()" in cookbook
    assert "chip.bare_state()" in cookbook
    assert "safe inside `jax.jit`" in cookbook
    assert "traced/jit construction genuinely requires" not in cookbook.lower()
    assert "simulate_batch" in cookbook
    assert "chip.dress()" in cookbook
    assert "store_states" in cookbook
    assert "QuTiP is the default backend" in cookbook
    assert "selects `mesolve`" in cookbook
    assert "effective Hamiltonian" in cookbook
    assert "nominal-pi" in cookbook
    assert "dispersive readout" in cookbook.lower()
    assert "later analysis example" in cookbook.lower()
    assert "unit 09" not in cookbook.lower()

    reader_docs = "\n".join((readme, docs_index, docs_example, cookbook)).lower()
    assert "iq paths" in reader_docs
    reader_prose = re.sub(r"https?://[^)\s]+", "", reader_docs)
    for stale_statistical_term in ("covariance", "ellipse", "circle", "blob"):
        assert stale_statistical_term not in reader_prose

    assert "canonical source" in contributing
    assert "executed `.ipynb`" in contributing
    assert "object references" in contributing
    assert "derived" in contributing and "tolerances" in contributing
    assert "exactly two compact" not in contributing
    assert "py:percent" not in contributing
    assert "2π" not in contributing
