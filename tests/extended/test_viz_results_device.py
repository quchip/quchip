"""Visualization coverage for result populations/state/Wigner plots and device-level plots."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from quchip import Resonator
from quchip.backend import SolverResult, get_default_backend
from quchip.results import SimulationResult


def _density_matrix_from_weights(weights: dict[tuple[int, ...], float], dims: list[int]) -> object:
    backend = get_default_backend()
    density_matrix = None
    for basis_state, weight in weights.items():
        kets = [backend.basis(dim, level) for dim, level in zip(dims, basis_state)]
        ket = kets[0] if len(kets) == 1 else backend.tensor_states(*kets)
        projector = backend.matmul(ket, backend.dag(ket))
        term = weight * projector
        density_matrix = term if density_matrix is None else density_matrix + term
    return density_matrix


@pytest.fixture()
def sample_result() -> SimulationResult:
    dims = [3, 2, 2]
    states = [
        _density_matrix_from_weights(
            {
                (0, 0, 0): 0.10,
                (0, 1, 0): 0.05,
                (1, 0, 0): 0.15,
                (1, 1, 0): 0.10,
                (2, 0, 0): 0.20,
                (2, 1, 0): 0.05,
                (0, 0, 1): 0.05,
                (0, 1, 1): 0.10,
                (1, 0, 1): 0.05,
                (1, 1, 1): 0.05,
                (2, 0, 1): 0.05,
                (2, 1, 1): 0.05,
            },
            dims,
        ),
        _density_matrix_from_weights(
            {
                (0, 0, 0): 0.05,
                (0, 1, 0): 0.05,
                (1, 0, 0): 0.10,
                (1, 1, 0): 0.15,
                (2, 0, 0): 0.10,
                (2, 1, 0): 0.15,
                (0, 0, 1): 0.05,
                (0, 1, 1): 0.05,
                (1, 0, 1): 0.10,
                (1, 1, 1): 0.10,
                (2, 0, 1): 0.05,
                (2, 1, 1): 0.05,
            },
            dims,
        ),
    ]
    solver_result = SolverResult(
        times=np.array([0.0, 5.0]),
        states=states,
        expect=None,
        final_state=states[-1],
        solver="mesolve",
    )
    backend = get_default_backend()
    return SimulationResult(
        solver_result=solver_result,
        backend=backend,
        dims=dims,
        device_info=[("q0", True), ("q1", True), ("r0", False)],
    )


def _line_by_label(ax: plt.Axes, label: str) -> plt.Line2D:
    for line in ax.lines:
        if line.get_label() == label:
            return line
    raise AssertionError(f"Missing line for label {label!r}")


def test_plot_populations_returns_figure(sample_result: SimulationResult) -> None:
    """plot_populations returns a Figure instance."""
    fig = sample_result.plot_populations()
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_populations_returns_supplied_figure(sample_result: SimulationResult) -> None:
    """plot_populations given ax= returns the figure that owns that axes."""
    fig, ax = plt.subplots()
    returned = sample_result.plot_populations(ax=ax)
    assert returned is fig
    plt.close(fig)


def test_plot_populations_trace_out_single_device_matches_expected_values(sample_result: SimulationResult) -> None:
    """Tracing out a single device sums the weights onto the remaining basis labels correctly."""
    fig = sample_result.plot_populations(trace_out="r0")
    line = _line_by_label(fig.axes[0], "|20>")
    assert np.allclose(line.get_ydata(), np.array([0.25, 0.15]))
    plt.close(fig)


def test_plot_populations_trace_out_list_matches_expected_values(sample_result: SimulationResult) -> None:
    """Tracing out a list of devices sums the weights onto the remaining basis labels correctly."""
    fig = sample_result.plot_populations(trace_out=["q1", "r0"])
    line = _line_by_label(fig.axes[0], "|2>")
    assert np.allclose(line.get_ydata(), np.array([0.35, 0.35]))
    plt.close(fig)


def test_plot_populations_computational_filter_restricts_basis_labels(sample_result: SimulationResult) -> None:
    """computational=True restricts plotted labels to the qubit computational subspace."""
    fig = sample_result.plot_populations(trace_out="r0", computational=True)
    labels = {line.get_label() for line in fig.axes[0].lines}
    assert labels == {"|00>", "|01>", "|10>", "|11>"}
    plt.close(fig)


def test_plot_populations_default_colors_follow_tab20_index_order(sample_result: SimulationResult) -> None:
    """Default line colors follow the tab20 colormap in basis-state index order."""
    fig = sample_result.plot_populations(trace_out=["q1", "r0"])
    cmap = plt.get_cmap("tab20")
    for idx, label in enumerate(["|0>", "|1>", "|2>"]):
        line = _line_by_label(fig.axes[0], label)
        assert to_rgba(line.get_color()) == pytest.approx(cmap(idx))
    plt.close(fig)


def test_plot_populations_explicit_color_override_wins(sample_result: SimulationResult) -> None:
    """An explicit colors override takes precedence over the default tab20 color."""
    fig = sample_result.plot_populations(trace_out=["q1", "r0"], colors={(1,): "#123456"})
    line = _line_by_label(fig.axes[0], "|1>")
    assert to_rgba(line.get_color()) == pytest.approx(to_rgba("#123456"))
    plt.close(fig)


def test_plot_state_population_respects_supplied_axes(sample_result: SimulationResult) -> None:
    """plot_state in population mode draws onto the figure owning the supplied axes."""
    fig, ax = plt.subplots()
    returned = sample_result.plot_state(0, trace_out=["q1", "r0"], ax=ax)
    assert returned is fig
    plt.close(fig)


def test_plot_state_dm_returns_figure_with_two_axes(sample_result: SimulationResult) -> None:
    """plot_state in "dm" mode returns a figure with two axes, one per real/imaginary panel."""
    fig = sample_result.plot_state(0, trace_out=["q1", "r0"], mode="dm")
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_wigner_matches_qutip_vacuum_reference(sample_result: SimulationResult) -> None:
    """The internal Wigner function computation matches QuTiP's reference for the vacuum state."""
    import qutip
    from quchip.viz.results import _wigner_from_density_matrix

    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0
    xvec = np.linspace(-3.0, 3.0, 41)

    actual = _wigner_from_density_matrix(rho, xvec, xvec)
    expected = qutip.wigner(qutip.Qobj(rho, dims=[[3], [3]]), xvec, xvec)

    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_plot_wigner_returns_figure_without_qutip_objects(sample_result: SimulationResult) -> None:
    """plot_wigner returns a Figure without requiring qutip objects as input."""
    fig = sample_result.plot_wigner(trace_out=["q0", "q1"])
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_device_plot_methods_return_figures_and_respect_axes() -> None:
    """Device plot_energy_levels and plot_wavefunction return figures and honor supplied axes."""
    device = Resonator(freq=6.0, levels=4, label="r0")

    fig = device.plot_energy_levels()
    assert isinstance(fig, Figure)
    plt.close(fig)

    fig, ax = plt.subplots()
    returned = device.plot_wavefunction(0, ax=ax)
    assert returned is fig
    plt.close(fig)
