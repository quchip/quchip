"""Visualization coverage: smoke tests for all nine plotters plus review regressions."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import qutip
from matplotlib.figure import Figure

import quchip as qc
from quchip.backend import SolverResult, get_default_backend
from quchip.results import SimulationResult
from quchip.viz._common import _normalize_time_index
from quchip.viz.chip import _collect_topology, _coupling_node_id, _device_node_id, _drive_node_id
from quchip.viz.results import _wigner_from_density_matrix

pytestmark = pytest.mark.viz


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def driven_chip() -> tuple[qc.Chip, qc.DuffingTransmon, qc.Resonator, qc.ChargeDrive]:
    """A small driven two-device chip: a capacitively coupled qubit and resonator with a charge drive."""
    q = qc.DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = qc.Resonator(freq=7.0, levels=4, label="r")
    coupling = qc.Capacitive(q, r, g=0.02, label="qr")
    chip = qc.Chip([q, r], couplings=[coupling], frame="rotating")
    drive = qc.ChargeDrive(target=q, label="d0")
    chip.wire(drive)
    return chip, q, r, drive


@pytest.fixture
def smoke_result(driven_chip: tuple[qc.Chip, qc.DuffingTransmon, qc.Resonator, qc.ChargeDrive]) -> SimulationResult:
    """A real simulate() result on driven_chip with dict-form e_ops and stored states."""
    chip, q, r, drive = driven_chip
    seq = qc.QuantumSequence(chip)
    seq.schedule(drive, envelope=qc.Gaussian(duration=20.0, amplitude=0.02, sigmas=3.0), freq=5.0)
    return seq.simulate(
        tlist=np.linspace(0.0, 20.0, 41),
        initial_state=chip.bare_state(q=0, r=0),
        e_ops=chip.e_ops(q="Z"),
    )


@pytest.fixture
def correlator_result() -> SimulationResult:
    """A free-evolution 2-qubit result whose e_ops include a tuple-keyed ZZ correlator.

    The initial state is an uncorrelated product state with q0 fixed in
    |0> (``<Z_q0> = +1``) and q1 in an equal superposition
    (``<Z_q1> = 0``), so ``<Z_q0> != <Z_q0 Z_q1>`` already at ``t=0`` —
    independent of any subsequent dynamics — giving a robust
    ground-truth pair for distinguishing the two traces.
    """
    q0 = qc.DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q0")
    q1 = qc.DuffingTransmon(freq=5.3, anharmonicity=-0.25, levels=2, label="q1")
    coupling = qc.Capacitive(q0, q1, g=0.01, label="qq")
    chip = qc.Chip([q0, q1], couplings=[coupling], frame="rotating")
    e_ops = chip.e_ops(q0="Z", q1="Z", correlators={("q0", "q1"): ("Z", "Z")})
    initial_state = chip.superposition({q0: 0, q1: 0}, {q0: 0, q1: 1})
    return qc.simulate(
        chip, [], np.linspace(0.0, 10.0, 11), initial_state=initial_state, e_ops=e_ops,
        check_truncation=False,
    )


def _density_matrix_from_weights(weights: dict[tuple[int, ...], float], dims: list[int]) -> object:
    """Build a diagonal (classically mixed) density matrix from basis-state weights, for a hand-built result."""
    backend = get_default_backend()
    density_matrix = None
    for basis_state, weight in weights.items():
        kets = [backend.basis(dim, level) for dim, level in zip(dims, basis_state)]
        ket = kets[0] if len(kets) == 1 else backend.tensor_states(*kets)
        projector = backend.matmul(ket, backend.dag(ket))
        term = weight * projector
        density_matrix = term if density_matrix is None else density_matrix + term
    return density_matrix


@pytest.fixture
def multi_device_state_result() -> SimulationResult:
    """A hand-built two-time, two-device SimulationResult with stored density matrices (no e_ops)."""
    dims = [3, 4]
    weights_t0 = {(0, 0): 0.5, (1, 1): 0.3, (2, 2): 0.2}
    weights_t1 = {(0, 0): 0.3, (1, 1): 0.3, (2, 2): 0.4}
    states = [_density_matrix_from_weights(weights_t0, dims), _density_matrix_from_weights(weights_t1, dims)]
    solver_result = SolverResult(
        times=np.array([0.0, 5.0]), states=states, expect=None, final_state=states[-1], solver="mesolve",
    )
    backend = get_default_backend()
    return SimulationResult(
        solver_result=solver_result, backend=backend, dims=dims,
        device_info=[("q", True), ("r", False)],
    )


# ---------------------------------------------------------------------------
# A. Smoke-Figure assertions for all nine plotter implementations.
# ---------------------------------------------------------------------------


def test_plot_graph_returns_html_path(driven_chip, tmp_path: Path) -> None:
    """plot_graph writes an HTML file and returns its path."""
    chip, _q, _r, _drive = driven_chip
    path = qc.plot_graph(chip, str(tmp_path / "graph.html"))
    assert Path(path).exists()
    assert path.endswith(".html")


def test_plot_energy_levels_chip_returns_figure(driven_chip) -> None:
    """plot_energy_levels(chip) returns a Figure."""
    chip, _q, _r, _drive = driven_chip
    fig = qc.plot_energy_levels(chip)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_energy_levels_device_returns_figure(driven_chip) -> None:
    """plot_energy_levels(device) returns a Figure."""
    _chip, q, _r, _drive = driven_chip
    fig = q.plot_energy_levels()
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_wavefunction_returns_figure(driven_chip) -> None:
    """plot_wavefunction(device, n) returns a Figure."""
    _chip, q, _r, _drive = driven_chip
    fig = qc.plot_wavefunction(q, 0)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_sequence_returns_figure(driven_chip) -> None:
    """plot_sequence returns a Figure."""
    chip, _q, _r, drive = driven_chip
    seq = qc.QuantumSequence(chip)
    seq.schedule(drive, envelope=qc.Gaussian(duration=20.0, amplitude=0.02, sigmas=3.0), freq=5.0)
    fig = qc.plot_sequence(seq)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_populations_returns_figure(smoke_result: SimulationResult) -> None:
    """plot_populations returns a Figure."""
    fig = qc.plot_populations(smoke_result)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_state_returns_figure(smoke_result: SimulationResult) -> None:
    """plot_state returns a Figure."""
    fig = qc.plot_state(smoke_result, -1)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_expectation_returns_figure(smoke_result: SimulationResult) -> None:
    """plot_expectation returns a Figure."""
    fig = qc.plot_expectation(smoke_result)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_wigner_returns_figure(smoke_result: SimulationResult) -> None:
    """plot_wigner returns a Figure once isolated to a single subsystem."""
    fig = qc.plot_wigner(smoke_result, -1, trace_out="q")
    assert isinstance(fig, Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# B (item A). Edge-pump topology: junction node present, chip renders.
# ---------------------------------------------------------------------------


def test_collect_topology_represents_coupling_as_junction_node() -> None:
    """Every coupling gets its own namespaced junction node splitting the device-device edge."""
    q0 = qc.DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = qc.DuffingTransmon(freq=5.3, anharmonicity=-0.25, levels=3, label="q1")
    coupling = qc.TunableCapacitive(q0, q1, g_0=0.01, label="tc")
    chip = qc.Chip([q0, q1], couplings=[coupling])

    nodes, edges = _collect_topology(chip)

    junction_id = _coupling_node_id("tc")
    assert junction_id in nodes
    assert nodes[junction_id]["kind"] == "coupling"
    edge_pairs = {(start, end) for start, end, _data in edges}
    assert (_device_node_id("q0"), junction_id) in edge_pairs
    assert (junction_id, _device_node_id("q1")) in edge_pairs


def test_collect_topology_attaches_edge_pump_to_coupling_junction() -> None:
    """A ParametricDrive (edge pump) attaches to its coupling's junction node, not a device."""
    q0 = qc.DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = qc.DuffingTransmon(freq=5.3, anharmonicity=-0.25, levels=3, label="q1")
    coupling = qc.TunableCapacitive(q0, q1, g_0=0.01, label="tc")
    chip = qc.Chip([q0, q1], couplings=[coupling])
    chip.wire(qc.ParametricDrive(coupling, label="pump"))

    nodes, edges = _collect_topology(chip)

    drive_id = _drive_node_id("pump")
    junction_id = _coupling_node_id("tc")
    assert drive_id in nodes
    assert (drive_id, junction_id) in {(start, end) for start, end, _data in edges}
    assert None not in nodes
    assert not any(start is None or end is None for start, end, _data in edges)


def test_collect_topology_excludes_edge_pump_when_couplings_excluded() -> None:
    """Excluding couplings from the render also drops their dependent edge-pump controls."""
    q0 = qc.DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = qc.DuffingTransmon(freq=5.3, anharmonicity=-0.25, levels=3, label="q1")
    coupling = qc.TunableCapacitive(q0, q1, g_0=0.01, label="tc")
    chip = qc.Chip([q0, q1], couplings=[coupling])
    chip.wire(qc.ParametricDrive(coupling, label="pump"))

    nodes, _edges = _collect_topology(chip, exclude={"coupling"})

    assert _coupling_node_id("tc") not in nodes
    assert _drive_node_id("pump") not in nodes


def test_collect_topology_crosstalk_never_resurrects_an_omitted_edge_pump() -> None:
    """A crosstalk edge onto an omitted edge pump is dropped rather than recreating the pump node."""
    q0 = qc.DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = qc.DuffingTransmon(freq=5.3, anharmonicity=-0.25, levels=3, label="q1")
    coupling = qc.TunableCapacitive(q0, q1, g_0=0.01, label="tc")
    chip = qc.Chip([q0, q1], couplings=[coupling])
    drive = qc.ChargeDrive(target=q0, label="d0")
    pump = qc.ParametricDrive(coupling, label="pump")
    chip.connect(qc.ControlEquipment(
        lines=[drive, pump],
        signal_chain=[qc.Crosstalk(source="pump", victim="d0", beta=0.1, theta=0.0, delay=0.0)],
    ))

    nodes, edges = _collect_topology(chip, exclude={"coupling"})

    pump_id = _drive_node_id("pump")
    assert pump_id not in nodes
    assert not any(pump_id in (start, end) for start, end, _data in edges)
    assert _drive_node_id("d0") in nodes


def test_plot_graph_renders_chip_with_edge_pump_control(tmp_path: Path) -> None:
    """plot_graph no longer crashes on a chip with an edge-pump (ParametricDrive) control."""
    q0 = qc.DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = qc.DuffingTransmon(freq=5.3, anharmonicity=-0.25, levels=3, label="q1")
    coupling = qc.TunableCapacitive(q0, q1, g_0=0.01, label="tc")
    chip = qc.Chip([q0, q1], couplings=[coupling])
    chip.wire(qc.ParametricDrive(coupling, label="pump"))

    path = qc.plot_graph(chip, str(tmp_path / "edge_pump.html"))

    assert Path(path).exists()


def test_plot_graph_rejects_unknown_layout(driven_chip, tmp_path: Path) -> None:
    """plot_graph raises ValueError listing valid layouts for an unrecognized layout string."""
    chip, _q, _r, _drive = driven_chip
    with pytest.raises(ValueError, match="force_atlas"):
        qc.plot_graph(chip, str(tmp_path / "bad.html"), layout="not_a_real_layout")


# ---------------------------------------------------------------------------
# C (item B). plot_wigner requires exactly one retained subsystem.
# ---------------------------------------------------------------------------


def test_plot_wigner_without_isolating_trace_out_raises_and_names_labels(smoke_result: SimulationResult) -> None:
    """plot_wigner on a multi-device state without an isolating trace_out raises, naming the retained labels."""
    with pytest.raises(ValueError, match="'q'") as exc_info:
        qc.plot_wigner(smoke_result)
    message = str(exc_info.value)
    assert "'q'" in message and "'r'" in message


def test_plot_wigner_isolated_to_single_subsystem_succeeds(smoke_result: SimulationResult) -> None:
    """plot_wigner succeeds once trace_out isolates exactly one subsystem."""
    fig = qc.plot_wigner(smoke_result, trace_out="q")
    assert isinstance(fig, Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# D (item E). Shared index normalization for plot_state / plot_wigner.
# ---------------------------------------------------------------------------


def test_normalize_time_index_accepts_full_negative_range() -> None:
    """_normalize_time_index accepts every index in [-N, N) and returns its non-negative form."""
    class _FakeResult:
        times = [0.0, 1.0, 2.0]

    result = _FakeResult()
    assert _normalize_time_index(result, -1) == 2
    assert _normalize_time_index(result, -3) == 0
    assert _normalize_time_index(result, 0) == 0
    assert _normalize_time_index(result, 2) == 2


def test_normalize_time_index_rejects_out_of_range() -> None:
    """_normalize_time_index raises IndexError outside [-N, N)."""
    class _FakeResult:
        times = [0.0, 1.0, 2.0]

    result = _FakeResult()
    with pytest.raises(IndexError, match="out of range"):
        _normalize_time_index(result, 3)
    with pytest.raises(IndexError, match="out of range"):
        _normalize_time_index(result, -4)


def test_plot_state_accepts_negative_index(smoke_result: SimulationResult) -> None:
    """plot_state(-1) succeeds — a regression for a prior explicit index<0 rejection."""
    fig = qc.plot_state(smoke_result, -1)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_state_out_of_range_index_raises(smoke_result: SimulationResult) -> None:
    """plot_state raises IndexError for an index outside [-N, N)."""
    with pytest.raises(IndexError, match="out of range"):
        qc.plot_state(smoke_result, 10_000)


def test_plot_wigner_accepts_negative_index(smoke_result: SimulationResult) -> None:
    """plot_wigner(-1) succeeds when isolated to a single subsystem."""
    fig = qc.plot_wigner(smoke_result, -1, trace_out="q")
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_wigner_out_of_range_index_raises(smoke_result: SimulationResult) -> None:
    """plot_wigner raises IndexError for an index outside [-N, N), with the same message shape as plot_state."""
    with pytest.raises(IndexError, match="out of range"):
        qc.plot_wigner(smoke_result, 10_000, trace_out="q")


# ---------------------------------------------------------------------------
# E (item H). Correlator tuple keys resolve as registered keys, not (key, index).
# ---------------------------------------------------------------------------


def test_plot_expectation_resolves_correlator_tuple_key(correlator_result: SimulationResult) -> None:
    """A tuple correlator key like ("q0", "q1") plots as itself, not as a (key, index) selector."""
    fig = qc.plot_expectation(correlator_result, keys=[("q0", "q1")])
    assert isinstance(fig, Figure)
    lines = fig.axes[0].lines
    assert len(lines) == 1
    assert lines[0].get_label() == "('q0', 'q1')"
    expected = np.real(np.asarray(correlator_result.observable_traces[("q0", "q1")].values))
    np.testing.assert_allclose(lines[0].get_ydata(), expected)
    plt.close(fig)


def test_plot_expectation_correlator_key_not_misread_as_index_selector(correlator_result: SimulationResult) -> None:
    """The correlator trace, not q0's single-device trace, is what gets plotted for the tuple key."""
    fig = qc.plot_expectation(correlator_result, keys=[("q0", "q1")])
    correlator_values = np.real(np.asarray(correlator_result.observable_traces[("q0", "q1")].values))
    q0_values = np.real(np.asarray(correlator_result.observable_traces["q0"].values))
    plotted = fig.axes[0].lines[0].get_ydata()
    np.testing.assert_allclose(plotted, correlator_values)
    assert not np.allclose(plotted, q0_values)
    plt.close(fig)


# ---------------------------------------------------------------------------
# F (item I). plot_state "dm" mode shares one symmetric normalization.
# ---------------------------------------------------------------------------


def test_plot_state_dm_heatmaps_share_symmetric_normalization(multi_device_state_result: SimulationResult) -> None:
    """Re(rho) and Im(rho) heatmaps in plot_state(mode="dm") share one vmin/vmax."""
    fig = qc.plot_state(multi_device_state_result, 0, trace_out="r", mode="dm")
    real_ax, imag_ax = fig.axes
    real_clim = real_ax.images[0].get_clim()
    imag_clim = imag_ax.images[0].get_clim()
    assert real_clim == imag_clim
    assert real_clim[0] == -real_clim[1]
    plt.close(fig)


# ---------------------------------------------------------------------------
# G (item G). Root lazy exports.
# ---------------------------------------------------------------------------


_LAZY_VIZ_NAMES = (
    "plot_energy_levels",
    "plot_expectation",
    "plot_graph",
    "plot_populations",
    "plot_sequence",
    "plot_state",
    "plot_wavefunction",
    "plot_wigner",
)


@pytest.mark.parametrize("name", _LAZY_VIZ_NAMES)
def test_viz_name_resolves_lazily_from_root(name: str) -> None:
    """Each plot_* name resolves via quchip's module __getattr__ and is callable."""
    assert callable(getattr(qc, name))


def test_viz_names_all_in_root_all() -> None:
    """Every plot_* name is listed in quchip.__all__."""
    for name in _LAZY_VIZ_NAMES:
        assert name in qc.__all__


def test_viz_names_all_in_root_dir() -> None:
    """dir(quchip) includes every lazily resolved plot_* name."""
    names = dir(qc)
    for name in _LAZY_VIZ_NAMES:
        assert name in names


# ---------------------------------------------------------------------------
# H. Durable numerical Wigner test — independent of the solver pipeline.
# ---------------------------------------------------------------------------


def test_wigner_from_density_matrix_matches_qutip_for_complex_superposition() -> None:
    """_wigner_from_density_matrix matches qutip.wigner to tight tolerance for a complex Fock superposition."""
    dim = 6
    psi = (qutip.basis(dim, 0) + 1j * qutip.basis(dim, 1)).unit()
    rho = psi * psi.dag()
    rho_np = rho.full()
    xvec = np.linspace(-4.0, 4.0, 81)

    actual = _wigner_from_density_matrix(rho_np, xvec, xvec)
    expected = qutip.wigner(rho, xvec, xvec, g=np.sqrt(2))

    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_wigner_from_density_matrix_matches_qutip_for_fock_state() -> None:
    """_wigner_from_density_matrix matches qutip.wigner to tight tolerance for a pure Fock state."""
    dim = 5
    rho = qutip.ket2dm(qutip.basis(dim, 2))
    rho_np = rho.full()
    xvec = np.linspace(-4.0, 4.0, 61)

    actual = _wigner_from_density_matrix(rho_np, xvec, xvec)
    expected = qutip.wigner(rho, xvec, xvec, g=np.sqrt(2))

    np.testing.assert_allclose(actual, expected, atol=1e-10)
