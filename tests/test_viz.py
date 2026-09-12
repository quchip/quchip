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
from quchip.viz.chip import _collect_topology, _coupling_node_id, _device_node_id, _drive_node_id
from quchip.viz.results import _wigner_from_density_matrix

pytestmark = pytest.mark.viz


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
    (``<Z_q1> = 0``), so ``<Z_q0> != <Z_q0 Z_q1>`` already at ``t=0``,
    independent of subsequent dynamics.
    """
    q0 = qc.DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q0")
    q1 = qc.DuffingTransmon(freq=5.3, anharmonicity=-0.25, levels=2, label="q1")
    coupling = qc.Capacitive(q0, q1, g=0.01, label="qq")
    chip = qc.Chip([q0, q1], couplings=[coupling], frame="rotating")
    e_ops = chip.e_ops(q0="Z", q1="Z", correlators={("q0", "q1"): ("Z", "Z")})
    initial_state = chip.superposition({q0: 0, q1: 0}, {q0: 0, q1: 1})
    return qc.simulate(
        chip, [], np.linspace(0.0, 10.0, 11), initial_state=initial_state, e_ops=e_ops,
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


def test_plot_graph_returns_html_path(driven_chip, tmp_path: Path) -> None:
    """plot_graph writes an HTML file and returns its path."""
    chip, _q, _r, _drive = driven_chip
    path = qc.plot_graph(chip, str(tmp_path / "graph.html"))
    assert Path(path).exists()
    assert path.endswith(".html")


def test_plot_graph_can_show_dressed_values(tmp_path: Path) -> None:
    """The dressed graph labels device frequencies and full-pull cross-Kerr values."""
    q = qc.DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = qc.Resonator(freq=7.0, levels=4, label="r")
    chip = qc.Chip([q, r], [qc.Capacitive(q, r, g=0.05, label="qr")])

    path = chip.plot_graph(str(tmp_path / "dressed.html"), values="dressed")
    content = Path(path).read_text()

    assert f"f01={float(chip.freq(q)):.6f} GHz" in content
    assert f"K={float(chip.kerr_matrix()[q, r]):.3g} GHz" in content
    assert "g=0.05 GHz" not in content


def test_plot_graph_can_show_bare_and_dressed_values(tmp_path: Path) -> None:
    """The combined view retains declared values alongside dressed observables."""
    q = qc.DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = qc.Resonator(freq=7.0, levels=4, label="r")
    chip = qc.Chip([q, r], [qc.Capacitive(q, r, g=0.05, label="qr")])

    path = chip.plot_graph(str(tmp_path / "both.html"), values="both")
    content = Path(path).read_text()

    assert "bare=5.000 GHz" in content
    assert f"f01={float(chip.freq(q)):.6f} GHz" in content
    assert "g=0.05 GHz" in content
    assert f"K={float(chip.kerr_matrix()[q, r]):.3g} GHz" in content


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
    assert nodes[_device_node_id("q0")]["label"] == "q0\n5.000 GHz"
    assert "g=0.01 GHz" in nodes[junction_id]["label"]
    assert "f01=" not in nodes[_device_node_id("q0")]["label"]
    assert "K=" not in nodes[junction_id]["label"]
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
    """plot_graph supports a chip with an edge-pump control."""
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


def test_plot_graph_rejects_unknown_value_mode(driven_chip, tmp_path: Path) -> None:
    """plot_graph rejects an unsupported value annotation mode."""
    chip, _q, _r, _drive = driven_chip
    with pytest.raises(ValueError, match="values must be one of"):
        qc.plot_graph(chip, str(tmp_path / "bad-values.html"), values="effective")


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


def test_plot_state_dm_heatmaps_share_symmetric_normalization(multi_device_state_result: SimulationResult) -> None:
    """Re(rho) and Im(rho) heatmaps in plot_state(mode="dm") share one vmin/vmax."""
    fig = qc.plot_state(multi_device_state_result, 0, trace_out="r", mode="dm")
    real_ax, imag_ax = fig.axes
    real_clim = real_ax.images[0].get_clim()
    imag_clim = imag_ax.images[0].get_clim()
    assert real_clim == imag_clim
    assert real_clim[0] == -real_clim[1]
    plt.close(fig)


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


def _fridge_chip() -> tuple[qc.Chip, qc.PortNetwork]:
    resonator = qc.Resonator(freq=6.0, levels=4, label="r")
    network = qc.PortNetwork(label="fridge")
    port = network.port("coupler", target=resonator, rate=0.04)
    circulator = network.circulator("circ")
    isolator = network.isolator("iso")
    loss = network.attenuator("cold_loss", eta=0.1)
    cable = network.delay("cable", duration=3.2)
    hemt = network.amplifier("hemt", gain=100.0, added_noise=2.0)
    network.link(loss, circulator.port(1))
    network.link(port, circulator.port(2))
    network.link(circulator.port(3), isolator, cable, hemt)
    network.expose("drive", at=loss.port(1))
    network.expose("readout", at=hemt.port(2))
    return qc.Chip([resonator], port_network=network), network


def test_plot_port_network_draws_every_component_and_plane() -> None:
    chip, network = _fridge_chip()

    fig = qc.viz.plot_port_network(chip)
    texts = {text.get_text() for text in fig.findobj(matplotlib.text.Text)}

    assert isinstance(fig, Figure)
    assert {"coupler", "circ", "iso", "cold_loss", "cable", "hemt", "drive", "readout"} <= texts
    assert any("port -> r" in text for text in texts)
    assert {"load", "vacuum_1", "vacuum_2"} <= texts
    assert any("3.2" in text for text in texts)

    quiet = qc.viz.plot_port_network(network, show_hidden=False)
    quiet_texts = {text.get_text() for text in quiet.findobj(matplotlib.text.Text)}
    assert not ({"load", "vacuum_1", "vacuum_2"} & quiet_texts)

    _, ax = plt.subplots()
    assert qc.viz.plot_port_network(chip, ax=ax) is ax.figure
    plt.close("all")


def test_plot_sparameters_supports_matrix_kinds_and_axis_selection() -> None:
    resonator = qc.Resonator(freq=6.0, levels=6, label="r")
    network = qc.PortNetwork(label="line")
    network.port("in", target=resonator, rate=0.04)
    network.port("out", target=resonator, rate=0.02)
    chip = qc.Chip([resonator], port_network=network)
    vna = qc.VNA(chip)
    frequencies = np.linspace(5.95, 6.05, 21)
    result = vna.sweep(frequencies)

    fig = qc.viz.plot_sparameters(result)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    assert len(fig.axes[0].get_lines()) == 4
    assert "dB" in fig.axes[0].get_ylabel()

    magnitude = qc.viz.plot_sparameters(result, pairs=[("out", "in")], kind="magnitude")
    assert len(magnitude.axes) == 1 and len(magnitude.axes[0].get_lines()) == 1
    iq = qc.viz.plot_sparameters(result, kind="iq")
    assert len(iq.axes[0].get_lines()) == 4

    mapped = vna.sweep(frequencies, qc.Sweep([5.98, 6.0, 6.02], name="r.freq"))
    selected = qc.viz.plot_sparameters(mapped, pairs=[("in", "in")], select={"r.freq": 2})
    line = selected.axes[0].get_lines()[0]
    np.testing.assert_allclose(line.get_ydata(), 20 * np.log10(np.abs(mapped.s("in", "in")[2])))

    with pytest.raises(ValueError, match="kind"):
        qc.viz.plot_sparameters(result, kind="smith")
    with pytest.raises(TypeError, match="SParameterResult"):
        qc.viz.plot_sparameters(vna.finite_power(6.0, 0.01, input="in"))
    plt.close("all")


def test_plot_sparameters_rejects_result_without_free_sweep_axis() -> None:
    """A single-frequency result has nothing to plot against and says so."""
    resonator = qc.Resonator(freq=6.0, levels=4, label="r")
    network = qc.PortNetwork(label="line")
    network.port("in", target=resonator, rate=0.02)
    result = qc.VNA(qc.Chip([resonator], port_network=network)).sweep(6.0)
    with pytest.raises(ValueError, match="sweep axis"):
        qc.viz.plot_sparameters(result)


def test_plot_sparameters_rejects_selecting_frequency() -> None:
    """select= indexes only non-frequency axes; frequency stays the x axis."""
    resonator = qc.Resonator(freq=6.0, levels=4, label="r")
    network = qc.PortNetwork(label="line")
    network.port("in", target=resonator, rate=0.02)
    result = qc.VNA(qc.Chip([resonator], port_network=network)).sweep(
        np.linspace(5.9, 6.1, 5), qc.Sweep(np.array([5.99, 6.01]), name="r.freq")
    )
    with pytest.raises(ValueError, match="frequency"):
        qc.viz.plot_sparameters(result, select={"frequency": 1})


def test_plot_port_network_draws_asymmetric_planes_with_two_arrows() -> None:
    """A plane exposed with separate input and output terminals gets one arrow per leg."""
    resonator = qc.Resonator(freq=6.0, levels=4, label="r")
    network = qc.PortNetwork(label="line")
    port = network.port("p", target=resonator, rate=0.02)
    circ = network.circulator("circ")
    pad = network.attenuator("pad", eta=0.5)
    network.link(port, circ.port(2))
    network.link(pad, circ.port(1))
    network.expose("through", input=pad.port(1).input, output=circ.port(3).output)
    fig = qc.viz.plot_port_network(network)
    arrows = [a for a in fig.findobj(matplotlib.text.Annotation) if a.arrow_patch is not None]
    styles = sorted(type(a.arrow_patch.get_arrowstyle()).__name__ for a in arrows)
    assert styles.count("CurveB") == 2
    assert styles.count("CurveAB") == 2

    same_component = qc.PortNetwork(label="one")
    circ = same_component.circulator("circ")
    same_component.link(same_component.port("p", target=resonator, rate=0.02), circ.port(2))
    same_component.expose("split", input=circ.port(1).input, output=circ.port(3).output)
    fig = qc.viz.plot_port_network(same_component)
    arrows = [a for a in fig.findobj(matplotlib.text.Annotation) if a.arrow_patch is not None]
    assert sorted(type(a.arrow_patch.get_arrowstyle()).__name__ for a in arrows) == ["CurveAB", "CurveB", "CurveB"]
