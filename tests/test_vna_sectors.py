"""Finite-probe VNA solves on the product states with total excitation N <= K."""

import warnings

import numpy as np
import pytest

from scipy import sparse

import quchip.analysis.measurement as measurement
import quchip.analysis.vna as vna_analysis
import quchip.engine.routes as routes
import quchip.backend._sector_stationary as sector_solver
from quchip import (RWA, VNA, Bath, Capacitive, ChargeBasisTransmon, Chip, DuffingTransmon, ExcitationCutoff,
                    PortNetwork, Resonator)
from quchip.declarative import CollapseChannel, CouplingModel, Scalar, parameter

OFFSETS = np.array([-0.02, -1e-3, -1e-5, 0.0, 1e-5, 1e-3, 0.02])


def feedline(*, dephasing=True, thermal_occupation=None, space=None, backend="qutip", drive=False):
    """Return a qubit and two resonators on a cascaded feedline, with 27 product states and N <= 6.

    ``drive=True`` adds a port ``"xy"`` on the qubit.
    """
    q = DuffingTransmon(freq=5.3, anharmonicity=-0.26, levels=3, label="q", T1=2e3, T2=1.5e3 if dephasing else None)
    r0 = Resonator(freq=6.5, levels=3, label="r0", internal_quality_factor=1e5, thermal_occupation=thermal_occupation)
    r1 = Resonator(freq=6.6, levels=3, label="r1", internal_quality_factor=1e5)
    network = PortNetwork(label="feedline")
    ports = [network.port(r.label, target=r, rate=2 * np.pi * 0.004) for r in (r0, r1)]
    network.cascade(*ports)
    network.expose("feed", input=ports[0], output=ports[-1])
    if drive:
        network.expose("xy", at=network.port("q", target=q, rate=2e-3))
    return Chip([q, r0, r1], [Capacitive(q, r0, g=0.06), Capacitive(r0, r1, g=0.01)], port_network=network,
                approximation=RWA(), space=space, backend=backend)


def probe_frequencies():
    """Return probe frequencies near the dressed first resonator."""
    return float(feedline().freq("r0")) + np.array([-0.002, 0.0007])


def dark_pair(max_total=None):
    """Return two resonant qubits on one bus; ``2|10> - |01>`` is dark to it."""
    lower = np.array([[0.0, 1.0], [0.0, 0.0]])
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q0")
    q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q1")
    network = PortNetwork(label="line")
    network.port("bus", target=(q0, q1), rate=1.0, operator=np.kron(lower, np.eye(2)) + 2 * np.kron(np.eye(2), lower))
    return Chip([q0, q1], port_network=network,
                space=None if max_total is None else ExcitationCutoff(max_total=max_total))


def readout_pairs(count, backend="qutip"):
    """Return ``count`` transmon-resonator pairs on one cascaded feedline, with ``9**count`` product states."""
    qubits = [DuffingTransmon(freq=5.3 - 0.1 * i, anharmonicity=-0.26, levels=3, label=f"q{i}", T1=3e4)
              for i in range(count)]
    resonators = [Resonator(freq=6.5 + 0.1 * i, levels=3, label=f"r{i}", internal_quality_factor=2e5)
                  for i in range(count)]
    network = PortNetwork(label="feedline")
    ports = [network.port(r.label, target=r, rate=2 * np.pi * 0.002) for r in resonators]
    network.cascade(*ports)
    network.expose("feed", input=ports[0], output=ports[-1])
    couplings = [Capacitive(q, r, g=0.04) for q, r in zip(qubits, resonators, strict=True)]
    return Chip(qubits + resonators, couplings, port_network=network, frame=5.2, approximation=RWA(), backend=backend)


class Shelving(CouplingModel):
    """Cross-Kerr shift and a jump that moves three photons from ``a`` into ``b``."""

    chi: Scalar = parameter(unit="GHz")
    rate: Scalar = parameter()

    def interaction(self, a, b, p):
        return p.chi * a.n * b.n

    def dissipation(self, a, b, p):
        return (CollapseChannel((a.a @ a.a @ a.a) * (b.adag @ b.adag @ b.adag), p.rate, "shelve"),)


def shelving_chip():
    """Return a probed resonator that shelves population in a slow mode, with 68 product states."""
    a = Resonator(freq=6.0, levels=17, label="a")
    b = Resonator(freq=4.0, levels=4, label="b", T1=1e6)
    network = PortNetwork(label="line")
    network.expose("line", at=network.port("a", target=a, rate=0.05))
    return Chip([a, b], [Shelving(a, b, chi=0.04, rate=0.05)], port_network=network, approximation=RWA())


def assert_same_measurement(actual, expected, *, rtol):
    """Compare means, mode values and every noise spectrum with ``rtol`` of each quantity's largest value."""
    def close(a, b):
        a, b = np.asarray(a), np.asarray(b)
        np.testing.assert_allclose(a, b, rtol=rtol, atol=rtol * np.max(np.abs(b)))

    for name in ("values", "mode_amplitudes", "photon_numbers", "mode_frequencies"):
        close(getattr(actual, name), getattr(expected, name))
    assert actual.noise_components.keys() == expected.noise_components.keys()
    for name, (white, excess) in expected.noise_components.items():
        close(actual.noise_components[name][0], white)
        close(actual.noise_components[name][1], excess)


def cutoff_warnings(caught):
    """Return the automatic-cutoff warnings among ``caught``."""
    return [item for item in caught if "not on the full product space" in str(item.message)]


def test_complete_space_matches_the_product_space(monkeypatch):
    """With N <= 6, the sector route gives the product-space means, mode values and IQ spectra, and does not warn."""
    # A cutoff that keeps every state is no cutoff, so the automatic route starts at the largest total instead.
    monkeypatch.setattr(routes, "AUTOMATIC_SECTOR_STATES", 0)
    monkeypatch.setattr(routes, "FIRST_CUTOFF", 6)
    chip, frequencies, beta = feedline(), probe_frequencies(), 0.018 * np.exp(0.7j)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sectors = VNA(chip).measure(frequencies, beta, noise_frequencies=OFFSETS)
    product = VNA(chip).measure(frequencies, beta, noise_frequencies=OFFSETS, options={})
    assert not cutoff_warnings(caught)
    assert all(item["solver"] == "sector_stationary" and item["excitation_cutoff"] == 6
               and item["retained_dimension"] == 27 and item["cutoff_check"] is None for item in sectors.diagnostics)
    assert all(item["solver"] != "sector_stationary" for item in product.diagnostics)
    # Pure dephasing gives anomalous IQ correlations.
    excess = np.asarray(product.noise_components["device.correlations"][1])
    assert np.max(np.abs(excess[..., 0, 0] - excess[..., 1, 1])) > 1e-3 * np.max(np.abs(excess))
    assert_same_measurement(sectors, product, rtol=1e-8)


def test_a_pump_at_the_probe_frequency_solves_on_the_sectors(monkeypatch):
    """A pump on the qubit at the probe frequency joins the sector drive; the result is the product-space one."""
    monkeypatch.setattr(routes, "AUTOMATIC_SECTOR_STATES", 0)
    monkeypatch.setattr(routes, "FIRST_CUTOFF", 6)
    frequency, beta = float(probe_frequencies()[1]), 0.018 * np.exp(0.7j)
    sector, product = (VNA(feedline(drive=True), ports=["feed"]) for _ in range(2))
    for vna in (sector, product):
        vna.pump("xy", freq=frequency, amplitude=0.03j)
    solved = sector.measure(frequency, beta, noise_frequencies=OFFSETS)
    reference = product.measure(frequency, beta, noise_frequencies=OFFSETS, options={})
    assert solved.diagnostics[0]["solver"] == "sector_stationary"
    assert_same_measurement(solved, reference, rtol=1e-7)
    # A pump at another frequency leaves no static frame for either route.
    detuned = VNA(feedline(drive=True), ports=["feed"])
    detuned.pump("xy", freq=frequency + 0.01, amplitude=0.03j)
    with pytest.raises(ValueError, match="distinct stationary tones"):
        detuned.finite_power(frequency, beta)


def test_sector_route_matches_the_compact_engine_at_the_same_cutoff():
    """At N <= 2, the sector solver and the direct compact solve give the same truncated model."""
    chip, frequencies, beta = feedline(space=ExcitationCutoff(max_total=2)), probe_frequencies(), 0.018 * np.exp(0.7j)
    sectors = VNA(chip).measure(frequencies, beta, noise_frequencies=OFFSETS)
    compact = VNA(chip).measure(frequencies, beta, noise_frequencies=OFFSETS, options={})
    assert sectors.diagnostics[0]["cutoff_choice"] == "explicit"
    assert compact.diagnostics[0]["solver"] == "stationary_resolvent"
    assert sectors.diagnostics[0]["retained_dimension"] == compact.diagnostics[0]["retained_dimension"] == 10
    assert_same_measurement(sectors, compact, rtol=1e-8)


def test_two_outputs_and_a_charge_basis_transmon_match_the_compact_engine():
    """A hybrid, a phase shift and a cable correlate two outputs; the transmon keeps its charge basis.

    At N <= 2, the sector route and the direct compact solve give the same means, mode values and cross-spectra.
    """
    q = ChargeBasisTransmon(E_C=0.25, E_J=12.0, n_g=0.1, num_basis=5, basis="native", label="q")
    r0 = Resonator(freq=6.5, levels=3, label="r0", internal_quality_factor=1e5)
    r1 = Resonator(freq=6.52, levels=3, label="r1", internal_quality_factor=1e5)
    network = PortNetwork(label="hybrid")
    p0 = network.port("r0", target=r0, rate=2 * np.pi * 0.004)
    p1 = network.port("r1", target=r1, rate=2 * np.pi * 0.003)
    mix, shift = network.hybrid90("mix"), network.phase_shift("shift", phase=0.7)
    cable = network.delay("cable", duration=3.0)
    network.connect(p0.output, mix.input_terminal("left"))
    network.connect(p1.output, mix.input_terminal("right"))
    network.cascade(mix.output_terminal("left"), shift)
    # The right output of the hybrid also drives the second resonator through a two-way cable.
    network.connect(mix.output_terminal("right"), cable.input_terminal("1"))
    network.connect(cable.output_terminal("1"), p1.input)
    network.expose("left", input=p0.input, output=shift)
    network.expose("right", at=cable.port(2))
    chip = Chip([q, r0, r1], [Capacitive(q, r0, g=0.06), Capacitive(r0, r1, g=0.01)], port_network=network,
                approximation=RWA(), space=ExcitationCutoff(max_total=2))
    assert chip.resolve_basis("q") == "native"
    frequencies, beta = float(chip.freq("r0")) + np.array([-0.002, 0.0007]), 0.018 * np.exp(0.7j)
    sectors, compact = (VNA(chip).measure(frequencies, beta, input="left", outputs=["left", "right"],
                                          noise_frequencies=OFFSETS, options=options) for options in (None, {}))
    assert sectors.diagnostics[0]["solver"] == "sector_stationary"
    assert compact.diagnostics[0]["solver"] == "stationary_resolvent"
    (excess,) = (np.asarray(excess) for _, excess in compact.noise_components.values())
    assert np.max(np.abs(excess[..., :2, 2:])) > 0.1 * np.max(np.abs(excess))
    assert_same_measurement(sectors, compact, rtol=1e-8)


def test_a_dark_state_converges_only_when_the_probe_reaches_it(monkeypatch):
    """N = 2 couples the dark state back to the bus; N <= 1 leaves a second stationary state and raises."""
    # The automatic route starts at N <= 2, which is the whole space here, so it is exact and does not warn.
    monkeypatch.setattr(routes, "AUTOMATIC_SECTOR_STATES", 0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        complete = VNA(dark_pair()).finite_power(5.0, 0.1)
    product = VNA(dark_pair()).finite_power(5.0, 0.1, options={})
    assert not cutoff_warnings(caught)
    assert complete.diagnostics[0]["solver"] == "sector_stationary" and complete.diagnostics[0]["cutoff_check"] is None
    np.testing.assert_allclose(complete.values, product.values, rtol=1e-9, atol=1e-12)
    with pytest.raises(ValueError, match="more than one stationary state"):
        VNA(dark_pair(max_total=1)).finite_power(5.0, 0.1)


def bus_pair_generator(max_total, *, decay=0.0, conserving=()):
    """Return the sector generator of two resonant qubits on the bus ``σ_1 + 2 σ_2``, with optional own decay."""
    lower = np.array([[0.0, 1.0], [0.0, 0.0]])
    first, second = np.kron(lower, np.eye(2)), np.kron(np.eye(2), lower)
    lowering = [first + 2 * second] + ([np.sqrt(decay) * first, np.sqrt(decay) * second] if decay else [])
    # The product order |00>, |01>, |10>, |11> lists the states by N, and N <= 1 keeps the first three.
    size = 3 if max_total == 1 else 4
    return sector_solver.SectorGenerator([1, 2, 1][:max_total + 1], np.zeros((size, size)),
                                         [op[:size, :size] for op in lowering], [op[:size, :size] for op in conserving])


def dense_stationary(lowering, drive):
    """Return the stationary state of ``H = i(c* L - c L†)`` with jumps ``L`` by a dense solve for ``ρ - |0><0|``."""
    dimension = lowering[0].shape[0]
    identity = np.eye(dimension)
    hamiltonian = sum(1j * (np.conj(c) * op - c * op.conj().T) for c, op in zip(drive, lowering))
    generator = -1j * (np.kron(hamiltonian, identity) - np.kron(identity, hamiltonian.T))
    for op in lowering:
        decay = op.conj().T @ op
        generator += np.kron(op, op.conj()) - 0.5 * np.kron(decay, identity) - 0.5 * np.kron(identity, decay.T)
    generator[0] = identity.reshape(-1)
    vacuum = np.zeros(dimension**2, dtype=complex)
    vacuum[0] = 1.0
    correction = np.linalg.solve(generator, vacuum - generator @ vacuum)
    return (vacuum + correction).reshape(dimension, dimension)


@pytest.mark.unit
@pytest.mark.parametrize("beta, conserving", [(0.0, ()), (0.1, (np.diag([0.0, 1.0, 1.0, 2.0]),)), (0.1, (np.eye(4),))],
                         ids=["no_probe", "number_dephasing", "identity_jump"])
def test_the_uniqueness_check_finds_a_second_stationary_state(beta, conserving):
    """At N <= 1 the dark state stays stationary without a probe, and conserving jumps do not hide it."""
    generator = bus_pair_generator(1, conserving=conserving)
    with pytest.raises(ValueError, match="more than one stationary state"):
        generator.stationary(0.0, [beta])


@pytest.mark.unit
def test_the_second_start_separates_two_symmetric_dark_states():
    """Jumps from b into two dark states make each stationary; a symmetric start would weight both equally."""
    def transition(row, column):
        matrix = np.zeros((4, 4))
        matrix[row, column] = 1.0
        return matrix

    # The states are the vacuum, b and the two dark states, all with N = 1 except the vacuum.
    generator = sector_solver.SectorGenerator([1, 3], np.zeros((4, 4)), [transition(0, 1)],
                                              [transition(2, 1), transition(3, 1)])
    with pytest.raises(ValueError, match="more than one stationary state"):
        generator.stationary(0.0, [0.1])


@pytest.mark.unit
def test_two_starts_resolve_a_slowly_emptied_dark_state():
    """Qubit decay at 1e-6 / ns empties the dark state slowly; the solve matches a dense one, and raises without it."""
    beta, vacuum = 1e-6, np.diag([1.0, 0.0, 0.0, 0.0])
    lower = np.array([[0.0, 1.0], [0.0, 0.0]])
    first, second = np.kron(lower, np.eye(2)), np.kron(np.eye(2), lower)
    expected = dense_stationary([first + 2 * second, np.sqrt(1e-6) * first, np.sqrt(1e-6) * second], [beta, 0.0, 0.0])
    solved = bus_pair_generator(2, decay=1e-6).stationary(0.0, [beta, 0.0, 0.0])
    np.testing.assert_allclose(solved.state - vacuum, expected - vacuum, rtol=0,
                               atol=1e-8 * np.linalg.norm(expected - vacuum))
    # Without it the dark state empties at a rate of order beta^2, which the solve cannot resolve.
    with pytest.raises(ValueError, match="cannot resolve"):
        bus_pair_generator(2).stationary(0.0, [beta])


@pytest.mark.unit
def test_the_uniqueness_bound_of_a_large_sector_counts_the_channels_of_each_device():
    """1100 one-excitation states share one port and each decay on their own; the bound proves no dark state."""
    count = 1100
    shape = (count + 1, count + 1)
    port = sparse.csr_matrix((np.full(count, 0.1), (np.zeros(count, dtype=int), np.arange(1, count + 1))), shape)
    local = [sparse.csr_matrix(([1e-2], ([0], [state])), shape) for state in range(1, count + 1)]
    # The port alone leaves count - 1 dark states. With the local channels the Gershgorin bound of the sum is
    # negative, but each state decays at 1e-4 / ns on its own.
    assert sector_solver.has_dark_state([1, count], [port])
    assert not sector_solver.has_dark_state([1, count], [port, *local])


@pytest.mark.unit
def test_a_solve_that_does_not_converge_raises(monkeypatch):
    """A residual above the limit raises instead of returning an unconverged state."""
    monkeypatch.setattr(sector_solver, "_TOLERANCE", 1e-300)
    monkeypatch.setattr(sector_solver, "_RESIDUAL_LIMIT", 1e-300)
    monkeypatch.setattr(sector_solver, "_CYCLES", 2)
    with pytest.raises(ValueError, match="did not converge"):
        VNA(feedline(space=ExcitationCutoff(max_total=1))).finite_power(6.5, 0.01)


def test_automatic_cutoff_finds_population_shelved_above_the_first_cutoff():
    """N <= 2 never reaches the shelving jump; the comparison with K + 1 raises K until the means agree."""
    chip, frequencies, beta = shelving_chip(), 6.0 + np.array([-0.004, 0.0, 0.004]), 0.02
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        automatic = VNA(chip).finite_power(frequencies, beta)
    product = VNA(chip).finite_power(frequencies, beta, options={})
    first = VNA(chip.with_space(ExcitationCutoff(max_total=2))).finite_power(frequencies, beta)
    (warning,) = cutoff_warnings(caught)
    diagnostics = automatic.diagnostics[0]
    assert diagnostics["cutoff_choice"] == "automatic" and diagnostics["excitation_cutoff"] == 4
    assert diagnostics["cutoff_check"]["status"] == "passed" and diagnostics["cutoff_check"]["compared_with"] == 5
    assert "N <= 4" in str(warning.message) and "N <= 5" in str(warning.message)
    assert warning.filename == __file__
    assert np.max(np.abs(np.asarray(first.values) - np.asarray(product.values))) > 0.1 * beta
    np.testing.assert_allclose(automatic.values, product.values, rtol=0, atol=1e-6 * beta)


@pytest.mark.unit
def test_the_cutoff_check_compares_every_grid_point():
    """A small point that changes fails the check, while changes below the floor pass."""
    def points(means):
        return [measurement._SectorPoint(None, np.atleast_1d(mean), np.zeros(1), np.zeros(1), None, {})
                for mean in means]

    # Relative to the grid maximum the change is 1e-7, but relative to its own point it is 9e-6.
    assert measurement._sector_change(points([1.0, 1e-2]), points([1.0, 1e-2 * (1 + 1e-5)]), False) > 1e-6
    # A point at 1e-10 of the maximum is below the floor of 1e-9 absolute, even when it doubles.
    assert measurement._sector_change(points([1.0, 1e-10]), points([1.0, 2e-10]), False) < 1e-6


@pytest.mark.unit
def test_the_cutoff_check_compares_every_noise_offset():
    """The noise check fails on a change in the background beside a narrow peak, not only at the peak."""
    background = np.full((7, 2, 2), 1e-2)
    background[3] = 1.0

    def points(excess):
        return [measurement._SectorPoint(None, np.ones(1), np.zeros(1), np.zeros(1), excess, {})]

    shifted = background.copy()
    shifted[0] *= 1 + 1e-5
    assert measurement._sector_change(points(background), points(shifted), True) > 1e-6
    assert measurement._sector_change(points(background), points(shifted), False) == 0.0


@pytest.mark.unit
def test_the_noise_solve_takes_each_missing_negative_offset(monkeypatch):
    """A grid that is symmetric only to round-off solves the negative of its first offset, not its reversed neighbor."""
    offsets = np.array([-0.02000000000001, -0.01, 0.0, 0.01, 0.02])
    grids, resolvent = [], sector_solver.SectorGenerator.resolvent

    def record(self, frequency, drive, source, grid, observables):
        grids.append(np.asarray(grid))
        return resolvent(self, frequency, drive, source, grid, observables)

    monkeypatch.setattr(sector_solver.SectorGenerator, "resolvent", record)
    VNA(feedline(space=ExcitationCutoff(max_total=2))).measure(6.5, 0.01, noise_frequencies=offsets)
    assert grids and all(np.any(grid == 0.02000000000001) and np.any(grid == -0.02) for grid in grids)


@pytest.mark.unit
def test_the_output_checks_compare_every_offset_and_delay():
    """A small sample beside a large one fails the check, although the norm over the grid barely changes."""
    zeros, peak, shifted = np.zeros(2), np.array([1.0, 0.0]), np.array([1.0, 5e-7])
    # At the second sample, the change is 5e-7 over the floor, 1e-3 of the peak.
    spectra = [(peak, zeros, 1.0, 0.5), (shifted, zeros, 1.0, 0.5), (shifted, zeros, 1.0, 0.5)]
    assert vna_analysis._spectrum_change(*spectra[:2]) == pytest.approx(5e-7 / (5e-7 + 1e-3))
    cutoff, _, check = routes.automatic_cutoff(4, lambda total, lower: spectra[total - 2],
                                               vna_analysis._spectrum_change, fits=lambda total: True,
                                               states=lambda total: total)
    assert cutoff == 3 and check.status == "passed"
    assert vna_analysis._correlation_change((peak, 1.0, 2.0), (shifted, 1.0, 2.0)) > routes.CUTOFF_TOLERANCE
    # Each intensity sets its own scale, so a small output intensity is compared with itself.
    assert vna_analysis._correlation_change((peak, 1.0, 1e-6), (peak, 1.0, 1e-6 * (1 + 1e-5))) > 1e-6


@pytest.mark.unit
@pytest.mark.parametrize("method", ["finite_power", "measure"])
@pytest.mark.parametrize("case", ["passed", "noise", "memory"])
def test_automatic_cutoff_warns_once_at_the_call(monkeypatch, method, case):
    """One warning per call names K and the result of its check, at the caller's line."""
    monkeypatch.setattr(routes, "AUTOMATIC_SECTOR_STATES", 0)
    if case == "noise":
        monkeypatch.setattr(measurement, "_NOISE_CHECK_STATES", 0)
    if case == "memory":
        monkeypatch.setattr(measurement, "fits_memory", lambda size: False)
    vna, frequencies = VNA(feedline(dephasing=False)), probe_frequencies()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if method == "measure":
            result = vna.measure(frequencies, 1e-3, noise_frequencies=OFFSETS)
        else:
            result = vna.finite_power(frequencies, 1e-3)
    (warning,) = cutoff_warnings(caught)
    text, check = str(warning.message), result.diagnostics[0]["cutoff_check"]
    assert warning.filename == __file__
    assert f"N <= {result.diagnostics[0]['excitation_cutoff']}," in text
    if case == "memory":
        assert check["status"] == "not run" and "not verified" in text
    else:
        assert check["status"] == "passed" and check["change"] <= 1e-6
        assert ("noise spectra by at most" in text) == (method == "measure" and case == "passed")
        assert ("noise spectra were not compared" in text) == (method == "measure" and case == "noise")


def test_modes_are_reported_in_the_probe_frame_of_the_solve():
    """A shared loss channel drives a mode detuned by 1 GHz at the probe frequency; both routes report that frame."""
    def chip(space=None):
        a = Resonator(freq=6.0, levels=3, label="a", T1=100.0)
        b = Resonator(freq=5.0, levels=3, label="b", T1=100.0)
        network = PortNetwork(label="line")
        network.expose("line", at=network.port("a", target=a, rate=0.1))
        return Chip([a, b], port_network=network, baths=[Bath("collective_decay", targets=[a, b], rate=0.1)],
                     approximation=RWA(), space=space)

    offsets = [-0.01, -0.001, 0.0, 0.001, 0.01]
    sectors = VNA(chip(ExcitationCutoff(max_total=2))).measure(6.0, 0.01, noise_frequencies=offsets)
    harmonic = VNA(chip()).measure(6.0, 0.01, noise_frequencies=offsets)
    assert sectors.diagnostics[0]["solver"] == "sector_stationary"
    assert harmonic.diagnostics[0]["solver"] == "linear_response"
    np.testing.assert_array_equal(sectors.mode_frequencies, [6.0, 6.0])
    np.testing.assert_array_equal(harmonic.mode_frequencies, [6.0, 6.0])
    # Three levels and N <= 2 differ from the harmonic modes only at order |<a>|^2.
    assert abs(sectors.mode_amplitude("b")) > 1e-4
    np.testing.assert_allclose(sectors.mode_amplitudes, harmonic.mode_amplitudes, rtol=1e-5)


def test_an_explicit_cutoff_with_heating_solves_the_compact_model():
    """A thermal occupation raises N, so the compact model is solved directly; it converges to the product space."""
    chip, frequencies, beta = feedline(dephasing=False, thermal_occupation=0.02), probe_frequencies(), 0.01
    product = VNA(chip).measure(frequencies, beta, noise_frequencies=OFFSETS, options={})
    errors = []
    for max_total, size in ((3, 17), (4, 23)):
        compact = VNA(chip.with_space(ExcitationCutoff(max_total=max_total))).measure(
            frequencies, beta, noise_frequencies=OFFSETS)
        assert compact.diagnostics[0]["solver"] == "stationary_resolvent"
        assert compact.diagnostics[0]["excitation_cutoff"] == max_total
        assert compact.diagnostics[0]["retained_dimension"] == size
        errors.append(np.max(np.abs(np.asarray(compact.values) - np.asarray(product.values))))
    assert 0 < errors[1] < 0.05 * errors[0]
    assert_same_measurement(compact, product, rtol=1e-6)


@pytest.mark.validation
def test_twelve_pairs_approach_the_small_signal_response_as_beta_squared():
    """Of 3^24 product states, N <= 2 keeps 325; mean/beta - S follows the third-order term c |beta|^2."""
    chip = readout_pairs(12)
    frequency = float(chip.freq("r0"))
    small_signal = complex(np.asarray(VNA(chip).sweep(frequency).matrix)[0, 0])
    truncated = VNA(chip.with_space(ExcitationCutoff(max_total=2)))
    deviations = [(complex(np.ravel(truncated.finite_power(frequency, beta).values)[0]) / beta - small_signal) / beta**2
                  for beta in (2.5e-5, 5e-5)]
    # N <= 2 holds the whole third-order response, so the normalized deviation is the same at both amplitudes.
    assert abs(deviations[0]) > 0.1
    np.testing.assert_allclose(deviations[0], deviations[1], rtol=5e-3)
    measured = truncated.measure(frequency, 1e-3, noise_frequencies=[-0.01, -1e-4, 0.0, 1e-4, 0.01])
    assert measured.diagnostics[0]["retained_dimension"] == 325
    # A weak probe leaves the resonator close to a coherent state, so <a†a> is close to |<a>|^2.
    np.testing.assert_allclose(measured.photon_number("r0"), abs(measured.mode_amplitude("r0")) ** 2, rtol=1e-3)


@pytest.mark.validation
def test_an_explicit_cutoff_is_differentiable_under_jax():
    """Under tracing, the backend solves the compact model and says so; value and derivative match the sector solve."""
    jax = pytest.importorskip("jax")
    pytest.importorskip("dynamiqs")
    import jax.numpy as jnp

    frequency = float(probe_frequencies()[1])
    traced = VNA(feedline(space=ExcitationCutoff(max_total=2), backend="dynamiqs"))
    concrete = VNA(feedline(space=ExcitationCutoff(max_total=2)))

    taken = []

    def transmission(amplitude):
        result = traced.finite_power(frequency, amplitude)
        taken.append(result.diagnostics[0]["route"])
        return jnp.abs(result.ratio("feed"))

    def reference(amplitude):
        return abs(complex(np.ravel(concrete.finite_power(frequency, amplitude).ratio("feed"))[0]))

    value, derivative = jax.jit(jax.value_and_grad(transmission))(jnp.asarray(0.02))
    (route,) = set(taken)
    assert route.name == "stationary_resolvent" and route.arrays == "jax" and route.dimension == 10
    assert "traced" in route.reason and "N <= 2" in route.reason
    assert concrete.finite_power(frequency, 0.02).diagnostics[0]["route"].name == "sector_stationary"
    # Richardson extrapolation removes the step^2 error of the central differences.
    central = [(reference(0.02 + step) - reference(0.02 - step)) / (2 * step) for step in (1e-3, 1e-4)]
    np.testing.assert_allclose(value, reference(0.02), rtol=1e-9)
    np.testing.assert_allclose(derivative, (100 * central[1] - central[0]) / 99, rtol=1e-6)


def pumped(chip, port, frequency, amplitude):
    """Return a VNA of ``chip`` with one pump."""
    vna = VNA(chip)
    vna.pump(port, freq=frequency, amplitude=amplitude)
    return vna


def assert_same_output(sector, product, offsets, delays, *, port="feed", rtol):
    """Compare the spectra, fluxes and raw correlations of two VNAs, with ``rtol`` of each largest value."""
    def close(a, b):
        a, b = np.asarray(a), np.asarray(b)
        np.testing.assert_allclose(a, b, rtol=rtol, atol=rtol * np.max(np.abs(b)))

    spectra = [vna.output_spectrum(port, frequencies=offsets, **options)
               for vna, options in ((sector, {}), (product, {"options": {}}))]
    assert spectra[0].diagnostics["solver"] == "sector_stationary"
    assert spectra[1].diagnostics["solver"] == "stationary_resolvent"
    for name in ("signal_fluctuation_spectrum", "added_noise_spectrum"):
        close(getattr(spectra[0], name), getattr(spectra[1], name))
    close([spectra[0].signal_photon_flux, spectra[0].signal_coherent_flux],
          [spectra[1].signal_photon_flux, spectra[1].signal_coherent_flux])
    for method in ("g1", "g2"):
        values = [getattr(vna, method)(port, delays, **options)
                  for vna, options in ((sector, {}), (product, {"options": {}}))]
        close(values[0].unnormalized, values[1].unnormalized)
        close(values[0].values, values[1].values)
    return spectra


def test_output_statistics_on_every_sector_match_the_product_space(monkeypatch):
    """With N <= 6, the sector spectrum, fluxes, g1, g2 and state agree with the product space."""
    # Every chip takes the sector route, and the automatic cutoff keeps every label, so nothing is truncated.
    monkeypatch.setattr(routes, "AUTOMATIC_SECTOR_STATES", 0)
    monkeypatch.setattr(routes, "FIRST_CUTOFF", 6)
    chip = feedline()
    frequency, beta = float(probe_frequencies()[1]), 0.02 * np.exp(0.7j)
    # The offsets are not symmetric, so the spectrum needs the resolvent at -f as well.
    offsets, delays = np.array([-0.004, 0.0, 0.0015, 0.009]), np.array([0.0, 3.0, 25.0])
    sector, product = (pumped(chip, "feed", frequency, beta) for _ in range(2))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spectra = assert_same_output(sector, product, offsets, delays, rtol=1e-7)
    assert not cutoff_warnings(caught)
    states = [spectrum.steady_state for spectrum in spectra]
    assert states[0].excitation_space is not None and states[0].frame.frequencies == states[1].frame.frequencies
    for label in ("q", "r0", "r1"):
        np.testing.assert_allclose(np.asarray(chip.backend.to_array(states[0].reduced_state(label))),
                                   np.asarray(chip.backend.to_array(states[1].reduced_state(label))), atol=1e-9)


def test_the_cutoff_check_raises_k_for_output_statistics():
    """N <= 2 misses the shelved population; the comparison with K + 1 raises K until the statistics agree."""
    chip, offsets, delays = shelving_chip(), np.array([-0.004, 0.0, 0.004]), np.array([0.0, 5.0])
    automatic, product = (pumped(chip, "line", 6.0, 0.02) for _ in range(2))
    first = pumped(chip.with_space(ExcitationCutoff(max_total=2)), "line", 6.0, 0.02)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spectra = assert_same_output(automatic, product, offsets, delays, port="line", rtol=1e-5)
    warned = cutoff_warnings(caught)
    assert len(warned) == 3 and all(warning.filename == __file__ for warning in warned)
    diagnostics = spectra[0].diagnostics
    assert diagnostics["cutoff_choice"] == "automatic" and diagnostics["excitation_cutoff"] > 2
    assert diagnostics["cutoff_check"]["status"] == "passed" and diagnostics["cutoff_check"]["change"] <= 1e-6
    truncated = first.output_spectrum("line", frequencies=offsets)
    assert truncated.diagnostics["cutoff_choice"] == "explicit"
    # At N <= 2 the incoherent emission is almost absent.
    reference = np.asarray(spectra[1].signal_fluctuation_spectrum)
    difference = np.asarray(truncated.signal_fluctuation_spectrum) - reference
    assert np.max(np.abs(difference)) > 0.5 * np.max(np.abs(reference))


def test_three_readout_pairs_solve_their_output_statistics_at_n_le_2():
    """Of 729 product states, N <= 2 keeps 28; each stationary workflow solves there."""
    chip = readout_pairs(3).with_space(ExcitationCutoff(max_total=2))
    vna = pumped(chip, "feed", float(chip.freq("r0")), 1e-3)
    results = [chip.steadystate(), vna.output_spectrum("feed", frequencies=np.linspace(-0.01, 0.01, 5)),
               vna.g1("feed", np.array([0.0, 10.0])), vna.g2("feed", np.array([0.0, 10.0]))]
    for result in results:
        assert result.diagnostics["route"].name == "sector_stationary"
    assert [result.diagnostics["excitation_cutoff"] for result in results[1:]] == [2, 2, 2]
    assert [result.diagnostics["retained_dimension"] for result in results[1:]] == [28, 28, 28]


@pytest.mark.parametrize("backend", ["qutip", pytest.param("dynamiqs", marks=pytest.mark.optional_backend)])
def test_an_undriven_conserving_chip_is_in_its_vacuum(backend):
    """On 81 product states, steadystate() proves the vacuum unique without a Liouvillian solve."""
    if backend == "dynamiqs":
        pytest.importorskip("dynamiqs")
    chip = readout_pairs(2, backend)
    resonator = next(device for device in chip.devices if device.label == "r1")
    result = chip.steadystate(e_ops={resonator: 1 - resonator.number_operator(), "q0": "n"})
    assert result.diagnostics["solver"] == "sector_stationary" and result.nullity == 1
    assert result.excitation_space is not None
    assert complex(result.expect(resonator)) == 1 and complex(result.expect("q0")) == 0
    for label in ("q0", "r1"):
        np.testing.assert_array_equal(np.asarray(chip.backend.to_array(result.reduced_state(label))),
                                      np.diag([1.0, 0.0, 0.0]))
    if backend == "qutip":
        product = chip.steadystate(options={})
        np.testing.assert_allclose(np.asarray(product.reduced_state("r1").full()), np.diag([1.0, 0.0, 0.0]),
                                   atol=1e-9)


def test_a_dark_state_leaves_steadystate_to_the_liouvillian():
    """The lowering channels leave a dark N = 1 state, so steadystate() does not claim the vacuum and raises."""
    with pytest.raises(ValueError, match="nullity 2"):
        dark_pair(max_total=1).steadystate()


@pytest.mark.validation
@pytest.mark.parametrize("backend", ["qutip", pytest.param("dynamiqs", marks=pytest.mark.optional_backend)])
def test_a_pumped_detuned_qubit_pair_matches_the_product_space(monkeypatch, backend):
    """Two detuned, exchange-coupled transmons under a pump near one of them, on every sector and the product space."""
    if backend == "dynamiqs":
        pytest.importorskip("dynamiqs")
    monkeypatch.setattr(routes, "AUTOMATIC_SECTOR_STATES", 0)
    monkeypatch.setattr(routes, "FIRST_CUTOFF", 4)
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0", T1=5e3)
    q1 = DuffingTransmon(freq=5.08, anharmonicity=-0.22, levels=3, label="q1", T1=8e3, T2=6e3)
    network = PortNetwork(label="line")
    network.expose("p0", at=network.port("q0", target=q0, rate=0.01))
    network.expose("p1", at=network.port("q1", target=q1, rate=0.004))
    chip = Chip([q0, q1], [Capacitive(q0, q1, g=0.02)], port_network=network, approximation=RWA(), backend=backend)
    # The pump is 2 MHz above the dressed q0 transition and strong enough to saturate it.
    frequency = float(chip.freq("q0")) + 0.002
    sector, product = (pumped(chip, "p0", frequency, 0.06 * np.exp(0.3j)) for _ in range(2))

    def close(a, b, rtol=1e-7):
        a, b = np.asarray(a), np.asarray(b)
        np.testing.assert_allclose(a, b, rtol=rtol, atol=rtol * np.max(np.abs(b)))

    for port in ("p0", "p1"):
        spectra = [sector.output_spectrum(port, frequencies=[0.02]),
                   product.output_spectrum(port, frequencies=[0.02], options={})]
        assert spectra[0].diagnostics["solver"] == "sector_stationary"
        assert abs(spectra[1].signal_fluctuation_spectrum[0]) > 0
        close(spectra[0].signal_fluctuation_spectrum, spectra[1].signal_fluctuation_spectrum)
        close(spectra[0].signal_photon_flux, spectra[1].signal_photon_flux)
    for label in ("q0", "q1"):
        close(chip.backend.to_array(spectra[0].steady_state.reduced_state(label)),
              chip.backend.to_array(spectra[1].steady_state.reduced_state(label)))
    delays = np.array([0.0, 7.3])
    for method in ("g1", "g2"):
        for output in ("p0", "p1"):
            values = [getattr(sector, method)(output, delays, input="p0"),
                      getattr(product, method)(output, delays, input="p0", options={})]
            close(values[0].unnormalized, values[1].unnormalized)
    # The cross-correlation of q1 with the pumped port has a large imaginary part, so the check sees its phase.
    cross = np.asarray(product.g1("p1", delays, input="p0", options={}).unnormalized)
    assert np.all(np.abs(cross.imag) > 0.5 * np.abs(cross))
