"""Weak-probe VNA of excitation-conserving chips through their one-excitation block."""
import jax
import numpy as np
import pytest

from quchip import RWA, VNA, Capacitive, Chip, DuffingTransmon, Exact, PortNetwork, Resonator, eliminate


def _readout_chip(alpha=-0.25, levels=3, *, approximation=RWA(), qubit_occupation=None, backend="qutip", g=0.06):
    qubit = DuffingTransmon(freq=5.0, anharmonicity=alpha, levels=levels, label="q", T1=2000.0, T2=1500.0,
                            thermal_occupation=qubit_occupation)
    resonator = Resonator(freq=6.0, levels=levels, label="r")
    network = PortNetwork(label="m")
    network.port("r", target=resonator, rate=0.05)
    return Chip([qubit, resonator], [Capacitive(qubit, resonator, g=g)], port_network=network,
                approximation=approximation, backend=backend)


def _feedline_chip(readouts, *, operators=None):
    resonators = [Resonator(freq=6.5 + 0.1 * index, levels=2, label=f"r{index}", internal_quality_factor=2e5)
                  for index in range(readouts)]
    qubits = [DuffingTransmon(freq=5.3 - 0.1 * index, anharmonicity=-0.26, levels=3, label=f"q{index}", T1=3e4)
              for index in range(readouts)]
    network = PortNetwork(label="feedline")
    ports = [network.port(resonator.label, target=resonator, rate=2 * np.pi * 0.002, operator=operator)
             for resonator, operator in zip(resonators, operators or (None,) * readouts, strict=True)]
    network.cascade(*ports)
    network.expose("feed", input=ports[0], output=ports[-1])
    couplings = [Capacitive(qubit, resonator, g=0.04) for qubit, resonator in zip(qubits, resonators)]
    return Chip([*qubits, *resonators], couplings, port_network=network, frame=5.2, approximation=RWA())


def _sweep(chip, frequencies, **options):
    result = VNA(chip).sweep(frequencies, **options)
    return np.asarray(result.matrix)[:, 0, 0], result.diagnostics[0]["solver"]


FREQUENCIES = np.array([4.99, 5.0, 5.01, 5.99, 6.0, 6.01])
FEEDLINE_FREQUENCIES = np.array([6.45, 6.5, 6.52, 6.6, 6.62, 6.7])


def test_nonlinear_conserving_chip_uses_the_one_excitation_block_at_any_cutoff():
    """A dephased Duffing readout chip matches the stationary route and does not depend on cutoff."""
    small, route = _sweep(_readout_chip(levels=3), FREQUENCIES)
    large, _ = _sweep(_readout_chip(levels=5), FREQUENCIES)
    stationary, stationary_route = _sweep(_readout_chip(levels=3), FREQUENCIES, options={})

    assert route == "vacuum_response" and stationary_route != "vacuum_response"
    # Both routes solve the same linear equations; the stationary route adds its solver tolerance.
    np.testing.assert_allclose(small, stationary, atol=1e-9)
    np.testing.assert_allclose(large, small, atol=1e-12)


def test_reduced_multi_device_chip_uses_the_one_excitation_block():
    """A reduced chip with retained terms, a joint port and a reflection section keeps the weak-probe route."""
    first = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q1", T1=2000.0)
    second = DuffingTransmon(freq=5.3, anharmonicity=-0.25, levels=3, label="q2", T1=2000.0)
    resonator = Resonator(freq=6.0, levels=3, label="r1")
    network = PortNetwork(label="m")
    network.port("readout", target=resonator, rate=0.05)
    chip = Chip([first, second, resonator], [Capacitive(first, resonator, g=0.06), Capacitive(first, second, g=0.01)],
                port_network=network, approximation=RWA())
    reduced = eliminate(chip, "r1", method="exact").chip
    frequencies = np.array([4.99, 5.0, 5.01, 5.3])

    weak, route = _sweep(reduced, frequencies)
    stationary, _ = _sweep(reduced, frequencies, options={})

    assert route == "vacuum_response"
    np.testing.assert_allclose(weak, stationary, atol=1e-9)


def test_chip_reduced_past_both_readout_modes_keeps_the_one_excitation_block():
    """Two transformed readout ports, one carried through a second elimination, keep the weak-probe route."""
    qubits = [DuffingTransmon(freq=freq, anharmonicity=-0.26, levels=2, label=label, T1=30000.0)
              for freq, label in ((5.326, "q1"), (5.192, "q2"))]
    bus = Resonator(freq=6.298, levels=2, label="bus")
    readouts = [Resonator(freq=freq, levels=2, label=label) for freq, label in ((6.558, "r1"), (6.657, "r2"))]
    network = PortNetwork(label="feed")
    for readout in readouts:
        network.port(f"p_{readout.label}", target=readout, rate=0.01)
    couplings = [Capacitive(qubit, bus, g=0.03) for qubit in qubits]
    couplings += [Capacitive(qubit, readout, g=0.04) for qubit, readout in zip(qubits, readouts, strict=True)]
    chip = Chip([*qubits, bus, *readouts], couplings, port_network=network, frame=5.2, approximation=RWA())
    reduced = eliminate(eliminate(chip, "r1", method="exact").chip, "r2", method="exact").chip
    frequencies = np.array([5.19, 5.25, 5.33])

    weak = VNA(reduced).sweep(frequencies)
    stationary = VNA(reduced).sweep(frequencies, options={})

    assert weak.diagnostics[0]["solver"] == "vacuum_response"
    np.testing.assert_allclose(np.asarray(weak.matrix), np.asarray(stationary.matrix), atol=1e-9)


def test_shared_feedline_with_transmons_uses_the_one_excitation_block():
    """A cascade feedline whose ports lower the level index keeps the weak-probe route and its stationary result."""
    weak, route = _sweep(_feedline_chip(2), FEEDLINE_FREQUENCIES)
    stationary, _ = _sweep(_feedline_chip(2), FEEDLINE_FREQUENCIES, options={})

    assert route == "vacuum_response"
    np.testing.assert_allclose(weak, stationary, atol=1e-9)


@pytest.mark.unit
def test_feedline_port_that_does_not_lower_the_level_index_breaks_conservation():
    """A cascade pair with a port coupled through sigma_x changes the level index, so the compact route declines."""
    from quchip.chip.effective import conserves_excitation_number
    from quchip.engine.linear_response import try_build_weak_probe_problem

    lowering = _feedline_chip(2)
    quadrature = _feedline_chip(2, operators=(None, "X"))

    assert conserves_excitation_number(lowering, lowering.approximation)
    assert not conserves_excitation_number(quadrature, quadrature.approximation)
    assert try_build_weak_probe_problem(quadrature, FEEDLINE_FREQUENCIES, plane_labels=("feed",)) is None


def test_weak_probe_route_requires_a_stationary_vacuum():
    """Thermal excitation or counter-rotating terms leave the vacuum, so the compact route declines."""
    from quchip.engine.linear_response import try_build_weak_probe_problem

    _, route = _sweep(_readout_chip(qubit_occupation=0.05), FREQUENCIES[:2])
    assert route != "vacuum_response"
    assert try_build_weak_probe_problem(_readout_chip(approximation=Exact()), FREQUENCIES, plane_labels=("r",)) is None


@pytest.mark.validation
@pytest.mark.optional_backend
def test_weak_probe_reflection_gradient_matches_finite_difference():
    """|S|² differentiated through the one-excitation route matches a central difference in g."""
    pytest.importorskip("dynamiqs")

    def reflection(g):
        result = VNA(_readout_chip(backend="dynamiqs", g=g)).sweep(np.array([5.0]))
        return jax.numpy.abs(result.matrix[0, 0, 0]) ** 2

    g, step = 0.06, 1e-5
    finite_difference = (reflection(g + step) - reflection(g - step)) / (2 * step)
    np.testing.assert_allclose(jax.grad(reflection)(g), finite_difference, rtol=1e-6)
