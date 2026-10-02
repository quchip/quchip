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


def _sweep(chip, frequencies, **options):
    result = VNA(chip).sweep(frequencies, **options)
    return np.asarray(result.matrix)[:, 0, 0], result.diagnostics[0]["solver"]


FREQUENCIES = np.array([4.99, 5.0, 5.01, 5.99, 6.0, 6.01])


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
