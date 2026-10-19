"""Weak probes of static lab-frame models, including counter-rotating terms."""

from __future__ import annotations

import numpy as np
import pytest
import qutip

from quchip import Capacitive, Chip, DuffingTransmon, Exact, PortNetwork, Resonator, RWA, Sweep, VNA


def _readout(*, backend="qutip", exact=True, levels=3, qfreq=5.46, rfreq=6.55, g=0.04,
             two_ports=False, delay=False, frame="lab"):
    q = DuffingTransmon(freq=qfreq, anharmonicity=-0.26, levels=levels, T1=50e3, label="q")
    r = Resonator(freq=rfreq, levels=levels, internal_quality_factor=2e5, label="r")
    network = PortNetwork(label="line")
    for index, rate in enumerate((2 * np.pi * 0.001, 0.003) if two_ports else (2 * np.pi * 0.001,)):
        port = network.port(f"p{index}", target=r, rate=rate, phase=(0.2, -0.35)[index])
        if delay:
            cable = network.delay(f"cable{index}", duration=(0.123, 0.057)[index])
            network.link(port, cable)
            network.expose(f"vna{index}", at=cable.port(2))
    return Chip([q, r], [Capacitive(q, r, g=g, label="qr")], port_network=network,
                backend=backend, approximation=Exact() if exact else RWA(), frame=frame)


def _reference(frequencies, *, exact=True, levels=3, qfreq=5.46, rfreq=6.55, g=0.04,
               two_ports=False, delay=False):
    """Assemble the declared finite master equation without quchip engine operators."""
    a = qutip.tensor(qutip.destroy(levels), qutip.qeye(levels))
    b = qutip.tensor(qutip.qeye(levels), qutip.destroy(levels))
    na, nb = a.dag() * a, b.dag() * b
    interaction = -(a - a.dag()) * (b - b.dag()) if exact else a.dag() * b + a * b.dag()
    h = 2 * np.pi * (qfreq * na - 0.13 * a.dag() ** 2 * a ** 2 + rfreq * nb + g * interaction)
    rates = (2 * np.pi * 0.001, 0.003) if two_ports else (2 * np.pi * 0.001,)
    ports = [np.sqrt(rate) * np.exp(1j * phase) * b for rate, phase in zip(rates, (0.2, -0.35))]
    collapse = [a / np.sqrt(50e3), np.sqrt(2 * np.pi * rfreq / 2e5) * b, *ports]
    rho = qutip.steadystate(h, collapse)
    liouvillian = qutip.liouvillian(h, collapse).full()
    normal = [port.dag() * rho - rho * port.dag() for port in ports]
    conjugate = [rho * port - port * rho for port in ports]
    sources = np.column_stack([source.full().reshape(-1, order="F") for source in normal + conjugate])
    sources[-1] = 0
    matrices, conjugates = [], []
    dimension = levels**2
    for frequency in np.atleast_1d(frequencies):
        shifted = liouvillian + 2j * np.pi * frequency * np.eye(dimension**2)
        shifted[-1] = np.eye(dimension).reshape(-1)
        solved = np.linalg.solve(shifted, sources)
        fields = np.array([[np.trace(port.full() @ solution.reshape((dimension, dimension), order="F"))
                            for solution in solved.T] for port in ports])
        transfer = np.exp(2j * np.pi * frequency * np.array((0.123, 0.057)[:len(ports)])) if delay else 1.0
        transfer = np.broadcast_to(transfer, (len(ports),))
        matrices.append(np.conj(transfer[:, None] * (np.eye(len(ports)) + fields[:, :len(ports)])
                                * transfer[None, :]))
        conjugates.append(np.conj(fields[:, len(ports):]) if frequency == 0 else np.zeros((len(ports), len(ports))))
    return np.array(matrices), np.array(conjugates)


@pytest.mark.parametrize("backend", ["qutip", pytest.param("dynamiqs", marks=pytest.mark.optional_backend)])
@pytest.mark.parametrize("exact", [False, True])
def test_weak_readout_probe_matches_independent_lab_liouvillian(backend, exact):
    """RWA and Exact reflection retain their declared Hamiltonian and engineering phase."""
    if backend == "dynamiqs":
        pytest.importorskip("dynamiqs")
    frequencies = np.array([6.5508, 6.5513, 6.5518])
    chip = _readout(backend=backend, exact=exact, frame="rotating")
    result = VNA(chip).sweep(frequencies)
    expected, conjugate = _reference(frequencies, exact=exact)
    # Same declared master equation. The bound covers direct-solve round-off of the 81x81 lab Liouvillian.
    np.testing.assert_allclose(result.matrix, expected, atol=2e-9)
    # T is zero away from DC in both calculations.
    np.testing.assert_allclose(result.conjugate_matrix, conjugate, atol=1e-12)
    general = VNA(chip).sweep(frequencies, options={})
    np.testing.assert_allclose(general.matrix, expected, atol=2e-9)
    if exact:
        assert {item["solver"] for item in result.diagnostics} == {"stationary_resolvent"}
        rwa, _ = _reference(frequencies, exact=False)
        # Counter-rotating terms change the response far beyond the solver bound.
        assert np.max(np.abs(result.matrix - rwa)) > 0.05


@pytest.mark.parametrize("backend", ["qutip", pytest.param("dynamiqs", marks=pytest.mark.optional_backend)])
def test_lab_response_keeps_port_selection_and_frequency_dependent_reference_planes(backend):
    """Every selected matrix element includes its own physical-frequency cable and port phases."""
    if backend == "dynamiqs":
        pytest.importorskip("dynamiqs")
    chip = _readout(backend=backend, two_ports=True, delay=True)
    frequencies = np.array([6.5509, 6.5515])
    expected, _ = _reference(frequencies, two_ports=True, delay=True)
    result = VNA(chip, ports=["vna1", "vna0"]).sweep(frequencies, options={})
    # Same declared master equation. The bound covers direct-solve round-off of the lab Liouvillian.
    np.testing.assert_allclose(result.matrix, expected[:, ::-1, ::-1], atol=2e-9)
    selected = VNA(chip, ports=["vna1"]).sweep(frequencies)
    np.testing.assert_allclose(selected.s11, expected[:, 1, 1], atol=2e-9)


def test_lab_response_reuses_one_state_per_chip_point_and_captures_diagnostics(monkeypatch):
    """A parameter sweep solves once per chip point and saved diagnostics survive source edits."""
    chip = _readout()
    frequencies = np.array([6.5509, 6.5515])
    calls = []
    solve = chip.backend.steadystate

    def recorded(*args, **kwargs):
        calls.append(None)
        return solve(*args, **kwargs)

    monkeypatch.setattr(chip.backend, "steadystate", recorded)
    result = VNA(chip).sweep(frequencies, Sweep([0.03, 0.04], name="qr.g"))
    assert len(calls) == 2
    expected = np.stack([_reference(frequencies, g=g)[0] for g in (0.03, 0.04)])
    # Same declared master equation. The bound covers direct-solve round-off of the lab Liouvillian.
    np.testing.assert_allclose(result.matrix, expected, atol=2e-9)
    chip.devices[0].freq = 5.0
    np.testing.assert_allclose(result.matrix, expected, atol=2e-9)
    # A physical stationary state has no negative eigenvalue beyond round-off.
    assert max(float(item["positivity_error"]) for item in result.diagnostics) < 1e-10
    assert len(calls) == 2


@pytest.mark.parametrize("backend", ["qutip", pytest.param("dynamiqs", marks=pytest.mark.optional_backend)])
def test_lab_conjugate_response_is_same_carrier_only_at_dc(backend):
    """Opposite-frequency response is excluded from T except when both inputs are DC."""
    if backend == "dynamiqs":
        pytest.importorskip("dynamiqs")
    frequencies = np.array([-1.6, 0.0, 1.6])
    chip = _readout(backend=backend, qfreq=1.1, rfreq=1.6, g=0.1)
    result = VNA(chip).sweep(frequencies)
    expected, conjugate = _reference(frequencies, qfreq=1.1, rfreq=1.6, g=0.1)
    # Same declared master equation. The bound covers direct-solve round-off of the lab Liouvillian.
    np.testing.assert_allclose(result.matrix, expected, atol=2e-9)
    np.testing.assert_allclose(result.conjugate_matrix, conjugate, atol=2e-9)
    # The DC conjugate response is far above round-off, so the quadrature checks below can see it.
    assert abs(conjugate[1, 0, 0]) > 1e-7
    step = 1e-4
    # Central differences of the finite-power field. The bound covers their O(step²) error and round-off.
    means = np.asarray(VNA(chip).finite_power(0.0, [step, -step, 1j * step, -1j * step]).mean("p0"))
    np.testing.assert_allclose((means[0] - means[1]) / (2 * step),
                               expected[1, 0, 0] + conjugate[1, 0, 0], atol=2e-8)
    np.testing.assert_allclose((means[2] - means[3]) / (2j * step),
                               expected[1, 0, 0] - conjugate[1, 0, 0], atol=2e-8)


def test_exact_finite_power_and_fixed_pumps_still_require_a_stationary_tone_frame():
    """A weak lab-frame probe does not enable a periodic finite-power operating point."""
    vna = VNA(_readout())
    with pytest.raises(ValueError, match="dynamic Hamiltonian"):
        vna.finite_power([6.55], 1e-4)
    vna.pump("p0", freq=6.55, amplitude=1e-4)
    with pytest.raises(ValueError, match="dynamic Hamiltonian"):
        vna.sweep([6.55])


def test_lab_dynamic_model_keeps_the_stationary_error():
    """An explicit frequency modulation still requires time evolution."""
    from quchip.declarative import CosineCoefficient, TimeDependentTerm

    class ModulatedResonator(Resonator):
        def time_terms(self, op, p):
            return (TimeDependentTerm(operator=op.n, coefficient=CosineCoefficient(0.02, 0.1)),)

    resonator = ModulatedResonator(freq=6.0, levels=2, label="r")
    network = PortNetwork()
    network.port("p", target=resonator, rate=0.02)
    with pytest.raises(ValueError, match="dynamic Hamiltonian"):
        VNA(Chip([resonator], port_network=network)).sweep([6.0])


@pytest.mark.validation
@pytest.mark.optional_backend
def test_compiled_lab_response_derivatives_match_independent_liouvillian_differences():
    """JIT derivatives include the stationary state's frequency dependence and probe resolvent shift."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    chip = _readout(backend="dynamiqs", levels=2)

    def response(values):
        bound = chip.with_params({"q.freq": values[0], "qr.g": values[1]})
        reflection = VNA(bound).sweep(values[2]).s11
        return jnp.stack((reflection.real, reflection.imag))

    def independent(values):
        matrix, _ = _reference([values[2]], levels=2, qfreq=values[0], g=values[1])
        return np.array((matrix[0, 0, 0].real, matrix[0, 0, 0].imag))

    compiled = jax.jit(jax.jacfwd(response))
    for values in (np.array([5.46, 0.04, 6.551]), np.array([5.48, 0.035, 6.552])):
        step = np.array([1e-5, 1e-6, 1e-7])
        finite_difference = np.column_stack([
            (independent(values + np.eye(3)[i] * step[i]) - independent(values - np.eye(3)[i] * step[i]))
            / (2 * step[i]) for i in range(3)
        ])
        # Same declared master equation. The bound covers direct-solve round-off of the lab Liouvillian.
        np.testing.assert_allclose(response(values), independent(values), atol=2e-9)
        # Central differences of the reference. The bound covers their O(step²) error and round-off.
        np.testing.assert_allclose(compiled(jnp.asarray(values)), finite_difference, rtol=2e-5, atol=2e-7)
