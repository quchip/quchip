"""Conditional weak-probe response from a frozen dressed reference."""
import numpy as np
import pytest

from quchip import RWA, VNA, Capacitive, Chip, DuffingTransmon, Exact, PortNetwork, Resonator


def readout_chip(*, g=0.06, alpha=-0.25, levels=3, backend="qutip", T1=30000., T2=40000.,
                 approximation=RWA(), q_levels=None):
    q = DuffingTransmon(freq=5., anharmonicity=alpha, levels=levels if q_levels is None else q_levels,
                        label="q", T1=T1, T2=T2)
    r = Resonator(freq=6., levels=levels, label="r", internal_quality_factor=60000.)
    network = PortNetwork()
    network.port("readout", target=r, rate=0.01)
    return Chip([q, r], [Capacitive(q, r, g=g, label="qr")], port_network=network,
                approximation=approximation, backend=backend)


def full_response(chip, frequencies, when):
    """Independent full-space Lindblad resolvent with the frozen commutator source."""
    import qutip as qt

    engine = chip.resolve(frame="lab")
    H = sum((term.coefficient * np.asarray(term.operator.to_dense()) for term in engine.slh.H.static_terms),
            np.zeros((np.prod(engine.dims),) * 2, complex))
    channels = [np.asarray(c.coupling.to_dense()) for c in engine.slh.channels]
    generator = qt.liouvillian(qt.Qobj(H), [qt.Qobj(c) for c in channels]).full()
    psi = np.asarray(chip.backend.to_array(chip.state(when))).reshape(-1)
    rho = np.outer(psi, psi.conj())
    scattering = np.asarray(engine.slh.S)
    external_keys = {c.key for c in engine.slh.external_channels}
    external = [i for i, c in enumerate(engine.slh.channels) if c.key in external_keys]
    source = []
    for i in external:
        Kdag = sum(scattering[k, i] * c.conj().T for k, c in enumerate(channels))
        source.append(-(Kdag @ rho - rho @ Kdag).reshape(-1, order="F"))
    sources = np.stack(source, axis=1)
    result = []
    for f in frequencies:
        x = np.linalg.solve(-2j * np.pi * f * np.eye(len(generator)) - generator, sources)
        system = np.array([[np.trace(channels[i] @ x[:, j].reshape(H.shape, order="F"))
                            for j in range(len(external))] for i in external])
        result.append((scattering[np.ix_(external, external)] + system).conj())
    return np.asarray(result)


def test_ground_condition_reproduces_vacuum_response():
    """Explicit ground labels preserve the existing complex vacuum scattering."""
    chip = readout_chip()
    frequencies = np.array([4.99, 5., 5.99, 6., 6.01])
    expected = VNA(chip).sweep(frequencies)
    actual = VNA(chip).sweep(frequencies, when={chip["q"]: 0})
    np.testing.assert_allclose(actual.matrix, expected.matrix, rtol=0, atol=1e-13)


def test_excited_response_matches_full_lindblad_resolvent():
    """All loss and both probe-source terms agree with an independent full-space solve."""
    chip = readout_chip(T1=2000., T2=1500.)
    frequencies = np.array([4.75, 5., 5.99, 6., 6.01])
    result = VNA(chip).sweep(frequencies, when={"q": 1})
    np.testing.assert_allclose(result.matrix, full_response(chip, frequencies, {"q": 1}), rtol=1e-9, atol=1e-10)
    assert result.diagnostics[0]["reference_escape_rate"] > 1 / 3000
    assert result.diagnostics[0]["approximation"] == "frozen_reference"


def test_harmonic_excitation_does_not_change_response():
    """Stimulated emission and jump recycling cancel the extra harmonic absorption weight."""
    r = Resonator(freq=6., levels=3, label="r", internal_quality_factor=60000.)
    network = PortNetwork()
    network.port("p", target=r, rate=0.01)
    chip = Chip([r], port_network=network)
    f = np.array([5.99, 6., 6.01])
    np.testing.assert_allclose(VNA(chip).sweep(f, when={"r": 1}).matrix,
                               VNA(chip).sweep(f).matrix, rtol=1e-10, atol=1e-11)


def test_uncoupled_spectator_decay_and_dephasing_do_not_broaden_readout():
    """A spectator cannot alter the readout even when its decay is fast."""
    chip = readout_chip(g=0., T1=40., T2=20.)
    f = np.array([5.99, 6., 6.01])
    np.testing.assert_allclose(VNA(chip).sweep(f, when={"q": 1}).matrix,
                               VNA(chip).sweep(f, when={"q": 0}).matrix, rtol=1e-10, atol=1e-11)


@pytest.mark.unit
@pytest.mark.parametrize("when", [{"missing": 1}, {"q": -1}, {"q": True}, {"q": 1.2}, {"q": 2}, {"q": 1, "r": 1}])
def test_invalid_reference_is_rejected(when):
    """Invalid labels or unsupported excitation totals cannot silently select the vacuum."""
    with pytest.raises((TypeError, ValueError)):
        VNA(readout_chip()).sweep([6.], when=when)


def test_nonconserving_model_and_pumps_are_rejected():
    """Explicit conditioning never falls back to an unrelated stationary operating point."""
    chip = readout_chip(approximation=Exact())
    with pytest.raises(ValueError, match="conserv"):
        VNA(chip).sweep([6.], when={"q": 1})
    chip = readout_chip()
    vna = VNA(chip)
    vna.pump("readout", freq=6., amplitude=0.01)
    with pytest.raises(ValueError, match="pump"):
        vna.sweep([6.], when={"q": 1})


def test_exact_qnd_model_matches_analytic_conditional_lorentzian():
    """A stationary QND reference gives the exact shifted cavity pole and internal linewidth."""
    from quchip import CrossKerr

    q = DuffingTransmon(freq=5., anharmonicity=-.25, levels=3, label="q", T2=100.)
    r = Resonator(freq=6., levels=3, label="r", internal_quality_factor=60000.)
    network = PortNetwork()
    network.port("p", target=r, rate=.01)
    chi = -.003
    chip = Chip([q, r], [CrossKerr(q, r, chi=chi)], port_network=network, approximation=Exact())
    frequencies = np.array([5.99, 6. + chi, 6., 6.01])
    result = VNA(chip).sweep(frequencies, when={"q": 1})
    kappa = .01 + 2 * np.pi * 6 / 60000
    expected = 1 - .01 / (kappa / 2 + 2j * np.pi * (frequencies - 6. - chi))
    np.testing.assert_allclose(result.matrix[:, 0, 0], expected, rtol=1e-10, atol=1e-11)
    assert result.diagnostics[0]["reference_escape_rate"] == pytest.approx(0., abs=1e-15)
    assert result.diagnostics[0]["reference_stationarity_residual"] == pytest.approx(0., abs=1e-15)


@pytest.mark.validation
def test_decay_correction_matches_analytic_recycling_and_scales_with_rate():
    """An independent QND coherence solution bounds the frozen-source correction on the cavity timescale."""
    from quchip import CrossKerr

    kappa, chi = .01, -.003
    f = np.array([6. + chi, 6. + chi + .0005])
    errors = []
    for gamma in [1e-5, 2e-5]:
        q = DuffingTransmon(freq=5., anharmonicity=-.25, levels=3, label="q", T1=1 / gamma)
        r = Resonator(freq=6., levels=3, label="r")
        network = PortNetwork()
        network.port("p", target=r, rate=kappa)
        chip = Chip([q, r], [CrossKerr(q, r, chi=chi)], port_network=network)
        result = VNA(chip).sweep(f, when={"q": 1})
        de = kappa / 2 + 2j * np.pi * (f - 6. - chi)
        dg = kappa / 2 + 2j * np.pi * (f - 6.)
        expected = 1 - kappa / (de + gamma) * (1 + gamma / dg)
        np.testing.assert_allclose(result.matrix[:, 0, 0], expected, rtol=1e-10, atol=1e-11)
        assert result.diagnostics[0]["reference_escape_rate"] == pytest.approx(gamma, rel=1e-12)
        error = np.max(np.abs(expected - (1 - kappa / de)))
        assert error < 4 * gamma / kappa
        errors.append(error)
    assert errors[1] / errors[0] == pytest.approx(2., rel=.005)


@pytest.mark.validation
def test_ef_transition_and_cutoffs_control_the_conditional_pull():
    """The three-level dispersive pull has the EF correction and converges with local cutoffs."""
    from scipy.optimize import minimize_scalar

    chip = readout_chip(g=.02, T1=None, T2=None)
    ground_frequency = float(chip.freq("r"))
    excited_frequency = float(chip.freq("r", when={"q": 1}))
    expected_pull = 2 * .02**2 * (-.25) / ((5. - 6.) * (5. - 6. - .25))
    assert excited_frequency - ground_frequency == pytest.approx(expected_pull, rel=.003)
    frequencies = excited_frequency + np.array([-.001, 0., .001])
    response = VNA(chip).sweep(frequencies, when={"q": 1}).matrix
    larger = readout_chip(g=.02, levels=4, T1=None, T2=None)
    np.testing.assert_allclose(response, VNA(larger).sweep(frequencies, when={"q": 1}).matrix, atol=1e-11)
    two_level = readout_chip(g=.02, q_levels=2, T1=None, T2=None)
    with pytest.warns(UserWarning, match="'q' keeps 2 levels"):
        two_level_pull = float(two_level.freq("r", when={"q": 1})) - ground_frequency
    assert two_level_pull == pytest.approx(2 * .02**2 / (5. - 6.), rel=.003)
    optimum = minimize_scalar(lambda f: abs(VNA(chip).sweep(f, when={"q": 1}).matrix[0, 0])**2,
                              bounds=(excited_frequency - .0005, excited_frequency + .0005),
                              method="bounded", options={"xatol": 1e-12})
    assert optimum.success
    assert optimum.x == pytest.approx(excited_frequency, abs=2e-6)


@pytest.mark.validation
def test_close_modes_include_internal_loss_and_shared_feedline_mixing():
    """A close readout/bus pair agrees with the full Lindblad matrix including shared-channel damping."""
    q = DuffingTransmon(freq=5., anharmonicity=-.25, levels=3, label="q", T1=30000.)
    r = Resonator(freq=6., levels=3, label="r", internal_quality_factor=20000.)
    b = Resonator(freq=6.0008, levels=3, label="b", internal_quality_factor=30000.)
    network = PortNetwork()
    p = network.port("r", target=r, rate=.01)
    z = network.port("b", target=b, rate=.008, phase=.4)
    network.cascade(p, z)
    network.expose("feed", input=p, output=z)
    chip = Chip([q, r, b], [Capacitive(q, r, g=.04), Capacitive(r, b, g=.0006)],
                port_network=network)
    f = np.array([5.998, 6., 6.003])
    expected = full_response(chip, f, {"q": 1})
    np.testing.assert_allclose(VNA(chip).sweep(f, when={"q": 1}).matrix, expected, rtol=1e-9, atol=1e-10)


@pytest.mark.optional_backend
@pytest.mark.validation
@pytest.mark.parametrize("iterative", [False, True])
def test_conditional_response_jit_gradient_matches_finite_difference(monkeypatch, iterative):
    """A traced dressed reference and loss response give the correct complex reflection derivative."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    if iterative:
        import quchip.backend._response as algebra

        monkeypatch.setattr(algebra, "_COHERENCE_DENSE_LIMIT", 0)

    frequencies = np.array([5.999, 6.002])

    def signal(parameters, backend="dynamiqs"):
        value = VNA(readout_chip(g=parameters[0], backend=backend)).sweep(
            frequencies + parameters[1], when={"q": 1}).matrix[:, 0, 0]
        return jnp.stack((value.real, value.imag))

    parameters = np.array([.06, 0.])
    derivative = jax.jit(jax.jacrev(signal))(parameters)
    # Central-difference truncation and roundoff at these steps, plus the 1e-11
    # solve tolerance, stay below this bound.
    for i, step in enumerate([1e-5, 1e-7]):
        delta = np.eye(2)[i] * step
        difference = (signal(parameters + delta, "qutip") - signal(parameters - delta, "qutip")) / (2 * step)
        np.testing.assert_allclose(derivative[..., i], difference, rtol=2e-6, atol=1e-7)
    compiled = jax.jit(signal)
    # Both backends solve the same blocks to a 1e-11 relative tolerance.
    for point in (parameters, np.array([.055, .0003])):
        np.testing.assert_allclose(compiled(point), signal(point, "qutip"), atol=1e-10)


@pytest.mark.optional_backend
@pytest.mark.parametrize("cascade", [False, True])
def test_compiled_response_rejects_reordered_levels(cascade):
    """A compiled diagonal model keeps its authored level order and rejects values that reorder it."""
    pytest.importorskip("dynamiqs")
    import jax
    from quchip.declarative import DeviceModel, Scalar, parameter

    class Diagonal(DeviceModel):
        freq: Scalar = parameter()

        def local_hamiltonian(self, op, p):
            return p.freq * op.n - 3. * (op.n @ op.n)

    q = Diagonal(freq=10., levels=3, label="q")
    r = Resonator(freq=6., levels=3, label="r")
    network = PortNetwork()
    first = network.port("q", target=q, rate=.01)
    last = network.port("r", target=r, rate=.008)
    if cascade:
        network.cascade(first, last)
        network.expose("feed", input=first, output=last)
    chip = Chip([q, r], port_network=network, backend="dynamiqs")

    def signal(frequency):
        result = VNA(chip.with_params({"q.freq": frequency})).sweep(6.002, when={"q": 0})
        assert result.diagnostics[0]["solver"] == "vacuum_response"
        return result.matrix

    compiled = jax.jit(signal)
    # Compiled and eager calls solve the same sector blocks, so only round-off differs.
    np.testing.assert_allclose(compiled(10.), signal(10.), atol=1e-12)
    for frequency in (5., -1.):
        with pytest.raises(ValueError, match="conserv"):
            signal(frequency)
        # The cached executable checks the traced level order during execution.
        with pytest.raises(jax.errors.JaxRuntimeError, match="out of their authored order"):
            compiled(frequency).block_until_ready()


def test_matrix_free_response_matches_direct_solve(monkeypatch):
    """Iterative sector solves preserve complex scattering and certify their residual."""
    import quchip.backend._response as algebra

    chip = readout_chip()
    network = PortNetwork()
    network.port("a", target=chip["r"], rate=.01)
    network.port("b", target=chip["r"], rate=.003, phase=.3)
    chip = Chip(chip.devices, chip.couplings, port_network=network)
    f = np.array([5.99, 6.002])
    direct = VNA(chip).sweep(f, when={"q": 1}).matrix
    np.testing.assert_allclose(direct, full_response(chip, f, {"q": 1}), rtol=1e-9, atol=1e-10)
    monkeypatch.setattr(algebra, "_COHERENCE_DENSE_LIMIT", 0)
    iterative = VNA(chip).sweep(f, when={"q": 1})
    np.testing.assert_allclose(iterative.matrix, direct, rtol=1e-8, atol=1e-9)
    assert max(d["residual"] for d in iterative.diagnostics) < 1e-8
    assert iterative.diagnostics[0]["block_condition_number"] is None


@pytest.mark.optional_backend
@pytest.mark.parametrize("iterative", [False, True])
def test_conditional_response_rejects_nonfinite_source(monkeypatch, iterative):
    """Compiled direct and iterative solves cannot return a nonfinite response as converged."""
    pytest.importorskip("dynamiqs")
    import jax
    from dataclasses import replace

    import quchip.backend._response as algebra
    from quchip.engine.conditional_response import build_conditional_response

    if iterative:
        monkeypatch.setattr(algebra, "_COHERENCE_DENSE_LIMIT", 0)
    chip = readout_chip(backend="dynamiqs")
    problem, _ = build_conditional_response(chip, np.array([6.002]), ("readout",), (1, 0))
    problem = replace(problem, sources=tuple(source * np.nan for source in problem.sources))
    with pytest.raises((ValueError, jax.errors.JaxRuntimeError), match="failed to converge"):
        chip.backend.coherence_response(problem).responses.block_until_ready()


def test_parameter_axes_ports_and_reference_delays_are_preserved():
    """Conditional parameter sweeps retain all ports and each selected reference-plane phase."""
    from quchip import Sweep

    chip = readout_chip()
    network = PortNetwork()
    port_a = network.port("a", target=chip["r"], rate=.01)
    network.port("b", target=chip["r"], rate=.003, phase=.3)
    chip = Chip(chip.devices, chip.couplings, port_network=network)
    f = np.array([5.999, 6.002])
    values = [-.25, -.2]
    sweep = VNA(chip).sweep(f, Sweep(values, name="q.anharmonicity"), when={chip["q"]: 1})
    expected = np.stack([full_response(chip.with_params({"q.anharmonicity": a}), f, {"q": 1}) for a in values])
    np.testing.assert_allclose(sweep.matrix, expected, atol=1e-10)
    subset = VNA(chip, ports=["b"]).sweep(f, when={"q": 1})
    np.testing.assert_allclose(subset.matrix[:, 0, 0], expected[0, :, 1, 1], atol=1e-10)
    cable = network.delay("cable", duration=.123)
    network.link(port_a, cable)
    network.expose("a_external", at=cable.port(2))
    delayed_chip = Chip(chip.devices, chip.couplings, port_network=network)
    delayed = VNA(delayed_chip, ports=["a_external", "b"]).sweep(f, when={"q": 1})
    legs = np.stack([np.exp(-2j * np.pi * f * .123), np.ones_like(f)], axis=1)
    np.testing.assert_allclose(delayed.matrix, expected[0] * legs[:, :, None] * legs[:, None, :], atol=1e-10)
    # A later model mutation cannot change diagnostics captured by the earlier solve.
    captured = sweep.diagnostics[0]["reference_escape_rate"]
    chip["q"].T1 = 60000.
    assert sweep.diagnostics[0]["reference_escape_rate"] == captured


def test_ambiguous_reference_and_nonvacuum_input_are_rejected():
    """Ambiguous dressed labels and thermal preparations need a different response model."""
    resonant = readout_chip().with_params({"q.freq": 6.})
    with pytest.raises(ValueError, match="unambiguous"):
        VNA(resonant).sweep(6.001, when={"q": 1})
    thermal = readout_chip()
    thermal["q"].thermal_occupation = .05
    with pytest.raises(ValueError, match="vacuum"):
        VNA(thermal).sweep(6., when={"q": 1})
    with pytest.raises(ValueError, match="options"):
        VNA(readout_chip()).sweep(6., when={"q": 1}, options={})


@pytest.mark.validation
def test_large_conditional_harmonic_model_matches_independent_mode_equations():
    """A 24-device excited reference has the same response as independent 24-mode linear equations."""
    count, g, rate = 24, .002, .01
    devices = [Resonator(freq=5. + .03 * i, levels=3, label=f"r{i}", internal_quality_factor=200000.)
               for i in range(count)]
    network = PortNetwork()
    for i in [0, count - 1]:
        network.port(f"p{i}", target=devices[i], rate=rate)
    chip = Chip(devices, [Capacitive(a, b, g=g) for a, b in zip(devices[:-1], devices[1:])], port_network=network)
    frequencies = np.array([5., 5. + .03 * (count - 1)])
    H = np.diag([d.freq for d in devices]) + np.diag(np.full(count - 1, g), 1) + np.diag(np.full(count - 1, g), -1)
    C = np.sqrt(rate) * np.eye(count)[[0, count - 1]]
    damping = np.diag([2 * np.pi * d.freq / 200000. for d in devices]) + C.T @ C
    expected = [np.eye(2) - C @ np.linalg.solve(damping / 2 + 2j * np.pi * (f * np.eye(count) - H), C.T)
                for f in frequencies]
    result = VNA(chip).sweep(frequencies, when={"r0": 1})
    np.testing.assert_allclose(result.matrix, expected, atol=1e-8, rtol=1e-8)
    assert result.diagnostics[0]["sector_dimensions"] == (1, 24, 300)
    assert result.diagnostics[0]["residual"] < 1e-8


def test_two_level_reference_keeps_stimulated_emission_without_an_upper_sector():
    """An excited two-level system emits into the probe even though its two-excitation sector is empty."""
    q = DuffingTransmon(freq=5., anharmonicity=-.25, levels=2, label="q")
    network = PortNetwork()
    network.port("p", target=q, rate=.01)
    chip = Chip([q], port_network=network)
    f = np.array([4.999, 5., 5.001])
    result = VNA(chip).sweep(f, when={"q": 1})
    expected = 1 + .01 / (.005 + 2j * np.pi * (f - 5.))
    np.testing.assert_allclose(result.matrix[:, 0, 0], expected, atol=1e-11)
    assert np.isfinite(result.diagnostics[0]["block_condition_number"])


def test_reduced_chip_uses_transformed_channels_in_conditional_response():
    """An eliminated bus carries its retained Hamiltonian and loss into the conditional sector solve."""
    from quchip import eliminate

    source = readout_chip()
    bus = Resonator(freq=7.5, levels=3, label="bus", internal_quality_factor=10000.)
    chip = Chip([*source.devices, bus], [*source.couplings, Capacitive(source["q"], bus, g=.04)],
                port_network=source.port_network)
    reduced = eliminate(chip, "bus", method="exact").chip
    f = np.array([5.99, 6.002])
    expected = full_response(reduced, f, {"q": 1})
    np.testing.assert_allclose(VNA(reduced).sweep(f, when={"q": 1}).matrix, expected, rtol=1e-9, atol=1e-10)
