"""The poles of the weak-probe response: frequency, linewidth, composition and residues (#192)."""

import numpy as np
import pytest

from quchip import RWA, VNA, Capacitive, Chip, DuffingTransmon, Exact, PortNetwork, Resonator, eliminate

TWO_PI = 2 * np.pi


def readout(g=0.1, backend="qutip", approximation=RWA()):
    """The chip of #192: a transmon dispersively coupled to a readout with a port and an internal Q."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=3, internal_quality_factor=1e5, label="r")
    network = PortNetwork(label="line")
    port = network.port("r_port", target=r, rate=TWO_PI * 0.001)
    network.expose("p", input=port.input, output=port.output)
    return Chip([q, r], [Capacitive(q, r, g=g, label="qr")], port_network=network, frame=6.0,
                approximation=approximation, backend=backend)


def readout_rate():
    return TWO_PI * 0.001 + TWO_PI * 7.0 / 1e5


def resonator_pair(first_rate, second_rate, g=None, second_freq=7.0, backend="qutip"):
    """Resonators at 7 GHz and second_freq with their own ports and, optionally, an exchange coupling g."""
    modes = [Resonator(freq=freq, levels=2, label=f"r{index}") for index, freq in ((1, 7.0), (2, second_freq))]
    network = PortNetwork(label="line")
    for mode, rate in zip(modes, (first_rate, second_rate)):
        port = network.port(f"{mode.label}_port", target=mode, rate=rate)
        network.expose(f"p{mode.label[-1]}", input=port.input, output=port.output)
    return Chip(modes, [Capacitive(*modes, g=g)] if g else [], port_network=network, backend=backend)


@pytest.mark.unit
def test_resonator_linewidth_adds_port_and_internal_loss():
    """A resonator with port rate κₑ and internal Q has one pole at f with κ = κₑ + 2πf/Q and residue iκₑ/2π."""
    resonator = Resonator(freq=7.0, levels=3, internal_quality_factor=2e4, label="r")
    network = PortNetwork(label="line")
    port = network.port("r_port", target=resonator, rate=0.003)
    network.expose("p", input=port.input, output=port.output)
    result = VNA(Chip([resonator], port_network=network)).resonances()

    kappa = 0.003 + TWO_PI * 7.0 / 2e4
    assert result.route.name == "linear_response" and len(result) == 1 and result.dominant == ("r",)
    np.testing.assert_allclose(result.frequencies, [7.0], rtol=1e-14)
    np.testing.assert_allclose(result.linewidths, [kappa], rtol=1e-12)
    np.testing.assert_allclose(result.poles, [7.0 + 1j * kappa / (2 * TWO_PI)], rtol=1e-14)
    np.testing.assert_allclose(result.residues, [[[1j * 0.003 / TWO_PI]]], rtol=1e-12)
    np.testing.assert_allclose(result.direct, [[1.0]], atol=1e-15)
    assert result["r"].composition == {"r": pytest.approx(1.0)} and not result["r"].dark


@pytest.mark.validation
def test_dispersive_poles_match_the_exact_two_mode_drift():
    """The one-excitation drift of #192 is the 2×2 matrix −i[[ω_q, g_ω], [g_ω, ω_r]] − diag(0, κ_r/2). Its closed-form
    eigenvalues give both poles, and the qubit linewidth approaches g_ω²κ_r / (Δ_ω² + κ_r²/4) as g/Δ → 0."""
    kappa = readout_rate()
    errors = []
    for g in (0.1, 0.05, 0.025):
        result = VNA(readout(g)).resonances()
        omega_q, omega_r, coupling = TWO_PI * 5.0, TWO_PI * 7.0, TWO_PI * g
        mean = (-1j * (omega_q + omega_r) - kappa / 2) / 2
        half = (-1j * (omega_q - omega_r) + kappa / 2) / 2
        root = np.sqrt(half**2 - coupling**2)
        poles = sorted([mean + root, mean - root], key=lambda value: -value.imag)
        assert result.route.name == "vacuum_response" and result.dominant == ("q", "r")
        np.testing.assert_allclose(result.frequencies, [-value.imag / TWO_PI for value in poles], rtol=1e-14)
        # The eigensolve is backward stable, so each eigenvalue errs by about ε |A_0| ≈ 2e-15 /ns.
        np.testing.assert_allclose(result.linewidths, [-2 * value.real for value in poles], rtol=0, atol=1e-14)
        # Σκ = −2 Re tr A = tr(C†C): the port and internal rates of the readout.
        np.testing.assert_allclose(np.sum(result.linewidths), kappa, rtol=1e-12)
        delta = omega_q - omega_r
        leading = coupling**2 * kappa / (delta**2 + kappa**2 / 4)
        errors.append(abs(result["q"].linewidth / leading - 1))
        assert errors[-1] < 4 * (g / 2.0) ** 2
    # The correction is of order (g/Δ)², so halving g divides it by about four.
    np.testing.assert_allclose(np.asarray(errors[:-1]) / np.asarray(errors[1:]), 4.0, rtol=0.01)


def test_poles_rebuild_the_swept_two_port_response_with_a_cable():
    """S rebuilt from the poles, the residues and the reference transfers equals VNA.sweep(), entry by entry."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=6.0, levels=2, label="r", internal_quality_factor=1e5)
    network = PortNetwork(label="line")
    left = network.port("left", target=r, rate=0.004)
    right = network.port("right", target=r, rate=0.006)
    cable = network.delay("cable", duration=0.125)
    network.link(right, cable)
    network.expose("a", input=left.input, output=left.output)
    network.expose("b", at=cable.port(2))
    chip = Chip([q, r], [Capacitive(q, r, g=0.05)], port_network=network, approximation=RWA())
    frequencies = np.linspace(5.99, 6.01, 9)

    result = VNA(chip).resonances()
    swept = VNA(chip).sweep(frequencies)
    assert result.ports == ("a", "b") and np.shape(result.residues) == (2, 2, 2)
    np.testing.assert_allclose(result.scattering(frequencies), swept.matrix, rtol=1e-10, atol=1e-12)


def test_retained_pole_enters_the_composition_and_the_rebuilt_response():
    """A readout that eliminate() removes stays a pole of the reduced chip, with its own mode label."""
    qubit = DuffingTransmon(freq=5.3, anharmonicity=-0.26, levels=2, label="q")
    kept = Resonator(freq=6.0, levels=2, internal_quality_factor=1e5, label="r0")
    removed = Resonator(freq=6.02, levels=2, label="r1")
    network = PortNetwork(label="line")
    right = [network.port("r0_right", target=kept, rate=0.006), network.port("r1_right", target=removed, rate=0.006)]
    left = [network.port("r1_left", target=removed, rate=0.006), network.port("r0_left", target=kept, rate=0.006)]
    network.cascade(*left)
    network.cascade(*right)
    network.expose("right", input=right[0].input, output=right[-1].output)
    network.expose("left", input=left[0].input, output=left[-1].output)
    chip = Chip([qubit, kept, removed], [Capacitive(qubit, removed, g=0.04)], port_network=network, frame=6.0,
                approximation=RWA())
    reduced = eliminate(chip, "r1").chip
    frequencies = np.linspace(5.98, 6.04, 13)

    result = VNA(reduced).resonances()
    assert result.mode_labels == ("q", "r0", "r1") and result.dominant == ("q", "r0", "r1")
    np.testing.assert_allclose(result.scattering(frequencies), VNA(reduced).sweep(frequencies).matrix,
                               rtol=1e-10, atol=1e-12)


@pytest.mark.unit
def test_a_pole_that_no_port_reaches_is_dark():
    """An uncoupled lossy resonator stays a pole of the drift, with zero residue at the port."""
    bright = Resonator(freq=7.0, levels=2, label="bright")
    hidden = Resonator(freq=7.3, levels=2, internal_quality_factor=1e4, label="hidden")
    network = PortNetwork(label="line")
    port = network.port("port", target=bright, rate=0.01)
    network.expose("p", input=port.input, output=port.output)
    result = VNA(Chip([bright, hidden], port_network=network)).resonances()

    assert result.dominant == ("bright", "hidden") and list(result.dark) == [False, True]
    np.testing.assert_allclose(result["hidden"].linewidth, TWO_PI * 7.3 / 1e4, rtol=1e-12)
    np.testing.assert_allclose(result["hidden"].residues, 0.0, atol=1e-18)


@pytest.mark.unit
@pytest.mark.parametrize("rate", [1e-12, 1e-18])
def test_a_weak_port_keeps_its_pole_bright_and_its_linewidth(rate):
    """Each port is compared with its own scale. A lossless resonator behind a weak port, next to a port of rate
    1 /ns, reflects −1 on resonance, so its pole is not dark. Its linewidth is the port rate to round-off, and the
    poles rebuild that reflection."""
    chip = resonator_pair(rate, 1.0, second_freq=5.0)
    result = VNA(chip).resonances()

    assert result.dominant == ("r2", "r1") and not np.any(result.dark)
    np.testing.assert_allclose(result["r1"].linewidth, rate, rtol=1e-12)
    np.testing.assert_allclose(result["r1"].residues, [[1j * rate / TWO_PI, 0.0], [0.0, 0.0]], rtol=1e-12)
    np.testing.assert_allclose(VNA(chip).sweep([7.0]).matrix[0, 0, 0], -1.0, atol=1e-9)
    np.testing.assert_allclose(result.scattering([7.0])[0, 0, 0], -1.0, atol=1e-9)


@pytest.mark.unit
def test_unresolved_poles_raise_and_a_nearby_drift_keeps_simple_poles():
    """At (κ₂ − κ₁)/4 = g_ω the drift is −0.2 I + N with N² = 0 after the common frequency: an exceptional point,
    where the eigensolve cannot resolve the two poles. Away from it, the simple poles rebuild the response."""
    coupling = 0.1 / TWO_PI
    with pytest.raises(ValueError, match=r"near 7(\.\d+)? GHz are ill-conditioned and too close to resolve"):
        VNA(resonator_pair(0.2, 0.6, coupling)).resonances()
    chip = resonator_pair(0.2, 0.61, coupling)
    frequencies = np.linspace(6.9, 7.1, 11)
    np.testing.assert_allclose(VNA(chip).resonances().scattering(frequencies), VNA(chip).sweep(frequencies).matrix,
                               rtol=1e-9, atol=1e-11)


@pytest.mark.unit
def test_pumps_and_chips_without_a_weak_probe_route_raise():
    """Pumped responses and chips that neither route covers raise with the reason of sweep()."""
    vna = VNA(readout())
    vna.pump("p", freq=7.0, amplitude=0.1)
    with pytest.raises(ValueError, match="without pumps"):
        vna.resonances()
    with pytest.raises(ValueError, match="needs a weak-probe route.*not passive and harmonic"):
        VNA(readout(approximation=Exact())).resonances()


@pytest.mark.optional_backend
def test_backends_give_the_same_poles():
    """QuTiP and dynamiqs build the same drift, so the poles and residues agree."""
    pytest.importorskip("dynamiqs")
    qutip, dynamiqs = VNA(readout()).resonances(), VNA(readout(backend="dynamiqs")).resonances()
    for name in ("frequencies", "linewidths", "residues", "composition"):
        np.testing.assert_allclose(np.asarray(getattr(dynamiqs, name)), np.asarray(getattr(qutip, name)),
                                   rtol=1e-12, atol=1e-16)


@pytest.mark.optional_backend
def test_repeated_poles_keep_a_complete_eigenbasis_under_jit():
    """Two identical uncoupled resonators give a repeated pole. Inside jax.jit, each copy keeps its own eigenvector,
    so the rebuilt response equals the reflection 1 − κ/(κ/2 + i 2π (f − 7)) of each port."""
    pytest.importorskip("dynamiqs")
    import jax

    chip = resonator_pair(0.1, 0.1, backend="dynamiqs")
    frequencies = np.linspace(6.95, 7.05, 5)

    def rebuilt(freq):
        result = VNA(chip.with_params({"r1.freq": freq})).resonances()
        return result.scattering(frequencies), result.composition

    matrix, composition = jax.jit(rebuilt)(7.0)
    reflection = 1 - 0.1 / (0.05 + 1j * TWO_PI * (frequencies - 7.0))
    np.testing.assert_allclose(np.asarray(matrix), reflection[:, None, None] * np.eye(2), rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.sum(np.asarray(composition), axis=0), [1.0, 1.0], rtol=0, atol=1e-13)


@pytest.mark.validation
@pytest.mark.optional_backend
def test_linewidth_derivative_matches_central_differences():
    """On dynamiqs, the derivative of the qubit linewidth in g matches central differences, also under jit.
    Selecting the pole by label stays possible in a derivative; eigenvector quantities raise there."""
    pytest.importorskip("dynamiqs")
    import jax

    chip = readout(backend="dynamiqs")

    def linewidth(g):
        return VNA(chip.with_params({"qr.g": g})).resonances()["q"].linewidth

    g, step = 0.1, 1e-5
    finite = (linewidth(g + step) - linewidth(g - step)) / (2 * step)
    np.testing.assert_allclose(jax.grad(linewidth)(g), finite, rtol=1e-6)
    traced = jax.jit(jax.grad(lambda value: VNA(chip.with_params({"qr.g": value})).resonances().linewidths[0]))
    np.testing.assert_allclose(traced(g), finite, rtol=1e-6)
    with pytest.raises(NotImplementedError):
        jax.grad(lambda value: VNA(chip.with_params({"qr.g": value})).resonances().composition[0, 0])(g)
    with pytest.raises(ValueError, match="concrete values"):
        jax.jit(lambda value: VNA(chip.with_params({"qr.g": value})).resonances()["q"].linewidth)(g)
