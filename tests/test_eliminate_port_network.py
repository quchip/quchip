"""Network-connected linear-mode elimination transforms the SLH boundary."""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from quchip import Capacitive, Chip, CrossKerr, DuffingTransmon, Exact, PortNetwork, Resonator, RWA, eliminate


def _readout_chip(*, phase_shift: float | None = None, approximation=RWA()) -> tuple[Chip, float]:
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
    resonator = Resonator(freq=6.0, levels=3, label="r")
    network = PortNetwork(label="readout_line")
    port = network.port("readout", target=resonator, rate=0.03, phase=0.2)
    if phase_shift is not None:
        shifter = network.phase_shift("cable", phase=phase_shift)
        line = network.delay("line", duration=0.125)
        network.cascade(port, shifter, line.input_terminal("1"))
        network.expose("feedline", input=port.input, output=line.output_terminal("2"))
    chip = Chip(
        [qubit, resonator],
        [Capacitive(qubit, resonator, g=0.04, label="qr")],
        port_network=network,
        approximation=approximation,
    )
    # The retained jump uses exp(-S), whose one-excitation transfer is sin(g/Delta).
    return chip, 0.03 * np.sin(0.04 / (5.0 - 6.0)) ** 2


def test_eliminate_retargets_port_without_double_counting_purcell() -> None:
    """The transformed external channel is retained without folding it into survivor T1."""
    chip, expected_rate = _readout_chip()

    reduced = eliminate(chip, "r").chip

    assert reduced.port_network is not None
    assert tuple(plane.label for plane in reduced.port_network.external_ports) == ("readout",)
    assert reduced.port("readout").resolve_targets(reduced) == ("q",)
    assert reduced["q"].T1 is None
    channel = reduced.resolve().slh.external_channels[0]
    matrix_element = channel.coupling.to_dense()[0, 1]
    np.testing.assert_allclose(np.abs(matrix_element) ** 2, expected_rate, rtol=1e-6)


def test_eliminate_keeps_scattering_and_adds_the_mode_reflection_to_the_plane() -> None:
    """Elimination keeps the core scattering and named plane, adding the mode's reflection innermost."""
    chip, _ = _readout_chip(phase_shift=0.37)
    before = chip.resolve().slh

    reduced = eliminate(chip, "r").chip
    after = reduced.resolve().slh

    assert reduced.port_network is not None
    assert tuple(plane.label for plane in reduced.port_network.external_ports) == ("feedline",)
    np.testing.assert_allclose(after.S, before.S)
    assert after.external_channels[0].key == "feedline"
    plane, original = after.external_channels[0].reference, before.external_channels[0].reference
    (section,) = plane.inbound[len(original.inbound):]
    assert plane.inbound[:len(original.inbound)] == original.inbound
    assert plane.outbound == (section, *original.outbound)
    frequencies = np.array([5.5, 5.99, 6.0, 6.01, 6.5])
    prefix = "network.component.r_reflection."
    freq, rate = float(reduced.parameters[prefix + "freq"]), float(reduced.parameters[prefix + "external_rate"])
    detuning = 2 * np.pi * (frequencies - freq)
    reflection = 1 - rate / (rate / 2 - 1j * detuning)
    np.testing.assert_allclose(np.asarray(section(frequencies)) ** 2, reflection, rtol=1e-12)


@pytest.mark.parametrize("method", ["sw", "exact"])
def test_reflection_section_sits_at_the_dressed_mode_frequency(method) -> None:
    """Without RWA, the section sits at the mode's dressed frequency, counter-rotating shifts included."""
    chip, _ = _readout_chip(approximation=Exact())

    reduced = eliminate(chip, "r", method=method).chip

    # The bare frequency is 1.5 MHz low. SW misses the fourth-order shift g⁴/Δ³ = 2.6 kHz.
    tolerance = 1e-9 if method == "exact" else 2 * 0.04**4
    assert abs(reduced.parameters["network.component.r_reflection.freq"] - chip.freq("r")) < tolerance


@pytest.mark.parametrize("approximation", [RWA(), Exact()])
def test_exact_elimination_also_retains_the_network_boundary(approximation) -> None:
    """Exact reduction carries the port operator through its dressed rotation."""
    chip, _ = _readout_chip(approximation=approximation)

    hamiltonian = np.asarray(chip.unresolved_hamiltonian().matrix(), dtype=complex)
    if isinstance(approximation, RWA):
        excitations = (np.arange(3)[:, None] + np.arange(3)).ravel()
        hamiltonian *= excitations[:, None] == excitations[None, :]
    _, eigenvectors = np.linalg.eigh(hamiltonian)
    kept = np.array([0, 3, 6])
    selected = np.argmax(np.abs(eigenvectors[kept]) ** 2, axis=1)
    overlap = eigenvectors[np.ix_(kept, selected)]
    values, vectors = np.linalg.eigh(overlap @ overlap.conj().T)
    inverse_sqrt = (vectors * values ** -.5) @ vectors.conj().T
    embedding = eigenvectors[:, selected] @ (inverse_sqrt @ overlap).conj().T
    lowering = np.diag(np.sqrt(np.arange(1, 3)), 1)
    mode_lowering = np.kron(np.eye(3), lowering)
    expected = np.sqrt(0.03) * np.exp(0.2j) * embedding.conj().T @ mode_lowering @ embedding

    reduced = eliminate(chip, "r", method="exact").chip

    assert reduced.port_network is not None
    actual = reduced.resolve().slh.external_channels[0].coupling.to_dense()
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.validation
@pytest.mark.optional_backend
def test_transformed_boundary_is_differentiable_in_coupling_strength() -> None:
    """SW port transfer, reflection frequency, and rate follow their analytic derivatives."""
    pytest.importorskip("dynamiqs")

    def boundary_values(g: jax.Array) -> jax.Array:
        qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        resonator = Resonator(freq=6.0, levels=3, label="r")
        network = PortNetwork(label="line")
        network.port("readout", target=resonator, rate=0.03)
        chip = Chip(
            [qubit, resonator],
            [Capacitive(qubit, resonator, g=g, label="qr")],
            port_network=network,
            backend="dynamiqs",
        )
        reduced = eliminate(chip, "r").chip
        coupling = reduced.resolve().slh.external_channels[0].coupling
        prefix = "network.component.r_reflection."
        return jnp.array([jnp.abs(coupling.to_dense()[0, 1]) ** 2,
                          reduced.parameters[prefix + "freq"],
                          reduced.parameters[prefix + "external_rate"]])

    g, detuning, rate = 0.04, 6.0 - 5.0, 0.03
    gradient = jax.jit(jax.jacrev(boundary_values))(jnp.asarray(g))
    # SW gives sin²(g/Δ) port transfer, f_r + g²/Δ, and κ cos²(g/Δ).
    rate_gradient = rate * np.sin(2 * g / detuning) / detuning
    np.testing.assert_allclose(gradient, [rate_gradient, 2 * g / detuning, -rate_gradient], rtol=1e-5)


def test_eliminate_rejects_port_connected_nonlinear_target() -> None:
    """Unsupported nonlinear boundary elimination fails instead of dropping its port."""
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
    network = PortNetwork(label="line")
    network.port("drive", target=qubit, rate=0.02)
    chip = Chip([qubit], port_network=network)

    with pytest.raises(NotImplementedError, match="linear Fock-mode"):
        eliminate(chip, "q")


def _feedline_chip() -> Chip:
    """Three qubits read out through resonators in series on one feedline."""
    qubits = [DuffingTransmon(freq=freq, anharmonicity=-0.26, levels=2, label=f"q{index}")
              for index, freq in enumerate((5.326, 5.192, 5.06), start=1)]
    readouts = [Resonator(freq=freq, levels=2, label=f"r{index}", internal_quality_factor=1e4)
                for index, freq in enumerate((6.558, 6.657, 6.756), start=1)]
    network = PortNetwork(label="feed")
    ports = [network.port(f"p{index}", target=readout, rate=0.01) for index, readout in enumerate(readouts, start=1)]
    network.cascade(*ports)
    network.expose("line", input=ports[0].input, output=ports[-1].output)
    couplings = [Capacitive(qubit, readout, g=0.04) for qubit, readout in zip(qubits, readouts)]
    return Chip([*qubits, *readouts], couplings, port_network=network, frame=5.2, approximation=RWA())


@pytest.mark.parametrize(
    ("modes", "options", "legs", "beyond"),
    [(("r2",), {"method": "exact"}, ([], ["r2_transmission"]), ("r3",)),
     (("r2", "r1"), {"method": "exact"}, (["r1_transmission"], ["r2_transmission"]), ()),
     (("r2", "r1"), {"method": "sw", "local": True}, (["r1_transmission"], ["r2_transmission"]), ())],
    ids=["middle-readout", "middle-then-first-readout", "middle-then-first-readout-local"],
)
def test_feedline_readout_elimination_keeps_the_line_transmission(modes, options, legs, beyond) -> None:
    """Readouts removed from a shared feedline keep their transmission as one-pass sections."""
    import json

    from quchip import VNA

    chip = _feedline_chip()
    legs_by_mode = {edge.device_b_label: edge for edge in chip.couplings}
    ports_by_mode = {port.resolve_targets(chip)[0]: port for port in chip.ports}
    detuning = min(abs(chip[mode].freq - chip[legs_by_mode[mode].device_a_label].freq) for mode in modes)
    g = max(legs_by_mode[mode].g for mode in modes)
    rate = float(ports_by_mode[modes[-1]].rate_value(chip))
    # Purcell dispersion for exact reduction; the fourth-order energy shift also limits SW.
    bound = (2 * (g / detuning)**2 * rate / (2 * np.pi * detuning) if options["method"] == "exact"
             else 2 * 8 * np.pi * g**4 / (rate * detuning**3))
    last_edge = legs_by_mode[modes[-1]]
    mode_freq = chip[modes[-1]].freq
    last_detuning = mode_freq - chip[last_edge.device_a_label].freq
    # The isolated RWA pair gives the removed mode's dressed lowering weight.
    weight = ((1 + last_detuning / np.hypot(last_detuning, 2 * last_edge.g)) / 2
              if options["method"] == "exact" else np.cos(last_edge.g / last_detuning)**2)
    kappa_over_delta = (weight * rate / (2 * np.pi * min(abs(chip[label].freq - mode_freq) for label in beyond))
                        if beyond else 0.0)
    reduced_chip = chip
    for mode in modes:
        result = eliminate(reduced_chip, mode, **options)
        reduced_chip = result.chip
    restored = Chip.from_dict(json.loads(json.dumps(reduced_chip.to_dict())))

    # A section filters the outbound leg, unless kept readouts lie only downstream of it.
    plane = restored.resolve().slh.external_channels[0].reference
    assert ([element.label for element in plane.inbound], [element.label for element in plane.outbound]) == legs
    # The diagnostic uses incoming detunings, so it is a scale rather than a strict field-error bound.
    section = result.validity[f"{modes[-1]}_transmission"]
    assert section["kappa_over_delta"] == pytest.approx(kappa_over_delta, rel=1e-12, abs=1e-12)
    assert section["is_valid"]
    assert f"{modes[-1]}_transmission κ/Δ={float(section['kappa_over_delta']):.2g} ✓" in result.describe()

    # The readout-like eigenvalue of each isolated RWA pair sets its resonance.
    poles = []
    for mode, edge in legs_by_mode.items():
        fr, fq = chip[mode].freq, chip[edge.device_a_label].freq
        poles.append((fr + fq + np.hypot(fr - fq, 2 * edge.g)) / 2)
    frequencies = np.concatenate([pole + np.array([-1e-3, 0.0, 1e-3]) for pole in poles])
    full = np.asarray(VNA(chip).sweep(frequencies).matrix)[:, 0, 0]
    reduced = np.asarray(VNA(restored).sweep(frequencies).matrix)[:, 0, 0]
    # Reducing the line's own Hamiltonian term with the chip moves the kept dips by 8e-3.
    # A second removal that drops its readout's internal loss errs by 0.6 at that dip.
    assert np.max(np.abs(full - reduced)) < bound


def _bus_readout_chip(port_count: int = 1) -> Chip:
    first = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q1")
    second = DuffingTransmon(freq=5.3, anharmonicity=-0.25, levels=3, label="q2")
    resonator = Resonator(freq=6.0, levels=3, label="r1")
    network = PortNetwork(label="line")
    for index in range(port_count):
        network.port(f"readout{index}", target=resonator, rate=0.05)
    return Chip(
        [first, second, resonator],
        [Capacitive(first, resonator, g=0.06), Capacitive(first, second, g=0.01)],
        port_network=network,
        approximation=RWA(),
    )


def test_local_feedline_section_keeps_spectator_frequency_corrections() -> None:
    """A readout outside the generator patch still contributes its incoming gap to the section's diagnostic."""
    from quchip import EffectiveTerms

    source = _feedline_chip()
    chip = Chip(source.devices, source.couplings, port_network=source.port_network.copy(), frame=source.frame,
                approximation=source.approximation,
                effective_terms=[EffectiveTerms(("r3",), (2,), np.diag([0.0, 0.07]), label="shift")])
    full, local = eliminate(chip, "r2"), eliminate(chip, "r2", local=True)
    key = "r2_transmission"
    rate = local.chip.parameters[f"network.component.{key}.external_rate"]
    expected = rate / (2 * np.pi * (chip["r3"].freq + 0.07 - chip["r2"].freq))
    assert local.validity[key]["kappa_over_delta"] == pytest.approx(expected, rel=1e-12)
    assert local.validity[key]["kappa_over_delta"] == pytest.approx(full.validity[key]["kappa_over_delta"], rel=1e-12)


def _spectator_dressed_boundary(g=0.12, *, backend="qutip", shared=True, effective=False) -> Chip:
    """A conserved boundary occupation shifts the spectator pair's virtual excitation energy."""
    from quchip import EffectiveTerms

    mode = Resonator(freq=7.0, levels=2, label="m")
    first = Resonator(freq=4.0, levels=2, label="a")
    second = Resonator(freq=5.0, levels=2, label="b")
    network = PortNetwork(label="line")
    port = network.port("pm", target=mode, rate=0.03)
    if shared:
        other = network.port("pb", target=second, rate=0.02)
        network.cascade(port, other)
        network.expose("feed", input=port.input, output=other.output)
    couplings = [CrossKerr(mode, first, chi=0.7, label="ma")]
    terms = []
    if effective:
        coordinate = np.array([[0.0, 1.0], [1.0, 0.0]])
        terms = [EffectiveTerms(("a", "b"), (2, 2), g * np.kron(coordinate, coordinate), label="pair")]
    else:
        couplings.append(Capacitive(first, second, g=g, label="ab"))
    return Chip([mode, first, second], couplings, port_network=network, approximation=Exact(),
                backend=backend, effective_terms=terms)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("effective", [False, True])
def test_local_section_includes_spectator_dressing(shared, effective) -> None:
    """The section pole and weight include virtual states outside the elimination generator's support."""
    chip = _spectator_dressed_boundary(shared=shared, effective=effective)
    full, local = eliminate(chip, "m"), eliminate(chip, "m", local=True)
    prefix = f"network.component.m_{'transmission' if shared else 'reflection'}."
    # Each m occupation gives a two-state spectator rotation with gap E or E + chi.
    gap, chi, g = 9.0, 0.7, 0.12
    frequency = 7.0 + g**2 / gap - g**2 / (gap + chi)
    weight = np.cos(g / gap - g / (gap + chi)) ** 2
    expected = np.array([frequency, 0.03 * weight])
    for result in (full, local):
        actual = [result.chip.parameters[prefix + key] for key in ("freq", "external_rate")]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-14)
    projection = next(terms.projection for terms in local.chip.effective_terms if terms.projection is not None)
    assert projection.source_labels == ("m", "a")


@pytest.mark.validation
@pytest.mark.optional_backend
@pytest.mark.parametrize("g", [0.0, 0.12])
def test_local_spectator_section_derivatives_match_analytic_rotation(g) -> None:
    """A traced spectator coupling retains the pole and weight curvature, including at zero coupling."""
    pytest.importorskip("dynamiqs")

    def section_values(strength):
        reduced = eliminate(_spectator_dressed_boundary(strength, backend="dynamiqs"), "m", local=True).chip
        prefix = "network.component.m_transmission."
        return jnp.array([reduced.parameters[prefix + key] for key in ("freq", "external_rate")])

    difference = 1 / 9.0 - 1 / 9.7
    actual = jax.jit(jax.jacfwd(jax.jacfwd(section_values)))(jnp.asarray(g))
    expected = [2 * difference, -0.06 * difference**2 * np.cos(2 * g * difference)]
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-13)


@pytest.mark.parametrize("method", ["sw", "exact"])
def test_field_elimination_transforms_the_port_onto_every_survivor(method) -> None:
    """A port-coupled mode on a multi-device chip reduces to one joint port with the Purcell rate."""
    reduced = eliminate(_bus_readout_chip(), "r1", method=method).chip

    port = reduced.port("readout0")
    assert port.resolve_targets(reduced) == ("q1", "q2")
    lowering = np.asarray(reduced.backend.to_array(port.operator.matrix(backend=reduced.backend)))
    excited_q1 = np.ravel_multi_index((1, 0), (3, 3))
    # Purcell rate κ(g/Δ)². Exact dressing gives sin²θ with tan 2θ = 2g/Δ, 0.94%
    # below (g/Δ)² here, and q1-q2 hybridization (J/Δ)² adds about 0.1%.
    np.testing.assert_allclose(0.05 * abs(lowering[0, excited_q1]) ** 2, 0.05 * 0.06**2, rtol=2e-2)


def _issue_readout_chip(g: float = 0.06, *, internal_quality_factor: float | None = None) -> Chip:
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q", T1=20000.0)
    resonator = Resonator(freq=6.0, levels=3, label="r", internal_quality_factor=internal_quality_factor)
    network = PortNetwork(label="m")
    network.port("r", target=resonator, rate=0.05)
    return Chip([qubit, resonator], [Capacitive(qubit, resonator, g=g)], port_network=network, approximation=RWA())


@pytest.mark.parametrize(
    ("build", "mode", "method"),
    [(_issue_readout_chip, "r", "sw"), (_issue_readout_chip, "r", "exact"), (_bus_readout_chip, "r1", "exact")],
    ids=["two-device-sw", "two-device-exact", "three-device-exact"],
)
def test_reduced_boundary_reproduces_the_full_reflection(build, mode, method) -> None:
    """VNA on the reduced chip matches the full chip beyond the eliminated mode's own reflection."""
    from quchip import VNA

    chip = build()
    frequencies = np.array([4.9, 4.99, 5.1])
    full = np.asarray(VNA(chip).sweep(frequencies).matrix)[:, 0, 0]
    reduced = np.asarray(VNA(eliminate(chip, mode, method=method).chip).sweep(frequencies).matrix)[:, 0, 0]

    # The dropped reflection |S_r - 1| is about κ/(2π|Δ|) = 8e-3. With the
    # dressed section kept, the exact route leaves 1e-7 here. SW leaves 5e-5,
    # 1.8 (g/Δ)² κ/(2π|Δ|), from its reduced qubit and port, not the section.
    next_order = 0.06**2 * 0.05 / (2 * np.pi)
    assert np.max(np.abs(full - reduced)) < 4 * next_order


@pytest.mark.parametrize("method", ["sw", "exact"])
def test_reduced_boundary_reproduces_the_reflection_across_the_eliminated_resonance(method) -> None:
    """VNA on the reduced chip matches the full chip across the eliminated mode's own dip."""
    from quchip import VNA

    chip = _issue_readout_chip(internal_quality_factor=1e4)
    frequencies = chip.freq("r") + np.linspace(-0.02, 0.02, 21)
    full = np.asarray(VNA(chip).sweep(frequencies).matrix)[:, 0, 0]
    reduced = np.asarray(VNA(eliminate(chip, "r", method=method).chip).sweep(frequencies).matrix)[:, 0, 0]

    # Bare section values put the dip 3.6 MHz, half a linewidth, too low. With
    # dressed values, the exact route leaves the Purcell dispersion, below
    # (g/Δ)² κ/(2π|Δ|). SW also misses the fourth-order energy g⁴/Δ³, which
    # moves |S| by about 8π g⁴/(κ|Δ|³) = 6.5e-3.
    next_order = 0.06**2 * 0.05 / (2 * np.pi)
    bound = 2 * next_order if method == "exact" else 2 * 8 * np.pi * 0.06**4 / 0.05
    assert np.max(np.abs(full - reduced)) < bound


@pytest.mark.validation
def test_reduced_boundary_residual_is_second_order_in_the_coupling() -> None:
    """Halving g quarters the residual across the eliminated dip, as Purcell dispersion predicts."""
    from quchip import VNA

    def residual(g: float) -> float:
        chip = _issue_readout_chip(g)
        frequencies = chip.freq("r") + np.linspace(-0.02, 0.02, 41)
        full = np.asarray(VNA(chip).sweep(frequencies).matrix)[:, 0, 0]
        reduced = np.asarray(VNA(eliminate(chip, "r", method="exact").chip).sweep(frequencies).matrix)[:, 0, 0]
        return float(np.max(np.abs(full - reduced)))

    np.testing.assert_allclose(residual(0.06) / residual(0.03), 4.0, rtol=0.05)


def test_reduced_boundary_serializes_with_the_chip() -> None:
    """A reduced chip with its mode-reflection section round-trips and keeps its scattering."""
    import json

    from quchip import VNA

    reduced = eliminate(_issue_readout_chip(), "r").chip
    restored = Chip.from_dict(json.loads(json.dumps(reduced.to_dict())))
    frequencies = np.array([4.95, 5.05])

    np.testing.assert_allclose(
        np.asarray(VNA(restored).sweep(frequencies).matrix), np.asarray(VNA(reduced).sweep(frequencies).matrix),
        atol=1e-12,
    )


def _two_readout_chip(backend: str = "qutip") -> Chip:
    """Two qubits on a bus, each read out through its own port-coupled resonator."""
    first = DuffingTransmon(freq=5.326, anharmonicity=-0.262, levels=2, label="q1", T1=30000.0)
    second = DuffingTransmon(freq=5.192, anharmonicity=-0.264, levels=2, label="q2", T1=30000.0)
    bus = Resonator(freq=6.298, levels=2, label="bus")
    readout1 = Resonator(freq=6.558, levels=2, label="r1")
    readout2 = Resonator(freq=6.657, levels=2, label="r2")
    network = PortNetwork(label="feed")
    network.port("p1", target=readout1, rate=0.01)
    network.port("p2", target=readout2, rate=0.01)
    couplings = [Capacitive(first, bus, g=0.03), Capacitive(second, bus, g=0.03),
                 Capacitive(first, readout1, g=0.04), Capacitive(second, readout2, g=0.04)]
    return Chip([first, second, bus, readout1, readout2], couplings, port_network=network, frame=5.2,
                approximation=RWA(), backend=backend)


def _eliminate_in_order(chip: Chip, order: tuple[str, ...], method: str) -> Chip:
    for mode in order:
        chip = eliminate(chip, mode, method=method).chip
    return chip


@pytest.mark.parametrize("method", ["sw", "exact"])
def test_second_readout_elimination_carries_the_earlier_transformed_port(method) -> None:
    """Both readout modes reduce in either order; the earlier transformed port moves onto the survivors."""
    from quchip import VNA

    frequencies = np.array([5.15, 5.25, 5.4])
    sweeps = []
    for order in (("r1", "r2"), ("r2", "r1")):
        reduced = _eliminate_in_order(_two_readout_chip(), order, method)
        assert {port.label: port.resolve_targets(reduced) for port in reduced.ports} == {
            "p1": ("q1", "q2", "bus"), "p2": ("q1", "q2", "bus"),
        }
        assert reduced.port_network is not None
        assert {"r1_reflection", "r2_reflection"} <= {component.label for component in reduced.port_network.components}
        sweeps.append(np.asarray(VNA(reduced).sweep(frequencies).matrix))
    # The orders differ by less than 1e-12, well below either reduction's
    # residual against the full chip (2e-8 exact, 1e-7 SW).
    np.testing.assert_allclose(sweeps[0], sweeps[1], atol=2e-7)


@pytest.mark.optional_backend
def test_reduced_chip_with_transformed_and_kept_ports_resolves_inside_jit() -> None:
    """A transformed joint port and a kept survivor port keep their declared changes inside jit."""
    pytest.importorskip("dynamiqs")
    reduced = eliminate(_two_readout_chip(backend="dynamiqs"), "r1", method="exact").chip
    eager = reduced.resolve()
    eager_jumps = [reduced.backend.to_array(op) for op in reduced.backend._collapse_operators(eager)]

    def jumps(t1):
        candidate = reduced.with_params({"q1.T1": t1})
        return [candidate.backend.to_array(op) for op in candidate.backend._collapse_operators(candidate.resolve())]

    traced = jax.jit(jumps)(30000.0)
    assert len(traced) == len(eager_jumps)
    for actual, expected in zip(traced, eager_jumps, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-12)


@pytest.mark.validation
def test_doubly_reduced_boundary_reproduces_the_full_two_port_response() -> None:
    """Eliminating both readout modes leaves each port with only its own Purcell dispersion residual."""
    from quchip import VNA

    chip = _two_readout_chip()
    frequencies = np.array([5.15, 5.19, 5.2, 5.25, 5.32, 5.33, 5.4])
    full = np.asarray(VNA(chip).sweep(frequencies).matrix)
    reduced = np.asarray(VNA(_eliminate_in_order(chip, ("r1", "r2"), "exact")).sweep(frequencies).matrix)

    # With dressed sections, each reflection misses less than 0.1 (g/Δ)² κ/(2π|Δ|)
    # of its own readout mode. Transmission misses less than 0.1 of the smaller one.
    next_order = [0.04**2 / detuning**2 * 0.01 / (2 * np.pi * detuning) for detuning in (1.232, 1.465)]
    for index, bound in enumerate(next_order):
        assert np.max(np.abs(full[:, index, index] - reduced[:, index, index])) < 4 * bound
    assert np.max(np.abs(full[:, 1, 0] - reduced[:, 1, 0])) < min(next_order)
    assert np.max(np.abs(full[:, 0, 1] - reduced[:, 0, 1])) < min(next_order)


def test_field_elimination_rejects_an_authored_joint_port() -> None:
    """A port authored across two modes reaches each directly and has no reduced boundary."""
    first = Resonator(freq=6.0, levels=2, label="r1")
    second = Resonator(freq=6.2, levels=2, label="r2")
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
    lowering = np.diag([1.0], 1)
    network = PortNetwork(label="line")
    network.port("joint", target=(first, second), rate=0.01,
                 operator=np.kron(lowering, np.eye(2)) + np.kron(np.eye(2), lowering))
    chip = Chip([qubit, first, second], [Capacitive(qubit, first, g=0.04), Capacitive(qubit, second, g=0.04)],
                port_network=network, approximation=RWA())

    with pytest.raises(NotImplementedError, match="unsupported ports: \\['joint'\\]"):
        eliminate(chip, "r1")


def test_field_elimination_rejects_a_mode_with_several_ports() -> None:
    """A mode transmitting between two ports has no reduced boundary yet."""
    with pytest.raises(NotImplementedError, match="transmission between them"):
        eliminate(_bus_readout_chip(port_count=2), "r1")


def test_field_elimination_rejects_a_plane_shared_with_another_port() -> None:
    """The mode's reflection cannot be factored out of a plane that also carries another port."""
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
    resonator = Resonator(freq=6.0, levels=2, label="r")
    other = Resonator(freq=7.0, levels=2, label="s")
    network = PortNetwork(label="line")
    readout = network.port("readout", target=resonator, rate=0.03)
    spectator = network.port("spectator", target=other, rate=0.03)
    splitter = network.beam_splitter("splitter", eta=0.5)
    network.connect(readout.output, splitter.input_terminal("left"))
    network.connect(spectator.output, splitter.input_terminal("right"))
    network.expose("a", input=readout.input, output=splitter.output_terminal("left"))
    network.expose("b", input=spectator.input, output=splitter.output_terminal("right"))
    chip = Chip(
        [qubit, resonator, other],
        [Capacitive(qubit, resonator, g=0.04, label="qr")],
        port_network=network,
    )

    with pytest.raises(NotImplementedError, match="shares its external plane"):
        eliminate(chip, "r")


def test_field_elimination_rejects_projected_survivor_basis() -> None:
    """A resolved-size port is not stored as an authored-size operator."""
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=4, label="q")
    qubit.basis = "eigen"
    qubit.projection_levels = 2
    resonator = Resonator(freq=6.0, levels=3, label="r")
    network = PortNetwork(label="line")
    network.port("readout", target=resonator, rate=0.03)
    chip = Chip(
        [qubit, resonator],
        [Capacitive(qubit, resonator, g=0.04, label="qr")],
        port_network=network,
        approximation=Exact(),
    )

    with pytest.raises(NotImplementedError, match="projected survivor basis"):
        eliminate(chip, "r")


def test_eliminate_custom_harmonic_boundary_matches_resonator() -> None:
    """A declared harmonic mode has the same reduced boundary as a Resonator."""
    from quchip import FockDevice, Scalar, parameter

    class HarmonicMode(FockDevice):
        freq: Scalar = parameter(positive=True)

        def local_hamiltonian(self, op, p):
            return p.freq * op.n

    reference, _ = _readout_chip()
    q = reference["q"].copy()
    mode = HarmonicMode(6.0, levels=3, label="r")
    network = PortNetwork()
    network.port("readout", target=mode, rate=0.03, phase=0.2)
    custom = Chip([q, mode], [Capacitive(q, mode, g=0.04)], port_network=network)
    expected = eliminate(reference, "r").chip.resolve()
    actual = eliminate(custom, "r").chip.resolve()
    np.testing.assert_allclose(actual.hamiltonian().matrix(), expected.hamiltonian().matrix(), atol=1e-12)
    np.testing.assert_allclose(actual.slh.external_channels[0].coupling.to_dense(),
                               expected.slh.external_channels[0].coupling.to_dense(), atol=1e-12)
