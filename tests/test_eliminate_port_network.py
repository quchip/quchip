"""Network-connected linear-mode elimination transforms the SLH boundary."""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from quchip import Capacitive, Chip, DuffingTransmon, Exact, PortNetwork, Resonator, RWA, eliminate


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
def test_transformed_port_is_differentiable_in_coupling_strength() -> None:
    """The effective external rate remains differentiable through SW reduction."""
    pytest.importorskip("dynamiqs")

    def effective_rate(g: jax.Array) -> jax.Array:
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
        coupling = eliminate(chip, "r").chip.resolve().slh.external_channels[0].coupling
        return jnp.abs(coupling.to_dense()[0, 1]) ** 2

    gradient = jax.jit(jax.grad(effective_rate))(jnp.asarray(0.04))

    np.testing.assert_allclose(gradient, 0.03 * np.sin(2.0 * 0.04), rtol=1e-5)


def test_eliminate_rejects_port_connected_nonlinear_target() -> None:
    """Unsupported nonlinear boundary elimination fails instead of dropping its port."""
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
    network = PortNetwork(label="line")
    network.port("drive", target=qubit, rate=0.02)
    chip = Chip([qubit], port_network=network)

    with pytest.raises(NotImplementedError, match="linear Fock-mode"):
        eliminate(chip, "q")


def test_eliminate_rejects_port_that_generates_a_cascade_hamiltonian() -> None:
    """A field reduction cannot double-count an active cascade Hamiltonian."""
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
    resonator = Resonator(freq=6.0, levels=2, label="r")
    network = PortNetwork(label="active_line")
    readout = network.port("readout", target=resonator, rate=0.03)
    downstream = network.port("downstream", target=qubit, rate=0.02)
    network.cascade(readout, downstream)
    network.expose("feedline", input=readout.input, output=downstream.output)
    chip = Chip(
        [qubit, resonator],
        [Capacitive(qubit, resonator, g=0.04, label="qr")],
        port_network=network,
    )

    with pytest.raises(NotImplementedError, match="cascade-generated Hamiltonian"):
        eliminate(chip, "r")


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
