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
    detuning = 2 * np.pi * (frequencies - 6.0)
    reflection = 1 - 0.03 / (0.015 - 1j * detuning)
    np.testing.assert_allclose(np.asarray(section(frequencies)) ** 2, reflection, rtol=1e-12)


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


def _issue_readout_chip(g: float = 0.06) -> Chip:
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q", T1=20000.0)
    resonator = Resonator(freq=6.0, levels=3, label="r")
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

    # The dropped reflection |S_r - 1| is about κ/(2π|Δ|) = 8e-3. With it kept,
    # the residual is the next order, measured at 2.8 (g/Δ)² κ/(2π|Δ|) = 8e-5.
    next_order = 0.06**2 * 0.05 / (2 * np.pi)
    assert np.max(np.abs(full - reduced)) < 4 * next_order


@pytest.mark.validation
def test_reduced_boundary_residual_is_second_order_in_the_coupling() -> None:
    """Halving g quarters the residual: it is Purcell dispersion, not a g-independent missing reflection."""
    from quchip import VNA

    frequencies = np.linspace(4.8, 5.2, 5)

    def residual(g: float) -> float:
        chip = _issue_readout_chip(g)
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
