"""Network-connected linear-mode elimination transforms the SLH boundary."""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from quchip import Capacitive, Chip, DuffingTransmon, Exact, PortNetwork, Resonator, eliminate


def _readout_chip(*, phase_shift: float | None = None) -> tuple[Chip, float]:
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


def test_eliminate_preserves_scattering_and_explicit_exposure() -> None:
    """Elimination retains scalar scattering and the named external reference plane."""
    chip, _ = _readout_chip(phase_shift=0.37)
    before = chip.resolve().slh

    reduced = eliminate(chip, "r").chip
    after = reduced.resolve().slh

    assert reduced.port_network is not None
    assert tuple(plane.label for plane in reduced.port_network.external_ports) == ("feedline",)
    np.testing.assert_allclose(after.S, before.S)
    assert after.external_channels[0].key == "feedline"
    assert after.external_channels[0].reference == before.external_channels[0].reference


def test_exact_elimination_also_retains_the_network_boundary() -> None:
    """Exact reduction carries the port operator through its dressed rotation."""
    chip, _ = _readout_chip()

    hamiltonian = np.asarray(chip.unresolved_hamiltonian().matrix(), dtype=complex)
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

    gradient = jax.grad(effective_rate)(jnp.asarray(0.04))

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


def test_exact_field_elimination_rejects_multiple_survivors() -> None:
    """Exact field reduction refuses incompatible multi-survivor basis choices."""
    first = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="a")
    second = DuffingTransmon(freq=5.2, anharmonicity=-0.22, levels=2, label="b")
    resonator = Resonator(freq=6.0, levels=2, label="r")
    network = PortNetwork(label="line")
    network.port("readout", target=resonator, rate=0.03)
    chip = Chip(
        [first, second, resonator],
        [
            Capacitive(first, resonator, g=0.03, label="ar"),
            Capacitive(second, resonator, g=0.04, label="br"),
        ],
        port_network=network,
    )

    with pytest.raises(NotImplementedError, match="one survivor"):
        eliminate(chip, "r", method="exact")


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
