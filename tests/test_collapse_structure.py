"""Collapse support remains structural through bases and retained coordinates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quchip import (
    RWA, VNA, Bath, Capacitive, ChargeDrive, Chip, DuffingTransmon, Gaussian, PortNetwork, QuantumSequence,
    Resonator, eliminate,
)


def _readout_chip(*, loss=False, backend="dynamiqs"):
    a = DuffingTransmon(freq=5., anharmonicity=-.25, levels=2, label="a", T1=200. if loss else None)
    b = DuffingTransmon(freq=5.2, anharmonicity=-.25, levels=2, label="b")
    bus = Resonator(freq=6.3, levels=2, label="bus")
    readout = Resonator(freq=7., levels=2, label="readout")
    network = PortNetwork(label="m")
    network.port("out", target=readout, rate=.05)
    return Chip([a, b, bus, readout], [Capacitive(a, bus, g=.08, label="leg"),
                Capacitive(b, bus, g=.08), Capacitive(a, readout, g=.06)],
                port_network=network, approximation=RWA(), backend=backend, frame="rotating")


def test_sw_spectator_port_keeps_its_operator_after_serialization():
    """An operator outside the SW generator support is unchanged in independent frames."""
    source = _readout_chip(backend="qutip")
    result = eliminate(source, "bus")
    embedding = np.asarray(result.mapping.embedding)
    original = np.asarray(source.resolve(frame="lab").collapse_terms[0].operator.to_dense())
    expected = embedding.conj().T @ original @ embedding
    for variant in (result.chip, Chip.from_dict(result.chip.to_dict())):
        actual = np.asarray(variant.resolve().collapse_terms[0].operator.to_dense())
        np.testing.assert_allclose(actual, expected, atol=2e-14)


@pytest.mark.optional_backend
def test_sw_spectator_port_accepts_traced_coupling():
    """A spectator port keeps its single phase when the SW rotation is differentiated."""
    pytest.importorskip("dynamiqs")
    source = _readout_chip()

    def amplitude(g):
        reduced = eliminate(source.with_params({"leg.g": g}), "bus").chip
        return jnp.real(reduced.resolve().collapse_terms[0].operator.to_dense()[0, 1])

    value, derivative = jax.jit(jax.value_and_grad(amplitude))(.08)
    assert float(value) == pytest.approx(float(amplitude(.08)), abs=2e-14)
    assert float(value) == pytest.approx(1., abs=2e-14)
    assert float(derivative) == 0.


@pytest.mark.validation
@pytest.mark.optional_backend
@pytest.mark.parametrize("method, local", [("sw", False), ("exact", False), ("sw", True)])
def test_reduced_channel_frequency_and_coupling_derivatives(method, local):
    """The transformed loss amplitude keeps its physical derivatives in a common frame."""
    pytest.importorskip("dynamiqs")
    source = _readout_chip(loss=True)
    source.set_frame(5.)

    def amplitude(frequency, coupling):
        reduced = eliminate(source.with_params({"a.freq": frequency, "leg.g": coupling}), "bus",
                            method=method, local=local).chip
        channel = next(term for term in reduced.resolve().collapse_terms if term.source == "a")
        return jnp.real(channel.operator.to_dense()[0, 4])

    value, gradients = jax.value_and_grad(amplitude, argnums=(0, 1))(5., .08)
    compiled, compiled_gradients = jax.jit(jax.value_and_grad(amplitude, argnums=(0, 1)))(5., .08)
    assert float(value) == pytest.approx(float(amplitude(5., .08)), abs=2e-13)
    np.testing.assert_allclose(compiled, value, atol=2e-13)
    np.testing.assert_allclose(compiled_gradients, gradients, atol=2e-12)
    step = 1e-4
    reference = [(amplitude(5. + step, .08) - amplitude(5. - step, .08)) / (2 * step),
                 (amplitude(5., .08 + step) - amplitude(5., .08 - step)) / (2 * step)]
    # Central differences have O(step^2) truncation error away from crossings.
    np.testing.assert_allclose(gradients, reference, rtol=2e-5, atol=1e-9)
    assert np.all(np.abs(gradients) > 1e-4)


@pytest.mark.validation
@pytest.mark.optional_backend
def test_collective_bath_frequency_gradient_matches_one_excitation_solution():
    """Collective decay in a common frame matches the independent two-amplitude solution."""
    from scipy.linalg import expm

    pytest.importorskip("dynamiqs")
    a = DuffingTransmon(freq=5., anharmonicity=-.25, levels=2, label="a")
    b = DuffingTransmon(freq=5.05, anharmonicity=-.25, levels=2, label="b")
    source = Chip([a, b], baths=[Bath("collective_decay", targets=[a, b], rate=.1, label="shared")],
                  frame=5., approximation=RWA(), backend="dynamiqs")

    def population(frequency):
        chip = source.with_params({"a.freq": frequency})
        result = QuantumSequence(chip).simulate(tlist=jnp.linspace(0., 5., 11), initial_state={"a": 1})
        return jnp.real(result.population("a", level=1)[-1])

    def reference(frequency):
        generator = -2j * np.pi * np.diag([frequency - 5., .05]) - .05 * np.ones((2, 2))
        return abs((expm(5. * generator) @ [1., 0.])[0]) ** 2

    point, step = 4.97, 1e-5
    plain = population(point)
    value, derivative = jax.value_and_grad(population)(point)
    compiled, compiled_derivative = jax.jit(jax.value_and_grad(population))(point)
    # The solver uses its default 1e-6 relative integration tolerance.
    np.testing.assert_allclose([plain, value, compiled], reference(point), rtol=3e-6, atol=1e-8)
    expected = (reference(point + step) - reference(point - step)) / (2 * step)
    np.testing.assert_allclose([derivative, compiled_derivative], expected, rtol=5e-5, atol=1e-7)


@pytest.mark.validation
@pytest.mark.optional_backend
def test_reduced_vna_frequency_derivative_matches_finite_difference():
    """The stationary probe frame accepts a traced survivor frequency after reduction."""
    pytest.importorskip("dynamiqs")
    source = _readout_chip(loss=True)

    def reflection(frequency):
        reduced = eliminate(source.with_params({"a.freq": frequency}), "bus").chip
        return jnp.abs(VNA(reduced).sweep(np.array([7.01])).matrix[0, 0, 0]) ** 2

    value, derivative = jax.jit(jax.value_and_grad(reflection))(5.)
    step = 1e-4
    reference = (reflection(5. + step) - reflection(5. - step)) / (2 * step)
    assert float(value) == pytest.approx(float(reflection(5.)), abs=2e-12)
    assert float(derivative) == pytest.approx(float(reference), rel=2e-5, abs=1e-9)
    assert abs(float(derivative)) > 1e-6


@pytest.mark.validation
@pytest.mark.optional_backend
@pytest.mark.parametrize("method", ["sw", "exact"])
def test_inherited_loss_traces_eliminated_frequency(method):
    """Removed-mode loss carries its integer energy weights through either reduction."""
    pytest.importorskip("dynamiqs")
    q = Resonator(freq=5., levels=2, label="q")
    bus = Resonator(freq=7., levels=2, T1=100., label="bus")
    source = Chip([q, bus], [Capacitive(q, bus, g=.08)], frame=5., approximation=RWA(), backend="dynamiqs")

    def amplitude(frequency):
        reduced = eliminate(source.with_params({"bus.freq": frequency}), "bus", method=method).chip
        return jnp.real(reduced.resolve().collapse_terms[0].operator.to_dense()[0, 1])

    value, derivative = jax.jit(jax.value_and_grad(amplitude))(7.)
    step = 1e-4
    reference = (amplitude(7. + step) - amplitude(7. - step)) / (2 * step)
    assert float(value) == pytest.approx(float(amplitude(7.)), abs=2e-13)
    assert float(derivative) == pytest.approx(float(reference), rel=2e-7, abs=1e-10)
    assert abs(float(derivative)) > 1e-3


@pytest.mark.validation
@pytest.mark.optional_backend
def test_reduced_pulse_coupling_derivative_in_per_device_frames():
    """A surviving readout port permits pulse coupling gradients in independent frames."""
    pytest.importorskip("dynamiqs")
    source = _readout_chip()
    source.wire(ChargeDrive(source["a"], label="xy"))

    def population(coupling):
        sequence = QuantumSequence(eliminate(source.with_params({"leg.g": coupling}), "bus").chip)
        sequence.schedule("xy", envelope=Gaussian(duration=8., sigmas=3., amplitude=.04), freq=4.98)
        result = sequence.simulate(tlist=jnp.linspace(0., 10., 21), initial_state={"a": 0})
        return jnp.real(result.population("a", level=1)[-1])

    value, derivative = jax.jit(jax.value_and_grad(population))(.08)
    step = 1e-4
    reference = (population(.08 + step) - population(.08 - step)) / (2 * step)
    assert float(value) == pytest.approx(float(population(.08)), rel=3e-6, abs=1e-8)
    # The finite difference also includes adaptive integration error.
    assert float(derivative) == pytest.approx(float(reference), rel=1e-3, abs=1e-6)
    assert abs(float(derivative)) > 1e-3


@pytest.mark.optional_backend
def test_sw_induced_loss_still_rejects_incompatible_frames():
    """The induced loss on the other survivor cannot share a phase in independent frames."""
    pytest.importorskip("dynamiqs")
    source = _readout_chip(loss=True)

    def amplitude(g):
        reduced = eliminate(source.with_params({"leg.g": g}), "bus").chip
        return jnp.real(reduced.resolve().collapse_terms[0].operator.to_dense()[0, 1])

    for evaluate in (amplitude, jax.grad(amplitude), jax.jit(amplitude)):
        with pytest.raises(ValueError, match="operator bands with different phases"):
            evaluate(.08)


@pytest.mark.optional_backend
def test_traced_diagonal_basis_keeps_energy_order_and_rejects_multiple_phases():
    """Structural support follows energy permutations and never assumes native level order."""
    from quchip.declarative import DeviceModel, Scalar, parameter

    pytest.importorskip("dynamiqs")

    class Diagonal(DeviceModel):
        freq: Scalar = parameter()

        def local_hamiltonian(self, op, p):
            return p.freq * op.n - 3. * (op.n @ op.n)

    def operator(frequency, frame=1.):
        a = Diagonal(freq=frequency, levels=3, label="a")
        b = Diagonal(freq=frequency, levels=3, label="b")
        chip = Chip([a, b], baths=[Bath("collective_decay", rate=.1)], frame=frame, backend="dynamiqs")
        return chip.resolve().collapse_terms[0].operator.to_dense()

    # At f=-1 the energies are strictly reversed: both native lowering operators
    # raise the energy index. At f=5 their native edges have distinct weights.
    np.testing.assert_allclose(jax.jit(operator)(-1.), operator(-1.), atol=2e-14)
    with pytest.raises(ValueError, match="operator bands with different phases"):
        operator(5.)
    with pytest.raises(jax.errors.JaxRuntimeError, match="operator bands with different phases"):
        jax.jit(operator)(5.).block_until_ready()
    np.testing.assert_allclose(jax.jit(lambda f: operator(f, 0.))(5.), operator(5., 0.), atol=2e-14)
