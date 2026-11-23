"""Physics checks for the reference classes of the extension guide and the built-in CosineEnvelope."""

from __future__ import annotations

from quchip.approximations import RWA


import jax
import jax.numpy as jnp
import numpy as np
import pytest

import quchip
from quchip import (
    ChargeDrive,
    Chip,
    CosineEnvelope,
    DuffingTransmon,
    GaussianDRAG,
    QuantumSequence,
    Square,
)
from quchip.control.envelopes import Envelope
from quchip.control.signal import AnalyticSignal, SignalTransform
from quchip.devices.spaces import CustomSpace
from tests.extension_guide import (
    CableLoss,
    ChargePhaseDrive,
    LossyKerrCavity,
    SpinHalf,
)


def test_spin_half_uses_a_custom_space_with_analytic_spectrum():
    spin = SpinHalf(freq=4.0, label="spin")

    assert isinstance(spin.local_space(), CustomSpace)
    np.testing.assert_allclose(
        spin.unresolved_hamiltonian().matrix(),
        np.diag([-2.0, 2.0]),
        atol=1e-12,
    )
    np.testing.assert_allclose(Chip([spin]).resolve().bases["spin"].energies, [-2.0, 2.0])


def test_spin_half_traverses_projection_drive_serialization_and_jax_paths():
    class ArrayBackend:
        @staticmethod
        def from_array(value, *, dims):
            _ = dims
            return jnp.asarray(value)

        @staticmethod
        def to_array(value):
            return jnp.asarray(value)

    spin = SpinHalf(freq=4.0, basis="eigen", label="spin")
    drive = ChargeDrive(spin, label="xy")
    chip = Chip([spin], frame="rotating", approximation=RWA())
    chip.wire(drive)
    sequence = QuantumSequence(chip)
    sequence.schedule(drive, envelope=Square(duration=5.0, amplitude=0.02), freq=4.0)

    assert len(sequence.resolve().dynamic_terms) == 2
    restored = Chip.from_dict(chip.to_dict())["spin"]
    assert isinstance(restored, SpinHalf)
    assert restored.freq == pytest.approx(4.0)

    gradient = jax.grad(
        lambda freq: jnp.real(SpinHalf(freq=freq).unresolved_hamiltonian().matrix(backend=ArrayBackend())[1, 1])
    )(4.0)
    assert gradient == pytest.approx(0.5)


def test_cosine_envelope_is_a_zero_endpoint_raised_cosine():
    envelope = CosineEnvelope(duration=10.0, amplitude=2.0)
    samples = envelope.sample(np.asarray([0.0, 2.5, 5.0, 7.5, 10.0]))

    np.testing.assert_allclose(samples, [0.0, 1.0, 2.0, 1.0, 0.0], atol=1e-12)
    restored = Envelope.from_dict(envelope.to_dict())
    assert isinstance(restored, CosineEnvelope)
    np.testing.assert_allclose(restored.sample(np.asarray([0.0, 5.0, 10.0])), [0.0, 2.0, 0.0])
    # quchip 0.4 saved this envelope under its former extensions module.
    saved = {**envelope.to_dict(), "type": "quchip.extensions.envelopes.CosineEnvelope"}
    assert isinstance(Envelope.from_dict(saved), CosineEnvelope)
    assert jax.grad(lambda amplitude: jnp.real(CosineEnvelope(10.0, amplitude).value(5.0)))(2.0) == pytest.approx(1.0)


def test_charge_phase_drive_maps_delivered_iq_to_two_observables():
    mode = DuffingTransmon(5.0, -0.25, levels=3, label="q")
    drive = ChargePhaseDrive(mode, label="iq")
    signal = AnalyticSignal.from_pulse(
        quchip.engine.ir.DriveOp(
            target_label="q",
            drive_label="iq",
            envelope=Square(duration=2.0, amplitude=0.02j),
        )
    )

    expression = drive.hamiltonian(mode, signal)

    assert expression.labels == ("q",)
    assert "I" in expression.latex() and "Q" in expression.latex()


def test_cable_loss_is_ordered_serializable_and_differentiable():
    raw = AnalyticSignal(quchip.engine.ir.Constant(1.0))
    transform = CableLoss(line="xy", loss_db=6.0)

    delivered = transform.apply({("xy", 0): raw})[("xy", 0)]
    restored = SignalTransform.from_dict(transform.to_dict())
    gradient = jax.grad(
        lambda loss_db: jnp.real(
            CableLoss(line="xy", loss_db=loss_db).apply({("xy", 0): raw})[("xy", 0)].evaluate(0.0, xp=jnp)
        )
    )(6.0)

    assert delivered.evaluate(0.0) == pytest.approx(10.0 ** (-6.0 / 20.0))
    assert isinstance(restored, CableLoss)
    assert np.isfinite(gradient) and gradient < 0.0


def test_gaussian_drag_serializes_and_keeps_signed_beta_differentiable():
    envelope = GaussianDRAG(duration=20.0, sigmas=4.0, amplitude=0.5, beta=-0.8)
    restored = Envelope.from_dict(envelope.to_dict())

    assert isinstance(restored, GaussianDRAG)
    assert restored.beta == pytest.approx(-0.8)
    time = 8.0
    gradient = jax.grad(
        lambda beta: jnp.imag(GaussianDRAG(duration=20.0, sigmas=4.0, amplitude=0.5, beta=beta).value(time))
    )(-0.8)
    assert np.isfinite(gradient)
    assert abs(float(gradient)) > 0.0


def test_lossy_kerr_cavity_authors_two_photon_loss():
    cavity = LossyKerrCavity(
        freq=5.0,
        kerr=0.02,
        two_photon_loss_rate=0.004,
        levels=4,
        label="cat",
    )

    channel = next(ch for ch in cavity.collapse_channels() if ch.name == "two_photon_loss")
    lowering = np.diag(np.sqrt(np.arange(1, 4)), k=1)
    np.testing.assert_allclose(channel.operator.matrix(), lowering @ lowering)
    assert np.asarray(channel.rate.matrix()).item() == pytest.approx(0.004)

    restored = Chip.from_dict(Chip([cavity]).to_dict())["cat"]
    assert isinstance(restored, LossyKerrCavity)
    assert restored.two_photon_loss_rate == pytest.approx(0.004)


def test_lossy_kerr_two_photon_fock_state_follows_exact_decay_law():
    rate = 0.004
    cavity = LossyKerrCavity(
        freq=5.0,
        kerr=0.02,
        two_photon_loss_rate=rate,
        levels=4,
        label="cat",
    )
    chip = Chip([cavity])
    times = np.linspace(0.0, 80.0, 5)

    result = quchip.simulate(
        chip,
        [],
        times,
        initial_state=chip.bare_state({cavity: 2}),
        e_ops={cavity: cavity.number_operator()},
        )

    np.testing.assert_allclose(
        np.real(result.expect(cavity)),
        2.0 * np.exp(-2.0 * rate * times),
        rtol=2e-5,
        atol=2e-7,
    )
