"""Physics checks for the shared authored dissipation surface."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import quchip
from quchip import Bath, Chip, DuffingTransmon
from quchip.control.drive import BaseDrive
from quchip.declarative import CollapseChannel
from quchip.declarative.dissipation import collapse_parameter_paths
from quchip.declarative.expr import ParameterNamespace
from quchip.declarative.ops import LocalOps
from quchip.declarative.parameters import Parameter
from quchip.devices.spaces import FockSpace
from quchip.extensions import CollectiveDecayCoupling, LossyChargeDrive


def _transmon(label: str) -> DuffingTransmon:
    return DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3, label=label)


def test_collapse_channel_is_public_immutable_authored_data():
    channel = CollapseChannel(
        operator=np.eye(2),
        rate=0.01,
        name="loss",
    )

    assert quchip.CollapseChannel is CollapseChannel
    with pytest.raises(FrozenInstanceError):
        channel.rate = 0.02
    with pytest.raises(ValueError, match="rate"):
        CollapseChannel(np.eye(2), -0.01, "gain")


def test_collapse_channel_contains_only_physics_values():
    channel = CollapseChannel(operator=np.eye(2), rate=0.2, name="loss")

    assert channel.operator.shape == (2, 2)
    assert channel.rate == 0.2
    assert channel.name == "loss"
    assert not hasattr(channel, "parameters")


def test_collapse_dependency_paths_come_from_operator_and_rate():
    fields = {
        "operator_scale": Parameter(),
        "loss_rate": Parameter(),
    }
    op = LocalOps("q", FockSpace(2))
    p = ParameterNamespace("q", fields)

    assert collapse_parameter_paths(op.a * p.operator_scale, p.loss_rate) == (
        "q.operator_scale",
        "q.loss_rate",
    )


def test_lossy_charge_drive_owns_one_local_relaxation_channel():
    device = _transmon("q")
    drive = LossyChargeDrive(device, line_loss_rate=0.004, label="xy")

    (channel_with_paths,) = drive._collapse_channels_with_paths(device)
    channel, paths = channel_with_paths
    assert channel.name == "line_relaxation"
    assert paths == ("drive.xy.line_loss_rate",)
    assert np.asarray(channel.rate.matrix()).item() == pytest.approx(0.004)

    actual = np.asarray(channel.operator.matrix())
    expected = np.diag(np.sqrt(np.arange(1, 3)), k=1)
    np.testing.assert_allclose(actual, expected)


def test_lossy_charge_drive_serializes_its_rate():
    device = _transmon("q")
    original = LossyChargeDrive(device, line_loss_rate=0.004, label="xy")

    restored = BaseDrive.from_dict(original.to_dict(), target=device)

    assert isinstance(restored, LossyChargeDrive)
    assert restored.line_loss_rate == pytest.approx(0.004)
    assert restored.target_label == "q"


def test_collective_decay_coupling_serializes_with_its_chip():
    a = _transmon("a")
    b = _transmon("b")
    chip = Chip(
        [a, b],
        couplings=[
            CollectiveDecayCoupling(
                a,
                b,
                exchange_strength=0.02,
                decay_rate=0.003,
                label="shared_line",
            )
        ],
    )

    restored = Chip.from_dict(chip.to_dict()).coupling("shared_line")

    assert isinstance(restored, CollectiveDecayCoupling)
    assert restored.exchange_strength == pytest.approx(0.02)
    assert restored.decay_rate == pytest.approx(0.003)


def test_collective_decay_coupling_authors_correlated_jump():
    a = _transmon("a")
    b = _transmon("b")
    coupling = CollectiveDecayCoupling(
        a,
        b,
        exchange_strength=0.02,
        decay_rate=0.003,
        label="shared_line",
    )
    chip = Chip([a, b], couplings=[coupling])
    resolved = chip.coupling("shared_line")

    (channel,) = resolved.collapse_channels()
    assert channel.name == "collective_decay"
    assert np.asarray(channel.rate.matrix()).item() == pytest.approx(0.003)

    actual = np.asarray(channel.operator.matrix())
    lowering = np.diag(np.sqrt(np.arange(1, 3)), k=1)
    expected = np.kron(lowering, np.eye(3)) + np.kron(np.eye(3), lowering)
    np.testing.assert_allclose(actual, expected)


def test_drive_and_coupling_channels_keep_owner_and_support_in_engine_terms():
    a = _transmon("a")
    b = _transmon("b")
    drive = LossyChargeDrive(a, line_loss_rate=0.004, label="xy")
    coupling = CollectiveDecayCoupling(
        a,
        b,
        exchange_strength=0.02,
        decay_rate=0.003,
        label="shared_line",
    )
    chip = Chip([a, b], couplings=[coupling])
    chip.wire(drive)

    contributions = chip.collapse_contributions()
    drive_term = next(term for term in contributions if term[3] == "xy")
    coupling_term = next(term for term in contributions if term[3] == "shared_line")

    assert drive_term[2:] == ((0,), "xy", "line_relaxation", ("drive.xy.line_loss_rate",))
    assert coupling_term[2:] == (
        (0, 1),
        "shared_line",
        "collective_decay",
        ("shared_line.decay_rate",),
    )


def test_bath_returns_collapse_channels_without_a_private_tuple_shape():
    a = _transmon("a")
    b = _transmon("b")
    chip = Chip([a, b])
    bath = Bath("collective_decay", targets=[a, b], rate=0.006, label="common")
    chip.add_bath(bath)

    (channel,) = bath.collapse_channels(chip)

    assert isinstance(channel, CollapseChannel)
    assert channel.name == "collective_decay"
    assert np.asarray(channel.rate.matrix()).item() == pytest.approx(0.006)
    bath_term = next(term for term in chip.collapse_contributions() if term[3] == "common")
    assert bath_term[2:] == ((), "common", "collective_decay", ("bath.common.rate",))


def test_component_hooks_reject_raw_collapse_tuples():
    class RawTupleDrive(LossyChargeDrive):
        def dissipation(self, target, op, p):
            return ((op.a, p.line_loss_rate),)

    device = _transmon("q")
    drive = RawTupleDrive(device, line_loss_rate=0.004, label="raw")
    chip = Chip([device])
    chip.wire(drive)

    with pytest.raises(TypeError, match="CollapseChannel"):
        chip.resolve()


def test_dissipation_rates_remain_differentiable_through_engine_resolution():
    def total_rate(line_rate, collective_rate):
        a = _transmon("a")
        b = _transmon("b")
        drive = LossyChargeDrive(a, line_loss_rate=line_rate, label="xy")
        coupling = CollectiveDecayCoupling(
            a,
            b,
            exchange_strength=0.02,
            decay_rate=collective_rate,
            label="shared_line",
        )
        chip = Chip([a, b], couplings=[coupling])
        chip.wire(drive)
        result = chip.resolve()
        return sum(jnp.asarray(term.rate) for term in result.collapse_terms)

    gradients = jax.grad(total_rate, argnums=(0, 1))(0.004, 0.003)

    np.testing.assert_allclose(gradients, (1.0, 1.0), atol=1e-12)


def test_lossy_charge_drive_relaxes_one_excitation_at_its_authored_rate():
    device = _transmon("q")
    rate = 0.004
    drive = LossyChargeDrive(device, line_loss_rate=rate, label="xy")
    chip = Chip([device])
    chip.wire(drive)
    times = np.linspace(0.0, 100.0, 6)

    result = quchip.simulate(
        chip,
        [],
        times,
        initial_state=chip.bare_state({device: 1}),
        e_ops={device: device.number_operator()},
        check_truncation=False,
    )

    np.testing.assert_allclose(
        np.real(result.expect(device)),
        np.exp(-rate * times),
        rtol=2e-5,
        atol=2e-7,
    )


def test_collective_decay_bright_state_decays_at_twice_the_channel_rate():
    a = _transmon("a")
    b = _transmon("b")
    rate = 0.003
    coupling = CollectiveDecayCoupling(
        a,
        b,
        exchange_strength=0.02,
        decay_rate=rate,
        label="shared_line",
    )
    chip = Chip([a, b], couplings=[coupling])
    bright = (
        chip.bare_state({a: 1, b: 0}) + chip.bare_state({a: 0, b: 1})
    ) / np.sqrt(2.0)
    times = np.linspace(0.0, 80.0, 5)

    result = quchip.simulate(
        chip,
        [],
        times,
        initial_state=bright,
        e_ops={a: a.number_operator(), b: b.number_operator()},
        check_truncation=False,
    )
    excitations = np.real(result.expect(a) + result.expect(b))

    np.testing.assert_allclose(
        excitations,
        np.exp(-2.0 * rate * times),
        rtol=2e-5,
        atol=2e-7,
    )
