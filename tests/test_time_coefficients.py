"""Contracts for public intrinsic time coefficients."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import quchip
from quchip.declarative import (
    CosineCoefficient,
    Scalar,
    TimeDependentTerm,
    TimeCoefficient,
    parameter,
    qnp,
)
from quchip.engine.ir import (
    Carrier,
    SignalNode,
    _as_time_coefficient,
    evaluate_signal_program,
)


class QuadraticCoefficient(TimeCoefficient):
    scale: Scalar = parameter()

    def value(self, t: object) -> object:
        return self.scale * qnp.asarray(t) ** 2


def _walk_signal(node: SignalNode) -> tuple[SignalNode, ...]:
    return (node, *(descendant for child in node.signal_children() for descendant in _walk_signal(child)))


def test_cosine_coefficient_uses_ghz_ns_and_phase() -> None:
    coefficient = CosineCoefficient(amplitude=0.2, frequency=0.25, phase=np.pi / 3)
    times = np.asarray([0.0, 1.0, 2.0])
    expected = 0.2 * np.cos(2.0 * np.pi * 0.25 * times + np.pi / 3)

    np.testing.assert_allclose(np.asarray(coefficient.value(times)), expected, atol=1e-12)


def test_cosine_coefficient_is_jax_differentiable() -> None:
    def sample(amplitude: float) -> jax.Array:
        coefficient = CosineCoefficient(amplitude=amplitude, frequency=0.25)
        return coefficient.value(jnp.asarray(0.5))

    np.testing.assert_allclose(jax.grad(sample)(0.2), np.sqrt(0.5), atol=1e-7)


def test_time_dependent_types_are_public() -> None:
    assert quchip.TimeCoefficient is TimeCoefficient
    assert quchip.CosineCoefficient is CosineCoefficient
    assert quchip.TimeDependentTerm is TimeDependentTerm


def test_cosine_coefficient_round_trip() -> None:
    original = CosineCoefficient(amplitude=0.2, frequency=0.25, phase=0.4)

    restored = TimeCoefficient.from_dict(original.to_dict())

    assert isinstance(restored, CosineCoefficient)
    assert restored.amplitude == original.amplitude
    assert restored.frequency == original.frequency
    assert restored.phase == original.phase


def test_custom_coefficient_round_trip() -> None:
    restored = TimeCoefficient.from_dict(QuadraticCoefficient(scale=0.5).to_dict())

    assert isinstance(restored, QuadraticCoefficient)
    assert restored.scale == pytest.approx(0.5)


def test_cosine_coefficient_pytree_leaves_are_numerical_fields() -> None:
    leaves = jax.tree_util.tree_leaves(CosineCoefficient(0.2, 0.25, 0.4))

    assert leaves == [0.2, 0.25, 0.4]


def test_time_dependent_term_requires_time_coefficient() -> None:
    with pytest.raises(TypeError, match="TimeCoefficient"):
        TimeDependentTerm(operator=np.eye(2), coefficient=lambda time: time)


def test_custom_coefficient_lowers_without_engine_extension() -> None:
    coefficient = QuadraticCoefficient(scale=0.5)
    signal = _as_time_coefficient(coefficient, owner="fixture").signal

    np.testing.assert_allclose(np.asarray(coefficient.value([0.0, 2.0])), [0.0, 2.0])
    np.testing.assert_allclose(
        evaluate_signal_program(signal, np.asarray([0.0, 2.0]), xp=np),
        [0.0, 2.0],
    )


def test_cosine_coefficient_exposes_analytic_carrier_to_engine() -> None:
    signal = _as_time_coefficient(
        CosineCoefficient(0.2, 0.25, 0.3), owner="fixture"
    ).signal

    assert any(isinstance(node, Carrier) for node in _walk_signal(signal))
