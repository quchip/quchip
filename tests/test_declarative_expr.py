from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from quchip.declarative.expr import DynamicScalar, PhysicsExpr
from quchip.declarative.ops import EndpointOps, LocalOps


def test_same_endpoint_matrix_composition_uses_matmul():
    """Composing same-endpoint operators with ``@`` produces a matmul-kind expression."""
    op = LocalOps(label="q", levels=3)
    expr = op.n @ (op.n - op.I)
    assert isinstance(expr, PhysicsExpr)
    assert expr.kind == "matmul"


def test_same_endpoint_star_errors_with_clear_message():
    """Multiplying same-endpoint operators with ``*`` raises, naming ``@`` as the fix."""
    op = LocalOps(label="q", levels=3)
    with pytest.raises(TypeError, match="same endpoint.*use @"):
        _ = op.n * op.I


def test_cross_endpoint_star_builds_tensor_product():
    """Multiplying cross-endpoint operators with ``*`` produces a tensor-kind expression preserving both labels."""
    a = EndpointOps(label="a", levels=3)
    b = EndpointOps(label="b", levels=4)
    expr = a.x * b.x
    assert expr.kind == "tensor"
    assert expr.labels == ("a", "b")


def test_cross_endpoint_matmul_errors_with_clear_message():
    """Composing cross-endpoint operators with ``@`` raises, naming ``*`` as the fix."""
    a = EndpointOps(label="a", levels=3)
    b = EndpointOps(label="b", levels=4)
    with pytest.raises(TypeError, match="different endpoints.*use \\*"):
        _ = a.x @ b.x


def test_dynamic_source_detection():
    """A ``DynamicScalar``-scaled expression reports a dynamic source."""
    op = LocalOps(label="q", levels=3)
    dynamic = DynamicScalar("env")
    expr = dynamic * op.n
    assert expr.has_dynamic_source()


def test_traced_scalar_left_multiply_does_not_raise():
    """A 0-d scalar left-multiplying an operator scales it without raising."""
    op = LocalOps(label="q", levels=3)
    omega = jnp.asarray(5.0)
    expr = omega * op.n
    assert isinstance(expr, PhysicsExpr)
    assert expr.kind == "op"
    assert float(expr.scalar) == 5.0


def test_traced_scalar_right_multiply_does_not_raise():
    """A 0-d scalar right-multiplying an operator scales it without raising."""
    op = LocalOps(label="q", levels=3)
    omega = jnp.asarray(5.0)
    expr = op.n * omega
    assert isinstance(expr, PhysicsExpr)
    assert float(expr.scalar) == 5.0


def test_traced_scalar_flows_through_jax_grad():
    """A scalar coefficient built from a traced input stays differentiable through ``jax.grad``."""
    op = LocalOps(label="q", levels=3)

    def coefficient(omega):
        return (omega * op.n).scalar

    grad_fn = jax.grad(coefficient)
    assert float(grad_fn(jnp.asarray(3.0))) == 1.0


def test_python_scalar_still_works():
    """Plain Python scalars scale an operator from either side."""
    op = LocalOps(label="q", levels=3)
    expr = 2.0 * op.n
    assert expr.scalar == 2.0
    assert (op.n * 3).scalar == 3


def test_array_operand_rejected():
    """A non-scalar (``ndim > 0``) array operand is rejected with ``TypeError``."""
    op = LocalOps(label="q", levels=3)
    arr = jnp.asarray([1.0, 2.0, 3.0])  # ndim == 1, not scalar
    with pytest.raises(TypeError):
        _ = arr * op.n


def test_dynamic_sources_accumulate_across_binary_ops():
    """Dynamic sources from both operands accumulate, in order, across ``+``."""
    op = LocalOps(label="q", levels=3)
    d1 = DynamicScalar("env1")
    d2 = DynamicScalar("env2")
    expr = d1 * op.n + d2 * op.n
    assert expr.dynamic_sources == (d1, d2)


def test_scalar_addition_raises():
    """Adding a bare scalar to an operator raises, pointing at the explicit-identity fix."""
    op = LocalOps(label="q", levels=3)
    with pytest.raises(TypeError, match="identity"):
        _ = op.n + 1.0


def test_mismatched_support_addition_raises():
    """Adding operators from different endpoints raises, naming ``*`` as the fix."""
    a = EndpointOps(label="a", levels=3)
    b = EndpointOps(label="b", levels=3)
    with pytest.raises(TypeError, match="same endpoint support"):
        _ = a.n + b.n


def test_overlapping_tensor_support_raises():
    """Tensoring an operator with an expression that already spans its endpoint raises."""
    a = EndpointOps(label="a", levels=3)
    b = EndpointOps(label="b", levels=3)
    with pytest.raises(TypeError, match="overlapping"):
        _ = a.n * (a.n * b.n)


def test_dynamic_scalar_times_scalar_times_operator_scales_not_tensors():
    """A dynamic-scalar-times-scalar chain multiplying an operator scales it rather than tensoring it."""
    op = LocalOps(label="q", levels=3)
    dynamic = DynamicScalar("env")
    expr = (dynamic * 2.0) * op.n
    assert expr.kind == "op"
    assert expr.labels == ("q",)
    assert expr.scalar == 2.0
    assert expr.dynamic_sources == (dynamic,)


def test_tensor_labels_preserve_authored_order_for_unequal_dimensions():
    """Tensor-product labels reflect authorship order, not endpoint dimension or sorting."""
    a = EndpointOps(label="a", levels=3)
    b = EndpointOps(label="b", levels=5)
    assert (a.n * b.n).labels == ("a", "b")
    assert (b.n * a.n).labels == ("b", "a")
