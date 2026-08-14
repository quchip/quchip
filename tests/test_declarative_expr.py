from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from quchip.declarative.expr import PhysicsExpr, materialize_array
from quchip.declarative.ops import EndpointOps, LocalOps
from quchip.devices.spaces import FockSpace


def test_same_endpoint_matrix_composition_uses_matmul():
    """Composing same-endpoint operators with ``@`` produces a matmul-kind expression."""
    op = LocalOps(label="q", space=FockSpace(3))
    expr = op.n @ (op.n - op.I)
    assert isinstance(expr, PhysicsExpr)
    assert expr.kind == "matmul"


def test_same_endpoint_star_errors_with_clear_message():
    """Multiplying same-endpoint operators with ``*`` raises, naming ``@`` as the fix."""
    op = LocalOps(label="q", space=FockSpace(3))
    with pytest.raises(TypeError, match="same endpoint.*use @"):
        _ = op.n * op.I


def test_cross_endpoint_star_builds_tensor_product():
    """Multiplying cross-endpoint operators with ``*`` produces a tensor-kind expression preserving both labels."""
    a = EndpointOps(label="a", space=FockSpace(3))
    b = EndpointOps(label="b", space=FockSpace(4))
    expr = a.x * b.x
    assert expr.kind == "tensor"
    assert expr.labels == ("a", "b")


def test_cross_endpoint_matmul_errors_with_clear_message():
    """Composing cross-endpoint operators with ``@`` raises, naming ``*`` as the fix."""
    a = EndpointOps(label="a", space=FockSpace(3))
    b = EndpointOps(label="b", space=FockSpace(4))
    with pytest.raises(TypeError, match="different endpoints.*use \\*"):
        _ = a.x @ b.x


def test_traced_scalar_left_multiply_does_not_raise():
    """A 0-d scalar left-multiplying an operator scales it without raising."""
    op = LocalOps(label="q", space=FockSpace(3))
    omega = jnp.asarray(5.0)
    expr = omega * op.n
    assert isinstance(expr, PhysicsExpr)
    assert expr.kind == "scale"
    assert float(expr.args[0].args[0]) == 5.0


def test_traced_scalar_right_multiply_does_not_raise():
    """A 0-d scalar right-multiplying an operator scales it without raising."""
    op = LocalOps(label="q", space=FockSpace(3))
    omega = jnp.asarray(5.0)
    expr = op.n * omega
    assert isinstance(expr, PhysicsExpr)
    assert expr.kind == "scale"
    assert float(expr.args[0].args[0]) == 5.0


def test_traced_scalar_flows_through_jax_grad():
    """A scalar coefficient built from a traced input stays differentiable through ``jax.grad``."""
    op = LocalOps(label="q", space=FockSpace(3))

    def coefficient(omega):
        return (omega * op.n).args[0].args[0]

    grad_fn = jax.grad(coefficient)
    assert float(grad_fn(jnp.asarray(3.0))) == 1.0


def test_python_scalar_still_works():
    """Plain Python scalars scale an operator from either side."""
    op = LocalOps(label="q", space=FockSpace(3))
    expr = 2.0 * op.n
    assert expr.args[0].args[0] == 2.0
    assert (op.n * 3).args[0].args[0] == 3


def test_array_operand_rejected():
    """A non-scalar (``ndim > 0``) array operand is rejected with ``TypeError``."""
    op = LocalOps(label="q", space=FockSpace(3))
    arr = jnp.asarray([1.0, 2.0, 3.0])  # ndim == 1, not scalar
    with pytest.raises(TypeError):
        _ = arr * op.n


def test_scalar_addition_raises():
    """Adding a bare scalar to an operator raises, pointing at the explicit-identity fix."""
    op = LocalOps(label="q", space=FockSpace(3))
    with pytest.raises(TypeError, match="identity"):
        _ = op.n + 1.0


def test_mismatched_support_addition_raises():
    """Adding operators from different endpoints raises, naming ``*`` as the fix."""
    a = EndpointOps(label="a", space=FockSpace(3))
    b = EndpointOps(label="b", space=FockSpace(3))
    with pytest.raises(TypeError, match="same endpoint support"):
        _ = a.n + b.n


def test_overlapping_tensor_support_raises():
    """Tensoring an operator with an expression that already spans its endpoint raises."""
    a = EndpointOps(label="a", space=FockSpace(3))
    b = EndpointOps(label="b", space=FockSpace(3))
    with pytest.raises(TypeError, match="overlapping"):
        _ = a.n * (a.n * b.n)


def test_tensor_labels_preserve_authored_order_for_unequal_dimensions():
    """Tensor-product labels reflect authorship order, not endpoint dimension or sorting."""
    a = EndpointOps(label="a", space=FockSpace(3))
    b = EndpointOps(label="b", space=FockSpace(5))
    assert (a.n * b.n).labels == ("a", "b")
    assert (b.n * a.n).labels == ("b", "a")


def test_nonadjacent_two_body_embedding_preserves_subsystem_axes():
    """Embedding across a spectator matches the explicit tensor product on unequal spaces."""
    a = EndpointOps(label="a", space=FockSpace(2))
    c = EndpointOps(label="c", space=FockSpace(3))
    embedded = (a.a * c.adag).embed(("a", "b", "c"), (2, 4, 3))
    expected = jnp.kron(jnp.kron(materialize_array(a.a), jnp.eye(4)), materialize_array(c.adag))
    assert jnp.allclose(materialize_array(embedded), expected)


def test_opaque_function_displays_arguments_and_stays_differentiable():
    """A matrix function remains opaque in display while its bound values trace through JAX."""
    omega = PhysicsExpr.parameter(scope="q", name="freq", symbol=r"\omega")
    expr = PhysicsExpr.from_function(
        lambda value: jnp.diag(jnp.asarray([0.0, value], dtype=complex)),
        omega,
        labels=("q",),
        dims=(2,),
        name=r"\hat X",
    )

    class ArrayBackend:
        @staticmethod
        def from_array(value, *, dims):
            _ = dims
            return value

        @staticmethod
        def to_array(value):
            return value

    assert expr.latex() == r"\hat X\!\left(\omega_{q}\right)"
    def loss(value):
        return jnp.real(expr.matrix({"q.freq": value}, backend=ArrayBackend())[1, 1])

    assert jax.grad(loss)(jnp.asarray(5.0)) == pytest.approx(1.0)

    from quchip.declarative import DeviceModel, Scalar, parameter

    class OpaqueDevice(DeviceModel):
        freq: Scalar = parameter(symbol=r"\omega")

        def local_hamiltonian(self, op, p):
            _ = (op, p)
            return lambda freq: jnp.diag(jnp.asarray([0.0, freq], dtype=complex))

    device = OpaqueDevice(freq=5.0, levels=2, label="opaque")
    authored = device.unresolved_hamiltonian()
    assert authored.latex() == r"\hat H_{opaque}\!\left(\omega_{opaque}\right)"

    def device_loss(value):
        rebound = OpaqueDevice(freq=value, levels=2, label="opaque")
        return jnp.real(rebound.unresolved_hamiltonian().matrix(backend=ArrayBackend())[1, 1])

    assert jax.grad(device_loss)(jnp.asarray(5.0)) == pytest.approx(1.0)
