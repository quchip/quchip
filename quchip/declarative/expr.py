"""Backend-neutral operator algebra for the declarative physics DSL.

:class:`PhysicsExpr` is the expression tree :class:`~quchip.declarative.ops.LocalOps`
handles compose into; :func:`compile_expr` lowers a tree into a backend-native
operator once endpoint labels are resolved to concrete Hilbert spaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quchip.utils.jax_utils import maybe_concrete_scalar


@dataclass(frozen=True)
class DynamicScalar:
    """A time-dependent scalar payload attached to a :class:`PhysicsExpr`.

    ``source`` is the time-dependent object (typically an envelope or a
    :class:`~quchip.engine.ir.ScalarModulation`) that the engine compiles
    into a :class:`~quchip.engine.ir.DynamicTerm` when assembling the
    Hamiltonian. Multiplying a ``DynamicScalar`` by a ``PhysicsExpr``
    records the source on the resulting expression's ``dynamic_sources``
    tuple — the operator structure itself is unchanged.
    """

    source: Any

    def __mul__(self, other: Any) -> "PhysicsExpr":
        """Attach this dynamic source to ``other``."""
        return ensure_expr(other)._with_dynamic(self)

    def __rmul__(self, other: Any) -> "PhysicsExpr":
        """Attach this dynamic source when it appears on the right."""
        return ensure_expr(other)._with_dynamic(self)


def _is_scalar_like(value: Any) -> bool:
    """Return True if ``value`` should be treated as a scalar coefficient.

    Accepts Python numeric types, 0-d JAX arrays, JAX tracers (which expose
    ``ndim``), and 0-d numpy scalars. Rejects :class:`PhysicsExpr` and
    :class:`DynamicScalar` (they have their own multiplication semantics)
    and any array-shaped operand (``ndim > 0``) — silently swallowing those
    would hide a real type error.
    """
    if isinstance(value, (int, float, complex)):
        return True
    if isinstance(value, (PhysicsExpr, DynamicScalar)):
        return False
    ndim = getattr(value, "ndim", 0)
    return ndim == 0


@dataclass(frozen=True)
class PhysicsExpr:
    """Backend-neutral expression tree for local physics operators.

    Built by composing :class:`~quchip.declarative.ops.LocalOps` operator
    handles, not constructed directly. Operators on the *same* endpoint
    compose with ``@`` (matrix product); operators on *different* endpoints
    combine with ``*`` (tensor product). Scalars scale via ``*``, and an
    envelope or modulation multiplied in is recorded as a dynamic source
    rather than altering the operator structure.

    Examples
    --------
    >>> from quchip.declarative import LocalOps
    >>> a, b = LocalOps("q0", 3), LocalOps("q1", 3)
    >>> coupling = 0.01 * (a.a * b.adag + a.adag * b.a)
    >>> coupling.kind
    'add'
    >>> coupling.labels
    ('q0', 'q1')
    """

    kind: str
    args: tuple[Any, ...] = ()
    labels: tuple[str, ...] = ()
    scalar: Any = 1.0
    dynamic_sources: tuple[DynamicScalar, ...] = ()

    def _binary(self, other: Any, kind: str) -> "PhysicsExpr":
        """Combine with *other* under *kind*, validating same-support algebra for ``add``/``sub``.

        Addition and subtraction require both operands to be
        :class:`PhysicsExpr` with identical endpoint support: a bare scalar
        is backend-ambiguous (it could mean scalar-times-identity or an
        element-wise array add), and combining terms from different
        endpoints without a tensor product is not an interaction any
        coupling declares. Cross-endpoint terms combine with ``*`` instead.
        """
        if kind in ("add", "sub"):
            if not isinstance(other, PhysicsExpr):
                raise TypeError(
                    f"Cannot add or subtract {type(other).__name__!r} and PhysicsExpr directly; "
                    "write the identity explicitly, e.g. `op.n + 1.0 * op.I`."
                )
            rhs = other
            if self.labels != rhs.labels:
                raise TypeError(
                    "Addition and subtraction require operands with the same endpoint support "
                    f"(got {self.labels!r} and {rhs.labels!r}); combine cross-endpoint terms with `*`."
                )
        else:
            rhs = ensure_expr(other)
        return PhysicsExpr(
            kind=kind,
            args=(self, rhs),
            labels=tuple(dict.fromkeys(self.labels + rhs.labels)),
            dynamic_sources=self.dynamic_sources + rhs.dynamic_sources,
        )

    def _with_dynamic(self, dynamic: DynamicScalar) -> "PhysicsExpr":
        """Return a copy of this expression with *dynamic* appended to its dynamic sources."""
        return PhysicsExpr(
            kind=self.kind,
            args=self.args,
            labels=self.labels,
            scalar=self.scalar,
            dynamic_sources=self.dynamic_sources + (dynamic,),
        )

    def without_dynamic_sources(self) -> "PhysicsExpr":
        """Return the same operator expression with dynamic sources stripped."""
        return PhysicsExpr(
            kind=self.kind,
            args=tuple(arg.without_dynamic_sources() if isinstance(arg, PhysicsExpr) else arg for arg in self.args),
            labels=self.labels,
            scalar=self.scalar,
            dynamic_sources=(),
        )

    def __add__(self, other: Any) -> "PhysicsExpr":
        """Add, requiring both operands to share the same endpoint support."""
        return self._binary(other, "add")

    def __sub__(self, other: Any) -> "PhysicsExpr":
        """Subtract, requiring both operands to share the same endpoint support."""
        return self._binary(other, "sub")

    def __matmul__(self, other: Any) -> "PhysicsExpr":
        """Compose via matrix product, requiring both operands to act on the same endpoint."""
        rhs = ensure_expr(other)
        if self.labels != rhs.labels:
            raise TypeError("Cannot use @ for operators on different endpoints; use * for tensor product.")
        return self._binary(rhs, "matmul")

    def _scale(self, other: Any) -> "PhysicsExpr | None":
        """Scale by a scalar or attach a dynamic source; ``None`` if neither applies."""
        if isinstance(other, DynamicScalar):
            return self._with_dynamic(other)
        if _is_scalar_like(other):
            return PhysicsExpr(self.kind, self.args, self.labels, self.scalar * other, self.dynamic_sources)
        return None

    def __mul__(self, other: Any) -> "PhysicsExpr":
        """Scale by a scalar, attach a dynamic source, or tensor disjoint endpoints."""
        scaled = self._scale(other)
        if scaled is not None:
            return scaled
        rhs = ensure_expr(other)
        dynamic_sources = self.dynamic_sources + rhs.dynamic_sources
        if not self.labels or not rhs.labels:
            # One side carries no operator support of its own (e.g. a
            # DynamicScalar chained through a bare scalar via ensure_expr,
            # producing a labels=() "scalar" node). Merge as a scale of the
            # operator-bearing side rather than tensoring, so the dynamic
            # source rides the real operator instead of landing on a tensor
            # node with no backend operator on one factor.
            base, factor = (self, rhs) if self.labels else (rhs, self)
            return PhysicsExpr(
                kind=base.kind, args=base.args, labels=base.labels,
                scalar=base.scalar * factor.scalar, dynamic_sources=dynamic_sources,
            )
        if set(self.labels) & set(rhs.labels):
            raise TypeError(
                "Cannot use * for operators with overlapping endpoint support "
                f"({self.labels!r} and {rhs.labels!r}); operators on the same endpoint use @ for "
                "matrix composition, and tensor product requires disjoint label support."
            )
        return PhysicsExpr(
            kind="tensor",
            args=(self, rhs),
            labels=tuple(dict.fromkeys(self.labels + rhs.labels)),
            dynamic_sources=dynamic_sources,
        )

    def __rmul__(self, other: Any) -> "PhysicsExpr":
        """Handle scalar or dynamic-source multiplication from the left."""
        scaled = self._scale(other)
        if scaled is not None:
            return scaled
        # If both operands are PhysicsExpr, Python dispatches to __mul__ on the left and
        # never reflects to __rmul__. Anything else here is a type the user shouldn't be
        # multiplying by an operator — surface it as a clear error.
        raise TypeError(f"Cannot multiply {type(other).__name__} by PhysicsExpr.")

    def has_dynamic_source(self) -> bool:
        """Return whether this expression carries any time-dependent source."""
        return bool(self.dynamic_sources)


def ensure_expr(value: Any) -> PhysicsExpr:
    """Coerce a scalar-like value into :class:`PhysicsExpr`."""
    if isinstance(value, PhysicsExpr):
        return value
    if _is_scalar_like(value):
        return PhysicsExpr(kind="scalar", scalar=value)
    raise TypeError(f"Expected PhysicsExpr-compatible value, got {type(value).__name__}")


def compile_expr(expr: PhysicsExpr, op_lookup: dict[tuple[str, str], Any], backend: Any) -> Any:
    """Compile a :class:`PhysicsExpr` into a backend-native operator.

    ``op_lookup`` maps ``(label, operator_name)`` to a backend operator
    (e.g. ``a``, ``adag``, ``n``, ``I`` for each device label). ``backend``
    is the active default backend; only its ``tensor()`` method is used
    here. Scalar coefficients carried by every node are preserved
    consistently — including traced (non-concrete) scalars, which are
    multiplied unconditionally so gradients still flow.
    """
    if expr.kind == "scalar":
        return expr.scalar
    if expr.kind == "op":
        label = expr.labels[0]
        return expr.scalar * op_lookup[(label, expr.args[0])]
    if expr.kind == "add":
        result = compile_expr(expr.args[0], op_lookup, backend) + compile_expr(expr.args[1], op_lookup, backend)
    elif expr.kind == "sub":
        result = compile_expr(expr.args[0], op_lookup, backend) - compile_expr(expr.args[1], op_lookup, backend)
    elif expr.kind == "matmul":
        result = compile_expr(expr.args[0], op_lookup, backend) @ compile_expr(expr.args[1], op_lookup, backend)
    elif expr.kind == "tensor":
        result = backend.tensor(
            compile_expr(expr.args[0], op_lookup, backend),
            compile_expr(expr.args[1], op_lookup, backend),
        )
    else:
        raise TypeError(f"Unknown PhysicsExpr kind {expr.kind!r}")
    scalar_concrete = maybe_concrete_scalar(expr.scalar)
    if scalar_concrete is None or scalar_concrete != 1.0:
        result = expr.scalar * result
    return result
