"""One backend-neutral expression tree for authored scalar and operator physics."""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from math import prod
from typing import Any, Mapping

import jax.numpy as jnp


class UnboundParameterError(ValueError):
    """Numerical materialization was requested without every required value."""


def is_opaque_callable(value: Any) -> bool:
    """Return whether *value* is a callable authoring function, not a matrix-like object."""
    return callable(value) and getattr(value, "shape", None) is None


@dataclass(frozen=True)
class DynamicScalar:
    """A time-dependent scalar payload attached to an operator expression."""

    source: Any

    def __mul__(self, other: Any) -> "PhysicsExpr":
        return ensure_expr(other)._with_dynamic(self)

    def __rmul__(self, other: Any) -> "PhysicsExpr":
        return ensure_expr(other)._with_dynamic(self)


@dataclass(frozen=True)
class PhysicsExpr:
    """Authored scalar and operator algebra, independent of numerical values."""

    kind: str
    args: tuple[Any, ...] = ()
    labels: tuple[str, ...] = ()
    dynamic_sources: tuple[DynamicScalar, ...] = ()
    _bindings: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def parameter(
        cls,
        *,
        scope: str,
        name: str,
        symbol: str | None = None,
        unit: str | None = None,
    ) -> "PhysicsExpr":
        """Create a symbolic declared-parameter leaf."""
        return cls("parameter", (f"{scope}.{name}", symbol or name, unit))

    @classmethod
    def literal(cls, value: Any) -> "PhysicsExpr":
        """Create a literal scalar leaf."""
        return cls("literal", (value,))

    @classmethod
    def from_matrix(
        cls,
        value: Any,
        *,
        labels: tuple[str, ...],
        dims: tuple[int, ...],
        name: str | None = None,
    ) -> "PhysicsExpr":
        """Create a named backend-neutral matrix contribution."""
        if len(labels) != len(dims):
            raise ValueError("Matrix labels and dimensions must have the same length.")
        return cls("matrix", (value, tuple(dims), name), tuple(labels))

    @classmethod
    def from_function(
        cls,
        function: Any,
        *arguments: Any,
        labels: tuple[str, ...],
        dims: tuple[int, ...],
        name: str | None = None,
    ) -> "PhysicsExpr":
        """Create an opaque matrix-valued contribution from a pure function.

        The function runs only during numerical materialization. Display keeps
        its declared name and arguments, such as ``X(a, b)``, without exposing
        the implementation as symbolic algebra.
        """
        if len(labels) != len(dims):
            raise ValueError("Function labels and dimensions must have the same length.")
        if not callable(function):
            raise TypeError("function must be callable.")
        display_name = name or getattr(function, "__name__", None)
        if not display_name or display_name == "<lambda>":
            raise ValueError("Anonymous functions require a symbolic name.")
        return cls(
            "function",
            (function, tuple(dims), display_name, *(ensure_expr(arg) for arg in arguments)),
            tuple(labels),
        )

    @classmethod
    def from_state(
        cls,
        value: Any,
        *,
        labels: tuple[str, ...],
        dims: tuple[int, ...],
        name: str,
    ) -> "PhysicsExpr":
        """Create a named authored ket contribution."""
        return cls("state", (value, tuple(dims), name), tuple(labels))

    @classmethod
    def from_state_function(
        cls,
        function: Any,
        *arguments: Any,
        labels: tuple[str, ...],
        dims: tuple[int, ...],
        name: str,
    ) -> "PhysicsExpr":
        """Create an opaque callable ket contribution."""
        if not callable(function):
            raise TypeError("function must be callable.")
        return cls(
            "state_function",
            (function, tuple(dims), name, *(ensure_expr(arg) for arg in arguments)),
            tuple(labels),
        )

    @classmethod
    def from_signal(cls, signal: Any, *, name: str = "f") -> "PhysicsExpr":
        """Create a scalar time-function leaf backed by an engine signal."""
        return cls("signal", (signal, name))

    def embed(self, labels: tuple[str, ...], dims: tuple[int, ...]) -> "PhysicsExpr":
        """Embed this local contribution into an ordered composite Hilbert space."""
        if len(labels) != len(dims):
            raise ValueError("Composite labels and dimensions must have the same length.")
        missing = set(self.labels) - set(labels)
        if missing:
            raise ValueError(f"Cannot embed labels absent from the composite space: {sorted(missing)}")
        return PhysicsExpr("embed", (self, tuple(labels), tuple(dims)), tuple(labels))

    def with_bindings(self, bindings: Mapping[str, Any]) -> "PhysicsExpr":
        """Attach default values used only by direct numerical inspection."""
        return replace(self, _bindings=dict(bindings))

    def parameter_paths(self) -> tuple[str, ...]:
        """Return referenced dotted parameter paths in authored order."""
        return tuple(dict.fromkeys(
            node.args[0] for node in _walk_expr(self) if node.kind == "parameter"
        ))

    def numeric_values(self) -> tuple[Any, ...]:
        """Return bound and matrix payloads for tracer-safe cache decisions."""
        values: list[Any] = []
        for node in _walk_expr(self):
            values.extend(node._bindings.values())
            if node.kind in ("literal", "matrix", "state", "signal"):
                values.append(node.args[0])
        return tuple(values)

    @property
    def shape(self) -> tuple[int, int]:
        """Matrix shape implied by this operator expression's static support."""
        if not self.labels:
            raise AttributeError("Scalar expressions do not have a matrix shape.")
        if self.kind == "op":
            dimension = self.args[1].dimension
            return (dimension, dimension)
        if self.kind in ("matrix", "function"):
            dimension = prod(self.args[1])
            return (dimension, dimension)
        if self.kind in ("state", "state_function"):
            return (prod(self.args[1]), 1)
        if self.kind == "embed":
            dimension = prod(self.args[2])
            return (dimension, dimension)
        if self.kind == "tensor":
            dimension = self.args[0].shape[0] * self.args[1].shape[0]
            return (dimension, dimension)
        for arg in self.args:
            if isinstance(arg, PhysicsExpr) and arg.labels:
                return arg.shape
        raise AttributeError("Operator shape is not available for this expression.")

    def _binary(self, other: Any, kind: str) -> "PhysicsExpr":
        rhs = ensure_expr(other)
        if kind in ("add", "sub"):
            if bool(self.labels) != bool(rhs.labels):
                raise TypeError(
                    "Cannot add a scalar directly to an operator; write the identity explicitly."
                )
            if self.labels != rhs.labels:
                raise TypeError(
                    "Addition and subtraction require operands with the same endpoint support "
                    f"(got {self.labels!r} and {rhs.labels!r})."
                )
        return PhysicsExpr(
            kind,
            (self, rhs),
            tuple(dict.fromkeys(self.labels + rhs.labels)),
            self.dynamic_sources + rhs.dynamic_sources,
        )

    def _with_dynamic(self, dynamic: DynamicScalar) -> "PhysicsExpr":
        return replace(self, dynamic_sources=self.dynamic_sources + (dynamic,))

    def without_dynamic_sources(self) -> "PhysicsExpr":
        """Return the same expression with time-dependent sources removed."""
        return replace(
            self,
            args=tuple(
                arg.without_dynamic_sources() if isinstance(arg, PhysicsExpr) else arg
                for arg in self.args
            ),
            dynamic_sources=(),
        )

    def __add__(self, other: Any) -> "PhysicsExpr":
        return self._binary(other, "add")

    def __radd__(self, other: Any) -> "PhysicsExpr":
        return ensure_expr(other)._binary(self, "add")

    def __sub__(self, other: Any) -> "PhysicsExpr":
        return self._binary(other, "sub")

    def __rsub__(self, other: Any) -> "PhysicsExpr":
        return ensure_expr(other)._binary(self, "sub")

    def __matmul__(self, other: Any) -> "PhysicsExpr":
        rhs = ensure_expr(other)
        if not self.labels or self.labels != rhs.labels:
            raise TypeError("Cannot use @ for operators on different endpoints; use * across endpoints.")
        return self._binary(rhs, "matmul")

    def __mul__(self, other: Any) -> "PhysicsExpr":
        if isinstance(other, DynamicScalar):
            return self._with_dynamic(other)
        rhs = ensure_expr(other)
        dynamic = self.dynamic_sources + rhs.dynamic_sources
        if self.labels and rhs.labels:
            if set(self.labels) & set(rhs.labels):
                raise TypeError(
                    "Cannot use * for operators on the same endpoint or with overlapping "
                    "endpoint support; use @ on one endpoint."
                )
            return PhysicsExpr(
                "tensor",
                (self, rhs),
                tuple(dict.fromkeys(self.labels + rhs.labels)),
                dynamic,
            )
        if self.labels or rhs.labels:
            scalar, operator = (rhs, self) if self.labels else (self, rhs)
            return PhysicsExpr("scale", (scalar, operator), operator.labels, dynamic)
        return PhysicsExpr("mul", (self, rhs), (), dynamic)

    def __rmul__(self, other: Any) -> "PhysicsExpr":
        if isinstance(other, DynamicScalar):
            return self._with_dynamic(other)
        return ensure_expr(other).__mul__(self)

    def __truediv__(self, other: Any) -> "PhysicsExpr":
        rhs = ensure_expr(other)
        if rhs.labels:
            raise TypeError("Division by an operator is not defined.")
        return self * PhysicsExpr("pow", (rhs, PhysicsExpr.literal(-1)))

    def __pow__(self, other: Any) -> "PhysicsExpr":
        rhs = ensure_expr(other)
        if self.labels or rhs.labels:
            raise TypeError("Use @ for operator powers; ** is scalar-only.")
        return PhysicsExpr("pow", (self, rhs))

    def __neg__(self) -> "PhysicsExpr":
        return -1 * self

    def has_dynamic_source(self) -> bool:
        return bool(self.dynamic_sources)

    def latex(self) -> str:
        """Render the authored expression with familiar mathematical notation."""
        return _latex(self)

    def _repr_latex_(self) -> str:
        return f"${self.latex()}$"

    def __str__(self) -> str:
        return self.latex()

    def matrix(
        self,
        bindings: Mapping[str, Any] | None = None,
        *,
        t: Any | None = None,
        backend: Any = None,
    ) -> Any:
        """Materialize this expression and return its dense numerical array."""
        if backend is None:
            from quchip.backend import get_default_backend

            backend = get_default_backend()
        native = materialize_expr(
            self,
            backend,
            bindings=bindings,
            t=t,
        )
        return backend.to_array(native)


class ParameterNamespace:
    """Attribute view exposing one owner's declared fields as symbolic leaves."""

    __slots__ = ("_scope", "_fields")

    def __init__(self, scope: str, fields: Mapping[str, Any]) -> None:
        self._scope = scope
        self._fields = fields

    def __getattr__(self, name: str) -> PhysicsExpr:
        try:
            spec = self._fields[name]
        except KeyError as exc:
            raise AttributeError(f"No declared parameter {name!r} on {self._scope!r}.") from exc
        return PhysicsExpr.parameter(
            scope=self._scope,
            name=name,
            symbol=spec.symbol,
            unit=spec.unit,
        )

    def __dir__(self) -> list[str]:
        return sorted(self._fields)


def ensure_expr(value: Any) -> PhysicsExpr:
    """Coerce a scalar value into the shared expression tree."""
    if isinstance(value, PhysicsExpr):
        return value
    if isinstance(value, DynamicScalar):
        return PhysicsExpr.literal(1)._with_dynamic(value)
    if isinstance(value, (int, float, complex)) or getattr(value, "ndim", None) == 0:
        return PhysicsExpr.literal(value)
    raise TypeError(f"Expected a scalar or PhysicsExpr, got {type(value).__name__}.")


def as_operator_expr(
    value: Any,
    *,
    labels: tuple[str, ...],
    dims: tuple[int, ...],
    name: str,
    arguments: tuple[Any, ...] = (),
    owner: Any | None = None,
    scope: str | None = None,
    allowed: Mapping[str, Any] | None = None,
) -> PhysicsExpr:
    """Normalize symbolic, matrix, or opaque callable operator authorship."""
    expected = (prod(dims), prod(dims))
    if isinstance(value, PhysicsExpr):
        if not value.labels:
            raise TypeError("An operator expression must carry at least one subsystem label.")
        if value.labels != labels:
            raise TypeError(
                f"Operator expression support {value.labels!r} does not match {labels!r}."
            )
        if value.shape != expected:
            raise ValueError(
                f"Operator expression has shape {value.shape}, expected {expected}."
            )
        return value
    if is_opaque_callable(value):
        arguments, bindings = _resolve_callable_arguments(
            value, arguments=arguments, owner=owner, scope=scope, allowed=allowed
        )
        return PhysicsExpr.from_function(
            value,
            *arguments,
            labels=labels,
            dims=dims,
            name=name,
        ).with_bindings(bindings)
    if getattr(value, "shape", None) != expected:
        raise TypeError(
            f"Operator must be a PhysicsExpr, callable, or matrix with shape {expected}; "
            f"got {type(value).__name__} with shape {getattr(value, 'shape', None)}."
        )
    return PhysicsExpr.from_matrix(value, labels=labels, dims=dims, name=name)


def as_scalar_expr(
    value: Any,
    *,
    name: str,
    arguments: tuple[Any, ...] = (),
    owner: Any | None = None,
    scope: str | None = None,
    allowed: Mapping[str, Any] | None = None,
) -> PhysicsExpr:
    """Normalize symbolic, numeric, or opaque callable scalar authorship."""
    if isinstance(value, PhysicsExpr):
        if value.labels:
            raise TypeError("A scalar expression cannot carry subsystem labels.")
        return value
    if is_opaque_callable(value):
        arguments, bindings = _resolve_callable_arguments(
            value, arguments=arguments, owner=owner, scope=scope, allowed=allowed
        )
        return PhysicsExpr.from_function(
            value,
            *arguments,
            labels=(),
            dims=(),
            name=name,
        ).with_bindings(bindings)
    return ensure_expr(value)


def as_state_expr(
    value: Any,
    *,
    labels: tuple[str, ...],
    dims: tuple[int, ...],
    name: str,
    arguments: tuple[Any, ...] = (),
    owner: Any | None = None,
    scope: str | None = None,
    allowed: Mapping[str, Any] | None = None,
) -> PhysicsExpr:
    """Normalize an authored ket array or opaque callable without evaluating it."""
    expected = (prod(dims), 1)
    if isinstance(value, PhysicsExpr):
        if value.labels != labels:
            raise ValueError(
                f"State expression support {value.labels!r} does not match {labels!r}."
            )
        if value.shape != expected:
            raise ValueError(
                f"State expression has shape {value.shape}, expected {expected}."
            )
        return value
    if is_opaque_callable(value):
        arguments, bindings = _resolve_callable_arguments(
            value, arguments=arguments, owner=owner, scope=scope, allowed=allowed
        )
        return PhysicsExpr.from_state_function(
            value,
            *arguments,
            labels=labels,
            dims=dims,
            name=name,
        ).with_bindings(bindings)
    shape = getattr(value, "shape", None)
    if shape not in (expected, (expected[0],)):
        raise TypeError(
            f"State must be a PhysicsExpr, callable, or ket with shape {expected} or {(expected[0],)}; "
            f"got {type(value).__name__} with shape {shape}."
        )
    return PhysicsExpr.from_state(value, labels=labels, dims=dims, name=name)


def _resolve_callable_arguments(
    function: Any,
    *,
    arguments: tuple[Any, ...],
    owner: Any | None,
    scope: str | None,
    allowed: Mapping[str, Any] | None = None,
) -> tuple[tuple[PhysicsExpr, ...], dict[str, Any]]:
    """Resolve explicit callable arguments or bind their names to an owner."""
    if not is_opaque_callable(function):
        return (), {}
    if arguments:
        if owner is not None:
            raise TypeError("Pass explicit callable arguments or an owner, not both.")
        return tuple(ensure_expr(argument) for argument in arguments), {}
    if owner is None or scope is None:
        if inspect.signature(function).parameters:
            raise TypeError("Opaque functions with parameters require an owner and scope.")
        return (), {}
    fields = getattr(type(owner), "__quchip_param_fields__", {})
    resolved_arguments: list[PhysicsExpr] = []
    bindings: dict[str, Any] = {}
    for item in inspect.signature(function).parameters.values():
        if item.kind not in (item.POSITIONAL_ONLY, item.POSITIONAL_OR_KEYWORD):
            raise TypeError("Opaque functions may only declare positional parameters.")
        if allowed is not None and item.name not in allowed:
            raise ValueError(
                f"Opaque function argument {item.name!r} is not declared; "
                f"available fields are {sorted(allowed)}."
            )
        if not hasattr(owner, item.name):
            raise ValueError(f"Opaque function argument {item.name!r} is not a field on {scope!r}.")
        spec = fields.get(item.name)
        path = f"{scope}.{item.name}"
        resolved_arguments.append(
            PhysicsExpr.parameter(
                scope=scope,
                name=item.name,
                symbol=getattr(spec, "symbol", None) or item.name,
                unit=getattr(spec, "unit", None),
            )
        )
        bindings[path] = getattr(owner, item.name)
    return tuple(resolved_arguments), bindings


def _walk_expr(expr: PhysicsExpr) -> Iterator[PhysicsExpr]:
    """Yield an expression tree in authored preorder."""
    yield expr
    for arg in expr.args:
        if isinstance(arg, PhysicsExpr):
            yield from _walk_expr(arg)


def materialize_expr(
    expr: Any,
    backend: Any,
    *,
    bindings: Mapping[str, Any] | None = None,
    t: Any | None = None,
) -> Any:
    """Lower symbolic physics, passing an already-native contribution through."""
    if not isinstance(expr, PhysicsExpr):
        return expr
    values: dict[str, Any] = {}
    for node in _walk_expr(expr):
        values.update(node._bindings)
    if bindings is not None:
        values.update(bindings)
    missing = [path for path in expr.parameter_paths() if path not in values]
    if missing:
        raise UnboundParameterError("Missing numerical bindings: " + ", ".join(missing))

    def lower(node: PhysicsExpr) -> Any:
        if node.kind == "literal":
            return node.args[0]
        if node.kind == "parameter":
            return values[node.args[0]]
        if node.kind == "signal":
            if t is None:
                raise ValueError("t is required to materialize a time-dependent expression.")
            from quchip.engine.ir import evaluate_signal_program

            return evaluate_signal_program(node.args[0], t, xp=backend.array_module)
        if node.kind == "matrix":
            value, dims, _name = node.args
            return backend.from_array(
                backend.to_array(value),
                dims=[list(dims), list(dims)],
            )
        if node.kind == "state":
            value, dims, _name = node.args
            return backend.from_array(
                backend.to_array(value),
                dims=[list(dims), [1]],
            )
        if node.kind == "function":
            function, dims, _name, *arguments = node.args
            value = function(*(lower(argument) for argument in arguments))
            if not dims:
                return value
            return backend.from_array(value, dims=[list(dims), list(dims)])
        if node.kind == "state_function":
            function, dims, _name, *arguments = node.args
            value = function(*(lower(argument) for argument in arguments))
            return backend.from_array(value, dims=[list(dims), [1]])
        if node.kind == "op":
            name, space = node.args
            return space.operator(name, backend)
        if node.kind == "embed":
            local = lower(node.args[0])
            labels, dims = node.args[1:]
            support = tuple(labels.index(label) for label in node.args[0].labels)
            if len(support) == 1:
                return backend.embed(local, support[0], dims)
            if len(support) == 2:
                return backend.embed_two_body(local, support[0], support[1], dims)
            if support == tuple(range(len(dims))):
                return local
            raise ValueError(f"Cannot embed a contribution with support {support}.")
        left = lower(node.args[0])
        right = lower(node.args[1])
        if node.kind == "add":
            return left + right
        if node.kind == "sub":
            return left - right
        if node.kind == "matmul":
            return backend.matmul(left, right)
        if node.kind == "tensor":
            return backend.tensor(left, right)
        if node.kind in ("scale", "mul"):
            return left * right
        if node.kind == "pow":
            return left ** right
        raise TypeError(f"Unknown PhysicsExpr kind {node.kind!r}.")

    return lower(expr)


class _ArrayLowerer:
    """Minimal operator algebra for backend-independent JAX materialization."""

    array_module = jnp

    @staticmethod
    def from_array(value: Any, dims: Any = None) -> Any:
        del dims
        return jnp.asarray(value, dtype=jnp.complex128)

    @staticmethod
    def to_array(value: Any) -> Any:
        if hasattr(value, "to_jax"):
            value = value.to_jax()
        elif hasattr(value, "full"):
            value = value.full()
        return jnp.asarray(value)

    @staticmethod
    def destroy(dimension: int) -> Any:
        return jnp.diag(jnp.sqrt(jnp.arange(1, dimension)), 1).astype(jnp.complex128)

    @staticmethod
    def create(dimension: int) -> Any:
        return _ARRAY_LOWERER.destroy(dimension).conj().T

    @staticmethod
    def number(dimension: int) -> Any:
        return jnp.diag(jnp.arange(dimension, dtype=jnp.complex128))

    @staticmethod
    def identity(dimension: int) -> Any:
        return jnp.eye(dimension, dtype=jnp.complex128)

    @staticmethod
    def basis(dimension: int, index: int) -> Any:
        return jnp.zeros(dimension, dtype=jnp.complex128).at[index].set(1)

    @staticmethod
    def dag(value: Any) -> Any:
        return jnp.asarray(value).conj().T

    @staticmethod
    def matmul(left: Any, right: Any) -> Any:
        return left @ right

    @staticmethod
    def tensor(left: Any, right: Any) -> Any:
        return jnp.kron(left, right)

    @staticmethod
    def embed(local: Any, target: int, dims: tuple[int, ...]) -> Any:
        factors = [jnp.eye(dim, dtype=jnp.complex128) for dim in dims]
        factors[target] = local
        result = factors[0]
        for factor in factors[1:]:
            result = jnp.kron(result, factor)
        return result

    @staticmethod
    def embed_two_body(local: Any, first: int, second: int, dims: tuple[int, ...]) -> Any:
        if first > second:
            first, second = second, first
            local = jnp.asarray(local).reshape(
                dims[second], dims[first], dims[second], dims[first]
            ).transpose(1, 0, 3, 2).reshape(
                dims[first] * dims[second], dims[first] * dims[second]
            )
        order = [first, second] + [index for index in range(len(dims)) if index not in (first, second)]
        ordered_dims = [dims[index] for index in order]
        result = jnp.asarray(local)
        for dimension in ordered_dims[2:]:
            result = jnp.kron(result, jnp.eye(dimension, dtype=jnp.complex128))
        inverse = [order.index(index) for index in range(len(dims))]
        axes = inverse + [len(dims) + index for index in inverse]
        return result.reshape(*(ordered_dims + ordered_dims)).transpose(*axes).reshape(prod(dims), prod(dims))


_ARRAY_LOWERER = _ArrayLowerer()


def materialize_array(
    expr: Any,
    *,
    bindings: Mapping[str, Any] | None = None,
    t: Any | None = None,
) -> Any:
    """Materialize authored physics as a backend-independent JAX array."""
    return _ARRAY_LOWERER.to_array(
        materialize_expr(expr, _ARRAY_LOWERER, bindings=bindings, t=t)
    )


def filter_expr_bands(expr: PhysicsExpr, keeps_band: Any) -> PhysicsExpr | None:
    """Keep additive operator terms whose excitation-change band is accepted."""
    kept: list[tuple[int, PhysicsExpr]] = []
    for sign, term in _expanded_terms(expr):
        weights = _band_weights(term)
        if keeps_band(*(weights.get(label, 0) for label in expr.labels)):
            kept.append((sign, term))
    if not kept:
        return None
    result = kept[0][1] if kept[0][0] > 0 else -kept[0][1]
    for sign, term in kept[1:]:
        result = result + term if sign > 0 else result - term
    return result.with_bindings(expr._bindings)


def _expanded_terms(expr: PhysicsExpr) -> list[tuple[int, PhysicsExpr]]:
    """Expand additive children just enough to expose independently filterable bands."""
    if expr.kind == "add":
        return _expanded_terms(expr.args[0]) + _expanded_terms(expr.args[1])
    if expr.kind == "sub":
        return _expanded_terms(expr.args[0]) + [(-sign, term) for sign, term in _expanded_terms(expr.args[1])]
    if expr.kind in ("scale", "mul", "matmul", "tensor"):
        left_terms = _expanded_terms(expr.args[0])
        right_terms = _expanded_terms(expr.args[1])
        terms: list[tuple[int, PhysicsExpr]] = []
        for left_sign, left in left_terms:
            for right_sign, right in right_terms:
                combined = left @ right if expr.kind == "matmul" else left * right
                terms.append((left_sign * right_sign, combined))
        return terms
    return [(1, expr)]


def _band_weights(expr: PhysicsExpr) -> dict[str, int]:
    """Return one excitation-change weight per endpoint for a monomial."""
    if expr.kind in ("literal", "parameter", "signal"):
        return {}
    if expr.kind == "op":
        name, _space = expr.args
        try:
            weight = _OPERATOR_BAND_WEIGHTS[name]
        except KeyError as exc:
            raise TypeError(f"Operator {name!r} does not have one excitation-change band.") from exc
        return {expr.labels[0]: weight}
    if expr.kind in ("matrix", "function", "embed"):
        raise TypeError(f"{expr.kind.capitalize()} contributions do not expose symbolic excitation bands.")
    if expr.kind == "pow":
        raise TypeError("Scalar powers do not define operator excitation bands.")
    if expr.kind in ("add", "sub"):
        raise TypeError("Additive expressions must be expanded before band inspection.")
    weights: dict[str, int] = {}
    for child in expr.args[:2]:
        for label, weight in _band_weights(child).items():
            weights[label] = weights.get(label, 0) + weight
    return weights


_OPERATOR_BAND_WEIGHTS = {
    "a": -1,
    "adag": 1,
    "n": 0,
    "I": 0,
    "sigma_plus": 1,
    "sigma_minus": -1,
    "sigma_z": 0,
}


def _latex(expr: PhysicsExpr, parent_precedence: int = 0) -> str:
    if expr.kind == "literal":
        value = expr.args[0]
        return f"{value:g}" if isinstance(value, (int, float, complex)) else str(value)
    if expr.kind == "parameter":
        path, symbol, _unit = expr.args
        scope = path.rsplit(".", 1)[0]
        return _scoped_symbol(symbol, scope)
    if expr.kind == "signal":
        _signal, name = expr.args
        return rf"{name}\!\left(t\right)"
    if expr.kind == "matrix":
        _value, _dims, name = expr.args
        if name is not None:
            return name
        return rf"\hat H_{{{','.join(expr.labels)}}}"
    if expr.kind == "state":
        _value, _dims, name = expr.args
        return name
    if expr.kind == "function":
        _function, _dims, name, *arguments = expr.args
        rendered = ", ".join(_latex(argument) for argument in arguments)
        return rf"{name}\!\left({rendered}\right)"
    if expr.kind == "state_function":
        _function, _dims, name, *arguments = expr.args
        rendered = ", ".join(_latex(argument) for argument in arguments)
        return rf"{name}\!\left({rendered}\right)"
    if expr.kind == "op":
        name, _space = expr.args
        scope = expr.labels[0]
        symbols = {
            "a": r"\hat a",
            "adag": r"\hat a^\dagger",
            "n": r"\hat n",
            "I": r"\hat I",
            "sigma_x": r"\hat\sigma_x",
            "sigma_y": r"\hat\sigma_y",
            "sigma_z": r"\hat\sigma_z",
            "sigma_plus": r"\hat\sigma_+",
            "sigma_minus": r"\hat\sigma_-",
        }
        symbol = symbols.get(name, rf"\hat{{\mathrm{{{name}}}}}")
        return _scoped_symbol(symbol, scope)
    if expr.kind == "embed":
        return _latex(expr.args[0], parent_precedence)
    precedence = 1 if expr.kind in ("add", "sub") else 2 if expr.kind in ("scale", "mul", "tensor") else 3
    left = _latex(expr.args[0], precedence)
    right = _latex(expr.args[1], precedence + (1 if expr.kind == "sub" else 0))
    if expr.kind == "add":
        text = f"{left} + {right}"
    elif expr.kind == "sub":
        text = f"{left} - {right}"
    elif expr.kind in ("matmul", "tensor", "mul", "scale"):
        text = rf"{left}\,{right}"
    elif expr.kind == "pow":
        text = f"{left}^{{{right}}}"
    else:
        raise TypeError(f"Unknown PhysicsExpr kind {expr.kind!r}.")
    return f"({text})" if precedence < parent_precedence else text


def _scoped_symbol(symbol: str, scope: str) -> str:
    """Attach an owner scope without producing nested LaTeX subscripts."""
    if "_" not in symbol:
        return f"{symbol}_{{{scope}}}"
    base, subscript = symbol.split("_", 1)
    subscript = subscript.removeprefix("{").removesuffix("}")
    return f"{base}_{{{subscript},{scope}}}"
