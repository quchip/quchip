"""One backend-neutral expression tree for authored scalar and operator physics."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping


class UnboundParameterError(ValueError):
    """Numerical materialization was requested without every required value."""


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

    def with_bindings(self, bindings: Mapping[str, Any]) -> "PhysicsExpr":
        """Attach default values used only by direct numerical inspection."""
        return replace(self, _bindings=dict(bindings))

    def parameter_paths(self) -> tuple[str, ...]:
        """Return referenced dotted parameter paths in authored order."""
        paths: list[str] = []

        def visit(node: PhysicsExpr) -> None:
            if node.kind == "parameter":
                path = node.args[0]
                if path not in paths:
                    paths.append(path)
            for arg in node.args:
                if isinstance(arg, PhysicsExpr):
                    visit(arg)

        visit(self)
        return tuple(paths)

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
        backend: Any = None,
        op_lookup: Mapping[tuple[str, str], Any] | None = None,
    ) -> Any:
        """Materialize this expression and return its dense numerical array."""
        if backend is None:
            from quchip.backend import get_default_backend

            backend = get_default_backend()
        native = materialize_expr(self, backend, bindings=bindings, op_lookup=op_lookup)
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


def materialize_expr(
    expr: Any,
    backend: Any,
    *,
    bindings: Mapping[str, Any] | None = None,
    op_lookup: Mapping[tuple[str, str], Any] | None = None,
) -> Any:
    """Lower symbolic physics, passing an already-native contribution through."""
    if not isinstance(expr, PhysicsExpr):
        return expr
    values = dict(expr._bindings)
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
        if node.kind == "op":
            name, levels = node.args
            label = node.labels[0]
            if op_lookup is not None and (label, name) in op_lookup:
                return op_lookup[(label, name)]
            return _standard_operator(backend, name, levels)
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


def _standard_operator(backend: Any, name: str, levels: int) -> Any:
    """Build a standard local operator without consulting a live component."""
    if name == "a":
        return backend.destroy(levels)
    if name == "adag":
        return backend.create(levels)
    if name == "n":
        return backend.number(levels)
    if name == "I":
        return backend.identity(levels)

    zero = backend.basis(levels, 0)
    one = backend.basis(levels, 1)
    p01 = backend.matmul(zero, backend.dag(one))
    p10 = backend.matmul(one, backend.dag(zero))
    if name == "sigma_x":
        return p01 + p10
    if name == "sigma_y":
        return -1j * p01 + 1j * p10
    if name == "sigma_z":
        return backend.matmul(zero, backend.dag(zero)) - backend.matmul(one, backend.dag(one))
    if name == "sigma_plus":
        return p10
    if name == "sigma_minus":
        return p01
    raise ValueError(f"Unknown standard operator {name!r}.")


def _latex(expr: PhysicsExpr, parent_precedence: int = 0) -> str:
    if expr.kind == "literal":
        value = expr.args[0]
        return f"{value:g}" if isinstance(value, (int, float, complex)) else str(value)
    if expr.kind == "parameter":
        path, symbol, _unit = expr.args
        scope = path.rsplit(".", 1)[0]
        return _scoped_symbol(symbol, scope)
    if expr.kind == "op":
        name, _levels = expr.args
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
        return _scoped_symbol(symbols[name], scope)
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
