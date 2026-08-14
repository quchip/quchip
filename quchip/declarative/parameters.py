"""Declared-parameter metadata, synthesized ``__init__`` signatures, and validation.

:func:`parameter` is the field-declaration surface concrete
:class:`~quchip.declarative.models.DeviceModel`,
:class:`~quchip.declarative.models.CouplingModel`, and
:class:`~quchip.control.envelopes.Envelope` subclasses use;
this module resolves those declarations into synthesized constructors and
runs their sign constraints, both at construction and on post-construction
writes.
"""

from __future__ import annotations

import inspect
from abc import ABCMeta
from dataclasses import dataclass
from typing import Any, TypeAlias, TypeVar, dataclass_transform

from quchip.utils.jax_utils import maybe_concrete_scalar

Scalar: TypeAlias = Any


class _Unbound:
    """Marker for a declared parameter that has no current numerical value."""

    def __repr__(self) -> str:
        return "unbound"


UNBOUND = _Unbound()
_DEFAULT_OMITTED = object()


@dataclass(frozen=True)
class Parameter:
    """Metadata for a declarative model parameter field.

    The metadata is intentionally lightweight: it records validation and
    serialization intent while leaving the runtime value fully traceable.
    Sign constraints (``positive`` / ``nonnegative``) are enforced only on
    concrete scalars, so traced values flow through unchecked.
    """

    default: Any = UNBOUND
    positive: bool = False
    nonnegative: bool = False
    serialize: bool = True
    unit: str | None = None
    symbol: str | None = None
    noise: bool = False
    kw_only: bool = False
    required: bool = False


@dataclass(frozen=True)
class Setting:
    """Metadata for a serialized, non-traceable structural model choice."""

    default: Any = UNBOUND
    serialize: bool = True
    kw_only: bool = True


@dataclass(frozen=True)
class _ConstructorField:
    """Runtime marker removed after static constructor inference."""


def constructor_field(
    *,
    default: Any = UNBOUND,
    kw_only: bool = False,
    runtime: Any = UNBOUND,
) -> Any:
    """Describe a synthesized constructor argument to static type checkers."""
    _ = (default, kw_only)
    return _ConstructorField() if runtime is UNBOUND else runtime


def parameter(
    *,
    default: Any = _DEFAULT_OMITTED,
    positive: bool = False,
    nonnegative: bool = False,
    serialize: bool = True,
    unit: str | None = None,
    symbol: str | None = None,
    noise: bool = False,
    kw_only: bool = False,
) -> Any:
    """Declare a traceable numerical parameter on a model class.

    ``unit`` is display metadata for human-readable surfaces such as
    :meth:`Chip.describe` — the package-wide units contract (GHz, ns, mK)
    still governs the value itself. ``None`` means dimensionless or unknown.
    Returns a :class:`Parameter` field descriptor that :func:`parameter_fields`
    collects at class-definition time.

    Parameters
    ----------
    default : Any, optional
        Declared default value. When omitted the parameter remains unbound
        until numerical materialization.
    positive : bool, optional
        Reject concrete values ``<= 0``. Traced values pass unchecked.
    nonnegative : bool, optional
        Reject concrete values ``< 0``. Traced values pass unchecked.
    serialize : bool, optional
        Include the field in :meth:`to_dict` output.
    unit : str or None, optional
        Display-only unit label (e.g. ``"GHz"``).
    symbol : str or None, optional
        Mathematical symbol used when displaying authored physics. The field
        name is used when omitted.
    noise : bool, optional
        Whether :meth:`Chip.set_noise` may configure this field while its
        current value is unset.

    Examples
    --------
    >>> from quchip.declarative import DeviceModel, parameter, Scalar
    >>> class Oscillator(DeviceModel):
    ...     freq: Scalar = parameter(positive=True, unit="GHz")
    ...     def local_hamiltonian(self, op, p):
    ...         return p.freq * op.n
    >>> Oscillator(freq=5.0, levels=3).freq
    5.0
    """
    required = default is _DEFAULT_OMITTED
    return Parameter(
        default=UNBOUND if required else default,
        positive=positive,
        nonnegative=nonnegative,
        serialize=serialize,
        unit=unit,
        symbol=symbol,
        noise=noise,
        kw_only=kw_only,
        required=required,
    )


def setting(*, default: Any = UNBOUND, serialize: bool = True, kw_only: bool = True) -> Any:
    """Declare serialized structural configuration on a model class."""
    if not kw_only:
        raise ValueError("Declarative settings are keyword-only.")
    return Setting(default=default, serialize=serialize, kw_only=kw_only)


@dataclass_transform(
    field_specifiers=(Parameter, parameter, Setting, setting, constructor_field),
)
class DeclarativeMeta(ABCMeta):
    """Expose synthesized declarative constructors without runtime fields."""

    def __new__(
        mcls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> type:
        namespace = dict(namespace)
        for field_name, value in tuple(namespace.items()):
            if isinstance(value, _ConstructorField):
                namespace.pop(field_name)
        return super().__new__(mcls, name, bases, namespace, **kwargs)


@dataclass_transform(
    field_specifiers=(Parameter, parameter, Setting, setting, constructor_field),
    kw_only_default=True,
)
class DriveDeclarativeMeta(DeclarativeMeta):
    """Expose drive parameters as keyword-only synthesized arguments."""


def serializable_value(value: Any) -> Any:
    """Prefer a concrete scalar for serialization while preserving tracers."""
    if value is UNBOUND:
        return None
    concrete = maybe_concrete_scalar(value)
    return concrete if concrete is not None else value


def validate_sign(name: str, spec: Parameter, value: Any) -> None:
    """Enforce a field's declared sign constraint on concrete scalars only.

    Shared by construction (:func:`resolve_declared_params`) and
    post-construction writes (``DeviceModel._validate_param_write``) so the
    two paths cannot drift. Traced values flow through unchecked;
    ``None`` means "unset" and always passes.
    """
    if value is UNBOUND:
        return
    concrete = maybe_concrete_scalar(value)
    if spec.positive and concrete is not None and concrete <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    if spec.nonnegative and concrete is not None and concrete < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def build_declared_signature(
    param_fields: dict[str, Parameter],
    trailing: tuple[inspect.Parameter, ...] = (),
) -> inspect.Signature:
    """Build a synthesized ``__init__`` signature from declared param fields.

    Declared parameters become optional positional-or-keyword arguments in
    declaration order. *trailing* appends structural keyword-only parameters
    such as ``levels`` and ``label``.
    """
    params = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    for name, spec in param_fields.items():
        if spec.kw_only:
            continue
        default = inspect.Parameter.empty if spec.required else spec.default
        params.append(inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default))
    for name, spec in param_fields.items():
        if not spec.kw_only:
            continue
        default = inspect.Parameter.empty if spec.required else spec.default
        params.append(inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=default))
    params.extend(trailing)
    return inspect.Signature(params)


def resolve_declared_params(
    cls: type,
    params: dict[str, Any],
    *,
    fields: dict[str, Parameter] | None = None,
) -> dict[str, Any]:
    """Resolve declared parameters into a {name: value} dict.

    Walks ``parameter_fields(cls)``: for each declared field, pops the
    matching kwarg from *params* (or uses its default), runs the
    concrete-only positivity check, and collects the result. Returns a dict
    of validated parameter values, one entry per declared field. Raises
    ``TypeError`` if a required field is missing or if *params* still
    contains unrecognized keys after the loop.
    """
    resolved_fields = parameter_fields(cls) if fields is None else fields
    values: dict[str, Any] = {}
    for name, spec in resolved_fields.items():
        if name not in params and spec.required:
            raise TypeError(f"Missing required parameter: {name}")
        value = params.pop(name, spec.default)
        validate_sign(name, spec, value)
        values[name] = value
    if params:
        unknown = ", ".join(sorted(params))
        raise TypeError(f"Unexpected parameter(s): {unknown}")
    return values


def resolve_declared_settings(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    """Pop declared structural settings from *values* and apply defaults."""
    return {
        name: values.pop(name, spec.default)
        for name, spec in setting_fields(cls).items()
    }


_Field = TypeVar("_Field", Parameter, Setting)


def _declared_fields(cls: type, field_type: type[_Field]) -> dict[str, _Field]:
    """Collect one declarative field type in base-first definition order."""
    fields: dict[str, _Field] = {}
    for base in reversed(cls.__mro__):
        for name in getattr(base, "__annotations__", {}):
            value = getattr(cls, name, None)
            if isinstance(value, field_type):
                fields[name] = value
    return fields


def parameter_fields(cls: type) -> dict[str, Parameter]:
    """Resolve declarative parameter fields for *cls*, walking the MRO.

    A field is included iff some class in the MRO annotates the name *and*
    the resolved class attribute (``getattr(cls, name)``) is a
    :class:`Parameter` instance. A subclass that shadows an inherited
    ``Parameter`` with a concrete value (e.g. ``freq: Scalar = 5.0``)
    silently drops the field — by design, so subclasses can elide a
    parent's parameter when they want a concrete override.
    """
    return _declared_fields(cls, Parameter)


def setting_fields(cls: type) -> dict[str, Setting]:
    """Resolve structural setting fields for *cls*, walking the MRO."""
    return _declared_fields(cls, Setting)


def validate_declared_fields(cls: type) -> None:
    """Validate cross-cutting constraints on a class's declared fields."""
    for name, spec in parameter_fields(cls).items():
        if spec.noise and not spec.serialize:
            raise TypeError(f"Noise parameter {name!r} must be serializable.")
