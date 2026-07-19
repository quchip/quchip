"""Declared-parameter metadata, synthesized ``__init__`` signatures, and validation.

:func:`parameter` is the field-declaration surface concrete
:class:`~quchip.declarative.models.DeviceModel`,
:class:`~quchip.declarative.models.CouplingModel`, and
:class:`~quchip.declarative.envelope_shape.EnvelopeShape` subclasses use;
this module resolves those declarations into synthesized constructors and
runs their sign constraints, both at construction and on post-construction
writes.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, TypeAlias

from quchip.utils.jax_utils import maybe_concrete_scalar

Scalar: TypeAlias = Any
Modulation: TypeAlias = Any
_MISSING = object()


@dataclass(frozen=True)
class Parameter:
    """Metadata for a declarative model parameter field.

    The metadata is intentionally lightweight: it records validation and
    serialization intent while leaving the runtime value fully traceable.
    Sign constraints (``positive`` / ``nonnegative``) are enforced only on
    concrete scalars, so traced values flow through unchecked.
    """

    default: Any = _MISSING
    positive: bool = False
    nonnegative: bool = False
    serialize: bool = True
    unit: str | None = None

    @property
    def has_default(self) -> bool:
        """Return whether this field has a declared default value."""
        return self.default is not _MISSING


def parameter(
    *,
    default: Any = _MISSING,
    positive: bool = False,
    nonnegative: bool = False,
    serialize: bool = True,
    unit: str | None = None,
) -> Any:
    """Declare a traceable scalar or modulation parameter on a model class.

    ``unit`` is display metadata for human-readable surfaces such as
    :meth:`Chip.describe` — the package-wide units contract (GHz, ns, mK)
    still governs the value itself. ``None`` means dimensionless or unknown.
    Returns a :class:`Parameter` field descriptor that :func:`parameter_fields`
    collects at class-definition time.

    Parameters
    ----------
    default : Any, optional
        Declared default value. When omitted the field is required in the
        synthesized ``__init__``.
    positive : bool, optional
        Reject concrete values ``<= 0``. Traced values pass unchecked.
    nonnegative : bool, optional
        Reject concrete values ``< 0``. Traced values pass unchecked.
    serialize : bool, optional
        Include the field in :meth:`to_dict` output.
    unit : str or None, optional
        Display-only unit label (e.g. ``"GHz"``).

    Examples
    --------
    >>> from quchip.declarative import DeviceModel, parameter, Scalar
    >>> class Oscillator(DeviceModel):
    ...     freq: Scalar = parameter(positive=True, unit="GHz")
    ...     def local_hamiltonian(self, op):
    ...         return self.freq * op.n
    >>> Oscillator(freq=5.0, levels=3).freq
    5.0
    """
    return Parameter(
        default=default, positive=positive, nonnegative=nonnegative,
        serialize=serialize, unit=unit,
    )


def serializable_value(value: Any) -> Any:
    """Prefer a concrete scalar for serialization while preserving tracers."""
    concrete = maybe_concrete_scalar(value)
    return concrete if concrete is not None else value


def validate_sign(name: str, spec: Parameter, value: Any) -> None:
    """Enforce a field's declared sign constraint on concrete scalars only.

    Shared by construction (:func:`resolve_declared_params`) and
    post-construction writes (``DeviceModel._validate_param_write``) so the
    two paths cannot drift. Traced values flow through unchecked;
    ``None`` means "unset" and always passes.
    """
    concrete = maybe_concrete_scalar(value)
    if spec.positive and concrete is not None and concrete <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    if spec.nonnegative and concrete is not None and concrete < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _check_declaration_order(param_fields: dict[str, Parameter], owner: type | None) -> None:
    """Raise if a required declared field follows an optional one.

    ``inspect.Signature`` enforces the same ordering rule a ``def`` header
    does (every parameter after the first one carrying a default must also
    carry one) and raises a bare ``ValueError`` from deep inside
    :mod:`inspect` when it does not hold — naming neither the declaring
    class nor the two offending fields. This walks *param_fields* first so
    the failure instead names the class and both fields, with the two
    available remedies.
    """
    last_optional: str | None = None
    for name, spec in param_fields.items():
        if spec.has_default:
            last_optional = name
            continue
        if last_optional is not None:
            owner_name = owner.__name__ if owner is not None else "<unknown class>"
            raise TypeError(
                f"{owner_name} declares required parameter {name!r} after optional parameter "
                f"{last_optional!r}. Reorder the declarations so required fields precede optional "
                f"ones, or give {name!r} a default."
            )


def build_declared_signature(
    param_fields: dict[str, Parameter],
    trailing: tuple[inspect.Parameter, ...] = (),
    *,
    owner: type | None = None,
) -> inspect.Signature:
    """Build a synthesized ``__init__`` signature from declared param fields.

    Declared parameters become positional-or-keyword arguments in
    declaration order, carrying their declared defaults. *trailing* appends
    extra (typically keyword-only) structural parameters — e.g. ``levels``,
    ``label`` and the noise kwargs for a :class:`DeviceModel`. *owner* names
    the class whose declarations are being synthesized, for the ordering
    error raised by :func:`_check_declaration_order`.
    """
    _check_declaration_order(param_fields, owner)
    params = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    for name, spec in param_fields.items():
        if spec.has_default:
            params.append(inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=spec.default))
        else:
            params.append(inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD))
    params.extend(trailing)
    return inspect.Signature(params)


def resolve_declared_params(cls: type, params: dict[str, Any]) -> dict[str, Any]:
    """Resolve declared parameters into a {name: value} dict.

    Walks ``parameter_fields(cls)``: for each declared field, pops the
    matching kwarg from *params* (or uses its default), runs the
    concrete-only positivity check, and collects the result. Returns a dict
    of validated parameter values, one entry per declared field. Raises
    ``TypeError`` if a required field is missing or if *params* still
    contains unrecognized keys after the loop.
    """
    fields = parameter_fields(cls)
    values: dict[str, Any] = {}
    for name, spec in fields.items():
        if name in params:
            value = params.pop(name)
        elif spec.has_default:
            value = spec.default
        else:
            raise TypeError(f"Missing required parameter {name!r}")
        validate_sign(name, spec, value)
        values[name] = value
    if params:
        unknown = ", ".join(sorted(params))
        raise TypeError(f"Unexpected parameter(s): {unknown}")
    return values


def parameter_fields(cls: type) -> dict[str, Parameter]:
    """Resolve declarative parameter fields for *cls*, walking the MRO.

    A field is included iff some class in the MRO annotates the name *and*
    the resolved class attribute (``getattr(cls, name)``) is a
    :class:`Parameter` instance. A subclass that shadows an inherited
    ``Parameter`` with a concrete value (e.g. ``freq: Scalar = 5.0``)
    silently drops the field — by design, so subclasses can elide a
    parent's parameter when they want a concrete override.
    """
    fields: dict[str, Parameter] = {}
    for base in reversed(cls.__mro__):
        annotations = getattr(base, "__annotations__", {})
        for name in annotations:
            value = getattr(cls, name, None)
            if isinstance(value, Parameter):
                fields[name] = value
    return fields
