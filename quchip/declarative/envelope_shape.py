"""Declarative envelope shape base class.

This module lives apart from :mod:`quchip.declarative.models` because
importing :class:`~quchip.control.envelopes.BaseEnvelope` pulls in
:mod:`quchip.control`, which imports :mod:`quchip.engine`, which imports
back into :mod:`quchip.declarative` — a cycle that a :class:`DeviceModel`
subclass, which does not need :class:`BaseEnvelope` at all, should not have
to pay for.
"""

from __future__ import annotations

from typing import Any

from quchip.control.envelopes import BaseEnvelope
from quchip.declarative.qnp import asarray as _asarray
from quchip.declarative.parameters import (
    build_declared_signature,
    parameter_fields,
    resolve_declared_params,
    serializable_value,
    validate_sign,
)


def _synthesize_envelope_init(cls: Any) -> Any:
    """Build a positional ``__init__`` for a declarative envelope subclass.

    Declared parameters become positional-or-keyword arguments in
    declaration order (with their declared defaults); the body forwards to
    :meth:`EnvelopeShape.__init__`, which resolves / validates them — so
    envelope authors get a clean signature without hand-writing one.
    """
    signature = build_declared_signature(parameter_fields(cls), owner=cls)

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        arguments.pop("self")
        EnvelopeShape.__init__(self, **arguments)

    __init__.__signature__ = signature  # type: ignore[attr-defined]
    __init__.__qualname__ = f"{cls.__qualname__}.__init__"
    __init__.__doc__ = f"Initialize {cls.__name__} from its declared parameters."
    return __init__


class EnvelopeShape(BaseEnvelope):
    """Declarative base for pulse envelope shapes.

    Subclasses declare their parameters as annotated class attributes via
    :func:`parameter` (e.g. ``duration: Scalar = parameter(positive=True)``)
    and implement :meth:`value`, which returns the complex envelope at a
    given time. The :meth:`waveform` method is supplied by this base and
    delegates to :meth:`value`, keeping the signal pipeline JAX-traceable
    end-to-end.

    Examples
    --------
    >>> from quchip.declarative import EnvelopeShape, parameter, Scalar
    >>> class LinearRise(EnvelopeShape):
    ...     duration: Scalar = parameter(positive=True)
    ...     amplitude: Scalar = parameter(default=1.0)
    ...     def value(self, t):
    ...         return self.amplitude * (t / self.duration)
    >>> env = LinearRise(duration=20.0, amplitude=0.5)
    >>> float(env.value(10.0))
    0.25
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Synthesize a positional ``__init__`` from declared fields unless the subclass defines its own."""
        super().__init_subclass__(**kwargs)
        if "__init__" not in cls.__dict__:
            cls.__init__ = _synthesize_envelope_init(cls)  # type: ignore[method-assign]

    def __init__(self, **params: Any) -> None:
        """Initialize an envelope from its declared parameter values."""
        values = resolve_declared_params(type(self), params)
        if "duration" not in values:
            raise TypeError(
                f"{type(self).__name__} does not declare a 'duration' parameter; EnvelopeShape "
                "subclasses must declare `duration: Scalar = parameter(positive=True)`."
            )
        duration = values["duration"]
        amplitude = values.get("amplitude", 1.0)
        super().__init__(duration=duration, amplitude=amplitude)
        for name, value in values.items():
            setattr(self, name, value)
        self.validate()

    def __setattr__(self, name: str, value: Any) -> None:
        """Run declared-sign validation on every public write.

        Unconditional (no construction-complete gate): :func:`validate_sign`
        checks concrete scalars only and lets tracers pass, so it is safe on
        every path — including construction and the JAX pytree
        ``BaseEnvelope._unflatten`` reconstruction, which bypasses
        ``__init__`` entirely.
        """
        if not name.startswith("_"):
            spec = parameter_fields(type(self)).get(name)
            if spec is not None:
                validate_sign(name, spec, value)
        super().__setattr__(name, value)

    def validate(self) -> None:
        """Cross-field validation hook, run at the end of construction.

        Default is a no-op. Subclasses override to enforce constraints that
        span multiple declared parameters (e.g. ``2 * edge_duration <=
        duration``). Checks must be gated on *concrete* scalars via
        :func:`quchip.utils.jax_utils.maybe_concrete_scalar` so traced
        parameters never force concretization.
        """

    def value(self, t: Any) -> Any:
        """Evaluate the envelope at time points *t* in ns.

        Parameters
        ----------
        t : Any
            Time or array of times in ns. May be a JAX tracer; the
            implementation must stay traceable (use :mod:`quchip.declarative.qnp`).

        Returns
        -------
        Any
            The complex envelope value at *t*, matching the shape of *t*.
        """
        raise NotImplementedError

    def __mul__(self, other: Any) -> Any:
        """Treat the envelope as a dynamic scalar multiplying an expression."""
        from quchip.declarative.expr import DynamicScalar

        return DynamicScalar(self) * other

    # Scalar/dynamic multiply is commutative for an envelope acting as a
    # dynamic scalar, so the reflected operator reuses ``__mul__`` verbatim.
    __rmul__ = __mul__

    def waveform(self, t: Any, *, xp: Any | None = None) -> Any:
        """Evaluate :meth:`value` through the BaseEnvelope waveform contract.

        Parameters
        ----------
        t : Any
            Time or array of times in ns, coerced to an array before dispatch.
        xp : module or None, optional
            Array-namespace hint from the caller; ignored here since
            :meth:`value` selects its own namespace via
            :mod:`quchip.declarative.qnp`.

        Returns
        -------
        Any
            The complex envelope value at *t*.
        """
        _ = xp
        return self.value(_asarray(t))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the envelope type and declared parameter values."""
        data: dict[str, Any] = {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
        }
        for name, spec in parameter_fields(type(self)).items():
            if spec.serialize:
                data[name] = serializable_value(getattr(self, name))
        return data

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EnvelopeShape":
        """Reconstruct the envelope from :meth:`to_dict` output."""
        fields = parameter_fields(cls)
        params = {name: d[name] for name, spec in fields.items() if spec.serialize and name in d}
        return cls(**params)
