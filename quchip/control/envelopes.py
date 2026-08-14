"""Pulse envelope models for quantum control.

Envelopes define the local complex pulse shape ``E(t)`` that a drive
line plays between ``t = 0`` and ``t = duration``. Subclasses are
auto-registered for serialization *and* as JAX pytrees (via
``__init_subclass__``), so envelope parameters — duration, amplitude,
edge width, DRAG coefficient, anything stored on the instance — remain
differentiable end-to-end.

Conventions
-----------
- Times are ns; values are complex, with real and imaginary parts carrying
  relative I/Q structure.
- Global phase belongs to scheduling, not to an envelope.
- ``value(local_time)`` stays JAX-traceable; it must not concretize time or
  stored parameters.

References
----------
- Motzoi et al., *Simple Pulses for Elimination of Leakage*,
  PRL 103, 110501 (2009) — motivates Gaussian envelopes with DRAG
  corrections for short transmon pulses.
- Krantz et al., APR 6, 021318 (2019), Sec. IV.C — flat-top
  (Gaussian-edge) pulses for two-qubit gates.

Examples
--------
>>> from quchip import Gaussian, LinearRamp, Square, SquareWithGaussianEdges
>>> g = Gaussian(duration=20.0, sigmas=3.0, amplitude=0.05)
>>> sq = Square(duration=10.0, amplitude=0.1)
>>> fg = SquareWithGaussianEdges(duration=40.0, amplitude=0.1)
>>> lr = LinearRamp(duration=60.0, ramp_duration=50.0, amplitude=4.0)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jax.tree_util as jtu

from quchip.declarative import qnp
from quchip.declarative.parameters import (
    DeclarativeMeta,
    Scalar,
    build_declared_signature,
    parameter,
    parameter_fields,
    resolve_declared_params,
    serializable_value,
    validate_sign,
)
from quchip.utils.jax_utils import array_namespace as _pick_namespace
from quchip.utils.jax_utils import maybe_concrete_scalar
from quchip.utils.registry import Registrable

def _synthesize_envelope_init(cls: type["Envelope"]) -> Any:
    """Build a constructor from declared envelope parameters."""
    signature = build_declared_signature(parameter_fields(cls))

    def __init__(self: Envelope, *args: Any, **kwargs: Any) -> None:
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        arguments.pop("self")
        Envelope.__init__(self, **arguments)

    __init__.__signature__ = signature  # type: ignore[attr-defined]
    __init__.__qualname__ = f"{cls.__qualname__}.__init__"
    __init__.__doc__ = f"Initialize {cls.__name__} from its declared parameters."
    return __init__


class Envelope(Registrable, ABC, registry_root=True, metaclass=DeclarativeMeta):
    """Local complex pulse shape evaluated relative to its scheduled start."""

    duration: Scalar

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "__init__" not in cls.__dict__:
            cls.__init__ = _synthesize_envelope_init(cls)  # type: ignore[method-assign]

        def _flatten(obj: Any) -> tuple[tuple[Any, ...], tuple[str, ...]]:
            names = tuple(parameter_fields(type(obj)))
            return tuple(getattr(obj, n) for n in names), names

        def _unflatten(field_names: tuple[str, ...], children: tuple[Any, ...]) -> Any:
            new = cls.__new__(cls)
            for name, value in zip(field_names, children):
                setattr(new, name, value)
            return new

        jtu.register_pytree_node(cls, _flatten, _unflatten)

    def __init__(self, **params: Any) -> None:
        """Initialize an envelope from its declared parameter values."""
        values = resolve_declared_params(type(self), params)
        if "duration" not in values:
            raise TypeError(
                f"{type(self).__name__} must declare "
                "`duration: Scalar = parameter(positive=True)`."
            )
        for name, value in values.items():
            setattr(self, name, value)
        self.validate()

    def __setattr__(self, name: str, value: Any) -> None:
        """Validate declared fields on concrete writes without tracing values."""
        if not name.startswith("_"):
            spec = parameter_fields(type(self)).get(name)
            if spec is not None:
                validate_sign(name, spec, value)
        super().__setattr__(name, value)

    def validate(self) -> None:
        """Validate relations between concrete parameters."""

    @abstractmethod
    def value(self, local_time: Any) -> Any:
        """Return complex I/Q shape at time relative to the pulse start."""
        ...

    def sample(self, local_time: Any, *, real: bool = False) -> Any:
        """Evaluate the shape on an array, optionally returning only I."""
        xp = _pick_namespace(local_time)
        values = xp.asarray(
            self.value(xp.asarray(local_time, dtype=float)),
            dtype=complex,
        )
        return values.real if real else values

    def to_dict(self) -> dict[str, Any]:
        """Serialize the concrete type and its declared parameters."""
        data = super().to_dict()
        for name, spec in parameter_fields(type(self)).items():
            if spec.serialize:
                data[name] = serializable_value(getattr(self, name))
        return data

    @classmethod
    def _from_dict_payload(cls, data: dict[str, Any]) -> "Envelope":
        """Reconstruct one concrete envelope from declared parameters."""
        params = {
            name: data[name]
            for name, spec in parameter_fields(cls).items()
            if spec.serialize and name in data
        }
        return cls(**params)

    def __repr__(self) -> str:
        """Return a constructor-like representation of declared parameters."""
        params = ", ".join(
            f"{name}={getattr(self, name)}" for name in parameter_fields(type(self))
        )
        return f"{type(self).__name__}({params})"


def _gaussian_flat_top(t: Any, duration: Any, edge_duration: Any, sigmas: Any, amplitude: Any) -> Any:
    r"""Complex flat-top waveform with Gaussian ramp-up / ramp-down edges.

    Shared by :class:`GaussianEdge` and :class:`SquareWithGaussianEdges`,
    which differ only in how they derive ``edge_duration`` (a stored
    parameter vs. ``edge_frac * duration``). Each edge is a Gaussian of
    width :math:`\sigma = \tau_e / (2 N_\sigma)` (``edge_duration`` =
    :math:`\tau_e`, ``sigmas`` = :math:`N_\sigma`); the plateau between the
    edges holds constant ``amplitude``.

    Pure ``qnp`` arithmetic throughout, so every argument (including
    ``duration``, ``edge_duration``, and ``amplitude``) may be a JAX tracer
    and the waveform stays differentiable end-to-end.
    """
    sigma = edge_duration / (2.0 * sigmas)
    two_sigma_sq = 2.0 * sigma**2
    out = qnp.ones_like(t, dtype=complex) * amplitude

    out = qnp.where(
        t < edge_duration,
        amplitude * qnp.exp(-((t - edge_duration) ** 2) / two_sigma_sq),
        out,
    )

    fall_center = duration - edge_duration
    out = qnp.where(
        t > fall_center,
        amplitude * qnp.exp(-((t - fall_center) ** 2) / two_sigma_sq),
        out,
    )

    return qnp.asarray(out, dtype=complex)


class Gaussian(Envelope):
    r"""Centered Gaussian pulse.

    .. math::

       E(t) = A \exp\!\left[-\frac{(t - \tau/2)^2}{2 \sigma^2}\right],
       \qquad \sigma = \frac{\tau}{2 N_\sigma}.

    The ``sigmas`` parameter :math:`N_\sigma` is the number of standard
    deviations from the pulse center to its edge at ``t = 0`` or
    ``t = duration``. Gaussian pulses minimize spectral leakage onto
    higher transmon levels and are the starting point for DRAG
    corrections (Motzoi et al., PRL 103, 110501 (2009)).

    The scheduled window ``[0, duration]`` starts and ends at
    ``amplitude * exp(-sigmas**2 / 2)``, not zero — about
    ``0.011 * amplitude`` at the default ``sigmas=3``. The pulse turns
    on and off with that jump; the Gaussian waveform itself is
    unchanged.
    """

    duration: Scalar = parameter(positive=True, unit="ns")
    sigmas: Scalar = parameter(default=3, positive=True)
    amplitude: Scalar = parameter(default=1.0)

    def value(self, t: Any) -> Any:
        """Evaluate the centered Gaussian envelope at time points *t*."""
        center = self.duration / 2.0
        sigma = self.duration / (2.0 * self.sigmas)
        return qnp.asarray(
            self.amplitude * qnp.exp(-((t - center) ** 2) / (2 * sigma**2)),
            dtype=complex,
        )


class GaussianDRAG(Envelope):
    r"""Gaussian pulse with a derivative quadrature.

    .. math::

       E(t) = I(t) + i\,\beta\,\frac{dI}{dt}, \qquad
       I(t) = A\exp\!\left[-\frac{(t-\tau/2)^2}{2\sigma^2}\right].

    ``beta`` is signed and measured in ns. Its sign therefore owns the
    quadrature convention without an additional polarity flag.
    """

    duration: Scalar = parameter(positive=True, unit="ns")
    sigmas: Scalar = parameter(default=3, positive=True)
    amplitude: Scalar = parameter(default=1.0)
    beta: Scalar = parameter(default=0.0, unit="ns")

    def value(self, t: Any) -> Any:
        center = self.duration / 2.0
        sigma = self.duration / (2.0 * self.sigmas)
        in_phase = self.amplitude * qnp.exp(-((t - center) ** 2) / (2.0 * sigma**2))
        derivative = -(t - center) * in_phase / sigma**2
        return qnp.asarray(in_phase + 1j * self.beta * derivative, dtype=complex)


class GaussianEdge(Envelope):
    r"""Flat-top pulse with Gaussian ramp-up and ramp-down edges.

    Each edge is a Gaussian of width :math:`\sigma = \tau_e / (2 N_\sigma)`
    where :math:`\tau_e` = ``edge_duration``; the plateau between edges
    holds a constant amplitude :math:`A`. Total ``duration`` includes
    both edges. Commonly used for two-qubit gates (Krantz et al. 2019,
    Sec. IV.C) because the flat top sets the gate area while the
    Gaussian edges suppress spectral leakage.

    Parameters
    ----------
    duration : float
        Total pulse length, including both edges, in ns.
    edge_duration : float
        Ramp time :math:`\tau_e` per edge, in ns. Must satisfy
        ``2 * edge_duration <= duration``.
    sigmas : float
        Number of standard deviations spanned by each edge.
    amplitude : float
        Plateau amplitude :math:`A`.

    See Also
    --------
    :class:`SquareWithGaussianEdges` : Same shape parameterized by
        ``edge_frac`` (fraction) instead of absolute ``edge_duration``.

    References
    ----------
    * Krantz et al., APR **6**, 021318 (2019), Sec. IV.C.
    """

    duration: Scalar = parameter(positive=True, unit="ns")
    edge_duration: Scalar = parameter(positive=True, unit="ns")
    sigmas: Scalar = parameter(default=3, positive=True)
    amplitude: Scalar = parameter(default=1.0)

    def validate(self) -> None:
        """Reject edges that overrun the pulse (``2 * edge_duration > duration``)."""
        edge = maybe_concrete_scalar(self.edge_duration)
        dur = maybe_concrete_scalar(self.duration)
        if edge is not None and dur is not None and 2 * edge > dur:
            raise ValueError(f"2 * edge_duration ({2 * self.edge_duration}) exceeds duration ({self.duration})")

    def value(self, t: Any) -> Any:
        """Evaluate the flat-top Gaussian-edge envelope at time points *t*."""
        return _gaussian_flat_top(t, self.duration, self.edge_duration, self.sigmas, self.amplitude)


class SquareWithGaussianEdges(Envelope):
    r"""Flat-top pulse with Gaussian ramp-up and ramp-down edges.

    Each ramp has duration :math:`\tau_e = f_e \cdot \tau` with
    :math:`f_e` = ``edge_frac``; the plateau between ramps holds
    amplitude :math:`A`. Total ``duration`` includes both edges. The
    Gaussian width is :math:`\sigma = \tau_e / (2 N_\sigma)` with
    :math:`N_\sigma` = ``sigmas``.

    This is the canonical shape used in Krantz et al. 2019
    (Sec. IV.C) for two-qubit gates — the flat top sets the gate area
    while the Gaussian edges suppress spectral leakage. Parametrizing
    the ramp as a fraction of the total duration makes the shape
    shape-invariant under changes of ``duration``.

    Parameters
    ----------
    duration : float
        Total pulse length in ns (includes both ramps).
    amplitude : float
        Plateau amplitude :math:`A`.
    edge_frac : float
        Ramp length as a fraction of the total duration. Must satisfy
        ``0 < edge_frac`` and ``2 * edge_frac <= 1``.
    sigmas : float
        Number of standard deviations spanned by each ramp.
    """

    duration: Scalar = parameter(positive=True, unit="ns")
    amplitude: Scalar = parameter(default=1.0)
    edge_frac: Scalar = parameter(default=0.25, positive=True)
    sigmas: Scalar = parameter(default=3, positive=True)

    def validate(self) -> None:
        """Reject ramps that overrun the pulse (``2 * edge_frac > 1``)."""
        frac = maybe_concrete_scalar(self.edge_frac)
        if frac is not None and 2 * frac > 1.0:
            raise ValueError(f"2 * edge_frac ({2 * self.edge_frac}) exceeds 1.0")

    @property
    def edge_duration(self) -> float:
        """Ramp duration in ns (``edge_frac * duration``)."""
        return self.edge_frac * self.duration

    def value(self, t: Any) -> Any:
        """Evaluate the fraction-parameterized Gaussian-edge envelope."""
        return _gaussian_flat_top(t, self.duration, self.edge_duration, self.sigmas, self.amplitude)


class LinearRamp(Envelope):
    r"""Linearly rising ramp that holds at peak amplitude.

    The envelope rises linearly from 0 to ``amplitude`` over the first
    ``ramp_duration`` nanoseconds, then holds constant at ``amplitude``
    for the remainder of the pulse.

    .. math::

       E(t) = A \cdot \min\!\left(\frac{t}{\tau_r},\, 1\right),
       \qquad 0 \le t \le \tau,

    where :math:`\tau_r` is ``ramp_duration`` and :math:`\tau` is
    ``duration``.

    Parameters
    ----------
    duration : float
        Total pulse duration in ns.  Must be > 0.
    ramp_duration : float
        Duration of the linear rise in ns.  Must satisfy
        ``0 < ramp_duration <= duration``.
    amplitude : float
        Peak amplitude :math:`A` (default 1.0).

    Notes
    -----
    For an adiabatic ramp into a Kerr-cat qubit, choose ``ramp_duration``
    long compared to ``1 / (2 * K)`` (the inverse gap at the bifurcation
    point).  See Grimm et al., Nature 584, 205 (2020).

    The waveform is JAX-traceable: ``ramp_duration`` and ``amplitude``
    may be JAX tracers so the ramp parameters are differentiable.

    Examples
    --------
    >>> from quchip.control.envelopes import LinearRamp
    >>> ramp = LinearRamp(duration=60.0, ramp_duration=50.0, amplitude=4.0)
    >>> import numpy as np
    >>> t = np.array([0.0, 25.0, 50.0, 55.0])
    >>> np.real(ramp.value(t)).tolist()
    [0.0, 2.0, 4.0, 4.0]
    """

    duration: Scalar = parameter(positive=True, unit="ns")
    ramp_duration: Scalar = parameter(positive=True, unit="ns")
    amplitude: Scalar = parameter(default=1.0)

    def validate(self) -> None:
        """Reject ramps longer than the pulse (``ramp_duration > duration``)."""
        ramp = maybe_concrete_scalar(self.ramp_duration)
        dur = maybe_concrete_scalar(self.duration)
        if ramp is not None and dur is not None and ramp > dur:
            raise ValueError(f"ramp_duration ({self.ramp_duration}) must be <= duration ({self.duration})")

    def value(self, t: Any) -> Any:
        """Evaluate the linear-ramp envelope at time points *t* (ns).

        Parameters
        ----------
        t : array-like
            1-D array of time points in nanoseconds.

        Returns
        -------
        array
            Complex-valued waveform: rises linearly over ``ramp_duration``,
            holds constant at ``amplitude`` afterward.
        """
        return qnp.asarray(
            self.amplitude * qnp.minimum(t / self.ramp_duration, 1.0),
            dtype=complex,
        )


class Square(Envelope):
    r"""Constant-amplitude pulse.

    .. math::

       E(t) = A, \qquad 0 \le t \le \tau.

    Parameters
    ----------
    duration : float
        Pulse length :math:`\tau` in ns.
    amplitude : float
        Real amplitude :math:`A` applied on top of :meth:`value`.
    """

    duration: Scalar = parameter(positive=True, unit="ns")
    amplitude: Scalar = parameter(default=1.0)

    def value(self, t: Any) -> Any:
        """Evaluate the constant envelope at time points *t*."""
        return qnp.ones_like(t, dtype=complex) * self.amplitude
