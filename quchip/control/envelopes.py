"""Pulse envelope models for quantum control.

Envelopes define the time-domain waveform shape ``E(t)`` that a drive
line plays between ``t = 0`` and ``t = duration``. Subclasses are
auto-registered for serialization *and* as JAX pytrees (via
``__init_subclass__``), so envelope parameters — duration, amplitude,
edge width, DRAG coefficient, anything stored on the instance — remain
differentiable end-to-end.

Conventions
-----------
- Times are ns, waveforms are complex (the real/imag parts are
  interpreted by the drive channel's
  :class:`~quchip.control.signal_spec.DriveModulation`).
- ``waveform(t, xp=jnp)`` must stay JAX-traceable when ``xp`` is
  ``jax.numpy``; no concretization of ``t`` or of stored parameters.

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
>>> sq = Square(duration=10.0, amplitude=0.1, phase=0.0)
>>> fg = SquareWithGaussianEdges(duration=40.0, amplitude=0.1)
>>> lr = LinearRamp(duration=60.0, ramp_duration=50.0, amplitude=4.0)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import jax.tree_util as jtu
import numpy as np

from quchip.utils.jax_utils import maybe_concrete_scalar
from quchip.utils.jax_utils import array_namespace as _pick_namespace
from quchip.utils.registry import Registrable

# Cache of public field names per envelope class. Populated lazily on the
# initial flatten to avoid re-sorting ``vars(obj)`` on every JAX pytree
# traversal (which happens once per traced call, not once per t-sample,
# but is still pure overhead for a hot path).
_FIELD_CACHE: dict[type, tuple[str, ...]] = {}


def _public_fields(obj: Any) -> tuple[str, ...]:
    cls = type(obj)
    cached = _FIELD_CACHE.get(cls)
    if cached is not None:
        return cached
    names = tuple(sorted(name for name in vars(obj) if not name.startswith("_")))
    _FIELD_CACHE[cls] = names
    return names


class BaseEnvelope(Registrable, ABC, registry_root=True):
    """Base class for pulse envelopes.

    Subclasses store their shape parameters as public attributes and
    implement :meth:`waveform`. Every public attribute is registered
    as a pytree leaf, so gradients flow through arbitrary envelope
    parameters without any extra bookkeeping. Serialization (the type
    registry and ``from_dict`` dispatch) is owned by the shared
    :class:`~quchip.utils.registry.Registrable` mixin; the JAX pytree
    registration below is independent of it.

    Parameters
    ----------
    duration : float
        Total pulse duration in ns. Must be positive at construction
        time when concrete; tracers are accepted unchanged.
    amplitude : float
        Scalar amplitude applied on top of :meth:`waveform`.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        def _flatten(obj: Any) -> tuple[tuple[Any, ...], tuple[str, ...]]:
            names = _public_fields(obj)
            return tuple(getattr(obj, n) for n in names), names

        def _unflatten(field_names: tuple[str, ...], children: tuple[Any, ...]) -> Any:
            new = cls.__new__(cls)
            for name, value in zip(field_names, children):
                setattr(new, name, value)
            return new

        jtu.register_pytree_node(cls, _flatten, _unflatten)

    def __init__(self, duration: float, *, amplitude: float = 1.0) -> None:
        """Store common envelope parameters after concrete-only validation."""
        dur_val = maybe_concrete_scalar(duration)
        if dur_val is not None and dur_val <= 0:
            raise ValueError(f"duration must be > 0, got {duration}")
        self.duration = duration
        self.amplitude = amplitude

    @abstractmethod
    def waveform(self, t: np.ndarray, *, xp: Any | None = None) -> np.ndarray:
        """Evaluate the complex envelope at time points *t* (ns).

        Pass ``xp=jax.numpy`` to keep the computation traceable;
        otherwise NumPy is used.
        """
        ...

    def sample(self, tlist: Any, *, real: bool = False) -> Any:
        """Vectorized waveform evaluation on a time array.

        Picks the array namespace from *tlist* (JAX if it is a JAX
        array/tracer, NumPy otherwise), so traced inputs stay traced
        and concrete inputs yield concrete outputs. Returns a complex
        array shaped like ``tlist`` (or the real part if ``real=True``).
        """
        xp = _pick_namespace(tlist)
        t_arr = xp.asarray(tlist, dtype=float)
        w = self.waveform(t_arr, xp=xp)
        return w.real if real else w

    def __repr__(self) -> str:
        """Return a constructor-like representation of public parameters."""
        params = ", ".join(f"{name}={getattr(self, name)}" for name in _public_fields(self))
        return f"{type(self).__name__}({params})"


# ----------------------------------------------------------------------
# Declarative envelopes
#
# The import is intentionally placed here (rather than at the module top) to
# break the circular dependency: ``EnvelopeShape`` imports ``BaseEnvelope`` from
# this module, so it can only be imported once ``BaseEnvelope`` is defined.
# ----------------------------------------------------------------------

if TYPE_CHECKING:
    # Direct (non-lazy) import so mypy resolves EnvelopeShape as a concrete
    # class for base-class/annotation positions below.
    from quchip.declarative.envelope_shape import EnvelopeShape
    from quchip.declarative import Scalar, parameter, qnp  # noqa: E402
else:
    # Runtime path stays on the lazy re-export to preserve the circular-import
    # avoidance described above.
    from quchip.declarative import EnvelopeShape, Scalar, parameter, qnp  # noqa: E402


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


class Gaussian(EnvelopeShape):
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


class GaussianEdge(EnvelopeShape):
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


class SquareWithGaussianEdges(EnvelopeShape):
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


class LinearRamp(EnvelopeShape):
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
    >>> np.real(ramp.waveform(t)).tolist()
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


class Square(EnvelopeShape):
    r"""Constant-amplitude pulse with optional global phase.

    .. math::

       E(t) = A\, e^{i\phi}, \qquad 0 \le t \le \tau.

    Parameters
    ----------
    duration : float
        Pulse length :math:`\tau` in ns.
    amplitude : float
        Real amplitude :math:`A` applied on top of :meth:`value`.
    phase : float
        Global phase :math:`\phi` in radians.
    """

    duration: Scalar = parameter(positive=True, unit="ns")
    amplitude: Scalar = parameter(default=1.0)
    phase: Scalar = parameter(default=0.0, unit="rad")

    def value(self, t: Any) -> Any:
        """Evaluate the constant complex envelope at time points *t*."""
        phase_factor = qnp.exp(qnp.asarray(1j * self.phase))
        return qnp.ones_like(t, dtype=complex) * self.amplitude * phase_factor
