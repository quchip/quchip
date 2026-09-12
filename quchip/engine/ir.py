"""IR types shared by engine responsibilities and backends.

This module is the *contract* between the engine and its backends. It
defines four families of immutable, JAX-pytree-friendly types:

1. **Signal Program AST** — subclasses of :class:`SignalNode`
   (:class:`Constant`, :class:`EnvelopeRef`, :class:`Window`,
   :class:`Shift`, :class:`Scale`, :class:`PolarScale`, :class:`Add`,
   :class:`Multiply`, :class:`Conjugate`, :class:`RealPart`,
   :class:`Carrier`). A pure functional description of a time-dependent
   scalar coefficient ``f(t) : ℝ → ℂ``. Every leaf that a user may sweep
   (envelope parameters, amplitudes, phases, carrier frequencies) is a
   pytree leaf so the whole program is differentiable through JAX.

2. :class:`CanonicalOperator` — backend-free operator storage in
   dense / CSR / DIA layouts plus subsystem metadata. Backends convert
   to and from this format.

3. Resolved open-system physics — :class:`ResolvedSLH` plus
   solve-applied Hamiltonian terms in :class:`EngineResult`.

4. Solve requests — :class:`SolveProblem` and :class:`SolveBatch`, the
   frozen hand-offs to backends. ``backend`` selection is chip-owned
   and is explicitly forbidden from ``options``.

A note on 2π: every operator here has already been scaled by 2π during
engine assembly. Carrier frequencies are stored in angular units
(rad/ns). IR consumers (backends, analyses) must not re-apply 2π.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeAlias, cast

import jax.tree_util as jtu
import numpy as np

from quchip.engine.reference import FieldChannel, ReferencePlane
from quchip.utils.jax_utils import (
    array_namespace,
    contains_tracer,
    is_jax_namespace,
    maybe_concrete_scalar,
    select_array_module,
)
from quchip.utils.constants import TWO_PI

if TYPE_CHECKING:
    from quchip.chip.dressing import BareProductReference
    from quchip.control.envelopes import Envelope
    from quchip.control.signal import SignalKey
    from quchip.engine.frames import FramePlan
    from quchip.declarative.dynamics import TimeCoefficient
    from quchip.declarative.expr import PhysicsExpr
    from quchip.devices.base import BaseDevice

# ── Signal Program AST ──────────────────────────────────────────────
#
# Every node is a frozen dataclass subclassing SignalNode. Defining the
# subclass is the *only* step needed to add a node: pytree registration,
# child traversal, rebuilding, pointwise evaluation, and carrier-band
# decomposition all derive from the class itself (its dataclass fields
# plus its ``evaluate`` / ``bands`` methods). See :class:`SignalNode`.


def _pytree_field_names(cls: type) -> tuple[str, ...]:
    """Collect a node's dataclass field names in definition order.

    Walks the MRO base-first (mirroring how ``@dataclass`` orders
    inherited fields) and skips ``ClassVar`` declarations and private
    names. Runs at class-creation time, before the ``@dataclass``
    decorator has produced ``dataclasses.fields`` metadata, so it reads
    ``__annotations__`` directly.
    """
    names: list[str] = []
    for klass in reversed(cls.__mro__):
        for name, annotation in getattr(klass, "__annotations__", {}).items():
            if name.startswith("_") or "ClassVar" in str(annotation):
                continue
            if name not in names:
                names.append(name)
    return tuple(names)


class SignalNode:
    """Base class for signal-program AST nodes.

    A node describes a time-dependent scalar ``f(t) : ℝ → ℂ``. Subclasses:

    * are ``@dataclass(frozen=True)``; **every dataclass field is a JAX
      pytree child** (registration happens automatically on subclass
      definition), so any field a user may sweep is differentiable;
    * name the fields that hold child nodes (or tuples of child nodes)
      in ``_signal_child_fields``, which powers generic traversal
      (:meth:`signal_children`) and rewriting (:meth:`rebuild_children`);
    * implement :meth:`evaluate` — the node's pointwise semantics;
    * override :meth:`bands` when (and only when) the node interacts
      with :class:`Carrier` leaves: the default treats any carrier-free
      subtree as a single zero-frequency band, which is exact for every
      envelope-like node.
    """

    _signal_child_fields: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        names = _pytree_field_names(cls)

        def flatten(obj: Any) -> tuple[tuple[Any, ...], tuple]:
            return tuple(getattr(obj, name) for name in names), ()

        def unflatten(_aux: tuple, children: tuple) -> Any:
            return cls(**dict(zip(names, children)))

        jtu.register_pytree_node(cls, flatten, unflatten)

    def signal_children(self) -> tuple[SignalNode, ...]:
        """Return this node's child nodes (flattening tuple-valued fields)."""
        out: list[SignalNode] = []
        for name in self._signal_child_fields:
            value = getattr(self, name)
            if isinstance(value, tuple):
                out.extend(value)
            else:
                out.append(value)
        return tuple(out)

    def rebuild_children(self, transform: Any) -> SignalNode:
        """Reconstruct this node with *transform* applied to each child.

        Non-child fields are preserved; nodes without children pass
        through untouched.

        Parameters
        ----------
        transform : callable
            Function applied to every direct signal child.
        """
        if not self._signal_child_fields:
            return self
        updates: dict[str, Any] = {}
        for name in self._signal_child_fields:
            value = getattr(self, name)
            if isinstance(value, tuple):
                updates[name] = tuple(transform(child) for child in value)
            else:
                updates[name] = transform(value)
        # Every concrete node is a frozen dataclass; the base class is not,
        # which is all mypy objects to here.
        return replace(self, **updates)  # type: ignore[type-var]

    def evaluate(self, t: Any, *, xp: Any) -> Any:
        """Evaluate the node in a selected array namespace.

        Parameters
        ----------
        t : scalar or array_like
            Evaluation times in ns.
        xp : module
            NumPy-like array namespace.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement evaluate(t, xp=...)."
        )

    def bands(self) -> tuple[CarrierBand, ...]:
        """Rewrite this subtree into carrier-normalized bands.

        Default: a carrier-free subtree is exactly one zero-frequency
        band whose envelope is the subtree itself. Nodes whose subtrees
        may contain :class:`Carrier` leaves must override this with
        their carrier algebra (see :func:`decompose_carrier_bands`).
        """
        if _contains_carrier(self):
            raise TypeError(
                f"{type(self).__name__} contains Carrier leaves but does not define "
                "carrier-band semantics; override bands() with this node's carrier algebra."
            )
        return (CarrierBand(envelope=self, freq=0.0),)


@dataclass(frozen=True)
class Constant(SignalNode):
    value: complex

    def evaluate(self, t: Any, *, xp: Any) -> Any:
        """Return the constant value, broadcast to the shape of *t*."""
        return xp.asarray(self.value, dtype=complex)

    def bands(self) -> tuple[CarrierBand, ...]:
        """Return the single zero-frequency band carrying this constant."""
        return (CarrierBand(envelope=self, freq=0.0),)


@dataclass(frozen=True)
class EnvelopeRef(SignalNode):
    """Reference to a pulse envelope evaluated at local time."""

    envelope: "Envelope"

    def evaluate(self, t: Any, *, xp: Any) -> Any:
        """Return the referenced envelope's complex ``value(t)``."""
        return xp.asarray(self.envelope.value(xp.asarray(t, dtype=float)), dtype=complex)

    def bands(self) -> tuple[CarrierBand, ...]:
        """Return the single zero-frequency band carrying this envelope."""
        return (CarrierBand(envelope=self, freq=0.0),)


@dataclass(frozen=True)
class CoefficientRef(SignalNode):
    """Internal signal leaf backed by a public component-owned coefficient."""

    coefficient: "TimeCoefficient"

    def evaluate(self, t: Any, *, xp: Any) -> Any:
        return xp.asarray(self.coefficient.value(t))


@dataclass(frozen=True)
class Window(SignalNode):
    """Gate *child* to ``[start, stop]``; zero outside."""

    child: SignalNode
    start: float
    stop: float

    _signal_child_fields: ClassVar[tuple[str, ...]] = ("child",)

    def evaluate(self, t: Any, *, xp: Any) -> Any:
        """Return the child value inside ``[start, stop]`` (ns), zero elsewhere."""
        value = self.child.evaluate(t, xp=xp)
        t_arr = xp.asarray(t, dtype=float)
        mask = (t_arr >= self.start) & (t_arr <= self.stop)
        return xp.where(mask, value, xp.zeros_like(t_arr, dtype=complex))

    def bands(self) -> tuple[CarrierBand, ...]:
        """Return the child bands with the same time gate applied to each envelope."""
        return tuple(
            CarrierBand(Window(b.envelope, self.start, self.stop), b.freq)
            for b in self.child.bands()
        )


@dataclass(frozen=True)
class Shift(SignalNode):
    """Time-shift *child* by ``delta_t``: ``child(t - delta_t)``."""

    child: SignalNode
    delta_t: float

    _signal_child_fields: ClassVar[tuple[str, ...]] = ("child",)

    def evaluate(self, t: Any, *, xp: Any) -> Any:
        """Return the child evaluated at ``t - delta_t`` (ns)."""
        return self.child.evaluate(xp.asarray(t, dtype=float) - self.delta_t, xp=xp)

    def bands(self) -> tuple[CarrierBand, ...]:
        """Return the child bands, each carrying the shift's carrier phase.

        A time shift distributes over bands and contributes the constant
        carrier phase ``exp(-i·freq·Δt)`` per band.
        """
        return tuple(
            CarrierBand(
                Scale(Shift(b.envelope, self.delta_t), _shift_phase(b.freq, self.delta_t)),
                b.freq,
            )
            for b in self.child.bands()
        )


@dataclass(frozen=True)
class Scale(SignalNode):
    """Multiply *child* by a complex scalar ``factor``."""

    child: SignalNode
    factor: complex

    _signal_child_fields: ClassVar[tuple[str, ...]] = ("child",)

    def evaluate(self, t: Any, *, xp: Any) -> Any:
        """Return the child value scaled by ``factor``."""
        return xp.asarray(self.factor) * self.child.evaluate(t, xp=xp)

    def bands(self) -> tuple[CarrierBand, ...]:
        """Return the child bands with ``factor`` folded into each envelope."""
        return tuple(
            CarrierBand(Scale(b.envelope, self.factor), b.freq)
            for b in self.child.bands()
        )


@dataclass(frozen=True)
class PolarScale(SignalNode):
    """Scale *child* by ``amplitude * exp(i * theta)`` (both are pytree leaves)."""

    child: SignalNode
    amplitude: float
    theta: float

    _signal_child_fields: ClassVar[tuple[str, ...]] = ("child",)

    def evaluate(self, t: Any, *, xp: Any) -> Any:
        """Return the child value scaled by ``amplitude * exp(i * theta)``."""
        return self.amplitude * xp.exp(1j * self.theta) * self.child.evaluate(t, xp=xp)

    def bands(self) -> tuple[CarrierBand, ...]:
        """Return the child bands with the polar scale folded into each envelope."""
        return tuple(
            CarrierBand(PolarScale(b.envelope, self.amplitude, self.theta), b.freq)
            for b in self.child.bands()
        )


@dataclass(frozen=True)
class Add(SignalNode):
    children: tuple[SignalNode, ...]

    _signal_child_fields: ClassVar[tuple[str, ...]] = ("children",)

    def evaluate(self, t: Any, *, xp: Any) -> Any:
        """Return the sum of the children evaluated at *t* (ns)."""
        total = xp.asarray(0.0 + 0.0j, dtype=complex)
        for child in self.children:
            total = total + child.evaluate(t, xp=xp)
        return total

    def bands(self) -> tuple[CarrierBand, ...]:
        """Return the concatenation of every child's bands."""
        return tuple(b for child in self.children for b in child.bands())


@dataclass(frozen=True)
class Multiply(SignalNode):
    children: tuple[SignalNode, ...]

    _signal_child_fields: ClassVar[tuple[str, ...]] = ("children",)

    def evaluate(self, t: Any, *, xp: Any) -> Any:
        """Return the product of the children evaluated at *t* (ns)."""
        total = xp.asarray(1.0 + 0.0j, dtype=complex)
        for child in self.children:
            total = total * child.evaluate(t, xp=xp)
        return total

    def bands(self) -> tuple[CarrierBand, ...]:
        """Return the frequency convolution: the Cartesian product of child bands."""
        bands = [CarrierBand(Constant(1.0 + 0.0j), 0.0)]
        for child in self.children:
            child_bands = child.bands()
            bands = [
                CarrierBand(_mul_envelope(b.envelope, cb.envelope), b.freq + cb.freq)
                for b in bands
                for cb in child_bands
            ]
        return tuple(bands)


@dataclass(frozen=True)
class Conjugate(SignalNode):
    child: SignalNode

    _signal_child_fields: ClassVar[tuple[str, ...]] = ("child",)

    def evaluate(self, t: Any, *, xp: Any) -> Any:
        """Return the complex conjugate of the child evaluated at *t* (ns)."""
        return xp.conj(self.child.evaluate(t, xp=xp))

    def bands(self) -> tuple[CarrierBand, ...]:
        """Return the child bands with each envelope conjugated and its frequency negated."""
        return tuple(
            CarrierBand(Conjugate(b.envelope), -b.freq)
            for b in self.child.bands()
        )


@dataclass(frozen=True)
class RealPart(SignalNode):
    child: SignalNode

    _signal_child_fields: ClassVar[tuple[str, ...]] = ("child",)

    def evaluate(self, t: Any, *, xp: Any) -> Any:
        """Return the real part of the child evaluated at *t* (ns)."""
        return xp.real(self.child.evaluate(t, xp=xp))

    def bands(self) -> tuple[CarrierBand, ...]:
        """Return each band split into ``±freq`` halves via ``Re z = (z + z̄) / 2``."""
        bands: list[CarrierBand] = []
        for b in self.child.bands():
            bands.append(CarrierBand(Scale(b.envelope, 0.5), b.freq))
            bands.append(CarrierBand(Scale(Conjugate(b.envelope), 0.5), -b.freq))
        return tuple(bands)


@dataclass(frozen=True)
class ImagPart(SignalNode):
    """Imaginary quadrature of a complex analytic signal."""

    child: SignalNode

    _signal_child_fields: ClassVar[tuple[str, ...]] = ("child",)

    def evaluate(self, t: Any, *, xp: Any) -> Any:
        """Return the imaginary part of the child evaluated at *t* (ns)."""
        return xp.imag(self.child.evaluate(t, xp=xp))

    def bands(self) -> tuple[CarrierBand, ...]:
        """Split bands using ``Im z = (z - z_bar) / (2i)``."""
        bands: list[CarrierBand] = []
        for band in self.child.bands():
            bands.append(CarrierBand(Scale(band.envelope, -0.5j), band.freq))
            bands.append(
                CarrierBand(Scale(Conjugate(band.envelope), 0.5j), -band.freq)
            )
        return tuple(bands)


@dataclass(frozen=True)
class SignalPower(SignalNode):
    """Pointwise power of a scalar signal program."""

    child: SignalNode
    exponent: Any

    _signal_child_fields: ClassVar[tuple[str, ...]] = ("child",)

    def evaluate(self, t: Any, *, xp: Any) -> Any:
        return self.child.evaluate(t, xp=xp) ** self.exponent

    def bands(self) -> tuple[CarrierBand, ...]:
        exponent = maybe_concrete_scalar(self.exponent)
        if exponent is None or int(exponent) != exponent or exponent < 0:
            raise TypeError("Carrier-band expansion requires a non-negative integer signal power.")
        result: SignalProgram = Constant(1.0 + 0.0j)
        for _ in range(int(exponent)):
            result = Multiply((result, self.child))
        return result.bands()


@dataclass(frozen=True)
class Carrier(SignalNode):
    """Oscillating carrier ``exp(sign · i · freq · t)``.

    ``freq`` is in angular units (rad/ns). The default ``sign = -1``
    matches the convention used in rotating-frame decompositions
    (Scully & Zubairy, *Quantum Optics*, §5), where a raising-type
    band on a ``+Δ`` detuning rotates as ``exp(−iΔt)``. Both fields are
    registered as pytree children (``freq`` may be traced; ``sign`` is
    semantically a static ``±1`` — do not map over it).

    Attributes
    ----------
    freq : scalar
        Angular carrier frequency in rad/ns.
    sign : {-1, 1}
        Sign in the complex exponential.
    """

    freq: float
    sign: Literal[-1, 1] = -1

    def evaluate(self, t: Any, *, xp: Any) -> Any:
        """Return ``exp(sign · i · freq · t)``.

        Parameters
        ----------
        t : scalar or array_like
            Evaluation times in ns.
        xp : module
            NumPy-like array namespace.
        """
        t_arr = xp.asarray(t, dtype=float)
        return xp.exp(1j * self.sign * self.freq * t_arr)

    def bands(self) -> tuple[CarrierBand, ...]:
        """Return the single ``sign·freq`` band with a unit-constant envelope."""
        return (CarrierBand(envelope=Constant(1.0 + 0.0j), freq=self.sign * self.freq),)


SignalProgram: TypeAlias = SignalNode


def _contains_carrier(node: SignalNode) -> bool:
    """True when the subtree rooted at *node* contains a :class:`Carrier` leaf."""
    if isinstance(node, Carrier):
        return True
    return any(_contains_carrier(child) for child in node.signal_children())


@dataclass(frozen=True)
class ScalarModulation:
    """Mark a :data:`SignalProgram` as a scalar modulation.

    Attributes
    ----------
    signal : SignalProgram
        Backend-neutral scalar time dependence.
    """

    signal: SignalProgram

    def __post_init__(self) -> None:
        # Reconstruct registered envelopes/coefficients as well as signal nodes.
        object.__setattr__(self, "signal", jtu.tree_map(_capture_solve_input, self.signal))


jtu.register_pytree_node(
    ScalarModulation,
    lambda obj: ((obj.signal,), ()),
    lambda _aux, children: ScalarModulation(signal=children[0]),
)


def _as_time_coefficient(value: Any, *, owner: str) -> ScalarModulation:
    """Lower a public component-owned coefficient to the private engine wrapper."""
    from quchip.declarative.dynamics import TimeCoefficient

    if not isinstance(value, TimeCoefficient):
        raise TypeError(
            f"{owner} coefficient must be a TimeCoefficient; got {type(value).__name__}."
        )
    return ScalarModulation(signal=value._signal_program())


def signal_children(node: Any) -> tuple:
    """Return the :data:`SignalProgram` child nodes of *node*.

    Dispatches to :meth:`SignalNode.signal_children`; a
    :class:`ScalarModulation` wrapper contributes its ``signal``.
    :attr:`EnvelopeRef.envelope` is an ``Envelope``, not a
    ``SignalProgram`` child, and so is *not* returned here.
    """
    if isinstance(node, ScalarModulation):
        return (node.signal,)
    if isinstance(node, SignalNode):
        return node.signal_children()
    return ()


def signal_window_bounds(signal: Any, shift: Any = 0.0) -> list[tuple[Any, Any]]:
    """Return absolute window edges, retaining native values and batch axes.

    Enclosing shifts move the clock of every descendant window. Collecting
    structure requires no numerical decisions, so timing stays differentiable.
    """
    if isinstance(signal, Shift):
        return signal_window_bounds(signal.child, shift + signal.delta_t)
    bounds = [(signal.start + shift, signal.stop + shift)] if isinstance(signal, Window) else []
    for child in signal_children(signal):
        bounds.extend(signal_window_bounds(child, shift))
    return bounds


def evaluate_signal_program(signal: SignalProgram, t: Any, *, xp: Any | None = None) -> Any:
    """Evaluate a signal program at time(s) *t* (ns); *xp* defaults to NumPy."""
    if not isinstance(signal, SignalNode):
        raise TypeError(f"Unsupported signal program node {type(signal).__name__}")
    xp = np if xp is None else xp
    return signal.evaluate(t, xp=xp)


# ── Signal Simplification ─────────────────────────────────────────


def simplify_signal(signal: SignalProgram) -> SignalProgram:
    """Recursively simplify a signal program by canceling exact opposing carrier pairs."""
    signal = signal.rebuild_children(simplify_signal)
    replacement = _cancel_opposing_carriers(signal)
    if replacement is not None:
        signal = replacement
    return signal


def _freq_key(freq: Any) -> Any:
    """Return a hashable key for a carrier frequency that is safe under JAX tracing.

    Concrete scalars hash directly and participate in the carrier
    cancellation rewrite. JAX tracers are unhashable, so the key falls
    back to ``id(freq)`` — deliberately conservative: two distinct tracer
    objects carrying the same traced value will NOT be merged (and so
    their carriers will not cancel), but no incorrect cancellation is
    ever introduced. The traced value itself is never branched on.
    """
    try:
        hash(freq)
        return freq
    except TypeError:
        return id(freq)


def _cancel_opposing_carriers(signal: SignalProgram) -> SignalProgram | None:
    """Cancel exact opposing ``Carrier`` pairs (``+freq`` / ``-freq``) inside a :class:`Multiply`."""
    if not isinstance(signal, Multiply):
        return None

    kept: list[SignalProgram] = []
    # (freq_key, sign) -> count of unmatched carriers
    carriers: dict[tuple[Any, int], int] = {}
    # Parallel map from freq_key back to actual freq value (for reconstruction)
    freq_for_key: dict[Any, Any] = {}
    for child in signal.children:
        if isinstance(child, Carrier):
            fk = _freq_key(child.freq)
            freq_for_key[fk] = child.freq
            key = (fk, child.sign)
            opposite = (fk, -child.sign)
            if carriers.get(opposite, 0):
                carriers[opposite] -= 1
                continue
            carriers[key] = carriers.get(key, 0) + 1
            continue
        kept.append(child)

    for (fk, sign), count in carriers.items():
        # `sign` is always -1 or 1 at runtime (negation of a Literal[-1, 1] widens to
        # int under mypy's numeric-literal rules); this is a pure typing gap.
        kept.extend(Carrier(freq=freq_for_key[fk], sign=cast(Literal[-1, 1], sign)) for _ in range(count))
    kept = [child for child in kept if child != Constant(1.0 + 0.0j)]
    if not kept:
        return Constant(1.0 + 0.0j)
    if len(kept) == 1:
        return kept[0]
    return Multiply(tuple(kept))


# ── Carrier-Band Normalization ────────────────────────────────────


@dataclass(frozen=True)
class CarrierBand:
    """One band of a carrier-normalized signal: ``envelope(t) · exp(i · freq · t)``.

    :func:`decompose_carrier_bands` rewrites any :data:`SignalProgram`
    into a sum of these bands, where ``envelope`` is guaranteed
    carrier-free (no :class:`Carrier` leaves) and therefore slow, and
    ``freq`` is the angular band frequency (rad/ns, sign folded in,
    JAX-traceable). Backends use this to keep the fast oscillation
    analytic while sampling only the slow envelope — exact regardless of
    how resonant the carrier is, unlike pre-sampling the whole product.
    """

    envelope: SignalProgram
    freq: Any


def _shift_phase(freq: Any, delta_t: float) -> Any:
    """Constant carrier phase ``exp(-i · freq · delta_t)`` from a time shift.

    Stays JAX-traceable when either the band frequency or the time shift is
    traced. NumPy is used only when both operands are concrete.
    """
    from quchip.utils.jax_utils import maybe_concrete_scalar

    concrete_freq = maybe_concrete_scalar(freq)
    concrete_time = maybe_concrete_scalar(delta_t)
    if concrete_freq is not None and concrete_time is not None:
        return complex(np.exp(-1j * concrete_freq * concrete_time))
    try:
        import jax.numpy as jnp
    except ImportError:  # pragma: no cover - JAX always present on traced paths
        return np.exp(-1j * freq * delta_t)
    return jnp.exp(-1j * freq * delta_t)


def _mul_envelope(a: SignalProgram, b: SignalProgram) -> SignalProgram:
    """Multiply two carrier-free envelopes, folding the trivial ``Constant(1)`` identity.

    The ``== unit`` comparisons are structural dataclass equality, not a
    branch on a traced value: the only ``Constant`` that appears here is
    the literal ``Constant(1.0 + 0.0j)`` Multiply seed (concrete), and any
    traced envelope node is a different type, so ``==`` short-circuits on
    the type mismatch without touching a tracer.
    """
    unit = Constant(1.0 + 0.0j)
    if a == unit:
        return b
    if b == unit:
        return a
    return Multiply((a, b))


def decompose_carrier_bands(signal: SignalProgram) -> tuple[CarrierBand, ...]:
    """Rewrite *signal* into ``Σ_k envelope_k(t) · exp(i · freq_k · t)`` with carrier-free envelopes.

    This is the scalar-coefficient analogue of the operator band
    decomposition in :mod:`quchip.engine.bands`: every :class:`Carrier`
    leaf is pulled out into a band frequency, leaving a slow, carrier-free
    ``envelope`` per band. The rewrite is exact and follows the carrier
    algebra, implemented node-locally in each :meth:`SignalNode.bands`:

    * ``Carrier(freq, sign)`` → one band ``(1, sign·freq)``.
    * ``Conjugate`` → conjugate the envelope, flip the band frequency.
    * ``RealPart`` → split each band into ``±freq`` (``Re z = (z+z̄)/2``).
    * ``Multiply`` → frequency convolution (Cartesian product of bands).
    * ``Add`` → concatenate bands.
    * ``Scale`` / ``PolarScale`` / ``Window`` / ``Shift`` → distribute over
      bands (``Shift`` also contributes the constant phase ``exp(-i·freq·Δt)``).

    All frequency arithmetic stays in JAX-traceable terms (no ``float()``,
    no branching on traced values).
    """
    if not isinstance(signal, SignalNode):
        raise TypeError(f"Unsupported signal program node {type(signal).__name__}")
    return signal.bands()


# ── Canonical Operator ──────────────────────────────────────────────

CanonicalLayout: TypeAlias = Literal["dense", "csr", "dia"]


@dataclass(frozen=True)
class CanonicalOperator:
    """Backend-free operator with explicit dense/CSR/DIA payload and subsystem metadata.

    For ``dense`` the payload is the full 2D matrix; for ``csr`` it is the
    1D nonzero value array paired with ``indices``/``indptr``; for ``dia``
    it is a 2D ``(n_diags, n_cols)`` array paired with ``offsets``.
    ``dims`` must multiply to ``shape[0]`` and ``subsystem_labels`` names
    each subsystem.

    Attributes
    ----------
    layout : {"dense", "csr", "dia"}
        Storage layout of ``values``.
    values : array_like
        Matrix or sparse payload.
    shape : tuple of int
        Square operator shape.
    dims : tuple of int
        Subsystem dimensions.
    basis : str
        Basis convention for the payload.
    subsystem_labels : tuple of str
        Labels matching ``dims``.
    indices, indptr, offsets : array_like or None
        Sparse CSR/DIA metadata.
    tag : str or None
        Optional diagnostic label.
    """

    layout: CanonicalLayout
    values: Any
    shape: tuple[int, int]
    dims: tuple[int, ...]
    basis: str
    subsystem_labels: tuple[str, ...]
    indices: Any | None = None
    indptr: Any | None = None
    offsets: Any | None = None
    tag: str | None = None

    def __post_init__(self) -> None:
        if self.shape[0] != self.shape[1]:
            raise ValueError(f"CanonicalOperator data must be square, got shape {self.shape}")
        expected_dim = 1
        for d in self.dims:
            expected_dim *= d
        if expected_dim != self.shape[0]:
            raise ValueError(f"Product of dims {self.dims} = {expected_dim} does not match matrix size {self.shape[0]}")
        if len(self.subsystem_labels) != len(self.dims):
            raise ValueError(
                f"subsystem_labels length {len(self.subsystem_labels)} does not match dims length {len(self.dims)}"
            )
        self._validate_payload()
        for name in ("values", "indices", "indptr", "offsets"):
            object.__setattr__(self, name, _capture_solve_input(getattr(self, name)))

    def _validate_payload(self) -> None:
        if self.layout == "dense":
            if self.values.ndim != 2:
                raise ValueError(f"dense CanonicalOperator values must be 2D, got {self.values.ndim}D")
            if tuple(self.values.shape) != self.shape:
                raise ValueError(f"shape {self.shape} does not match dense payload shape {self.values.shape}")
            if any(part is not None for part in (self.indices, self.indptr, self.offsets)):
                raise ValueError("dense CanonicalOperator must not provide sparse payload fields")
            return

        if self.layout == "csr":
            if self.values is None or self.indices is None or self.indptr is None:
                raise ValueError("csr CanonicalOperator requires values, indices, and indptr")
            if self.offsets is not None:
                raise ValueError("csr CanonicalOperator must not provide offsets")
            if self.values.ndim != 1 or self.indices.ndim != 1 or self.indptr.ndim != 1:
                raise ValueError("csr CanonicalOperator payload arrays must be 1D")
            if self.values.shape[0] != self.indices.shape[0]:
                raise ValueError("csr CanonicalOperator values and indices must have the same length")
            if self.indptr.shape[0] != self.shape[0] + 1:
                raise ValueError("csr CanonicalOperator indptr length must be n_rows + 1")
            return

        if self.layout == "dia":
            if self.values is None or self.offsets is None:
                raise ValueError("dia CanonicalOperator requires values and offsets")
            if self.indices is not None or self.indptr is not None:
                raise ValueError("dia CanonicalOperator must not provide CSR payload fields")
            if self.values.ndim != 2 or self.offsets.ndim != 1:
                raise ValueError("dia CanonicalOperator values must be 2D and offsets must be 1D")
            if self.values.shape[0] != self.offsets.shape[0]:
                raise ValueError("dia CanonicalOperator values rows must match offsets length")
            if self.values.shape[1] != self.shape[1]:
                raise ValueError("dia CanonicalOperator values columns must match matrix width")
            return

        raise ValueError(f"Unknown canonical layout {self.layout!r}")

    @property
    def is_sparse(self) -> bool:
        """True for the ``csr`` / ``dia`` layouts, False for ``dense``."""
        return self.layout in {"csr", "dia"}

    @classmethod
    def from_dense(
        cls,
        values: Any,
        *,
        dims: tuple[int, ...],
        basis: str,
        subsystem_labels: tuple[str, ...],
        tag: str | None = None,
    ) -> "CanonicalOperator":
        """Build a dense canonical operator.

        Parameters
        ----------
        values : array_like
            Square dense matrix.
        dims : tuple of int
            Subsystem dimensions.
        basis : str
            Basis convention for ``values``.
        subsystem_labels : tuple of str
            Labels matching ``dims``.
        tag : str or None, optional
            Diagnostic label.
        """
        shape = tuple(values.shape)
        return cls(
            layout="dense",
            values=values,
            shape=(shape[0], shape[1]),
            dims=dims,
            basis=basis,
            subsystem_labels=subsystem_labels,
            tag=tag,
        )

    @classmethod
    def from_csr(
        cls,
        values: Any,
        indices: Any,
        indptr: Any,
        *,
        shape: tuple[int, int],
        dims: tuple[int, ...],
        basis: str,
        subsystem_labels: tuple[str, ...],
        tag: str | None = None,
    ) -> "CanonicalOperator":
        """Build a CSR canonical operator.

        Parameters
        ----------
        values, indices, indptr : array_like
            CSR payload arrays.
        shape : tuple of int
            Square matrix shape.
        dims : tuple of int
            Subsystem dimensions.
        basis : str
            Basis convention for the payload.
        subsystem_labels : tuple of str
            Labels matching ``dims``.
        tag : str or None, optional
            Diagnostic label.
        """
        return cls(
            layout="csr",
            values=values,
            indices=indices,
            indptr=indptr,
            shape=shape,
            dims=dims,
            basis=basis,
            subsystem_labels=subsystem_labels,
            tag=tag,
        )

    @classmethod
    def from_dia(
        cls,
        values: Any,
        offsets: Any,
        *,
        shape: tuple[int, int],
        dims: tuple[int, ...],
        basis: str,
        subsystem_labels: tuple[str, ...],
        tag: str | None = None,
    ) -> "CanonicalOperator":
        """Build a DIA canonical operator.

        Parameters
        ----------
        values, offsets : array_like
            Diagonal payload and integer offsets.
        shape : tuple of int
            Square matrix shape.
        dims : tuple of int
            Subsystem dimensions.
        basis : str
            Basis convention for the payload.
        subsystem_labels : tuple of str
            Labels matching ``dims``.
        tag : str or None, optional
            Diagnostic label.
        """
        return cls(
            layout="dia",
            values=values,
            offsets=offsets,
            shape=shape,
            dims=dims,
            basis=basis,
            subsystem_labels=subsystem_labels,
            tag=tag,
        )

    def with_metadata(
        self,
        *,
        dims: tuple[int, ...] | None = None,
        basis: str | None = None,
        subsystem_labels: tuple[str, ...] | None = None,
        tag: str | None = None,
    ) -> "CanonicalOperator":
        """Return a metadata-adjusted copy with unchanged payload.

        Parameters
        ----------
        dims : tuple of int or None, optional
            Replacement subsystem dimensions.
        basis : str or None, optional
            Replacement basis convention.
        subsystem_labels : tuple of str or None, optional
            Replacement subsystem labels.
        tag : str or None, optional
            Replacement diagnostic label.
        """
        return replace(
            self,
            dims=self.dims if dims is None else dims,
            basis=self.basis if basis is None else basis,
            subsystem_labels=self.subsystem_labels if subsystem_labels is None else subsystem_labels,
            tag=self.tag if tag is None else tag,
        )

    def scaled(self, factor: Any, *, tag: str | None = None) -> "CanonicalOperator":
        """Return a scalar multiple without changing the operator layout.

        Parameters
        ----------
        factor : scalar
            Multiplier applied to the payload.
        tag : str or None, optional
            Replacement diagnostic label.
        """
        return replace(
            self,
            values=self.values * factor,
            tag=self.tag if tag is None else tag,
        )

    def diagonal(self) -> Any:
        """Return the main diagonal without materializing a sparse matrix."""
        xp = array_namespace(self.values)
        values = xp.asarray(self.values, dtype=complex)

        if self.layout == "dense":
            return xp.diagonal(values)

        if self.layout == "dia":
            offsets = xp.asarray(self.offsets, dtype=int)
            return xp.sum(
                xp.where(offsets[:, None] == 0, values, 0),
                axis=0,
            )

        indices = xp.asarray(self.indices, dtype=int)
        indptr = xp.asarray(self.indptr, dtype=int)
        counts = indptr[1:] - indptr[:-1]
        repeat_kwargs = (
            {"total_repeat_length": self.values.shape[0]}
            if is_jax_namespace(xp)
            else {}
        )
        rows = xp.repeat(xp.arange(self.shape[0], dtype=int), counts, **repeat_kwargs)
        selected = xp.where(indices == rows, values, 0)
        diagonal = xp.zeros(self.shape[0], dtype=values.dtype)
        if is_jax_namespace(xp):
            return diagonal.at[rows].add(selected)
        xp.add.at(diagonal, rows, selected)
        return diagonal

    def to_dense(self) -> Any:
        """Materialize the payload as a dense ``shape``-sized matrix.

        Vectorized and array-namespace-preserving (JAX-safe): a traced
        JAX payload yields a JAX array via ``.at[].set`` / ``.add``, a
        concrete NumPy payload yields a NumPy array. Callers that need a
        guaranteed concrete NumPy matrix must wrap the result in
        ``np.asarray(..., dtype=complex)`` themselves.
        """
        payload = next(
            component for component in (self.values, self.indices, self.offsets) if component is not None
        )
        xp = array_namespace(payload)

        if self.layout == "dense":
            return xp.asarray(self.values, dtype=complex)

        if self.layout == "csr":
            values = xp.asarray(self.values, dtype=complex)
            indices = xp.asarray(self.indices, dtype=int)
            indptr = xp.asarray(self.indptr, dtype=int)
            counts = indptr[1:] - indptr[:-1]
            repeat_kwargs = (
                {"total_repeat_length": self.values.shape[0]}
                if is_jax_namespace(xp)
                else {}
            )
            rows = xp.repeat(xp.arange(self.shape[0], dtype=int), counts, **repeat_kwargs)
            dense = xp.zeros(self.shape, dtype=values.dtype)
            if is_jax_namespace(xp):
                return dense.at[rows, indices].set(values)
            dense[rows, indices] = values
            return dense

        offsets = xp.asarray(self.offsets, dtype=int)
        values = xp.asarray(self.values, dtype=complex)
        n_rows, n_cols = self.shape
        col_grid = xp.broadcast_to(xp.arange(n_cols, dtype=int), values.shape)
        row_grid = col_grid - offsets[:, None]
        valid = (row_grid >= 0) & (row_grid < n_rows)
        dense = xp.zeros(self.shape, dtype=values.dtype)
        if is_jax_namespace(xp):
            safe_rows = xp.where(valid, row_grid, 0)
            safe_vals = xp.where(valid, values, 0)
            return dense.at[safe_rows, col_grid].add(safe_vals)
        dense[row_grid[valid], col_grid[valid]] = values[valid]
        return dense

    def fingerprint(self) -> tuple:
        """Batching key: value-sensitive, with an automatic tracer-safe fallback.

        Two crosstalk-rebuilt operators carrying the same coefficients
        collapse to the same key so they batch into one solve slot.
        Under ``jax.jit`` the payload is a tracer (possibly hidden inside
        a backend qarray wrapper, e.g. dynamiqs ``SparseDIAQArray``);
        :func:`contains_tracer` detects that and the key falls back to
        layout + shape/dtype structure only, so ``tobytes()`` is never
        called on a tracer and two equivalent traced operators in
        different batch slots still produce identical keys.
        """
        if contains_tracer((self.values, self.indices, self.indptr, self.offsets)):
            return self._structural_fingerprint()
        base: tuple[Any, ...] = (
            self.layout, tuple(self.shape), tuple(self.dims),
            str(self.basis), tuple(self.subsystem_labels),
        )
        try:
            values_arr = np.ascontiguousarray(np.asarray(self.values))
        except Exception:
            return self._structural_fingerprint()
        base = base + ((values_arr.shape, values_arr.dtype.str, values_arr.tobytes()),)
        if self.layout == "csr":
            idx = np.ascontiguousarray(np.asarray(self.indices, dtype=np.int64))
            indptr = np.ascontiguousarray(np.asarray(self.indptr, dtype=np.int64))
            return base + (idx.tobytes(), indptr.tobytes())
        if self.layout == "dia":
            offsets = np.ascontiguousarray(np.asarray(self.offsets, dtype=np.int64))
            return base + (offsets.tobytes(),)
        return base

    def _structural_fingerprint(self) -> tuple:
        """Tracer-safe fallback key: layout + shape/dtype metadata, never payload values."""

        def _shape_dtype(a: Any) -> Any:
            if a is None:
                return None
            shape = getattr(a, "shape", None)
            if shape is None:
                shape = tuple(np.shape(a))
            dtype = getattr(a, "dtype", None)
            return (tuple(shape), str(dtype) if dtype is not None else None)

        return (
            self.layout, tuple(self.shape), tuple(self.dims),
            str(self.basis), tuple(self.subsystem_labels), "traced",
            _shape_dtype(self.values), _shape_dtype(self.indices),
            _shape_dtype(self.indptr), _shape_dtype(self.offsets),
        )


# ── Hamiltonian Terms ───────────────────────────────────────────────

TermOrigin: TypeAlias = Literal[
    "device", "coupling", "drive", "crosstalk", "flux", "port", "network"
]


@dataclass(frozen=True)
class StaticTerm:
    """Time-independent Hamiltonian contribution.

    The ``operator`` payload has already been scaled by 2π during
    engine assembly; backends must not re-apply it. ``coefficient``
    multiplies ``operator`` and may be a concrete scalar or a JAX
    tracer (sweeps over static couplings, detunings, etc.). ``origin``
    is purely advisory metadata.

    Attributes
    ----------
    operator : CanonicalOperator
        Angular-frequency operator payload.
    coefficient : scalar
        Scalar multiplier, default one.
    origin : str
        Component category that authored the term.
    metadata : dict
        Advisory term metadata.
    """

    operator: CanonicalOperator
    coefficient: complex = 1.0
    origin: TermOrigin = "device"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DynamicTerm:
    """Time-dependent Hamiltonian contribution ``operator · f(t)``.

    ``f(t)`` is wrapped in :class:`ScalarModulation`, which each backend
    lowers into its native coefficient representation (QuTiP callback,
    dynamiqs sampled array, etc.). The ``operator`` is 2π-scaled already
    (see module docstring). ``tag`` is an optional human label; it does
    not participate in physics.

    Attributes
    ----------
    operator : CanonicalOperator
        Angular-frequency operator payload.
    time_dependence : ScalarModulation
        Scalar coefficient evaluated in time.
    origin : str
        Component category that authored the term.
    tag : str or None
        Optional diagnostic label.
    """

    operator: CanonicalOperator
    time_dependence: ScalarModulation
    origin: TermOrigin = "drive"
    tag: str | None = None


@dataclass(frozen=True)
class CollapseTerm:
    """Backend-neutral Lindblad channel, optionally exposed as a port.

    Attributes
    ----------
    operator : CanonicalOperator
        Dimensionless local jump operator before rate scaling.
    rate : scalar
        Lindblad rate in ``1/ns``.
    source, channel : str
        Owning component and local channel labels.
    parameter_paths : tuple of str
        Parameter paths controlling the channel.
    phase : scalar or None
        Coupling phase in radians.
    frame_frequency : scalar or None
        Accessible-channel carrier in GHz; ``None`` marks a hidden channel.
    """

    operator: CanonicalOperator
    rate: Any
    source: str
    channel: str
    parameter_paths: tuple[str, ...] = ()
    phase: Any = None
    frame_frequency: Any = None

    @property
    def label(self) -> str:
        """Return the input-output label for an accessible channel."""
        return self.source

    def latex(self) -> str:
        """Render this collapse channel as an opaque named operator."""
        symbols = {
            "T1": "T_1",
            "T2": "T_2",
            "thermal_occupation": r"\bar n",
            "internal_quality_factor": "Q_\\mathrm{int}",
            "external_quality_factor": "Q_\\mathrm{ext}",
        }
        rendered: list[str] = []
        for path in self.parameter_paths:
            scope, name = path.rsplit(".", 1)
            symbol = symbols.get(name, name)
            if "_" in symbol and not symbol.startswith("\\"):
                base, subscript = symbol.split("_", 1)
                rendered.append(rf"{base}_{{{subscript},{scope}}}")
            else:
                rendered.append(rf"{symbol}_{{{scope}}}")
        arguments = ", ".join(rendered)
        suffix = rf"\!\left({arguments}\right)" if arguments else ""
        return rf"\hat L_{{{self.source},{self.channel}}}{suffix}"


ChannelAccess: TypeAlias = Literal["exposed", "hidden"]


@dataclass(frozen=True)
class HamiltonianProgram:
    """Resolved static and time-dependent Hamiltonian contributions.

    Attributes
    ----------
    static_terms : tuple of StaticTerm
        Time-independent contributions.
    dynamic_terms : tuple of DynamicTerm
        Time-dependent contributions.
    """

    static_terms: tuple[StaticTerm, ...] = ()
    dynamic_terms: tuple[DynamicTerm, ...] = ()


@dataclass(frozen=True)
class SLHChannel:
    """One resolved Markov channel and its boundary accessibility.

    Attributes
    ----------
    key : str
        Unique resolved channel key.
    accessibility : {"exposed", "hidden"}
        Whether the channel reaches the modeled experiment boundary.
    collapse : CollapseTerm
        Component-authored channel record.
    coupling_operator : CanonicalOperator or None
        Resolved physical coupling operator, when already composed.
    reference : ReferencePlane
        External reference-plane transfer.
    input_occupation : scalar or None
        Markov input occupation for a thermal field.
    """

    key: str
    accessibility: ChannelAccess
    collapse: CollapseTerm
    coupling_operator: CanonicalOperator | None = None
    reference: ReferencePlane = field(default_factory=ReferencePlane)
    input_occupation: Any = None

    @property
    def carrier(self) -> Any:
        """Return this channel's stationary carrier in GHz, zero when unframed."""
        return 0.0 if self.collapse.frame_frequency is None else self.collapse.frame_frequency

    @property
    def coupling(self) -> CanonicalOperator:
        """Return the physical coupling operator for this channel."""
        if self.coupling_operator is not None:
            return self.coupling_operator
        phase = 0.0 if self.collapse.phase is None else self.collapse.phase
        prefer_jax = contains_tracer((self.collapse.rate, phase)) or any(
            is_jax_namespace(array_namespace(value))
            for value in (self.collapse.rate, phase)
        )
        xp = select_array_module(prefer_jax)
        return self.collapse.operator.scaled(
            xp.exp(1j * xp.asarray(phase)) * xp.sqrt(xp.asarray(self.collapse.rate)),
            tag=f"slh:{self.key}",
        )

    @property
    def collapse_term(self) -> CollapseTerm:
        """Return the solver-facing collapse record for this resolved channel."""
        if self.coupling_operator is None:
            return self.collapse
        return replace(
            self.collapse,
            operator=self.coupling_operator,
            rate=1.0,
            source=self.key,
            channel="resolved_slh",
            phase=None,
        )


@dataclass(frozen=True)
class ResolvedSLH:
    """Immutable, input-free normal form for resolved Markovian physics.

    Attributes
    ----------
    scattering : array_like
        Unitary scalar channel matrix ``S``.
    hamiltonian : HamiltonianProgram
        Resolved Hamiltonian ``H``.
    channels : tuple of SLHChannel
        Coupling channels ``L`` in scattering order.
    support : array_like or None
        Boolean structural reachability matrix.
    output_network : object or None
        Captured external output processing.

    Notes
    -----
    See :doc:`/physics` for the SLH composition and channel conventions.
    """

    scattering: Any
    hamiltonian: HamiltonianProgram
    channels: tuple[SLHChannel, ...] = ()
    support: Any = None
    output_network: Any = None

    @property
    def has_network_hamiltonian(self) -> bool:
        """Whether connection resolution generated coherent Hamiltonian terms."""
        return any(term.origin == "network" for term in self.hamiltonian.static_terms)

    def __post_init__(self) -> None:
        size = len(self.channels)
        support = (
            np.ones((size, size), dtype=bool)
            if self.support is None
            else np.array(self.support, dtype=bool, copy=True)
        )
        if support.shape != (size, size):
            raise ValueError(
                f"ResolvedSLH support must be one boolean per channel pair; got {support.shape} for {size} channels."
            )
        support.setflags(write=False)
        object.__setattr__(self, "support", support)
        scattering = self.scattering
        if isinstance(scattering, np.ndarray) or not hasattr(scattering, "shape"):
            scattering = np.array(scattering, dtype=complex, copy=True)
            scattering.setflags(write=False)
            object.__setattr__(self, "scattering", scattering)
        shape = tuple(scattering.shape)
        expected_shape = (len(self.channels), len(self.channels))
        if shape != expected_shape:
            raise ValueError(
                "ResolvedSLH scattering must have one row and column per channel; "
                f"got {shape} for {len(self.channels)} channels."
            )

        keys = tuple(channel.key for channel in self.channels)
        if len(set(keys)) != len(keys):
            raise ValueError("ResolvedSLH channel keys must be unique.")

        seen_hidden = False
        for channel in self.channels:
            if channel.accessibility == "hidden":
                seen_hidden = True
            elif channel.accessibility == "exposed" and seen_hidden:
                raise ValueError("ResolvedSLH exposed channels must appear before hidden channels.")

        if not contains_tracer(scattering):
            concrete = np.asarray(scattering, dtype=complex)
            identity = np.eye(len(self.channels), dtype=complex)
            if not np.allclose(concrete.conj().T @ concrete, identity, rtol=1e-10, atol=1e-12):
                raise ValueError("ResolvedSLH concrete scattering must be unitary.")

    @classmethod
    def from_terms(
        cls,
        *,
        static_terms: tuple[StaticTerm, ...],
        dynamic_terms: tuple[DynamicTerm, ...],
        collapse_terms: tuple[CollapseTerm, ...],
    ) -> "ResolvedSLH":
        """Build an identity-scattering model from engine terms.

        Parameters
        ----------
        static_terms : tuple of StaticTerm
            Static Hamiltonian contributions.
        dynamic_terms : tuple of DynamicTerm
            Dynamic Hamiltonian contributions.
        collapse_terms : tuple of CollapseTerm
            Component-authored Lindblad channels.
        """
        exposed: list[SLHChannel] = []
        hidden: list[SLHChannel] = []
        key_counts: dict[str, int] = {}
        for collapse in collapse_terms:
            accessibility: ChannelAccess = (
                "exposed" if collapse.frame_frequency is not None else "hidden"
            )
            prefix = "external" if accessibility == "exposed" else "hidden"
            base_key = f"{prefix}.{collapse.source}.{collapse.channel}"
            occurrence = key_counts.get(base_key, 0) + 1
            key_counts[base_key] = occurrence
            channel = SLHChannel(
                key=base_key if occurrence == 1 else f"{base_key}#{occurrence}",
                accessibility=accessibility,
                collapse=collapse,
            )
            (exposed if accessibility == "exposed" else hidden).append(channel)
        channels = tuple((*exposed, *hidden))
        return cls(
            scattering=np.eye(len(channels), dtype=complex),
            hamiltonian=HamiltonianProgram(
                static_terms=tuple(static_terms),
                dynamic_terms=tuple(dynamic_terms),
            ),
            channels=channels,
            support=np.eye(len(channels), dtype=bool),
        )

    @property
    def S(self) -> Any:
        """Return the scalar scattering matrix."""
        return self.scattering

    def feeds(self, output_index: int, input_index: int) -> bool:
        """Return whether one input structurally reaches one output.

        Parameters
        ----------
        output_index, input_index : int
            Output row and input column in channel order.
        """
        return bool(self.support[output_index, input_index])

    @property
    def L(self) -> tuple[CanonicalOperator, ...]:
        """Return physical coupling operators in channel order."""
        return tuple(channel.coupling for channel in self.channels)

    @property
    def H(self) -> HamiltonianProgram:
        """Return the resolved Hamiltonian program."""
        return self.hamiltonian

    @property
    def external_channels(self) -> tuple[SLHChannel, ...]:
        """Return the exposed boundary channels."""
        return tuple(channel for channel in self.channels if channel.accessibility == "exposed")

    @property
    def hidden_channels(self) -> tuple[SLHChannel, ...]:
        """Return channels traced out by the modeled experiment."""
        return tuple(channel for channel in self.channels if channel.accessibility == "hidden")

    @property
    def collapse_terms(self) -> tuple[CollapseTerm, ...]:
        """Return solver-facing collapse records in resolved channel order."""
        terms = [channel.collapse_term for channel in self.channels]
        thermal = [(index, channel) for index, channel in enumerate(self.channels)
                   if channel.input_occupation is not None]
        if not thermal:
            return tuple(terms)
        xp = select_array_module(contains_tracer((self.S, *(c.input_occupation for _, c in thermal),
                                                  *(operator.values for operator in self.L))))
        operators = xp.stack([xp.asarray(operator.to_dense()) for operator in self.L])
        for index, channel in thermal:
            # Input j couples through K_j = (S† L)_j, not the output row L_j.
            values = xp.einsum("i,ijk->jk", xp.conj(xp.asarray(self.S)[:, index]), operators)
            for name, matrix in (("emission", values), ("absorption", xp.conj(values.T))):
                template = channel.coupling
                operator = CanonicalOperator.from_dense(
                    matrix, dims=template.dims, basis=template.basis,
                    subsystem_labels=template.subsystem_labels, tag=f"thermal:{channel.key}:{name}",
                )
                terms.append(CollapseTerm(operator=operator, rate=channel.input_occupation,
                                          source=channel.key, channel=f"thermal_{name}"))
        return tuple(terms)

@dataclass(frozen=True)
class DroppedTerm:
    """Advisory record for a Hamiltonian term elided by an approximation.

    Compare a dropped band's amplitude with its oscillation frequency to
    assess RWA validity; the leading Bloch-Siegert correction scales as
    amplitude²/frequency. Numeric fields use ordinary GHz and may be traced.
    Static ``band_weights`` let assembly derive the frequency from the frame.

    Parameters
    ----------
    source : str
        Label of the owning component (coupling / drive / …) that
        dropped the term.
    operator : str
        Human-readable operator string (e.g. ``"a_q0 · a_q1"``).
    reason : str
        Short reason (e.g. ``"counter-rotating under RWA"``).
    band_weights : tuple[int, ...] | None
        Excitation-change weights of the dropped band, one per endpoint
        mode in the owner's declared order (e.g. ``(-1, -1)`` for
        ``a·b``). ``None`` when not applicable.
    amplitude : Any | None
        Static prefactor of the dropped term in GHz (e.g. the coupling
        ``g``); possibly traced. ``None`` when the prefactor is
        time-dependent (drive envelopes) or unknown.
    frequency : Any | None
        Oscillation frequency of the dropped band in the assembly
        frame, GHz, positive; possibly traced. ``None`` until resolved
        (assembly fills it from the frame and ``band_weights``).
    """

    source: str
    operator: str
    reason: str
    band_weights: tuple[int, ...] | None = None
    amplitude: Any = None
    frequency: Any = None


@dataclass(frozen=True)
class BoundCoherentInput:
    """One solve-bound incident field, retained outside input-free SLH.

    ``reference_beta`` is the signal scheduled at the authored external
    reference plane. ``beta`` is shifted by the total duration of the exposure's
    inbound reference run and is the field composed with the Markov boundary.
    """

    exposure: str
    source_label: str
    beta: SignalProgram
    reference_beta: SignalProgram

    def __post_init__(self) -> None:
        for name in ("beta", "reference_beta"):
            object.__setattr__(self, name, jtu.tree_map(_capture_solve_input, getattr(self, name)))


@dataclass(frozen=True)
class _ResolvedDressingContext:
    """Frozen basis/backend data needed to analyze one resolved snapshot.

    This is deliberately private engine metadata: backends still consume the
    canonical operators in :class:`EngineResult`, while ``dress()`` uses the
    backend that created the snapshot to preserve native eigenstate objects.
    Local reference factors come from captured basis resolution, so later
    mutation of the source chip cannot change their meaning.
    """

    backend: Any = field(repr=False, compare=False)
    reference: BareProductReference = field(repr=False, compare=False)


@dataclass(frozen=True)
class EngineResult:
    """Backend-neutral resolved physics passed to backends.

    Represents

    .. math::
        H(t) \\;=\\; \\sum_s c_s \\, O_s
                   \\;+\\; \\sum_d O_d \\, f_d(t)

    ``slh`` is the input-free Markov model; ``applied_hamiltonian`` holds
    scheduled controls and solve-bound coherent drives. Operators already
    include the 2π conversion. Backends may use ``metadata`` integration
    hints but remain responsible for resolving finite-support dynamics.
    ``dropped_terms`` is advisory approximation metadata.

    Attributes
    ----------
    slh : ResolvedSLH
        Input-free resolved scattering, Hamiltonian, and channel description.
    applied_hamiltonian : HamiltonianProgram
        Solve-bound static and dynamic terms added after the input-free model.
    coherent_inputs : tuple
        Captured coherent source bindings used to assemble those terms.
    dims : tuple of int
        Solver Hilbert-space dimensions in chip device order.
    metadata : dict
        Advisory solver hints, with frequencies in GHz and time scales in ns.
    dropped_terms : tuple of DroppedTerm
        Terms removed by the selected approximation, retained for diagnostics.
    bases : mapping
        Per-device authored-to-solver basis records.
    authored : object or None
        Captured authored physics description.
    resolved_frame : ResolvedFrame
        Frame frequencies and demodulation convention used by assembly.
    approximation : Approximation or None
        Approximation captured by this resolved snapshot.
    dynamical_supports : tuple of tuple of str
        Device groups coupled by retained dynamical terms.
    dissipation : bool
        Whether :attr:`collapse_terms` exposes the resolved dissipators.
    """

    slh: ResolvedSLH
    applied_hamiltonian: HamiltonianProgram = field(default_factory=HamiltonianProgram)
    coherent_inputs: tuple[BoundCoherentInput, ...] = ()
    dims: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    dropped_terms: tuple[DroppedTerm, ...] = ()
    bases: Mapping[str, Any] = field(default_factory=dict)
    authored: Any = None
    resolved_frame: Any = None
    approximation: Any = None
    dynamical_supports: tuple[tuple[str, ...], ...] = ()
    dissipation: bool = True
    _dressing_context: _ResolvedDressingContext | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def static_terms(self) -> tuple[StaticTerm, ...]:
        """Return resolved and solve-applied static Hamiltonian terms."""
        return self.slh.H.static_terms + self.applied_hamiltonian.static_terms

    @property
    def dynamic_terms(self) -> tuple[DynamicTerm, ...]:
        """Return resolved and solve-applied time-dependent Hamiltonian terms."""
        return self.slh.H.dynamic_terms + self.applied_hamiltonian.dynamic_terms

    @property
    def collapse_terms(self) -> tuple[CollapseTerm, ...]:
        """Return active collapse records in SLH channel order."""
        if not self.dissipation:
            return ()
        return self.slh.collapse_terms

    def dress(
        self,
        *,
        at_time: Any | None = None,
        overlap_threshold: float = 0.5,
        labeling: str = "DE",
    ) -> Any:
        """Dress this resolved Hamiltonian, optionally at one instant.

        Unlike :meth:`Chip.dress <quchip.Chip.dress>`, this method analyzes
        the selected frame and approximation stored in this snapshot. A
        snapshot carrying dynamic Hamiltonian terms requires ``at_time``;
        the result is an instantaneous eigensystem, not a Floquet analysis.

        Parameters
        ----------
        at_time : scalar or None, optional
            Instant in ns for dynamic snapshots.
        overlap_threshold : float, default=0.5
            Minimum bare-state overlap accepted for labeling.
        labeling : str, default="DE"
            Bare-to-dressed assignment strategy.
        """
        from quchip.chip.analysis import dress_engine_result

        return dress_engine_result(
            self,
            at_time=at_time,
            overlap_threshold=overlap_threshold,
            labeling=labeling,
        )

    def with_applied_hamiltonian_terms(
        self,
        *,
        static_terms: tuple[StaticTerm, ...] | None = None,
        dynamic_terms: tuple[DynamicTerm, ...] | None = None,
    ) -> "EngineResult":
        """Return a copy with solve-bound Hamiltonian terms replaced.

        Parameters
        ----------
        static_terms : tuple of StaticTerm or None, optional
            Replacement static terms; ``None`` preserves them.
        dynamic_terms : tuple of DynamicTerm or None, optional
            Replacement dynamic terms; ``None`` preserves them.
        """
        applied = replace(
            self.applied_hamiltonian,
            static_terms=(
                self.applied_hamiltonian.static_terms
                if static_terms is None
                else tuple(static_terms)
            ),
            dynamic_terms=(
                self.applied_hamiltonian.dynamic_terms
                if dynamic_terms is None
                else tuple(dynamic_terms)
            ),
        )
        return replace(self, applied_hamiltonian=applied)

    def _contains_tracer(self) -> bool:
        """Return whether any value-bearing field belongs to a JAX trace.

        Engine IR containers are frozen contracts rather than JAX pytrees, so
        cache guards must inspect their array-bearing fields explicitly.
        """
        operators = (
            tuple(term.operator for term in self.static_terms)
            + tuple(term.operator for term in self.dynamic_terms)
            + tuple(term.operator for term in self.collapse_terms)
        )
        operator_payloads = tuple(
            (
                operator.values,
                operator.indices,
                operator.indptr,
                operator.offsets,
            )
            for operator in operators
        )
        basis_payloads = tuple(
            (record.vectors, record.energies, record.energy_vectors,
             record.authored_hamiltonian.numeric_values()
             if hasattr(record.authored_hamiltonian, "numeric_values") else record.authored_hamiltonian)
            for record in self.bases.values()
        )
        authored_values = (
            self.authored.numeric_values()
            if hasattr(self.authored, "numeric_values")
            else self.authored
        )
        return contains_tracer(
            (
                operator_payloads,
                tuple(term.coefficient for term in self.static_terms),
                tuple(term.time_dependence for term in self.dynamic_terms),
                tuple(
                    (term.rate, term.phase, term.frame_frequency)
                    for term in self.collapse_terms
                ),
                tuple(
                    (term.amplitude, term.frequency)
                    for term in self.dropped_terms
                ),
                tuple(
                    (item.beta, item.reference_beta)
                    for item in self.coherent_inputs
                ),
                basis_payloads,
                authored_values,
                self.resolved_frame,
                self.metadata,
            )
        )

    @property
    def port_terms(self) -> tuple[CollapseTerm, ...]:
        """Return active collapse channels that cross an accessible port boundary."""
        if not self.dissipation:
            return ()
        return tuple(channel.collapse_term for channel in self.slh.external_channels)

    def hamiltonian(self) -> PhysicsExpr:
        """Return the exact canonical Hamiltonian as an inspectable expression.

        This view is derived from the same terms backends receive. Matrix
        leaves remain opaque, while each dynamic coefficient renders as a
        named function of time.
        """
        from quchip.declarative.expr import PhysicsExpr

        expressions: list[PhysicsExpr] = []
        for index, static_term in enumerate(self.static_terms):
            tag = static_term.operator.tag or static_term.origin
            operator = PhysicsExpr.from_matrix(
                static_term.operator.to_dense() / TWO_PI,
                labels=static_term.operator.subsystem_labels,
                dims=static_term.operator.dims,
                name=r"\hat H_0" if tag == "H0" else rf"\hat H_{{{tag},{index}}}",
            )
            expressions.append(static_term.coefficient * operator)
        for index, dynamic_term in enumerate(self.dynamic_terms):
            tag = dynamic_term.tag or dynamic_term.operator.tag or dynamic_term.origin
            operator = PhysicsExpr.from_matrix(
                dynamic_term.operator.to_dense() / TWO_PI,
                labels=dynamic_term.operator.subsystem_labels,
                dims=dynamic_term.operator.dims,
                name=rf"\hat H_{{{tag},{index}}}",
            )
            signal = PhysicsExpr.from_signal(
                dynamic_term.time_dependence.signal,
                name=rf"f_{{{tag},{index}}}",
            )
            expressions.append(signal * operator)
        if not expressions:
            raise ValueError("EngineResult contains no Hamiltonian terms.")
        return sum(expressions[1:], start=expressions[0])

    def latex(self) -> str:
        """Render the canonical Hamiltonian with named time functions."""
        return self.hamiltonian().latex()

    def _repr_latex_(self) -> str:
        return f"${self.latex()}$"

    def dropped_terms_summary(self) -> str:
        """Format :attr:`dropped_terms` as a multi-line human-readable string.

        Traced ``amplitude`` / ``frequency`` values print as ``traced``
        rather than being concretized.
        """
        if not self.dropped_terms:
            return "No dropped terms."

        def _fmt(value: Any) -> str:
            concrete = maybe_concrete_scalar(value)
            return f"{concrete:.6g} GHz" if concrete is not None else "traced"

        lines = [f"{len(self.dropped_terms)} term(s) dropped:"]
        for term in self.dropped_terms:
            extras = [
                f"{name} {_fmt(value)}"
                for name, value in (("amp", term.amplitude), ("freq", term.frequency))
                if value is not None
            ]
            detail = f"{term.reason}; {', '.join(extras)}" if extras else term.reason
            lines.append(f"  [{term.source}] {term.operator}  ({detail})")
        return "\n".join(lines)


def _aggregate_batch_metadata(engine_results: list[EngineResult]) -> dict[str, Any]:
    """Conservatively combine advisory solver hints across batch points."""
    metadata = dict(engine_results[0].metadata)
    for key in ("max_carrier_freq_ghz", "spectral_bound_ghz", "max_step_ns"):
        metadata.pop(key, None)

    carrier_values = [
        result.metadata["max_carrier_freq_ghz"]
        for result in engine_results
        if "max_carrier_freq_ghz" in result.metadata
    ]
    if carrier_values:
        metadata["max_carrier_freq_ghz"] = max(carrier_values)

    spectral_values = [
        result.metadata["spectral_bound_ghz"]
        for result in engine_results
        if "spectral_bound_ghz" in result.metadata
    ]
    if spectral_values:
        metadata["spectral_bound_ghz"] = max(spectral_values)

    step_values = [result.metadata.get("max_step_ns") for result in engine_results]
    non_none = [value for value in step_values if value is not None]
    if len(non_none) == len(step_values) and non_none:
        metadata["max_step_ns"] = min(non_none)
    return metadata


# ── Compiled Sweep Templates ────────────────────────────────────────
#
# Pure caches reused across homogeneous drive sweeps: the underlying
# physics is fully defined by assembly. A sweep over envelope parameters,
# drive frequencies, phases, or frame scalars leaves every
# CanonicalOperator invariant and changes only the signal-program leaves
# that describe f(t). Produced by
# assembly.compile_hamiltonian_template and instantiated per sweep
# point by assembly.instantiate_engine_result, so a
# single JAX ``jit`` trace covers every variant in a homogeneous sweep.


@dataclass(frozen=True)
class HamiltonianTemplate:
    """Chip-topology-invariant Hamiltonian skeleton.

    Contains:

    * ``static_terms`` — already assembled ``H₀`` and any static
      (same-frame) coupling folds.
    * ``invariant_dynamic_terms`` — dynamic terms whose signal programs
      do not depend on drive variants (e.g. band-decomposed couplings),
      already simplified at template-compile time.
    * ``drive_terms`` — pre-embedded, 2π-scaled drive bands
      (:class:`~quchip.engine.assembly.CompiledDriveTerm`) ready
      for per-variant reinstantiation.
    * ``collapse_terms`` — canonical Lindblad channels, including accessible
      port metadata where present.
    * ``reference_drive_ops`` — the structural yardstick used by
      :func:`~quchip.engine.assembly.instantiate_engine_result`
      to reject drive-ops that change the template's skeleton (device,
      drive, envelope type, or drive type).

    Sweep leaves (envelope parameters, drive frequencies, phases, frame
    scalars) are *not* in the template; they rebuild on every
    instantiation.
    """

    resolved_frame: Any  # ResolvedFrame
    approximation: Any
    dims: tuple[int, ...]
    slh: Any  # ResolvedSLH
    static_terms: tuple[Any, ...] = ()              # tuple[StaticTerm, ...]
    invariant_dynamic_terms: tuple[Any, ...] = ()   # tuple[DynamicTerm, ...]
    drive_terms: tuple[Any, ...] = ()               # tuple[assembly.CompiledDriveTerm, ...]
    coherent_terms: tuple[Any, ...] = ()            # tuple[assembly.CompiledCoherentTerm, ...]
    reference_drive_ops: tuple[Any, ...] = ()       # tuple[DriveOp, ...]
    delivered_keys: frozenset[SignalKey] = frozenset()
    dropped_terms: tuple[Any, ...] = ()             # tuple[DroppedTerm, ...]
    #: Single-tone weight-zero bands dropped structurally under RWA during engine assembly.
    #: time (:func:`~quchip.engine.assembly._compile_drive_terms`).
    #: The drop decision needs no drive frequency; resolving each entry into
    #: a :class:`DroppedTerm` does, so this stays a pointer
    #: (``tuple[assembly._StructuralDrop, ...]``) until instantiation.
    weight_zero_drops: tuple[Any, ...] = ()
    #: Advisory spectral-bound hint (ordinary GHz) for the *static* terms.
    #: Computed once at template compile — the static terms are invariant
    #: across a sweep, so re-materializing their dense diagonal on every
    #: instantiation is wasted work. ``None`` when empty, oversized, or not
    #: fully concrete (a traced coefficient stays dynamic). Only the
    #: variant-specific carrier-frequency hint is recomputed per instantiation.
    static_spectral_bound_ghz: float | None = None
    collapse_terms: tuple[Any, ...] = ()            # tuple[CollapseTerm, ...]
    bases: Mapping[str, Any] = field(default_factory=dict)
    authored: Any = None
    dressing_context: Any = None
    dynamical_supports: tuple[tuple[str, ...], ...] = ()


# ── Frame Types ─────────────────────────────────────────────────────

# Python's type system cannot express "scalar-like with JAX tracer support",
# so _is_scalar_like() is the runtime check.
ScalarLike = int | float

if TYPE_CHECKING:
    FrameSpec: TypeAlias = Literal["lab", "rotating", "auto"] | ScalarLike | dict[str | BaseDevice, ScalarLike]


def _is_scalar_like(value: Any) -> bool:
    """Python scalar or 0-d array (including JAX tracers)."""
    return getattr(value, "shape", None) == () or isinstance(value, (int, float))


@dataclass(frozen=True)
class ResolvedFrame:
    """Resolved per-device frame information.

    Describes the rotating-frame transformation applied uniformly to
    the chip:

    * ``frequencies[label]`` — the per-device integration-frame
      frequency ``ω_frame`` in GHz. The static Hamiltonian gets the
      counter-term ``−Σᵢ ω_frame,ᵢ nᵢ``.
    * ``demod_freqs[label] = reference_freq − ω_frame`` — the
      demodulation frequency used post-solve to rotate
      expectations back into the user's control frame.
      ``reference_freq`` is the device attribute (see
      :attr:`~quchip.devices.base.BaseDevice.reference_freq`); it
      merely defaults to the dressed drive frequency when not set
      explicitly.
    * ``mode`` — one of ``"lab"`` / ``"rotating"`` / ``"auto"`` /
      ``"float"`` / ``"dict"``.
    * ``plan`` — the :class:`~quchip.engine.frames.FramePlan` selected for
      ``"auto"``, or ``None`` for an explicit frame.
    """

    frequencies: dict[str, Any]
    demod_freqs: dict[str, Any]
    mode: str
    plan: FramePlan | None = None


# ── Solve Problem ───────────────────────────────────────────────────


def _reject_backend_option(options: dict[str, Any], *, cls_name: str) -> dict[str, Any]:
    """Reject a chip-owned ``"backend"`` key and return a defensive copy of ``options``.

    Backend selection is chip-owned, so a ``"backend"`` key in solver options
    is a contract violation. The returned dict is a fresh copy so callers cannot
    mutate the stored options after construction.
    """
    if "backend" in options:
        raise ValueError(
            f"{cls_name}.options must not contain 'backend'. "
            "Backend selection is chip-owned -- use chip.backend instead."
        )
    return dict(options)


StateStorage: TypeAlias = Literal["all", "final", "none"]


@dataclass(frozen=True)
class SolveProblem:
    """Immutable simulation request handed from the chip pipeline to a backend.

    Backend selection is captured separately from ``options``.
    ``e_ops_meta`` reconstructs flattened band expectations into public
    dict-keyed observables.

    Attributes
    ----------
    chip : Chip
        Source chip captured by the request.
    engine_result : EngineResult
        Frozen resolved physics.
    initial_state : state_like
        Initial ket or density matrix in solver coordinates.
    tlist : array_like
        Solver times in ns.
    e_ops : sequence or None
        Backend-ready expectation operators.
    e_ops_meta : object or None
        Metadata used to reconstruct public observables.
    resolved_frame : ResolvedFrame
        Captured integration and demodulation frame.
    solver : str or None
        Explicit solver selection.
    options : dict
        Backend solver options; a ``"backend"`` key is rejected.
    run_args : dict
        Native trajectory call keywords, separate from integrator options.
    monitoring : dict or None
        Captured channel-index to (efficiency, phase) selection; prepare with_monitoring().
    truncation : object or None
        Captured boundary-population plan.
    states : {"all", "final", "none"} or None
        None retains native stochastic defaults; deterministic requests resolve it to all.
    backend : Backend
        Captured backend owner.
    device_info : tuple
        Device labels and computational flags in subsystem order.
    """

    chip: Any  # Chip (typed as Any to avoid runtime import cycles)
    engine_result: Any  # EngineResult
    initial_state: Any
    tlist: Any
    e_ops: Any = None
    e_ops_meta: Any = None
    resolved_frame: Any = None
    solver: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    truncation: Any = field(default=None, repr=False, compare=False, kw_only=True)
    states: StateStorage | None = field(default=None, kw_only=True)
    run_args: dict[str, Any] = field(default_factory=dict, kw_only=True)
    monitoring: dict[int, tuple[Any, Any]] | None = field(default=None, kw_only=True)
    backend: Any = field(default=None, repr=False, compare=False, kw_only=True)
    device_info: tuple[tuple[str, bool], ...] = field(default=(), kw_only=True)

    @property
    def stochastic(self) -> bool:
        """Whether an explicitly selected native solver produces trajectories."""
        return self.solver not in (None, "sesolve", "mesolve")

    @property
    def dissipation(self) -> bool:
        """Whether this calculation includes the resolved dissipators."""
        return self.engine_result.dissipation

    def solver_name(self, backend: Any) -> str:
        """Return the selected solver name.

        An explicit ``solver`` takes precedence, but ``sesolve`` is rejected for a
        density matrix. Otherwise select ``sesolve`` only for a ket with no collapse
        terms; select ``mesolve`` for a density matrix or any problem with collapse terms.

        Parameters
        ----------
        backend : Backend
            Backend used to classify the initial state.
        """
        is_ket = backend.is_ket(self.initial_state)
        if self.solver is not None:
            if self.solver == "sesolve" and not is_ket:
                raise ValueError("sesolve evolves kets only; a density matrix needs mesolve.")
            if self.solver == "sesolve" and self.engine_result.collapse_terms:
                raise ValueError(
                    "sesolve cannot include dissipation; use mesolve or explicitly build with dissipation=False."
                )
            return self.solver
        return "sesolve" if is_ket and not self.engine_result.collapse_terms else "mesolve"

    def __post_init__(self) -> None:
        if self.solver not in (None, "sesolve", "mesolve", "mcsolve", "ssesolve", "smesolve",
                               "jssesolve", "dssesolve", "dsmesolve"):
            raise ValueError(f"Unknown native solver {self.solver!r}.")
        if self.run_args and not self.stochastic:
            raise ValueError("run_args requires an explicitly selected native trajectory solver.")
        if self.states is None and not self.stochastic:
            object.__setattr__(self, "states", "all")
        if self.states not in (None, "all", "final", "none"):
            raise ValueError('states must be "all", "final", or "none".')
        options = _reject_backend_option(self.options, cls_name="SolveProblem")
        if "t0" in options:
            raise ValueError("The initial-state time is tlist[0]; set tlist instead of option 't0'.")
        if not self.stochastic and {"store_states", "store_final_state"} & options.keys():
            raise ValueError('Use states="all"|"final"|"none" instead of storage flags in options.')
        reserved = {"H", "state", "psi0", "rho0", "tlist", "tsave", "e_ops", "exp_ops", "options",
                    "c_ops", "sc_ops", "jump_ops"} & self.run_args.keys()
        if reserved:
            raise ValueError(f"run_args cannot override assembled inputs: {sorted(reserved)}")
        object.__setattr__(self, "monitoring", _capture_solve_input(self.monitoring))
        run_args = _capture_solve_input(self.run_args)
        if "seeds" in run_args:
            from copy import deepcopy

            run_args["seeds"] = deepcopy(run_args["seeds"])
        object.__setattr__(self, "run_args", run_args)
        object.__setattr__(self, "options", _capture_solve_input(options))
        object.__setattr__(self, "tlist", _capture_solve_input(self.tlist))
        object.__setattr__(self, "e_ops", _capture_solve_input(self.e_ops))
        if self.backend is None and self.chip is not None:
            object.__setattr__(self, "backend", self.chip.backend)
        if not self.device_info:
            object.__setattr__(self, "device_info", tuple(
                (device.label, device.computational) for device in getattr(self.chip, "devices", ())
            ))
        if self.truncation is None and self.chip is not None and isinstance(self.engine_result, EngineResult):
            from quchip.engine.truncation import capture_truncation

            object.__setattr__(self, "truncation", capture_truncation(self.chip, self.engine_result))
        copy_state = getattr(self.initial_state, "copy", None)
        if copy_state is not None:
            object.__setattr__(self, "initial_state", copy_state())


def _capture_solve_input(value: Any) -> Any:
    """Copy mutable input containers and NumPy buffers, retaining native JAX values."""
    from quchip.utils.values import copy_value

    return copy_value(value, readonly=True)


@dataclass(frozen=True)
class LinearResponseProblem:
    """Passive-linear input-output request handed to a backend.

    ``hamiltonian`` is the number-conserving mode matrix in angular units,
    ``couplings`` stacks the channel rows of ``L = C a``, and ``scattering``
    is the complete instantaneous SLH matrix including hidden vacuum and loss
    channels. Frequencies remain ordinary GHz at the public boundary.
    ``plane_indices`` selects the external channels used as matrix rows and
    columns. ``inbound_transfer`` and ``outbound_transfer`` contain the
    per-frequency reference factors for each external channel. Backends return
    the undecorated Markov response; the engine applies both factors.

    Attributes
    ----------
    frequencies : array_like
        Probe frequencies in GHz.
    mode_labels : tuple of str
        Linear Fock modes in matrix order.
    hamiltonian : array_like
        Number-conserving mode matrix in rad/ns.
    couplings : array_like
        Channel-to-mode coupling matrix in ``1/sqrt(ns)``.
    scattering : array_like
        Instantaneous SLH scattering matrix.
    plane_indices : tuple of int
        External channel indices used for result rows and columns.
    inbound_transfer, outbound_transfer : array_like
        Frequency-dependent external reference factors.
    field_channels : tuple of FieldChannel
        Captured input field states.
    """

    frequencies: Any
    mode_labels: tuple[str, ...]
    hamiltonian: Any
    couplings: Any
    scattering: Any
    plane_indices: tuple[int, ...]
    inbound_transfer: Any
    outbound_transfer: Any
    field_channels: tuple[FieldChannel, ...] = ()


@dataclass(frozen=True)
class SteadyStateProblem:
    """Immutable static Lindblad request handed from a chip to its backend.

    Attributes
    ----------
    chip : Chip
        Source chip captured by the request.
    engine_result : EngineResult
        Frozen static resolved physics.
    e_ops : sequence or None
        Backend-ready expectation operators.
    e_ops_meta : object or None
        Metadata used to reconstruct public observables.
    resolved_frame : ResolvedFrame
        Captured stationary frame.
    options : dict
        Backend solver options.
    backend : Backend
        Captured backend owner.
    device_info : tuple
        Device labels and computational flags in subsystem order.
    """

    chip: Any
    engine_result: EngineResult
    e_ops: Any = None
    e_ops_meta: Any = None
    resolved_frame: Any = None
    options: dict[str, Any] = field(default_factory=dict)
    backend: Any = field(default=None, repr=False, compare=False, kw_only=True)
    device_info: tuple[tuple[str, bool], ...] = field(default=(), kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "options",
            _capture_solve_input(_reject_backend_option(self.options, cls_name="SteadyStateProblem")),
        )
        object.__setattr__(self, "e_ops", _capture_solve_input(self.e_ops))
        if self.backend is None:
            object.__setattr__(self, "backend", self.chip.backend)
        if not self.device_info:
            object.__setattr__(self, "device_info", tuple(
                (device.label, device.computational) for device in self.chip.devices
            ))


@dataclass(frozen=True)
class SolveBatch:
    """Explicit solve problems sharing one dispatch owner and sweep shape.

    Attributes
    ----------
    chip : Chip
        Source chip for the batch.
    problems : tuple of SolveProblem
        Compatible frozen requests in flat sweep order.
    params : array_like or None
        Bound parameter mappings on the sweep grid.
    shape : tuple of int
        Sweep-grid shape.
    axes : tuple
        Named sweep-axis metadata.
    """

    chip: Any
    problems: tuple[SolveProblem, ...]
    params: Any = None
    shape: tuple[int, ...] = ()
    axes: tuple[tuple[str, Any], ...] = ()
    # Original collection coordinates for errors raised during compiled execution.
    _failure_context: tuple[tuple[int, dict[str, Any]], ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", _capture_solve_input(self.axes))
        if not self.problems:
            return
        reference = self.problems[0]
        expected_dynamic = len(reference.engine_result.dynamic_terms)
        expected_dims = tuple(reference.engine_result.dims)
        for index, problem in enumerate(self.problems):
            if problem.backend is not reference.backend:
                raise ValueError("Every SolveProblem in a SolveBatch must share a captured backend.")
            actual_dynamic = len(problem.engine_result.dynamic_terms)
            if actual_dynamic != expected_dynamic:
                raise ValueError(
                    f"SolveProblem {index} has {actual_dynamic} dynamic terms; expected {expected_dynamic}."
                )
            if tuple(problem.engine_result.dims) != expected_dims:
                raise ValueError(
                    f"SolveProblem {index} has dims {tuple(problem.engine_result.dims)}; "
                    f"expected {expected_dims}. Structural settings cannot vary in a SolveBatch."
                )
            if (problem.solver != reference.solver or problem.options != reference.options
                    or problem.states != reference.states):
                raise ValueError("Every SolveProblem in a SolveBatch must share solver options.")
            if (problem.tlist is not reference.tlist
                    and not contains_tracer((problem.tlist, reference.tlist))
                    and np.array_equal(np.asarray(problem.tlist), np.asarray(reference.tlist))):
                object.__setattr__(problem, "tlist", reference.tlist)

    @property
    def has_shared_tlist(self) -> bool:
        """Whether every point has the same captured time grid."""
        return all(problem.tlist is self.problems[0].tlist for problem in self.problems)

    @property
    def batch_size(self) -> int:
        return len(self.problems)

    @property
    def initial_states(self) -> tuple[Any, ...]:
        return tuple(problem.initial_state for problem in self.problems)

    @property
    def tlist(self) -> Any:
        if not self.has_shared_tlist:
            raise ValueError("Batch points have different time grids; inspect each problem's tlist.")
        return self.problems[0].tlist

    def signals_for(self, slot: int) -> tuple[ScalarModulation, ...]:
        """Return one dynamic slot across all batch points.

        Parameters
        ----------
        slot : int
            Dynamic-term index.
        """
        return tuple(
            problem.engine_result.dynamic_terms[slot].time_dependence
            for problem in self.problems
        )

    def __len__(self) -> int:
        return self.batch_size

    def __iter__(self) -> Iterator[SolveProblem]:
        for index in range(self.batch_size):
            yield self.element(index)

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, slice):
            return [self.element(index) for index in range(*item.indices(self.batch_size))]
        return self.element(item)

    def params_at(self, point: int | tuple[int, ...]) -> dict[str, Any]:
        """Return sweep values at one grid coordinate.

        Parameters
        ----------
        point : int or tuple of int
            Flat index or multidimensional sweep coordinate.
        """
        if self.params is None:
            return {}
        if self.shape == ():
            if point not in (0, ()):
                raise IndexError(f"Scalar batch only accepts 0 or (), got {point!r}")
            return dict(self.params.item().items())
        coordinate = point if isinstance(point, tuple) else np.unravel_index(point, self.shape)
        return dict(self.params[coordinate].items())

    def element(self, index: int) -> SolveProblem:
        """Return one solve request by flat batch index.

        Parameters
        ----------
        index : int
            Flat batch index.
        """
        return self.problems[index]


# ── Drive Operation ─────────────────────────────────────────────────


@dataclass(frozen=True)
class DriveOp:
    """Drive operation scheduled on a device or a modulable coupling.

    ``freq`` is in GHz; ``None`` selects flux drive (or baseband edge
    pump). ``start_time`` and ``phase_offset`` apply in the control
    frame. ``drive_label`` resolves the drive in the chip's control
    equipment (e.g. ``"charge_0"``). ``target_label`` resolves in the
    chip's device or coupling label space.

    The pulse window retains its absolute scheduled time. A solve may
    select a partial interval; a window wholly outside that interval or
    touching only an endpoint contributes no evolution.
    """

    target_label: str
    envelope: Envelope
    freq: float | None = None
    start_time: float = 0.0
    phase_offset: float = 0.0
    drive_label: str = ""


@dataclass(frozen=True)
class CoherentOp:
    """Coherent field operation scheduled on an external SLH exposure."""

    coherent_input: Any
    envelope: Envelope
    freq: float | None = None
    start_time: float = 0.0
    phase_offset: float = 0.0

    @property
    def exposure(self) -> str:
        """Return the external exposure receiving this incident field."""
        return self.coherent_input.exposure

    @property
    def target_label(self) -> str:
        """Alias the exposure for common solve-window diagnostics."""
        return self.exposure

    @property
    def drive_label(self) -> str:
        """Alias the endpoint label for common scheduling diagnostics."""
        return self.coherent_input.label


ControlOp: TypeAlias = DriveOp | CoherentOp
