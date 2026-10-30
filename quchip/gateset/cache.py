"""Gate propagators that are reused at any start time and virtual-Z frame.

Each pulse of a native is lowered alone on the lab-frame problem, which gives
its delivered terms after the signal chain (delays and crosstalk included) and
their time windows. The native splits into time-disjoint pieces where no
delivered window is open. A piece is covariant when the conditions below hold,
and then its propagator at start time ``τ`` and frame offset ``F`` follows
from the template without a new solve:

.. math::

    U(\\tau + t_1, \\tau + t_0) = C\\, U(t_1, t_0)\\, C^\\dagger,
    \\qquad C = e^{i(F - 2\\pi f_c \\tau) N},

with ``N`` the total excitation number and ``f_c`` the piece's carrier
``freq``. The start-referenced ``detuning`` does not enter. The conditions are
structural:

1. every static term has N-charge 0 and every collapse operator one N-charge
   (checked once for the chip);
2. the chip has no time-dependent terms of its own, and every transform of its
   signal chain declares ``linear_time_invariant``, as the built-in delays,
   gains, and crosstalk do;
3. every carrier pulse of the piece has the same ``freq`` ``f_c`` and frame;
4. every delivered term of a pulse has one N-charge q; a term of a carrier
   pulse turns by ``exp(iqθ)`` when the pulse phase turns by θ, and a term of
   a baseband pulse in a carrier piece has charge 0;
5. the pulse's drive builds its signal with the shared ``signal()``, and
   every term is built from the pulse's envelope, carriers, and the standard
   signal algebra, with no coefficient of absolute time.

Condition 2 lets each pulse be lowered alone: the delivered field of a piece
is the sum of its pulses' delivered fields, and each moves with its pulse.
Under conditions 2, 3, and 5, a carrier pulse that starts ``τ`` later gives
the field of the same pulse with its phase turned by ``-2π f_c τ``, shifted
by ``τ``, so condition 4 also covers the start time.

The gate set proves condition 4 on the structure of each delivered
coefficient. It lowers a carrier pulse a second time with a stand-in for its
phase that has no arithmetic, comparison, hash, conversion, or truth value, so
the lowering can only store it. The stand-in must reach each delivered tree only as the angle
of a polar scale, and the tree must equal the first lowering everywhere else.
Such a polar scale carries phase harmonic 1, and the node types combine
harmonics: a sum takes their union, a product adds them, a conjugate negates
them, a real or imaginary part takes both signs, and an integer power adds
them that many times. A term passes when its only harmonic is its charge, so
the result holds for every phase, amplitude, and envelope. A lowering that
computes with the phase, for example one that folds ``cos θ`` into a scale
factor, fails the term. Signal bands cannot show the harmonic: at zero
delivered frequency both sidebands of a pulse fall on one band.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from functools import reduce
from typing import Any

import jax.tree_util as jtu
import numpy as np

from quchip.control.drive import BaseDrive
from quchip.engine import ir
from quchip.engine.ir import signal_children, signal_window_bounds
from quchip.results.propagator import solver_levels

_CHARGE_ATOL = 1e-12
_PULSE_LOCAL_NODES = (ir.Constant, ir.EnvelopeRef, ir.Carrier, ir.Window, ir.Shift, ir.Scale, ir.PolarScale,
                      ir.Add, ir.Multiply, ir.Conjugate, ir.RealPart, ir.ImagPart, ir.SignalPower)


def _dense(term_operator: Any) -> np.ndarray:
    return np.asarray(term_operator.to_dense())


def _charges(matrix: np.ndarray, total: np.ndarray) -> frozenset[int]:
    rows, cols = np.nonzero(np.abs(matrix) > _CHARGE_ATOL * max(1.0, float(np.abs(matrix).max(initial=0.0))))
    return frozenset((total[rows] - total[cols]).tolist())


def _foreign_node(signal: Any) -> str:
    """Return the name of a node that is not pulse-local, or ``""``."""
    if not isinstance(signal, _PULSE_LOCAL_NODES):
        return type(signal).__name__
    for child in signal_children(signal):
        name = _foreign_node(child)
        if name:
            return name
    return ""


def model_check(problem: Any) -> str:
    """Return why the chip's own terms break placement reuse, or ``""``."""
    engine_result = problem.engine_result
    total = solver_levels(engine_result).sum(axis=1)
    for term in engine_result.static_terms:
        if _charges(_dense(term.operator), total) - {0}:
            return "a static term changes the excitation number (no RWA)"
    for term in engine_result.collapse_terms:
        if len(_charges(_dense(term.operator), total)) > 1:
            return f"collapse operator {term.source}.{term.channel} mixes excitation numbers"
    return ""


def _reached(matrix: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Return a mask of the devices on which a term does not act as the identity.

    A term acts as the identity on a device when it keeps the device's level
    and each of its elements is the same for every level of the device that
    the basis holds.
    """
    magnitude = np.abs(matrix)
    tolerance = _CHARGE_ATOL * max(1.0, float(magnitude.max(initial=0.0)))
    rows, cols = np.nonzero(magnitude > tolerance)
    reached = np.any(levels[rows] != levels[cols], axis=0)
    index = {state: i for i, state in enumerate(map(tuple, levels.tolist()))}
    for device in np.flatnonzero(~reached):
        for row, col in zip(rows, cols):
            row_state, col_state = list(levels[row]), list(levels[col])
            for level in range(int(levels[:, device].max()) + 1):
                row_state[device] = col_state[device] = level
                i, j = index.get(tuple(row_state)), index.get(tuple(col_state))
                if i is not None and j is not None and abs(matrix[i, j] - matrix[row, col]) > tolerance:
                    reached[device] = True
                    break
            if reached[device]:
                break
    return reached


def signal_check(chip: Any) -> str:
    """Return why the chip's signal chain breaks placement reuse, or ``""``."""
    equipment = chip.control_equipment
    for transform in () if equipment is None else equipment.signal_chain:
        if not getattr(transform, "linear_time_invariant", False):
            return f"the signal transform {type(transform).__name__} does not declare linear_time_invariant"
    return ""


@dataclass(frozen=True)
class DeliveredTerm:
    """One delivered term of a pulse: N-charges, a foreign signal node, the devices on which it does not act as
    the identity, and whether its structure proves that it turns by ``exp(iqθ)`` when the phase of its carrier
    pulse turns by θ (always true for a baseband pulse, which has no carrier phase)."""

    charges: frozenset[int]
    foreign: str
    devices: frozenset[str] = frozenset()
    follows_phase: bool = True


@dataclass(frozen=True)
class Delivered:
    """The delivered terms of one pulse, the hull of their windows (``None`` when nothing is delivered), and the
    class of the pulse's drive when that class builds the signal with its own ``signal()`` (else ``""``)."""

    support: tuple[float, float] | None
    terms: tuple[DeliveredTerm, ...]
    own_signal: str = ""


class _Phase:
    """A stand-in for a pulse phase. It has no arithmetic, comparison, hash, conversion, or truth value, so a
    lowering can only store it."""

    __slots__ = ()
    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        raise TypeError("the phase stand-in of a gate-set check has no value to compare")

    def __ne__(self, other: object) -> bool:
        raise TypeError("the phase stand-in of a gate-set check has no value to compare")

    def __bool__(self) -> bool:
        raise TypeError("the phase stand-in of a gate-set check has no truth value")

    def __repr__(self) -> str:
        return "<pulse phase>"


def _lowered(chip: Any, op: Any) -> Any:
    from quchip.engine.problem import build_solve_problem

    stop = float(op.start_time) + float(op.envelope.duration)
    return build_solve_problem(chip, [op], np.array([0.0, max(stop, 1.0)]), states="final", frame="lab",
                               dissipation=False, _product_space=True).engine_result


def _probed(chip: Any, op: Any) -> list[tuple[np.ndarray, Any]]:
    """Lower a pulse with the phase stand-in and return each term's operator and signal; empty when the lowering
    computes with the phase."""
    try:
        terms = _lowered(chip, replace(op, phase_offset=_Phase())).dynamic_terms
        return [(_dense(term.operator), term.time_dependence.signal) for term in terms]
    except Exception:  # any use of the stand-in other than storage
        return []


def _holds_phase(value: Any) -> bool:
    """Whether a node field holds the phase stand-in among its leaves."""
    return any(isinstance(leaf, _Phase) or (isinstance(leaf, np.ndarray) and leaf.dtype == object
                                            and any(isinstance(item, _Phase) for item in leaf.flat))
               for leaf in jtu.tree_leaves(value))


def _same(before: Any, after: Any) -> bool:
    """Whether two node fields hold the same values; the engine rebuilds envelopes, so compare pytree leaves."""
    if before is after:
        return True
    left, left_tree = jtu.tree_flatten(before)
    right, right_tree = jtu.tree_flatten(after)
    # PyTreeDef compares by value at runtime; jax's stubs give mypy no signature for it.
    same_tree = left_tree == right_tree  # type: ignore[operator]
    return same_tree and all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(left, right))


def _harmonics(signal: Any, probed: Any, phase: Any) -> frozenset[int] | None:
    """Return the harmonics of the pulse phase in a delivered signal, or ``None`` when its structure does not
    prove them. ``probed`` is the same signal lowered with the phase stand-in, which may appear only as the angle
    of a polar scale; every other field must equal its value in ``signal``, and that angle must equal ``phase``.
    """
    if type(signal) is not type(probed) or not isinstance(signal, _PULSE_LOCAL_NODES):
        return None
    children = getattr(type(signal), "_signal_child_fields", ())
    own = 0
    for item in fields(signal):
        if item.name in children:
            continue
        value, stand_in = getattr(signal, item.name), getattr(probed, item.name)
        if isinstance(signal, ir.PolarScale) and item.name == "theta" and isinstance(stand_in, _Phase):
            if not _same(value, phase):
                return None
            own = 1
        elif _holds_phase(stand_in) or not _same(value, stand_in):
            return None
    pairs = list(zip(signal_children(signal), signal_children(probed)))
    if len(pairs) != len(signal_children(signal)) or len(pairs) != len(signal_children(probed)):
        return None
    parts = [_harmonics(a, b, phase) for a, b in pairs]
    if any(part is None for part in parts):
        return None
    sets: list[frozenset[int]] = [part for part in parts if part is not None]

    def plus(left: frozenset[int], right: frozenset[int]) -> frozenset[int]:
        return frozenset(a + b for a in left for b in right)

    if isinstance(signal, ir.Add):
        combined = frozenset().union(*sets)
    elif isinstance(signal, ir.Multiply):
        combined = reduce(plus, sets, frozenset({0}))
    elif isinstance(signal, ir.Conjugate):
        combined = frozenset(-h for h in sets[0])
    elif isinstance(signal, (ir.RealPart, ir.ImagPart)):
        combined = sets[0] | frozenset(-h for h in sets[0])
    elif isinstance(signal, ir.SignalPower):
        exponent = float(signal.exponent)
        if sets[0] == frozenset({0}):
            combined = sets[0]
        elif exponent.is_integer() and exponent >= 0:
            combined = reduce(plus, [sets[0]] * int(exponent), frozenset({0}))
        else:
            return None
    else:
        combined = sets[0] if sets else frozenset({0})
    return frozenset(h + own for h in combined)


def lower_pulse(chip: Any, op: Any) -> Delivered:
    """Lower one scheduled op alone and return its delivered terms.

    A carrier pulse is lowered a second time with a stand-in for its phase,
    which shows the phase harmonics of each term. Charges, windows, and
    harmonics depend on the pulse's structure, not on its amplitude or phase.
    """
    engine_result = _lowered(chip, op)
    levels = solver_levels(engine_result)
    total = levels.sum(axis=1)
    labels = np.array([device.label for device in chip.devices])
    probed = _probed(chip, op) if isinstance(op, ir.DriveOp) and op.freq is not None else None
    windows: list[tuple[float, float]] = []
    terms = []
    for position, term in enumerate(engine_result.dynamic_terms):
        signal = term.time_dependence.signal
        bounds = signal_window_bounds(signal)
        if not bounds:
            raise ValueError(f"A delivered field of the pulse on {op.drive_label!r} at {float(op.start_time):g} ns "
                             "has no time window, so the gate set cannot bound the gate.")
        windows.extend((float(lo), float(hi)) for lo, hi in bounds)
        matrix = _dense(term.operator)
        charges = _charges(matrix, total)
        follows = True
        if probed is not None:
            harmonics = None
            if len(probed) == len(engine_result.dynamic_terms) and np.array_equal(probed[position][0], matrix):
                harmonics = _harmonics(signal, probed[position][1], op.phase_offset)
            follows = harmonics is not None and len(charges) <= 1 and harmonics <= {next(iter(charges), 0)}
        terms.append(DeliveredTerm(charges, _foreign_node(signal), frozenset(labels[_reached(matrix, levels)].tolist()),
                                   follows))
    support = (min(lo for lo, _ in windows), max(hi for _, hi in windows)) if windows else None
    own_signal = ""
    if isinstance(op, ir.DriveOp) and chip.control_equipment is not None:
        drive = next((line for line in chip.control_equipment.lines if line.label == op.drive_label), None)
        if drive is not None and type(drive).signal is not BaseDrive.signal:
            own_signal = type(drive).__name__
    return Delivered(support, tuple(terms), own_signal)


@dataclass(frozen=True)
class Piece:
    """A time-disjoint part of a native gate.

    Attributes
    ----------
    start, stop : float
        Delivered span in ns, relative to the gate start.
    carrier : float or None
        Common carrier ``freq`` in GHz, or ``None`` for baseband pulses only.
    frame : str or None
        Common virtual-Z frame of the carrier pulses.
    covariant : bool
        Whether placement and frame shifts reuse the template propagator.
    reason : str
        Why reuse is not exact, or ``""``.
    matrix : numpy.ndarray
        Lab-frame template propagator over the span, in solver coordinates.
    """

    start: float
    stop: float
    carrier: float | None
    frame: str | None
    covariant: bool
    reason: str
    matrix: np.ndarray = field(repr=False)


def split_pieces(supports: list[tuple[float, float] | None]) -> list[tuple[float, float, list[int]]]:
    """Group pulses into time-disjoint pieces by their delivered spans; touching spans stay apart."""
    spans = sorted((lo, hi, index) for index, span in enumerate(supports) if span is not None
                   for lo, hi in [span])
    pieces: list[tuple[float, float, list[int]]] = []
    for start, stop, index in spans:
        if pieces and start < pieces[-1][1] - 1e-12:
            last = pieces[-1]
            pieces[-1] = (last[0], max(last[1], stop), last[2] + [index])
        else:
            pieces.append((start, stop, [index]))
    return pieces


def covariance(ops: list[Any], delivered: list[Delivered], piece_indices: list[int],
               model_reason: str = "") -> tuple[bool, str, float | None, str | None]:
    """Check the structural conditions for one piece.

    Returns ``(covariant, reason, carrier, frame)``.
    """
    if model_reason:
        return False, model_reason, None, None
    piece_ops = [ops[i] for i in piece_indices]
    if any(not isinstance(op, ir.DriveOp) for op in piece_ops):
        return False, "the piece holds an operation that is not a drive pulse", None, None
    carriers = {float(op.freq) for op in piece_ops if op.freq is not None}
    frames = {op.frame for op in piece_ops if op.freq is not None}
    if len(carriers) > 1:
        return False, f"pulses use {len(carriers)} carriers", None, None
    if len(frames) > 1:
        return False, f"carrier pulses use {len(frames)} frames", None, None
    carrier = next(iter(carriers)) if carriers else None
    frame = next(iter(frames)) if frames else None
    for index in piece_indices:
        op = ops[index]
        owner = delivered[index].own_signal
        if owner:
            return False, (f"the drive {op.drive_label!r} builds its signal with {owner}.signal(), so the gate set "
                           "cannot follow how its field moves with the start time"), carrier, frame
        for term in delivered[index].terms:
            if term.foreign:
                return False, f"a term of the pulse on {op.drive_label!r} uses {term.foreign}", carrier, frame
            if len(term.charges) > 1:
                return False, "a drive term mixes excitation numbers (no RWA)", carrier, frame
            charge = next(iter(term.charges)) if term.charges else 0
            # The placement conjugation turns every charge-q term at the piece's carrier, baseband pulses included.
            if op.freq is None and carrier is not None and charge != 0:
                return False, (f"a baseband term of the pulse on {op.drive_label!r} has charge {charge}, but the "
                               "piece has a carrier pulse, so only charge 0 keeps reuse"), carrier, frame
            if not term.follows_phase:
                how = "stays the same" if charge == 0 else f"turns by exp({charge}iθ)"
                return False, (f"the gate set cannot show that a charge-{charge} term of the pulse on "
                               f"{op.drive_label!r} {how} when the pulse phase turns by θ"), carrier, frame
    return True, "", carrier, frame


@dataclass
class GatePropagator:
    """Cached propagator of one calibrated native.

    Attributes
    ----------
    duration : float
        Span in ns from the gate start to the end of its authored window or
        of its last delivered field, whichever is later.
    pieces : list of Piece
        Time-disjoint pieces in time order.
    covariant : bool
        True when every piece is covariant.
    defect : float
        Largest unitarity defect of the piece propagators.
    """

    duration: float
    pieces: list[Piece]
    defect: float

    @property
    def covariant(self) -> bool:
        return all(piece.covariant for piece in self.pieces)

    def placed(self, start: float, offsets: dict[str, float], total: np.ndarray, idle: "Idle") -> np.ndarray:
        """Return the lab-frame propagator over ``[start, start + duration]``.

        Parameters
        ----------
        start : float
            Gate start time in ns.
        offsets : dict of str to float
            Accumulated virtual-Z angle of each frame at the gate start.
        total : numpy.ndarray
            Total excitation number of each solver coordinate.
        idle : Idle
            Free evolution between pieces.
        """
        if not self.covariant:
            raise ValueError("A non-covariant gate needs a solve for each placement.")
        result = np.eye(total.size, dtype=complex)
        cursor = 0.0
        for piece in self.pieces:
            if piece.start > cursor:
                result = idle(piece.start - cursor) @ result
            angle = offsets.get(piece.frame, 0.0) if piece.frame is not None else 0.0
            if piece.carrier is not None:
                angle = angle - 2 * np.pi * piece.carrier * start
            phases = np.exp(1j * angle * total)
            result = (phases[:, None] * piece.matrix * phases.conj()[None, :]) @ result
            cursor = piece.stop
        if self.duration > cursor:
            result = idle(self.duration - cursor) @ result
        return result


class Idle:
    """Free evolution ``exp(-i H0 Δt)`` of the static lab Hamiltonian in solver coordinates."""

    def __init__(self, problem: Any) -> None:
        engine_result = problem.engine_result
        hamiltonian = sum(term.coefficient * _dense(term.operator) for term in engine_result.static_terms)
        hamiltonian = np.asarray(hamiltonian, dtype=complex)
        self.frequencies, self.vectors = np.linalg.eigh(0.5 * (hamiltonian + hamiltonian.conj().T))
        self._cache: dict[float, np.ndarray] = {}

    def __call__(self, interval: float) -> np.ndarray:
        key = round(float(interval), 12)
        if key not in self._cache:
            self._cache[key] = (self.vectors * np.exp(-1j * self.frequencies * key)[None, :]) @ self.vectors.conj().T
        return self._cache[key]
