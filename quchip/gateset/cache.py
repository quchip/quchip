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
5. every term is built from the pulse's envelope, carriers, and the standard
   signal algebra, with no coefficient of absolute time.

Condition 2 lets each pulse be lowered alone: the delivered field of a piece
is the sum of its pulses' delivered fields, and each moves with its pulse.
Under conditions 2, 3, and 5, a carrier pulse that starts ``τ`` later gives
the field of the same pulse with its phase turned by ``-2π f_c τ``, shifted
by ``τ``, so condition 4 also covers the start time.

The gate set proves condition 4 on the structure of each delivered
coefficient. It lowers a carrier pulse a second time with its phase turned by
a small step and walks both signal trees together. A node whose scalar turns
by ``exp(ik·step)`` carries phase harmonic ``k``, and the node types combine
harmonics: a sum takes their union, a product adds them, a conjugate negates
them, a real or imaginary part takes both signs, and an integer power adds
them that many times. A term passes when its only harmonic is its charge, so
the result holds for every phase, amplitude, and envelope. Any other change
between the two trees fails the term. Signal bands cannot show the harmonic:
at zero delivered frequency both sidebands of a pulse fall on one band.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from functools import reduce
from typing import Any

import jax.tree_util as jtu
import numpy as np

from quchip.engine import ir
from quchip.engine.ir import signal_children, signal_window_bounds
from quchip.results.propagator import solver_levels

_CHARGE_ATOL = 1e-12
_TURN_ATOL = 1e-9
_PHASE_STEP = 2 * np.pi / 4099  # rad; exp(ik·step) differs for every |k| <= 2049, so a turn names its harmonic
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
    """The delivered terms of one pulse and the hull of their windows (``None`` when nothing is delivered)."""

    support: tuple[float, float] | None
    terms: tuple[DeliveredTerm, ...]


def _lowered(chip: Any, op: Any) -> Any:
    from quchip.engine.problem import build_solve_problem

    stop = float(op.start_time) + float(op.envelope.duration)
    return build_solve_problem(chip, [op], np.array([0.0, max(stop, 1.0)]), states="final", frame="lab",
                               dissipation=False, _product_space=True).engine_result


def _turn(before: Any, after: Any) -> int | None:
    """Return ``k`` when a node scalar turned by ``exp(ik·step)`` between the two lowerings, or ``None``."""
    before, after = complex(before), complex(after)
    if before == after:
        return 0
    if before == 0 or abs(abs(after / before) - 1.0) > _TURN_ATOL:
        return None
    angle = float(np.angle(after / before))
    k = round(angle / _PHASE_STEP)
    return k if abs(angle - k * _PHASE_STEP) <= _TURN_ATOL else None


def _same(before: Any, after: Any) -> bool:
    """Whether two node fields hold the same values; the engine rebuilds envelopes, so compare pytree leaves."""
    if before is after:
        return True
    left, left_tree = jtu.tree_flatten(before)
    right, right_tree = jtu.tree_flatten(after)
    return left_tree == right_tree and all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(left, right))


def _harmonics(before: Any, after: Any) -> frozenset[int] | None:
    """Return the harmonics of the pulse phase in a delivered signal, or ``None`` when its structure does not
    prove them. ``before`` and ``after`` are the signal lowered at the pulse phase and at the phase plus the step.
    """
    if type(before) is not type(after) or not isinstance(before, _PULSE_LOCAL_NODES):
        return None
    if isinstance(before, ir.PolarScale):
        own = _turn(before.amplitude * np.exp(1j * before.theta), after.amplitude * np.exp(1j * after.theta))
    elif isinstance(before, ir.Scale):
        own = _turn(before.factor, after.factor)
    elif isinstance(before, ir.Constant):
        own = _turn(before.value, after.value)
    else:
        own = 0
    scalars = {"amplitude", "theta", "factor", "value"}
    children = getattr(type(before), "_signal_child_fields", ())
    if own is None or not all(_same(getattr(before, item.name), getattr(after, item.name))
                              for item in fields(before) if item.name not in scalars and item.name not in children):
        return None
    pairs = list(zip(signal_children(before), signal_children(after)))
    if len(pairs) != len(signal_children(before)) or len(pairs) != len(signal_children(after)):
        return None
    parts = [_harmonics(a, b) for a, b in pairs]
    if any(part is None for part in parts):
        return None
    sets: list[frozenset[int]] = [part for part in parts if part is not None]

    def plus(left: frozenset[int], right: frozenset[int]) -> frozenset[int]:
        return frozenset(a + b for a in left for b in right)

    if isinstance(before, ir.Add):
        combined = frozenset().union(*sets)
    elif isinstance(before, ir.Multiply):
        combined = reduce(plus, sets, frozenset({0}))
    elif isinstance(before, ir.Conjugate):
        combined = frozenset(-h for h in sets[0])
    elif isinstance(before, (ir.RealPart, ir.ImagPart)):
        combined = sets[0] | frozenset(-h for h in sets[0])
    elif isinstance(before, ir.SignalPower):
        exponent = float(before.exponent)
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

    A carrier pulse is lowered a second time with its phase turned by a small
    step, which shows the phase harmonics of each term. Charges, windows, and
    harmonics depend on the pulse's structure, not on its amplitude or phase.
    """
    engine_result = _lowered(chip, op)
    levels = solver_levels(engine_result)
    total = levels.sum(axis=1)
    labels = np.array([device.label for device in chip.devices])
    turned = None
    if isinstance(op, ir.DriveOp) and op.freq is not None:
        turned = _lowered(chip, replace(op, phase_offset=float(op.phase_offset) + _PHASE_STEP)).dynamic_terms
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
        if turned is not None:
            harmonics = None
            if len(turned) == len(engine_result.dynamic_terms) and np.array_equal(_dense(turned[position].operator),
                                                                                    matrix):
                harmonics = _harmonics(signal, turned[position].time_dependence.signal)
            follows = harmonics is not None and len(charges) <= 1 and harmonics <= {next(iter(charges), 0)}
        terms.append(DeliveredTerm(charges, _foreign_node(signal), frozenset(labels[_reached(matrix, levels)].tolist()),
                                   follows))
    support = (min(lo for lo, _ in windows), max(hi for _, hi in windows)) if windows else None
    return Delivered(support, tuple(terms))


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
                how = "changes" if charge == 0 else f"does not turn by exp({charge}iθ)"
                return False, (f"a charge-{charge} term of the pulse on {op.drive_label!r} {how} when the pulse "
                               "phase turns by θ"), carrier, frame
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
