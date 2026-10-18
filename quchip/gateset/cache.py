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
4. every delivered term of a pulse has one N-charge q, and each of its signal
   bands sits at ``-q (f_c + δ)`` for the piece's carrier ``f_c`` (0 when the
   piece has no carrier) and the pulse's detuning δ, so a baseband pulse in a
   carrier piece needs charge 0;
5. every term is built from the pulse's envelope, carriers, and the standard
   signal algebra, with no coefficient of absolute time.

Condition 2 lets each pulse be lowered alone: the delivered field of a piece
is the sum of its pulses' delivered fields, and each moves with its pulse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quchip.engine import ir
from quchip.engine.ir import signal_children, signal_window_bounds
from quchip.results.propagator import solver_levels

_CHARGE_ATOL = 1e-12
_FREQUENCY_RTOL = 1e-9
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
    """Return a mask of the devices whose level a term changes or on whose level its diagonal depends."""
    magnitude = np.abs(matrix)
    tolerance = _CHARGE_ATOL * max(1.0, float(magnitude.max(initial=0.0)))
    rows, cols = np.nonzero(magnitude > tolerance)
    reached = np.any(levels[rows] != levels[cols], axis=0)
    diagonal = np.diag(matrix)
    for device in np.flatnonzero(~reached):
        first: dict[tuple[int, ...], complex] = {}
        for key, value in zip(map(tuple, np.delete(levels, device, axis=1)), diagonal):
            if abs(value - first.setdefault(key, value)) > tolerance:
                reached[device] = True
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
    """One delivered term of a pulse: N-charges, signal bands in GHz, a foreign signal node, and the devices
    whose levels it changes or on whose levels it depends."""

    charges: frozenset[int]
    bands: tuple[float, ...]
    foreign: str
    devices: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Delivered:
    """The delivered terms of one pulse and the hull of their windows (``None`` when nothing is delivered)."""

    support: tuple[float, float] | None
    terms: tuple[DeliveredTerm, ...]


def lower_pulse(chip: Any, op: Any) -> Delivered:
    """Lower one scheduled op alone and return its delivered terms."""
    from quchip.engine.problem import build_solve_problem

    stop = float(op.start_time) + float(op.envelope.duration)
    problem = build_solve_problem(chip, [op], np.array([0.0, max(stop, 1.0)]), states="final", frame="lab",
                                  dissipation=False, _product_space=True)
    levels = solver_levels(problem.engine_result)
    total = levels.sum(axis=1)
    labels = np.array([device.label for device in chip.devices])
    windows: list[tuple[float, float]] = []
    terms = []
    for term in problem.engine_result.dynamic_terms:
        signal = term.time_dependence.signal
        bounds = signal_window_bounds(signal)
        if not bounds:
            raise ValueError(f"A delivered field of the pulse on {op.drive_label!r} at {float(op.start_time):g} ns "
                             "has no time window, so the gate set cannot bound the gate.")
        windows.extend((float(lo), float(hi)) for lo, hi in bounds)
        matrix = _dense(term.operator)
        terms.append(DeliveredTerm(_charges(matrix, total),
                                   tuple(float(band.freq) / (2 * np.pi) for band in signal.bands()),
                                   _foreign_node(signal), frozenset(labels[_reached(matrix, levels)].tolist())))
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
        # The placement conjugation rotates every charge-q term at the piece's carrier, baseband pulses included.
        tone = (0.0 if carrier is None else carrier) + (0.0 if op.detuning is None else float(op.detuning))
        for term in delivered[index].terms:
            if term.foreign:
                return False, f"a term of the pulse on {op.drive_label!r} uses {term.foreign}", carrier, frame
            if len(term.charges) > 1:
                return False, "a drive term mixes excitation numbers (no RWA)", carrier, frame
            charge = next(iter(term.charges)) if term.charges else 0
            allowed = -charge * tone
            for nu in term.bands:
                if abs(nu - allowed) > _FREQUENCY_RTOL * max(1.0, abs(allowed)):
                    return False, (f"a charge-{charge} term of the pulse on {op.drive_label!r} oscillates at "
                                   f"{nu:.6g} GHz, not at {allowed:.6g} GHz"), carrier, frame
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
