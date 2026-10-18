"""Gate propagators that are reused at any start time and virtual-Z frame.

A native splits into time-disjoint pieces at every instant that no pulse
spans. A piece is covariant when the conditions below hold, and then its
propagator at start time ``τ`` and frame offset ``F`` follows from the
template without a new solve:

.. math::

    U(\\tau + t_1, \\tau + t_0) = C\\, U(t_1, t_0)\\, C^\\dagger,
    \\qquad C = e^{i(F - 2\\pi f_c \\tau) N},

with ``N`` the total excitation number and ``f_c`` the piece's carrier
``freq``. The start-referenced ``detuning`` does not enter. The conditions are
structural and use the lab-frame terms of the solve:

1. every static term has N-charge 0;
2. every dynamic term has one N-charge q, and each of its signal bands sits at
   ``-q (f_c + δ)`` for a pulse of the piece with detuning δ, or at 0 for q = 0;
3. every carrier pulse of the piece has the same ``freq`` and frame;
4. every collapse operator has one N-charge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quchip.engine.ir import signal_window_bounds
from quchip.results.propagator import solver_levels

_CHARGE_ATOL = 1e-12
_FREQUENCY_RTOL = 1e-9


def _dense(term_operator: Any) -> np.ndarray:
    return np.asarray(term_operator.to_dense())


def _charges(matrix: np.ndarray, total: np.ndarray) -> set[int]:
    rows, cols = np.nonzero(np.abs(matrix) > _CHARGE_ATOL * max(1.0, float(np.abs(matrix).max(initial=0.0))))
    return set((total[rows] - total[cols]).tolist())


@dataclass(frozen=True)
class Piece:
    """A time-disjoint part of a native gate.

    Attributes
    ----------
    start, stop : float
        Span in ns, relative to the gate start.
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


def split_pieces(ops: list[Any]) -> list[tuple[float, float, list[int]]]:
    """Group ops into time-disjoint pieces; touching windows stay apart."""
    spans = sorted(
        ((float(op.start_time), float(op.start_time) + float(op.envelope.duration), index)
         for index, op in enumerate(ops)),
    )
    pieces: list[tuple[float, float, list[int]]] = []
    for start, stop, index in spans:
        if pieces and start < pieces[-1][1] - 1e-12:
            last = pieces[-1]
            pieces[-1] = (last[0], max(last[1], stop), last[2] + [index])
        else:
            pieces.append((start, stop, [index]))
    return pieces


def covariance(problem: Any, ops: list[Any], piece_indices: list[int], span: tuple[float, float]) -> tuple[
        bool, str, float | None, str | None]:
    """Check the structural conditions for one piece on a lab-frame problem.

    Returns ``(covariant, reason, carrier, frame)``.
    """
    engine_result = problem.engine_result
    total = solver_levels(engine_result).sum(axis=1)
    for term in engine_result.static_terms:
        if _charges(_dense(term.operator), total) - {0}:
            return False, "a static term changes the excitation number (no RWA)", None, None
    for term in engine_result.collapse_terms:
        if len(_charges(_dense(term.operator), total)) > 1:
            return False, f"collapse operator {term.source}.{term.channel} mixes excitation numbers", None, None
    piece_ops = [ops[i] for i in piece_indices]
    carriers = {float(op.freq) for op in piece_ops if op.freq is not None}
    frames = {op.frame for op in piece_ops if op.freq is not None}
    if len(carriers) > 1:
        return False, f"pulses use {len(carriers)} carriers", None, None
    if len(frames) > 1:
        return False, f"carrier pulses use {len(frames)} frames", None, None
    carrier = next(iter(carriers)) if carriers else None
    frame = next(iter(frames)) if frames else None
    detunings = {0.0 if op.detuning is None else float(op.detuning) for op in piece_ops if op.freq is not None}
    for term in engine_result.dynamic_terms:
        windows = signal_window_bounds(term.time_dependence.signal)
        if windows and not any(float(lo) < span[1] - 1e-12 and float(hi) > span[0] + 1e-12 for lo, hi in windows):
            continue
        charges = _charges(_dense(term.operator), total)
        if len(charges) > 1:
            return False, "a drive term mixes excitation numbers (no RWA)", carrier, frame
        charge = next(iter(charges)) if charges else 0
        bands = [float(band.freq) / (2 * np.pi) for band in term.time_dependence.signal.bands()]
        if charge == 0:
            if any(abs(nu) > 1e-12 for nu in bands):
                return False, "an excitation-preserving term oscillates", carrier, frame
            continue
        if carrier is None:
            return False, "a baseband pulse drives an excitation-changing term", carrier, frame
        allowed = [-charge * (carrier + delta) for delta in detunings]
        for nu in bands:
            if not any(abs(nu - a) <= _FREQUENCY_RTOL * max(1.0, abs(a)) for a in allowed):
                return False, f"a charge-{charge} term oscillates at {nu:.6g} GHz, not at its carrier", carrier, frame
    return True, "", carrier, frame


@dataclass
class GatePropagator:
    """Cached propagator of one calibrated native.

    Attributes
    ----------
    duration : float
        Gate duration in ns.
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
