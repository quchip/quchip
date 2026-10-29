"""Least-squares calibration of a native gate on its computational block.

For knob values ``x``, free Z phases ``φ``, and a global phase ``γ``, the fit
minimizes ``|r|²`` with

.. math::

    r = \\left[\\frac{M - e^{i\\gamma} I}{\\sqrt{d + 1}},\\ \\frac{\\ell}{\\sqrt{d}}\\right],
    \\qquad M = U_\\mathrm{ideal}^\\dagger Z(\\varphi) V(x),

where ``V`` is the gate's computational block, ``ℓ`` the leaked amplitudes,
and ``Z(φ) = Π_q exp(iφ_q n_q)``. At ``γ = arg Tr M``,
``|r|² = 1 - F_avg + (d - |Tr M|)² / (d (d + 1))``. The second term is of
higher order in the gate error, so near a good gate ``|r|²`` and ``1 - F_avg``
agree to leading order; their optima can still differ slightly.

The knob columns of the Jacobian are forward differences with a fixed step in
the scaled coordinates, and the phase columns are analytic.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.linalg import logm
from scipy.optimize import least_squares, minimize

from quchip.gateset.ideals import pauli_components
from quchip.gateset.native import ResolvedKnob
from quchip.results.propagator import average_fidelity
from quchip.inverse_design.fit import _identifiability_receipt

if TYPE_CHECKING:
    from quchip.results.channel import ChannelResult

#: Finite-difference step of the scaled knobs.
DIFF_STEP = 1e-4


def error_generator(block: np.ndarray, ideal: np.ndarray) -> tuple[dict[str, float], float]:
    """Return the Pauli coefficients of ``i log U`` and the worst-case distance of ``U``.

    ``U`` is the unitary polar factor of ``block U_ideal†`` with its global
    phase removed: the error after the ideal gate, ``block ≈ U U_ideal``, as in
    cycle benchmarking. The error before the gate is ``U_ideal† U U_ideal``. A
    rotation by θ about a Pauli axis has the coefficient θ/2. The distance is
    ``sin(Δ/2)``, where Δ is the smallest arc that holds the eigenphases of
    ``U``, and 1 when Δ ≥ π.
    """
    left, _, right = np.linalg.svd(block @ ideal.conj().T)
    unitary = left @ right
    unitary = unitary * np.exp(-1j * np.angle(np.trace(unitary)))
    generator = 1j * logm(unitary)
    generator = 0.5 * (generator + generator.conj().T)
    phases = np.sort(np.angle(np.linalg.eigvals(unitary)))
    gaps = np.diff(np.concatenate([phases, phases[:1] + 2 * np.pi]))
    arc = 2 * np.pi - float(gaps.max())
    distance = 1.0 if arc >= np.pi else float(np.sin(arc / 2))
    return pauli_components(generator), distance


@dataclass(frozen=True)
class KnobResult:
    """One knob's calibration.

    Attributes
    ----------
    initial, value : float
        Template and calibrated values.
    lower, upper : float
        Bounds.
    scale : float
        Fit scale.
    at_bound : bool
        True when the value sits on a bound.
    change : float
        ``(value - initial) / scale``.
    """

    initial: float
    value: float
    lower: float
    upper: float
    scale: float
    at_bound: bool
    change: float


@dataclass(eq=False)
class GateCalibration:
    """Calibration report of one native gate.

    Attributes
    ----------
    name : str
        Gate name.
    qubits : tuple of str
        Gate qubits.
    knobs : dict of str to KnobResult
        Calibrated knobs.
    free_z : dict of str to float
        Post-gate Z phases ``φ_q``, applied as trailing ``vz(q, -φ_q)``.
    global_phase : float
        Fitted global phase ``γ``.
    fidelity : float
        Average gate fidelity ``F_avg`` of ``Z(φ) V`` against the ideal.
    leakage : float
        Mean leakage over the computational inputs.
    leakage_per_input : numpy.ndarray
        Leakage of each computational input.
    cost : float
        Final ``|r|²``.
    solves : int
        Propagator solves of the calibration.
    seed : str
        Provenance of the winning fit start.
    history : numpy.ndarray
        ``(solves, best |r|²)`` after each solve.
    block : numpy.ndarray
        Calibrated computational block ``Z(φ) V``.
    ideal : numpy.ndarray
        Ideal unitary.
    error_generator : dict of str to float
        Pauli coefficients, in rad, of ``i log U``, with ``U`` the unitary polar
        factor of ``Z(φ) V U_ideal†`` without its global phase: the error after
        the ideal gate.
    worst_case_distance : float
        ``sin(Δ/2)`` from the eigenphase arc of ``U``. For a unitary error it
        equals the diamond distance.
    identifiability : dict
        Jacobian rank, condition number, and weak directions of the final fit,
        and ``jacobian_step_change``: the relative change of the knob columns
        when the difference step shrinks tenfold.
    unitarity_defect : float
        ``max |U†U - I|`` over the computed columns, from the integrator.
    options, frame, cutoff
        Solver settings of the calibration.
    status : str
        ``"ok"``, ``"at_bound"``, ``"weak_direction"``, ``"unconverged"``
        (the fit used all its evaluations), ``"solver_floor"``, ``"poor_fit"``
        (infidelity above the calibration target at start time 0, at the
        worst start time of ``placement_infidelity``, or in the state of
        ``worst_spectators``), or ``"stale"`` (the chip changed after the
        calibration, or a native inside this composite was calibrated again
        after it).
    sequence : QuantumSequence
        Calibrated pulses with the trailing ``vz`` entries.
    spectators : dict of str to float
        Average gate fidelity of the gate's own block with each other gate-set
        qubit in dressed level 1, with the implemented corrections.
    ground_fidelity : float or None
        Average gate fidelity of the gate's own block with every spectator in
        level 0, with the implemented corrections. ``None`` without
        spectators in the calibration block.
    worst_spectators : tuple of (dict of str to int, float) or None
        Computational state of the other gate-set qubits, as dressed levels 0
        and 1, in which the gate's own block has its lowest average gate
        fidelity, and that fidelity, with the implemented corrections.
        ``None`` when the gate acts on every gate-set qubit.
    placement_infidelity : tuple of float or None
        Smallest and largest ``1 - F_avg`` over start times in [0, 100) ns, with
        the implemented corrections. A native whose pieces use different
        carriers, such as the echoed cross-resonance gate, turns each piece
        with its own carrier, so its error can change with the start time.
    repetitions : dict of int to float
        Infidelity of N applications against the ideal to the N.
    dependencies : dict
        Versions of the natives inside a composite gate.
    seconds : float
        Wall time of the fit.
    version : int
        Calibration count of this native in its gate set.
    cutoff_difference : float or None
        Largest change of the block or leakage at the next excitation cutoff,
        with the other gate-set qubits in level 0 and in level 1, for an
        automatic cutoff.
    block_qubits : tuple of str
        Qubits of the fitted block: the gate's qubits, then its spectators.
        ``block`` and ``ideal`` use this order, and ``fidelity`` and
        ``leakage`` refer to this block.
    """

    name: str
    qubits: tuple[str, ...]
    knobs: dict[str, KnobResult]
    free_z: dict[str, float]
    global_phase: float
    fidelity: float
    leakage: float
    leakage_per_input: np.ndarray
    cost: float
    solves: int
    seed: str
    history: np.ndarray
    block: np.ndarray
    ideal: np.ndarray
    error_generator: dict[str, float]
    worst_case_distance: float
    identifiability: dict[str, Any]
    unitarity_defect: float
    options: dict[str, Any]
    frame: Any
    cutoff: int | None
    status: str
    sequence: Any = None
    spectators: dict[str, float] = field(default_factory=dict)
    placement_infidelity: tuple[float, float] | None = None
    ground_fidelity: float | None = None
    worst_spectators: tuple[dict[str, int], float] | None = None
    repetitions: dict[int, float] = field(default_factory=dict)
    dependencies: dict[Any, int] = field(default_factory=dict)
    seconds: float = 0.0
    version: int = 0
    cutoff_difference: float | None = None
    block_qubits: tuple[str, ...] = ()

    @property
    def infidelity(self) -> float:
        """``1 - F_avg``."""
        return 1.0 - self.fidelity

    @property
    def channel(self) -> ChannelResult:
        """Channel ``ρ ↦ B ρ B†`` of the calibrated block ``B`` on the computational states of ``block_qubits``.

        Population that leaves the computational states is lost, so the
        channel decreases the trace by :attr:`leakage_per_input`. The record
        has no solves of its own: ``batch`` is ``None``.
        """
        from quchip.results.channel import ChannelResult

        block = np.asarray(self.block)
        qubits = self.block_qubits or self.qubits
        duration = 0.0 if self.sequence is None else float(self.sequence.total_duration)
        return ChannelResult(superoperator=np.kron(block.conj(), block),
                             labels=tuple(itertools.product((0, 1), repeat=len(qubits))), devices=tuple(qubits),
                             leakage=np.asarray(self.leakage_per_input), frame="dressed", frequencies={},
                             times=(0.0, duration), channels=(), batch=None)

    @property
    def eigenvalue_infidelity(self) -> float:
        """Eigenvalue infidelity of :attr:`channel` against the ideal; see :meth:`ChannelResult.eigenvalue_infidelity`.

        A spectral score, not an operational fidelity. It leaves out small
        rotations that only tilt the gate against preparation and
        measurement, so it is close to what gate-set tomography reports,
        while ``infidelity`` counts them.
        """
        return self.channel.eigenvalue_infidelity(self.ideal)

    def summary(self) -> str:
        """Return a text report."""
        spectators = [q for q in self.block_qubits if q not in self.qubits]
        joint = f" with spectators {', '.join(spectators)}" if spectators else ""
        # The eigenvalue score diagonalizes the d² × d² channel; a block of up to five qubits keeps that quick.
        spectral = (f", eigenvalue infidelity = {self.eigenvalue_infidelity:.3e}"
                    if len(self.block_qubits or self.qubits) <= 5 else "")
        lines = [
            f"{self.name} on {', '.join(self.qubits)}{joint}: 1 - F_avg = {self.infidelity:.3e}{spectral}, "
            f"leakage = {self.leakage:.2e}, status = {self.status}",
            f"  {self.solves} solves in {self.seconds:.1f} s from {self.seed}; |r|^2 = {self.cost:.3e}; "
            f"unitarity defect = {self.unitarity_defect:.1e}",
        ]
        for name, knob in self.knobs.items():
            flags = " (at bound)" if knob.at_bound else ""
            lines.append(f"  {name:>18s} = {knob.value:+.6g}  from {knob.initial:+.6g}, "
                         f"{knob.change:+.2f} scales{flags}")
        for qubit, phase in self.free_z.items():
            lines.append(f"  {'z ' + qubit:>18s} = {phase:+.6f} rad")
        largest = sorted(self.error_generator.items(), key=lambda item: -abs(item[1]))[:3]
        lines.append("  error generator after the gate (rad): " + ", ".join(f"{k} {v:+.2e}" for k, v in largest))
        lines.append(f"  worst-case distance = {self.worst_case_distance:.2e}; Jacobian condition = "
                     f"{self.identifiability.get('jacobian_condition_number', float('nan')):.2e}")
        if self.ground_fidelity is not None:
            lines.append(f"  spectators in |0>: 1 - F = {1 - self.ground_fidelity:.2e}")
        if self.spectators:
            lines.append("  spectators in |1>: " + ", ".join(
                f"{q} 1 - F = {1 - f:.2e}" for q, f in self.spectators.items()))
        if self.worst_spectators is not None:
            state, fidelity = self.worst_spectators
            lines.append("  worst state of the other qubits: " + ", ".join(f"{q}={v}" for q, v in state.items())
                         + f": 1 - F = {1 - fidelity:.2e}")
        if self.repetitions:
            lines.append("  repetitions: " + ", ".join(f"N={n} 1 - F = {v:.2e}"
                                                      for n, v in self.repetitions.items()))
        if self.placement_infidelity is not None:
            low, high = self.placement_infidelity
            lines.append(f"  start times in [0, 100) ns: 1 - F = {low:.2e} to {high:.2e}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"GateCalibration({self.name!r}, qubits={self.qubits}, 1-F={self.infidelity:.2e}, "
                f"status={self.status!r})")


@dataclass
class FitOutcome:
    """Result of :func:`fit_block` before the report adds gate-set checks."""

    values: np.ndarray
    phases: np.ndarray
    global_phase: float
    block: np.ndarray
    leaked: np.ndarray
    defect: float
    cost: float
    solves: int
    seed: str
    history: np.ndarray
    jacobian: np.ndarray
    names: list[str]
    seconds: float
    jacobian_step_change: float = 0.0
    converged: bool = True


class _Residual:
    """Residual vector with one solve per distinct knob vector."""

    def __init__(
        self,
        evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, float]],
        ideal: np.ndarray,
        z_bits: np.ndarray,
        knobs: Sequence[ResolvedKnob],
    ) -> None:
        self.evaluate = evaluate
        self.ideal_dagger = ideal.conj().T
        self.z_bits = z_bits
        self.d = ideal.shape[0]
        self.initial = np.array([knob.initial for knob in knobs], dtype=float)
        self.scale = np.array([knob.scale for knob in knobs], dtype=float)
        self.n_knobs = len(knobs)
        self.cache: dict[bytes, tuple[np.ndarray, np.ndarray, float]] = {}
        self.solves = 0
        self.best = np.inf
        self.history: list[tuple[int, float]] = []
        self.lower = np.full(self.n_knobs, -np.inf)
        self.upper = np.full(self.n_knobs, np.inf)
        self.step = DIFF_STEP

    def values(self, z: np.ndarray) -> np.ndarray:
        return self.initial + self.scale * z[: self.n_knobs]

    def block(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        key = np.asarray(x, dtype=float).tobytes()
        if key not in self.cache:
            self.cache[key] = self.evaluate(x)
            self.solves += 1
        return self.cache[key]

    def vector(self, block: np.ndarray, leaked: np.ndarray, phases: np.ndarray, gamma: float) -> np.ndarray:
        product = self.ideal_dagger @ (np.exp(1j * (self.z_bits @ phases))[:, None] * block)
        diff = (product - np.exp(1j * gamma) * np.eye(self.d)) / np.sqrt(self.d + 1)
        leak = leaked / np.sqrt(self.d)
        return np.concatenate([diff.real.ravel(), diff.imag.ravel(), leak.real.ravel(), leak.imag.ravel()])

    def __call__(self, z: np.ndarray) -> np.ndarray:
        n_free = self.n_knobs
        n_phases = self.z_bits.shape[1]
        before = self.solves
        block, leaked, _ = self.block(self.values(z))
        r = self.vector(block, leaked, z[n_free:n_free + n_phases], float(z[-1]))
        self._record(r, before)
        return r

    def _record(self, r: np.ndarray, before: int) -> None:
        self.best = min(self.best, float(r @ r))
        if self.solves != before:
            self.history.append((self.solves, self.best))
        elif self.history:
            self.history[-1] = (self.solves, self.best)

    def jacobian(self, z: np.ndarray, step: float | None = None) -> np.ndarray:
        """Return ``∂r/∂z``: forward differences for the knobs, exact columns for the phases."""
        step = self.step if step is None else step
        n_free = self.n_knobs
        n_phases = self.z_bits.shape[1]
        phases, gamma = z[n_free:n_free + n_phases], float(z[-1])
        block, leaked, _ = self.block(self.values(z))
        r0 = self.vector(block, leaked, phases, gamma)
        jacobian = np.empty((r0.size, z.size))
        for j in range(n_free):
            # Step toward the side with room, shortened to the feasible distance when both bounds are near.
            room_up, room_down = self.upper[j] - z[j], z[j] - self.lower[j]
            h = min(step, room_up) if room_up >= min(step, room_down) else -min(step, room_down)
            if h == 0.0:
                jacobian[:, j] = 0.0
                continue
            shifted = z.copy()
            shifted[j] += h
            before = self.solves
            b, ell, _ = self.block(self.values(shifted))
            r = self.vector(b, ell, phases, gamma)
            self._record(r, before)
            jacobian[:, j] = (r - r0) / h
        rotated = np.exp(1j * (self.z_bits @ phases))[:, None] * block
        zero_leak = np.zeros(2 * leaked.size)
        for k in range(n_phases):
            column = self.ideal_dagger @ (1j * self.z_bits[:, k][:, None] * rotated) / np.sqrt(self.d + 1)
            jacobian[:, n_free + k] = np.concatenate([column.real.ravel(), column.imag.ravel(), zero_leak])
        column = -1j * np.exp(1j * gamma) * np.eye(self.d) / np.sqrt(self.d + 1)
        jacobian[:, -1] = np.concatenate([column.real.ravel(), column.imag.ravel(), zero_leak])
        return jacobian

    def phase_start(self, block: np.ndarray, leaked: np.ndarray) -> np.ndarray:
        """Fit only the phases at a fixed block; costs no solve."""
        n_phases = self.z_bits.shape[1]
        product = self.ideal_dagger @ block
        start = np.zeros(n_phases + 1)
        start[-1] = float(np.angle(np.trace(product)))

        def residual(p: np.ndarray) -> np.ndarray:
            return self.vector(block, leaked, p[:n_phases], float(p[-1]))

        best = least_squares(residual, start, method="lm", xtol=1e-14, ftol=1e-14)
        for trial in itertools.product((0.0, np.pi / 2, np.pi, -np.pi / 2), repeat=min(n_phases, 2)):
            if not any(trial):
                continue
            guess = start.copy()
            guess[: len(trial)] = trial
            candidate = least_squares(residual, guess, method="lm", xtol=1e-14, ftol=1e-14)
            if candidate.cost < best.cost:
                best = candidate
        return best.x


def fit_block(
    evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, float]],
    ideal: np.ndarray,
    z_bits: np.ndarray,
    knobs: Sequence[ResolvedKnob],
    *,
    starts: Sequence[tuple[np.ndarray, str]],
    max_nfev: int,
    explore: bool,
    phase_names: Sequence[str],
) -> FitOutcome:
    """Fit knobs, free Z phases, and the global phase to the ideal.

    Parameters
    ----------
    evaluate : callable
        Knob values -> ``(block, leaked, unitarity_defect)`` of the gate.
    ideal : numpy.ndarray
        Ideal unitary.
    z_bits : numpy.ndarray
        ``(2^n, n_free_z)`` bit of each free-Z qubit in each computational state.
    knobs : sequence of ResolvedKnob
        Resolved knobs in fit order.
    starts : sequence of (values, provenance)
        Knob vectors to start from.
    max_nfev : int
        Bound on residual evaluations of each fit, finite differences excluded.
        ``0`` skips the knob search: the best start keeps its knobs, and only
        the phases are fitted.
    explore : bool
        Run a Nelder-Mead stage on ``|r|²`` before the fits.
    phase_names : sequence of str
        Labels of the free-Z qubits.
    """
    began = time.perf_counter()
    residual = _Residual(evaluate, ideal, z_bits, knobs)
    n_knobs = len(knobs)
    n_phases = z_bits.shape[1]
    lower = np.array([knob.lower for knob in knobs], dtype=float)
    upper = np.array([knob.upper for knob in knobs], dtype=float)
    initial, scale = residual.initial, residual.scale
    lb = np.concatenate([(lower - initial) / scale, np.full(n_phases + 1, -np.inf)])
    ub = np.concatenate([(upper - initial) / scale, np.full(n_phases + 1, np.inf)])

    def start_vector(values: np.ndarray) -> np.ndarray:
        u = (np.asarray(values, dtype=float) - initial) / scale
        block, leaked, _ = residual.block(residual.values(np.concatenate([u, np.zeros(n_phases + 1)])))
        return np.concatenate([u, residual.phase_start(block, leaked)])

    residual.lower = lb[:n_knobs]
    residual.upper = ub[:n_knobs]
    vectors = [(start_vector(values), provenance) for values, provenance in starts]
    if explore and n_knobs and max_nfev:
        best_start = min(vectors, key=lambda item: float(np.sum(residual(item[0]) ** 2)))
        found = minimize(lambda z: float(np.sum(residual(z) ** 2)), best_start[0], method="Nelder-Mead",
                         bounds=list(zip(lb, ub)), options={"maxfev": 40 * (n_knobs + 1), "xatol": 1e-6,
                                                            "fatol": 1e-12})
        values = residual.values(found.x)
        vectors.append((start_vector(values), "explore"))
    best: tuple[np.ndarray, float, bool, str] | None = None
    for vector, provenance in vectors:
        vector = np.clip(vector, lb, ub)
        if max_nfev == 0:
            r = residual(vector)
            x, cost, converged = vector, 0.5 * float(r @ r), True
        else:
            result = least_squares(residual, vector, jac=residual.jacobian, method="trf", bounds=(lb, ub),
                                   x_scale=1.0, xtol=1e-12, ftol=1e-14, gtol=1e-14, max_nfev=max_nfev)
            x, cost, converged = result.x, float(result.cost), bool(result.status != 0)
        if best is None or cost < best[1]:
            best = (x, cost, converged, provenance)
    assert best is not None
    x, cost, converged, provenance = best
    values = residual.values(x)
    block, leaked, defect = residual.block(values)
    phases = x[n_knobs:n_knobs + n_phases]
    names = [knob.name for knob in knobs] + [f"z {name}" for name in phase_names] + ["global phase"]
    jacobian = residual.jacobian(x)
    change = 0.0
    if n_knobs:
        finer = residual.jacobian(x, step=residual.step / 10)
        norms = np.linalg.norm(jacobian[:, :n_knobs], axis=0)
        differences = np.linalg.norm(finer[:, :n_knobs] - jacobian[:, :n_knobs], axis=0)
        change = float(np.max(differences / np.maximum(norms, np.finfo(float).tiny)))
    history = np.array(residual.history, dtype=float).reshape(-1, 2)
    return FitOutcome(values=values, phases=np.asarray(phases, dtype=float), global_phase=float(x[-1]),
                      block=block, leaked=leaked, defect=defect, cost=2 * cost, solves=residual.solves,
                      seed=provenance, history=history, jacobian=jacobian, names=names,
                      seconds=time.perf_counter() - began, jacobian_step_change=change, converged=converged)


def report(
    outcome: FitOutcome,
    *,
    name: str,
    qubits: tuple[str, ...],
    knobs: Sequence[ResolvedKnob],
    ideal: np.ndarray,
    z_bits: np.ndarray,
    phase_names: Sequence[str],
    options: dict[str, Any],
    frame: Any,
    cutoff: int | None,
    condition_limit: float = 1e6,
) -> GateCalibration:
    """Build the report of a fit."""
    knob_results = {}
    at_bound = False
    for knob, value in zip(knobs, outcome.values):
        tolerance = 1e-6 * knob.scale
        hit = value - knob.lower < tolerance or knob.upper - value < tolerance
        at_bound = at_bound or hit
        knob_results[knob.name] = KnobResult(initial=knob.initial, value=float(value), lower=knob.lower,
                                             upper=knob.upper, scale=knob.scale, at_bound=bool(hit),
                                             change=float((value - knob.initial) / knob.scale))
    z = np.exp(1j * (z_bits @ outcome.phases))
    block = z[:, None] * outcome.block
    generator, distance = error_generator(block, ideal)
    leakage = np.real(np.sum(np.conj(outcome.leaked) * outcome.leaked, axis=0))
    receipt = _identifiability_receipt(outcome.jacobian, outcome.names, np.ones(len(outcome.names)))
    receipt["jacobian_step_change"] = outcome.jacobian_step_change
    fidelity = float(average_fidelity(block, ideal))
    if at_bound:
        status = "at_bound"
    elif receipt["rank_deficient"] or receipt["jacobian_condition_number"] > condition_limit:
        status = "weak_direction"
    elif not outcome.converged:
        status = "unconverged"
    elif 1.0 - fidelity < 10 * outcome.defect:
        status = "solver_floor"
    else:
        status = "ok"
    return GateCalibration(
        name=name, qubits=qubits, knobs=knob_results,
        free_z={q: float(np.angle(np.exp(1j * phi))) for q, phi in zip(phase_names, outcome.phases)},
        global_phase=float(np.angle(np.exp(1j * outcome.global_phase))), fidelity=fidelity,
        leakage=float(np.mean(leakage)), leakage_per_input=leakage, cost=outcome.cost, solves=outcome.solves,
        seed=outcome.seed, history=outcome.history, block=block, ideal=ideal, error_generator=generator,
        worst_case_distance=distance, identifiability=receipt, unitarity_defect=float(outcome.defect),
        options=dict(options), frame=frame, cutoff=cutoff, status=status, seconds=outcome.seconds,
    )
