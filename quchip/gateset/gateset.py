"""Gate sets: calibrated natives, virtual RZ, decomposition rules, and circuits."""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from math import prod
from typing import TYPE_CHECKING, Any

import numpy as np

from quchip.gateset.cache import (Delivered, GatePropagator, Idle, Piece, covariance, lower_pulse, model_check,
                                  signal_check,
                                  split_pieces)
from quchip.gateset.calibration import GateCalibration, fit_block, report
from quchip.gateset.ideals import IDEALS, canonical_name, embed, ideal_unitary, rule_table
from quchip.gateset.native import NativeGate, bind_pulses
from quchip.gateset.blocks import computational_states, project_computational
from quchip.results.propagator import average_fidelity, solver_levels
from quchip.utils.labeling import resolve_label

if TYPE_CHECKING:
    from quchip.chip.chip import Chip
    from quchip.control.sequence import QuantumSequence

#: Integration settings of gate-set solves (Section "Gate-set solver settings" of the guide).
DEFAULT_OPTIONS = {"atol": 1e-12, "rtol": 1e-10, "nsteps": 10**7}


def _tight_options(backend: Any) -> dict[str, Any]:
    """Return the default gate-set tolerances in the form of the backend that solves."""
    import sys

    from quchip.backend import _coerce_backend

    resolved = _coerce_backend(backend)
    module = sys.modules.get("quchip.backend.dynamiqs")
    if module is not None and isinstance(resolved, module.DynamiqsBackend):
        import dynamiqs as dq

        return {"method": dq.method.Dopri8(rtol=DEFAULT_OPTIONS["rtol"], atol=DEFAULT_OPTIONS["atol"],
                                           max_steps=DEFAULT_OPTIONS["nsteps"])}
    return dict(DEFAULT_OPTIONS)

#: Full product spaces up to this size skip the automatic excitation cutoff.
AUTO_CUTOFF_DIMENSION = 64


def _defect(propagator: Any) -> float:
    defect = propagator.unitarity_defect
    return 0.0 if defect is None else float(defect)


def _static_problem(chip: "Chip", dissipation: bool = False) -> Any:
    """Return the lab-frame problem of the chip's own terms on its product space."""
    from quchip.engine.problem import build_solve_problem

    return build_solve_problem(chip, [], np.array([0.0, 1.0]), states="final", frame="lab",
                               dissipation=dissipation, _product_space=True)


class _Unresolved(KeyError):
    """No native or rule gives an instruction."""


@dataclass
class CircuitUnitary:
    """Propagator of a circuit composed from cached gates.

    Attributes
    ----------
    full : numpy.ndarray
        Lab-frame propagator ``U(T, 0)`` in solver coordinates.
    computational : numpy.ndarray
        Circuit unitary on the computational states, with the virtual-Z
        frame correction ``Π_q RZ_q(-F_q)`` applied.
    leakage : numpy.ndarray
        Leakage of each computational input.
    frame_offsets : dict of str to float
        Accumulated ``vz`` angle ``F_q`` of each qubit at the end.
    duration : float
        Circuit duration in ns.
    qubits : tuple of str
        Qubit order of ``computational``.
    ideal : numpy.ndarray
        Ideal circuit unitary in the same order.
    solves : int
        Propagator solves that composition needed (overlapping or
        non-covariant gates).
    """

    full: np.ndarray
    computational: np.ndarray
    leakage: np.ndarray
    frame_offsets: dict[str, float]
    duration: float
    qubits: tuple[str, ...]
    ideal: np.ndarray
    solves: int = 0

    def fidelity(self, ideal: Any = None) -> float:
        """Return the average gate fidelity against the ideal circuit, global phase removed.

        Parameters
        ----------
        ideal : array_like, optional
            Reference unitary in the order of ``qubits``. The default is the
            ideal circuit.
        """
        target = self.ideal if ideal is None else np.asarray(ideal, dtype=complex)
        return float(average_fidelity(self.computational, target))


@dataclass
class CircuitRun:
    """Result of :meth:`GateSet.simulate`.

    Attributes
    ----------
    state : numpy.ndarray
        Final computational state, a ket or a density matrix, after the frame
        correction.
    probabilities : dict of str to float
        Population of each computational bit string.
    leakage : float
        Population outside the computational states.
    result : object
        The underlying propagator or simulation result.
    """

    state: np.ndarray
    probabilities: dict[str, float]
    leakage: float
    result: Any = field(repr=False, default=None)


class GateSet:
    """Calibrated native gates of a chip and the circuits built from them.

    Gate-set solves integrate in the rotating frame with tight tolerances and
    return lab-frame propagators, so the integration frame changes only cost
    and accuracy. Gate matrices use dressed computational states: each
    gate-set qubit in dressed level 0 or 1, every other device in dressed
    level 0, the first qubit most significant, and the phases of
    ``chip.state()``. RZ is virtual on every qubit.

    Parameters
    ----------
    chip : Chip
        Chip whose natives the gate set holds.
    qubits : sequence of str or device
        Computational qubits in matrix order.
    frame : FrameSpec, default "rotating"
        Integration frame of gate-set solves.
    options : dict, optional
        Solver options. The default is ``atol=1e-12`` and ``rtol=1e-10`` in
        the form of the backend that solves: QuTiP options, or a dynamiqs
        ``Dopri8`` method.
    backend : object or str, optional
        Backend of gate-set solves.
    space : "auto", ExcitationCutoff, or None, default "auto"
        Solver space. ``"auto"`` keeps the chip's policy when it has one.
        Otherwise it keeps the full product space up to 64 states and above
        that picks ``ExcitationCutoff(n + 1)`` for ``n`` qubits, raising the
        cutoff until the calibrated gates change by less than
        ``cutoff_tolerance`` at the next cutoff, with a warning. ``None``
        forces the full product space.
    cutoff_tolerance : float, default 1e-6
        Largest change of a gate's computational amplitudes or leakage
        accepted when the cutoff is raised by one.
    spectators : {"coupled", "all", "none"}, default "coupled"
        Other gate-set qubits in each calibration block. The fit then targets
        ``U ⊗ I`` on the joint block, with a free virtual Z on each spectator,
        so static ZZ and drive crosstalk during the gate enter the objective.
        ``"coupled"`` takes the gate-set qubits that a coupling path reaches
        without passing another gate-set qubit, ``"all"`` takes every other
        gate-set qubit, and ``"none"`` fits with the others in level 0.
    """

    def __init__(
        self,
        chip: "Chip",
        qubits: Sequence[Any],
        *,
        frame: Any = "rotating",
        options: Mapping[str, Any] | None = None,
        backend: Any = None,
        space: Any = "auto",
        cutoff_tolerance: float = 1e-6,
        spectators: str = "coupled",
    ) -> None:
        self.chip = chip
        self.qubits = tuple(resolve_label(qubit) for qubit in qubits)
        unknown = [label for label in self.qubits if label not in chip.device_map]
        if unknown:
            raise ValueError(f"Unknown qubits {unknown}.")
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError(f"Gate-set qubits must be distinct, got {list(self.qubits)}.")
        self.frame = frame
        self._default_options: dict[Any, dict[str, Any]] = {}
        self.options = options
        self.backend = backend
        self.cutoff_tolerance = float(cutoff_tolerance)
        if spectators not in ("coupled", "all", "none"):
            raise ValueError(f"spectators must be 'coupled', 'all', or 'none'; got {spectators!r}.")
        self.spectators = spectators
        self._space_request = space
        self._natives: dict[tuple[str, tuple[str, ...]], NativeGate] = {}
        self._calibrations: dict[tuple[str, tuple[str, ...]], GateCalibration] = {}
        self._entries: dict[tuple[str, tuple[str, ...]], GatePropagator] = {}
        self._user_rules: dict[str, list[tuple[int, Callable[..., Any]]]] = {}
        self._versions: dict[tuple[str, tuple[str, ...]], int] = {}
        self._calibrating: list[tuple[str, tuple[str, ...]]] = []
        self._used: list[set[tuple[str, tuple[str, ...]]]] = []
        self._solver_chip: Any = None
        self._cutoff_checked = False
        self._lab: Any = None
        self._idle: Idle | None = None
        self._placed: dict[Any, np.ndarray] = {}
        self._delivered: dict[Any, Delivered] = {}
        self._model_reason: str | None = None
        self._warm: dict[tuple[str, tuple[str, ...]], dict[str, float]] = {}

    # ------------------------------------------------------------------ setup

    @property
    def options(self) -> dict[str, Any]:
        """Solver options of gate-set solves."""
        if self._options is not None:
            return self._options
        backend = self.chip.backend if self.backend is None else self.backend
        # One options object per backend, so that a backend that caches compiled solves can reuse them.
        key = backend if isinstance(backend, str) else id(backend)
        if key not in self._default_options:
            self._default_options[key] = _tight_options(backend)
        return self._default_options[key]

    @options.setter
    def options(self, value: Mapping[str, Any] | None) -> None:
        self._options = None if value is None else dict(value)

    @classmethod
    def standard(
        cls,
        chip: "Chip",
        qubits: Sequence[Any],
        *,
        pairs: Iterable[tuple[Any, Any]] = (),
        sx_duration: float = 40.0,
        x_duration: float | None = None,
        cr_duration: float | None = None,
        echo: bool = True,
        **options: Any,
    ) -> "GateSet":
        """Return a gate set with DRAG SX and X on each qubit and a ZX90 on each directed pair.

        Parameters
        ----------
        chip : Chip
            Chip with one charge line per qubit.
        qubits : sequence of str or device
            Computational qubits.
        pairs : iterable of (control, target), default ()
            Directed cross-resonance pairs.
        sx_duration : float, default 40.0
            SX pulse duration in ns.
        x_duration : float or None, default None
            X pulse duration in ns. The default is ``sx_duration``.
        cr_duration : float or None
            Total CR tone duration in ns. Required with pairs.
        echo : bool, default True
            Echo each CR with X pulses on the control.
        **options
            Passed to :class:`GateSet`.
        """
        from quchip.gateset.builtins import cross_resonance, drag_sx

        pairs = list(pairs)
        if pairs and cr_duration is None:
            raise ValueError("cr_duration is required with pairs; the right value depends on g and the detuning.")
        gates = cls(chip, qubits, **options)
        for qubit in gates.qubits:
            gates.add(drag_sx(chip, qubit, duration=sx_duration))
            gates.add(drag_sx(chip, qubit, duration=sx_duration if x_duration is None else x_duration,
                              angle=np.pi, name="X"))
        for control, target in pairs:
            assert cr_duration is not None
            gates.add(cross_resonance(chip, control, target, duration=cr_duration, echo=echo))
        return gates

    def add(self, native: NativeGate) -> NativeGate:
        """Add a native gate after checking its template.

        Parameters
        ----------
        native : NativeGate
            Gate on gate-set qubits whose template uses the gate set's chip.
        """
        outside = [label for label in native.qubits if label not in self.qubits]
        if outside:
            raise ValueError(f"{native.name} acts on {outside}, which are not gate-set qubits {list(self.qubits)}.")
        if native.key in self._natives:
            raise ValueError(f"The gate set already has {native.name} on {native.qubits}.")
        if not native.is_composite:
            self._check_template(native)
        self._natives[native.key] = native
        return native

    def rule(self, name: str, qubits: int, expansion: Callable[..., list[tuple[Any, ...]]]) -> None:
        """Add a decomposition rule ahead of the built-in ones.

        Parameters
        ----------
        name : str
            Gate name.
        qubits : int
            Qubit count of the gate.
        expansion : callable
            Function of the gate's parameters that returns a time-ordered list
            of instructions on qubit slots ``0, 1, ...``.
        """
        self._user_rules.setdefault(canonical_name(name), []).append((int(qubits), expansion))

    def _check_template(self, native: NativeGate) -> None:
        sequence = native.sequence
        assert sequence is not None
        if sequence._chip is not self.chip:
            raise ValueError(f"The template of {native.name} uses another chip object than the gate set.")
        native.resolved_knobs()
        for op in sequence.scheduled_ops:
            stop = float(op.start_time) + float(op.envelope.duration)
            if float(op.start_time) < -1e-12 or stop > native.duration + 1e-9:
                raise ValueError(f"A pulse of {native.name} lies outside [0, {native.duration:g}] ns.")
            if op.freq is not None and op.frame not in native.qubits:
                raise ValueError(
                    f"A carrier pulse of {native.name} uses frame {op.frame!r}; use one of the gate qubits "
                    f"{list(native.qubits)} so virtual-Z frames follow circuit order."
                )

    @property
    def natives(self) -> dict[tuple[str, tuple[str, ...]], NativeGate]:
        """Native gates by ``(name, qubits)``."""
        return dict(self._natives)

    def native(self, name: str, qubits: Any) -> NativeGate:
        """Return one native gate.

        Parameters
        ----------
        name : str
            Gate name or alias.
        qubits : str, device, or sequence of them
            Gate qubits in gate order.
        """
        return self._natives[self._key(name, qubits)]

    def calibration(self, name: str, qubits: Any) -> GateCalibration:
        """Return the calibration report of one native.

        Parameters
        ----------
        name : str
            Gate name or alias.
        qubits : str, device, or sequence of them
            Gate qubits in gate order.
        """
        key = self._key(name, qubits)
        if key not in self._calibrations:
            raise KeyError(f"{key[0]} on {key[1]} is not calibrated.")
        return self._calibrations[key]

    @staticmethod
    def _key(name: str, qubits: Any) -> tuple[str, tuple[str, ...]]:
        labels = (resolve_label(qubits),) if isinstance(qubits, str) or not isinstance(qubits, Iterable) else tuple(
            resolve_label(q) for q in qubits)
        return (canonical_name(name), labels)

    # ------------------------------------------------------------ solver space

    @property
    def solver_chip(self) -> "Chip":
        """Chip of gate-set solves, with the excitation cutoff in use."""
        if self._solver_chip is None:
            self._solver_chip = self._initial_space()
        return self._solver_chip

    @property
    def cutoff(self) -> int | None:
        """Excitation cutoff of gate-set solves, or ``None`` for the product space."""
        space = self.solver_chip.space
        return None if space is None else int(space.max_total)

    def _initial_space(self) -> "Chip":
        from quchip.cutoff import ExcitationCutoff

        request = self._space_request
        if request is None:
            return self.chip.with_space(None) if self.chip.space is not None else self.chip
        if isinstance(request, ExcitationCutoff):
            return self.chip.with_space(request)
        if request != "auto":
            raise ValueError(f"space must be 'auto', an ExcitationCutoff, or None; got {request!r}.")
        if self.chip.space is not None or prod(int(d) for d in self.chip.dims) <= AUTO_CUTOFF_DIMENSION:
            return self.chip
        return self.chip.with_space(ExcitationCutoff(max_total=len(self.qubits) + 1))

    def _set_solver_chip(self, chip: "Chip") -> None:
        self._solver_chip = chip
        self._lab = None
        self._idle = None
        self._entries.clear()
        self._placed.clear()
        self._delivered.clear()
        self._model_reason = None

    def _on_solver(self, sequence: "QuantumSequence", chip: "Chip | None" = None) -> "QuantumSequence":
        target = self.solver_chip if chip is None else chip
        return sequence if sequence._chip is target else sequence._copy_with_chip(target)

    def _lab_problem(self) -> Any:
        if self._lab is None:
            lab = _static_problem(self.solver_chip)
            if lab.engine_result.dynamic_terms:
                raise ValueError("The chip has time-dependent terms of its own; gate sets need a chip whose "
                                 "Hamiltonian changes only through scheduled pulses.")
            self._lab = lab
        return self._lab

    def _model(self) -> str:
        """Why the chip's own terms break placement reuse, or ``""``."""
        if self._model_reason is None:
            self._model_reason = model_check(self._lab_problem()) or signal_check(self.chip)
        return self._model_reason

    def _delivered_of(self, op: Any) -> Delivered:
        """Delivered terms and window of one scheduled op, cached by its structure and timing."""
        def number(value: Any) -> Any:
            return None if value is None else round(float(value), 12)

        key = (type(op).__name__, getattr(op, "target_label", None), getattr(op, "drive_label", None),
               number(getattr(op, "freq", None)), number(getattr(op, "detuning", None)), getattr(op, "frame", None),
               number(op.start_time), type(op.envelope).__name__, number(op.envelope.duration))
        if key not in self._delivered:
            self._lab_problem()
            self._delivered[key] = lower_pulse(self.solver_chip, op)
        return self._delivered[key]

    def _span(self, ops: Sequence[Any], duration: float, start: float = 0.0) -> float:
        """Return the end of ``[start, start + duration]`` extended to the last delivered field."""
        stop = start + float(duration)
        for op in ops:
            support = self._delivered_of(op).support
            if support is None:
                continue
            if support[0] < start - 1e-9:
                raise ValueError(f"The pulse on {op.drive_label!r} delivers a field at {support[0]:g} ns, before "
                                 f"its gate starts at {start:g} ns; negative line delays are not supported.")
            stop = max(stop, support[1])
        return stop

    def _states(self, qubits: Sequence[str], background: Mapping[str, int] | None = None,
                chip: "Chip | None" = None) -> tuple[np.ndarray, np.ndarray]:
        problem = self._lab_problem() if chip is None else _static_problem(chip)
        levels = solver_levels(problem.engine_result)
        columns, energies = computational_states(problem, tuple(qubits), levels, background)
        return np.asarray(columns), np.asarray(energies)

    def _block(self, sequence: "QuantumSequence", qubits: Sequence[str], duration: float, *,
               background: Mapping[str, int] | None = None, chip: "Chip | None" = None) -> tuple[Any, float]:
        """Return the gate block ``Π_q RZ_q(-F_q) V`` of a sequence and the unitarity defect.

        The block spans ``[0, duration]``, extended to the last delivered
        field when the signal chain delays a pulse. ``F_q`` sums the
        sequence's own ``vz`` angles, as in a circuit.
        """
        columns, energies = self._states(qubits, background, chip)
        stop = self._span(list(self._on_solver(sequence).scheduled_ops), duration)
        propagator = self._on_solver(sequence, chip).propagator(
            tlist=[0.0, stop], columns=columns, frame=self.frame, options=self.options, dissipation=False,
            backend=self.backend)
        image = np.asarray(propagator.matrix)
        block = project_computational(image, columns, energies, 0.0, stop, tuple(qubits))
        correction = self._frame_correction(tuple(qubits), self._offsets(sequence))
        block = replace(block, matrix=correction[:, None] * np.asarray(block.matrix))
        return block, _defect(propagator)

    # ------------------------------------------------------------- calibration

    def calibrate(
        self,
        *selection: Any,
        max_nfev: int = 200,
        explore: bool | None = None,
        target: float = 1e-3,
        progress: bool = False,
    ) -> dict[tuple[str, tuple[str, ...]], GateCalibration]:
        """Calibrate natives against their ideals and fill the cache.

        Parameters
        ----------
        *selection : str, (name, qubits), or NativeGate
            Natives to calibrate. Empty means all. Natives inside a selected
            composite run first when they are uncalibrated or stale.
        max_nfev : int, default 200
            Bound on residual evaluations of each fit. ``0`` skips the knob
            search: each knob keeps its best start, the template value or a
            seed, and only the virtual-Z phases are fitted. The templates of
            the built-in natives hold closed-form seeds from the chip.
        explore : bool or None, default None
            Override each native's ``explore`` flag.
        target : float, default 1e-3
            Infidelity above which the status is ``"poor_fit"``: at start
            time 0, at the worst start time of ``placement_infidelity``, or
            with one other qubit in level 1, as in ``spectators``.
        progress : bool, default False
            Print one line per native as it finishes.

        Returns
        -------
        dict
            Calibration reports by ``(name, qubits)``. A ``UserWarning`` names
            each native whose status is not ``"ok"``.
        """
        selected = keys = self._select(selection)
        self._check_cutoff([key for key in keys if not self._natives[key].is_composite])
        while True:
            results = {}
            for key in keys:
                results[key] = self._calibrate(key, max_nfev=max_nfev, explore=explore, target=target,
                                               progress=progress)
            failing = self._check_cutoff_after(list(results))
            if not failing:
                self._warm.clear()
                return {key: results[key] for key in selected}
            # Recalibrate every calibrated native in the larger space, starting from its fitted knobs.
            self._warm = {key: {name: knob.value for name, knob in calibration.knobs.items()}
                          for key, calibration in self._calibrations.items()}
            keys = sorted(set(keys) | set(self._calibrations), key=lambda k: self._natives[k].is_composite)
            self._calibrations.clear()
            self._versions.clear()

    def _select(self, selection: Sequence[Any]) -> list[tuple[str, tuple[str, ...]]]:
        if not selection:
            keys = list(self._natives)
        else:
            keys = []
            for item in selection:
                if isinstance(item, NativeGate):
                    keys.append(item.key)
                elif isinstance(item, str):
                    found = [key for key in self._natives if key[0] == canonical_name(item)]
                    if not found:
                        raise KeyError(f"No native named {item!r}.")
                    keys.extend(found)
                else:
                    keys.append(self._key(*item))
        missing = [key for key in keys if key not in self._natives]
        if missing:
            raise KeyError(f"Unknown natives {missing}.")
        plain = [key for key in keys if not self._natives[key].is_composite]
        return plain + [key for key in keys if self._natives[key].is_composite]

    def _calibrate(self, key: tuple[str, tuple[str, ...]], *, max_nfev: int, explore: bool | None,
                   target: float = 1e-3, progress: bool = False) -> GateCalibration:
        if key in self._calibrating:
            cycle = " -> ".join(f"{name}{qubits}" for name, qubits in [*self._calibrating, key])
            raise RecursionError(f"Composite natives form a cycle: {cycle}.")
        native = self._natives[key]
        self._calibrating.append(key)
        try:
            dependencies: dict[Any, int] = {}
            if native.is_composite:
                assert native._build is not None
                self._used.append(set())
                try:
                    sequence, knobs = native._build(self)
                finally:
                    used = self._used.pop()
                dependencies = {dep: self._versions.get(dep, 0) for dep in used}
                native.sequence, native.knobs = sequence, dict(knobs)
                self._check_template(native)
            calibration = self._fit(native, max_nfev=max_nfev, explore=native.explore if explore is None else explore)
            calibration.dependencies = dependencies
        finally:
            self._calibrating.pop()
        self._versions[key] = self._versions.get(key, 0) + 1
        calibration.version = self._versions[key]
        self._calibrations[key] = calibration
        for other in self._calibrations.values():
            if other.dependencies.get(key, calibration.version) != calibration.version:
                other.status = "stale"
        self._entries.pop(key, None)
        self._placed.clear()
        self._post_checks(native, calibration)
        # The gate must also meet the target at other start times and with any other qubit excited.
        worst = max((calibration.infidelity, *(calibration.placement_infidelity or ()),
                     *(1.0 - fidelity for fidelity in calibration.spectators.values())))
        if calibration.status == "ok" and worst > target:
            calibration.status = "poor_fit"
        if calibration.status != "ok":
            warnings.warn(f"{native.name} on {native.qubits}: calibration status {calibration.status!r}.\n"
                          f"{calibration.summary()}", UserWarning, stacklevel=3)
        if progress:
            print(calibration.summary().splitlines()[0])
        return calibration

    def spectators_of(self, native: NativeGate) -> tuple[str, ...]:
        """Return the gate-set qubits that join a native's calibration block, in gate-set order.

        Parameters
        ----------
        native : NativeGate
            Native gate.
        """
        others = [q for q in self.qubits if q not in native.qubits]
        if self.spectators == "none":
            return ()
        if self.spectators == "all":
            return tuple(others)
        # The native's own pulses reach their devices directly, through the signal chain's crosstalk too.
        sources = set(native.qubits)
        for op in () if native.sequence is None else native.sequence.scheduled_ops:
            for term in self._delivered_of(op).terms:
                sources |= term.devices
        links: dict[str, set[str]] = {}
        for coupling in self.chip.couplings:
            a, b = getattr(coupling, "device_a_label", None), getattr(coupling, "device_b_label", None)
            if a is None or b is None:
                continue
            links.setdefault(a, set()).add(b)
            links.setdefault(b, set()).add(a)
        reached = {device for device in sources if device in others}
        frontier = [device for device in sources if device not in reached]
        seen = set(sources)
        while frontier:
            device = frontier.pop()
            for neighbor in links.get(device, ()):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                if neighbor in self.qubits:
                    reached.add(neighbor)
                else:
                    frontier.append(neighbor)
        return tuple(q for q in others if q in reached)

    def _fit(self, native: NativeGate, *, max_nfev: int, explore: bool) -> GateCalibration:
        template = native.sequence
        assert template is not None
        knobs = native.resolved_knobs()
        spectators = self.spectators_of(native)
        block_qubits = (*native.qubits, *spectators)
        ideal = np.kron(native.ideal_matrix(), np.eye(2 ** len(spectators)))
        free_z = (*native.free_z, *spectators)
        duration = native.duration
        bits = np.array(list(itertools.product((0, 1), repeat=len(block_qubits))), dtype=float).reshape(
            -1, len(block_qubits))
        z_bits = bits[:, [block_qubits.index(q) for q in free_z]]

        def bindings(values: np.ndarray) -> dict[str, float]:
            out: dict[str, float] = {}
            for knob, value in zip(knobs, values):
                out.update(knob.bindings(float(value)))
            return out

        def evaluate(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
            block, defect = self._block(bind_pulses(template, bindings(values)), block_qubits, duration)
            return np.asarray(block.matrix), np.asarray(block.leaked), defect

        initial = np.array([knob.initial for knob in knobs], dtype=float)
        starts = [(initial, "template")]
        names = [knob.name for knob in knobs]
        warm = self._warm.get(native.key)
        if warm is not None and set(warm) == set(names):
            starts.append((np.array([warm[name] for name in names], dtype=float), "previous space"))
        for seed in native.seeds:
            unknown = set(seed.values) - set(names)
            if unknown:
                raise KeyError(f"Seed {seed.provenance!r} of {native.name} names unknown knobs {sorted(unknown)}.")
            values = initial.copy()
            for name, value in seed.values.items():
                values[names.index(name)] = float(value)
            starts.append((values, seed.provenance))
        outcome = fit_block(evaluate, ideal, z_bits, knobs, starts=starts, max_nfev=max_nfev, explore=explore,
                            phase_names=free_z)
        calibration = report(outcome, name=native.name, qubits=native.qubits, knobs=knobs, ideal=ideal,
                             z_bits=z_bits, phase_names=free_z, options=self.options, frame=self.frame,
                             cutoff=self.cutoff)
        calibration.block_qubits = block_qubits
        sequence = bind_pulses(template, bindings(outcome.values))
        for qubit, phase in calibration.free_z.items():
            sequence.vz(qubit, -phase)
        calibration.sequence = sequence
        return calibration

    def _post_checks(self, native: NativeGate, calibration: GateCalibration) -> None:
        """Fill the spectator, placement, and repetition fields of a fresh calibration."""
        key = native.key
        if calibration.block_qubits and len(calibration.block_qubits) > len(native.qubits):
            block, _ = self._block(calibration.sequence, native.qubits, native.duration)
            calibration.ground_fidelity = block.fidelity(native.ideal_matrix())
        for spectator in self.qubits:
            if spectator in native.qubits:
                continue
            block, _ = self._block(calibration.sequence, native.qubits, native.duration, background={spectator: 1})
            calibration.spectators[spectator] = block.fidelity(native.ideal_matrix())
        entry = self._entry(key)
        block_qubits = calibration.block_qubits or native.qubits
        if entry.covariant:
            columns, energies = self._states(block_qubits)
            total = solver_levels(self._lab_problem().engine_result).sum(axis=1)
            correction = self._frame_correction(block_qubits, self._offsets(calibration.sequence))
            errors = []
            # Start times spread over 100 ns without a common period, so no carrier aliases them.
            for start in np.concatenate([[0.0], 100.0 * ((np.arange(1, 21) * 0.6180339887498949) % 1.0)]):
                placed = entry.placed(float(start), {}, total, self._idle_of())
                block = project_computational(placed @ columns, columns, energies, float(start),
                                              float(start) + entry.duration, block_qubits)
                errors.append(1.0 - float(average_fidelity(correction[:, None] * np.asarray(block.matrix),
                                                           calibration.ideal)))
            calibration.placement_infidelity = (min(errors), max(errors))
        for count in (1, 4, 20) if entry.covariant else (1, 4):
            composed = self.unitary([(native.name, native.qubits)] * count, qubits=block_qubits)
            calibration.repetitions[count] = 1.0 - composed.fidelity()

    def _check_cutoff(self, keys: list[tuple[str, tuple[str, ...]]]) -> None:
        """Raise an automatic cutoff until the native templates converge."""
        if self._cutoff_checked or self._space_request != "auto" or self.chip.space is not None:
            return
        self._cutoff_checked = True
        if self.cutoff is None:
            return
        from quchip.cutoff import ExcitationCutoff

        maximum = sum(int(d) - 1 for d in self.chip.dims)
        cutoff = self.cutoff
        difference = 0.0
        while cutoff < maximum:
            current = self.chip.with_space(ExcitationCutoff(max_total=cutoff))
            larger = self.chip.with_space(ExcitationCutoff(max_total=cutoff + 1))
            difference = 0.0
            for key in keys:
                native = self._natives[key]
                assert native.sequence is not None
                block_qubits = (*native.qubits, *self.spectators_of(native))
                left, _ = self._block(native.sequence, block_qubits, native.duration, chip=current)
                right, _ = self._block(native.sequence, block_qubits, native.duration, chip=larger)
                difference = max(difference, float(np.abs(np.asarray(left.matrix) - np.asarray(right.matrix)).max()),
                                 float(np.abs(np.asarray(left.leakage) - np.asarray(right.leakage)).max()))
            if difference <= self.cutoff_tolerance:
                break
            cutoff += 1
        if cutoff >= maximum:
            self._set_solver_chip(self.chip)
            warnings.warn("The gate set solves on the full product space; no cutoff converged.", UserWarning,
                          stacklevel=3)
            return
        self._set_solver_chip(self.chip.with_space(ExcitationCutoff(max_total=cutoff)))
        warnings.warn(
            f"The gate set solves with ExcitationCutoff(max_total={cutoff}) of {prod(int(d) for d in self.chip.dims)} "
            f"product states; the native templates change by {difference:.1e} at max_total={cutoff + 1}. "
            "Pass space= to choose the space.", UserWarning, stacklevel=3)

    def _check_cutoff_after(self, keys: list[tuple[str, tuple[str, ...]]]) -> list[tuple[str, tuple[str, ...]]]:
        """Compare calibrated gates with the next cutoff; on a change, raise the cutoff and return the gates.

        Only an automatic cutoff is raised. The caller recalibrates in the
        larger space.
        """
        if self.cutoff is None or self._space_request != "auto" or self.chip.space is not None:
            return []
        from quchip.cutoff import ExcitationCutoff

        larger = self.chip.with_space(ExcitationCutoff(max_total=self.cutoff + 1))
        failing = []
        worst = 0.0
        for key in keys:
            native = self._natives[key]
            calibration = self._calibrations[key]
            block_qubits = calibration.block_qubits or native.qubits
            left, _ = self._block(calibration.sequence, block_qubits, native.duration)
            right, _ = self._block(calibration.sequence, block_qubits, native.duration, chip=larger)
            difference = max(float(np.abs(np.asarray(left.matrix) - np.asarray(right.matrix)).max()),
                             float(np.abs(np.asarray(left.leakage) - np.asarray(right.leakage)).max()))
            calibration.cutoff_difference = difference
            if difference > self.cutoff_tolerance:
                failing.append(key)
                worst = max(worst, difference)
        if failing:
            maximum = sum(int(d) - 1 for d in self.chip.dims)
            names = ", ".join(f"{name} on {','.join(qubits)}" for name, qubits in failing)
            if self.cutoff + 1 >= maximum:
                self._set_solver_chip(self.chip)
                where = "the full product space"
            else:
                self._set_solver_chip(larger)
                where = f"max_total={self.cutoff}"
            warnings.warn(f"{names} changed by {worst:.1e} at the next excitation cutoff; recalibrating in {where}.",
                          UserWarning, stacklevel=3)
        return failing

    # ------------------------------------------------------------------- cache

    def _idle_of(self) -> Idle:
        if self._idle is None:
            self._idle = Idle(self._lab_problem())
        return self._idle

    def _entry(self, key: tuple[str, tuple[str, ...]]) -> GatePropagator:
        """Return the cached propagator of a calibrated native."""
        if key not in self._entries:
            calibration = self._calibrations[key]
            native = self._natives[key]
            sequence = self._on_solver(calibration.sequence)
            ops = list(sequence.scheduled_ops)
            duration = self._span(ops, native.duration)
            delivered = [self._delivered_of(op) for op in ops]
            pieces = []
            defect = 0.0
            for start, stop, indices in split_pieces([item.support for item in delivered]):
                covariant, reason, carrier, frame = covariance(ops, delivered, indices, self._model())
                propagator = sequence.propagator(tlist=[start, stop], frame=self.frame, options=self.options,
                                                 dissipation=False, backend=self.backend)
                defect = max(defect, _defect(propagator))
                pieces.append(Piece(start, stop, carrier, frame, covariant, reason, np.asarray(propagator.matrix)))
            self._entries[key] = GatePropagator(duration, pieces, defect)
        return self._entries[key]

    def entry(self, name: str, qubits: Any) -> GatePropagator:
        """Return the cached propagator of a calibrated native.

        Parameters
        ----------
        name : str
            Gate name or alias.
        qubits : str, device, or sequence of them
            Gate qubits in gate order.
        """
        return self._entry(self._key(name, qubits))

    # ---------------------------------------------------------------- circuits

    def _normalize(self, instruction: Any) -> tuple[str, tuple[Any, ...], tuple[float, ...]]:
        if not isinstance(instruction, (tuple, list)) or not instruction:
            raise TypeError(f"An instruction is (name, qubits) or (name, qubits, params); got {instruction!r}.")
        name = canonical_name(instruction[0])
        qubits: Any = instruction[1] if len(instruction) > 1 else ()
        params: Any = instruction[2] if len(instruction) > 2 else ()
        if len(instruction) > 3:
            raise TypeError(f"An instruction has at most three parts; got {instruction!r}.")
        if isinstance(qubits, (str, int)) or not isinstance(qubits, Iterable):
            qubits = (qubits,)
        labels = tuple(q if isinstance(q, int) else resolve_label(q) for q in qubits)
        if isinstance(params, (int, float, np.floating, np.integer)):
            params = (params,)
        return name, labels, tuple(float(p) for p in params)

    def _expand(self, circuit: Iterable[Any], depth: int = 0,
                stack: tuple[tuple[str, tuple[str, ...]], ...] = ()) -> list[tuple[Any, ...]]:
        """Expand a circuit into natives and virtual instructions.

        ``stack`` holds the gates whose rules are being expanded, so a rule
        that leads back to one of them does not resolve it.
        """
        if depth > 32:
            raise RecursionError("Decomposition rules nest more than 32 levels.")
        rules = rule_table(self._user_rules)
        out: list[tuple[Any, ...]] = []
        for instruction in circuit:
            name, qubits, params = self._normalize(instruction)
            unknown = [q for q in qubits if q not in self.qubits]
            if unknown:
                raise ValueError(f"{name} acts on {unknown}, which are not gate-set qubits {list(self.qubits)}.")
            if name == "RZ":
                out.append(("RZ", qubits[0], params[0]))
            elif name == "IDLE":
                out.extend(("IDLE", q, params[0]) for q in qubits)
            elif name == "GLOBALPHASE":
                out.append(("GLOBALPHASE", params[0] if params else 0.0))
            elif (name, qubits) in self._natives:
                out.append(("native", (name, qubits)))
            elif (name, qubits) in stack:
                raise _Unresolved(f"The rules for {name} on {qubits} lead back to it.")
            else:
                for count, expansion in rules.get(name, []):
                    if count != len(qubits):
                        continue
                    body = []
                    for item in expansion(*params):
                        slots = item[1]
                        mapped = tuple(qubits[s] for s in slots) if isinstance(slots, tuple) else qubits[slots]
                        body.append((item[0], mapped, *item[2:]))
                    try:
                        out.extend(self._expand(body, depth + 1, (*stack, (name, qubits))))
                    except _Unresolved:
                        continue
                    break
                else:
                    raise _Unresolved(f"No native or rule gives {name} on {qubits}.")
        return out

    def schedule(self, circuit: Iterable[Any], *, policy: str = "asap") -> "QuantumSequence":
        """Return the pulse schedule of a circuit.

        Parameters
        ----------
        circuit : iterable of instructions
            ``(name, qubits)`` or ``(name, qubits, params)``. RZ compiles to
            ``vz(q, -λ)``, IDLE(T) to a delay, and other names to natives
            through the decomposition rules.
        policy : {"asap", "serial"}, default "asap"
            ``"asap"`` overlaps gates on disjoint qubits; ``"serial"`` runs
            one gate at a time.
        """
        return self._compile(circuit, policy)[0]

    def _compile(self, circuit: Iterable[Any], policy: str) -> tuple[Any, list[tuple[Any, int, int]], float]:
        from quchip.control.sequence import QuantumSequence, _PulseEntry

        if policy not in ("asap", "serial"):
            raise ValueError(f"policy must be 'asap' or 'serial'; got {policy!r}.")
        sequence = QuantumSequence(self.chip)
        placements: list[tuple[Any, int, int]] = []
        global_phase = 0.0
        count = 0
        for item in self._expand(circuit):
            kind = item[0]
            if kind == "RZ":
                sequence.vz(item[1], -item[2])
            elif kind == "IDLE":
                sequence.delay(item[1], item[2])
            elif kind == "GLOBALPHASE":
                global_phase += item[1]
            else:
                key = item[1]
                calibration = self._calibrated(key)
                if policy == "serial":
                    sequence.barrier()
                sequence.extend(calibration.sequence)
                pulses = sum(isinstance(entry, _PulseEntry) for entry in calibration.sequence._entries)
                placements.append((key, count, pulses))
                count += pulses
                if self._used:
                    self._used[-1].add(key)
        return sequence, placements, global_phase

    def _calibrated(self, key: tuple[str, tuple[str, ...]]) -> GateCalibration:
        if key not in self._calibrations:
            if not self._calibrating:
                raise KeyError(f"{key[0]} on {key[1]} is not calibrated; run calibrate() first.")
            self._calibrate(key, max_nfev=200, explore=None)
        calibration = self._calibrations[key]
        if calibration.status == "stale":
            warnings.warn(f"{key[0]} on {key[1]} holds natives that were calibrated again after it; its pulses "
                          "keep the earlier ones. Calibrate it again to use the new ones.", UserWarning, stacklevel=4)
        return calibration

    def _offsets(self, sequence: Any) -> dict[str, float]:
        from quchip.control.sequence import _FrameShiftEntry

        offsets = dict.fromkeys(self.qubits, 0.0)
        for entry in sequence._entries:
            if isinstance(entry, _FrameShiftEntry):
                offsets[entry.device_label] = offsets.get(entry.device_label, 0.0) + float(entry.angle)
        return offsets

    def ideal(self, circuit: Iterable[Any], *, qubits: Sequence[Any] | None = None) -> np.ndarray:
        """Return the ideal unitary of a circuit.

        Parameters
        ----------
        circuit : iterable of instructions
            As for :meth:`schedule`.
        qubits : sequence of str or device, optional
            Qubit order of the matrix, first most significant. The default is
            every gate-set qubit.
        """
        order = self.qubits if qubits is None else tuple(resolve_label(q) for q in qubits)
        result = np.eye(2 ** len(order), dtype=complex)
        for instruction in circuit:
            result = self._ideal_of(instruction, order) @ result
        return result

    def _ideal_of(self, instruction: Any, order: tuple[str, ...]) -> np.ndarray:
        name, qubits, params = self._normalize(instruction)
        dimension = 2 ** len(order)
        if name == "GLOBALPHASE":
            return np.exp(1j * (params[0] if params else 0.0)) * np.eye(dimension)
        if name == "IDLE":
            return np.eye(dimension, dtype=complex)
        outside = [q for q in qubits if q not in order]
        if outside:
            raise ValueError(f"{name} acts on {outside}, outside the qubits {list(order)}.")
        positions = [order.index(q) for q in qubits]
        if name in IDEALS:
            return embed(ideal_unitary(name, params), positions, len(order))
        if (name, qubits) in self._natives:
            return embed(self._natives[(name, qubits)].ideal_matrix(), positions, len(order))
        rules = rule_table(self._user_rules)
        for count, expansion in rules.get(name, []):
            if count == len(qubits):
                body = []
                for item in expansion(*params):
                    slots = item[1]
                    mapped = tuple(qubits[s] for s in slots) if isinstance(slots, tuple) else qubits[slots]
                    body.append((item[0], mapped, *item[2:]))
                return self.ideal(body, qubits=order)
        raise KeyError(f"No ideal for {name} on {qubits}.")

    def unitary(self, circuit: Iterable[Any], *, qubits: Sequence[Any] | None = None,
                policy: str = "asap") -> CircuitUnitary:
        """Compose a circuit's propagator from cached gates.

        Gates whose pieces are covariant reuse their cached propagator at any
        start time and frame. Overlapping gates and non-covariant gates run
        one solve per placement. Idle intervals use the static lab
        Hamiltonian's eigensystem.

        Parameters
        ----------
        circuit : iterable of instructions
            As for :meth:`schedule`.
        qubits : sequence of str or device, optional
            Computational qubits of the result; other devices stay in dressed
            level 0. The default is every gate-set qubit.
        policy : {"asap", "serial"}, default "asap"
            Scheduling policy.
        """
        circuit = list(circuit)
        order = self.qubits if qubits is None else tuple(resolve_label(q) for q in qubits)
        sequence, placements, global_phase = self._compile(circuit, policy)
        solver_sequence = self._on_solver(sequence)
        ops = list(solver_sequence.scheduled_ops)
        total = solver_levels(self._lab_problem().engine_result).sum(axis=1)
        idle = self._idle_of()
        instances = []
        for key, first, count in placements:
            template = list(self._on_solver(self._calibrations[key].sequence).scheduled_ops)
            placed = ops[first:first + count]
            start = float(placed[0].start_time) - float(template[0].start_time) if count else None
            offsets: dict[str, float] = {}
            for op, reference in zip(placed, template):
                if op.frame is not None:
                    offsets[op.frame] = float(op.phase_offset) - float(reference.phase_offset)
            if start is not None:
                instances.append((start, start + self._entry(key).duration, key, offsets))
        instances.sort(key=lambda item: item[0])
        segments: list[list[Any]] = []
        for instance in instances:
            if segments and instance[0] < max(item[1] for item in segments[-1]) - 1e-12:
                segments[-1].append(instance)
            else:
                segments.append([instance])
        duration = max([float(sequence.total_duration), *(item[1] for item in instances)])
        result = np.eye(total.size, dtype=complex)
        cursor = 0.0
        solves = 0
        for segment in segments:
            start = segment[0][0]
            stop = max(item[1] for item in segment)
            if start > cursor:
                result = idle(start - cursor) @ result
            entry = self._entry(segment[0][2]) if len(segment) == 1 else None
            if entry is not None and entry.covariant:
                matrix = entry.placed(start, segment[0][3], total, idle)
            else:
                cache_key = tuple((round(item[0], 9), item[2], tuple(sorted((k, round(v, 12))
                                                                             for k, v in item[3].items())))
                                  for item in segment)
                if cache_key not in self._placed:
                    propagator = solver_sequence.propagator(tlist=[start, stop], frame=self.frame,
                                                            options=self.options, dissipation=False,
                                                            backend=self.backend)
                    self._placed[cache_key] = np.asarray(propagator.matrix)
                    solves += 1
                matrix = self._placed[cache_key]
            result = matrix @ result
            cursor = stop
        if duration > cursor:
            result = idle(duration - cursor) @ result
        columns, energies = self._states(order)
        block = project_computational(result @ columns, columns, energies, 0.0, duration, order)
        frame_offsets = self._offsets(sequence)
        correction = self._frame_correction(order, frame_offsets) * np.exp(1j * global_phase)
        return CircuitUnitary(full=result, computational=correction[:, None] * np.asarray(block.matrix),
                              leakage=np.asarray(block.leakage), frame_offsets=frame_offsets, duration=duration,
                              qubits=order, ideal=self.ideal(circuit, qubits=order), solves=solves)

    @staticmethod
    def _frame_correction(order: Sequence[str], offsets: Mapping[str, float]) -> np.ndarray:
        """Diagonal of ``Π_q RZ_q(-F_q)`` on computational states, first qubit most significant."""
        diagonal = np.ones(1, dtype=complex)
        for qubit in order:
            angle = -offsets.get(qubit, 0.0)
            diagonal = np.kron(diagonal, np.array([np.exp(-0.5j * angle), np.exp(0.5j * angle)]))
        return diagonal

    def initial_state(self, bits: Any) -> Any:
        """Return the chip state of a dressed computational bit string, in the gauge of ``chip.state()``.

        Parameters
        ----------
        bits : str or sequence of int
            One bit per gate-set qubit, first qubit first.
        """
        mapping = self._bits(bits, self.qubits)
        return self.chip.state(mapping)

    @staticmethod
    def _bits(bits: Any, order: Sequence[str]) -> dict[Any, int]:
        values = [int(b) for b in (bits if not isinstance(bits, str) else list(bits))]
        if len(values) != len(order) or any(b not in (0, 1) for b in values):
            raise ValueError(f"Expected {len(order)} bits for {list(order)}; got {bits!r}.")
        return dict(zip(order, values))

    def simulate(
        self,
        circuit: Iterable[Any],
        initial: Any = None,
        *,
        qubits: Sequence[Any] | None = None,
        policy: str = "asap",
        dissipation: bool = True,
    ) -> CircuitRun:
        """Simulate a circuit's pulse schedule and return the computational state.

        The schedule runs with the gate-set solver settings. The final state
        is mapped into the computational frame with the virtual-Z correction.

        Parameters
        ----------
        circuit : iterable of instructions
            As for :meth:`schedule`.
        initial : str, tuple of bits, or array_like, optional
            Bit string, or a ``2^n`` ket or density matrix in computational
            order. The default is all zeros.
        qubits : sequence of str or device, optional
            Computational qubits; other devices start and are read in dressed
            level 0. The default is every gate-set qubit.
        policy : {"asap", "serial"}, default "asap"
            Scheduling policy.
        dissipation : bool, default True
            Include the chip's collapse channels.
        """
        order = self.qubits if qubits is None else tuple(resolve_label(q) for q in qubits)
        circuit = list(circuit)
        sequence, _, global_phase = self._compile(circuit, policy)
        solver_sequence = self._on_solver(sequence)
        duration = self._span(list(solver_sequence.scheduled_ops), float(sequence.total_duration))
        columns, energies = self._states(order)
        dimension = 2 ** len(order)
        if initial is None:
            initial = "0" * len(order)
        if isinstance(initial, (str, tuple, list)) and len(initial) == len(order) and all(
                str(b) in ("0", "1") for b in initial):
            vector = np.zeros(dimension, dtype=complex)
            vector[int("".join(str(int(b)) for b in initial), 2)] = 1.0
            initial = vector
        initial = np.asarray(initial, dtype=complex)
        correction = self._frame_correction(order, self._offsets(sequence)) * np.exp(1j * global_phase)
        mapping = correction[:, None] * np.exp(2j * np.pi * energies * duration)[:, None] * columns.conj().T
        if duration <= 0.0:
            # Only virtual gates: no time passes, so the frame correction is the whole circuit.
            step = mapping @ columns
            state = step @ initial if initial.ndim == 1 else step @ initial @ step.conj().T
            populations = np.abs(state) ** 2 if initial.ndim == 1 else np.real(np.diag(state))
            return CircuitRun(state=state, probabilities=self._probabilities(populations, order),
                              leakage=float(1.0 - populations.sum()))
        open_system = dissipation and bool(_static_problem(self.solver_chip, dissipation=True)
                                           .engine_result.collapse_terms)
        if initial.ndim == 1 and not open_system:
            propagator = solver_sequence.propagator(tlist=[0.0, duration], columns=columns @ initial[:, None],
                                                    frame=self.frame, options=self.options, dissipation=False,
                                                    backend=self.backend)
            state = mapping @ np.asarray(propagator.matrix)[:, 0]
            populations = np.abs(state) ** 2
            return CircuitRun(state=state, probabilities=self._probabilities(populations, order),
                              leakage=float(1.0 - populations.sum()), result=propagator)
        rho = initial if initial.ndim == 2 else np.outer(initial, initial.conj())
        superoperator = solver_sequence.propagator(tlist=[0.0, duration], frame=self.frame, options=self.options,
                                                   dissipation=dissipation, backend=self.backend)
        start = columns @ rho @ columns.conj().T
        if superoperator.kind == "superoperator":
            d = start.shape[0]
            final = (np.asarray(superoperator.matrix) @ start.reshape(-1, order="F")).reshape(d, d, order="F")
        else:
            unitary = np.asarray(superoperator.matrix)
            final = unitary @ start @ unitary.conj().T
        state = mapping @ final @ mapping.conj().T
        populations = np.real(np.diag(state))
        return CircuitRun(state=state, probabilities=self._probabilities(populations, order),
                          leakage=float(1.0 - populations.sum()), result=superoperator)

    @staticmethod
    def _probabilities(populations: np.ndarray, order: Sequence[str]) -> dict[str, float]:
        return {"".join(map(str, bits)): float(populations[i])
                for i, bits in enumerate(itertools.product((0, 1), repeat=len(order)))}

    # ----------------------------------------------------------------- report

    def describe(self) -> str:
        """Return one line per native: name, qubits, duration, infidelity, leakage, and status."""
        lines = [f"GateSet on {', '.join(self.qubits)}; solver {self.frame} frame, "
                 f"{'product space' if self.cutoff is None else f'ExcitationCutoff({self.cutoff})'}"]
        lines.append(f"{'gate':>8s} {'qubits':>12s} {'ns':>7s} {'1 - F_avg':>10s} {'leakage':>9s}  status")
        for key, native in self._natives.items():
            calibration = self._calibrations.get(key)
            if calibration is None:
                lines.append(f"{native.name:>8s} {','.join(native.qubits):>12s} {'':>7s} {'-':>10s} {'-':>9s}  "
                             "uncalibrated")
                continue
            status = calibration.status
            lines.append(f"{native.name:>8s} {','.join(native.qubits):>12s} {native.duration:7.1f} "
                         f"{calibration.infidelity:10.2e} {calibration.leakage:9.1e}  {status}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"GateSet(qubits={self.qubits}, natives={[f'{n}{q}' for n, q in self._natives]})"
