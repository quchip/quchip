"""qutip-qip processor and compiler backed by a quchip gate set.

This module imports qutip-qip. Use :func:`quchip.interop.qutip_qip.to_qutip_qip`
to build the processor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import qutip
from qutip_qip.compiler import GateCompiler
from qutip_qip.device import ModelProcessor

from quchip.backend import _backend_context, _coerce_backend
from quchip.engine.problem import build_solve_problem
from quchip.gateset.gateset import DEFAULT_OPTIONS
from quchip.results.propagator import _frame_phases, solver_levels

from . import from_qutip_qip

if TYPE_CHECKING:
    from quchip.control.sequence import QuantumSequence
    from quchip.gateset import GateSet

__all__ = ["ChipCompiler", "ChipProcessor"]

#: Spacing in ns of the pulse samples that the compiler returns.
SAMPLE_STEP = 1.0


class ChipCompiler(GateCompiler):
    """Compile qutip-qip gates into the pulse schedule of a quchip gate set.

    Each gate expands through the gate set's natives and decomposition rules.
    RZ becomes a virtual Z. The exact schedule stays in :attr:`schedule`. The
    returned coefficients sample each line's complex envelope ``I + iQ`` in
    the frame of its carrier, one ``"<line>.i"`` and ``"<line>.q"`` pair per
    line and frame, for plotting. A tone on a line at another qubit's
    frequency, such as a cross-resonance tone, is ``"<line>@<qubit>"``.

    Parameters
    ----------
    num_qubits : int
        Number of circuit qubits.
    params : dict
        Processor parameters, with the gate set under ``"gates"``.

    Attributes
    ----------
    schedule : QuantumSequence
        Schedule of the last compiled circuit, on the gate set's solver chip,
        or on a larger space when gates run at the same time.
    duration : float
        End time of its last delivered field in ns.
    tlist : numpy.ndarray
        Sample times in ns.
    readout : numpy.ndarray
        Map from the final lab-frame state to computational amplitudes, with
        the dressed phases and the virtual-Z phases removed.
    embedding : numpy.ndarray or None
        Map of the solver coordinates into the space of ``schedule`` when it
        is larger, else ``None``.
    """

    def __init__(self, num_qubits: int, params: dict[str, Any]) -> None:
        super().__init__(num_qubits, params=params)
        self.gates: GateSet = params["gates"]

    def compile(self, circuit: Any, schedule_mode: Any = "ASAP", args: Any = None) -> tuple[dict, dict]:
        """Compile a circuit and return sampled pulse coefficients.

        Parameters
        ----------
        circuit : QubitCircuit or iterable of GateInstruction
            Circuit to compile.
        schedule_mode : {"ASAP", None}, default "ASAP"
            ``"ASAP"`` overlaps gates on disjoint qubits. ``None`` or
            ``False`` runs one gate at a time.
        args : None
            Not used.

        Returns
        -------
        tlist, coeffs : dict
            Sample times and coefficients by pulse label.
        """
        policies = {"ASAP": "asap", None: "serial", False: "serial"}
        if schedule_mode not in policies:
            raise ValueError(f"schedule_mode must be 'ASAP' or None; got {schedule_mode!r}.")
        sequence, placements, self.global_phase = self.gates._compile(from_qutip_qip(circuit, self.gates),
                                                                      policies[schedule_mode])
        self.schedule, self.duration, _, self.readout, self.embedding = self.gates._readout(
            sequence, placements, self.global_phase, self.gates.qubits)
        self.tlist, coeffs = _samples(self.schedule, self.duration)
        return {label: self.tlist for label in coeffs}, coeffs


def _samples(schedule: "QuantumSequence", duration: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Sample each line's complex envelope in its carrier frame."""
    ops = schedule.scheduled_ops
    edges = [float(t) for op in ops for t in (op.start_time, op.start_time + op.envelope.duration)]
    count = int(np.ceil(duration / SAMPLE_STEP)) + 1 if duration > 0 else 1
    tlist = np.unique(np.concatenate([np.linspace(0.0, duration, count), edges]))
    signals: dict[str, np.ndarray] = {}
    for op in ops:
        label = op.drive_label if op.frame in (None, op.target_label) else f"{op.drive_label}@{op.frame}"
        local = tlist - float(op.start_time)
        inside = (local >= 0.0) & (local <= float(op.envelope.duration))
        phase = float(op.phase_offset) - 2 * np.pi * float(op.detuning or 0.0) * local[inside]
        values = signals.setdefault(label, np.zeros(tlist.size, dtype=complex))
        values[inside] += np.asarray(op.envelope.sample(local[inside]), dtype=complex) * np.exp(1j * phase)
    coeffs = {}
    for label, values in signals.items():
        coeffs[f"{label}.i"] = values.real
        coeffs[f"{label}.q"] = values.imag
    return tlist, coeffs


class _ChipModel:
    """The chip in qutip-qip's model interface.

    The chip owns the Hamiltonian, so the pulse labels carry a zero operator.
    """

    def __init__(self, gates: "GateSet") -> None:
        self.num_qubits = len(gates.qubits)
        self.dims: list[int] = []
        self.params: dict[str, Any] = {"gates": gates}
        self.labels: list[str] = []

    def get_all_drift(self) -> list[Any]:
        return []

    def get_control(self, label: str) -> tuple[Any, list[int]]:
        return qutip.qzero(self.dims), list(range(len(self.dims)))

    def get_control_labels(self) -> list[str]:
        return list(self.labels)

    def get_control_latex(self) -> list[dict[str, str]]:
        return [{label: label for label in self.labels}]

    def get_noise(self) -> list[Any]:
        return []


class ChipProcessor(ModelProcessor):
    """qutip-qip processor that runs circuits on a quchip chip.

    :class:`ChipCompiler` compiles each circuit with the gate set, and
    ``run_state`` integrates the chip's Hamiltonian in the gate set's frame
    with ``qutip.mesolve``. The default solver options are the gate set's
    tight QuTiP tolerances and quchip's step bound; an ``options`` argument
    overrides them. Register states use the gate set's dressed computational
    basis, first qubit most significant.
    :meth:`generate_init_processor_state` maps a register state onto the
    chip, and :meth:`get_final_circuit_state` maps the final chip state back,
    with the dressed phases and the virtual-Z phases removed. Leakage shows
    as missing norm or trace.

    Noise comes from the chip: ``run_state(noisy=True)``, the default,
    includes its collapse channels. ``add_noise`` and the matrix-exponential
    ``run_analytically`` are not supported.

    Parameters
    ----------
    gates : GateSet
        Calibrated gate set.
    """

    def __init__(self, gates: "GateSet") -> None:
        super().__init__(model=_ChipModel(gates))
        self.gates = gates
        self._default_compiler = ChipCompiler
        self.pulse_mode = "continuous"
        self._compiled: ChipCompiler | None = None
        self._columns()

    def _columns(self) -> np.ndarray:
        """Return the dressed computational states and set the dimensions of the solver space.

        A cutoff that calibration raises later changes the space, so this
        runs again before each use.
        """
        columns, _ = self.gates._states(self.gates.qubits)
        dims = [int(d) for d in self.gates.solver_chip.dims]
        self.model.dims = dims if int(np.prod(dims)) == columns.shape[0] else [columns.shape[0]]
        return columns

    def load_circuit(self, qc: Any, schedule_mode: Any = "ASAP", compiler: ChipCompiler | None = None) -> Any:
        """Compile a circuit and keep its schedule for the next run.

        Parameters
        ----------
        qc : QubitCircuit
            Circuit on ``len(gates.qubits)`` qubits.
        schedule_mode : {"ASAP", None}, default "ASAP"
            Scheduling, see :meth:`ChipCompiler.compile`.
        compiler : ChipCompiler, optional
            Compiler to use instead of a new one.

        Returns
        -------
        tlist, coeffs : dict
            Sample times and coefficients by pulse label.
        """
        compiler = ChipCompiler(self.num_qubits, self.params) if compiler is None else compiler
        tlist, coeffs = compiler.compile(qc, schedule_mode=schedule_mode)
        self._columns()
        self.model.labels = list(coeffs)
        self.set_coeffs(coeffs)
        self.set_tlist(tlist)
        schedule = compiler.schedule
        # A circuit of virtual gates takes no time; its Hamiltonian is the static one.
        stop = compiler.duration if compiler.duration > 0 else 1.0
        with _backend_context(_coerce_backend("qutip")):
            problem = build_solve_problem(schedule._chip, schedule._materialize_drive_ops(), [0.0, stop],
                                          states="final", frame=self.gates.frame, _product_space=True)
        backend = problem.backend
        prepared = backend.prepare_hamiltonian(problem.engine_result, problem.tlist, rtol=DEFAULT_OPTIONS["rtol"])
        self._hamiltonian = backend._coerce_solver_rhs(prepared.rhs)
        self._collapse = backend._collapse_operators(problem.engine_result)
        self._options = backend.resolve_solver_options(dict(DEFAULT_OPTIONS), metadata=prepared.metadata,
                                                       tlist=problem.tlist)
        frame = problem.resolved_frame.frequencies
        frequencies = np.array([float(frame.get(device.label, 0.0)) for device in problem.chip.devices])
        rotation = _frame_phases(solver_levels(problem.engine_result), frequencies, compiler.duration, np)
        self._readout = compiler.readout * rotation[None, :]
        self._compiled = compiler
        return tlist, coeffs

    def get_full_tlist(self, tol: float = 1.0e-10) -> Any:
        return None if self._compiled is None else self._compiled.tlist

    def get_qobjevo(self, args: Any = None, noisy: bool = False) -> tuple[Any, list[Any]]:
        """Return the chip Hamiltonian of the loaded circuit and, when noisy, its collapse operators."""
        if self._compiled is None:
            raise RuntimeError("Load a circuit first.")
        if args:
            raise ValueError("The chip Hamiltonian takes no QobjEvo arguments.")
        return self._hamiltonian, list(self._collapse) if noisy else []

    def run_state(self, init_state: Any = None, analytical: bool = False, qc: Any = None, states: Any = None,
                  **kwargs: Any) -> Any:
        """Load ``qc`` when given and integrate the chip from ``init_state``.

        Parameters
        ----------
        init_state : qutip.Qobj
            Chip state from :meth:`generate_init_processor_state`. When the
            circuit runs gates at once in a larger space, the state is put
            into that space.
        analytical : bool, default False
            Not supported.
        qc : QubitCircuit, optional
            Circuit to load first.
        states : qutip.Qobj, optional
            qutip-qip's older name for ``init_state``.
        **kwargs
            Passed to ``qutip.mesolve``, such as ``noisy``, ``options``, and
            ``tlist``.
        """
        if analytical:
            self.run_analytically()
        if qc is not None:
            self.load_circuit(qc)
        if self._compiled is None:
            raise RuntimeError("Load a circuit first.")
        kwargs["options"] = {**self._options, **(kwargs.get("options") or {})}
        embedding = self._compiled.embedding
        if embedding is not None:
            init_state, states = (self._embed(state, embedding) for state in (init_state, states))
        return super().run_state(init_state=init_state, states=states, **kwargs)

    def _embed(self, state: Any, embedding: np.ndarray) -> Any:
        """Put a state on the solver coordinates into the larger space of the loaded circuit."""
        if state is None or state.shape[0] != embedding.shape[1]:
            return state
        dims = list(self._hamiltonian.dims[0])
        matrix = state.full()
        if state.isket:
            return qutip.Qobj(embedding @ matrix, dims=[dims, [1] * len(dims)])
        return qutip.Qobj(embedding @ matrix @ embedding.T, dims=[dims, dims])

    def run_analytically(self, init_state: Any = None, qc: Any = None) -> Any:
        raise NotImplementedError("ChipProcessor integrates the chip with run_state; it has no "
                                  "matrix-exponential run. GateSet.unitary() composes the gate propagators.")

    def add_noise(self, noise: Any) -> None:
        raise NotImplementedError("Noise comes from the chip. Set T1, T2, and the other collapse channels on its "
                                  "devices; run_state(noisy=True) includes them.")

    def generate_init_processor_state(self, init_circuit_state: Any = None) -> Any:
        """Return the chip state of a register state, ``|0...0>`` by default.

        Parameters
        ----------
        init_circuit_state : qutip.Qobj, optional
            Register ket or density matrix in computational order.
        """
        columns = self._columns()
        if init_circuit_state is None:
            return qutip.Qobj(columns[:, 0], dims=[self.dims, [1] * len(self.dims)])
        register = init_circuit_state.full()
        if register.shape[0] != columns.shape[1]:
            raise ValueError(f"Expected a register state of dimension {columns.shape[1]}; got {register.shape[0]}.")
        if init_circuit_state.isket:
            return qutip.Qobj(columns @ register, dims=[self.dims, [1] * len(self.dims)])
        return qutip.Qobj(columns @ register @ columns.conj().T, dims=[self.dims, self.dims])

    def get_final_circuit_state(self, final_processor_state: Any) -> Any:
        """Return the register state of a final chip state of the loaded circuit.

        Parameters
        ----------
        final_processor_state : qutip.Qobj
            Ket or density matrix at the end of the circuit, in the frame of
            ``run_state``.
        """
        if self._compiled is None:
            raise RuntimeError("Load a circuit first.")
        state = final_processor_state.full()
        n = self.num_qubits
        if final_processor_state.isket:
            return qutip.Qobj(self._readout @ state, dims=[[2] * n, [1] * n])
        return qutip.Qobj(self._readout @ state @ self._readout.conj().T, dims=[[2] * n, [2] * n])
