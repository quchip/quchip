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
from quchip.gateset.gateset import DEFAULT_OPTIONS, _chip_fingerprint
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
        Processor parameters, with the gate set under ``"gateset"``.

    Attributes
    ----------
    schedule : QuantumSequence
        Schedule of the last compiled circuit, on the gate set's solver chip,
        or on a larger space when gates run at the same time.
    duration : float
        End time of its last delivered field in ns.
    tlist : numpy.ndarray
        Sample times in ns.
    chip_to_register : numpy.ndarray
        Map from the lab-frame chip state at :attr:`duration` to register
        amplitudes, with the dressed phases and the virtual-Z phases removed.
    embedding : numpy.ndarray or None
        Map of the solver coordinates into the space of ``schedule`` when it
        is larger, else ``None``.
    """

    def __init__(self, num_qubits: int, params: dict[str, Any]) -> None:
        super().__init__(num_qubits, params=params)
        self.gateset: GateSet = params["gateset"]

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
        sequence, placements, self.global_phase = self.gateset._compile(from_qutip_qip(circuit, self.gateset),
                                                                      policies[schedule_mode])
        self.schedule, self.duration, _, self.chip_to_register, self.embedding = self.gateset._readout(
            sequence, placements, self.global_phase, self.gateset.qubits)
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

    def __init__(self, gateset: "GateSet") -> None:
        self.num_qubits = len(gateset.qubits)
        self.dims: list[int] = []
        self.params: dict[str, Any] = {"gateset": gateset}
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
    basis, first qubit most significant. ``run_state`` starts in the dressed
    ``|0...0>`` or takes a bit string or a register state;
    :meth:`generate_init_processor_state` maps a register state onto the
    chip, and :meth:`get_final_circuit_state` maps the final chip state back,
    with the dressed phases and the virtual-Z phases removed. Leakage shows
    as missing norm or trace.

    Noise comes from the chip: ``run_state(noisy=True)``, the default,
    includes its collapse channels. ``add_noise`` and the matrix-exponential
    ``run_analytically`` are not supported.

    Parameters
    ----------
    gateset : GateSet
        Calibrated gate set.
    """

    def __init__(self, gateset: "GateSet") -> None:
        super().__init__(model=_ChipModel(gateset))
        self.gateset = gateset
        self._default_compiler = ChipCompiler
        self.pulse_mode = "continuous"
        self._compiled: ChipCompiler | None = None
        self._spaces: list[np.ndarray] = []
        self._loaded: tuple[Any, Any, ChipCompiler | None] | None = None
        self._loaded_state: Any = None
        self._pulse_record: list[tuple[Any, np.ndarray, np.ndarray]] = []
        self._columns()

    def _columns(self) -> np.ndarray:
        """Return the dressed computational states and set the dimensions of the solver space.

        A cutoff that calibration raises later changes the space, so this
        runs again before each use. Each space is recorded, so a chip state
        prepared in an earlier one can be put into the current one.
        """
        columns, _ = self.gateset._states(self.gateset.qubits)
        dims = [int(d) for d in self.gateset.solver_chip.dims]
        self.model.dims = dims if int(np.prod(dims)) == columns.shape[0] else [columns.shape[0]]
        levels = solver_levels(self.gateset._lab_problem().engine_result)
        if not self._spaces or not np.array_equal(self._spaces[-1], levels):
            self._spaces.append(levels)
        return columns

    def load_circuit(self, qc: Any, schedule_mode: Any = "ASAP", compiler: ChipCompiler | None = None) -> Any:
        """Compile a circuit and keep its schedule for the next run.

        Parameters
        ----------
        qc : QubitCircuit
            Circuit on ``len(gateset.qubits)`` qubits.
        schedule_mode : {"ASAP", None}, default "ASAP"
            Scheduling, see :meth:`ChipCompiler.compile`.
        compiler : ChipCompiler, optional
            Compiler to use instead of a new one.

        Returns
        -------
        tlist, coeffs : dict
            Sample times and coefficients by pulse label.
        """
        requested = compiler
        compiler = ChipCompiler(self.num_qubits, self.params) if compiler is None else compiler
        tlist, coeffs = compiler.compile(qc, schedule_mode=schedule_mode)
        self._columns()
        self.model.labels = list(coeffs)
        self.set_coeffs(coeffs)
        self.set_tlist(tlist)
        self._pulse_record = [(pulse.label, np.array(pulse.coeff, copy=True), np.array(pulse.tlist, copy=True))
                              for pulse in self.pulses]
        schedule = compiler.schedule
        # A circuit of virtual gates takes no time; its Hamiltonian is the static one.
        stop = compiler.duration if compiler.duration > 0 else 1.0
        with _backend_context(_coerce_backend("qutip")):
            problem = build_solve_problem(schedule._chip, schedule._materialize_drive_ops(), [0.0, stop],
                                          states="final", frame=self.gateset.frame, _product_space=True)
        backend = problem.backend
        prepared = backend.prepare_hamiltonian(problem.engine_result, problem.tlist, rtol=DEFAULT_OPTIONS["rtol"])
        self._hamiltonian = backend._coerce_solver_rhs(prepared.rhs)
        self._collapse = backend._collapse_operators(problem.engine_result)
        self._options = backend.resolve_solver_options(dict(DEFAULT_OPTIONS), metadata=prepared.metadata,
                                                       tlist=problem.tlist)
        frame = problem.resolved_frame.frequencies
        frequencies = np.array([float(frame.get(device.label, 0.0)) for device in problem.chip.devices])
        rotation = _frame_phases(solver_levels(problem.engine_result), frequencies, compiler.duration, np)
        self._chip_to_register = compiler.chip_to_register * rotation[None, :]
        self._compiled = compiler
        self._loaded, self._loaded_state = (qc, schedule_mode, requested), self._gates_state()
        return tlist, coeffs

    def _gates_state(self) -> tuple[Any, ...]:
        """Return what a compiled schedule depends on: the cutoff, the calibration versions, and the chip."""
        return (self.gateset.cutoff, tuple(sorted(self.gateset._versions.items())),
                _chip_fingerprint(self.gateset.chip))

    def get_full_tlist(self, tol: float = 1.0e-10) -> Any:
        return None if self._compiled is None else self._compiled.tlist

    def get_qobjevo(self, args: Any = None, noisy: bool = False) -> tuple[Any, list[Any]]:
        """Return the chip Hamiltonian of the loaded circuit and, when noisy, its collapse operators."""
        if self._compiled is None:
            raise RuntimeError("Load a circuit first, with load_circuit(qc) or run_state(qc=qc).")
        if self._gates_state() != self._loaded_state:
            raise RuntimeError("The gate set or the chip changed after load_circuit; load the circuit again.")
        if args:
            raise ValueError("The chip Hamiltonian takes no QobjEvo arguments.")
        pulses = [(pulse.label, pulse.coeff, pulse.tlist) for pulse in self.pulses]
        if len(pulses) != len(self._pulse_record) or any(
                label != old_label or not np.array_equal(coeff, old_coeff) or not np.array_equal(tlist, old_tlist)
                for (label, coeff, tlist), (old_label, old_coeff, old_tlist) in zip(pulses, self._pulse_record)):
            raise ValueError("The pulses changed after load_circuit. ChipProcessor solves the gate set's schedule, "
                             "so pulse edits would not take effect; change the natives or recalibrate instead.")
        return self._hamiltonian, list(self._collapse) if noisy else []

    def run_state(self, init_state: Any = None, analytical: bool = False, qc: Any = None, states: Any = None,
                  **kwargs: Any) -> Any:
        """Load ``qc`` when given and integrate the chip from ``init_state``.

        Parameters
        ----------
        init_state : str, qutip.Qobj, or None, default None
            Initial state. ``None`` starts in the dressed ``|0...0>``. A bit
            string such as ``"01"``, circuit qubit 0 first, or a ket or
            density matrix on the register qubits maps onto the chip as in
            :meth:`generate_init_processor_state`. A state with the
            processor's dimensions is a chip state, except on a chip of
            two-level gate-set qubits only, where it reads as a register
            state. A chip state prepared
            before a calibration raised the excitation cutoff is put into the
            current space, and when the circuit runs gates at once in a larger
            space, into that space.
        analytical : bool, default False
            Not supported.
        qc : QubitCircuit, optional
            Circuit to load first. Without it, the loaded circuit runs, compiled
            again with its compiler when the gate set or the chip changed
            after it was loaded.
        states : str or qutip.Qobj, optional
            qutip-qip's older name for ``init_state``.
        **kwargs
            Passed to ``qutip.mesolve``, such as ``noisy``, ``options``, and
            ``tlist``.
        """
        from qutip_qip.circuit import QubitCircuit

        if isinstance(init_state, QubitCircuit):
            raise TypeError("The first argument of run_state is the initial state; pass the circuit as qc=.")
        if analytical:
            self.run_analytically()
        if qc is not None:
            self.load_circuit(qc)
        elif self._loaded is not None and self._gates_state() != self._loaded_state:
            # The gate set or the chip changed after the circuit was loaded: compile it again with its compiler.
            self.load_circuit(*self._loaded)
        if self._compiled is None:
            raise RuntimeError("Load a circuit first, with load_circuit(qc) or run_state(qc=qc).")
        kwargs["options"] = {**self._options, **(kwargs.get("options") or {})}
        if states is None:
            init_state = self._chip_state(init_state)
        else:
            states = self._chip_state(states)
        embedding = self._compiled.embedding
        if embedding is not None:
            init_state, states = (self._embed(state, embedding) for state in (init_state, states))
        return super().run_state(init_state=init_state, states=states, **kwargs)

    def _chip_state(self, state: Any) -> Any:
        """Return the chip state of ``None``, a bit string, a register state, or a chip state."""
        n = self.num_qubits
        if state is None:
            return self._on_chip(qutip.basis([2] * n, [0] * n))
        if isinstance(state, (str, tuple, list)):
            bits = [str(bit) for bit in state]
            if len(bits) != n or set(bits) - {"0", "1"}:
                raise ValueError(f"A bit string needs {n} digits of 0 and 1; got {state!r}.")
            return self._on_chip(qutip.basis([2] * n, [int(bit) for bit in bits]))
        if not (state.isket or state.isoper):
            raise ValueError("The initial state must be a ket or a density matrix.")
        dims = list(state.dims[0])
        if dims == [2] * n:
            # A register state. On a chip of two-level gate-set qubits only, chip states have these
            # dimensions too, and they read as register states there.
            return self._on_chip(state)
        if dims in (list(self.dims), list(self._hamiltonian.dims[0])):
            return state
        # A chip state prepared before a calibration raised the excitation cutoff.
        for levels in self._spaces[:-1]:
            if levels.shape[0] == state.shape[0]:
                return self._lift(state, self._space_embedding(levels), list(self.dims))
        raise ValueError(f"Expected a bit string, a state on {n} register qubits, or a chip state with "
                         f"dimensions {list(self.dims)}; got dimensions {state.dims}.")

    def _space_embedding(self, levels: np.ndarray) -> np.ndarray:
        """Return the map of an earlier space's coordinates into the current space, by their level labels."""
        index = {tuple(int(level) for level in row): i for i, row in enumerate(self._spaces[-1])}
        labels = [tuple(int(level) for level in row) for row in levels]
        missing = [label for label in labels if label not in index]
        if missing:
            raise ValueError(f"The state's space has levels {missing[:3]} that the current space does not keep.")
        embedding = np.zeros((len(index), len(labels)))
        embedding[[index[label] for label in labels], np.arange(len(labels))] = 1.0
        return embedding

    def _embed(self, state: Any, embedding: np.ndarray) -> Any:
        """Put a state on the solver coordinates into the larger space of the loaded circuit."""
        if state is None or state.shape[0] != embedding.shape[1]:
            return state
        return self._lift(state, embedding, list(self._hamiltonian.dims[0]))

    @staticmethod
    def _lift(state: Any, embedding: np.ndarray, dims: list[int]) -> Any:
        """Map a ket or density matrix with an isometry onto a space of dimensions ``dims``."""
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

        On a chip whose only devices are the gate-set qubits, each with two
        levels, chip and register states have the same dimensions. There
        this returns the register state, and :meth:`run_state` maps it onto
        the chip.

        Parameters
        ----------
        init_circuit_state : qutip.Qobj, optional
            Register ket or density matrix in computational order.
        """
        n = self.num_qubits
        if init_circuit_state is None:
            init_circuit_state = qutip.basis([2] * n, [0] * n)
        self._columns()
        if list(self.dims) == [2] * n:
            return init_circuit_state
        return self._on_chip(init_circuit_state)

    def _on_chip(self, init_circuit_state: Any) -> Any:
        """Map a register ket or density matrix onto the dressed computational states of the chip."""
        columns = self._columns()
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
            raise RuntimeError("Load a circuit first, with load_circuit(qc) or run_state(qc=qc).")
        state = final_processor_state.full()
        n = self.num_qubits
        if final_processor_state.isket:
            return qutip.Qobj(self._chip_to_register @ state, dims=[[2] * n, [1] * n])
        return qutip.Qobj(self._chip_to_register @ state @ self._chip_to_register.conj().T, dims=[[2] * n, [2] * n])
