"""Propagators of scheduled sequences.

A :class:`Propagator` holds ``U(t1, t0)`` on a problem's solver coordinates:
the product of each device's local energy levels, or the retained labels of an
:class:`~quchip.cutoff.ExcitationCutoff`. The matrix is always stored in the
lab frame,

.. math::

    U_\\mathrm{lab}(t_1, t_0) = R(t_1)\\, U_\\mathrm{solver}(t_1, t_0)\\, R(t_0)^\\dagger,
    \\qquad R(t) = e^{-2\\pi i t \\sum_i f_i n_i},

where ``f_i`` are the integration-frame frequencies and ``n_i`` the level
indices. The integration frame therefore changes only cost and accuracy.
:meth:`Propagator.computational` expresses the gate on dressed computational
states in the frame that rotates with each device's dressed frequency.

A :class:`PropagatorResult` holds the propagator on the dressed states of a
product of levels, from one ket solve per input.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np

from quchip.utils.constants import TWO_PI
from quchip.utils.jax_utils import contains_tracer
from quchip.utils.labeling import resolve_label

FRAMES = ("dressed", "lab", "integration")


@dataclass(frozen=True)
class PropagatorResult:
    """Propagator of a sequence on the dressed states of a product of levels.

    ``matrix[i, j]`` is ``⟨d_i|U(T, t₀)|d_j⟩`` in the record's frame. Here
    ``|d_k⟩`` is the dressed state of label k under the solve's approximation
    and excitation cutoff, and U is the propagator from the first saved time
    t₀ to the last saved time T. A frame gives each label the energy
    ``E_k = ground_energy + Σᵢ fᵢ kᵢ``, with the frequencies fᵢ of
    :attr:`frequencies`, and its matrix is
    ``exp(i 2π T E_i) ⟨d_i|U|d_j⟩_lab exp(−i 2π t₀ E_j)``. The dressed frame
    turns each selected device at its dressed frequency and removes the
    dressed ground energy. An idle label with at most one excitation then
    keeps amplitude 1, and other labels turn at their residual shifts, such
    as the static ZZ of two qubits. The integration frame gives the label
    amplitudes of
    :meth:`~quchip.results.results.SimulationResult.amplitude_array`, and the
    lab frame keeps every energy.

    Attributes
    ----------
    matrix : array_like, shape (d, d)
        Final label in rows and initial label in columns.
    labels : tuple of tuple of int
        Level of each selected device, for each row and column.
    devices : tuple of str
        Selected devices. Every other device starts and is projected in level 0.
    leakage : array_like, shape (d,)
        Probability ``1 − Σᵢ|matrix[i, j]|²`` that input j leaves the labels. It
        does not depend on the frame.
    frame : {"dressed", "lab", "integration"}
        Frame of :attr:`matrix`.
    frequencies : dict of str to scalar
        Frame frequency of each selected device, in GHz.
    ground_energy : scalar
        Frame energy of the all-ground label, in GHz.
    times : tuple of scalar
        Start time t₀ and end time T, in ns.
    omitted_channels : tuple of str
        Collapse channels that the solve left out. The matrix is the
        propagator of the Hamiltonian alone.
    batch : SimulationBatchResult
        The solve of each input, in the order of :attr:`labels`.
    """

    matrix: Any
    labels: tuple[tuple[int, ...], ...]
    devices: tuple[str, ...]
    leakage: Any
    frame: str
    frequencies: dict[str, Any]
    ground_energy: Any
    times: tuple[Any, Any]
    omitted_channels: tuple[str, ...]
    batch: Any

    def fidelity(self, target: Any) -> Any:
        """Return the average gate fidelity of the propagator against a unitary target.

        For a target V of dimension d, the average over pure inputs in the
        labels is ``(Tr(M†M) + |Tr(V†M)|²) / (d(d + 1))``, with M the
        :attr:`matrix`. Leakage lowers ``Tr(M†M)``, so it lowers the fidelity.
        The target is written in the frame and label order of this record.

        Parameters
        ----------
        target : array_like, shape (d, d)
            Unitary target in the frame and label order of this record.
        """
        xp = _namespace(self.matrix, target)
        matrix = xp.asarray(self.matrix)
        target = xp.asarray(target, dtype=complex)
        dimension = matrix.shape[0]
        if target.shape != matrix.shape:
            raise ValueError(f"The target has shape {target.shape}; the propagator has shape {matrix.shape}.")
        kept = xp.real(xp.sum(xp.conj(matrix) * matrix))
        overlap = xp.abs(xp.sum(xp.conj(target) * matrix)) ** 2
        return (kept + overlap) / (dimension * (dimension + 1))


def _namespace(*values: Any) -> Any:
    if contains_tracer(values) or any(type(value).__module__.startswith("jax") for value in values):
        import jax.numpy as jnp

        return jnp
    return np


def _label_frame(batch: Any, devices: tuple[str, ...], labels: tuple[tuple[int, ...], ...], frame: str) -> Any:
    """Return the solves, the label targets at T, and the record frame of a batch that starts in ``labels``.

    The targets are integration-frame vectors at the last saved time T. Label
    k gets the shift ``E_k − F_k`` from the integration-frame energy F_k to
    the record frame's energy E_k.
    """
    if frame not in FRAMES:
        raise ValueError(f"frame must be one of {FRAMES}; got {frame!r}.")
    results = tuple(batch)
    if len(results) < len(labels):
        raise ValueError(f"Expected at least {len(labels)} solves, got {len(results)}.")
    first = results[0]
    targets = first._label_targets
    if targets is None or first.device_info is None:
        raise RuntimeError("These results have no captured chip for label targets; they were not created by a solve.")
    backend = first._backend
    order = tuple(label for label, _ in first.device_info)
    end = first.times[-1:]
    vectors = []
    for levels in labels:
        fixed, moving = first._resolve_target(dict(zip(devices, levels)), "integration", end)
        vectors.append(backend.to_array(fixed).reshape(-1) if moving is None else moving[0])

    frame_map = first._frame_map()
    positions = [order.index(device) for device in devices]
    integration = {device: frame_map.frequencies[position] for device, position in zip(devices, positions)}
    if frame == "integration":
        frequencies, ground = integration, 0.0
    elif frame == "lab":
        frequencies, ground = {device: 0.0 for device in devices}, 0.0
    else:
        vacuum = (0,) * len(order)
        ground = targets.energy(vacuum)
        frequencies = {}
        for device, position in zip(devices, positions):
            excited = list(vacuum)
            excited[position] = 1
            frequencies[device] = targets.energy(tuple(excited)) - ground
    # The integration frame gives label k the energy Σᵢ fᵢ kᵢ; the record's frame replaces it with its own.
    shifts = [ground + sum(frequencies[device] * level - integration[device] * level
                           for device, level in zip(devices, levels)) for levels in labels]
    return results, vectors, frequencies, ground, shifts, (first.times[0], first.times[-1])


def subspace_propagator(batch: Any, devices: tuple[str, ...], labels: tuple[tuple[int, ...], ...], frame: str,
                        omitted_channels: tuple[str, ...]) -> PropagatorResult:
    """Return the propagator of a batch whose points start in the dressed states of ``labels``.

    Point j of ``batch`` starts in the dressed state of ``labels[j]``. Row i
    of the integration-frame matrix projects every final state on the label
    target of ``labels[i]``.

    Parameters
    ----------
    batch : SimulationBatchResult
        Ket solves that share one chip, approximation, frame and time grid.
    devices : tuple of str
        Selected devices, in the order of the levels of each label.
    labels : tuple of tuple of int
        Level of each selected device, for each input.
    frame : {"dressed", "lab", "integration"}
        Frame of the returned matrix.
    omitted_channels : tuple of str
        Collapse channels that the solves left out.
    """
    results, vectors, frequencies, ground, shifts, (start, stop) = _label_frame(batch, devices, labels, frame)
    if len(results) != len(labels):
        raise ValueError(f"Expected {len(labels)} solves, got {len(results)}.")
    backend = results[0]._backend
    finals = []
    for result in results:
        if not backend.is_ket(result._raw_final_state()):
            raise TypeError("A propagator needs ket solves.")
        finals.append(backend.to_array(result._raw_final_state()).reshape(-1))
    xp = _namespace(vectors, finals)
    amplitudes = xp.conj(xp.stack(vectors)) @ xp.stack(finals).T
    leakage = 1.0 - xp.sum(xp.abs(amplitudes) ** 2, axis=0)
    shift = xp.stack([xp.asarray(value, dtype=float) for value in shifts])
    matrix = (xp.exp(1j * TWO_PI * stop * shift)[:, None] * amplitudes
              * xp.exp(-1j * TWO_PI * start * shift)[None, :])
    return PropagatorResult(
        matrix=matrix,
        labels=tuple(labels),
        devices=tuple(devices),
        leakage=leakage,
        frame=frame,
        frequencies=frequencies,
        ground_energy=ground,
        times=(start, stop),
        omitted_channels=tuple(omitted_channels),
        batch=batch,
    )


def solver_levels(engine_result: Any) -> np.ndarray:
    """Return the level index of every device for each solver coordinate, shape ``(d, devices)``.

    Parameters
    ----------
    engine_result : EngineResult
        Resolved problem description whose coordinates are listed.
    """
    space = engine_result.excitation_space
    if space is not None:
        return np.asarray(space.levels, dtype=int)
    return np.array(list(itertools.product(*(range(int(d)) for d in engine_result.dims))), dtype=int).reshape(
        -1, len(engine_result.dims)
    )


def _frame_phases(levels: np.ndarray, frequencies: np.ndarray, time: Any, xp: Any) -> Any:
    """Diagonal of ``R(t) = exp(-2πi t Σ f_i n_i)`` on the solver coordinates."""
    generator = levels @ frequencies
    return xp.exp(-2j * np.pi * time * xp.asarray(generator))


@dataclass(frozen=True, eq=False)
class ComputationalBlock:
    """A propagator restricted to dressed computational states.

    Attributes
    ----------
    matrix : array
        ``(2^n, k)`` amplitudes ``<b| V |input>`` in the computational frame, with
        the first qubit most significant. For a full propagator, ``k = 2^n``.
    leaked : array
        ``(d, k)`` amplitudes of each output outside the computational states,
        in solver coordinates and the lab frame.
    leakage : array
        Population of each output outside the computational states.
    qubits : tuple of str
        Qubit labels in matrix order.
    """

    matrix: Any
    leaked: Any
    leakage: Any
    qubits: tuple[str, ...]

    def fidelity(self, ideal: Any) -> float:
        """Return the average gate fidelity against an ideal unitary, global phase removed.

        The value is ``(Tr M M† + |Tr M|²) / (d (d + 1))`` with ``M = ideal† V``
        (Pedersen, Møller and Mølmer, Phys. Lett. A 367, 47 (2007)). Leakage
        lowers ``Tr M M†``.

        Parameters
        ----------
        ideal : array_like
            ``(2^n, 2^n)`` target unitary in the same qubit order.
        """
        ideal = np.asarray(ideal, dtype=complex)
        matrix = np.asarray(self.matrix)
        if matrix.shape != ideal.shape:
            raise ValueError(f"The block has shape {matrix.shape}; the ideal has shape {ideal.shape}.")
        product = ideal.conj().T @ matrix
        dimension = ideal.shape[0]
        return float((np.trace(product @ product.conj().T).real + abs(np.trace(product)) ** 2)
                     / (dimension * (dimension + 1)))


@dataclass(frozen=True, eq=False)
class Propagator:
    """Lab-frame propagator of a sequence over ``[t0, t1]``.

    Create it with :meth:`QuantumSequence.propagator
    <quchip.control.sequence.QuantumSequence.propagator>`.

    Attributes
    ----------
    matrix : array
        ``(d, d)`` unitary, ``(d, k)`` image of the input states, or
        ``(d², d²)`` superoperator on column-stacked density matrices.
    kind : {"unitary", "superoperator"}
        Closed or open evolution.
    t0, t1 : float
        Start and end times in ns.
    labels : tuple of str
        Device labels in coordinate order.
    levels : numpy.ndarray
        ``(d, devices)`` level index of each solver coordinate.
    dims : tuple of int
        Retained levels of each device.
    frame_frequencies : mapping
        Integration-frame frequency of each device in GHz.
    inputs : array or None
        ``(d, k)`` lab-frame input states at ``t0`` when columns were requested.
    options : mapping
        Solver options of the calculation.
    cutoff : int or None
        Excitation cutoff ``K`` of the solve, or ``None`` for the product space.
    """

    matrix: Any
    kind: str
    t0: float
    t1: float
    labels: tuple[str, ...]
    levels: np.ndarray
    dims: tuple[int, ...]
    frame_frequencies: Mapping[str, Any]
    inputs: Any = None
    options: Mapping[str, Any] = field(default_factory=dict)
    cutoff: int | None = None
    _problem: Any = field(default=None, repr=False)

    @classmethod
    def from_problem(cls, problem: Any, *, columns: Any = None) -> "Propagator":
        """Solve a prepared problem for its propagator.

        Parameters
        ----------
        problem : SolveProblem
            Deterministic problem built on the full retained coordinates.
        columns : array_like or None, default None
            ``(d, k)`` lab-frame input states at the first time.
        """
        backend = problem.backend
        xp = backend.array_module
        engine_result = problem.engine_result
        levels = solver_levels(engine_result)
        labels = tuple(device.label for device in problem.chip.devices)
        frame = problem.resolved_frame.frequencies
        frequencies = np.array([float(frame.get(label, 0.0)) for label in labels])
        times = xp.asarray(problem.tlist, dtype=float)
        t0, t1 = times[0], times[-1]
        start = _frame_phases(levels, frequencies, t0, xp)
        end = _frame_phases(levels, frequencies, t1, xp)
        inputs = None
        solver_columns = None
        if columns is not None:
            inputs = xp.asarray(columns, dtype=complex)
            if inputs.ndim == 1:
                inputs = inputs[:, None]
            if inputs.shape[0] != levels.shape[0]:
                raise ValueError(f"columns have {inputs.shape[0]} rows; the solver space has {levels.shape[0]}.")
            solver_columns = xp.conj(start)[:, None] * inputs
        native = backend.solve_propagator(problem, solver_columns)
        matrix = xp.asarray(native)
        if engine_result.collapse_terms:
            kind = "superoperator"
            # Column stacking: vec(R rho R†) = (R* ⊗ R) vec(rho), so element (i + d j) carries R_i conj(R_j).
            lift = lambda phases: (phases[:, None] * xp.conj(phases)[None, :]).T.reshape(-1)  # noqa: E731
            matrix = lift(end)[:, None] * matrix * xp.conj(lift(start))[None, :]
        else:
            kind = "unitary"
            matrix = end[:, None] * matrix if columns is not None else end[:, None] * matrix * xp.conj(start)[None, :]
        reduction = engine_result.metadata.get("excitation_reduction") or {}
        return cls(
            matrix=matrix,
            kind=kind,
            t0=t0,
            t1=t1,
            labels=labels,
            levels=levels,
            dims=tuple(int(d) for d in problem.chip.dims),
            frame_frequencies=dict(frame),
            inputs=inputs,
            options=dict(problem.options),
            cutoff=reduction.get("cutoff"),
            _problem=problem,
        )

    @cached_property
    def unitarity_defect(self) -> float | None:
        """Return ``max |U† U - I|`` over the computed columns, or ``None`` for a superoperator.

        For a block of input states, the reference is the inputs' own overlap
        matrix. A large defect means the integrator, not the physics, limits a
        fit.
        """
        if self.kind != "unitary":
            return None
        matrix = np.asarray(self.matrix)
        gram = matrix.conj().T @ matrix
        reference = np.eye(matrix.shape[1]) if self.inputs is None else (
            np.asarray(self.inputs).conj().T @ np.asarray(self.inputs))
        return float(np.abs(gram - reference).max())

    def solver_frame(self) -> Any:
        """Return the matrix in the problem's integration frame."""
        xp = self._problem.backend.array_module
        frequencies = np.array([float(self.frame_frequencies.get(label, 0.0)) for label in self.labels])
        start = _frame_phases(self.levels, frequencies, self.t0, xp)
        end = _frame_phases(self.levels, frequencies, self.t1, xp)
        if self.kind == "superoperator":
            lift = lambda phases: (phases[:, None] * xp.conj(phases)[None, :]).T.reshape(-1)  # noqa: E731
            return xp.conj(lift(end))[:, None] * self.matrix * lift(start)[None, :]
        if self.inputs is not None:
            return xp.conj(end)[:, None] * self.matrix
        return xp.conj(end)[:, None] * self.matrix * start[None, :]

    def computational(self, qubits: Sequence[Any]) -> ComputationalBlock:
        """Express a unitary propagator on dressed computational states.

        The computational states hold each listed qubit in dressed level 0 or 1
        and every other device in dressed level 0, with the first qubit most
        significant. Their phases follow ``chip.state()``. The frame removes
        each device's dressed precession, so an idle interval leaves only the
        static ZZ phases:

        .. math::

            V = W_c^\\dagger e^{2\\pi i F t_1} U(t_1, t_0) e^{-2\\pi i F t_0} W_c,
            \\qquad F = \\sum_i f_i \\tilde n_i,

        with ``f_i = chip.freq(i)`` and ``ñ_i`` the dressed level label. A full
        propagator gives the square block on the computational inputs. A
        propagator computed from ``columns`` gives ``<b| e^{2πiF t1} U |input>``
        for its own inputs, with the input frame phase applied when the inputs
        are these computational states.

        Parameters
        ----------
        qubits : sequence of str or device
            Computational qubits in matrix order.
        """
        if self.kind != "unitary":
            raise ValueError("computational() needs a unitary propagator; pass dissipation=False.")
        names = tuple(resolve_label(qubit) for qubit in qubits)
        columns, energies = computational_states(self._problem, names, self.levels)
        xp = self._problem.backend.array_module
        start = xp.exp(-2j * np.pi * energies * self.t0)
        end = xp.exp(2j * np.pi * energies * self.t1)
        if self.inputs is None:
            image = self.matrix @ columns
            inputs_phase = start
        else:
            image = self.matrix
            inputs = np.asarray(self.inputs)
            same = inputs.shape == np.shape(columns) and np.allclose(inputs, np.asarray(columns), atol=1e-12)
            inputs_phase = start if same else None
        block = xp.conj(columns).T @ image
        leaked = image - columns @ block
        if inputs_phase is not None:
            block = block * inputs_phase[None, :]
            leaked = leaked * inputs_phase[None, :]
        block = end[:, None] * block
        leakage = xp.real(xp.sum(xp.conj(leaked) * leaked, axis=0))
        return ComputationalBlock(matrix=block, leaked=leaked, leakage=leakage, qubits=names)


def computational_states(problem: Any, qubits: Sequence[str], levels: np.ndarray) -> tuple[Any, Any]:
    """Return dressed computational kets in solver coordinates and their frame frequencies.

    Each ket is ``chip.compact_state(label, dressed=True)`` scattered onto the
    solver coordinates, so the gauge matches ``chip.state()``. The frequency of
    a computational state is ``Σ_q f_q b_q`` with ``f_q = chip.freq(q)``.

    Parameters
    ----------
    problem : SolveProblem
        Problem whose chip and coordinates define the states.
    qubits : sequence of str
        Computational qubit labels, first most significant.
    levels : numpy.ndarray
        ``(d, devices)`` level index of each solver coordinate.
    """
    chip = problem.chip
    xp = problem.backend.array_module
    labels = [device.label for device in chip.devices]
    unknown = [label for label in qubits if label not in labels]
    if unknown:
        raise ValueError(f"Unknown qubits {unknown}. Devices: {labels}.")
    if len(set(qubits)) != len(qubits):
        raise ValueError(f"Qubits must be distinct, got {list(qubits)}.")
    positions = {tuple(row): index for index, row in enumerate(levels.tolist())}
    frequencies = np.array([float(chip.freq(label)) for label in qubits])
    columns = []
    energies = []
    for bits in itertools.product((0, 1), repeat=len(qubits)):
        mapping = dict(zip(qubits, bits))
        state = chip.compact_state(mapping, dressed=True)
        values = xp.asarray(state.values).reshape(-1)
        rows = []
        for row in np.asarray(state.space.levels).tolist():
            index = positions.get(tuple(row))
            if index is None:
                raise ValueError(
                    f"The dressed state {mapping} reaches the label {tuple(row)}, which the solve does not retain."
                )
            rows.append(index)
        column = xp.zeros(levels.shape[0], dtype=complex)
        column = column.at[np.array(rows)].set(values) if hasattr(column, "at") else _scatter(column, rows, values)
        columns.append(column)
        energies.append(float(frequencies @ np.array(bits, dtype=float)))
    return xp.stack(columns, axis=1), xp.asarray(energies)


def _scatter(column: Any, rows: list[int], values: Any) -> Any:
    column[np.array(rows)] = values
    return column
