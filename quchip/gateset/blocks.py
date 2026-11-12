"""Gate blocks on dressed computational states, projected from lab-frame propagators.

The gate set fits and composes gates on the dressed computational states of
``chip.state()``, in the dressed frame of PHYSICS.md §9.4 without its
ground-energy phase. :meth:`QuantumSequence.subspace_propagator
<quchip.control.sequence.QuantumSequence.subspace_propagator>` gives the same
matrix from one solve per input. These helpers take it from one propagator
solve and keep the leaked amplitudes that the calibration residual needs.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from quchip.results.propagator import _namespace, average_fidelity


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
        return float(average_fidelity(self.matrix, ideal))


def project_computational(
    image: Any, columns: Any, energies: Any, t0: Any, t1: Any, qubits: Sequence[str]
) -> ComputationalBlock:
    """Project lab-frame output states onto dressed computational states.

    Parameters
    ----------
    image : array
        ``(d, k)`` lab-frame states at ``t1`` in solver coordinates.
    columns, energies : array
        Computational kets and frame frequencies from :func:`computational_states`.
    t0 : float or None
        Start time whose input frame phase ``exp(-2πi e t0)`` multiplies each
        output, or ``None`` when the inputs are not the computational states.
    t1 : float
        End time of the outputs.
    qubits : sequence of str
        Qubit labels in matrix order.
    """
    xp = _namespace(image)
    block = xp.conj(columns).T @ image
    leaked = image - columns @ block
    if t0 is not None:
        start = xp.exp(-2j * np.pi * energies * t0)
        block = block * start[None, :]
        leaked = leaked * start[None, :]
    block = xp.exp(2j * np.pi * energies * t1)[:, None] * block
    leakage = xp.real(xp.sum(xp.conj(leaked) * leaked, axis=0))
    return ComputationalBlock(matrix=block, leaked=leaked, leakage=leakage, qubits=tuple(qubits))


def computational_states(
    problem: Any, qubits: Sequence[str], levels: np.ndarray, background: Mapping[str, int] | None = None
) -> tuple[Any, Any]:
    """Return dressed computational kets in solver coordinates and their frame frequencies.

    Each ket is ``chip.compact_state(label, dressed=True)`` scattered onto the
    solver coordinates, so the gauge matches ``chip.state()``. The frequency of
    a state is ``Σ_i f_i ñ_i`` with ``f_i = chip.freq(i)`` over its dressed label.

    Parameters
    ----------
    problem : SolveProblem
        Problem whose chip and coordinates define the states.
    qubits : sequence of str
        Computational qubit labels, first most significant.
    levels : numpy.ndarray
        ``(d, devices)`` level index of each solver coordinate.
    background : mapping or None, default None
        Dressed levels of other devices, which are otherwise in level 0.
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
    background = dict(background or {})
    overlap = set(background) & set(qubits)
    if overlap:
        raise ValueError(f"Background levels repeat computational qubits {sorted(overlap)}.")
    columns = []
    energies = []
    product_space = problem.engine_result.excitation_space is None
    conserving = not product_space or _conserves(chip)
    for bits in itertools.product((0, 1), repeat=len(qubits)):
        mapping = {**background, **dict(zip(qubits, bits))}
        if not conserving:
            # Without excitation conservation, the dressed state spans the whole product space.
            columns.append(xp.asarray(np.asarray(chip.state(mapping).full()).reshape(-1)))
            energies.append(sum(float(chip.freq(label)) * level for label, level in mapping.items()))
            continue
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
        energies.append(sum(float(chip.freq(label)) * level for label, level in mapping.items()))
    return xp.stack(columns, axis=1), xp.asarray(energies)


def _conserves(chip: Any) -> bool:
    """True when the chip's static Hamiltonian conserves the total excitation number."""
    from quchip.engine.excitation import excitation_structure

    return excitation_structure(chip, chip.approximation).conserves


def _scatter(column: Any, rows: list[int], values: Any) -> Any:
    column[np.array(rows)] = values
    return column
