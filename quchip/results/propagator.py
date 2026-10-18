"""Propagators of a sequence on the dressed states of a product of levels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from quchip.utils.constants import TWO_PI
from quchip.utils.jax_utils import contains_tracer

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
    if frame not in FRAMES:
        raise ValueError(f"frame must be one of {FRAMES}; got {frame!r}.")
    results = tuple(batch)
    if len(results) != len(labels):
        raise ValueError(f"Expected {len(labels)} solves, got {len(results)}.")
    first = results[0]
    targets = first._label_targets
    if targets is None or first.device_info is None:
        raise RuntimeError("These results have no captured chip for label targets; they were not created by a solve.")
    backend = first._backend
    order = tuple(label for label, _ in first.device_info)
    end = first.times[-1:]
    vectors, finals = [], []
    for levels, result in zip(labels, results):
        if not backend.is_ket(result._raw_final_state()):
            raise TypeError("A propagator needs ket solves.")
        fixed, moving = first._resolve_target(dict(zip(devices, levels)), "integration", end)
        vectors.append(backend.to_array(fixed).reshape(-1) if moving is None else moving[0])
        finals.append(backend.to_array(result._raw_final_state()).reshape(-1))
    xp = _namespace(vectors, finals)
    amplitudes = xp.conj(xp.stack(vectors)) @ xp.stack(finals).T
    leakage = 1.0 - xp.sum(xp.abs(amplitudes) ** 2, axis=0)

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
    shift = xp.stack([
        xp.asarray(ground + sum(frequencies[device] * level - integration[device] * level
                                for device, level in zip(devices, levels)), dtype=float)
        for levels in labels
    ])
    start, stop = first.times[0], first.times[-1]
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
