"""Channels of a sequence on the dressed states of a product of levels, with collapse channels."""

from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np

from quchip.results.propagator import _label_frame, _namespace
from quchip.utils.constants import TWO_PI
from quchip.utils.jax_utils import contains_tracer

_PAULIS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1.0, -1.0]).astype(complex),
}


@dataclass(frozen=True)
class ChannelResult:
    """Channel of a sequence on the dressed states of a product of levels.

    ``superoperator`` maps the column-stacked input ``vec(ρ)`` to the output
    ``vec(Λ(ρ))``, with ``vec(ρ)[i + d j] = ρ[i, j]`` for d labels. Its
    entry ``[i + d j, k + d l]`` is ``⟨d_i|Λ(|d_k⟩⟨d_l|)|d_j⟩`` in the record's
    frame, where ``|d_k⟩`` is the dressed state of label k under the solve's
    approximation and excitation cutoff and Λ is the evolution from the first
    saved time t₀ to the last saved time T with the chip's collapse channels.
    The frame is applied to both sides, as in
    :class:`~quchip.results.propagator.PropagatorResult`. Population that
    leaves the labels is lost from the record, so the channel decreases the
    trace when there is leakage.

    The computational labels are those whose levels are all 0 or 1. The
    infidelities, the error map and the Pauli quantities use the block of
    the channel on them.

    Attributes
    ----------
    superoperator : array_like, shape (d², d²)
        Column-stacked superoperator on the labels.
    labels : tuple of tuple of int
        Level of each selected device, for each label.
    devices : tuple of str
        Selected devices. Every other device starts and is projected in level 0.
    leakage : array_like, shape (d,)
        Probability that input label k leaves the labels,
        ``1 − Σᵢ Λ[(i, i), (k, k)]``. It does not depend on the frame.
    frame : {"dressed", "lab", "integration"}
        Frame of :attr:`superoperator`.
    frequencies : dict of str to scalar
        Frame frequency of each selected device, in GHz.
    times : tuple of scalar
        Start time t₀ and end time T, in ns.
    channels : tuple of str
        Collapse channels that the solves included.
    batch : SimulationBatchResult or None
        The pure-state solves: each label, then for each pair of labels i < j
        the inputs ``(|d_i⟩ + |d_j⟩)/√2`` and ``(|d_i⟩ + i|d_j⟩)/√2``.
    """

    superoperator: Any
    labels: tuple[tuple[int, ...], ...]
    devices: tuple[str, ...]
    leakage: Any
    frame: str
    frequencies: dict[str, Any]
    times: tuple[Any, Any]
    channels: tuple[str, ...]
    batch: Any

    def average_leakage(self) -> Any:
        """Return the average probability that a computational input leaves the computational labels.

        This is ``1 − Tr[P_C Λ(P_C/d_C)]`` for the projector P_C on the d_C
        computational labels (Wood and Gambetta, PRA 97, 032306 (2018)). It
        counts population that reaches labeled leakage levels and
        population that leaves the labels, such as an excited spectator.
        """
        xp = _namespace(self.superoperator)
        computational = self._computational()
        populations = self._populations(xp)
        kept = xp.sum(populations[np.ix_(computational, computational)])
        return 1.0 - kept / len(computational)

    def seepage(self) -> Any:
        """Return the seepage from the labeled leakage states: the average probability that they return.

        This is ``Tr[P_C Λ(P_L/d_L)]`` for the d_L labels with a level above
        1. Wood and Gambetta, PRA 97, 032306 (2018), average over the whole
        complement of the computational labels instead; states outside the
        labels, or outside the solve's levels, do not enter here.

        Raises
        ------
        ValueError
            If every label is computational.
        """
        xp = _namespace(self.superoperator)
        computational = self._computational()
        leaked = [index for index in range(len(self.labels)) if index not in computational]
        if not leaked:
            raise ValueError("Seepage needs labels with a level above 1, such as levels=(0, 1, 2).")
        populations = self._populations(xp)
        return xp.sum(populations[np.ix_(computational, leaked)]) / len(leaked)

    def infidelity(self, target: Any) -> Any:
        """Return 1 − F_avg against a unitary target, in quchip's frame, with ideal preparation and measurement.

        On the computational block, with process fidelity
        ``F_e = Tr(S_U† S_Λ)/d²`` and the :meth:`average_leakage` L,
        ``1 − F_avg = (d(1 − F_e) + L)/(d + 1)`` (Wood and Gambetta, PRA 97,
        032306 (2018)). Leaked population counts as zero overlap, and the
        block is not renormalized. The number takes the record's frame and
        ideal preparation and measurement in the dressed states as given, so
        a small frame rotation of the gate counts as error. Randomized
        benchmarking need not agree with it.

        Parameters
        ----------
        target : array_like, shape (d_C, d_C)
            Unitary target on the computational labels, in the frame and label
            order of this record.
        """
        xp = _namespace(self.superoperator, target)
        block, dimension = self._block(xp)
        target = self._target(xp, target, dimension)
        process = xp.real(xp.sum(xp.conj(_conjugation(xp, target)) * block)) / dimension**2
        return (dimension * (1.0 - process) + self.average_leakage()) / (dimension + 1)

    def eigenvalue_infidelity(self, target: Any) -> float:
        """Return the eigenvalue infidelity against a unitary target: a spectral score, not an operational fidelity.

        The eigenvalues λ of the computational block and μ of the target's
        superoperator are paired by a minimum-weight matching of ``|λ − μ|``.
        With ``F = |Σ conj(μ) λ|/d²`` the score is ``d(1 − F)/(d + 1)``, as
        pyGSTi's ``eigenvalue_avg_gate_infidelity``. A fixed change of basis
        within the computational labels, applied to the channel and the
        target, leaves it unchanged, so at first order it leaves out small
        rotations that only tilt a gate against preparation and measurement.
        A frame change that differs at the start and the end can change it.
        Leakage is not added, but it lowers the eigenvalues: a
        uniform loss ℓ gives ``dℓ/(d + 1)``. Report :meth:`average_leakage`
        beside the score. NumPy only.

        Parameters
        ----------
        target : array_like, shape (d_C, d_C)
            Unitary target on the computational labels, in the label order of
            this record.
        """
        from scipy.optimize import linear_sum_assignment

        block, dimension = self._block(np)
        target = self._target(np, target, dimension)
        found = np.linalg.eigvals(np.asarray(block))
        ideal = np.linalg.eigvals(_conjugation(np, target))
        rows, columns = linear_sum_assignment(np.abs(found[:, None] - ideal[None, :]))
        overlap = abs(np.sum(np.conj(ideal[columns]) * found[rows])) / dimension**2
        return float(dimension * (1.0 - overlap) / (dimension + 1))

    def error(self, target: Any) -> ChannelResult:
        """Return the error map after the ideal gate, ``E = Λ ∘ U⁻¹``, so that ``Λ = E ∘ U``.

        For a target on the computational labels, the map acts on them; for a
        target on every label, it acts on every label. Twirl the error map,
        not the gate channel, to get the Pauli channel of cycle error
        reconstruction.

        Parameters
        ----------
        target : array_like, shape (d_C, d_C) or (d, d)
            Unitary target in the frame and label order of this record.
        """
        xp = _namespace(self.superoperator, target)
        shape = np.shape(target)
        if shape == (len(self.labels),) * 2:
            superoperator, labels = xp.asarray(self.superoperator), self.labels
            _require_unitary(target)
        else:
            superoperator, dimension = self._block(xp)
            target = self._target(xp, target, dimension)
            labels = tuple(self.labels[index] for index in self._computational())
        superoperator = superoperator @ xp.conj(_conjugation(xp, xp.asarray(target, dtype=complex))).T
        return self._replace(xp, superoperator, labels)

    def ptm(self) -> Any:
        """Return the Pauli transfer matrix ``R[a, b] = Tr(P_a Λ(P_b))/d``.

        The Pauli products are unnormalized, with the first device most
        significant, in the order of :meth:`pauli_fidelities`. This is
        pyGSTi's ``pp`` basis. The channel preserves Hermiticity, so R is
        real.

        Raises
        ------
        ValueError
            If the labels are not every product of levels 0 and 1.
        """
        xp = _namespace(self.superoperator)
        names, vectors = self._paulis(xp)
        dimension = len(self.labels)
        return xp.real(xp.conj(vectors).T @ xp.asarray(self.superoperator) @ vectors) / dimension

    def pauli_fidelities(self) -> dict[str, Any]:
        """Return the diagonal of the :meth:`ptm`, ``f(P) = Tr(P Λ(P))/d``, by Pauli name.

        Names give one of I, X, Y, Z per device, first device first, such as
        ``"XI"``. On an error map, ``f(I…I)`` is 1 minus the average leakage.
        """
        names, _ = self._paulis(np)
        diagonal = _namespace(self.superoperator).diagonal(self.ptm())
        return dict(zip(names, diagonal))

    def pauli_rates(self) -> dict[str, Any]:
        """Return the Pauli error probabilities of the twirled map, by Pauli name.

        ``p(P) = 4⁻ⁿ Σ_Q (−1)^⟨P,Q⟩ f(Q)``, where ⟨P,Q⟩ is 1 when P and Q
        anticommute and f are the :meth:`pauli_fidelities`. On an error map
        ``p(I…I)`` is the weight of the identity in the twirled map, and the
        rates sum to ``f(I…I)``, which is below 1 when there is leakage.
        """
        xp = _namespace(self.superoperator)
        names, _ = self._paulis(np)
        fidelities = self.pauli_fidelities()
        signs = np.array([[-1.0 if _anticommute(p, q) else 1.0 for q in names] for p in names])
        rates = xp.asarray(signs) @ xp.stack([xp.asarray(fidelities[name]) for name in names]) / len(names)
        return dict(zip(names, rates))

    def twirl(self) -> ChannelResult:
        """Return the Pauli twirl ``4⁻ⁿ Σ_P P Λ(P ρ P) P``.

        Its :meth:`ptm` is the diagonal of this record's. Twirl the
        :meth:`error` map to get the Pauli channel that randomized compiling
        and cycle benchmarking see.
        """
        xp = _namespace(self.superoperator)
        names, _ = self._paulis(np)
        superoperator = xp.asarray(self.superoperator)
        total = 0.0
        for name in names:
            pauli = xp.asarray(_conjugation(np, _pauli(name)))
            total = total + pauli @ superoperator @ pauli
        return self._replace(xp, total / len(names), self.labels)

    def _computational(self) -> list[int]:
        computational = [index for index, levels in enumerate(self.labels)
                         if all(level in (0, 1) for level in levels)]
        if not computational:
            raise ValueError("The labels have no computational state; include levels 0 and 1.")
        return computational

    def _populations(self, xp: Any) -> Any:
        """Return ``Λ[(i, i), (k, k)]``: the population of label i for input label k."""
        dimension = len(self.labels)
        diagonal = [index * (dimension + 1) for index in range(dimension)]
        return xp.real(xp.asarray(self.superoperator)[np.ix_(diagonal, diagonal)])

    def _block(self, xp: Any) -> tuple[Any, int]:
        computational = self._computational()
        dimension = len(self.labels)
        indices = [i + dimension * j for j in computational for i in computational]
        return xp.asarray(self.superoperator)[np.ix_(indices, indices)], len(computational)

    def _target(self, xp: Any, target: Any, dimension: int) -> Any:
        target = xp.asarray(target, dtype=complex)
        if target.shape != (dimension, dimension):
            raise ValueError(f"The target has shape {target.shape}; the computational labels need "
                             f"{(dimension, dimension)}.")
        _require_unitary(target)
        return target

    def _paulis(self, xp: Any) -> tuple[list[str], Any]:
        count = len(self.devices)
        if self.labels != tuple(itertools.product((0, 1), repeat=count)):
            raise ValueError("Pauli quantities need the labels of levels (0, 1) on every selected device; "
                             "use error(target) with a target on the computational labels.")
        names = ["".join(letters) for letters in itertools.product("IXYZ", repeat=count)]
        vectors = np.stack([_pauli(name).reshape(-1, order="F") for name in names], axis=1)
        return names, xp.asarray(vectors)

    def _replace(self, xp: Any, superoperator: Any, labels: tuple[tuple[int, ...], ...]) -> ChannelResult:
        dimension = len(labels)
        diagonal = [index * (dimension + 1) for index in range(dimension)]
        leakage = 1.0 - xp.sum(xp.real(superoperator[np.ix_(diagonal, diagonal)]), axis=0)
        return dataclasses.replace(self, superoperator=superoperator, labels=labels, leakage=leakage)


def _require_unitary(target: Any) -> None:
    if contains_tracer(target):
        return
    matrix = np.asarray(target)
    if np.abs(matrix.conj().T @ matrix - np.eye(len(matrix))).max() > 1e-8:
        raise ValueError("The target must be unitary. For a leaky gate, use the unitary part of its block.")


def _pauli(name: str) -> np.ndarray:
    matrix = np.ones((1, 1), dtype=complex)
    for letter in name:
        matrix = np.kron(matrix, _PAULIS[letter])
    return matrix


def _anticommute(first: str, second: str) -> bool:
    return sum(a != "I" and b != "I" and a != b for a, b in zip(first, second)) % 2 == 1


def _conjugation(xp: Any, unitary: Any) -> Any:
    """Return the column-stacked superoperator of ``ρ ↦ U ρ U†``."""
    return xp.kron(xp.conj(unitary), unitary)


def subspace_channel(batch: Any, devices: tuple[str, ...], labels: tuple[tuple[int, ...], ...], frame: str,
                     channels: tuple[str, ...]) -> ChannelResult:
    """Return the channel of a batch whose points start in the pure inputs of ``labels``.

    The first d points start in the dressed states of ``labels``. For each
    pair i < j, in the order of ``itertools.combinations``, the next two
    start in ``(|d_i⟩ + |d_j⟩)/√2`` and ``(|d_i⟩ + i|d_j⟩)/√2``. Linearity
    gives the image of every ``|d_i⟩⟨d_j|``.

    Parameters
    ----------
    batch : SimulationBatchResult
        Solves that share one chip, approximation, frame and time grid.
    devices : tuple of str
        Selected devices, in the order of the levels of each label.
    labels : tuple of tuple of int
        Level of each selected device, for each label.
    frame : {"dressed", "lab", "integration"}
        Frame of the returned superoperator.
    channels : tuple of str
        Collapse channels that the solves included.
    """
    results, vectors, frequencies, _, shifts, (start, stop) = _label_frame(batch, devices, labels, frame)
    dimension = len(labels)
    pairs = list(itertools.combinations(range(dimension), 2))
    if len(results) != dimension + 2 * len(pairs):
        raise ValueError(f"Expected {dimension**2} solves, got {len(results)}.")
    backend = results[0]._backend
    finals = [backend.to_array(result._raw_final_state()) for result in results]
    xp = _namespace(vectors, finals)
    targets = xp.stack(vectors)
    shift = xp.stack([xp.asarray(value, dtype=float) for value in shifts])
    after = xp.exp(1j * TWO_PI * stop * shift)
    before = xp.exp(-1j * TWO_PI * start * shift)

    def project(final: Any) -> Any:
        if backend.is_ket(final) or final.ndim == 1 or final.shape[-1] == 1:
            amplitudes = xp.conj(targets) @ final.reshape(-1)
            block = amplitudes[:, None] * xp.conj(amplitudes)[None, :]
        else:
            block = xp.conj(targets) @ final @ targets.T
        return after[:, None] * block * xp.conj(after)[None, :]

    projected = [project(final) for final in finals]
    images: dict[tuple[int, int], Any] = {(k, k): projected[k] for k in range(dimension)}
    for index, (i, j) in enumerate(pairs):
        plus, plus_i = projected[dimension + 2 * index], projected[dimension + 2 * index + 1]
        image = plus + 1j * plus_i - 0.5 * (1 + 1j) * (images[(i, i)] + images[(j, j)])
        images[(i, j)], images[(j, i)] = image, xp.conj(image).T
    columns = [images[(i, j)] * before[i] * xp.conj(before[j]) for j in range(dimension) for i in range(dimension)]
    superoperator = xp.stack([column.T.reshape(-1) for column in columns], axis=1)
    diagonal = [index * (dimension + 1) for index in range(dimension)]
    leakage = 1.0 - xp.sum(xp.real(superoperator[np.ix_(diagonal, diagonal)]), axis=0)
    return ChannelResult(
        superoperator=superoperator,
        labels=tuple(labels),
        devices=tuple(devices),
        leakage=leakage,
        frame=frame,
        frequencies=frequencies,
        times=(start, stop),
        channels=tuple(channels),
        batch=batch,
    )
