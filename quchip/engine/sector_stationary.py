"""Finite-probe stationary states of excitation-conserving chips on sectors.

A coherent probe ``i(c* L - c L†)`` couples every total excitation ``N`` to
``N ± 1``, so no finite set of sectors is closed. This module keeps the
product energy labels with ``N <= K`` and the projected model ``P H P`` and
``P L P`` on them. That model differs from the full model only by the probe
elements from ``N = K`` to ``N = K + 1``. Every operator comes from the terms'
local matrices on their supports, so the product space is never formed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from quchip.backend._sector_stationary import SectorGenerator, SectorSolve
from quchip.engine.linear_response import _UnsupportedLinearModel, _sector_response_model
from quchip.engine.reference import FieldChannel, cw_transfer
from quchip.engine.excitation import sector_levels, sector_size
from quchip.engine.sectors import _level_pairs
from quchip.utils.constants import TWO_PI
from quchip.utils.jax_utils import contains_tracer, maybe_concrete_scalar


class SectorProbeModel:
    """Sector operators of one conserving chip under a finite probe, for any cutoff.

    Parameters
    ----------
    chip : Chip
        Chip with a static model that conserves the total excitation.
    response : _SectorResponseModel
        The chip's channels, scattering and reference planes on sectors.
    labels : tuple of str
        Selected output ports.
    input_label : str
        Probe input port.
    modes : tuple of str
        Fock-mode devices whose amplitude and photon number are read.
    """

    def __init__(self, chip: Any, response: Any, labels: tuple[str, ...], input_label: str,
                 modes: tuple[str, ...]) -> None:
        self.chip = chip
        self.response = response
        sectors = response.sectors
        self.sectors = sectors
        self.dims = sectors.dims
        self.max_total = sum(dimension - 1 for dimension in self.dims)
        self.keys = response.keys
        self.probe = self.keys.index(input_label)
        self.labels = labels
        self.modes = modes
        self.scattering = np.asarray(self.response.scattering, dtype=complex)
        self.fields = tuple(FieldChannel(key, reference, None)
                            for key, reference in zip(self.keys, self.response.references, strict=True))
        conserving = set(self.response.conserving_indices)
        self.lowering_indices = tuple(k for k in range(len(self.response.channels)) if k not in conserving)
        self.conserving_indices = tuple(sorted(conserving))
        self._operators: dict[int, tuple[Any, tuple[Any, ...], tuple[Any, ...]]] = {}
        self._generators: dict[int, SectorGenerator] = {}

    def size(self, cutoff: int) -> int:
        """Return the number of product labels with ``N <= cutoff``."""
        return sum(sector_size(self.dims, total) for total in range(cutoff + 1))

    def required_memory(self, cutoff: int) -> float:
        """Return the bytes that a solve on ``N <= cutoff`` needs at the shortest Krylov restart."""
        from quchip.backend._sector_stationary import _MINIMUM_RESTART, sector_solve_bytes

        return sector_solve_bytes(self.sector_sizes(cutoff), _MINIMUM_RESTART)

    def sector_sizes(self, cutoff: int) -> tuple[int, ...]:
        """Return the size of each sector ``N = 0..cutoff``."""
        return tuple(sector_size(self.dims, total) for total in range(cutoff + 1))

    def _levels(self, cutoff: int) -> np.ndarray:
        return np.concatenate([sector_levels(self.dims, total) for total in range(cutoff + 1)])

    def _sparse(self, levels: np.ndarray, matrix: Any, support: tuple[int, ...]) -> Any:
        """Return an operator on ``support`` as a sparse matrix on the retained labels."""
        rows, columns, local_rows, local_columns = _level_pairs(self.dims, support, levels, levels)
        values = np.asarray(matrix, dtype=complex)[local_rows, local_columns]
        keep = values != 0
        size = len(levels)
        return sparse.csr_matrix((values[keep], (rows[keep], columns[keep])), shape=(size, size))

    def operators(self, cutoff: int) -> tuple[Any, tuple[Any, ...], tuple[Any, ...]]:
        """Return the Hamiltonian, the channels and the mode operators on ``N <= cutoff``.

        The Hamiltonian is in rad/ns, in the lab frame and relative to the
        vacuum energy. The channels follow the SLH order. Each mode has its
        lowering operator and its number operator.
        """
        if cutoff in self._operators:
            return self._operators[cutoff]
        from quchip.engine.observables import prepare_local_op
        from quchip.engine.assembly import _energy_operator

        sectors, levels = self.sectors, self._levels(cutoff)
        size = len(levels)
        diagonal = np.zeros(size)
        for index, energies in enumerate(sectors.energies):
            diagonal = diagonal + np.asarray(energies, dtype=float)[levels[:, index]]
        hamiltonian = sparse.diags(diagonal - diagonal[0]).astype(complex).tocsr()
        for operator, support in sectors.terms:
            hamiltonian = hamiltonian + self._sparse(levels, operator, support)
        channels = tuple(
            sum((complex(coefficient) * self._sparse(levels, matrix, support)
                 for coefficient, matrix, support in parts), start=sparse.csr_matrix((size, size), dtype=complex))
            for parts in self.response.channels)
        backend, labels = self.chip.backend, tuple(device.label for device in self.chip.devices)
        modes = []
        for label in self.modes:
            index, device = labels.index(label), self.chip.device_map[label]
            local = [_energy_operator(prepare_local_op(device, spec, sectors.bases[label], backend), (label,),
                                      sectors.bases, backend) for spec in ("a", "n")]
            modes.append(tuple(self._sparse(levels, matrix, (index,)) for matrix in local))
        self._operators[cutoff] = (TWO_PI * hamiltonian.tocsr(), channels, tuple(modes))
        return self._operators[cutoff]

    def generator(self, cutoff: int) -> SectorGenerator:
        """Return the truncated generator on ``N <= cutoff``."""
        if cutoff not in self._generators:
            hamiltonian, channels, _ = self.operators(cutoff)
            self._generators[cutoff] = SectorGenerator(
                self.sector_sizes(cutoff), hamiltonian,
                [channels[k] for k in self.lowering_indices], [channels[k] for k in self.conserving_indices])
        return self._generators[cutoff]

    def incident(self, frequency: float, amplitude: complex) -> complex:
        """Return the probe at the Markov boundary, in the physics convention and ``1/sqrt(ns)``."""
        inbound = self.response.references[self.probe].inbound
        return complex(amplitude) * complex(cw_transfer(inbound, frequency, np))

    def drive(self, frequency: float, amplitude: complex) -> np.ndarray:
        """Return the probe coefficient ``c`` of each lowering channel."""
        coefficients = self.scattering[:, self.probe] * self.incident(frequency, amplitude)
        return coefficients[list(self.lowering_indices)]

    def stationary(self, cutoff: int, frequency: float, amplitude: complex,
                   guess: np.ndarray | None = None) -> SectorSolve:
        """Solve the stationary state on ``N <= cutoff`` for one probe."""
        return self.generator(cutoff).stationary(frequency, self.drive(frequency, amplitude), guess)

    def means(self, cutoff: int, frequency: float, amplitude: complex, state: np.ndarray) -> np.ndarray:
        """Return ``<b_out>`` at the selected ports in the physics convention."""
        _, channels, _ = self.operators(cutoff)
        incident = self.incident(frequency, amplitude)
        values = []
        for label in self.labels:
            index = self.keys.index(label)
            boundary = self.scattering[index, self.probe] * incident + channels[index].multiply(state.T).sum()
            values.append(complex(cw_transfer(self.response.references[index].outbound, frequency, np)) * boundary)
        return np.asarray(values)

    def mode_values(self, cutoff: int, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``<a>`` and ``<a†a>`` of each mode in the common probe frame."""
        _, _, modes = self.operators(cutoff)
        amplitudes = np.asarray([lowering.multiply(state.T).sum() for lowering, _ in modes], dtype=complex)
        numbers = np.asarray([number.multiply(state.T).sum().real for _, number in modes], dtype=float)
        return amplitudes, numbers

    def weights(self, cutoff: int, state: np.ndarray) -> tuple[float, ...]:
        """Return the population of each total excitation ``N = 0..cutoff``."""
        offsets = np.concatenate(([0], np.cumsum(self.sector_sizes(cutoff))))
        diagonal = np.real(np.diag(state))
        return tuple(float(np.sum(diagonal[offsets[n]:offsets[n + 1]])) for n in range(cutoff + 1))

    def quadrature_spectrum(self, cutoff: int, frequency: float, amplitude: complex, state: np.ndarray,
                            offsets: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Return the normally ordered excess IQ spectrum of the selected ports.

        It follows :func:`quchip.engine.field_statistics.quadrature_spectrum`
        for vacuum inputs. The source of output ``p`` is ``(L_p - <L_p>) rho``.
        A resolvent at ``-f`` gives the adjoint source at ``f``, so each output
        needs one solve per offset.
        """
        _, channels, _ = self.operators(cutoff)
        generator, drive = self.generator(cutoff), self.drive(frequency, amplitude)
        offsets = np.asarray(offsets, dtype=float)
        indices = [self.keys.index(label) for label in self.labels]
        observables = [operator for index in indices for operator in (channels[index], channels[index].conj().T)]
        count = len(indices)
        # values[f, r, p, 0] = Tr(L_r Y_p(f)) and values[f, r, p, 1] = Tr(L_r† Y_p(f)).
        values = np.zeros((len(offsets), count, count, 2), dtype=complex)
        iterations, residual, trace = 0, 0.0, 0.0
        for p, index in enumerate(indices):
            lowering = channels[index]
            mean = lowering.multiply(state.T).sum()
            source = lowering @ state - mean * state
            solved, steps, error, drift = generator.resolvent(frequency, drive, source, offsets, observables)
            values[:, :, p, :] = solved.reshape(len(offsets), count, 2)
            iterations, residual, trace = max(iterations, steps), max(residual, error), max(trace, drift)
        reverse = values[::-1]
        factors = (1.0, -1j)
        causal = np.zeros((len(offsets), 2 * count, 2 * count), dtype=complex)
        for r in range(count):
            for a, fa in enumerate(factors):
                for p in range(count):
                    direct = (fa * values[:, r, p, 0] + np.conj(fa) * values[:, r, p, 1]) / 2
                    adjoint = (fa * np.conj(reverse[:, r, p, 1]) + np.conj(fa) * np.conj(reverse[:, r, p, 0])) / 2
                    for b, fb in enumerate(factors):
                        # The regression source is minus the conditioned emission.
                        causal[:, 2 * r + a, 2 * p + b] = -(fb * direct + np.conj(fb) * adjoint) / 2
        excess = causal + np.conj(np.swapaxes(causal, -1, -2))
        return excess, {"noise_iterations": iterations, "noise_residual": residual, "noise_trace_error": trace}


def try_sector_probe_model(chip: Any, labels: tuple[str, ...], input_label: str,
                           modes: tuple[str, ...]) -> SectorProbeModel | None:
    """Return the sector model of a chip under a finite probe, or ``None`` when it does not apply.

    The route needs a static model that conserves ``N``, ports that lower
    ``N`` by one, other channels that lower or conserve it, vacuum inputs, no
    thermal device occupation, no output network, no retained mode responses
    and concrete values.
    """
    network = chip.port_network
    if network is None or network._mode_responses:
        return None
    if any(device.thermal_occupation is not None and maybe_concrete_scalar(device.thermal_occupation) != 0.0
           for device in chip.devices):
        return None
    try:
        response = _sector_response_model(chip, labels)
    except _UnsupportedLinearModel:
        return None
    sectors = response.sectors
    references = tuple(value for reference in response.references
                       for element in (*reference.inbound, *reference.outbound) for value in element.tracked_values)
    if sectors.traced or contains_tracer((response.channels, response.scattering, references)):
        return None
    # Under a cutoff, the sector model exists for any chip; a conserving model keeps one total per class.
    if sectors.classes is not None and any(len(group) > 1 for group in sectors.classes):
        return None
    return SectorProbeModel(chip, response, labels, input_label, modes)
