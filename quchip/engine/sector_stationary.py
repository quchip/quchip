"""Stationary states and output statistics of excitation-conserving chips on sectors.

A coherent drive ``i(c* L - c L†)`` couples every total excitation ``N`` to
``N ± 1``, so no finite set of sectors is closed. This module keeps the
product energy labels with ``N <= K`` and the projected model ``P H P`` and
``P L P`` on them. That model differs from the full model only by the drive
elements from ``N = K`` to ``N = K + 1``. Every operator comes from the terms'
local matrices on their supports, so the product space is never formed. The
backend supplies the generator that solves on these labels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from quchip.engine.linear_response import _UnsupportedLinearModel, _sector_response_model
from quchip.engine.reference import FieldChannel, cw_transfer
from quchip.engine.excitation import sector_levels, sector_size
from quchip.engine.sectors import _level_pairs
from quchip.utils.constants import TWO_PI
from quchip.utils.jax_utils import contains_tracer, maybe_concrete_scalar
from quchip.utils.memory import dense_bytes, fits_memory


class SectorProbeModel:
    """Sector operators of one conserving chip under coherent inputs, for any cutoff.

    Parameters
    ----------
    chip : Chip
        Chip with a static model that conserves the total excitation.
    response : _SectorResponseModel
        The chip's channels, scattering and reference planes on sectors.
    labels : tuple of str
        Selected output ports.
    inputs : tuple of str
        Ports that carry coherent inputs at one carrier frequency.
    modes : tuple of str
        Fock-mode devices whose amplitude and photon number are read.
    """

    def __init__(self, chip: Any, response: Any, labels: tuple[str, ...], inputs: tuple[str, ...],
                 modes: tuple[str, ...]) -> None:
        self.chip = chip
        self.response = response
        sectors = response.sectors
        self.sectors = sectors
        self.dims = sectors.dims
        self.max_total = sum(dimension - 1 for dimension in self.dims)
        self.keys = response.keys
        self.inputs = tuple(self.keys.index(label) for label in inputs)
        self.labels = labels
        self.modes = modes
        self.scattering = np.asarray(self.response.scattering, dtype=complex)
        self.fields = tuple(FieldChannel(key, reference, None)
                            for key, reference in zip(self.keys, self.response.references, strict=True))
        conserving = set(self.response.conserving_indices)
        self.lowering_indices = tuple(k for k in range(len(self.response.channels)) if k not in conserving)
        self.conserving_indices = tuple(sorted(conserving))
        self._operators: dict[int, tuple[Any, tuple[Any, ...], tuple[Any, ...]]] = {}
        self._generators: dict[int, Any] = {}

    def size(self, cutoff: int) -> int:
        """Return the number of product labels with ``N <= cutoff``."""
        return sum(sector_size(self.dims, total) for total in range(cutoff + 1))

    def required_memory(self, cutoff: int) -> float:
        """Return the bytes that a solve on ``N <= cutoff`` needs at the shortest Krylov restart."""
        return self.chip.backend.sector_solve_bytes(self.sector_sizes(cutoff), smallest=True)

    def peak_bytes(self, cutoff: int) -> int:
        """Return the estimated peak bytes of a solve on ``N <= cutoff``."""
        return self.chip.backend.sector_solve_bytes(self.sector_sizes(cutoff))

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

    def generator(self, cutoff: int) -> Any:
        """Return the backend's truncated generator on ``N <= cutoff``."""
        if cutoff not in self._generators:
            hamiltonian, channels, _ = self.operators(cutoff)
            self._generators[cutoff] = self.chip.backend.sector_generator(
                self.sector_sizes(cutoff), hamiltonian,
                [channels[k] for k in self.lowering_indices], [channels[k] for k in self.conserving_indices])
        return self._generators[cutoff]

    def unique(self, cutoff: int) -> bool:
        """Return whether the lowering channels prove one stationary state on ``N <= cutoff``."""
        _, channels, _ = self.operators(cutoff)
        return self.chip.backend.sector_unique(self.sector_sizes(cutoff),
                                               [channels[k] for k in self.lowering_indices])

    def steady_state(self, cutoff: int, solve: Any, frame: Any, diagnostics: Any) -> Any:
        """Return a solve on ``N <= cutoff`` as a steady-state result on those labels.

        ``frame`` is the frame of the coherent inputs, in which the solve is
        static. The lowering channels check that the state is unique, and an
        unproven state reports ``nullity=None``.
        """
        from types import MappingProxyType

        from quchip.engine.pulse_space import PulseSpace
        from quchip.results.steady_state import SteadyStateResult

        chip, backend = self.chip, self.chip.backend
        space = PulseSpace.create(self.dims, tuple(device.label for device in chip.devices),
                                  tuple(range(cutoff + 1)), self.sectors.bases)
        unique = not self.generator(cutoff).dark
        return SteadyStateResult(
            state=backend.from_array(solve.state, dims=[[space.size], [space.size]]),
            residual=solve.residual,
            nullity=1 if unique else None,
            _condition_number=None,
            dims=space.dims,
            device_info=tuple((device.label, device.computational) for device in chip.devices),
            stats=MappingProxyType({"iterations": solve.iterations, "uniqueness_checked": unique}),
            _backend=backend,
            _expectations=MappingProxyType({}),
            frame=frame,
            excitation_space=space,
            diagnostics=MappingProxyType(dict(diagnostics)),
        )

    def incident(self, frequency: float, amplitudes: Any) -> np.ndarray:
        """Return each input at its Markov boundary, in the physics convention and ``1/sqrt(ns)``."""
        return np.asarray([
            complex(amplitude) * complex(cw_transfer(self.response.references[index].inbound, frequency, np))
            for index, amplitude in zip(self.inputs, amplitudes, strict=True)], dtype=complex)

    def backgrounds(self, frequency: float, amplitudes: Any) -> np.ndarray:
        """Return the coherent field ``sum_i S_ki beta_i`` that the inputs carry to each channel ``k``."""
        if not self.inputs:
            return np.zeros(len(self.keys), dtype=complex)
        return self.scattering[:, list(self.inputs)] @ self.incident(frequency, amplitudes)

    def drive(self, frequency: float, amplitudes: Any) -> np.ndarray:
        """Return the drive coefficient ``c`` of each lowering channel."""
        return self.backgrounds(frequency, amplitudes)[list(self.lowering_indices)]

    def stationary(self, cutoff: int, frequency: float, amplitudes: Any, guess: np.ndarray | None = None) -> Any:
        """Solve the stationary state on ``N <= cutoff`` for coherent inputs at ``frequency``."""
        return self.generator(cutoff).stationary(frequency, self.drive(frequency, amplitudes), guess)

    def field(self, cutoff: int, frequency: float, amplitudes: Any, label: str) -> Any:
        """Return the output field ``h (b I + L)`` of one port on ``N <= cutoff`` as a sparse matrix.

        ``b`` is the coherent field that the inputs carry to the port, ``L``
        the port's channel and ``h`` its outbound transfer at ``frequency``.
        """
        _, channels, _ = self.operators(cutoff)
        index = self.keys.index(label)
        background = self.backgrounds(frequency, amplitudes)[index]
        gain = complex(cw_transfer(self.response.references[index].outbound, frequency, np))
        identity = sparse.identity(self.size(cutoff), dtype=complex, format="csr")
        return (gain * (background * identity + channels[index])).tocsr()

    def means(self, cutoff: int, frequency: float, amplitudes: Any, state: np.ndarray) -> np.ndarray:
        """Return ``<b_out>`` at the selected ports in the physics convention."""
        _, channels, _ = self.operators(cutoff)
        backgrounds = self.backgrounds(frequency, amplitudes)
        values = []
        for label in self.labels:
            index = self.keys.index(label)
            boundary = backgrounds[index] + channels[index].multiply(state.T).sum()
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

    def quadrature_spectrum(self, cutoff: int, frequency: float, amplitudes: Any, state: np.ndarray,
                            offsets: Any, starts: Any = None) -> tuple[np.ndarray, dict[str, Any], list[Any] | None]:
        """Return the normally ordered excess IQ spectrum of the selected ports, and its resolvent solutions.

        It follows :func:`quchip.engine.field_statistics.quadrature_spectrum`
        for vacuum inputs. The source of output ``p`` is ``(L_p - <L_p>) rho``.
        The adjoint source at ``f`` needs the resolvent at ``-f``, so the solve
        covers every offset and its negative. Only an exact negative reuses a
        point, so an exactly symmetric grid needs no extra solves. The third
        value holds the solution ``Y`` of each port at each solved offset, as
        square matrices, or ``None`` when they would not fit in memory with
        room to spare. Passing it as ``starts`` on a larger cutoff starts each
        solve there from it.
        """
        _, channels, _ = self.operators(cutoff)
        generator, drive = self.generator(cutoff), self.drive(frequency, amplitudes)
        offsets = np.asarray(offsets, dtype=float)
        grid, mirror = list(offsets), []
        for value in -offsets:
            match = np.flatnonzero(offsets == value)
            if match.size:
                mirror.append(int(match[0]))
            else:
                grid.append(float(value))
                mirror.append(len(grid) - 1)
        indices = [self.keys.index(label) for label in self.labels]
        observables = [operator for index in indices for operator in (channels[index], channels[index].conj().T)]
        count = len(indices)
        # values[f, r, p, 0] = Tr(L_r Y_p(f)) and values[f, r, p, 1] = Tr(L_r† Y_p(f)).
        solved_values = np.zeros((len(grid), count, count, 2), dtype=complex)
        iterations, residual, trace = 0, 0.0, 0.0
        size = self.size(cutoff)
        # The next cutoff solves beside the kept solutions, so they must fit with room to spare.
        keep = fits_memory(4 * dense_bytes(len(grid) * count, size, size))
        responses: list[Any] | None = [] if keep else None
        for p, index in enumerate(indices):
            lowering = channels[index]
            mean = lowering.multiply(state.T).sum()
            source = lowering @ state - mean * state
            solved, steps, error, drift, states = generator.resolvent(
                frequency, drive, source, grid, observables, None if starts is None else starts[p], keep)
            if responses is not None:
                responses.append([vector.reshape(size, size) for vector in states])
            solved_values[:, :, p, :] = solved.reshape(len(grid), count, 2)
            iterations, residual, trace = max(iterations, steps), max(residual, error), max(trace, drift)
        values, reverse = solved_values[:len(offsets)], solved_values[mirror]
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
        return excess, {"noise_iterations": iterations, "noise_residual": residual,
                        "noise_trace_error": trace}, responses

    def correlation(self, cutoff: int, frequency: float, amplitudes: Any, state: np.ndarray, order: int,
                    output: str, input: str, delays: Any) -> tuple[np.ndarray, float, float]:
        """Return a raw stationary output correlation and the two intensities, in the physics convention.

        ``G1(τ) = Tr(b_out† e^{Gτ}[b_in rho])`` and
        ``G2(τ) = Tr(b_out† b_out e^{Gτ}[b_in rho b_in†])``, where each field
        ``b`` keeps its coherent background as :meth:`field` gives it.
        """
        output_field = self.field(cutoff, frequency, amplitudes, output)
        input_field = self.field(cutoff, frequency, amplitudes, input)
        output_number = (output_field.conj().T @ output_field).tocsr()
        input_number = (input_field.conj().T @ input_field).tocsr()
        output_intensity = float(np.real(output_number.multiply(state.T).sum()))
        input_intensity = float(np.real(input_number.multiply(state.T).sum()))
        initial = input_field @ state
        if order == 2:
            initial = (input_field @ initial.conj().T).conj().T
        observable = output_field.conj().T.tocsr() if order == 1 else output_number
        raw = self.generator(cutoff).propagate(frequency, self.drive(frequency, amplitudes), initial, delays,
                                               [observable])[:, 0]
        return raw, input_intensity, output_intensity


def sector_probe_model(chip: Any, labels: tuple[str, ...], inputs: tuple[str, ...],
                       modes: tuple[str, ...]) -> tuple[SectorProbeModel | None, str]:
    """Return the sector model of a chip under coherent inputs, or ``None``, with the reason.

    The route needs a static model that conserves ``N``, ports that lower
    ``N`` by one, other channels that lower or conserve it, vacuum inputs, no
    thermal device occupation, no output network, no retained mode responses
    and concrete values.
    """
    network = chip.port_network
    if network is not None and network._mode_responses:
        return None, "The port network keeps retained mode responses."
    if contains_tracer(dict(chip.parameters)):
        return None, "The values are traced, and the sector solve runs on host NumPy arrays."
    thermal = [device.label for device in chip.devices
               if device.thermal_occupation is not None and maybe_concrete_scalar(device.thermal_occupation) != 0.0]
    if thermal:
        return None, f"Device {thermal[0]!r} has a thermal occupation, so its channels also raise N."
    try:
        response = _sector_response_model(chip, labels)
    except _UnsupportedLinearModel as error:
        return None, str(error) or "The model does not keep the excitation structure of the sector route."
    sectors = response.sectors
    references = tuple(value for reference in response.references
                       for element in (*reference.inbound, *reference.outbound) for value in element.tracked_values)
    if sectors.traced or contains_tracer((response.channels, response.scattering, references)):
        return None, "The values are traced, and the sector solve runs on host NumPy arrays."
    # Under a cutoff, the sector model exists for any chip; a conserving model keeps one total per class.
    if sectors.classes is not None and any(len(group) > 1 for group in sectors.classes):
        return None, "The static model does not conserve the total excitation N."
    return SectorProbeModel(chip, response, labels, inputs, modes), "The static model conserves N."
