"""Total excitation-number sectors of a conserving chip's static model.

When a chip's static model conserves the total energy-level index ``N``, its
Hamiltonian is block diagonal in the product energy basis.
:class:`SectorModel` builds the block of one ``N`` from each term's local
operator on its own support. Two product states couple only when they agree
outside that support, so no operator on the full product space is formed.
Dressed queries diagonalize only the sectors that their labels occupy. The
weak-probe VNA route reads its vacuum and one-excitation blocks the same way.
"""

from __future__ import annotations

import dataclasses
import functools
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Sequence

import jax.numpy as jnp
import numpy as np

from quchip.chip.dressing import BareProductReference, Labeling, assign_rowwise_greedy, label_eigensystem
from quchip.chip.ports import Port
from quchip.chip.sw import _sector_eigh
from quchip.declarative.expr import materialize_expr
from quchip.engine.assembly import (
    _interaction_contributions,
    _project_on_support,
    _resolve_system,
    _support_semantic_transform,
)
from quchip.utils.constants import TWO_PI
from quchip.utils.jax_utils import concrete_array_module, contains_tracer

if TYPE_CHECKING:
    from quchip.approximations import Approximation
    from quchip.chip.chip import Chip


@functools.lru_cache(maxsize=1024)
def _sector_levels(dims: tuple[int, ...], total: int) -> np.ndarray:
    """Return every level product below ``dims`` that sums to ``total``, in C order."""
    if not dims:
        return np.zeros((1 if total == 0 else 0, 0), dtype=np.int64)
    blocks = []
    for level in range(min(dims[0] - 1, total) + 1):
        rest = _sector_levels(dims[1:], total - level)
        if rest.shape[0]:
            blocks.append(np.column_stack((np.full(rest.shape[0], level, dtype=np.int64), rest)))
    levels = np.concatenate(blocks) if blocks else np.zeros((0, len(dims)), dtype=np.int64)
    levels.flags.writeable = False
    return levels


def sector_size(dims: Sequence[int], total: int) -> int:
    """Count the product states of ``dims`` whose levels sum to ``total``.

    Parameters
    ----------
    dims : sequence of int
        Level count of each device.
    total : int
        Total energy-level index.
    """
    counts = [1] + [0] * total
    for dimension in dims:
        counts = [sum(counts[n - level] for level in range(min(dimension - 1, n) + 1)) for n in range(total + 1)]
    return counts[total]


@dataclass(frozen=True, eq=False)
class SectorSpace:
    """Product energy states whose levels sum to one total.

    Attributes
    ----------
    dims : tuple[int, ...]
        Level count of each device, in chip order.
    total : int
        Total energy-level index of every state.
    levels : numpy.ndarray
        Levels with shape ``(size, len(dims))``, one state per row, in C order.
    """

    dims: tuple[int, ...]
    total: int
    levels: np.ndarray

    @property
    def size(self) -> int:
        """Number of product states in the sector."""
        return int(self.levels.shape[0])

    @functools.cached_property
    def labels(self) -> tuple[tuple[int, ...], ...]:
        """Bare labels of the states, in row order."""
        return tuple(map(tuple, self.levels.tolist()))

    @functools.cached_property
    def _rows(self) -> dict[tuple[int, ...], int]:
        return {label: row for row, label in enumerate(self.labels)}

    def row(self, label: tuple[int, ...]) -> int:
        """Return the row of a bare label in this sector.

        Parameters
        ----------
        label : tuple[int, ...]
            Level of each device, in chip order.
        """
        return self._rows[tuple(label)]


@dataclass(frozen=True)
class SectorEigensystem:
    """Labeled eigenpairs of one excitation sector.

    Attributes
    ----------
    space : SectorSpace
        Product states of the sector.
    eigenvalues : array_like
        Ascending eigenvalues in GHz.
    eigenvectors : array_like
        Eigenvectors as columns in the sector's product energy basis.
    labeling : Labeling
        Assignment of each bare label in ``space`` to one eigenvector.
    """

    space: SectorSpace
    eigenvalues: Any
    eigenvectors: Any
    labeling: Labeling

    def energy(self, label: tuple[int, ...]) -> Any:
        """Return the dressed energy assigned to a bare label, in GHz.

        Parameters
        ----------
        label : tuple[int, ...]
            Bare label in this sector.
        """
        index = self.labeling.indices[self.space.row(label)]
        values = self.eigenvalues
        if contains_tracer(index) and not contains_tracer(values):
            values = jnp.asarray(values)
        return values[index]


class SectorModel:
    """Static lab-frame model of a conserving chip, built one excitation sector at a time.

    The chip's static model must conserve the total energy-level index, as
    :func:`~quchip.chip.effective.conserves_excitation_number` checks. A block
    holds the device energies, each coupling band that the chip's approximation
    keeps, each retained term, and the Hamiltonian that cascaded ports generate.

    Parameters
    ----------
    chip : Chip
        Chip whose static model conserves the total energy-level index.
    resolution : optional
        Resolved local bases to reuse instead of resolving the devices again.
    """

    def __init__(self, chip: "Chip", *, resolution: Any = None) -> None:
        resolution = _resolve_system(chip, chip.backend) if resolution is None else resolution
        self.chip = chip
        self.bases = resolution.bases
        self.dims = tuple(int(dimension) for dimension in resolution.dims)
        self.network = resolution.network
        self.energies = tuple(self.bases[device.label].energies for device in chip.devices)
        self.terms = tuple(self._static_terms())
        self._ports: dict[str, tuple[Any, tuple[int, ...]]] | None = None
        self._spaces: dict[int, SectorSpace] = {}
        self._pairs: dict[tuple[tuple[int, ...], int, int],
                          tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        self._eigensystems: dict[int, SectorEigensystem] = {}

    @property
    def traced(self) -> bool:
        """Return whether the device energies or static terms hold JAX tracers."""
        return contains_tracer((self.energies, tuple(operator for operator, _ in self.terms)))

    @functools.cached_property
    def contributions(self) -> tuple[tuple[Any, ...], ...]:
        """Collapse contributions of the chip, with retained coordinate maps applied."""
        return tuple(self.chip._collapse_contributions_with_owners(self.bases))

    def space(self, total: int) -> SectorSpace:
        """Return the product states with total energy-level index ``total``.

        Parameters
        ----------
        total : int
            Total energy-level index.
        """
        if total not in self._spaces:
            self._spaces[total] = SectorSpace(self.dims, total, _sector_levels(self.dims, total))
        return self._spaces[total]

    def local_operator(self, operator: Any, support: tuple[int, ...]) -> Any:
        """Return an operator on ``support`` as a matrix in that support's product energy basis.

        Parameters
        ----------
        operator : PhysicsExpr or backend operator
            Operator in the authored coordinates of ``support``.
        support : tuple[int, ...]
            Device indices in the operator's tensor order. An empty support
            denotes an operator on the whole chip.
        """
        chip, backend = self.chip, self.chip.backend
        matrix = backend.to_array(_project_on_support(chip, operator, support, self.bases, backend))
        transform = _support_semantic_transform(chip, support or tuple(range(len(self.dims))), self.bases)
        if transform is not None:
            xp = concrete_array_module(matrix, transform)
            transform = xp.asarray(transform)
            matrix = transform.conj().T @ xp.asarray(matrix) @ transform
        return matrix if contains_tracer(matrix) else np.asarray(matrix, dtype=complex)

    def channel_operator(self, operator: Any, rate: Any, support: tuple[int, ...], phase: Any = None) -> Any:
        """Return a channel coupling ``exp(i phase) sqrt(rate) A`` in its support's energy basis.

        Parameters
        ----------
        operator : PhysicsExpr or backend operator
            Channel operator ``A`` on ``support``.
        rate : scalar or PhysicsExpr
            Channel rate.
        support : tuple[int, ...]
            Device indices of ``A``. An empty support denotes the whole chip.
        phase : scalar or None, default=None
            Port phase in radians.
        """
        matrix = self.local_operator(operator, support)
        rate = materialize_expr(rate, self.chip.backend)
        xp = concrete_array_module(matrix, rate, phase)
        scale = xp.sqrt(xp.asarray(rate))
        if phase is not None:
            scale = xp.exp(1j * xp.asarray(phase)) * scale
        return scale * matrix

    def port_couplings(self) -> dict[str, tuple[Any, tuple[int, ...]]]:
        """Return each port's coupling and support, keyed by port label."""
        if self._ports is None:
            self._ports = {
                owner.label: (self.channel_operator(operator, rate, support, owner.phase), support)
                for operator, rate, support, _source, _channel, _paths, owner in self.contributions
                if isinstance(owner, Port)
            }
        return self._ports

    def block(self, operator: Any, support: tuple[int, ...], row_total: int, column_total: int) -> Any:
        """Return the block of an operator from one sector to another.

        Parameters
        ----------
        operator : array_like
            Matrix on ``support`` in its product energy basis.
        support : tuple[int, ...]
            Device indices in the matrix's tensor order. An empty support
            denotes the whole chip.
        row_total, column_total : int
            Total energy-level index of the rows and of the columns.
        """
        rows, columns, local_rows, local_columns = self._pairs_for(support, row_total, column_total)
        shape = (self.space(row_total).size, self.space(column_total).size)
        if contains_tracer(operator):
            return jnp.zeros(shape, dtype=complex).at[rows, columns].set(operator[local_rows, local_columns])
        out = np.zeros(shape, dtype=complex)
        out[rows, columns] = np.asarray(operator)[local_rows, local_columns]
        return out

    def hamiltonian(self, total: int) -> Any:
        """Return the static lab-frame Hamiltonian block of one sector, in GHz.

        Parameters
        ----------
        total : int
            Total energy-level index of the sector.
        """
        space = self.space(total)
        generated = self._generated(total)
        xp = concrete_array_module(self.energies, tuple(operator for operator, _ in self.terms), generated)
        diagonal = xp.zeros(space.size)
        for index, energies in enumerate(self.energies):
            diagonal = diagonal + xp.asarray(energies)[space.levels[:, index]]
        hamiltonian = xp.diag(diagonal).astype(complex)
        gathered = [(self._pairs_for(support, total, total), operator) for operator, support in self.terms]
        if gathered:
            rows = np.concatenate([pairs[0] for pairs, _ in gathered])
            columns = np.concatenate([pairs[1] for pairs, _ in gathered])
            values = xp.concatenate([xp.asarray(operator)[pairs[2], pairs[3]] for pairs, operator in gathered])
            if xp is np:
                np.add.at(hamiltonian, (rows, columns), values)
            else:
                hamiltonian = hamiltonian.at[rows, columns].add(values)
        return hamiltonian if generated is None else hamiltonian + generated

    def eigensystem(self, total: int) -> SectorEigensystem:
        """Diagonalize one sector and assign its bare labels.

        Each sector is diagonalized once per model. A concrete block is
        diagonalized on the host, and a traced block uses one ``eigh``.

        Parameters
        ----------
        total : int
            Total energy-level index of the sector.
        """
        if total not in self._eigensystems:
            space = self.space(total)
            eigenvalues, eigenvectors = _sector_eigh(self.hamiltonian(total), np.full(space.size, total))
            labeling = label_eigensystem(eigenvectors, BareProductReference((space.size,)),
                                         policy=assign_rowwise_greedy)
            if not contains_tracer(eigenvalues):
                # Concrete energies take the backend's array type, as those of the full eigensystem do.
                eigenvalues = self.chip.backend.array_module.asarray(eigenvalues)
            self._eigensystems[total] = SectorEigensystem(
                space, eigenvalues, eigenvectors, dataclasses.replace(labeling, keys=space.labels),
            )
        return self._eigensystems[total]

    def labeled_subspace(self, labels: Sequence[tuple[int, ...]]) -> tuple[Any, Any, list[int], Any]:
        """Return the eigenpairs of the sectors that ``labels`` occupy and the labels' positions.

        The returned eigenvalues and block-diagonal eigenvector matrix hold the
        occupied sectors in increasing total. The list gives each label's row,
        and the array gives the column of its assigned eigenvector.

        Parameters
        ----------
        labels : sequence of tuple[int, ...]
            Bare labels, each with one level per device.
        """
        totals = sorted({sum(label) for label in labels})
        systems = {total: self.eigensystem(total) for total in totals}
        offsets: dict[int, int] = {}
        size = 0
        for total in totals:
            offsets[total] = size
            size += systems[total].space.size
        xp = concrete_array_module(tuple((system.eigenvectors, system.labeling.indices) for system in systems.values()))
        eigenvalues = xp.concatenate([xp.asarray(systems[total].eigenvalues) for total in totals])
        eigenvectors = xp.zeros((size, size), dtype=complex)
        for total in totals:
            span = slice(offsets[total], offsets[total] + systems[total].space.size)
            if xp is np:
                eigenvectors[span, span] = systems[total].eigenvectors
            else:
                eigenvectors = eigenvectors.at[span, span].set(systems[total].eigenvectors)
        rows = [systems[sum(label)].space.row(label) for label in labels]
        positions = [offsets[sum(label)] + row for label, row in zip(labels, rows, strict=True)]
        columns = xp.stack([
            offsets[sum(label)] + systems[sum(label)].labeling.indices[row]
            for label, row in zip(labels, rows, strict=True)
        ])
        return eigenvalues, eigenvectors, positions, columns

    def _static_terms(self) -> Iterator[tuple[Any, tuple[int, ...]]]:
        """Yield every coupling and retained term on its support, in the energy basis and GHz."""
        chip = self.chip
        approximation = chip.approximation
        for contribution in _interaction_contributions(chip):
            support = tuple(chip.device_index(label) for label in contribution.labels)
            operator = self.local_operator(contribution.expression(), support)
            if approximation.filters_terms and not contribution.retained:
                kept = operator * self._band_mask(support, approximation)
                if not contains_tracer(operator):
                    norm = np.linalg.norm(operator)
                    if norm > 0.0 and np.linalg.norm(kept) <= 1e-12 * norm:
                        warnings.warn(f"Coupling {contribution.label!r} vanishes entirely under RWA().",
                                      UserWarning, stacklevel=2)
                operator = kept
            yield operator, support

    def _band_mask(self, support: tuple[int, ...], approximation: "Approximation") -> np.ndarray:
        """Mark the matrix elements on ``support`` whose band the approximation keeps.

        An element's band weight on each device is its column level minus its
        row level, in the energy basis.
        """
        dims = tuple(self.dims[index] for index in support)
        levels = np.indices(dims).reshape(len(dims), -1).T
        weights = (levels[None, :, :] - levels[:, None, :]).reshape(-1, len(dims))
        bands, inverse = np.unique(weights, axis=0, return_inverse=True)
        kept = np.array([approximation.keeps_operator_band(tuple(int(weight) for weight in band)) for band in bands])
        return kept[inverse.reshape(-1)].reshape(levels.shape[0], levels.shape[0])

    def _generated(self, total: int) -> Any | None:
        """Return the Hamiltonian block that cascaded port pairs generate, in GHz, or ``None``.

        Each pair contributes ``(L_down^dagger c L_up - h.c.) / 2i``. Both port
        operators lower the total level index by one, so the block needs only
        the port blocks from sector ``total`` to sector ``total - 1``.
        """
        pairs = () if self.network is None else self.network.generated_pairs
        if not pairs or total == 0:
            return None
        ports = self.port_couplings()
        blocks: dict[str, Any] = {}
        for downstream, upstream, _ in pairs:
            for label in (downstream, upstream):
                if label not in blocks:
                    blocks[label] = self.block(*ports[label], total - 1, total)
        xp = concrete_array_module(tuple(blocks.values()), tuple(coefficient for *_, coefficient in pairs))
        size = self.space(total).size
        generated = xp.zeros((size, size), dtype=complex)
        for downstream, upstream, coefficient in pairs:
            product = xp.asarray(blocks[downstream]).conj().T @ (xp.asarray(coefficient) * xp.asarray(blocks[upstream]))
            generated = generated + (product - product.conj().T) / 2j
        return generated / TWO_PI

    def _pairs_for(self, support: tuple[int, ...], row_total: int,
                   column_total: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the sector rows and columns that an operator on ``support`` connects.

        Two product states connect when their levels agree outside the support.
        The result holds the rows, the columns, and the matching row and column
        indices of the local matrix.
        """
        key = (support, row_total, column_total)
        if key not in self._pairs:
            rows, columns = self.space(row_total), self.space(column_total)
            inside = list(support) if support else list(range(len(self.dims)))
            outside = [index for index in range(len(self.dims)) if index not in inside]
            local_dims = tuple(self.dims[index] for index in inside)
            if rows.size == 0 or columns.size == 0:
                empty = np.zeros(0, dtype=np.int64)
                self._pairs[key] = (empty, empty, empty, empty)
                return self._pairs[key]
            local_rows = np.ravel_multi_index(tuple(rows.levels[:, inside].T), local_dims)
            local_columns = np.ravel_multi_index(tuple(columns.levels[:, inside].T), local_dims)
            if outside:
                stacked = np.concatenate((rows.levels[:, outside], columns.levels[:, outside]))
                groups = np.unique(stacked, axis=0, return_inverse=True)[1].reshape(-1)
                row_groups, column_groups = groups[:rows.size], groups[rows.size:]
            else:
                row_groups = np.zeros(rows.size, dtype=np.int64)
                column_groups = np.zeros(columns.size, dtype=np.int64)
            order = np.argsort(column_groups, kind="stable")
            sorted_groups = column_groups[order]
            start = np.searchsorted(sorted_groups, row_groups, side="left")
            counts = np.searchsorted(sorted_groups, row_groups, side="right") - start
            pair_rows = np.repeat(np.arange(rows.size), counts)
            offsets = np.arange(pair_rows.size) - np.repeat(np.cumsum(counts) - counts, counts)
            pair_columns = order[np.repeat(start, counts) + offsets]
            self._pairs[key] = (pair_rows, pair_columns, local_rows[pair_rows], local_columns[pair_columns])
        return self._pairs[key]
