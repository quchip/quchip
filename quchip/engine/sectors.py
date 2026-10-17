"""Excitation sectors and cutoff classes of a chip's static model.

A product energy label gives each device one isolated local energy level, and
its total excitation is ``N``. When a chip's static model conserves ``N``, its
Hamiltonian is block diagonal in the product energy basis, one block per
``N``. Under an :class:`~quchip.cutoff.ExcitationCutoff`, the retained static
model ``P H P`` is block diagonal in classes: sets of totals in ``[0, K]`` that
the static terms' changes of ``N`` connect. :class:`SectorModel` builds each
block from each term's local operator on its own support. Two product states
couple only when they agree outside that support, so no operator on the full
product space is formed. Dressed queries diagonalize only the blocks that
their labels occupy. The weak-probe VNA route reads its vacuum and
one-excitation blocks the same way.
"""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Any, Sequence

import jax.numpy as jnp
import numpy as np

from quchip.chip.dressing import (
    BareProductReference,
    Labeling,
    assign_rowwise_greedy,
    gauge_phases,
    label_eigensystem,
)
from quchip.chip.ports import Port
from quchip.chip.sw import _sector_eigh
from quchip.declarative.expr import materialize_expr
from quchip.engine.assembly import (
    _energy_operator,
    _resolved_energy_interactions,
    _project_on_support,
    _resolve_system,
)
from quchip.utils.constants import TWO_PI
from quchip.utils.jax_utils import concrete_array_module, contains_tracer
from quchip.utils.values import scoped_entry, scoped_hit

if TYPE_CHECKING:
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


def _level_pairs(dims: Sequence[int], support: tuple[int, ...], row_levels: np.ndarray,
                 column_levels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the rows and columns that an operator on ``support`` connects.

    Two product states connect when their levels agree outside the support.
    The result holds the rows, the columns, and the matching row and column
    indices of the local matrix. An empty support denotes the whole chip.
    """
    if len(row_levels) == 0 or len(column_levels) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty, empty
    inside = list(support) if support else list(range(len(dims)))
    outside = [index for index in range(len(dims)) if index not in inside]
    local_dims = tuple(dims[index] for index in inside)
    local_rows = np.ravel_multi_index(tuple(row_levels[:, inside].T), local_dims)
    local_columns = np.ravel_multi_index(tuple(column_levels[:, inside].T), local_dims)
    if outside:
        stacked = np.concatenate((row_levels[:, outside], column_levels[:, outside]))
        groups = np.unique(stacked, axis=0, return_inverse=True)[1].reshape(-1)
        row_groups, column_groups = groups[:len(row_levels)], groups[len(row_levels):]
    else:
        row_groups = np.zeros(len(row_levels), dtype=np.int64)
        column_groups = np.zeros(len(column_levels), dtype=np.int64)
    order = np.argsort(column_groups, kind="stable")
    sorted_groups = column_groups[order]
    start = np.searchsorted(sorted_groups, row_groups, side="left")
    counts = np.searchsorted(sorted_groups, row_groups, side="right") - start
    pair_rows = np.repeat(np.arange(len(row_levels)), counts)
    offsets = np.arange(pair_rows.size) - np.repeat(np.cumsum(counts) - counts, counts)
    pair_columns = order[np.repeat(start, counts) + offsets]
    return pair_rows, pair_columns, local_rows[pair_rows], local_columns[pair_columns]


def excitation_classes(changes: frozenset[int] | None, maximum: int) -> tuple[tuple[int, ...], ...]:
    """Partition the totals ``0..maximum`` into the classes that ``changes`` connect.

    A class holds the totals that steps ``n -> n +- c`` with every
    intermediate total in ``[0, maximum]`` connect. Unknown changes give one
    class.

    Parameters
    ----------
    changes : frozenset of int or None
        Changes of the total energy-level index, or ``None`` when unknown.
    maximum : int
        Largest retained total.
    """
    if changes is None:
        return (tuple(range(maximum + 1)),)
    steps = sorted({abs(change) for change in changes} - {0})
    seen: set[int] = set()
    classes: list[tuple[int, ...]] = []
    for start in range(maximum + 1):
        if start in seen:
            continue
        members, frontier = {start}, [start]
        while frontier:
            total = frontier.pop()
            for step in steps:
                for reached in (total - step, total + step):
                    if 0 <= reached <= maximum and reached not in members:
                        members.add(reached)
                        frontier.append(reached)
        seen |= members
        classes.append(tuple(sorted(members)))
    return tuple(classes)


def generated_products(ports: dict[str, tuple[Any, tuple[int, ...]]], pairs: Sequence[tuple[str, str, Any]],
                       dims: Sequence[int]) -> tuple[tuple[Any, tuple[int, ...]], ...]:
    """Return the Hamiltonian of cascaded port pairs as local terms on joint supports.

    Each pair contributes ``(L_down^dagger c L_up - h.c.) / 2i``, in the units
    of the port couplings' product. The product is formed on the joint support
    of the two ports, so every intermediate level of those devices is kept.
    Gathering a term onto any set of product labels therefore gives
    ``P H_gen P`` exactly. A support whose elements are all below the
    round-off threshold of their products contributes no term.

    Parameters
    ----------
    ports : dict
        Each port's coupling matrix and support, keyed by port label. All
        couplings use the same local coordinates.
    pairs : sequence of tuple
        ``(downstream, upstream, coefficient)`` of each generated pair.
    dims : sequence of int
        Level count of each device.
    """
    from quchip.chip.port_network import _cascade_hamiltonian

    everything = tuple(range(len(dims)))
    products: dict[tuple[int, ...], list[tuple[Any, Any, Any]]] = {}
    for downstream, upstream, coefficient in pairs:
        (left, left_support), (right, right_support) = ports[downstream], ports[upstream]
        left_support, right_support = left_support or everything, right_support or everything
        joint = tuple(sorted({*left_support, *right_support}))
        products.setdefault(joint, []).append(
            (_widen(left, left_support, joint, dims), _widen(right, right_support, joint, dims), coefficient))
    terms = ((_cascade_hamiltonian(group), joint) for joint, group in products.items())
    return tuple((term, joint) for term, joint in terms if term is not None)


def _widen(matrix: Any, support: tuple[int, ...], joint: tuple[int, ...], dims: Sequence[int]) -> Any:
    """Return a matrix on ``support`` as a matrix on the larger support ``joint``."""
    from quchip.backend._memory import require_memory

    extra = tuple(index for index in joint if index not in support)
    order = (*support, *extra)
    local = [int(dims[index]) for index in order]
    size = prod(local)
    require_memory(16 * 4 * size**2, task=f"Cascaded port product on devices {list(joint)}",
                   remedy="Use ports on fewer devices or reduce the device cutoffs.")
    xp = concrete_array_module(matrix)
    widened = xp.kron(xp.asarray(matrix), xp.eye(prod(local[len(support):]), dtype=complex))
    permutation = [order.index(index) for index in joint]
    tensor = xp.transpose(widened.reshape(local + local), permutation + [len(order) + axis for axis in permutation])
    return tensor.reshape(size, size)


def _key(totals: int | Sequence[int]) -> tuple[int, ...]:
    """Return the sorted totals of a sector or class key."""
    if isinstance(totals, (int, np.integer)):
        return (int(totals),)
    return tuple(sorted({int(total) for total in totals}))


@dataclass(frozen=True, eq=False)
class SectorSpace:
    """Product energy states whose levels sum to one of a set of totals.

    Attributes
    ----------
    dims : tuple[int, ...]
        Level count of each device, in chip order.
    totals : tuple[int, ...]
        Total energy-level indices of the states, in increasing order.
    levels : numpy.ndarray
        Levels with shape ``(size, len(dims))``, one state per row, ordered by
        total and then in C order.
    """

    dims: tuple[int, ...]
    totals: tuple[int, ...]
    levels: np.ndarray

    @property
    def size(self) -> int:
        """Number of product states in the space."""
        return int(self.levels.shape[0])

    @functools.cached_property
    def labels(self) -> tuple[tuple[int, ...], ...]:
        """Bare labels of the states, in row order."""
        return tuple(map(tuple, self.levels.tolist()))

    @functools.cached_property
    def _rows(self) -> dict[tuple[int, ...], int]:
        return {label: row for row, label in enumerate(self.labels)}

    def row(self, label: tuple[int, ...]) -> int:
        """Return the row of a bare label in this space.

        Parameters
        ----------
        label : tuple[int, ...]
            Level of each device, in chip order.
        """
        return self._rows[tuple(label)]


@dataclass(frozen=True)
class SectorEigensystem:
    """Labeled eigenpairs of one sector or cutoff class.

    Attributes
    ----------
    space : SectorSpace
        Product states of the block.
    eigenvalues : array_like
        Ascending eigenvalues in GHz.
    eigenvectors : array_like
        Eigenvectors as columns in the block's product energy basis. Each
        column has a real, nonnegative amplitude on its assigned label, as
        :func:`~quchip.chip.dressing.gauge_phases` sets.
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
            Bare label in this block.
        """
        index = self.labeling.indices[self.space.row(label)]
        values = self.eigenvalues
        if contains_tracer(index) and not contains_tracer(values):
            values = jnp.asarray(values)
        return values[index]


class SectorModel:
    """Static lab-frame model of a chip, built one block of totals at a time.

    Without a cutoff, the chip's static model must conserve the total
    energy-level index, as
    :func:`~quchip.chip.effective.conserves_excitation_number` checks, and each
    block is one sector. Under an active
    :class:`~quchip.cutoff.ExcitationCutoff`, the model is ``P H P`` on the
    labels with ``N <= K``, and each block is one class of
    :func:`excitation_classes`. A block holds the device energies, each
    coupling band that the chip's approximation keeps, each retained term, and
    the Hamiltonian that cascaded ports generate.

    Parameters
    ----------
    chip : Chip
        Chip whose static model is built.
    resolution : optional
        Resolved local bases to reuse instead of resolving the devices again.
    advisories : list of str, optional
        List that receives approximation advisories instead of warnings.
    approximation : Approximation, optional
        Approximation that selects the static terms. ``None`` uses the chip's.
    """

    def __init__(self, chip: "Chip", *, resolution: Any = None, advisories: list[str] | None = None,
                 approximation: Any = None) -> None:
        from quchip.chip.effective import static_excitation_changes
        from quchip.cutoff import active_cutoff

        resolution = _resolve_system(chip, chip.backend) if resolution is None else resolution
        self.chip = chip
        self.bases = resolution.bases
        self.dims = tuple(int(dimension) for dimension in resolution.dims)
        self.network = resolution.network
        self.energies = tuple(self.bases[device.label].energies for device in chip.devices)
        self.approximation = chip.approximation if approximation is None else approximation
        policy = active_cutoff(chip)
        self.cutoff: int | None = None if policy is None else policy.max_total
        self.classes = None if self.cutoff is None else excitation_classes(
            static_excitation_changes(chip, self.approximation, resolution=resolution), self.cutoff)
        self._ports: dict[str, tuple[Any, tuple[int, ...]]] | None = None
        pairs = () if self.network is None else self.network.generated_pairs
        generated = () if not pairs else generated_products(self.port_couplings(), pairs, self.dims)
        self.terms = (*_resolved_energy_interactions(chip, resolution, self.approximation, advisories),
                      *((term / TWO_PI, joint) for term, joint in generated))
        self._spaces: dict[tuple[int, ...], SectorSpace] = {}
        self._pairs: dict[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
                          tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        self._eigensystems: dict[tuple[int, ...], tuple[Any, Any, SectorEigensystem]] = {}

    @property
    def traced(self) -> bool:
        """Return whether the device energies, static terms or port couplings hold JAX tracers."""
        return contains_tracer((self.energies, tuple(operator for operator, _ in self.terms), self._ports))

    @property
    def contributions(self) -> tuple[tuple[Any, ...], ...]:
        """Collapse contributions of the chip, with retained coordinate maps applied."""
        return tuple(self.chip._collapse_contributions_with_owners(self.bases))

    def class_of(self, total: int) -> tuple[int, ...]:
        """Return the totals of the block that holds ``total``.

        Parameters
        ----------
        total : int
            Total energy-level index.
        """
        if self.classes is None or self.cutoff is None:
            return (int(total),)
        if not 0 <= total <= self.cutoff:
            raise ValueError(
                f"Total excitation {total} is outside the excitation cutoff N <= {self.cutoff}. "
                "Raise max_total or use chip.with_space(None)."
            )
        return next(group for group in self.classes if total in group)

    def space(self, totals: int | Sequence[int]) -> SectorSpace:
        """Return the product states whose levels sum to ``totals``.

        Parameters
        ----------
        totals : int or sequence of int
            One total energy-level index, or the totals of a class.
        """
        key = _key(totals)
        if key not in self._spaces:
            from quchip.backend._memory import require_memory

            size = sum(sector_size(self.dims, total) for total in key)
            block = f"sector N={key[0]}" if len(key) == 1 else f"class N in {list(key)}"
            require_memory(16 * 6 * size**2 + 8 * len(self.dims) * size,
                           task=f"Excitation {block} (dimension {size})",
                           remedy="Request lower-excitation labels or reduce the model." if self.cutoff is None
                           else "Request lower-excitation labels, lower max_total, or reduce the model.")
            levels = (_sector_levels(self.dims, key[0]) if len(key) == 1
                      else np.concatenate([_sector_levels(self.dims, total) for total in key]))
            levels.flags.writeable = False
            self._spaces[key] = SectorSpace(self.dims, key, levels)
        return self._spaces[key]

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
        projected = _project_on_support(chip, operator, support, self.bases, backend)
        labels = tuple(chip.devices[index].label for index in support or range(len(chip.devices)))
        return _energy_operator(projected, labels, self.bases, backend)

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

    def block(self, operator: Any, support: tuple[int, ...], row_totals: int | Sequence[int],
              column_totals: int | Sequence[int]) -> Any:
        """Return the block of an operator from one set of totals to another.

        Parameters
        ----------
        operator : array_like
            Matrix on ``support`` in its product energy basis.
        support : tuple[int, ...]
            Device indices in the matrix's tensor order. An empty support
            denotes the whole chip.
        row_totals, column_totals : int or sequence of int
            Total energy-level indices of the rows and of the columns.
        """
        rows, columns, local_rows, local_columns = self._pairs_for(support, row_totals, column_totals)
        shape = (self.space(row_totals).size, self.space(column_totals).size)
        if contains_tracer(operator):
            return jnp.zeros(shape, dtype=complex).at[rows, columns].set(operator[local_rows, local_columns])
        out = np.zeros(shape, dtype=complex)
        out[rows, columns] = np.asarray(operator)[local_rows, local_columns]
        return out

    def hamiltonian(self, totals: int | Sequence[int]) -> Any:
        """Return the static lab-frame Hamiltonian block on ``totals``, in GHz.

        Parameters
        ----------
        totals : int or sequence of int
            One total energy-level index, or the totals of a class.
        """
        space = self.space(totals)
        xp = concrete_array_module(self.energies, tuple(operator for operator, _ in self.terms))
        diagonal = xp.zeros(space.size)
        for index, energies in enumerate(self.energies):
            diagonal = diagonal + xp.asarray(energies)[space.levels[:, index]]
        hamiltonian = xp.diag(diagonal).astype(complex)
        gathered = [(self._pairs_for(support, totals, totals), operator) for operator, support in self.terms]
        if gathered:
            rows = np.concatenate([pairs[0] for pairs, _ in gathered])
            columns = np.concatenate([pairs[1] for pairs, _ in gathered])
            values = xp.concatenate([xp.asarray(operator)[pairs[2], pairs[3]] for pairs, operator in gathered])
            if xp is np:
                np.add.at(hamiltonian, (rows, columns), values)
            else:
                hamiltonian = hamiltonian.at[rows, columns].add(values)
        return hamiltonian

    def eigensystem(self, total: int) -> SectorEigensystem:
        """Diagonalize the block that holds ``total`` and assign its bare labels.

        Each block is diagonalized once per model, and a result that holds
        tracers is reused only inside its trace. A concrete block is
        diagonalized on the host, and a traced block uses one ``eigh``.

        Parameters
        ----------
        total : int
            Total energy-level index in the block.
        """
        key = self.class_of(total)
        entry = self._eigensystems.get(key)
        if scoped_hit(entry, key):
            return entry[2]
        space = self.space(key)
        matrix = self.hamiltonian(key)
        eigenvalues: Any
        eigenvectors: Any
        if not contains_tracer(matrix) and not np.any(np.asarray(matrix).imag):
            # A real symmetric block diagonalizes several times faster than a complex one.
            eigenvalues, real_vectors = np.linalg.eigh(np.asarray(matrix).real)
            eigenvectors = real_vectors.astype(complex)
        else:
            eigenvalues, eigenvectors = _sector_eigh(matrix, np.zeros(space.size, dtype=np.int64))
        labeling = label_eigensystem(eigenvectors, BareProductReference((space.size,)), policy=assign_rowwise_greedy)
        # The reference is the sector's product energy basis, so the amplitudes are the vectors.
        eigenvectors = eigenvectors * gauge_phases(eigenvectors, labeling.indices)
        if not contains_tracer(eigenvalues):
            # Concrete energies take the backend's array type, as those of the full eigensystem do.
            eigenvalues = self.chip.backend.array_module.asarray(eigenvalues)
        system = SectorEigensystem(space, eigenvalues, eigenvectors, dataclasses.replace(labeling, keys=space.labels))
        # Inside a trace, even a concrete block gives a traced labeling.
        traced = contains_tracer((eigenvalues, eigenvectors, labeling.indices, labeling.overlaps, labeling.margins))
        self._eigensystems[key] = scoped_entry(key, system, traced=traced)
        return system

    def embed(self, totals: int | Sequence[int], vector: Any) -> Any:
        """Return a vector of one block as a product-space vector in solver coordinates.

        Parameters
        ----------
        totals : int or sequence of int
            Total energy-level indices of the block.
        vector : array_like
            Amplitudes in the block's product energy basis.
        """
        from quchip.backend._memory import require_memory

        dimension = prod(self.dims)
        # The zero vector, each basis transform and the native ket can each hold a product-space copy.
        require_memory(16 * 4 * dimension, task=f"Product-space state (dimension {dimension})",
                       remedy="Use chip.compact_state(..., dressed=True), which keeps the block coordinates.")
        space = self.space(totals)
        rows = np.ravel_multi_index(tuple(space.levels.T), self.dims)
        transforms = [self.bases[device.label].energy_to_solver() for device in self.chip.devices]
        xp = concrete_array_module(vector, transforms)
        product: Any
        if xp is np:
            product = np.zeros(dimension, dtype=complex)
            product[rows] = vector
        else:
            product = jnp.zeros(dimension, dtype=complex).at[rows].set(vector)
        tensor = product.reshape(self.dims)
        for axis, transform in enumerate(transforms):
            if transform is not None:
                tensor = xp.moveaxis(xp.tensordot(xp.asarray(transform), tensor, axes=([1], [axis])), 0, axis)
        return tensor.reshape(-1)

    def labeled_subspace(self, labels: Sequence[tuple[int, ...]]) -> tuple[Any, Any, list[int], Any]:
        """Return the eigenpairs of the blocks that ``labels`` occupy and the labels' positions.

        The returned eigenvalues and block-diagonal eigenvector matrix hold the
        occupied blocks in increasing total. The list gives each label's row,
        and the array gives the column of its assigned eigenvector.

        Parameters
        ----------
        labels : sequence of tuple[int, ...]
            Bare labels, each with one level per device.
        """
        keys = sorted({self.class_of(sum(label)) for label in labels})
        systems = {key: self.eigensystem(key[0]) for key in keys}
        offsets: dict[tuple[int, ...], int] = {}
        size = 0
        for key in keys:
            offsets[key] = size
            size += systems[key].space.size
        from quchip.backend._memory import require_memory

        require_memory(16 * 3 * size**2, task=f"Combined excitation blocks (dimension {size})",
                       remedy="Request fewer blocks or reduce the model.")
        xp = concrete_array_module(tuple((system.eigenvectors, system.labeling.indices) for system in systems.values()))
        eigenvalues = xp.concatenate([xp.asarray(systems[key].eigenvalues) for key in keys])
        eigenvectors = xp.zeros((size, size), dtype=complex)
        for key in keys:
            span = slice(offsets[key], offsets[key] + systems[key].space.size)
            if xp is np:
                eigenvectors[span, span] = systems[key].eigenvectors
            else:
                eigenvectors = eigenvectors.at[span, span].set(systems[key].eigenvectors)
        owners = [self.class_of(sum(label)) for label in labels]
        rows = [systems[key].space.row(label) for label, key in zip(labels, owners, strict=True)]
        positions = [offsets[key] + row for key, row in zip(owners, rows, strict=True)]
        columns = xp.stack([
            offsets[key] + systems[key].labeling.indices[row] for key, row in zip(owners, rows, strict=True)
        ])
        return eigenvalues, eigenvectors, positions, columns

    def _pairs_for(self, support: tuple[int, ...], row_totals: int | Sequence[int],
                   column_totals: int | Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the rows and columns that an operator on ``support`` connects.

        Two product states connect when their levels agree outside the support.
        The result holds the rows, the columns, and the matching row and column
        indices of the local matrix.
        """
        key = (support, _key(row_totals), _key(column_totals))
        if key not in self._pairs:
            self._pairs[key] = _level_pairs(self.dims, support, self.space(row_totals).levels,
                                            self.space(column_totals).levels)
        return self._pairs[key]
