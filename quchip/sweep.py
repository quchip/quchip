"""Declarative parameter sweeps for chip studies.

This module provides the user-facing sweep abstractions and a
dressed-spectrum sweep driver:

- :class:`Sweep` declares a 1-D axis of values for a single named
  parameter. Multiple :class:`Sweep` axes compose as a Cartesian
  product.
- :class:`ZippedSweep` (built with :meth:`Sweep.zip`) pairs axes
  element-wise rather than expanding the product, which is the right
  shape for correlated scans (e.g. frequency and drive amplitude moved
  together along a calibration trace).
- :class:`SpectrumSweep` walks a sweep grid, dresses the chip at each
  point via :meth:`Chip.dress`, and records eigenvalues and bare→dressed
  label assignments into a :class:`SpectrumSweepResult`.

Sweep values are stored in their native physical units (GHz for
frequencies, ns for times, mK for temperatures). :class:`Sweep`
preserves JAX arrays verbatim so downstream code can differentiate
through swept values when desired; everything else is promoted to
``np.ndarray`` so ``len`` and indexing behave predictably.

The dressed-spectrum machinery here computes eigenvalues and overlaps
of the lab-frame chip Hamiltonian; for a general reference on the
dressed-state picture of coupled circuit qubits, see Blais, Grimsmo,
Girvin & Wallraff, *Circuit quantum electrodynamics*, Rev. Mod. Phys.
93, 025005 (2021).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np

from quchip.utils.labeling import bare_label_from_mapping, top_components

if TYPE_CHECKING:
    from quchip.chip.chip import Chip


class ZippedSweep:
    """Element-wise pairing of sweep axes.

    Built via :meth:`Sweep.zip`. All bundled axes must share the same
    ``size``; iteration steps through them together, producing one
    parameter dict per element rather than a Cartesian-product grid.
    """

    def __init__(self, sweeps: tuple[Sweep, ...]) -> None:
        self.sweeps = sweeps
        self.size = sweeps[0].size

    def __repr__(self) -> str:
        names = [s.name for s in self.sweeps]
        return f"ZippedSweep({names}, size={self.size})"


class Sweep:
    """Declarative 1-D sweep axis over a named parameter.

    ``values`` are in the swept parameter's own physical units (the
    package-wide contract: GHz for frequencies and couplings, ns for
    times, mK for temperatures).

    ``values`` may be a Python sequence, a NumPy array, or a JAX array;
    JAX arrays are preserved as-is so any consumer that threads the
    sweep through a JAX-traced path keeps full differentiability.
    Non-JAX inputs are normalized through :func:`numpy.asarray` for
    uniform ``len``/indexing behavior.

    Examples
    --------
    >>> import numpy as np
    >>> from quchip.sweep import Sweep
    >>> freqs = Sweep(np.linspace(4.9, 5.1, 5), name="freq")
    >>> freqs.size
    5
    >>> drives = Sweep([0.01, 0.02, 0.03], name="amp")
    >>> points = Sweep.expand([freqs, drives])
    >>> len(points)
    15
    """

    def __init__(self, values: Any, *, name: str | None = None) -> None:
        from quchip.utils.jax_utils import is_jax_array
        self.values = values if is_jax_array(values) else np.asarray(values)
        self.name = name or "unnamed"
        self.size = len(self.values)

    def __repr__(self) -> str:
        return f"Sweep({self.name!r}, size={self.size})"

    @staticmethod
    def zip(*sweeps: Sweep) -> ZippedSweep:
        """Pair axes for element-wise iteration.

        At least two sweeps are required; all sweeps must have equal size.
        A :class:`ValueError` is raised otherwise.

        Parameters
        ----------
        *sweeps
            Two or more :class:`Sweep` axes to bundle element-wise. Their
            ``name`` attributes must all be distinct — a :class:`ZippedSweep`
            with repeated axis names raises :class:`ValueError` when it is
            later enumerated (:meth:`expand`, :class:`SpectrumSweep`).

        Returns
        -------
        ZippedSweep
            Bundle iterated element-wise instead of taken in Cartesian
            product with other axes.
        """
        if len(sweeps) < 2:
            raise ValueError(f"Sweep.zip() requires at least two sweeps, got {len(sweeps)}")
        sizes = {s.size for s in sweeps}
        if len(sizes) != 1:
            raise ValueError(f"Zipped sweeps must have equal lengths, got {sorted(sizes)}")
        return ZippedSweep(sweeps)

    @staticmethod
    def expand(axes: Sequence[Sweep | ZippedSweep]) -> list[dict[str, Any]]:
        """Expand sweep axes into a flat list of parameter dicts.

        The Cartesian product of independent :class:`Sweep` axes is
        taken; :class:`ZippedSweep` bundles remain element-wise. The
        return order is the grid's C-order (last axis varies fastest).

        This is the params-only view of :func:`_iter_axis_points` — the
        single enumeration code path — with the grid coordinates dropped.

        Parameters
        ----------
        axes
            :class:`Sweep` and/or :class:`ZippedSweep` axes. Every axis name
            (including each member of a zipped bundle) must be unique across
            ``axes``, else :class:`ValueError` is raised — a repeated name
            would silently overwrite itself in each point's parameter dict.

        Returns
        -------
        list[dict[str, Any]]
            One parameter dict per grid point, length equal to the product
            of the independent axes' sizes (zipped bundles contribute their
            shared size once).
        """
        _shape, points = _iter_axis_points(axes)
        return [params for _coord, params in points]


def _axis_groups(
    axes: Sequence[Sweep | ZippedSweep],
) -> tuple[list[list[dict[str, Any]]], tuple[int, ...]]:
    """Normalize axes into per-group parameter dicts plus a Cartesian shape."""
    groups: list[list[dict[str, Any]]] = []
    for axis in axes:
        if isinstance(axis, ZippedSweep):
            groups.append([{s.name: s.values[i] for s in axis.sweeps} for i in range(axis.size)])
        else:
            groups.append([{axis.name: value} for value in axis.values])
    return groups, tuple(len(group) for group in groups)


def _check_unique_axis_names(axes: Sequence[Sweep | ZippedSweep]) -> None:
    """Raise ``ValueError`` if any name repeats across ``axes``, including within a zip.

    ``_iter_axis_points`` merges each grid point's per-axis param dicts with
    ``dict.update``, so two axes (or two members of one
    :class:`ZippedSweep`, or a zipped member colliding with an independent
    axis) sharing a name silently overwrite each other instead of raising.
    """
    names: list[str] = []
    for axis in axes:
        if isinstance(axis, ZippedSweep):
            names.extend(s.name for s in axis.sweeps)
        else:
            names.append(axis.name)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Sweep axis names must be unique across all axes; duplicated: {duplicates}")


def _iter_axis_points(
    axes: Sequence[Sweep | ZippedSweep],
) -> tuple[tuple[int, ...], list[tuple[tuple[int, ...], dict[str, Any]]]]:
    """Yield every (N-D coordinate, param dict) pair on the sweep grid."""
    _check_unique_axis_names(axes)
    groups, shape = _axis_groups(axes)
    if not shape:
        return (), [((), {})]

    expanded: list[tuple[tuple[int, ...], dict[str, Any]]] = []
    for coord in np.ndindex(*shape):
        params: dict[str, Any] = {}
        for group_index, entry_index in enumerate(coord):
            params.update(groups[group_index][entry_index])
        expanded.append((coord, params))
    return shape, expanded


@dataclass
class SpectrumSweepResult:
    """Dressed-spectrum data collected across a :class:`SpectrumSweep` grid.

    Attributes
    ----------
    device_labels
        Device labels of the underlying chip, ordered as the chip's
        tensor-product order (:attr:`~quchip.chip.chip.Chip.devices`).
    bare_labels
        Tuple of bare-state labels (one tuple of occupation numbers per
        device) that the sweep tracks.
    eigenvalues
        ``(*grid_shape, n_evals)`` array of eigenvalues in GHz, one row
        per sweep point.
    dressed_indices
        ``(*grid_shape, n_bare)`` array giving the dressed-state index
        assigned to each bare label at each sweep point, or ``NaN``
        when no confident assignment is found.
    assignment_overlaps
        ``(*grid_shape, n_bare)`` overlap weights accompanying
        ``dressed_indices``.
    overlap_threshold
        Minimum overlap below which an assignment is treated as
        unreliable and returned as ``NaN``.
    params
        Object array of parameter dicts, one per grid point.
    eigenvector_matrices
        Optional object array of dressed eigenvector matrices (one per
        grid point), produced when ``store_eigenstates=True``.
    eigenstates
        Optional object array of dressed eigenstates aligned with
        ``eigenvector_matrices``.
    """

    device_labels: tuple[str, ...]
    bare_labels: tuple[tuple[int, ...], ...]
    eigenvalues: np.ndarray
    dressed_indices: np.ndarray
    assignment_overlaps: np.ndarray
    overlap_threshold: float
    params: np.ndarray
    eigenvector_matrices: np.ndarray | None = None
    eigenstates: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        """Sweep-grid shape (excludes the trailing eigenvalue axis)."""
        return self.eigenvalues.shape[:-1]

    def _normalize_point(self, point: int | tuple[int, ...] | None) -> tuple[int, ...]:
        if self.shape == ():
            return ()
        if point is None:
            raise ValueError(
                f"This result spans a sweep grid of shape {self.shape}; pass a "
                "point index (int for 1-D sweeps, tuple of ints otherwise) to "
                "select which grid point to read."
            )
        normalized = point if isinstance(point, tuple) else (point,)
        if len(normalized) != len(self.shape):
            raise ValueError(
                f"Point {point!r} has {len(normalized)} coordinate(s) but this result's sweep "
                f"grid has shape {self.shape} ({len(self.shape)} axes); pass a coordinate tuple "
                "matching the grid's dimensionality."
            )
        return normalized

    def _normalize_label(
        self,
        device_states: Mapping[str, int] | None,
        device_state_kwargs: Mapping[str, int],
    ) -> tuple[int, ...]:
        """Resolve partial device-state specifications into a full bare-label tuple.

        Delegates to :func:`quchip.utils.labeling.bare_label_from_mapping`,
        the single canonical spec-to-tuple definition: keys may be device
        objects or labels, unspecified devices default to Fock index 0, and
        duplicate or unknown labels raise :class:`ValueError`.
        """
        return bare_label_from_mapping(self.device_labels, device_states, device_state_kwargs)

    def _bare_label_position(self, label: tuple[int, ...]) -> int:
        """Return the column index of ``label`` in ``bare_labels`` or raise."""
        try:
            return self.bare_labels.index(label)
        except ValueError:
            raise ValueError(f"Bare-state label {label} is not present in this sweep") from None

    def dressed_index(
        self,
        device_states: Mapping[str, int] | None = None,
        /,
        **device_state_kwargs: int,
    ) -> np.ndarray:
        """Dressed-state indices across the sweep for one bare label.

        Entries where the assignment overlap falls below
        ``overlap_threshold`` (or was never found) are returned as
        ``NaN``.

        Parameters
        ----------
        device_states
            Mapping from device (label or object) to occupation number,
            resolved via :func:`~quchip.utils.labeling.bare_label_from_mapping`;
            devices left unspecified default to Fock index 0. Mutually
            exclusive with ``device_state_kwargs``.
        **device_state_kwargs
            Keyword form of ``device_states`` (device label as keyword).

        Returns
        -------
        numpy.ndarray
            Floating-point array of shape :attr:`shape` (the sweep grid,
            excluding the bare-label axis). Floating dtype is required to
            represent unreliable or missing assignments as ``NaN`` — a grid
            point whose overlap with this bare label falls below
            ``overlap_threshold``, or where no dressed state was assigned
            to it at all.

        Examples
        --------
        >>> # Track the bare |1, 0> state across a sweep
        >>> # result.dressed_index({qubit_a: 1, qubit_b: 0})  # doctest: +SKIP
        """
        label = self._normalize_label(device_states, device_state_kwargs)
        bare_pos = self._bare_label_position(label)

        indices = np.asarray(self.dressed_indices[..., bare_pos], dtype=float)
        overlaps = np.asarray(self.assignment_overlaps[..., bare_pos], dtype=float)
        invalid = np.isnan(indices) | (overlaps < self.overlap_threshold)
        result = indices.copy()
        result[invalid] = np.nan
        return result

    def energy_by_bare_label(
        self,
        device_states: Mapping[str, int] | None = None,
        /,
        **device_state_kwargs: int,
    ) -> np.ndarray:
        """Dressed energies (GHz) traced out for one bare label across the sweep.

        Invalid points inherit the ``NaN`` mask from
        :meth:`dressed_index`.

        Parameters
        ----------
        device_states
            Mapping from device (label or object) to occupation number; see
            :meth:`dressed_index`. Mutually exclusive with
            ``device_state_kwargs``.
        **device_state_kwargs
            Keyword form of ``device_states`` (device label as keyword).

        Returns
        -------
        numpy.ndarray
            Floating-point array of shape :attr:`shape`, in GHz. Entries
            where :meth:`dressed_index` returns ``NaN`` (overlap below
            ``overlap_threshold``, or no assignment found) are ``NaN``.
        """
        indices = self.dressed_index(device_states, **device_state_kwargs)
        flat_indices = indices.reshape(-1)
        flat_evals = self.eigenvalues.reshape(-1, self.eigenvalues.shape[-1])
        flat_out = np.full(flat_indices.shape, np.nan, dtype=float)
        valid = ~np.isnan(flat_indices)
        flat_out[valid] = flat_evals[np.nonzero(valid)[0], flat_indices[valid].astype(int)]
        return flat_out.reshape(indices.shape)

    def state_components_at(
        self,
        point: int | tuple[int, ...] | None,
        state: int | Mapping[str, int] | None = None,
        /,
        *,
        n_components: int = 5,
        **device_state_kwargs: int,
    ) -> dict[tuple[int, ...], float]:
        """Top-``n_components`` bare-state probabilities of a dressed eigenstate.

        ``state`` may either be an explicit dressed index (``int``) or
        a bare-label specification that is resolved via the same
        overlap-based map used by :meth:`dressed_index`. Requires the
        sweep to have been run with ``store_eigenstates=True``.

        Returns a dict mapping bare labels to squared-amplitude
        probabilities, ordered from largest to smallest.
        """
        if self.eigenvector_matrices is None:
            raise ValueError(
                "Eigenvector matrices were not stored for this sweep. "
                "Re-run it with store_eigenstates=True."
            )
        if n_components <= 0:
            raise ValueError(f"n_components must be positive, got {n_components}")

        point_index = self._normalize_point(point)
        eigenvector_matrix = self.eigenvector_matrices[point_index]

        if isinstance(state, int) and not isinstance(state, bool):
            dressed_idx: int = state
        else:
            if state is not None and not isinstance(state, Mapping):
                raise TypeError(f"state must be an int or mapping, got {type(state).__name__}")
            label = self._normalize_label(state, device_state_kwargs)
            bare_pos = self._bare_label_position(label)
            raw_idx = self.dressed_indices[point_index + (bare_pos,)]
            overlap = self.assignment_overlaps[point_index + (bare_pos,)]
            if np.isnan(raw_idx) or float(overlap) < self.overlap_threshold:
                raise ValueError(
                    f"No confidently labeled dressed state for bare label {label} at point {point_index}"
                )
            dressed_idx = int(raw_idx)

        return top_components(eigenvector_matrix, self.bare_labels, dressed_idx, n_components)


class SpectrumSweep:
    """Sequential dressed-spectrum sweep driver.

    At each grid point the chip is cloned via :meth:`Chip.updated`,
    ``update_fn`` mutates the clone with the swept values, and
    :meth:`Chip.dress` computes the lab-frame dressed spectrum. The
    per-point eigenvalues and overlap-based bare→dressed assignments
    are collected into a :class:`SpectrumSweepResult`.

    This is the standard tool for two-tone spectroscopy maps, avoided
    crossings, and any study that requires following dressed states
    across parameter space. See Blais, Grimsmo, Girvin & Wallraff,
    Rev. Mod. Phys. 93, 025005 (2021), for the dressed-state picture;
    the overlap-based labeling follows the usual practice of assigning
    a dressed eigenstate to the bare state with which it has maximum
    overlap.

    Examples
    --------
    >>> # Sweep a qubit frequency and record dressed levels
    >>> # sweep = SpectrumSweep(
    >>> #     chip,
    >>> #     [Sweep(np.linspace(4.8, 5.2, 41), name="freq")],
    >>> #     update_fn=lambda c, p: setattr(c.device(qubit), "frequency", p["freq"]),
    >>> # )
    >>> # result = sweep.run()  # doctest: +SKIP
    """

    def __init__(
        self,
        chip: "Chip",
        axes: Sequence[Sweep | ZippedSweep],
        *,
        update_fn: Callable[["Chip", dict[str, Any]], None],
        evals_count: int | None = None,
        store_eigenstates: bool = False,
        overlap_threshold: float = 0.5,
    ) -> None:
        self.chip = chip
        self.axes = list(axes)
        self.update_fn = update_fn
        self.evals_count = evals_count
        self.store_eigenstates = bool(store_eigenstates)
        self.overlap_threshold = float(overlap_threshold)

    def run(self, progress: bool = True) -> SpectrumSweepResult:
        """Execute the sweep and collect the dressed-spectrum analysis.

        Parameters
        ----------
        progress
            Whether to display a ``tqdm`` progress bar.

        Returns
        -------
        SpectrumSweepResult
            Populated result.
        """
        from tqdm import tqdm

        for axis in self.axes:
            if axis.size == 0:
                name = axis.name if isinstance(axis, Sweep) else [s.name for s in axis.sweeps]
                raise ValueError(
                    f"Sweep axis {name!r} has zero length; SpectrumSweep.run() cannot iterate an empty grid."
                )

        shape, expanded = _iter_axis_points(self.axes)
        bare_labels = self.chip._canonical_bare_labels()
        n_bare = len(bare_labels)

        total_dim = self.chip.total_dim
        if self.evals_count is None:
            n_evals = total_dim
        else:
            n_evals = self.evals_count
            if isinstance(n_evals, bool) or not isinstance(n_evals, int) or not (1 <= n_evals <= total_dim):
                raise ValueError(
                    f"evals_count must be an integer in [1, {total_dim}] (chip.total_dim), got {n_evals!r}"
                )

        param_store = np.empty(shape, dtype=object)
        dressed_indices = np.full(shape + (n_bare,), np.nan, dtype=float)
        assignment_overlaps = np.full(shape + (n_bare,), np.nan, dtype=float)

        matrices_store = np.empty(shape, dtype=object) if self.store_eigenstates else None
        states_store = np.empty(shape, dtype=object) if self.store_eigenstates else None
        eigenvalues_store: list[np.ndarray] = []

        # Snapshotted once, pre-sweep: the label/shape scaffold above
        # (bare_labels, n_evals, device_labels on the result) is fixed from
        # this signature, so update_fn changing device count, order, labels,
        # or levels at any grid point would silently misalign every array
        # collected here against a scaffold that no longer matches.
        topology_signature = tuple((device.label, device.levels) for device in self.chip.devices)

        iterator = tqdm(expanded, desc="SpectrumSweep") if progress else expanded
        for coord, params in iterator:
            chip_point = self.chip.updated(lambda cloned: self.update_fn(cloned, params))
            point_signature = tuple((device.label, device.levels) for device in chip_point.devices)
            if point_signature != topology_signature:
                raise ValueError(
                    f"update_fn {self.update_fn!r} changed chip topology at grid point {coord}: "
                    f"expected devices (label, levels) = {topology_signature}, got {point_signature}."
                )
            dressed = chip_point.dress(overlap_threshold=self.overlap_threshold)
            evals = np.asarray(dressed.eigenvalues, dtype=float)[:n_evals]
            eigenvalues_store.append(evals)
            param_store[coord] = dict(params)

            for bare_pos, bare_label in enumerate(bare_labels):
                dressed_idx = dressed.state_map.get(bare_label)
                if dressed_idx is None or dressed_idx >= len(evals):
                    continue
                dressed_indices[coord + (bare_pos,)] = dressed_idx
                assignment_overlaps[coord + (bare_pos,)] = dressed.assignment_overlaps[bare_label]

            if matrices_store is not None and states_store is not None:
                matrices_store[coord] = np.asarray(dressed.eigenvector_matrix[:, : len(evals)])
                states_store[coord] = list(dressed.eigenstates[: len(evals)])

        if shape:
            eigenvalues = np.asarray(eigenvalues_store, dtype=float).reshape(shape + (len(eigenvalues_store[0]),))
        else:
            eigenvalues = np.asarray(eigenvalues_store[0], dtype=float)

        return SpectrumSweepResult(
            device_labels=tuple(device.label for device in self.chip.devices),
            bare_labels=bare_labels,
            eigenvalues=eigenvalues,
            dressed_indices=dressed_indices,
            assignment_overlaps=assignment_overlaps,
            overlap_threshold=self.overlap_threshold,
            params=param_store,
            eigenvector_matrices=matrices_store,
            eigenstates=states_store,
        )
