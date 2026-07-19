"""Batch / sweep-axis machinery for :class:`~quchip.control.sequence.QuantumSequence`.

This module owns everything about turning a scheduled sequence into a
*batched* solve request: the sweepable :class:`BatchAxis` /
:class:`ZippedBatchAxis` descriptors, the per-entry handles
(:class:`PulseHandle`, :class:`DelayHandle`) that create them, the axis
expansion into per-point overrides, and the list-like :class:`ProblemBatch`
view over the resulting :class:`~quchip.engine.ir.SolveBatch`.

:class:`QuantumSequence` (in ``sequence.py``) is the scheduler; it delegates
its ``vary`` / ``zip`` / ``build_batch`` surface here. The split keeps the
scheduler focused on timing semantics and the sweep bookkeeping isolated.

All sweep axes stay JAX-traceable: pulse parameters, delays, and envelope
fields flow into ``SolveProblem`` without Python-side concretization.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from quchip.control.sequence import QuantumSequence
    from quchip.engine.ir import SolveBatch, SolveProblem


@dataclass(frozen=True)
class BatchAxis:
    """One batchable axis over a scheduled entry or sequence-level field.

    Created by :meth:`PulseHandle.vary`, :meth:`DelayHandle.vary`, or
    :meth:`QuantumSequence.vary`, and consumed by
    :meth:`QuantumSequence.build_batch`.
    """

    owner: "QuantumSequence"
    target_kind: str  # "entry" | "sequence"
    field: str
    values: Any
    name: str
    entry_index: int | None = None
    #: The scheduled entry object this axis references (``None`` for
    #: sequence-level axes). ``build_batch`` validates it against the
    #: sequence by identity, so an axis silently pointing at the wrong
    #: entry after the sequence was modified is impossible.
    entry: Any = None

    @property
    def size(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class ZippedBatchAxis:
    """Multiple :class:`BatchAxis` objects swept pairwise as one dimension."""

    axes: tuple[BatchAxis, ...]

    @property
    def size(self) -> int:
        return self.axes[0].size


class ProblemBatch(Sequence["SolveProblem"]):
    """List-like view of a batched solve request with sweep bookkeeping.

    Wraps a :class:`~quchip.engine.ir.SolveBatch` (the structural source of
    truth); per-element :class:`~quchip.engine.ir.SolveProblem` views are
    materialised on demand.
    """

    __slots__ = ("batch", "params", "shape", "axes")

    def __init__(
        self,
        batch: "SolveBatch",
        params: np.ndarray,
        shape: tuple[int, ...],
        axes: tuple[tuple[str, Any], ...],
    ) -> None:
        self.batch = batch
        self.params = params
        self.shape = shape
        self.axes = axes

    def __len__(self) -> int:
        return self.batch.batch_size

    def __iter__(self):
        for idx in range(self.batch.batch_size):
            yield self.batch.element(idx)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return [self.batch.element(i) for i in range(*item.indices(self.batch.batch_size))]
        return self.batch.element(int(item))

    def params_at(self, point: int | tuple[int, ...]) -> dict[str, Any]:
        """Return the sweep parameter dictionary at the given grid coordinate."""
        if self.shape == ():
            if point not in (0, ()):
                raise IndexError(f"Scalar batch only accepts 0 or (), got {point!r}")
            return dict(self.params.item().items())
        if not isinstance(point, tuple):
            point = (point,)
        return dict(self.params[point].items())


def _axis_metadata(axis: BatchAxis | ZippedBatchAxis) -> tuple[str, Any]:
    """Return ``(name, values)`` for *axis*, joining zipped-axis names with ``"/"``."""
    if isinstance(axis, ZippedBatchAxis):
        names = tuple(member.name for member in axis.axes)
        values = tuple({member.name: member.values[i] for member in axis.axes} for i in range(axis.size))
        return ("/".join(names), values)
    return (axis.name, axis.values)


class _BaseEntryHandle:
    """Reference to one scheduled entry, used to build batch sweeps.

    Holds both the entry's index and the entry object itself; every use
    re-validates the two against the sequence by identity, so a handle can
    never silently act on the wrong entry after the sequence changes.
    """

    def __init__(self, sequence: "QuantumSequence", entry_index: int) -> None:
        self._sequence = sequence
        self._entry_index = entry_index
        self._entry = sequence._entries[entry_index]

    def _resolve_entry(self) -> Any:
        entries = self._sequence._entries
        if self._entry_index >= len(entries) or entries[self._entry_index] is not self._entry:
            raise RuntimeError(
                "This handle no longer matches its scheduled entry — the sequence was "
                "modified after the handle was created. Re-schedule and use a fresh handle."
            )
        return self._entry

    def vary(self, field: str, values: Any, *, name: str | None = None) -> BatchAxis:
        """Create a :class:`BatchAxis` that sweeps *field* over *values*."""
        normalized = self._normalize_field(field)
        return BatchAxis(
            owner=self._sequence,
            target_kind="entry",
            field=normalized,
            values=values,
            name=name or normalized,
            entry_index=self._entry_index,
            entry=self._entry,
        )

    def _normalize_field(self, field: str) -> str:
        raise NotImplementedError


class PulseHandle(_BaseEntryHandle):
    """Reference to one scheduled pulse entry.

    Sweepable fields: ``freq``, ``phase``, ``start_time``, and any public
    envelope attribute (e.g. ``amplitude``, ``duration``, ``sigmas``).
    """

    _reserved_fields = ("freq", "phase", "start_time")

    def _normalize_field(self, field: str) -> str:
        from quchip.control.sequence import _PulseEntry

        entry = self._resolve_entry()
        if not isinstance(entry, _PulseEntry):
            raise TypeError("PulseHandle does not point to a pulse entry")
        if field in self._reserved_fields:
            return field
        if field.startswith("_") or not hasattr(entry.envelope, field):
            sweepable = list(self._reserved_fields) + [
                name for name in vars(entry.envelope) if not name.startswith("_")
            ]
            raise ValueError(f"Pulse field '{field}' is not sweepable. Available pulse fields: {sweepable}")
        return field


class DelayHandle(_BaseEntryHandle):
    """Reference to a scheduled delay entry. Only ``duration`` is sweepable."""

    def _normalize_field(self, field: str) -> str:
        self._resolve_entry()
        if field != "duration":
            raise ValueError("Delay entries only support varying 'duration'")
        return field


def _expand_axis_overrides(
    axes: Sequence[BatchAxis | ZippedBatchAxis],
) -> tuple[tuple[int, ...], list[tuple[tuple[int, ...], dict[tuple[int | None, str], Any]]]]:
    """Expand batch axes into ``(shape, [(coord, overrides)...])``.

    ``overrides`` maps ``(entry_index, field) -> value`` for entry-level axes
    and ``(None, field) -> value`` for sequence-level axes.
    """
    if not axes:
        return (), [((), {})]

    axis_slices: list[list[dict[tuple[int | None, str], Any]]] = []
    for axis in axes:
        if isinstance(axis, ZippedBatchAxis):
            slice_: list[dict[tuple[int | None, str], Any]] = []
            for i in range(axis.size):
                point: dict[tuple[int | None, str], Any] = {}
                for subaxis in axis.axes:
                    point[(subaxis.entry_index, subaxis.field)] = subaxis.values[i]
                slice_.append(point)
            axis_slices.append(slice_)
        else:
            axis_slices.append(
                [{(axis.entry_index, axis.field): axis.values[i]} for i in range(axis.size)]
            )

    shape = tuple(len(s) for s in axis_slices)
    expanded: list[tuple[tuple[int, ...], dict[tuple[int | None, str], Any]]] = []
    for coord in np.ndindex(*shape):
        merged: dict[tuple[int | None, str], Any] = {}
        for dim_idx, point_idx in enumerate(coord):
            merged.update(axis_slices[dim_idx][point_idx])
        expanded.append((coord, merged))
    return shape, expanded
