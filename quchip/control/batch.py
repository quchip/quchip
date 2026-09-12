"""Sweep axes and entry handles for scheduled sequences.

Axes expand into per-point overrides for :class:`~quchip.engine.ir.SolveBatch`.
Numerical pulse, delay, and envelope values remain JAX-traceable.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from quchip.utils.batching import expand_axis_groups

if TYPE_CHECKING:
    from quchip.control.sequence import QuantumSequence


@dataclass(frozen=True)
class BatchAxis:
    """One batchable axis over a scheduled entry or sequence-level field.

    Created by :meth:`PulseHandle.vary`, :meth:`DelayHandle.vary`, or
    :meth:`QuantumSequence.vary`, and consumed by
    :meth:`QuantumSequence.build_batch`.

    Attributes
    ----------
    owner : QuantumSequence
        Sequence that owns the axis.
    field : str
        Swept field name.
    values : sequence
        Values evaluated along this axis.
    name : str
        Display name in result metadata.
    target_kind : {"entry", "sequence", "parameter"}
        Kind of value the axis overrides.
    entry_index : int or None
        Scheduled-entry index for entry axes.
    entry_field : str or None
        Normalized field on the scheduled entry.
    entry : object or None
        Scheduled entry captured by identity.
    """

    owner: "QuantumSequence"
    target_kind: str  # "entry" | "sequence" | "parameter"
    field: str
    values: Any
    name: str
    entry_index: int | None = None
    entry_field: str | None = None
    #: Scheduled entry, checked by identity when building the batch.
    #: ``None`` for sequence-level axes.
    entry: Any = None

    @property
    def size(self) -> int:
        return len(self.values)

    @property
    def override_key(self) -> tuple[int | None, str]:
        """Identify the bound value independently of the display name."""
        return self.entry_index, self.entry_field or self.field


@dataclass(frozen=True)
class ZippedBatchAxis:
    """Multiple :class:`BatchAxis` objects swept pairwise as one dimension.

    Attributes
    ----------
    axes : tuple of BatchAxis
        Equal-length axes evaluated at matching indices.
    """

    axes: tuple[BatchAxis, ...]

    @property
    def size(self) -> int:
        return self.axes[0].size


def _axis_metadata(axis: BatchAxis | ZippedBatchAxis) -> tuple[str, Any]:
    """Return ``(name, values)`` for *axis*, joining zipped-axis names with ``"/"``."""
    if isinstance(axis, ZippedBatchAxis):
        names = tuple(member.name for member in axis.axes)
        values = tuple({member.name: member.values[i] for member in axis.axes} for i in range(axis.size))
        return ("/".join(names), values)
    return (axis.name, axis.values)


class _BaseEntryHandle:
    """A scheduled entry's index and identity, used to build batch sweeps.

    Rejects use after the sequence changes which entry occupies that index.
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
        """Create a batch axis over one entry field.

        Parameters
        ----------
        field : str
            Pulse or delay field name.
        values : array-like
            Values along the axis.
        name : str or None, default=None
            Display name; defaults to the normalized field.
        """
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

    Sweepable fields: ``freq``, ``phase``, ``start_time``, and declared
    envelope parameters (e.g. ``amplitude``, ``duration``, ``sigmas``).

    Parameters
    ----------
    sequence : QuantumSequence
        Sequence owning the scheduled pulse.
    entry_index : int
        Pulse entry index in the sequence.
    """

    _reserved_fields = ("freq", "phase", "start_time")

    def _normalize_field(self, field: str) -> str:
        from quchip.control.sequence import _PulseEntry

        entry = self._resolve_entry()
        if not isinstance(entry, _PulseEntry):
            raise TypeError("PulseHandle does not point to a pulse entry")
        if field in self._reserved_fields:
            return field
        parameters = entry.envelope.parameter_values()
        if field not in parameters:
            sweepable = list(self._reserved_fields) + list(parameters)
            raise ValueError(f"Pulse field '{field}' is not sweepable. Available pulse fields: {sweepable}")
        return field


class DelayHandle(_BaseEntryHandle):
    """Reference to a scheduled delay entry.

    Only ``duration`` is sweepable.

    Parameters
    ----------
    sequence : QuantumSequence
        Sequence owning the scheduled delay.
    entry_index : int
        Delay entry index in the sequence.
    """

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
    axis_slices: list[list[dict[tuple[int | None, str], Any]]] = []
    for axis in axes:
        members = axis.axes if isinstance(axis, ZippedBatchAxis) else (axis,)
        axis_slices.append([
            {member.override_key: member.values[i] for member in members}
            for i in range(axis.size)
        ])

    return expand_axis_groups(axis_slices)
