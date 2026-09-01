"""Shared immutable result-grid mechanics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, Generic, Self, TypeVar, cast, overload

import numpy as np

from quchip.backend import Backend

ResultT = TypeVar("ResultT")


class BatchResult(Generic[ResultT]):
    """Ordered results with named Cartesian or zipped sweep axes."""

    _results: tuple[ResultT, ...]
    _shape: tuple[int, ...]
    _axes: tuple[tuple[str, Any], ...]

    def __init__(
        self,
        results: list[ResultT],
        *,
        shape: tuple[int, ...] | None = None,
        axes: tuple[tuple[str, Any], ...] | None = None,
    ) -> None:
        object.__setattr__(self, "_results", tuple(results))
        if shape is None:
            object.__setattr__(self, "_shape", (len(self._results),))
            object.__setattr__(
                self,
                "_axes",
                (("batch", tuple(range(len(self._results)))),) if axes is None else tuple(axes),
            )
        else:
            object.__setattr__(self, "_shape", tuple(shape))
            object.__setattr__(self, "_axes", () if axes is None else tuple(axes))
        points = int(np.prod(self._shape, dtype=int))
        if points != len(self._results):
            raise ValueError(
                f"Batch shape {self._shape} has {points} points, "
                f"but results contains {len(self._results)} elements."
            )
        if self._axes and len(self._axes) != len(self._shape):
            raise ValueError(f"Expected {len(self._shape)} axis descriptors, got {len(self._axes)}.")

    def __setattr__(self, name: str, value: Any) -> None:
        raise FrozenInstanceError(f"cannot assign to field {name!r}")

    @property
    def results(self) -> tuple[ResultT, ...]:
        """Return per-point results in C-order sweep order."""
        return self._results

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the natural sweep-grid shape."""
        return self._shape

    @property
    def axes(self) -> tuple[tuple[str, Any], ...]:
        """Return named sweep-axis metadata."""
        return self._axes

    @property
    def backend(self) -> Backend:
        """Return the backend shared by every result."""
        if not self._results:
            raise RuntimeError("Empty batch has no backend.")
        return cast(Backend, getattr(self._results[0], "_backend"))

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self):
        return iter(self._results)

    @staticmethod
    def _axis_member_names(axis_name: Any) -> tuple[str, ...]:
        if isinstance(axis_name, tuple):
            return tuple(str(part) for part in axis_name)
        return tuple(part for part in str(axis_name).split("/") if part)

    def _coordinate_from_dict(self, item: dict[str, int]) -> tuple[int, ...]:
        if not self._axes:
            raise TypeError("Dictionary indexing requires named sweep axes.")

        coord: list[int] = []
        consumed: set[str] = set()
        missing: list[str] = []
        for axis_name, _ in self._axes:
            direct_name = str(axis_name)
            member_names = self._axis_member_names(axis_name)
            provided_names = [name for name in member_names if name in item]
            if direct_name in item:
                provided_names.append(direct_name)
            if not provided_names:
                missing.append("/".join(member_names))
                continue
            provided_indices = {int(item[name]) for name in provided_names}
            if len(provided_indices) != 1:
                raise ValueError(f"Zipped axis {direct_name!r} constituent names must use the same index.")
            coord.append(provided_indices.pop())
            consumed.update(provided_names)

        unknown = sorted(set(item) - consumed)
        if unknown:
            axis_names = [name for axis_name, _ in self._axes for name in self._axis_member_names(axis_name)]
            raise KeyError(f"Unknown sweep axis names {unknown}. Available: {axis_names}")
        if missing:
            raise KeyError(f"Missing sweep axis indices for {missing}.")
        return tuple(coord)

    @overload
    def __getitem__(self, item: int | dict[str, int]) -> ResultT: ...

    @overload
    def __getitem__(self, item: slice) -> Self: ...

    def __getitem__(self, item: int | slice | dict[str, int]) -> ResultT | Self:
        if isinstance(item, dict):
            coord = self._coordinate_from_dict(item)
            return self._results[int(np.ravel_multi_index(coord, self._shape))]
        if isinstance(item, slice):
            selected = self._results[item]
            indices = tuple(range(*item.indices(len(self._results))))
            return type(self)(list(selected), shape=(len(indices),), axes=(("batch", indices),))
        return self._results[item]

    def _stack(self, values: list[Any]) -> Any:
        return self.backend.array_module.asarray(values)

    def _reshape(self, values: list[Any]) -> Any:
        array = self._stack(values)
        return self.backend.array_module.reshape(array, self._shape + tuple(array.shape[1:]))

    def with_sweep_metadata(
        self,
        *,
        shape: tuple[int, ...],
        axes: tuple[tuple[str, Any], ...],
    ) -> Self:
        """Return an equivalent batch annotated with sweep-axis metadata."""
        return type(self)(list(self._results), shape=shape, axes=axes)

    def __repr__(self) -> str:
        axis_names = [str(name) for name, _ in self._axes]
        return f"{type(self).__name__}(n={len(self._results)}, shape={self._shape}, axes={axis_names})"
