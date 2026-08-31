"""Backend-neutral results for stationary Lindblad calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from quchip.backend import Backend, SteadyStateSolverResult
from quchip.devices.base import BaseDevice
from quchip.utils.labeling import resolve_label


@dataclass(frozen=True)
class SteadyStateResult:
    """One normalized stationary density matrix and its diagnostics."""

    state: Any
    residual: Any
    trace: Any
    trace_error: Any
    hermiticity_error: Any
    minimum_eigenvalue: Any
    positivity_error: Any
    nullity: Any
    condition_number: Any
    dims: tuple[int, ...]
    device_info: tuple[tuple[str, bool], ...]
    stats: Mapping[str, Any]
    _backend: Backend = field(repr=False, compare=False)
    _expectations: Mapping[Any, Any] = field(repr=False, compare=False)

    @property
    def is_unique(self) -> Any:
        """Return uniqueness, or ``None`` when the backend skipped that diagnostic."""
        if self.nullity is None:
            return None
        return self.nullity == 1

    def expect(self, key: Any, index: int | None = None) -> Any:
        """Return one named stationary expectation value."""
        resolved = tuple(resolve_label(item) for item in key) if isinstance(key, tuple) else resolve_label(key)
        value = self._expectations[resolved]
        if isinstance(value, tuple):
            if index is None:
                raise ValueError(
                    f"expect[{key!r}] contains {len(value)} values; specify index=0..{len(value) - 1}"
                )
            return value[index]
        if index is not None:
            raise ValueError(f"expect[{key!r}] is a single value; drop the index argument")
        return value

    def reduced_state(self, device: str | BaseDevice) -> Any:
        """Partial-trace the stationary state down to one device."""
        label = resolve_label(device)
        for index, (candidate, _) in enumerate(self.device_info):
            if candidate == label:
                return self._backend.ptrace(self.state, index, list(self.dims))
        available = [candidate for candidate, _ in self.device_info]
        raise ValueError(f"Device '{label}' not found. Available: {available}")


def build_steady_state_result(
    solver_result: SteadyStateSolverResult,
    problem: Any,
    backend: Backend,
) -> SteadyStateResult:
    """Wrap one backend stationary solve without concretizing native arrays."""
    xp = backend.array_module
    state_array = xp.asarray(backend.to_array(solver_result.state), dtype=complex)
    adjoint = xp.conj(xp.swapaxes(state_array, -1, -2))
    hermitian_part = 0.5 * (state_array + adjoint)
    trace = xp.trace(state_array)
    trace_error = xp.abs(trace - 1.0)
    hermiticity_error = xp.linalg.norm(state_array - adjoint)
    minimum_eigenvalue = xp.min(xp.linalg.eigvalsh(hermitian_part))
    positivity_error = xp.maximum(0.0, -minimum_eigenvalue)

    expectations: dict[Any, Any] = {}
    if problem.e_ops_meta is not None:
        from quchip.engine.observables import recombine_expect

        flat = [xp.asarray([value], dtype=complex) for value in solver_result.expect or ()]
        _, recombined = recombine_expect(
            flat,
            problem.e_ops_meta,
            xp.asarray([0.0]),
            problem.resolved_frame.demod_freqs,
        )
        for key, value in recombined.items():
            if isinstance(value, list):
                expectations[key] = tuple(item[0] for item in value)
            else:
                expectations[key] = value[0]

    return SteadyStateResult(
        state=solver_result.state,
        residual=solver_result.residual,
        trace=trace,
        trace_error=trace_error,
        hermiticity_error=hermiticity_error,
        minimum_eigenvalue=minimum_eigenvalue,
        positivity_error=positivity_error,
        nullity=solver_result.nullity,
        condition_number=solver_result.condition_number,
        dims=tuple(problem.engine_result.dims),
        device_info=tuple((device.label, device.computational) for device in problem.chip.devices),
        stats=MappingProxyType(dict(solver_result.stats)),
        _backend=backend,
        _expectations=MappingProxyType(expectations),
    )


@dataclass(frozen=True, init=False)
class SteadyStateBatchResult:
    """Immutable stationary results reshaped to their declared sweep grid."""

    _results: tuple[SteadyStateResult, ...]
    _shape: tuple[int, ...]
    _axes: tuple[tuple[str, Any], ...]

    def __init__(
        self,
        results: list[SteadyStateResult],
        *,
        shape: tuple[int, ...],
        axes: tuple[tuple[str, Any], ...],
    ) -> None:
        if int(np.prod(shape, dtype=int)) != len(results):
            raise ValueError(f"Batch shape {shape} does not match {len(results)} results.")
        object.__setattr__(self, "_results", tuple(results))
        object.__setattr__(self, "_shape", tuple(shape))
        object.__setattr__(self, "_axes", tuple(axes))

    @property
    def results(self) -> tuple[SteadyStateResult, ...]:
        """Return stationary results in C-order sweep order."""
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
        return self._results[0]._backend

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self):
        return iter(self._results)

    def __getitem__(self, item: int) -> SteadyStateResult:
        return self._results[item]

    def expect(self, key: Any, index: int | None = None) -> Any:
        """Return one expectation value reshaped to the sweep grid."""
        values = self.backend.array_module.asarray(
            [result.expect(key, index=index) for result in self._results]
        )
        return self.backend.array_module.reshape(values, self._shape + tuple(values.shape[1:]))
