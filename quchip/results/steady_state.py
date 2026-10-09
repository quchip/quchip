"""Backend-neutral results for stationary Lindblad calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from quchip.utils.values import DeferredValue
from quchip.backend import Backend, SteadyStateSolverResult
from quchip.devices.base import BaseDevice
from quchip.results._batch import BatchResult
from quchip.utils.labeling import resolve_label


@dataclass(frozen=True)
class SteadyStateResult:
    """One normalized stationary density matrix and its diagnostics.

    Attributes
    ----------
    state : backend state
        Stationary density matrix in the solver basis.
    residual : scalar
        Residual norm of the stationary equation.
    nullity : scalar
        Null-space dimension of the trace-constrained generator.
    dims : tuple of int
        Hilbert-space dimensions in chip order.
    device_info : tuple
        ``(label, computational)`` records used by accessors.
    stats : mapping
        Backend solver statistics.
    """

    state: Any
    residual: Any
    nullity: Any
    _condition_number: DeferredValue | None = field(repr=False, compare=False)
    dims: tuple[int, ...]
    device_info: tuple[tuple[str, bool], ...]
    stats: Mapping[str, Any]
    _backend: Backend = field(repr=False, compare=False)
    _expectations: Mapping[Any, Any] = field(repr=False, compare=False)

    @property
    def condition_number(self) -> Any:
        """Condition number of the trace-constrained generator, calculated on request."""
        return None if self._condition_number is None else self._condition_number()

    @property
    def trace(self) -> Any:
        """Trace of the captured stationary density matrix."""
        return self._backend.array_module.trace(self._backend.to_array(self.state))

    @property
    def trace_error(self) -> Any:
        """Absolute deviation from unit trace."""
        return self._backend.array_module.abs(self.trace - 1.0)

    @property
    def hermiticity_error(self) -> Any:
        """Frobenius norm of rho minus its adjoint, calculated on request."""
        xp = self._backend.array_module
        state = self._backend.to_array(self.state)
        return xp.linalg.norm(state - xp.conj(xp.swapaxes(state, -1, -2)))

    @property
    def minimum_eigenvalue(self) -> Any:
        """Smallest eigenvalue of the Hermitian part, calculated on request."""
        xp = self._backend.array_module
        state = self._backend.to_array(self.state)
        hermitian = 0.5 * (state + xp.conj(xp.swapaxes(state, -1, -2)))
        return xp.min(xp.linalg.eigvalsh(hermitian))

    @property
    def positivity_error(self) -> Any:
        """Magnitude of a negative minimum eigenvalue, or zero."""
        return self._backend.array_module.maximum(0.0, -self.minimum_eigenvalue)

    @property
    def is_unique(self) -> Any:
        """Return uniqueness, or ``None`` when the backend skipped that diagnostic."""
        if self.nullity is None:
            return None
        return self.nullity == 1

    def expect(self, key: Any, index: int | None = None) -> Any:
        """Return one named stationary expectation value.

        Parameters
        ----------
        key : object or str
            Captured observable key.
        index : int or None, optional
            Entry of a list-valued observable.
        """
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
        """Partial-trace the stationary state down to one device.

        Parameters
        ----------
        device : device object or str
            Device to keep.
        """
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
        nullity=solver_result.nullity,
        _condition_number=solver_result._condition_number,
        dims=tuple(problem.engine_result.dims),
        device_info=problem.device_info,
        stats=MappingProxyType(dict(solver_result.stats)),
        _backend=backend,
        _expectations=MappingProxyType(expectations),
    )


class SteadyStateBatchResult(BatchResult[SteadyStateResult]):
    """Immutable stationary results reshaped to their declared sweep grid.

    Attributes
    ----------
    results : tuple of SteadyStateResult
        Per-point stationary results.
    shape : tuple of int
        Sweep-grid shape.
    axes : tuple
        Named sweep-axis metadata.
    """

    def expect(self, key: Any, index: int | None = None) -> Any:
        """Return one expectation value on the sweep grid.

        Parameters
        ----------
        key : object or str
            Captured observable key.
        index : int or None, optional
            Entry of a list-valued observable.
        """
        return self._reshape([result.expect(key, index=index) for result in self._results])
