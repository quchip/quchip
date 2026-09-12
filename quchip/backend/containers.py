"""Backend-agnostic solver-result and IR-lowering containers.

These frozen/auto dataclasses are the *payloads* exchanged across the backend
boundary. The engine emits them (or backends produce them) without committing
to any solver's native storage: a :class:`SolverResult` holds native states but
exposes a backend-free shape, a :class:`PreparedHamiltonian` / :class:`PreparedBatch`
carries whatever RHS the backend's solver accepts opaquely, and an
:class:`EigensystemData` defers per-column ket materialization so the dressing /
sweep hot path never pays for allocations it does not use.

Kept separate from :mod:`quchip.backend.protocol` (the :class:`Backend` ABC) so
the contract and its payloads can evolve independently.

References
----------
* Johansson, Nation, Nori — *QuTiP 2*, Comput. Phys. Commun. 183, 1760 (2012)
* Guilmin et al. — *dynamiqs: an open-source Python library for GPU-accelerated
  and differentiable simulation of quantum systems* (2024)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TypeAlias


from quchip.utils.values import DeferredValue


class BatchSolveError(RuntimeError):
    r"""A numerical or configuration failure at one batch point, safe across workers.

    Parameters
    ----------
    index : int
        Zero-based flat batch index that failed.
    detail : str
        Underlying numerical or configuration failure.
    parameters : dict or None, default None
        Parameter values at the failing point; None stores an empty mapping.
    """

    def __init__(self, index: int, detail: str, parameters: dict[str, Any] | None = None) -> None:
        self.index = index
        self.detail = detail
        self.parameters = {} if parameters is None else dict(parameters)
        super().__init__(index, detail, self.parameters)

    def __str__(self) -> str:
        context = f" with parameters {self.parameters!r}" if self.parameters else ""
        return f"Batch point {self.index} failed{context}: {self.detail}"


@dataclass
class SolverResult:
    """Backend-agnostic container for a single time-evolution solve.

    Attributes
    ----------
    times
        1D array of save times (ns) matching the solver's ``tlist``.
    states
        Saved states, one per save time, in the backend's native state type
        (``Qobj`` / ``QArray``). ``None`` if ``store_states`` was disabled.
    expect
        Expectation-value traces. ``list[list[float]]`` indexed by
        ``e_ops``-then-time, or a ``dict`` when the engine supplied labelled
        observables (keys are either strings or ``(drive_label, op)`` pairs).
    final_state
        Final state (``states[-1]`` when states are stored). Useful for
        multi-segment protocols without retaining full trajectories.
    stats
        Solver-specific diagnostics (integrator step count, batch index,
        etc.). Purely informational; no physics depends on it.
    solver
        The dispatched native solver name.
    native
        Unmodified stochastic result, or None for a deterministic payload.
    """

    times: Any
    states: list[Any] | None = None
    expect: list[list[float]] | dict[str | tuple[str, str], Any] | None = None
    final_state: Any | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    solver: str = ""
    native: Any = None


@dataclass(frozen=True)
class SteadyStateSolverResult:
    r"""Backend payload for one stationary Lindblad solve.

    Attributes
    ----------
    state : State
        Stationary density matrix in the native backend basis.
    expect : list or None
        Expectations in the requested observable order; None when not evaluated.
    stats : dict
        Backend diagnostics, including whether uniqueness was checked.
    residual : scalar or None
        Norm of the stationary Liouvillian residual, or None if not computed.
    nullity : int or None
        Estimated Liouvillian null-space dimension, or None when unchecked.
    """

    state: Any
    expect: list[Any] | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    residual: Any = None
    nullity: Any = None
    _condition_number: DeferredValue | None = field(default=None, repr=False)

    @property
    def condition_number(self) -> Any:
        return None if self._condition_number is None else self._condition_number()


@dataclass(frozen=True)
class LinearResponseSolverResult:
    r"""Backend payload for one batched passive-linear scattering solve.

    Attributes
    ----------
    responses : array_like, shape (n_freq, n_ports, n_ports)
        Complex scattering matrix indexed by frequency, output, then input.
    mode_amplitudes : array_like, shape (n_freq, n_modes, n_ports)
        Internal mode response to unit incident amplitude at each input port.
    residuals : array_like, shape (n_freq,)
        Linear-system residual norms at each probe frequency.
    """

    responses: Any
    mode_amplitudes: Any
    residuals: Any
    _condition_numbers: DeferredValue = field(repr=False)
    _mode_covariance: DeferredValue = field(repr=False)

    @property
    def mode_covariance(self) -> Any:
        """Return centered normal covariance N_ij = <delta a_j† delta a_i>."""
        return self._mode_covariance()

    @property
    def condition_numbers(self) -> Any:
        return self._condition_numbers()


@dataclass
class PreparedHamiltonian:
    r"""Backend-native Hamiltonian produced by :meth:`Backend.prepare_hamiltonian`.

    ``rhs`` is whatever the backend's solver accepts directly — a ``Qobj`` /
    ``QobjEvo`` for QuTiP, a dynamiqs ``TimeQArray`` / sum of them for
    dynamiqs. ``metadata`` passes engine-level hints (e.g.
    ``spectral_bound_ghz`` for integrator step heuristics) through opaquely.

    Attributes
    ----------
    rhs : object
        Native Hamiltonian consumed by the solver, already in angular-frequency units.
    metadata : dict
        Integration hints, including spectral/carrier bounds when available.
    """

    rhs: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedStationary:
    r"""Temporary native generator shared by one stationary operating point.

    Attributes
    ----------
    backend : Backend
        Backend instance that owns this preparation.
    engine_result : EngineResult
        Exact captured operating point associated with the generator.
    liouvillian : object
        Native stationary generator, with rates in 1/ns.
    """

    backend: Any = field(repr=False, compare=False)
    engine_result: Any = field(repr=False, compare=False)
    liouvillian: Any = field(repr=False, compare=False)


@dataclass
class EagerBatch:
    r"""Batched-solve payload with one native RHS per element.

    The default :meth:`Backend.solve_batch` dispatches each element's RHS
    through :meth:`Backend.batched_sesolve` / :meth:`Backend.batched_mesolve`.

    Attributes
    ----------
    rhs_list : list
        One native right-hand side per batch element.
    batch_size : int
        Number of batch elements.
    metadata : dict
        Shared lowering and integration hints.
    tlist : array_like or None
        Common save-time grid in ns, or None when unspecified.
    """

    rhs_list: list[Any]
    batch_size: int
    metadata: dict[str, Any] = field(default_factory=dict)
    tlist: Any | None = None


@dataclass
class VmappedBatch:
    r"""Batched-solve payload with a single natively batched RHS.

    ``rhs`` covers every element at once — the dynamiqs path, where the
    per-element signals are stacked along a leading batch axis and the
    solver runs one vmapped call.

    Attributes
    ----------
    rhs : object
        Native right-hand side including the batch axis.
    batch_size : int
        Number of batch elements.
    metadata : dict
        Shared lowering and integration hints.
    tlist : array_like or None
        Common save-time grid in ns, or None when unspecified.
    """

    rhs: Any
    batch_size: int
    metadata: dict[str, Any] = field(default_factory=dict)
    tlist: Any | None = None


@dataclass
class DeferredBatch:
    r"""Batched-solve payload whose RHS construction is deferred.

    ``shared`` carries backend-private state; the producing backend must
    override :meth:`Backend.solve_batch` to consume it. QuTiP assembles final
    ``QobjEvo`` objects inside its workers; Dynamiqs assembles a vmapped RHS
    inside its cached JIT.

    Attributes
    ----------
    shared : object
        Backend-private lowering payload consumed by solve_batch.
    batch_size : int
        Number of batch elements.
    metadata : dict
        Shared lowering and integration hints.
    tlist : array_like or None
        Common save-time grid in ns, or None when unspecified.
    """

    shared: Any
    batch_size: int
    metadata: dict[str, Any] = field(default_factory=dict)
    tlist: Any | None = None


#: What :meth:`Backend.prepare_batch` may return — the batching strategy is
#: explicit in the type, and :meth:`Backend.solve_batch` dispatches on it.
PreparedBatch: TypeAlias = EagerBatch | VmappedBatch | DeferredBatch


@dataclass
class EigensystemData:
    r"""Hermitian eigensystem returned in a single diagonalization call.

    ``eigenvalues`` is ascending. ``eigenvector_matrix`` stacks the
    eigenvectors as columns in the bare-product basis used for the
    diagonalization. The per-column backend-native kets are exposed lazily
    via the :attr:`eigenstates` property so the dressing / sweep hot path
    (which reads only ``eigenvalues`` + ``eigenvector_matrix`` + labeling)
    never pays for ``D`` backend-ket allocations and a second ``O(D**2)``
    densification it does not use.

    Backends populate either ``_states_builder`` (a callable that materializes
    the ket list on demand) or prime ``_states_cache`` directly (when the
    diagonalizer already produced the kets, e.g. QuTiP's ``Qobj.eigenstates``).

    Attributes
    ----------
    eigenvalues : array_like, shape (D,)
        Ascending eigenvalues in the input operator's units.
    eigenvector_matrix : array_like, shape (D, D)
        Eigenvectors as columns in the input operator basis.
    """

    eigenvalues: Any
    eigenvector_matrix: Any
    _states_builder: Callable[[], list[Any]] | None = None
    _states_cache: list[Any] | None = None

    @property
    def eigenstates(self) -> list[Any]:
        """Return per-column backend-native eigenstate kets (built on first access).

        Memoizes only when the materialized kets are tracer-free: under
        ``jit``/``grad``/``vmap`` the states carry tracers bound to the
        current trace, so caching them would let a stale tracer escape into
        a later trace. Concrete states are cached.
        """
        if self._states_cache is not None:
            return self._states_cache
        if self._states_builder is None:
            raise RuntimeError(
                "EigensystemData has neither a primed states cache nor a "
                "states builder; cannot materialize eigenstates."
            )
        states = self._states_builder()
        from quchip.utils.jax_utils import contains_tracer

        if not contains_tracer(states):
            self._states_cache = states
        return states


# Default options common to every solve — surfaces full state history so the
# user-facing ``SimulationResult`` has trajectories to plot.
_DEFAULT_SOLVE_OPTIONS: dict[str, Any] = {"store_states": True, "store_final_state": True}
