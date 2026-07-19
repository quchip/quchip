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
        The dispatched solver name (``"sesolve"`` or ``"mesolve"``).
    """

    times: Any
    states: list[Any] | None = None
    expect: list[list[float]] | dict[str | tuple[str, str], Any] | None = None
    final_state: Any | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    solver: str = ""


@dataclass
class PreparedHamiltonian:
    """Backend-native Hamiltonian produced by :meth:`Backend.prepare_hamiltonian`.

    ``rhs`` is whatever the backend's solver accepts directly — a ``Qobj`` /
    ``QobjEvo`` for QuTiP, a dynamiqs ``TimeQArray`` / sum of them for
    dynamiqs. ``metadata`` passes engine-level hints (e.g.
    ``spectral_bound_ghz`` for integrator step heuristics) through opaquely.
    """

    rhs: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EagerBatch:
    """Batched-solve payload with one native RHS per element.

    The default :meth:`Backend.solve_batch` dispatches each element's RHS
    through :meth:`Backend.batched_sesolve` / :meth:`Backend.batched_mesolve`.
    """

    rhs_list: list[Any]
    batch_size: int
    metadata: dict[str, Any] = field(default_factory=dict)
    tlist: Any | None = None


@dataclass
class VmappedBatch:
    """Batched-solve payload with a single natively batched RHS.

    ``rhs`` covers every element at once — the dynamiqs path, where the
    per-element signals are stacked along a leading batch axis and the
    solver runs one vmapped call.
    """

    rhs: Any
    batch_size: int
    metadata: dict[str, Any] = field(default_factory=dict)
    tlist: Any | None = None


@dataclass
class DeferredBatch:
    """Batched-solve payload whose RHS construction is deferred.

    ``shared`` carries backend-private state; the producing backend must
    override :meth:`Backend.solve_batch` to consume it (the QuTiP path —
    final ``QobjEvo`` assembly happens inside loky workers so main-process
    overhead stays O(1) in batch size).
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
    """Hermitian eigensystem returned in a single diagonalization call.

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
