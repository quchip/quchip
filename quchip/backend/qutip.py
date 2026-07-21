"""QuTiP backend implementation of the :mod:`quchip.backend.protocol` contract.

QuTiP stores operators as ``qutip.Qobj`` (sparse CSR by default, dense on
request). This backend lowers engine IR into ``Qobj`` / ``QobjEvo`` form and
dispatches the ``sesolve``/``mesolve`` drivers directly.

Batched sweeps and heterogeneous problem lists are parallelized via the
``loky`` reusable process pool. QuTiP solvers release the GIL only partially,
so processes beat threads. Final ``QobjEvo`` assembly happens inside the
workers so the main process never pays for it.

References
----------
* Johansson, Nation, Nori — *QuTiP 2*, Comput. Phys. Commun. 183, 1760 (2012)
* Breuer & Petruccione — *The Theory of Open Quantum Systems* (OUP, 2002)
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import qutip
from qutip import Qobj
from qutip.solver.mesolve import MESolver
from qutip.solver.sesolve import SESolver
from scipy import sparse

from quchip.backend._dims import (
    compute_two_body_permutation,
    default_solver_steps,
    validate_two_body_indices,
)
from quchip.backend.containers import (
    _DEFAULT_SOLVE_OPTIONS,
    EigensystemData,
    DeferredBatch,
    PreparedHamiltonian,
    SolverResult,
)
from quchip.backend.protocol import Backend, Operator, State
from quchip.engine.ir import (
    Shift,
    Window,
    decompose_carrier_bands,
    evaluate_signal_program,
    signal_children,
)
from quchip.utils.jax_utils import maybe_concrete_scalar


@dataclass(frozen=True)
class _QuTiPBatchShared:
    """Shared per-batch state travelling through :attr:`DeferredBatch.shared`.

    Opaque to the engine; consumed by :meth:`QuTiPBackend.solve_batch` to
    build one ``QobjEvo`` per element inside a loky worker.
    """

    static_rhs: Qobj | None
    dynamic_qobjs: tuple[Qobj, ...]
    sample_tlist: Any
    dynamic_signals: tuple[tuple[Any, ...], ...]


def _coeff_callable(signal: Any) -> Callable[[float, Any], complex]:
    """Build a scalar coefficient callable ``c(t)`` from a signal program AST."""
    def _coeff(time: float, _args: Any = None) -> complex:
        value = np.asarray(evaluate_signal_program(signal, time, xp=np), dtype=complex)
        return complex(value.item()) if value.ndim == 0 else complex(value.reshape(-1)[0])

    return _coeff


# Minimum number of samples used to interpolate a slow (carrier-free) envelope.
# The user's output tlist can be far coarser than the envelope's features —
# the extreme being a 2-point [t0, t_end] "final state only" grid. Envelope
# evaluation is a vectorized AST pass, so a ~1k-point floor costs microseconds
# per band. Windowed envelopes get additional local resolution regardless
# (see :func:`_augmented_sample_grid`); this floor is what a non-windowed
# envelope relies on alone.
_MIN_ENVELOPE_SAMPLES = 1001

# Local per-window subgrid density (samples/ns) merged into the base sample
# grid around each Window node's edges (see :func:`_augmented_sample_grid`).
# Bounded by the window's own width, not the solve span, so cost does not
# scale with idle-span length. Value pinned by the accuracy regression in
# tests/test_backend_hamiltonian_prep.py (TestEnvelopeSampleGrid).
_WINDOW_SUBGRID_POINTS_PER_NS = 40.0

# Minimum interior samples spanning a window, regardless of width. Caps the
# density-based count from below: a sub-ns window (e.g. 0.01 ns) would
# otherwise get only 1-2 interior points from _WINDOW_SUBGRID_POINTS_PER_NS
# alone, silently corrupting the sampled pulse area. Bounded by the
# window's own width still, not the solve span.
_MIN_WINDOW_INTERIOR_SAMPLES = 41

# Extra span (ns) sampled just outside each window edge so the interpolant
# sees the envelope's true zero on both sides of the start/stop discontinuity,
# not just a single grid cell straddling it.
_WINDOW_EDGE_PADDING_NS = 1.0

# Canonical skeleton size spanning the full base-grid range for a windowed
# envelope's coefficient grid. Outside any window the envelope is exactly
# zero (Window.evaluate's mask), so a handful of skeleton points represents
# that region exactly regardless of count — a windowed envelope's
# coefficient fidelity therefore never depends on how dense the user's
# output tlist happens to be, only on the window's own local subgrid.
_CANONICAL_BASE_SKELETON_POINTS = 3


def _collect_window_bounds(signal: Any, shift: float = 0.0) -> list[tuple[float, float]]:
    """Recursively collect every ``Window`` node's absolute ``(start, stop)`` bounds (ns).

    A ``Window``'s own ``start``/``stop`` are local to its child's time
    frame; an enclosing ``Shift`` translates that frame by ``delta_t``
    before the window's mask applies (``Shift`` wraps ``Window`` in the
    scheduled-drive AST — see
    :func:`~quchip.engine.stage2_assembly._spec_to_raw_signal`), so *shift*
    accumulates while descending through any ``Shift`` ancestor. A subtree
    whose window bounds or enclosing shift are JAX tracers is skipped —
    concrete placement is required to add refinement samples, and a QuTiP
    solve never runs under tracing regardless, so this is purely defensive.
    """
    bounds: list[tuple[float, float]] = []
    if isinstance(signal, Window):
        start = maybe_concrete_scalar(signal.start)
        stop = maybe_concrete_scalar(signal.stop)
        if start is not None and stop is not None and stop > start:
            bounds.append((start + shift, stop + shift))
        bounds.extend(_collect_window_bounds(signal.child, shift))
        return bounds
    if isinstance(signal, Shift):
        delta = maybe_concrete_scalar(signal.delta_t)
        if delta is None:
            return bounds
        bounds.extend(_collect_window_bounds(signal.child, shift + delta))
        return bounds
    for child in signal_children(signal):
        bounds.extend(_collect_window_bounds(child, shift))
    return bounds


def _local_window_subgrid(start: float, stop: float) -> np.ndarray:
    """Build a dense grid resolving ``[start, stop]`` plus a small zero-valued margin on each side.

    Interior samples span ``[start, stop]`` via ``np.linspace`` (which
    places *start* and *stop* at exact grid points by construction — no
    floating-point miss), at ``_WINDOW_SUBGRID_POINTS_PER_NS`` density
    floored at ``_MIN_WINDOW_INTERIOR_SAMPLES`` so a window narrower than
    ``1 / _WINDOW_SUBGRID_POINTS_PER_NS`` ns is never left under-resolved.
    *start* and *stop* are unioned in once more explicitly (belt-and-
    braces against any future construction change). A small zero-valued
    margin (``_WINDOW_EDGE_PADDING_NS``) is sampled on each side outside
    ``[start, stop]`` so the interpolant sees the true zero on both sides
    of the boundary discontinuity, not just a single grid cell straddling it.
    """
    width = stop - start
    interior_n = max(int(np.ceil(width * _WINDOW_SUBGRID_POINTS_PER_NS)) + 1, _MIN_WINDOW_INTERIOR_SAMPLES)
    interior = np.linspace(start, stop, interior_n)

    pad = min(_WINDOW_EDGE_PADDING_NS, max(width, 1e-9))
    outside_n = max(int(np.ceil(pad * _WINDOW_SUBGRID_POINTS_PER_NS)), 3)
    before = np.linspace(start - pad, start, outside_n, endpoint=False)
    after = np.linspace(stop, stop + pad, outside_n + 1)[1:]

    return np.unique(np.concatenate([before, interior, after, [start, stop]]))


def _augmented_sample_grid(envelope: Any, base_grid: Any) -> np.ndarray:
    """Return a window-aware coefficient grid, decoupled from the user's output-tlist density.

    A windowed envelope is exactly zero outside ``[start, stop]`` while
    generally nonzero *at* the boundary (e.g. a truncated Gaussian) —
    cubic-spline interpolation across that jump smears or rings regardless
    of how dense *base_grid* is globally, and a windowed pulse's fidelity
    must not depend on what output times the user happened to request. So
    when the envelope carries at least one concrete window bound (see
    :func:`_collect_window_bounds`), its coefficient grid is built
    canonically: a fixed-size skeleton spanning the full solve range
    (exact outside any window, where the value is identically zero,
    regardless of point count) unioned with a locally dense subgrid
    resolving each window (:func:`_local_window_subgrid`, bounded by the
    window's own width, not the solve span). *base_grid* contributes only
    its endpoints in that case. A non-windowed envelope uses *base_grid*
    directly instead — its fidelity is governed by
    :meth:`QuTiPBackend._resolve_envelope_sample_tlist`.
    """
    bounds = _collect_window_bounds(envelope)
    base = np.asarray(base_grid, dtype=float)
    if not bounds:
        return base
    lo, hi = base[0], base[-1]
    pieces = [np.linspace(lo, hi, _CANONICAL_BASE_SKELETON_POINTS)]
    for start, stop in bounds:
        clipped_start, clipped_stop = max(start, lo), min(stop, hi)
        if clipped_stop <= clipped_start:
            continue
        subgrid = _local_window_subgrid(clipped_start, clipped_stop)
        pieces.append(subgrid[(subgrid >= lo) & (subgrid <= hi)])
    return np.unique(np.concatenate(pieces))


def _sample_coeff_array(signal: Any, sample_tlist: Any) -> np.ndarray:
    """Sample a signal program over *sample_tlist* as a 1-D complex array.

    A time-independent envelope (e.g. a carrier band's ``Constant``) yields
    a scalar; broadcast it to the grid length so it matches *sample_tlist*.
    """
    arr = np.asarray(evaluate_signal_program(signal, sample_tlist, xp=np), dtype=complex).ravel()
    return np.broadcast_to(arr, (len(sample_tlist),)).copy()


# Interpolation order for a windowed envelope's sampled coefficient array.
# Linear (order=1) is mathematically bounded by its two bracketing node
# values, so it cannot overshoot the envelope's zero plateau. The default
# cubic (order=3) extrapolates across the canonical grid's highly
# non-uniform knot spacing — a dense local subgrid immediately adjacent to
# the sparse full-span skeleton (see _augmented_sample_grid) — and rings
# far outside the window: measured -2423 at t=50 ns for a 0.1 ns pulse
# placed at t=150.17 ns in a [0, 308] ns span, order=3. A non-windowed
# envelope has no discontinuity to ring across and keeps the default cubic
# order (unspecified below).
_WINDOWED_COEFFICIENT_ORDER = 1


def _envelope_coefficient(envelope: Any, sample_tlist: Any) -> Any:
    """Build the QuTiP coefficient for one carrier-free envelope: sampled array or exact callable.

    A windowed envelope (see :func:`_collect_window_bounds`) is sampled on
    the canonical, tlist-density-independent grid from
    :func:`_augmented_sample_grid` and interpolated at
    :data:`_WINDOWED_COEFFICIENT_ORDER` to avoid cubic ringing across the
    window-edge discontinuity. A non-windowed envelope is interpolated
    from *sample_tlist* directly at the default cubic order. Falls back to
    an exact callable when no concrete sample grid is available at all.
    """
    if sample_tlist is None:
        return qutip.coefficient(_coeff_callable(envelope))
    bounds = _collect_window_bounds(envelope)
    grid = _augmented_sample_grid(envelope, sample_tlist)
    arr = _sample_coeff_array(envelope, grid)
    if bounds:
        return qutip.coefficient(
            np.asarray(arr, dtype=complex),
            tlist=np.asarray(grid, dtype=float),
            order=_WINDOWED_COEFFICIENT_ORDER,
        )
    return qutip.coefficient(np.asarray(arr, dtype=complex), tlist=np.asarray(grid, dtype=float))


def _carrier_coefficient(freq: Any) -> Any:
    """Build the analytic carrier ``exp(i·freq·t)`` as a QuTiP coefficient (``freq`` angular, rad/ns).

    Kept analytic — never sampled — so a resonant carrier integrates to
    solver tolerance instead of accumulating cubic-spline interpolation
    error (the lab-frame frame-invariance bug).

    QuTiP-only boundary: the closure captures ``freq`` and is invoked by
    the QuTiP solver at concrete times. QuTiP is not JAX-native, so this
    callable is never evaluated under tracing — a traced ``freq`` would be
    concretized by the solver, never inside a ``jit``.
    """
    def _carrier(t: float, *args: Any, **kwargs: Any) -> complex:
        return complex(np.exp(1j * freq * t))

    return qutip.coefficient(_carrier)


def _band_coefficient(band: Any, sample_tlist: Any) -> Any:
    """Build the QuTiP coefficient for one carrier band: sampled slow envelope × analytic carrier.

    The carrier-free envelope coefficient is built by
    :func:`_envelope_coefficient` — canonical, tlist-density-independent
    sampling at a ringing-safe interpolation order for a windowed
    envelope, or direct interpolation of *sample_tlist* (or an exact
    callable) otherwise. A concretely zero band frequency needs no
    carrier, so the common resonant rotating-frame case stays a pure
    envelope coefficient.
    """
    env_coeff = _envelope_coefficient(band.envelope, sample_tlist)
    freq = maybe_concrete_scalar(band.freq)
    if freq is not None and freq == 0.0:
        return env_coeff
    return env_coeff * _carrier_coefficient(band.freq)


def _dynamic_term_entries(op: Qobj, signal: Any, sample_tlist: Any) -> list[list[Any]]:
    """Return QuTiP ``[op, coeff]`` pairs — one per carrier band of *signal*.

    The signal is band-normalized into ``Σ_k envelope_k(t)·exp(i·freq_k·t)``
    so each fast carrier stays analytic while only the slow envelope is
    sampled — physics a backend can represent exactly is never forced
    through pre-sampling.
    """
    return [[op, _band_coefficient(band, sample_tlist)] for band in decompose_carrier_bands(signal)]


def _assemble_qobjevo(static_rhs: Qobj | None, op_signal_pairs: Any, sample_tlist: Any) -> qutip.QobjEvo:
    """Seed-extend-assemble a ``QobjEvo`` from a static ``Qobj`` and dynamics.

    The term list is seeded with *static_rhs* (when present) and extended with
    one ``[op, coeff]`` band entry per dynamic ``(Qobj, signal)`` pair before a
    single ``QobjEvo`` assembly. Coefficients carry their own sample grids.
    Shared by :meth:`QuTiPBackend.prepare_hamiltonian` and the per-element
    batch RHS builder so the assembly path lives in one place.
    """
    terms: list[Any] = []
    if static_rhs is not None:
        terms.append(static_rhs)
    for op, signal in op_signal_pairs:
        terms.extend(_dynamic_term_entries(op, signal, sample_tlist))
    return qutip.QobjEvo(terms)


# A loky reusable executor respawns its worker pool after a short idle window
# (10 s by default), and that respawn costs ~3 s on the next sweep. Sweeps in an
# interactive session arrive minutes apart, so the pool is kept warm for an hour.
_POOL_IDLE_TIMEOUT_S = 3600


def _warmup_noop(_idx: Any = None) -> None:
    """Force loky fork/import out of timed regions (top-level no-op).

    Must be a module-level function (loky pickles tasks by reference; lambdas
    and closures are not picklable).
    """
    return None


class QuTiPBackend(Backend):
    """Concrete backend backed by QuTiP. ``Operator`` = ``State`` = ``qutip.Qobj``.

    Example
    -------
    >>> from quchip.backend.qutip import QuTiPBackend
    >>> backend = QuTiPBackend()
    >>> a = backend.destroy(3)
    >>> float((a.dag() * a).diag()[2].real)  # doctest: +SKIP
    2.0
    """

    # Batches smaller than this run sequentially in-process: the loky pool's
    # fork/import/IPC overhead dominates a handful of fast QuTiP solves.
    _PARALLEL_MIN_BATCH = 8

    def __init__(self) -> None:
        # Per-(kind, dim) memo of the Fock-space operator factories. These are
        # rebuilt many times during Hamiltonian assembly yet depend only on the
        # truncation ``n`` (always a concrete int). Kept strictly QuTiP-local:
        # a *shared* operator cache would leak ``DynamicJaxprTracer`` operators
        # when poisoned inside a ``jax.jit`` trace and break a later
        # ``grad``/``vmap`` — QuTiP factories always return pure ``Qobj``, so
        # this cache is provably tracer-free.
        self._op_cache: dict[tuple[str, int], Operator] = {}

    # ------------------------------------------------------------------
    # Array / scalar surface — QuTiP natively returns Qobj-flavoured answers
    # ------------------------------------------------------------------

    @property
    def array_module(self) -> Any:
        """Return the array module for backend-aware numeric code (``numpy``)."""
        return np

    def to_array(self, op: Operator) -> Any:
        """Return a dense ``numpy`` array for *op*."""
        if isinstance(op, Qobj):
            return np.asarray(op.full(), dtype=complex)
        return np.asarray(op, dtype=complex)

    def overlap(self, a: State, b: State) -> complex:
        """Return the scalar inner product ⟨a|b⟩ for two kets."""
        value = a.dag() @ b
        if isinstance(value, Qobj):
            return complex(value.full()[0, 0])
        return complex(value)

    def norm(self, state_or_op: State | Operator) -> float:
        """Return the norm of a state or operator via ``Qobj.norm``."""
        return float(state_or_op.norm())

    def trace(self, op: Operator) -> complex:
        """Return the scalar trace ``Tr(op)`` via ``Qobj.tr``."""
        return complex(op.tr())

    # ------------------------------------------------------------------
    # Operator / state factories — defer to QuTiP's own constructors
    # ------------------------------------------------------------------

    def _cached_op(self, kind: str, n: int, factory: Callable[[int], Operator]) -> Operator:
        key = (kind, n)
        cached = self._op_cache.get(key)
        if cached is None:
            cached = factory(n)
            self._op_cache[key] = cached
        return cached

    def destroy(self, n: int) -> Operator:
        """Return the annihilation operator for an *n*-level Fock space (``qutip.destroy``, memoized)."""
        return self._cached_op("destroy", n, qutip.destroy)

    def create(self, n: int) -> Operator:
        """Return the creation operator for an *n*-level Fock space (``qutip.create``, memoized)."""
        return self._cached_op("create", n, qutip.create)

    def number(self, n: int) -> Operator:
        """Return the number operator for an *n*-level Fock space (``qutip.num``, memoized)."""
        return self._cached_op("number", n, qutip.num)

    def identity(self, n: int) -> Operator:
        """Return the identity operator for an *n*-level space (``qutip.qeye``, memoized)."""
        return self._cached_op("identity", n, qutip.qeye)

    def from_array(self, data: Any, dims: list[list[int]] | None = None) -> Operator:
        """Construct a ``Qobj`` from a dense matrix with optional row/col *dims*."""
        return Qobj(data, dims=dims)

    def to_canonical_operator(self, op: Operator) -> Any:
        """Serialize a ``Qobj`` into the backend-agnostic canonical IR (CSR or dense)."""
        from quchip.engine.ir import CanonicalOperator

        if isinstance(op, Qobj):
            dims = tuple(op.dims[0])
            labels = tuple(str(i) for i in range(len(dims)))
            if type(op.data).__name__ != "Dense":
                csr = op.to("CSR").data_as("csr_matrix")
                return CanonicalOperator.from_csr(
                    csr.data, csr.indices, csr.indptr,
                    shape=op.shape, dims=dims, basis="fock", subsystem_labels=labels,
                )
            return CanonicalOperator.from_dense(
                np.asarray(op.full(), dtype=complex),
                dims=dims, basis="fock", subsystem_labels=labels,
            )

        arr = np.asarray(op, dtype=complex)
        return CanonicalOperator.from_dense(
            arr, dims=(arr.shape[0],), basis="fock", subsystem_labels=("0",),
        )

    def from_canonical_operator(self, canonical: Any) -> Operator:
        """Reconstruct a ``Qobj`` from the canonical IR payload."""
        return self._canonical_to_qobj(canonical)

    def coerce_operator(self, op: Operator) -> Operator:
        """Wrap an array-like operator into a ``Qobj`` (native ``Qobj`` passthrough)."""
        if isinstance(op, Qobj):
            return op
        return self.from_array(np.asarray(op, dtype=complex))

    def dag(self, op: Operator) -> Operator:
        """Return the Hermitian conjugate ``op†`` via ``Qobj.dag`` (array-likes coerced first)."""
        return self.coerce_operator(op).dag()

    def eigenenergies(self, op: Operator) -> Any:
        """Return the ascending eigenvalues of a Hermitian operator (``Qobj.eigenenergies``)."""
        return op.eigenenergies()

    def eigensystem_data(self, op: Operator) -> EigensystemData:
        """Return ascending eigenvalues, eigenvector matrix, and primed eigenstate kets.

        A ``Qobj`` uses ``Qobj.eigenstates`` verbatim to preserve the exact
        degenerate-subspace basis; dense inputs fall through to the protocol
        default.
        """
        if isinstance(op, Qobj):
            # Keep ``Qobj.eigenstates()`` verbatim: it preserves the exact
            # degenerate-subspace basis that ``operator_in_dressed_basis``
            # relies on (np.linalg.eigh reorders degenerate eigenvectors).
            # The kets are produced regardless, so prime the cache directly.
            evals, states = op.eigenstates()
            states = list(states)
            evecs = np.column_stack(
                [np.asarray(state.full(), dtype=complex).reshape(-1) for state in states]
            )
            return EigensystemData(
                eigenvalues=evals,
                eigenvector_matrix=evecs,
                _states_cache=states,
            )

        # Dense (non-Qobj) inputs use the protocol default verbatim: it densifies
        # via this backend's ``to_array``/``from_array``, so the kets are the same
        # ``Qobj`` columns (dims inferred to ``[[n], [1]]``) as a hand-rolled branch.
        return super().eigensystem_data(op)

    def expect(self, op: Operator, state: State) -> complex:
        """Return the expectation value ⟨op⟩ for a ket or density matrix via ``qutip.expect``."""
        return qutip.expect(op, state)

    def ptrace(self, state: State, keep: int | list[int], dims: list[int]) -> State:
        """Reduce onto subsystem(s) *keep* via ``Qobj.ptrace`` (partial trace).

        Rebuilds the composite *dims* when the incoming state carries flat dims.
        """
        # QuTiP's Qobj.ptrace needs the correct composite dims. Rebuild when the
        # incoming state carries flat dims (as when assembled outside tensor).
        if isinstance(state, Qobj):
            if state.dims[0] != dims and len(dims) > 1:
                new_dims = [dims, [1] * len(dims)] if state.isket else [dims, dims]
                state = Qobj(state.data, dims=new_dims)
        return state.ptrace(keep)

    def permute_state(self, state: State, dims: Sequence[int], order: Sequence[int]) -> State:
        """Reorder a ``Qobj``'s subsystems via ``Qobj.permute`` (sparse-friendly, keeps dims).

        Rebuilds composite ``dims`` first when the incoming state carries
        flat dims, mirroring :meth:`ptrace`. Non-``Qobj`` inputs fall
        through to the protocol default.
        """
        if not isinstance(state, Qobj):
            return super().permute_state(state, dims, order)
        dims = list(dims)
        if state.dims[0] != dims and len(dims) > 1:
            new_dims = [dims, [1] * len(dims)] if state.isket else [dims, dims]
            state = Qobj(state.data, dims=new_dims)
        return state.permute(list(order))

    def tensor(self, *operators: Operator) -> Operator:
        """Return the tensor product of operators via ``qutip.tensor`` (array-likes coerced first)."""
        return qutip.tensor([self.coerce_operator(op) for op in operators])

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def embed_two_body(
        self,
        op_ab: Operator,
        index_a: int,
        index_b: int,
        dims: Sequence[int],
    ) -> Operator:
        """Embed a two-body operator on devices *index_a* ⊗ *index_b* into the full space.

        Reorders subsystems (via sparse SWAP when ``index_a > index_b``) and
        identity-pads spectators without densifying the full matrix.
        """
        op_ordered, first_idx, second_idx = self._reorder_two_body_op(op_ab, index_a, index_b, dims)
        return self._embed_ordered_two_body(op_ordered, first_idx, second_idx, dims)

    @staticmethod
    def _reorder_two_body_op(
        op_ab: Qobj, index_a: int, index_b: int, dims: Sequence[int]
    ) -> tuple[Qobj, int, int]:
        """Validate indices; SWAP-permute subsystems when ``index_a > index_b``.

        The SWAP is done on the sparse CSR structure (COO-remap) so nothing
        gets densified here.
        """
        validate_two_body_indices(index_a, index_b, dims)
        expected_dim = dims[index_a] * dims[index_b]
        if op_ab.shape[0] != expected_dim:
            raise ValueError(
                f"Two-body operator dimension {op_ab.shape[0]} does not match "
                f"dims[{index_a}]*dims[{index_b}] = {expected_dim}"
            )

        d_a, d_b = dims[index_a], dims[index_b]
        if index_a < index_b:
            target_dims = [[d_a, d_b], [d_a, d_b]]
            op_ordered = op_ab if op_ab.dims == target_dims else Qobj(op_ab.data, dims=target_dims)
            return op_ordered, index_a, index_b

        # SWAP via sparse index remapping to avoid densifying sparse operators.
        coo = op_ab.to("CSR").data_as("csr_matrix").tocoo()
        row_a, row_b = np.divmod(coo.row, d_b)
        col_a, col_b = np.divmod(coo.col, d_b)
        dim_total = d_a * d_b
        swapped = sparse.coo_matrix(
            (coo.data, (row_b * d_a + row_a, col_b * d_a + col_a)),
            shape=(dim_total, dim_total),
        ).tocsr()
        return Qobj(swapped, dims=[[d_b, d_a], [d_b, d_a]]), index_b, index_a

    @staticmethod
    def _embed_ordered_two_body(
        op_ordered: Qobj, first_idx: int, second_idx: int, dims: Sequence[int]
    ) -> Qobj:
        """Embed a positionally-ordered two-body op; adjacent → direct tensor, else permute."""
        n_devices = len(dims)
        if second_idx == first_idx + 1:
            ops: list[Qobj] = []
            for i in range(n_devices):
                if i == first_idx:
                    ops.append(op_ordered)
                elif i == second_idx:
                    continue
                else:
                    ops.append(qutip.qeye(dims[i]))
            return qutip.tensor(ops)

        reorder, inverse_reorder = compute_two_body_permutation(first_idx, second_idx, dims)
        reordered_dims = [dims[j] for j in reorder]
        reordered_ops: list[Qobj] = [op_ordered]
        reordered_ops.extend(qutip.qeye(reordered_dims[j]) for j in range(2, len(reorder)))
        return qutip.tensor(reordered_ops).permute(inverse_reorder)

    # ------------------------------------------------------------------
    # State factories
    # ------------------------------------------------------------------

    def basis(self, n: int, k: int) -> State:
        """Return the Fock basis ket |k⟩ in an *n*-level space (``qutip.basis``)."""
        return qutip.basis(n, k)

    def tensor_states(self, *states: State) -> State:
        """Return the tensor product of states via ``qutip.tensor``."""
        return qutip.tensor(list(states))

    def coherent(self, n: int, alpha: complex) -> State:
        """Return the coherent state |α⟩ truncated to *n* Fock levels (``qutip.coherent``)."""
        return qutip.coherent(n, alpha)

    def state_to_dm(self, state: State) -> State:
        """Return a density matrix; pass through if *state* is already one."""
        if isinstance(state, Qobj) and not state.isket:
            return state
        return qutip.ket2dm(state)

    def is_ket(self, state: State) -> bool:
        """Return whether *state* is a ket rather than a density matrix (``Qobj.isket``)."""
        return state.isket

    # ------------------------------------------------------------------
    # Solver options / heuristics
    # ------------------------------------------------------------------

    def resolve_solver_options(
        self,
        options: dict[str, Any],
        *,
        metadata: dict[str, Any],
        tlist: Any,
    ) -> dict[str, Any]:
        """Fill in ``nsteps`` and ``max_step`` from Hamiltonian-metadata heuristics when unset.

        ``nsteps`` is an abort ceiling on the integrator's total internal
        step count, not a step-size bound — see :func:`default_solver_steps`
        for the heuristic that derives it from the Hamiltonian's fastest
        frequency scale.

        ``max_step`` bounds the size of an individual adaptive step. Without
        it, QuTiP's adaptive integrator can step clean over a finite-support
        pulse that sits inside a long idle span, sampling it only at points
        where its envelope happens to be near zero. When the user has not
        set ``max_step`` and ``metadata["max_step_ns"]`` carries a concrete,
        positive, finite value — half the narrowest window across the
        Hamiltonian's dynamic terms, computed by
        :func:`~quchip.engine.solver_hints._solver_hint_metadata` — it is
        used as the ceiling. An explicit user ``max_step`` is always
        authoritative, including QuTiP's own ``max_step=0`` (unbounded):
        the key's *presence* in ``options`` decides, not its truthiness.

        This is QuTiP's single option-merge boundary (shared by the single
        and batched solve paths), so the dynamiqs-only ``gradient`` knob —
        which QuTiP's ``SolverOptions`` rejects — is stripped here exactly
        once for portability; downstream runners trust the merged dict.
        """
        resolved = dict(options)
        if "nsteps" not in resolved:
            default = self._default_nsteps(metadata, tlist)
            if default is not None:
                resolved["nsteps"] = default
        if "max_step" not in resolved:
            max_step_ns = maybe_concrete_scalar(metadata.get("max_step_ns"))
            if max_step_ns is not None and max_step_ns > 0 and np.isfinite(max_step_ns):
                resolved["max_step"] = max_step_ns
        resolved.pop("gradient", None)
        return resolved

    _default_nsteps = staticmethod(default_solver_steps)

    def coerce_state(self, state: State, dims: tuple[int, ...] | None = None) -> State:
        """Wrap a foreign-native array state (ket or density matrix) into a ``Qobj``.

        Used when a per-call ``backend="qutip"`` override consumes states
        built under the dynamiqs backend. ``dims`` restores the tensor
        structure QuTiP's solvers require to match the Hamiltonian's dims.
        """
        if isinstance(state, Qobj):
            return state
        arr = np.asarray(state)
        if arr.ndim == 1:
            arr = arr[:, None]
        subsystem = [int(d) for d in dims] if dims else [arr.shape[0]]
        qdims = [subsystem, subsystem] if arr.shape[1] == arr.shape[0] else [subsystem, [1] * len(subsystem)]
        return Qobj(arr, dims=qdims)

    # ------------------------------------------------------------------
    # Single-problem solver dispatch
    # ------------------------------------------------------------------

    def sesolve(
        self,
        H: Any,
        psi0: State,
        tlist: Any,
        e_ops: list[Operator] | None = None,
        options: dict[str, Any] | None = None,
    ) -> SolverResult:
        """Solve the Schrödinger equation — wraps ``qutip.solver.sesolve.SESolver``."""
        runner = SESolver(self._coerce_solver_rhs(H), options=self._runner_options(options))
        result = runner.run(psi0, tlist, e_ops=e_ops)
        return self._wrap_result(result, solver="sesolve", extra_stats=self._solver_stats(runner))

    def mesolve(
        self,
        H: Any,
        rho0: State,
        tlist: Any,
        c_ops: list[Operator] | None = None,
        e_ops: list[Operator] | None = None,
        options: dict[str, Any] | None = None,
    ) -> SolverResult:
        """Solve the Lindblad master equation — wraps ``qutip.solver.mesolve.MESolver``."""
        runner = MESolver(self._coerce_solver_rhs(H), c_ops, options=self._runner_options(options))
        result = runner.run(rho0, tlist, e_ops=e_ops)
        return self._wrap_result(result, solver="mesolve", extra_stats=self._solver_stats(runner))

    @staticmethod
    def _runner_options(options: dict[str, Any] | None) -> dict[str, Any]:
        """Trust the already-merged options; only fill defaults when ``None``.

        The single option-merge boundary (:meth:`resolve_solver_options`,
        reached via :meth:`_merge_options`) has already applied
        ``_DEFAULT_SOLVE_OPTIONS`` and stripped the dynamiqs-only ``gradient``
        knob, so a supplied dict is forwarded as-is (a defensive copy) —
        re-merging here would duplicate that work. A bare ``None`` (a direct
        solver call without the merge boundary) still gets the store-state
        defaults.
        """
        if options is None:
            return dict(_DEFAULT_SOLVE_OPTIONS)
        return dict(options)

    # ------------------------------------------------------------------
    # Batched solver dispatch (parallel via loky)
    # ------------------------------------------------------------------

    def parallel_solve_problems(
        self,
        problems: list[Any],
        *,
        progress: bool = True,
    ) -> list[SolverResult] | None:
        """Solve large structurally heterogeneous problem lists through loky workers."""
        if len(problems) < self._PARALLEL_MIN_BATCH:
            return None
        return self._parallel_map(
            task=self.solve_problem,
            items=problems,
            n_jobs=-1,
            progress=progress,
            desc="Sweep (independent)",
        )

    def batched_sesolve(
        self,
        problems: list[dict[str, Any]],
        *,
        n_jobs: int = -1,
        progress: bool = True,
    ) -> list[SolverResult]:
        """Run sesolve in parallel via loky workers; fall back to sequential on failure."""
        return self._batched_solve(problems, solver_fn="sesolve", n_jobs=n_jobs, progress=progress)

    def batched_mesolve(
        self,
        problems: list[dict[str, Any]],
        *,
        n_jobs: int = -1,
        progress: bool = True,
    ) -> list[SolverResult]:
        """Run mesolve in parallel via loky workers; fall back to sequential on failure."""
        return self._batched_solve(problems, solver_fn="mesolve", n_jobs=n_jobs, progress=progress)

    def _batched_solve(
        self,
        problems: list[dict[str, Any]],
        *,
        solver_fn: str,
        n_jobs: int = -1,
        progress: bool = True,
    ) -> list[SolverResult]:
        """Dispatch each problem dict through :meth:`sesolve` / :meth:`mesolve` in parallel."""
        solve = getattr(self, solver_fn)
        return self._parallel_map(
            task=lambda problem: solve(**problem),
            items=problems,
            n_jobs=n_jobs,
            progress=progress,
            desc=f"Sweep ({solver_fn})",
        )

    # ------------------------------------------------------------------
    # Typed batched-IR surface
    # ------------------------------------------------------------------

    def prepare_hamiltonian(
        self,
        description: Any,
        tlist: Any | None = None,
    ) -> PreparedHamiltonian:
        """Convert a :class:`HamiltonianDescription` into a ``Qobj`` or ``QobjEvo``.

        Each dynamic coefficient is band-normalized: every carrier stays
        analytic while only its slow, carrier-free envelope is sampled
        (on *tlist*, locally densified around any window edge — see
        :func:`_band_coefficient` / :func:`_augmented_sample_grid`). This
        is exact regardless of how resonant a carrier is — the lab frame
        no longer accumulates the cubic-spline error that pre-sampling the
        full ``envelope·carrier`` product caused.
        """
        static_rhs = self._sum_terms(description.static_terms, self._canonical_to_qobj)
        metadata = dict(description.metadata)

        if not description.dynamic_terms:
            if static_rhs is None:
                raise ValueError("HamiltonianDescription must contain at least one static or dynamic term.")
            return PreparedHamiltonian(rhs=static_rhs, metadata=metadata)

        sample_tlist = self._resolve_envelope_sample_tlist(tlist)
        op_signal_pairs = (
            (self._canonical_to_qobj(operator), signal)
            for operator, signal in self._scalar_dynamic_terms(description)
        )
        rhs = _assemble_qobjevo(static_rhs, op_signal_pairs, sample_tlist)
        return PreparedHamiltonian(rhs=rhs, metadata=metadata)

    def prepare_batch(self, description: Any, tlist: Any) -> DeferredBatch:
        """Build a deferred-construction batch; per-element ``QobjEvo`` is built in workers.

        Each unique :class:`CanonicalOperator` is converted exactly once
        (shared across elements) and only the slow, carrier-free envelope
        is sampled on the user grid, locally densified around any window
        edge (carriers stay analytic — see :func:`_band_coefficient`).
        Final ``QobjEvo`` assembly lives in :meth:`solve_batch` so it runs
        inside loky workers, keeping the main process overhead O(1) in
        batch size.
        """
        cached_qobj = self._make_op_cache()
        static_rhs = self._sum_terms(description.static_terms, cached_qobj)
        dynamic_qobjs = tuple(cached_qobj(op) for op in description.dynamic_operators)
        sample_tlist: Any = None
        if dynamic_qobjs:
            sample_tlist = self._resolve_envelope_sample_tlist(tlist)

        shared = _QuTiPBatchShared(
            static_rhs=static_rhs,
            dynamic_qobjs=dynamic_qobjs,
            sample_tlist=sample_tlist,
            dynamic_signals=tuple(description.dynamic_signals),
        )
        return DeferredBatch(
            shared=shared,
            batch_size=description.batch_size,
            metadata=dict(description.metadata),
            tlist=tlist,
        )

    def solve_batch(self, batch: Any, *, progress: bool = True) -> list[SolverResult]:
        """Solve a :class:`SolveBatch` with per-element ``QobjEvo`` built in loky workers."""
        if batch.batch_size == 0:
            return []

        prepared = self.prepare_batch(batch.hamiltonian, batch.tlist)
        tlist_arr, c_ops, solver_name, opts, e_ops_arg = self._resolve_batch_config(batch, prepared)

        shared = prepared.shared
        if not isinstance(shared, _QuTiPBatchShared):
            raise RuntimeError(
                "QuTiPBackend.solve_batch requires DeferredBatch.shared to be a "
                f"_QuTiPBatchShared instance, got {type(shared).__name__}."
            )

        solve = getattr(self, solver_name)
        chip_dims = getattr(batch.chip, "dims", None)
        initial_states = [self.coerce_state(s, dims=chip_dims) for s in batch.initial_states]

        def build_and_solve(idx: int) -> SolverResult:
            rhs = self._build_element_rhs(shared, idx)
            kwargs = self._element_solver_kwargs(
                solver_name,
                rhs,
                initial_states[idx],
                tlist_arr,
                e_ops=e_ops_arg,
                c_ops=c_ops,
                options=dict(opts),
            )
            return solve(**kwargs)

        return self._parallel_map(
            task=build_and_solve,
            items=list(range(batch.batch_size)),
            n_jobs=-1,
            progress=progress,
            desc=f"Sweep ({solver_name})",
        )

    # ------------------------------------------------------------------
    # Internal: per-element RHS construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_element_rhs(shared: _QuTiPBatchShared, idx: int) -> Any:
        """Build the ``QobjEvo`` for batch element *idx* from shared state."""
        op_signal_pairs = (
            (op, shared.dynamic_signals[slot][idx].signal)
            for slot, op in enumerate(shared.dynamic_qobjs)
        )
        return _assemble_qobjevo(shared.static_rhs, op_signal_pairs, shared.sample_tlist)

    # ------------------------------------------------------------------
    # Internal: sample-tlist sizing
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_envelope_sample_tlist(tlist: Any) -> Any:
        """Return the base sample grid for interpolating carrier-free slow envelopes; ``None`` otherwise.

        Carriers are kept analytic (see :func:`_band_coefficient`), so no
        dense per-carrier oversampling is needed — only the slow envelope
        is interpolated. This grid's *density* governs fidelity for a
        **non-windowed** envelope only: a too-coarse output tlist is
        replaced by a uniform grid of ``_MIN_ENVELOPE_SAMPLES`` points
        over the same span, and a user grid at least that dense is used
        as-is. For a **windowed** envelope this method's output feeds only
        :func:`_augmented_sample_grid`'s canonical skeleton, which reads
        just this grid's *endpoints* — window fidelity there comes
        entirely from the window's own local subgrid, independent of
        whatever density this method happened to return, so the output
        tlist never determines a windowed pulse's interpolation fidelity.
        Returns ``None`` (callable coefficients) when the grid is too
        short, degenerate, or its endpoints are JAX tracers, mirroring the
        dynamiqs path.
        """
        if tlist is None or len(tlist) < 2:
            return None
        t0 = maybe_concrete_scalar(tlist[0])
        t1 = maybe_concrete_scalar(tlist[-1])
        if t0 is None or t1 is None or t1 <= t0:
            return None
        if len(tlist) >= _MIN_ENVELOPE_SAMPLES:
            return tlist
        return np.linspace(float(t0), float(t1), _MIN_ENVELOPE_SAMPLES)

    # ------------------------------------------------------------------
    # Internal: loky parallel-map helper
    # ------------------------------------------------------------------

    def warmup(self, n_jobs: int = -1) -> None:
        """Pre-spawn the reusable loky worker pool out of any timed region.

        Entirely optional: ``solve_batch`` spins the pool up on demand, so
        the only reason to call this is to move that one-time spin-up cost
        out of a region you are timing (benchmarks). Ordinary scripts and
        notebooks never need it.

        Forks the workers and pays the per-worker import cost up front by
        mapping a no-op over the worker count. A subsequent sweep then reuses
        the live pool instead of paying cold spin-up inside the timed solve.
        No-op (with a warning) when loky is unavailable.
        """
        try:
            executor = self._get_executor(n_jobs)
        except (ModuleNotFoundError, ImportError) as exc:
            warnings.warn(
                f"QuTiP sweep warmup unavailable ({exc!r}); the loky pool will "
                "spin up lazily on the first sweep instead.",
                stacklevel=2,
            )
            return
        workers = getattr(executor, "_max_workers", None) or (os.cpu_count() or 1)
        list(executor.map(_warmup_noop, range(workers)))

    @staticmethod
    def _get_executor(n_jobs: int) -> Any:
        """Return the process-wide reusable loky executor.

        ``reuse=True`` hands back the same live pool across sweeps; the long
        idle timeout avoids loky's ~10 s-default respawn, which otherwise
        stalls the following sweep by ~3 s. Worker count follows ``n_jobs``
        (``-1``/``None`` → all cores), clamped to at least one.
        """
        from loky import get_reusable_executor

        workers = os.cpu_count() if n_jobs in (-1, None) else n_jobs
        workers = max(1, int(workers or 1))
        return get_reusable_executor(
            max_workers=workers,
            reuse=True,
            timeout=_POOL_IDLE_TIMEOUT_S,
        )

    @staticmethod
    def _shutdown_executor() -> None:
        """Shut down the pool (best-effort) so a poisoned reused pool self-heals next sweep."""
        try:
            from loky import get_reusable_executor

            get_reusable_executor(reuse=True).shutdown(wait=False)
        except Exception:
            pass

    def _parallel_map(
        self,
        *,
        task: Callable[[Any], SolverResult],
        items: list[Any],
        n_jobs: int,
        progress: bool,
        desc: str,
    ) -> list[SolverResult]:
        """Run *task* over *items*, picking the cheapest dispatch path.

        Small batches (``< _PARALLEL_MIN_BATCH``) run sequentially in-process —
        the loky pool's fork/import/IPC overhead dominates a handful of fast
        QuTiP solves. Larger batches dispatch through the process-wide reusable
        loky executor. On any worker-pool failure the pool is shut down (so it
        self-heals next time) and execution falls back to sequential, keeping
        SolverResult semantics identical to the parallel path.
        """
        from tqdm import tqdm

        def sequential() -> list[SolverResult]:
            iterator = tqdm(items, desc=desc) if progress else items
            return [task(item) for item in iterator]

        if len(items) < self._PARALLEL_MIN_BATCH:
            return sequential()

        try:
            executor = self._get_executor(n_jobs)
        except (ModuleNotFoundError, ImportError) as exc:
            warnings.warn(
                f"QuTiP batched solve parallelism unavailable ({exc!r}); "
                "falling back to sequential execution.",
                stacklevel=2,
            )
            return sequential()

        try:
            mapped = executor.map(task, items)
            if progress:
                return list(tqdm(mapped, total=len(items), desc=desc))
            return list(mapped)
        except Exception as exc:
            self._shutdown_executor()
            warnings.warn(
                f"QuTiP batched solve parallelism unavailable ({exc!r}); "
                "falling back to sequential execution.",
                stacklevel=2,
            )
            return sequential()

    # ------------------------------------------------------------------
    # Internal: Qobj <-> CanonicalOperator conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_solver_rhs(H: Any) -> Any:
        """Ensure *H* is a ``Qobj`` or ``QobjEvo`` accepted by QuTiP solvers."""
        if isinstance(H, (Qobj, qutip.QobjEvo)):
            return H
        return qutip.QobjEvo(H)

    @staticmethod
    def _canonical_to_qobj(canonical: Any) -> Qobj:
        """Reconstruct a ``Qobj`` from a ``CanonicalOperator`` (dense or sparse)."""
        dims = [list(canonical.dims), list(canonical.dims)]
        if canonical.layout == "dense":
            return Qobj(np.asarray(canonical.values, dtype=complex), dims=dims, dtype="Dense")
        return Qobj(QuTiPBackend._canonical_to_csr_matrix(canonical), dims=dims, dtype="CSR")

    @staticmethod
    def _canonical_to_csr_matrix(canonical: Any) -> sparse.csr_matrix:
        """Convert any canonical layout (``csr``/``dia``/dense fallback) to SciPy CSR."""
        if canonical.layout == "csr":
            return sparse.csr_matrix(
                (
                    np.asarray(canonical.values, dtype=complex),
                    np.asarray(canonical.indices, dtype=int),
                    np.asarray(canonical.indptr, dtype=int),
                ),
                shape=canonical.shape,
            )
        if canonical.layout == "dia":
            dia = sparse.dia_matrix(
                (
                    np.asarray(canonical.values, dtype=complex),
                    np.asarray(canonical.offsets, dtype=int),
                ),
                shape=canonical.shape,
            )
            return dia.tocsr()
        return sparse.csr_matrix(np.asarray(canonical.values, dtype=complex))

    # ------------------------------------------------------------------
    # Internal: SolverResult packaging
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_result(
        qutip_result: Any,
        solver: str,
        *,
        extra_stats: dict[str, Any] | None = None,
    ) -> SolverResult:
        """Convert a ``qutip.Result`` to the backend-agnostic :class:`SolverResult`."""
        states = list(qutip_result.states) if qutip_result.states else None
        expect = [list(e) for e in qutip_result.expect] if qutip_result.expect else None
        stats = dict(qutip_result.stats) if getattr(qutip_result, "stats", None) else {}
        if extra_stats:
            stats.update({key: value for key, value in extra_stats.items() if value is not None})

        # Prefer QuTiP's own ``final_state`` (populated when ``store_final_state=True``)
        # so the final state survives even when the full trajectory is not stored
        # (``store_states=False`` — e.g. the auto policy for ``e_ops``-only solves).
        # Fall back to the last trajectory entry when only ``store_states`` is on.
        final_state = getattr(qutip_result, "final_state", None)
        if final_state is None and states:
            final_state = states[-1]

        return SolverResult(
            times=qutip_result.times,
            states=states,
            expect=expect,
            final_state=final_state,
            stats=stats,
            solver=solver,
        )

    @staticmethod
    def _solver_stats(solver_runner: Any) -> dict[str, Any]:
        """Return integrator diagnostics — currently only ``nsteps`` when exposed."""
        nsteps = QuTiPBackend._extract_nsteps(solver_runner)
        return {"nsteps": nsteps} if nsteps is not None else {}

    @staticmethod
    def _extract_nsteps(solver_runner: Any) -> int | None:
        """Return the ODEPACK/LSODA step count via QuTiP/scipy internals; ``None`` if unavailable.

        Reaches into undocumented QuTiP/scipy plumbing — may break on
        upgrades. Purely diagnostic; no physics depends on it.
        """
        integrator = getattr(solver_runner, "_integrator", None)
        ode_solver = getattr(integrator, "_ode_solver", None)
        inner = getattr(ode_solver, "_integrator", None)
        if inner is None:
            return None

        iwork = getattr(inner, "iwork", None)
        if iwork is not None and len(iwork) > 10:
            # ODEPACK stores NST (successful internal steps) in IWORK(11).
            nsteps = int(iwork[10])
            if nsteps > 0:
                return nsteps

        nst = getattr(inner, "nst", None)
        if nst is not None:
            nsteps = int(nst)
            if nsteps > 0:
                return nsteps

        return None
