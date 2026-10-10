"""QuTiP backend implementation of the :mod:`quchip.backend.protocol` contract.

QuTiP stores operators as ``qutip.Qobj`` (sparse CSR by default, dense on
request). This backend lowers engine IR into ``Qobj`` / ``QobjEvo`` form and
dispatches the ``sesolve``/``mesolve`` drivers directly.

The reusable ``loky`` process pool parallelizes batched sweeps and
heterogeneous problem lists.

References
----------
* Johansson, Nation, Nori — *QuTiP 2*, Comput. Phys. Commun. 183, 1760 (2012)
* Breuer & Petruccione — *The Theory of Open Quantum Systems* (OUP, 2002)
"""
# QuTiP solvers release the GIL only partially, so processes are faster than
# threads. The workers do the final `QobjEvo` assembly, so the main process
# never pays for it.

from __future__ import annotations

import cmath
import math
import multiprocessing
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Sequence

import numpy as np
import qutip
from qutip import Qobj
from qutip.solver.mesolve import MESolver
from qutip.solver.sesolve import SESolver
from scipy import sparse

from quchip.backend._memory import require_memory
from quchip.backend._response import linear_response, stationary_condition_number
from quchip.utils.values import DeferredValue
from quchip.backend._dims import (
    default_solver_steps,
    validate_two_body_indices,
)
from quchip.backend.containers import (
    BatchSolveError,
    _DEFAULT_SOLVE_OPTIONS,
    EigensystemData,
    DeferredBatch,
    PreparedHamiltonian,
    PreparedStationary,
    LinearResponseSolverResult,
    SolverResult,
    SteadyStateSolverResult,
)
from quchip.backend.protocol import Backend, Operator, State
from quchip.engine.ir import (
    decompose_carrier_bands,
    evaluate_signal_program,
    signal_window_bounds,
)
from quchip.utils.jax_utils import maybe_concrete_scalar


@dataclass(frozen=True)
class _QuTiPBatchShared:
    """Shared per-batch state travelling through :attr:`DeferredBatch.shared`.

    Opaque to the engine; consumed by :meth:`QuTiPBackend.solve_batch` to
    build one ``QobjEvo`` per element inside a loky worker.
    """

    static_rhs: tuple[Qobj | None, ...]
    dynamic_qobjs: tuple[tuple[Qobj, ...], ...]
    sample_tlist: Any
    dynamic_signals: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True)
class _QuTiPBatchTask:
    """One compact point handed to a sequential or loky batch worker."""

    static_rhs: Qobj | None
    dynamic_qobjs: tuple[Qobj, ...]
    dynamic_signals: tuple[Any, ...]
    initial_state: Any
    c_ops: tuple[Qobj, ...]
    solver_name: str
    options: dict[str, Any]
    e_ops: tuple[Qobj, ...] | None


def _coeff_callable(signal: Any) -> Callable[[float, Any], complex]:
    """Build a scalar coefficient callable ``c(t)`` from a signal program AST."""
    def _coeff(time: float, _args: Any = None) -> complex:
        value = np.asarray(evaluate_signal_program(signal, time, xp=np), dtype=complex)
        return complex(value.item()) if value.ndim == 0 else complex(value.reshape(-1)[0])

    return _coeff


# Resolve slow, carrier-free envelopes even with a two-point output tlist.
_MIN_ENVELOPE_SAMPLES = 1001
# Window cost scales with pulse width, not idle-span length; the floor resolves sub-ns pulses.
_WINDOW_SUBGRID_POINTS_PER_NS = 40.0
_MIN_WINDOW_INTERIOR_SAMPLES = 41
# Zero-valued margins expose each discontinuity; a sparse skeleton covers the remaining zero plateau.
_WINDOW_EDGE_PADDING_NS = 1.0
_CANONICAL_BASE_SKELETON_POINTS = 3


def _collect_window_bounds(signal: Any) -> list[tuple[float, float]]:
    """Concrete absolute windows used to refine QuTiP's coefficient grid."""
    bounds = []
    for start, stop in signal_window_bounds(signal):
        start, stop = maybe_concrete_scalar(start), maybe_concrete_scalar(stop)
        if start is not None and stop is not None and stop > start:
            bounds.append((start, stop))
    return bounds


def _local_window_subgrid(start: float, stop: float) -> np.ndarray:
    """Resolve a window with exact edge nodes, at least 41 interior points, and zero-valued margins."""
    width = stop - start
    interior_n = max(int(np.ceil(width * _WINDOW_SUBGRID_POINTS_PER_NS)) + 1, _MIN_WINDOW_INTERIOR_SAMPLES)
    interior = np.linspace(start, stop, interior_n)

    pad = min(_WINDOW_EDGE_PADDING_NS, max(width, 1e-9))
    outside_n = max(int(np.ceil(pad * _WINDOW_SUBGRID_POINTS_PER_NS)), 3)
    before = np.linspace(start - pad, start, outside_n, endpoint=False)
    after = np.linspace(stop, stop + pad, outside_n + 1)[1:]

    return np.unique(np.concatenate([before, interior, after, [start, stop]]))


def _augmented_sample_grid(envelope: Any, base_grid: Any) -> np.ndarray:
    """Use the base grid without windows; otherwise combine the solve-span skeleton,
    feature times, adjacent edge floats and local subgrids. Window accuracy must
    not depend on output-tlist density or idle-span length."""
    bounds = _collect_window_bounds(envelope)
    base = np.asarray(base_grid, dtype=float)
    if not bounds:
        return base
    lo, hi = base[0], base[-1]
    from quchip.engine.sampling import signal_feature_times

    pieces = [np.linspace(lo, hi, _CANONICAL_BASE_SKELETON_POINTS)]
    for features in signal_feature_times(envelope):
        pieces.append(features[(features >= lo) & (features <= hi)])
    for start, stop in bounds:
        clipped_start, clipped_stop = max(start, lo), min(stop, hi)
        if clipped_stop < clipped_start:
            continue
        # Adjacent floats preserve a discontinuity without giving it a
        # finite interpolation ramp, including a pulse touching an endpoint.
        edges = np.array([np.nextafter(start, -np.inf), start, stop, np.nextafter(stop, np.inf)])
        pieces.append(edges[(edges >= lo) & (edges <= hi)])
        if clipped_stop > clipped_start:
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


# Linear interpolation cannot overshoot the discontinuous zero plateau, unlike cubic interpolation
# across dense pulse knots and sparse idle knots. Non-windowed envelopes retain the cubic default.
_WINDOWED_COEFFICIENT_ORDER = 1

# QuTiP 5 assembles a Lindblad superoperator term by term and keeps one term per
# time-dependent Hamiltonian part. With dense inputs, measured assembly peaks near
# six dense D²×D² arrays plus four per time-dependent part. With CSR inputs each
# term is added into a running sum whose output is sized for both operands before
# trimming, so the previous sum, the new term and the new sum coexist: measured
# peaks stay below three copies of the final storage.
_DENSE_SUPEROPERATOR_PEAK_COPIES = 6
_DENSE_SUPEROPERATOR_COPIES_PER_PART = 4
_SPARSE_SUPEROPERATOR_PEAK_COPIES = 3


def _operator_parts(operator: Any) -> list[Qobj]:
    """Return the constant operators of a ``Qobj`` or ``QobjEvo``."""
    if isinstance(operator, qutip.QobjEvo):
        return [part[0] if isinstance(part, list) else part for part in operator.to_list()]
    return [operator]


def _pairs_reaching(magnitudes: np.ndarray, atol: float) -> int:
    """Count ordered pairs of stored magnitudes whose product reaches ``atol``."""
    ordered = np.sort(magnitudes)
    if atol <= 0.0:
        return int(ordered.size**2)
    with np.errstate(divide="ignore"):
        partners = np.searchsorted(ordered, atol / ordered)
    return int(ordered.size**2 - partners.sum())


def _superoperator_peak_bytes(hamiltonian: Any, collapse_ops: Sequence[Any]) -> int:
    """Estimate QuTiP's peak memory while assembling a Lindblad superoperator.

    The sparse estimate counts the entries QuTiP keeps. spre(H) and spost(H)
    store nnz(H)·D entries each. A jump term c ⊗ c* keeps only pairs of entries
    whose product reaches QuTiP's tidy-up tolerance, and the jump terms of all
    operators share positions in the running sum, so their union is bounded by
    the pairs of the elementwise largest magnitude. Each c†c keeps the entries
    where |c|ᵀ|c| reaches the tolerance, and its two lifts store D copies.
    """
    hamiltonian_parts = _operator_parts(hamiltonian)
    collapse_parts = [_operator_parts(op) for op in collapse_ops]
    dimension = int(hamiltonian_parts[0].shape[0])
    copies = _DENSE_SUPEROPERATOR_PEAK_COPIES + _DENSE_SUPEROPERATOR_COPIES_PER_PART * (len(hamiltonian_parts) - 1)
    dense = copies * 16 * dimension**4
    parts = [*hamiltonian_parts, *(part for group in collapse_parts for part in group)]
    if any(isinstance(part.data, qutip.data.Dense) for part in parts):
        return dense

    def magnitude(group: list[Qobj]) -> Any:
        matrix = abs(group[0].to("CSR").data_as("csr_matrix"))
        for part in group[1:]:
            matrix = matrix + abs(part.to("CSR").data_as("csr_matrix"))
        return matrix.tocsr()

    atol = float(qutip.settings.core["auto_tidyup_atol"]) if qutip.settings.core["auto_tidyup"] else 0.0
    entries = 2 * dimension * sum(part.to("CSR").data_as("csr_matrix").nnz for part in hamiltonian_parts)
    if collapse_parts:
        magnitudes = [magnitude(group) for group in collapse_parts]
        # Every part pair of a time-dependent jump keeps its own coefficient.
        separate = sum(len(group) ** 2 * _pairs_reaching(matrix.data, atol)
                       for group, matrix in zip(collapse_parts, magnitudes))
        envelope = magnitudes[0]
        for matrix in magnitudes[1:]:
            envelope = envelope.maximum(matrix)
        shared = _pairs_reaching(envelope.data, atol)
        entries += separate if any(len(group) > 1 for group in collapse_parts) else min(separate, shared)
        products = [(matrix.T @ matrix).tocsr() for matrix in magnitudes]
        for product in products:
            product.data[product.data < atol] = 0.0
            product.eliminate_zeros()
        entries += 2 * dimension * sum(products[1:], start=products[0]).nnz
    entry_bytes = 16 + np.dtype(qutip.core.data.base.idxint_dtype).itemsize
    return min(dense, _SPARSE_SUPEROPERATOR_PEAK_COPIES * entry_bytes * entries)


def _require_superoperator_memory(hamiltonian: Any, collapse_ops: Sequence[Any], *, task: str, remedy: str) -> None:
    """Raise before QuTiP assembles a Liouvillian that cannot fit in memory."""
    dimension = int(_operator_parts(hamiltonian)[0].shape[0])
    require_memory(
        _superoperator_peak_bytes(hamiltonian, collapse_ops),
        task=f"{task} at Hilbert dimension D = {dimension}",
        remedy=remedy,
    )


# Dense canonical payloads with at most this fraction of nonzero entries are
# stored as CSR. Below it CSR uses less memory than dense storage, including in
# every superoperator term built from the operator.
_CSR_MAX_FILL = 0.25

# Cap diag at Hilbert D=64 for sesolve and Liouvillian D²=1024 (Hilbert D=32) for mesolve.
# The mesolve setup scales roughly as D⁶ in time and D⁴ in memory.
_MAX_STATIC_HILBERT_DIM = 64
_MAX_STATIC_LIOUVILLIAN_DIM = 1024


def _envelope_coefficient(envelope: Any, sample_tlist: Any) -> Any:
    """Sample windowed envelopes locally with linear interpolation to avoid edge ringing.
    Use the base grid otherwise, or an exact callable when no concrete grid is available."""
    if sample_tlist is None:
        return qutip.coefficient(_coeff_callable(envelope))
    bounds = _collect_window_bounds(envelope)
    try:
        grid = _augmented_sample_grid(envelope, sample_tlist)
    except NotImplementedError:
        return qutip.coefficient(_coeff_callable(envelope))
    arr = _sample_coeff_array(envelope, grid)
    if bounds:
        return qutip.coefficient(
            np.asarray(arr, dtype=complex),
            tlist=np.asarray(grid, dtype=float),
            order=_WINDOWED_COEFFICIENT_ORDER,
        )
    return qutip.coefficient(np.asarray(arr, dtype=complex), tlist=np.asarray(grid, dtype=float))


def _carrier_coefficient(freq: Any) -> Any:
    """Keep exp(i·freq·t) analytic, with angular freq in rad/ns.
    QuTiP evaluates the closure only at concrete times; this is not a JAX-traced path."""
    rate = 1j * complex(freq)

    def _carrier(t: float, *args: Any, **kwargs: Any) -> complex:
        return cmath.exp(rate * t)

    return qutip.coefficient(_carrier)


def _summed_envelope_coefficient(envelopes: Sequence[Any], sample_tlist: Any) -> Any:
    """Return one coefficient equal to the sum of the envelopes' own coefficients.

    Linear interpolants of windowed envelopes add exactly on the union of
    their grids, and cubic interpolants on one shared grid add their samples,
    since interpolation is linear in the data. Envelopes without a concrete
    grid keep their exact callables.
    """
    if sample_tlist is None or len(envelopes) == 1:
        parts = [_envelope_coefficient(envelope, sample_tlist) for envelope in envelopes]
    else:
        parts = []
        linear: list[tuple[np.ndarray, np.ndarray]] = []
        cubic: list[tuple[np.ndarray, np.ndarray]] = []
        for envelope in envelopes:
            try:
                grid = np.asarray(_augmented_sample_grid(envelope, sample_tlist), dtype=float)
            except NotImplementedError:
                parts.append(qutip.coefficient(_coeff_callable(envelope)))
                continue
            samples = _sample_coeff_array(envelope, grid)
            (linear if _collect_window_bounds(envelope) else cubic).append((grid, samples))
        if cubic and all(np.array_equal(grid, cubic[0][0]) for grid, _ in cubic):
            parts.append(qutip.coefficient(sum(samples for _, samples in cubic), tlist=cubic[0][0]))
        else:
            parts.extend(qutip.coefficient(samples, tlist=grid) for grid, samples in cubic)
        if linear:
            union = np.unique(np.concatenate([grid for grid, _ in linear]))
            total = np.zeros(union.shape, dtype=complex)
            for grid, samples in linear:
                nonzero = np.flatnonzero(samples)
                if nonzero.size == 0:
                    continue
                # The interpolant vanishes beyond the nodes bracketing its nonzero samples.
                lo = np.searchsorted(union, grid[max(nonzero[0] - 1, 0)])
                hi = np.searchsorted(union, grid[min(nonzero[-1] + 1, grid.size - 1)], side="right")
                nodes = union[lo:hi]
                total[lo:hi] += np.interp(nodes, grid, samples.real) + 1j * np.interp(nodes, grid, samples.imag)
            parts.append(qutip.coefficient(total, tlist=union, order=_WINDOWED_COEFFICIENT_ORDER))
    total = parts[0]
    for part in parts[1:]:
        total = total + part
    return total


def _qobj_key(op: Qobj) -> tuple:
    """Exact content key of an operator; equal operators stored differently may get different keys."""
    payload: tuple[bytes, ...]
    if isinstance(op.data, qutip.data.CSR):
        matrix = op.data_as("csr_matrix")
        payload = (matrix.indptr.tobytes(), matrix.indices.tobytes(), matrix.data.tobytes())
    else:
        payload = (np.ascontiguousarray(op.full()).tobytes(),)
    return (repr(op.dims), type(op.data).__name__, *payload)


def _assemble_qobjevo(static_rhs: Qobj | None, op_signal_pairs: Any, sample_tlist: Any) -> qutip.QobjEvo:
    """Combine an optional static operator and dynamic entries into a QobjEvo.

    Carrier bands acting through equal operators at one concrete frequency
    share a coefficient, ``(Σ_k envelope_k(t))·exp(i·freq·t)``, so a solver
    step applies each operator once per carrier frequency however many
    pulses and crosstalk paths drive it.
    """
    terms: list[Any] = [] if static_rhs is None else [static_rhs]
    groups: dict[tuple, tuple[Qobj, Any, list[Any]]] = {}
    for op, signal in op_signal_pairs:
        key = _qobj_key(op)
        for band in decompose_carrier_bands(signal):
            freq = maybe_concrete_scalar(band.freq)
            groups.setdefault((key, id(band) if freq is None else freq), (op, band.freq, []))[2].append(band.envelope)
    for op, freq, envelopes in groups.values():
        coefficient = _summed_envelope_coefficient(envelopes, sample_tlist)
        if maybe_concrete_scalar(freq) != 0.0:
            coefficient = coefficient * _carrier_coefficient(freq)
        terms.append([op, coefficient])
    return qutip.QobjEvo(terms)


def _canonical_key(op: Qobj) -> tuple:
    """Content key of an operator, equal for equal matrices however they are stored."""
    matrix = (op.data_as("csr_matrix") if isinstance(op.data, qutip.data.CSR) else sparse.csr_matrix(op.full())).copy()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return (repr(op.dims), matrix.indptr.astype(np.int64).tobytes(), matrix.indices.astype(np.int64).tobytes(),
            matrix.data.tobytes())


# Times at which a Hamiltonian's paired coefficients must be conjugate before
# its terms are lifted as Hermitian.
_HERMITIAN_CHECK_TIMES = 129


def _is_hermitian_sum(terms: Sequence[list], tlist: Any) -> bool:
    """Whether ``Σ_k c_k(t)·A_k`` is Hermitian: each operator's adjoint carries the conjugate coefficient."""
    classes: dict[tuple, tuple[Qobj, list[Any]]] = {}
    for op, coefficient in terms:
        classes.setdefault(_canonical_key(op), (op, []))[1].append(coefficient)
    span = np.asarray(tlist, dtype=float)
    times = np.linspace(span[0], span[-1], _HERMITIAN_CHECK_TIMES)
    sums = {key: np.array([sum(c(t) for c in coefficients) for t in times])
            for key, (_, coefficients) in classes.items()}
    for key, (op, _) in classes.items():
        partner = sums.get(_canonical_key(op.dag()))
        values = sums[key]
        if partner is None or not np.allclose(partner, values.conj(), rtol=1e-12,
                                              atol=1e-12 * np.abs(values).max(initial=0.0)):
            return False
    return True


def _lindblad_generator(hamiltonian: Any, collapse_ops: Sequence[Any], tlist: Any) -> tuple[Any, Qobj] | None:
    """Lift a Hermitian time-dependent Hamiltonian and its dissipators with one part per coefficient.

    QuTiP lifts ``ρH†`` separately, so a term ``c(t)·A`` becomes ``spre(A)``
    with ``c`` and ``spost(A†)`` with ``c̄``, and the dissipators join as a
    second constant part. For a Hermitian ``H(t)``, ``ρH† = ρH``: each term
    lifts to the single part ``-i(spre(A) - spost(A))`` and the static
    Hamiltonian shares one constant part with every dissipator, halving the
    time-dependent sparse products per step. A quchip Hamiltonian is
    Hermitian because every drive band carries its Hermitian partner; the
    pairing is checked on the operators and their coefficients.

    Returns the time-dependent generator and the constant part, or ``None``
    to leave the input to QuTiP's construction.
    """
    if not isinstance(hamiltonian, qutip.QobjEvo) or not all(isinstance(op, Qobj) for op in collapse_ops):
        return None
    static, dynamic = [], []
    for part in hamiltonian.to_list():
        if isinstance(part, Qobj):
            static.append(part)
        elif isinstance(part, list) and len(part) == 2 and isinstance(part[0], Qobj) and callable(part[1]):
            dynamic.append(part)
        else:
            return None
    if not dynamic or not (static or collapse_ops) or not _is_hermitian_sum(dynamic, tlist):
        return None
    constant = qutip.liouvillian(sum(static[1:], start=static[0]) if static else None, list(collapse_ops))
    generator = qutip.QobjEvo([[-1j * (qutip.spre(op) - qutip.spost(op)), coefficient] for op, coefficient in dynamic])
    return generator, constant


# Integrators that only apply the right-hand side to their state, so they can integrate a packed one.
_PACKED_STATE_METHODS = frozenset({"adams", "bdf", "lsoda", "dop853", "vern7", "vern9", "tsit5"})
# The packed right-hand side runs in Python; below this Hilbert dimension its call
# overhead outweighs the halved sparse and vector work (break-even near 40).
_PACKED_STATE_MIN_DIMENSION = 48
# SciPy releases the GIL in sparse products, so a packed product with at least
# this many stored entries splits its rows over threads; a smaller one finishes
# before the hand-off pays.
_THREADED_PRODUCT_MIN_NNZ = 200_000
# Sparse products are bound by memory bandwidth and stop gaining near four threads.
_MAX_PRODUCT_THREADS = 4
# Scaling the inputs by the coefficients joins the threads when it writes at least
# this many entries; a shorter pass finishes before a second hand-off pays.
_THREADED_SCALING_MIN_ENTRIES = 500_000
# Thread pools by (process id, size): a forked child starts its own pool.
_product_pools: dict[tuple[int, int], ThreadPoolExecutor] = {}


def _thread_setting(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    return int(value) if value.isdigit() and int(value) > 0 else None


def _product_threads() -> int:
    """Threads for a packed master-equation product.

    ``QUCHIP_NUM_THREADS`` sets the count. Otherwise a child process, such as
    a sweep worker, uses one thread because its siblings share the machine;
    ``OMP_NUM_THREADS`` applies next, and the default is up to four of the
    CPUs available to the process.
    """
    explicit = _thread_setting("QUCHIP_NUM_THREADS")
    if explicit is not None:
        return explicit
    if multiprocessing.parent_process() is not None:
        return 1
    available = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    return _thread_setting("OMP_NUM_THREADS") or max(1, min(_MAX_PRODUCT_THREADS, available))


def _row_blocks(matrix: sparse.csr_matrix, count: int) -> list[tuple[int, int, sparse.csr_matrix]]:
    """Contiguous row blocks of a CSR matrix holding about equal numbers of stored entries."""
    if count <= 1:
        return [(0, matrix.shape[0], matrix)]
    bounds = np.searchsorted(matrix.indptr, np.linspace(0, matrix.nnz, count + 1)).clip(0, matrix.shape[0])
    bounds[0], bounds[-1] = 0, matrix.shape[0]
    return [(int(lo), int(hi), matrix[lo:hi]) for lo, hi in zip(bounds[:-1], bounds[1:]) if hi > lo]


def _share(pool: ThreadPoolExecutor, task: Callable[[Any], None], items: Sequence[Any]) -> None:
    """Run ``task`` over ``items``: the first on this thread, the rest on ``pool``."""
    pending = [pool.submit(task, item) for item in items[1:]]
    task(items[0])
    for future in pending:
        future.result()


def _product_pool(threads: int) -> ThreadPoolExecutor:
    key = (os.getpid(), threads)
    pool = _product_pools.get(key)
    if pool is None:
        pool = _product_pools[key] = ThreadPoolExecutor(threads, thread_name_prefix="quchip-product")
    return pool


class _HermitianPacking:
    """Index maps between a column-stacked density matrix and its packed upper triangle."""

    def __init__(self, dimension: int) -> None:
        rows, cols = np.triu_indices(dimension)
        order = np.lexsort((rows, cols))
        rows, cols = rows[order], cols[order]
        self.dimension, self.size = dimension, rows.size
        self.upper = rows + cols * dimension
        self.index = np.full((dimension, dimension), -1, dtype=np.int64)
        self.index[rows, cols] = np.arange(rows.size)
        lower_rows, lower_cols = np.tril_indices(dimension, -1)
        self.lower = lower_rows + lower_cols * dimension
        self.mirror = self.index[lower_cols, lower_rows]

    def split(self, superoperator: Qobj) -> list[sparse.csr_matrix]:
        """Rows (i ≤ j) of a superoperator as its parts acting on the packed ``x`` and on ``conj(x)``."""
        data = superoperator.data
        matrix = data.as_scipy() if hasattr(data, "as_scipy") else sparse.csr_matrix(data.to_array())
        rows = sparse.csr_matrix(matrix)[self.upper].tocoo()
        i, j = rows.col % self.dimension, rows.col // self.dimension
        upper = i <= j
        shape = (self.size, self.size)
        return [
            sparse.csr_matrix((rows.data[upper], (rows.row[upper], self.index[i[upper], j[upper]])), shape=shape),
            sparse.csr_matrix((rows.data[~upper], (rows.row[~upper], self.index[j[~upper], i[~upper]])), shape=shape),
        ]

    def pack(self, vector: np.ndarray) -> np.ndarray:
        return vector[self.upper]

    def unpack(self, packed: np.ndarray) -> np.ndarray:
        vector = np.empty(self.dimension**2, dtype=complex)
        vector[self.upper] = packed
        vector[self.lower] = np.conj(packed[self.mirror])
        return vector


class _PackedLindbladian(qutip.QobjEvo):
    """Lindblad generator acting on the packed upper triangle of a Hermitian density matrix.

    The rows (i ≤ j) of ``L·vec(ρ)`` read the lower triangle of ``ρ`` as the
    conjugate of the upper one, so each part acts as ``A·x + B·conj(x)`` on the
    packed state ``x``, and one stacked sparse product applies every part. A
    large product splits into row blocks on threads (see ``_product_threads``);
    each row sums in the same order, so the result does not depend on the split.
    """

    def __init__(self, constant: Qobj, dynamic: Sequence[list], packing: _HermitianPacking) -> None:
        super().__init__(qutip.qeye(packing.size))
        superoperators = (constant, *(op for op, _ in dynamic))
        stacked = sparse.hstack([block for op in superoperators for block in packing.split(op)], format="csr")
        threads = _product_threads() if stacked.nnz >= _THREADED_PRODUCT_MIN_NNZ else 1
        self._blocks = _row_blocks(stacked, threads)
        # The calling thread takes the first share of each step while the pool runs the rest.
        self._pool = _product_pool(len(self._blocks) - 1) if len(self._blocks) > 1 else None
        bounds = np.linspace(0, 2 * packing.size, len(self._blocks) + 1).astype(int)
        threaded_scaling = 2 * packing.size * len(dynamic) >= _THREADED_SCALING_MIN_ENTRIES
        self._chunks = list(zip(bounds[:-1], bounds[1:])) if threaded_scaling else [(0, 2 * packing.size)]
        self._coefficients = [coefficient for _, coefficient in dynamic]
        self._size = packing.size
        self._inputs = np.empty(2 * len(superoperators) * packing.size, dtype=complex)

    def _scale_inputs(self, factors: list[complex], chunk: tuple[int, int]) -> None:
        """Write ``c_k·[x, conj(x)]`` over one chunk of every coefficient's input block."""
        low, high = chunk
        span = 2 * self._size
        for k, factor in enumerate(factors, start=1):
            np.multiply(self._inputs[low:high], factor, out=self._inputs[k * span + low:k * span + high])

    def _apply(self, result: np.ndarray, added: np.ndarray | None, scale: complex,
               block: tuple[int, int, sparse.csr_matrix]) -> None:
        low, high, rows = block
        product = rows @ self._inputs
        if scale != 1:
            product *= scale
        if added is not None:
            product += added[low:high]
        result[low:high] = product

    def matmul_data(self, t: Any, state: Any, out: Any = None, scale: complex = 1) -> Any:
        size, inputs = self._size, self._inputs
        x = (state.as_ndarray() if isinstance(state, qutip.data.Dense) else state.to_array()).reshape(-1)
        inputs[:size] = x
        np.conjugate(x, out=inputs[size:2 * size])
        factors = [coefficient(t) for coefficient in self._coefficients]
        added = None
        if out is not None:
            added = (out.as_ndarray() if isinstance(out, qutip.data.Dense) else out.to_array()).reshape(-1)
        result = np.empty(size, dtype=complex)
        if self._pool is None:
            self._scale_inputs(factors, (0, 2 * size))
            self._apply(result, added, scale, self._blocks[0])
        else:
            _share(self._pool, partial(self._scale_inputs, factors), self._chunks)
            _share(self._pool, partial(self._apply, result, added, scale), self._blocks)
        return qutip.data.Dense(result.reshape(-1, 1), copy=False)


class _HermitianMESolver(MESolver):
    """``MESolver`` integrating the packed upper triangle of a Hermitian density matrix.

    A Hermiticity-preserving generator keeps ``ρ`` Hermitian, so the strictly
    lower triangle carries no information: the integrator advances half the
    entries and applies half the sparse rows. Saved states are unpacked, so
    results, expectation values and options behave as for ``MESolver``.
    """

    def __init__(self, generator: Any, constant: Qobj, *, options: dict[str, Any]) -> None:
        dynamic = generator.to_list()
        self._packed = None
        super().__init__(generator, [constant], options=options)
        self._packing = _HermitianPacking(math.isqrt(constant.shape[0]))
        self._packed = _PackedLindbladian(constant, dynamic, self._packing)
        self._integrator = self._get_integrator()

    def _get_integrator(self) -> Any:
        if self._packed is None:
            return super()._get_integrator()
        return self.avail_integrators()[self._options["method"]](self._packed, self.options)

    def _prepare_state(self, state: Qobj) -> Any:
        vector = super()._prepare_state(state).to_array().reshape(-1)
        return qutip.data.Dense(self._packing.pack(vector).reshape(-1, 1), copy=False)

    def _restore_state(self, data: Any, *, copy: bool = True) -> Qobj:
        vector = self._packing.unpack(data.to_array().reshape(-1))
        return super()._restore_state(qutip.data.Dense(vector.reshape(-1, 1), copy=False), copy=False)


def _packs_hermitian_state(state: Any, options: dict[str, Any]) -> bool:
    """Whether a lifted master equation from ``state`` may integrate a packed Hermitian state."""
    method = options.get("method", MESolver.solver_options["method"])
    if not isinstance(method, str) or method not in _PACKED_STATE_METHODS:
        return False
    return state.shape[0] >= _PACKED_STATE_MIN_DIMENSION and (state.isket or (state.isoper and state.isherm))


# A reused stationary state may leave a residual of at most this many machine
# epsilons of the generator's scale, as a direct solve does.
_STATIONARY_REUSE_EPS = 16


def _annihilates(liouvillian: sparse.csr_matrix, state: Any) -> bool:
    """Whether a superoperator maps ``state`` to zero to round-off."""
    size = math.isqrt(liouvillian.shape[0])
    if not isinstance(state, Qobj) or not state.isoper or state.shape != (size, size):
        return False
    vector = np.asarray(qutip.operator_to_vector(state).full(), dtype=complex).reshape(-1)
    scale = float(abs(liouvillian).sum(axis=0).max()) * float(np.linalg.norm(vector))
    return bool(np.linalg.norm(liouvillian @ vector) <= _STATIONARY_REUSE_EPS * np.finfo(float).eps * scale)


def _direct_steady_state(liouvillian: sparse.csr_matrix, dims: Sequence[int]) -> Qobj | None:
    """Solve as ``qutip.steadystate``'s default direct method does, without its option scope.

    QuTiP adds ``w·vec(1)ᵀ`` to the generator's first row, with ``w`` the mean
    magnitude of its non-negligible entries, solves for ``w·e₀`` and keeps the
    Hermitian part. Its solve runs inside an option scope whose entry and exit
    rebuild every data-layer dispatcher, which takes longer than the solve for
    small systems. Returns ``None`` for a generator without entries.
    """
    atol = qutip.settings.core["atol"]
    data = liouvillian.data
    significant = (np.abs(data.real) > atol) | (np.abs(data.imag) > atol)
    if not significant.any():
        return None
    weight = float(np.abs(data[significant]).mean())
    size = liouvillian.shape[0]
    n = math.isqrt(size)
    trace = sparse.csr_matrix(
        (np.full(n, weight, dtype=complex), (np.zeros(n, dtype=int), np.arange(0, size, n + 1))), shape=(size, size),
    )
    target = np.zeros(size, dtype=complex)
    target[0] = weight
    vector = sparse.linalg.spsolve((liouvillian + trace).tocsr(), target)
    rho = vector.reshape(n, n, order="F")
    return Qobj(0.5 * (rho + rho.conj().T), dims=[list(dims), list(dims)], isherm=True)


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
    """Concrete backend that uses QuTiP. ``Operator`` = ``State`` = ``qutip.Qobj``.

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
        return np

    def to_array(self, op: Operator) -> Any:
        if isinstance(op, Qobj):
            return np.asarray(op.full(), dtype=complex)
        if hasattr(op, "to_jax"):
            return np.asarray(op.to_jax(), dtype=complex)
        return np.asarray(op, dtype=complex)

    def overlap(self, a: State, b: State) -> complex:
        value = a.dag() @ b
        if isinstance(value, Qobj):
            return complex(value.full()[0, 0])
        return complex(value)

    def norm(self, state_or_op: State | Operator) -> float:
        r"""Return the native QuTiP norm.

        Parameters
        ----------
        state_or_op : qutip.Qobj
            Ket or bra for the Euclidean norm, or matrix for the trace norm.

        Returns
        -------
        float
            ``Qobj.norm()`` with its default norm selection.
        """
        return float(state_or_op.norm())

    def trace(self, op: Operator) -> complex:
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
        return self._cached_op("destroy", n, qutip.destroy)

    def create(self, n: int) -> Operator:
        return self._cached_op("create", n, qutip.create)

    def number(self, n: int) -> Operator:
        return self._cached_op("number", n, qutip.num)

    def identity(self, n: int) -> Operator:
        return self._cached_op("identity", n, qutip.qeye)

    def from_array(self, data: Any, dims: list[list[int]] | None = None) -> Operator:
        if isinstance(data, Qobj):
            return data if dims is None or data.dims == dims else Qobj(data.data, dims=dims)
        if hasattr(data, "to_jax"):
            data = np.asarray(data.to_jax(), dtype=complex)
        return Qobj(data, dims=dims)

    def to_canonical_operator(self, op: Operator) -> Any:
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
        return self._canonical_to_qobj(canonical)

    def coerce_operator(self, op: Operator) -> Operator:
        if isinstance(op, Qobj):
            return op
        return self.from_array(np.asarray(op, dtype=complex))

    def dag(self, op: Operator) -> Operator:
        return self.coerce_operator(op).dag()

    def eigenenergies(self, op: Operator) -> Any:
        return op.eigenenergies()

    def eigensystem_data(self, op: Operator) -> EigensystemData:
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
        return qutip.expect(op, state)

    def ptrace(self, state: State, keep: int | list[int], dims: list[int]) -> State:
        # QuTiP's Qobj.ptrace needs the correct composite dims. Rebuild when the
        # incoming state carries flat dims (as when assembled outside tensor).
        if isinstance(state, Qobj):
            if state.dims[0] != dims and len(dims) > 1:
                new_dims = [dims, [1] * len(dims)] if state.isket else [dims, dims]
                state = Qobj(state.data, dims=new_dims)
        return state.ptrace(keep)

    def permute_state(self, state: State, dims: Sequence[int], order: Sequence[int]) -> State:
        if not isinstance(state, Qobj):
            return super().permute_state(state, dims, order)
        dims = list(dims)
        if state.dims[0] != dims and len(dims) > 1:
            new_dims = [dims, [1] * len(dims)] if state.isket else [dims, dims]
            state = Qobj(state.data, dims=new_dims)
        return state.permute(list(order))

    def tensor(self, *operators: Operator) -> Operator:
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
        validate_two_body_indices(index_a, index_b, dims)
        local_dims = [dims[index_a], dims[index_b]]
        expected_dim = local_dims[0] * local_dims[1]
        if op_ab.shape[0] != expected_dim:
            raise ValueError(
                f"Two-body operator dimension {op_ab.shape[0]} does not match "
                f"dims[{index_a}]*dims[{index_b}] = {expected_dim}"
            )
        local = op_ab if op_ab.dims == [local_dims, local_dims] else Qobj(op_ab.data, dims=[local_dims, local_dims])
        if index_b == index_a + 1:
            # Keep native tensor layout selection for adjacent, ordered targets.
            return qutip.tensor([local if index == index_a else qutip.qeye(dim)
                                 for index, dim in enumerate(dims) if index != index_b])
        return qutip.expand_operator(local.to("CSR"), list(dims), targets=[int(index_a), int(index_b)])

    # ------------------------------------------------------------------
    # State factories
    # ------------------------------------------------------------------

    def basis(self, n: int, k: int) -> State:
        return qutip.basis(n, k)

    def tensor_states(self, *states: State) -> State:
        return qutip.tensor(list(states))

    def coherent(self, n: int, alpha: complex) -> State:
        return qutip.coherent(n, alpha)

    def state_to_dm(self, state: State) -> State:
        state = self.coerce_state(state)
        return state if not state.isket else qutip.ket2dm(state)

    def is_ket(self, state: State) -> bool:
        return state.isket if isinstance(state, Qobj) else super().is_ket(state)

    def is_native_state(self, state: Any) -> bool:
        return isinstance(state, Qobj)

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
        r"""Resolve QuTiP integration options without changing the input mapping.

        Parameters
        ----------
        options : dict
            Native QuTiP settings, e.g. ``method``, ``atol``, ``rtol``, ``nsteps`` (step
            ceiling), and ``max_step`` (ns). Unspecified step controls use engine hints, and
            explicit settings take priority. For method-specific keys, see
            `QuTiP solver options <https://qutip.readthedocs.io/en/stable/apidoc/solver.html>`_.
            Select the saved-state storage via the simulation's ``states`` argument.
        metadata : dict
            Lowering hints, including ordinary-GHz spectral/carrier bounds.
        tlist : array_like
            Save-time grid in ns, used to estimate integration budgets.

        Returns
        -------
        dict
            Copied options, with the missing integration defaults filled.
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
        max_step = maybe_concrete_scalar(resolved.get("max_step"))
        span = maybe_concrete_scalar(tlist[-1] - tlist[0])
        if ("nsteps" not in options and max_step is not None and max_step > 0
                and span is not None and np.isfinite(span / max_step)):
            resolved["nsteps"] = max(resolved.get("nsteps", 200_000), 2 * math.ceil(span / max_step))
        return resolved

    def _resolve_automatic_solver_options(
        self,
        options: dict[str, Any],
        *,
        user_options: dict[str, Any],
        engine_result: Any,
        solver_name: str,
        tlist: Any,
    ) -> dict[str, Any]:
        """Select QuTiP's diagonal propagator for small constant generators."""
        _ = tlist
        resolved = dict(options)

        adaptive_options = {"atol", "rtol", "nsteps", "max_step"}
        explicit_method = user_options.get("method")
        if explicit_method == "diag":
            incompatible = adaptive_options & user_options.keys()
            if incompatible:
                raise ValueError(f"QuTiP method='diag' does not support options {sorted(incompatible)}.")
            for key in adaptive_options:
                resolved.pop(key, None)
            return resolved

        # Cascade-generated terms can leave a degenerate Liouvillian non-diagonalizable,
        # so this reads slh.H (the resolved model), not the drive-augmented static terms.
        if (
            explicit_method is not None
            or bool(adaptive_options & user_options.keys())
            or engine_result.dynamic_terms
            or engine_result.slh.has_network_hamiltonian
        ):
            return resolved

        dimension = math.prod(engine_result.dims)
        generator_dim = dimension if solver_name == "sesolve" else dimension**2
        limit = _MAX_STATIC_HILBERT_DIM if solver_name == "sesolve" else _MAX_STATIC_LIOUVILLIAN_DIM
        if generator_dim > limit:
            return resolved

        for key in adaptive_options:
            resolved.pop(key, None)

        resolved["method"] = "diag"
        return resolved

    _default_nsteps = staticmethod(default_solver_steps)

    def coerce_state(self, state: State, dims: tuple[int, ...] | None = None) -> State:
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

    def solve_problem(self, problem: Any) -> SolverResult:
        if not problem.stochastic:
            return super().solve_problem(problem)
        if problem.solver not in ("mcsolve", "ssesolve", "smesolve"):
            raise ValueError(f"QuTiP does not provide {problem.solver!r}.")
        options = dict(problem.options)
        if problem.states is not None:
            if {"store_states", "store_final_state"} & options.keys():
                raise ValueError("Conflicting states and native storage options.")
            options.update(store_states=problem.states == "all", store_final_state=problem.states == "final")
        rhs = self.prepare_hamiltonian(problem.engine_result, problem.tlist).rhs
        from copy import deepcopy

        kwargs = dict(e_ops=problem.e_ops, options=options, **problem.run_args)
        if "seeds" in kwargs:
            kwargs["seeds"] = deepcopy(kwargs["seeds"])
        if problem.solver == "mcsolve":
            operators = self._collapse_operators(problem.engine_result)
            if not operators:
                raise ValueError(
                    "mcsolve requires a declared jump channel; use a deterministic solver for closed evolution."
                )
            kwargs["c_ops"] = operators
        else:
            from quchip.engine.monitoring import monitored_operators

            loss, monitored, etas = monitored_operators(problem)
            kwargs["sc_ops"] = [np.sqrt(eta) * op for eta, op in zip(etas, monitored)]
            if problem.solver == "smesolve":
                kwargs["c_ops"] = loss + [np.sqrt(1 - eta) * op for eta, op in zip(etas, monitored)]
                _require_superoperator_memory(
                    rhs, [*kwargs["c_ops"], *kwargs["sc_ops"]], task="QuTiP smesolve's D²×D² Liouvillian",
                    remedy="Reduce the device cutoffs.",
                )
        native = getattr(qutip, problem.solver)(
            rhs, self.coerce_state(problem.initial_state, dims=problem.engine_result.dims), problem.tlist, **kwargs)
        return SolverResult(times=problem.tlist, solver=problem.solver, native=native)

    def sesolve(
        self,
        H: Any,
        psi0: State,
        tlist: Any,
        e_ops: list[Operator] | None = None,
        options: dict[str, Any] | None = None,
    ) -> SolverResult:
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
        rhs = self._coerce_solver_rhs(H)
        _require_superoperator_memory(
            rhs, c_ops or [], task="QuTiP mesolve's D²×D² Liouvillian",
            remedy="Use backend='dynamiqs', whose mesolve applies the Lindblad generator without "
                   "forming it, or reduce the device cutoffs.",
        )
        options = self._runner_options(options)
        lifted = None if options.get("matrix_form") else _lindblad_generator(rhs, c_ops or [], tlist)
        if lifted is None:
            runner = MESolver(rhs, c_ops, options=options)
        else:
            # A superoperator collapse term joins the generator as its constant part.
            generator, constant = lifted
            if _packs_hermitian_state(rho0, options):
                runner = _HermitianMESolver(generator, constant, options=options)
            else:
                runner = MESolver(generator, [constant], options=options)
        result = runner.run(rho0, tlist, e_ops=e_ops)
        return self._wrap_result(result, solver="mesolve", extra_stats=self._solver_stats(runner))

    def _stationary_liouvillian(self, engine_result: Any) -> Qobj:
        """Lower one static engine description to QuTiP's native stationary system."""
        if engine_result.dynamic_terms:
            raise ValueError("Stationary analysis requires a static resolved Hamiltonian.")
        hamiltonian = self.prepare_hamiltonian(engine_result).rhs
        collapse_ops = self._collapse_operators(engine_result)
        _require_superoperator_memory(
            hamiltonian, collapse_ops, task="The D²×D² stationary Liouvillian",
            remedy="Reduce the device cutoffs.",
        )
        return qutip.liouvillian(hamiltonian, collapse_ops)

    @staticmethod
    def _scipy_liouvillian(liouvillian: Qobj) -> Any:
        """Expose a QuTiP Liouvillian as SciPy CSR without densifying it."""
        if hasattr(liouvillian.data, "as_scipy"):
            return liouvillian.data.as_scipy().tocsr()
        return sparse.csr_matrix(liouvillian.data.to_array())

    def steadystate(
        self, problem: Any, *, prepared: PreparedStationary | None = None, guess: State | None = None,
    ) -> SteadyStateSolverResult:
        r"""Solve a static Lindblad generator with :func:`qutip.steadystate`.

        Parameters
        ----------
        problem : SteadyStateProblem
            Captured static model, observables, and stationary solver options.
        prepared : PreparedStationary or None, default None
            Matching prepared generator. ``None`` builds it.
        guess : Qobj or None, default None
            Stationary state of a related generator. If this generator
            annihilates the state to round-off, the method returns it without
            solving and the stats record ``guess_reused``.

        Returns
        -------
        SteadyStateSolverResult
            Stationary density matrix and convergence diagnostics.

        See Also
        --------
        quchip.backend.protocol.Backend.steadystate

        Notes
        -----
        ``problem.options`` accepts native QuTiP stationary-solver keywords, e.g.
        ``method`` (default ``"direct"``) and ``solver`` (default None). quchip
        consumes ``rank_tolerance`` (singular-value cutoff, default automatic) and
        ``diagnostic_max_dimension`` (default 16). Above that Hilbert dimension, the
        nullity is not computed, and ``None`` means unchecked, not a unique state.
        """
        liouvillian = self.prepare_stationary(problem.engine_result, prepared=prepared).liouvillian
        options = dict(problem.options)
        rank_tolerance = options.pop("rank_tolerance", None)
        diagnostic_max_dimension = int(options.pop("diagnostic_max_dimension", 16))
        if diagnostic_max_dimension < 0:
            raise ValueError("diagnostic_max_dimension must be non-negative.")
        method = options.pop("method", "direct")
        solver = options.pop("solver", None)
        sparse_liouvillian = self._scipy_liouvillian(liouvillian)
        reused = guess is not None and _annihilates(sparse_liouvillian, guess)
        if guess is not None and reused:
            state = guess
        else:
            direct = method == "direct" and solver is None and not options
            state = _direct_steady_state(sparse_liouvillian, problem.engine_result.dims) if direct else None
            if state is None:
                state = qutip.steadystate(liouvillian, method=method, solver=solver, **options)

        state_vector = np.asarray(qutip.operator_to_vector(state).full(), dtype=complex).reshape(-1)
        residual = float(np.linalg.norm(sparse_liouvillian @ state_vector))
        dimension = state.shape[0]
        nullity = None
        if dimension <= diagnostic_max_dimension:
            dense_liouvillian = np.asarray(sparse_liouvillian.toarray(), dtype=complex)
            singular_values = np.linalg.svd(dense_liouvillian, compute_uv=False)
            if rank_tolerance is None:
                scale = singular_values[0] if singular_values.size else 0.0
                rank_tolerance = max(dense_liouvillian.shape) * np.finfo(float).eps * scale
            nullity = int(np.count_nonzero(singular_values <= rank_tolerance))
        expectations = None
        if isinstance(problem.e_ops, list):
            expectations = [qutip.expect(operator, state) for operator in problem.e_ops]

        return SteadyStateSolverResult(
            state=state,
            expect=expectations,
            stats={
                "method": method,
                "solver": solver,
                "uniqueness_checked": nullity is not None,
                "diagnostic_max_dimension": diagnostic_max_dimension,
                "guess_reused": reused,
            },
            residual=residual,
            nullity=nullity,
            _condition_number=(
                DeferredValue(partial(self._stationary_condition_number, problem.engine_result))
                if dimension <= diagnostic_max_dimension else None
            ),
        )

    def _stationary_condition_number(self, engine_result: Any) -> float:
        liouvillian = self._scipy_liouvillian(self._stationary_liouvillian(engine_result)).toarray()
        return float(stationary_condition_number(liouvillian, math.prod(engine_result.dims), xp=np))

    def linear_response(self, problem: Any) -> LinearResponseSolverResult:
        return linear_response(problem, xp=np)

    def stationary_resolvent(
        self,
        engine_result: Any,
        sources: tuple[tuple[str, Any], ...],
        observables: tuple[tuple[str, Any], ...],
        frequencies: Any,
        *,
        prepared: PreparedStationary | None = None,
    ) -> dict[tuple[str, str], Any]:
        liouvillian = self.prepare_stationary(engine_result, prepared=prepared).liouvillian
        matrix = self._scipy_liouvillian(liouvillian)
        native_sources = [(label, self.from_canonical_operator(operator)) for label, operator in sources]
        dims = native_sources[0][1].dims
        dimension = math.prod(engine_result.dims)
        targets = np.stack(
            [
                np.asarray(qutip.operator_to_vector(operator).full(), dtype=complex).reshape(-1)
                for _, operator in native_sources
            ],
            axis=1,
        )
        targets[-1, :] = 0.0
        size = matrix.shape[0]
        keep = sparse.diags([1.0] * (size - 1) + [0.0], format="csr", dtype=complex)
        trace_row = sparse.csr_matrix(
            (np.ones(dimension, dtype=complex), (np.full(dimension, size - 1), np.arange(0, size, dimension + 1))),
            shape=(size, size),
        )
        base = keep @ matrix + trace_row
        native_observables = tuple(
            (label, self.from_canonical_operator(operator))
            for label, operator in observables
        )
        values: dict[tuple[str, str], list[Any]] = {
            (source_label, label): [] for source_label, _ in sources for label, _ in observables
        }

        for frequency in np.atleast_1d(np.asarray(frequencies, dtype=float)):
            constrained = (base + 1j * (2.0 * np.pi) * frequency * keep).tocsc()
            solutions = sparse.linalg.splu(constrained).solve(targets)
            for column, (source_label, _) in enumerate(native_sources):
                response = Qobj(
                    solutions[:, column].reshape((dimension, dimension), order="F"),
                    dims=dims,
                )
                for label, observable in native_observables:
                    values[(source_label, label)].append((observable * response).tr())

        return {key: np.asarray(items) for key, items in values.items()}

    def stationary_propagate(
        self,
        engine_result: Any,
        initial: Any,
        observables: tuple[tuple[str, Any], ...],
        times: Any,
        *,
        prepared: PreparedStationary | None = None,
    ) -> dict[str, Any]:
        liouvillian = self.prepare_stationary(engine_result, prepared=prepared).liouvillian
        matrix = self._scipy_liouvillian(liouvillian)
        initial_operator = self.from_canonical_operator(initial)
        initial_vector = np.asarray(
            qutip.operator_to_vector(initial_operator).full(), dtype=complex
        ).reshape(-1)
        dimension = initial_operator.shape[0]
        native_observables = tuple(
            (label, self.from_canonical_operator(operator))
            for label, operator in observables
        )
        values: dict[str, list[Any]] = {label: [] for label, _ in observables}

        for time in np.atleast_1d(np.asarray(times, dtype=float)):
            evolved_vector = sparse.linalg.expm_multiply(matrix * time, initial_vector)
            evolved = Qobj(
                evolved_vector.reshape((dimension, dimension), order="F"),
                dims=initial_operator.dims,
            )
            for label, observable in native_observables:
                values[label].append((observable * evolved).tr())

        return {label: np.asarray(items) for label, items in values.items()}

    @staticmethod
    def _runner_options(options: dict[str, Any] | None) -> dict[str, Any]:
        """Trust the already-merged options; only fill defaults when ``None``.

        The single option-merge boundary (:meth:`resolve_solver_options`,
        reached via :meth:`_merge_options`) has already applied
        ``_DEFAULT_SOLVE_OPTIONS``, so a supplied dict is forwarded as a defensive copy —
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
        r"""Solve large, structurally heterogeneous problem lists through loky workers.

        Parameters
        ----------
        problems
            Problems that cannot be merged into one structural batch.
        progress
            Display solver progress if the backend supports it.

        Returns
        -------
        list[SolverResult] | None
            Backend results in input order, or ``None`` to keep the engine's
            structural-group dispatch.

        See Also
        --------
        quchip.backend.protocol.Backend.parallel_solve_problems
        """
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
        return self._batched_solve(problems, solver_fn="sesolve", n_jobs=n_jobs, progress=progress)

    def batched_mesolve(
        self,
        problems: list[dict[str, Any]],
        *,
        n_jobs: int = -1,
        progress: bool = True,
    ) -> list[SolverResult]:
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
        r"""Convert a :class:`EngineResult` into a ``Qobj`` or ``QobjEvo``.

        Each dynamic coefficient is band-normalized. Every carrier stays
        analytic, and only its slow, carrier-free envelope is sampled on
        *tlist*, locally densified around each window edge. This avoids the
        cubic-spline error of a pre-sampled full ``envelope·carrier`` product,
        even for resonant carriers in the lab frame.

        Parameters
        ----------
        description : EngineResult
            Captured engine Hamiltonian and channels in solver units.
        tlist : array_like
            Requested save times in ns, used by native time-dependent lowering.

        Returns
        -------
        PreparedHamiltonian
            Native right-hand side and numerical integration hints.

        See Also
        --------
        quchip.backend.protocol.Backend.prepare_hamiltonian
        """
        # The envelope sampling and edge densification are implemented in
        # `_assemble_qobjevo` / `_augmented_sample_grid`.
        static_rhs = self._sum_terms(description.static_terms, self._canonical_to_qobj)
        metadata = dict(description.metadata)

        if not description.dynamic_terms:
            if static_rhs is None:
                raise ValueError("EngineResult must contain at least one static or dynamic term.")
            return PreparedHamiltonian(rhs=static_rhs, metadata=metadata)

        sample_tlist = self._resolve_envelope_sample_tlist(tlist)
        op_signal_pairs = (
            (self._canonical_to_qobj(operator), signal)
            for operator, signal in self._scalar_dynamic_terms(description)
        )
        rhs = _assemble_qobjevo(static_rhs, op_signal_pairs, sample_tlist)
        return PreparedHamiltonian(rhs=rhs, metadata=metadata)

    def prepare_batch(self, batch: Any) -> DeferredBatch:
        r"""Build a deferred-construction batch whose workers build each element's ``QobjEvo``.

        Only the slow, carrier-free envelope is sampled on the user grid,
        locally densified around each window edge, and the carriers stay
        analytic.

        Parameters
        ----------
        batch : SolveBatch
            Batch of captured problems with compatible save grids.

        Returns
        -------
        PreparedBatch
            Eager, native-vectorized, or deferred representation selected by this backend.

        See Also
        --------
        quchip.backend.protocol.Backend.prepare_batch
        """
        # Each unique `CanonicalOperator` is converted once and shared across
        # elements. The envelope sampling is implemented in `_assemble_qobjevo`.
        # The final `QobjEvo` assembly occurs in `solve_batch`, so loky workers
        # can build and solve each point of larger concrete sweeps.
        cached_qobj = self._make_op_cache()
        engine_results = tuple(problem.engine_result for problem in batch.problems)
        static_cache: dict[tuple[int, ...], Qobj | None] = {}
        static_rhs: list[Qobj | None] = []
        for result in engine_results:
            key = result._static_term_ids
            if key not in static_cache:
                static_cache[key] = self._sum_terms(result.static_terms, cached_qobj)
            static_rhs.append(static_cache[key])
        dynamic_qobjs = tuple(
            tuple(cached_qobj(term.operator) for term in result.dynamic_terms)
            for result in engine_results
        )
        sample_tlist: Any = None
        if any(dynamic_qobjs):
            sample_tlist = self._resolve_envelope_sample_tlist(batch.tlist)

        shared = _QuTiPBatchShared(
            static_rhs=tuple(static_rhs),
            dynamic_qobjs=dynamic_qobjs,
            sample_tlist=sample_tlist,
            dynamic_signals=tuple(
                tuple(term.time_dependence for term in result.dynamic_terms)
                for result in engine_results
            ),
        )
        return DeferredBatch(
            shared=shared,
            batch_size=batch.batch_size,
            metadata=dict(engine_results[0].metadata),
            tlist=batch.tlist,
        )

    def solve_batch(self, batch: Any, *, progress: bool = True) -> list[SolverResult]:
        r"""Solve a :class:`SolveBatch`, with the ``QobjEvo`` for each element built in loky workers.

        Parameters
        ----------
        batch : SolveBatch
            Captured problems and batch parameter values.
        progress : bool, default True
            Request a batch progress display.

        Returns
        -------
        list of SolverResult
            Results in batch order.

        See Also
        --------
        quchip.backend.protocol.Backend.solve_batch
        """
        if any(problem.stochastic for problem in batch.problems):
            return self._parallel_map(task=self.solve_problem, items=list(batch.problems),
                                      n_jobs=1, progress=progress, desc="quchip trajectories")
        if batch.batch_size == 0:
            return []

        prepared = self.prepare_batch(batch)
        tlist_arr = self.array_module.asarray(batch.tlist, dtype=float)

        shared = prepared.shared
        if not isinstance(shared, _QuTiPBatchShared):
            raise RuntimeError(
                "QuTiPBackend.solve_batch requires DeferredBatch.shared to be a "
                f"_QuTiPBatchShared instance, got {type(shared).__name__}."
            )
        sample_tlist = shared.sample_tlist

        tasks: list[_QuTiPBatchTask] = []
        for index, problem in enumerate(batch.problems):
            engine_result = problem.engine_result
            c_ops = self._collapse_operators(engine_result)
            solver_name = problem.solver_name(self)
            opts = self._resolve_problem_options(
                problem,
                metadata=engine_result.metadata,
                tlist=tlist_arr,
                solver_name=solver_name,
            )
            tasks.append(
                _QuTiPBatchTask(
                    static_rhs=shared.static_rhs[index],
                    dynamic_qobjs=shared.dynamic_qobjs[index],
                    dynamic_signals=shared.dynamic_signals[index],
                    initial_state=self.coerce_state(
                        problem.initial_state,
                        dims=problem.engine_result.dims,
                    ),
                    c_ops=tuple(c_ops),
                    solver_name=solver_name,
                    options=dict(opts),
                    e_ops=(tuple(problem.e_ops) if isinstance(problem.e_ops, list) else None),
                )
            )

        def build_and_solve(task: _QuTiPBatchTask) -> SolverResult:
            op_signal_pairs = (
                (operator, signal.signal)
                for operator, signal in zip(task.dynamic_qobjs, task.dynamic_signals)
            )
            rhs = _assemble_qobjevo(task.static_rhs, op_signal_pairs, sample_tlist)
            kwargs = self._element_solver_kwargs(
                task.solver_name,
                rhs,
                task.initial_state,
                tlist_arr,
                e_ops=list(task.e_ops) if task.e_ops is not None else None,
                c_ops=list(task.c_ops),
                options=task.options,
            )
            return getattr(self, task.solver_name)(**kwargs)

        reference_solver = tasks[0].solver_name

        return self._parallel_map(
            task=build_and_solve,
            items=tasks,
            n_jobs=-1,
            progress=progress,
            desc=f"Sweep ({reference_solver})",
        )

    # ------------------------------------------------------------------
    # Internal: per-element RHS construction
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_envelope_sample_tlist(tlist: Any) -> Any:
        """Return the base sample grid for interpolating carrier-free slow envelopes; ``None`` otherwise.

        Carriers are kept analytic (see :func:`_assemble_qobjevo`), so no
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
        """Start the reusable solver pool.

        Parameters
        ----------
        n_jobs : int, default -1
            Worker count. ``-1`` selects all CPUs.
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
        loky executor. Worker-pool failures may use sequential execution;
        failures raised by an individual solve retain their point index and
        propagate without retrying the numerical work.
        """
        from tqdm import tqdm

        def indexed_task(entry: tuple[int, Any]) -> SolverResult:
            index, item = entry
            try:
                return task(item)
            except Exception as exc:
                raise BatchSolveError(index, f"{type(exc).__name__}: {exc}") from exc

        def sequential() -> list[SolverResult]:
            iterator = tqdm(enumerate(items), total=len(items), desc=desc) if progress else enumerate(items)
            return [indexed_task(entry) for entry in iterator]

        if n_jobs == 1 or len(items) < self._PARALLEL_MIN_BATCH:
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
            mapped = executor.map(indexed_task, enumerate(items))
            if progress:
                return list(tqdm(mapped, total=len(items), desc=desc))
            return list(mapped)
        except BatchSolveError:
            raise
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
        """Reconstruct a ``Qobj`` from a ``CanonicalOperator`` (dense or sparse).

        A dense payload that is mostly exact zeros, such as a band of a captured
        reduced-model matrix, is stored as CSR so the superoperators QuTiP builds
        from it stay sparse.
        """
        dims = [list(canonical.dims), list(canonical.dims)]
        if canonical.layout == "dense":
            values = np.asarray(canonical.values, dtype=complex)
            if np.count_nonzero(values) > _CSR_MAX_FILL * values.size:
                return Qobj(values, dims=dims, dtype="Dense")
            return Qobj(sparse.csr_matrix(values), dims=dims, dtype="CSR")
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

        final_state = states[-1] if states else getattr(qutip_result, "final_state", None)

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
        """Capture effective solver options and the step count when exposed."""
        stats: dict[str, Any] = {"options": dict(solver_runner.options)}
        nsteps = QuTiPBackend._extract_nsteps(solver_runner)
        if nsteps is not None:
            stats["nsteps"] = nsteps
        return stats

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
