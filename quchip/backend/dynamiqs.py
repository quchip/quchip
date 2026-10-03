"""Dynamiqs backend — JAX-native solver for differentiable, vmappable simulation.

This backend wraps `dynamiqs <https://github.com/dynamiqs/dynamiqs>`_ (which
builds on `JAX <https://github.com/google/jax>`_) to provide:

* **Full JAX traceability.** Every operator, state, and signal program
  evaluates inside JAX, so gradients flow through frame resolution, carrier
  phases, envelope parameters, crosstalk mixing, and dissipator strengths.
* **Native batched solves.** A typed :class:`~quchip.engine.ir.SolveBatch`
  is stacked along a batch axis and integrated with native Dynamiqs integrators
  under ``vmap`` in :meth:`solve_batch`.
  Structurally heterogeneous batches fail loudly — no silent sequential
  fallback — so the caller can regroup them via
  :func:`quchip.engine.solve_many`.

References
----------
* Guilmin et al. — *dynamiqs: an open-source Python library for GPU-accelerated
  and differentiable simulation of quantum systems* (2024)
* Bradbury et al. — *JAX: composable transformations of Python+NumPy programs*
  (2018)
* Lindblad — Commun. Math. Phys. 48, 119 (1976)
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from functools import lru_cache, partial, reduce
from typing import Any, Sequence

import dynamiqs as dq
import equinox as eqx
import numpy as np
from dynamiqs.qarrays.layout import get_layout
from dynamiqs.qarrays.qarray import QArray
from dynamiqs.qarrays.sparsedia_dataarray import SparseDIADataArray
from dynamiqs.time_qarray import SummedTimeQArray

# x64 is enabled at the package boundary in ``quchip/__init__.py``.

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.scipy.linalg as jsp_linalg  # noqa: E402
import jax.tree_util as jtu  # noqa: E402

from quchip.backend._memory import require_memory
from quchip.backend._response import linear_response, stationary_condition_number
from quchip.utils.jax_utils import contains_tracer
from quchip.utils.values import DeferredValue
from quchip.backend._dims import (  # noqa: E402
    _embed_array,
    default_solver_steps,
    normalize_dims_from_list,
    validate_two_body_indices,
)
from quchip.backend.containers import (  # noqa: E402
    DeferredBatch,
    EigensystemData,
    PreparedHamiltonian,
    PreparedStationary,
    LinearResponseSolverResult,
    SolverResult,
    SteadyStateSolverResult,
)
from quchip.backend.protocol import Backend, Operator, State, _is_unit  # noqa: E402
from quchip.engine.ir import (  # noqa: E402
    Add,
    ScalarModulation,
    _aggregate_batch_metadata,
    evaluate_signal_program,
    signal_window_bounds,
)

# Default integrator tolerances; pulse-parameter gradients converge at these,
# not at the native Tsit5 defaults (rtol = atol = 1e-6).
_DEFAULT_RTOL = 1e-9
_DEFAULT_ATOL = 1e-11


def _shared_operator_slots(results: Sequence[Any]) -> list[list[int]]:
    """Group dynamic slots whose operator is one object in every result.

    Terms sharing an operator form one modulated term whose signal is their
    sum, so each step applies the operator once however many pulses and
    crosstalk paths drive it.
    """
    groups: dict[tuple[int, ...], list[int]] = {}
    for slot in range(len(results[0].dynamic_terms)):
        groups.setdefault(tuple(id(result.dynamic_terms[slot].operator) for result in results), []).append(slot)
    return list(groups.values())


def _summed_modulation(modulations: Sequence[Any]) -> Any:
    """One scalar modulation equal to the sum of *modulations*."""
    if len(modulations) == 1:
        return modulations[0]
    return ScalarModulation(signal=Add(tuple(modulation.signal for modulation in modulations)))


@lru_cache(maxsize=None)
def _dia_structure(dims: tuple[int, ...], offsets: tuple[int, ...]) -> Any:
    """Pytree structure of an unbatched sparse-DIA qarray whose one leaf is its diagonals.

    The instances are assembled field by field: running their constructors,
    even abstractly, traces dynamiqs' per-diagonal checks for every offset set.
    """
    data = object.__new__(SparseDIADataArray)
    object.__setattr__(data, "offsets", offsets)
    object.__setattr__(data, "diags", 0)
    qarray = object.__new__(QArray)
    for name, value in (("dims", dims), ("vectorized", False), ("data", data)):
        object.__setattr__(qarray, name, value)
    return jtu.tree_structure(qarray)


def _dia_qarray(dims: tuple[int, ...], offsets: tuple[int, ...], diags: Any) -> QArray:
    """Build an operator in sparse-DIA layout; *diags* may be traced, *offsets* are static.

    Concrete diagonals are checked on the host and wrapped directly. Dynamiqs
    checks each diagonal with a device computation that compiles once per
    shape, which dominates building a chip's operators outside JIT; inside a
    trace the same check stages a host callback per diagonal. Traced
    diagonals are instead zeroed outside the matrix bounds.
    """
    dims = tuple(int(dim) for dim in dims)
    offsets = tuple(int(offset) for offset in offsets)
    if np.ndim(diags) != 2:
        return QArray(dims, False, SparseDIADataArray(offsets, jnp.asarray(diags, dtype=jnp.complex128)))
    if contains_tracer(diags):
        inside = np.ones(np.shape(diags), dtype=bool)
        for row, offset in enumerate(offsets):
            if offset > 0:
                inside[row, :offset] = False
            elif offset < 0:
                inside[row, offset:] = False
        values = jnp.where(inside, jnp.asarray(diags, dtype=jnp.complex128), 0.0)
        return jtu.tree_unflatten(_dia_structure(dims, offsets), [values])
    host = np.asarray(diags, dtype=complex)
    for row, offset in zip(host, offsets):
        if np.any(row[:offset] if offset >= 0 else row[offset:]):
            raise ValueError("Sparse-DIA diagonals must be zero outside the matrix bounds.")
    return jtu.tree_unflatten(_dia_structure(dims, offsets), [jnp.asarray(host)])


def _complex_payload(values: Any) -> Any:
    """Complex JAX array of *values*, converted on the host when they are concrete."""
    return jnp.asarray(values if contains_tracer(values) else np.asarray(values, dtype=complex), dtype=jnp.complex128)


def _concrete_coefficient(value: Any) -> np.ndarray | None:
    """A concrete numeric scalar as a 0-d NumPy array (real stays real), else ``None``."""
    if contains_tracer(value) or np.ndim(value) != 0:
        return None
    array = np.asarray(value)
    return array if array.dtype.kind in "iufc" else None


def _dia_parts(op: Any) -> tuple[tuple[int, ...], Any] | None:
    """Offsets and diagonals, concrete or traced, of an unbatched sparse-DIA qarray, else ``None``."""
    if isinstance(op, QArray) and op.layout is dq.dia and op.ndim == 2:
        return tuple(int(offset) for offset in op.data.offsets), op.data.diags
    return None


def _kron_dia_traced(left: tuple[tuple[int, ...], Any],
                     right: tuple[tuple[int, ...], Any]) -> tuple[tuple[int, ...], Any]:
    """:func:`_kron_dia` for diagonals that may be traced."""
    (left_offsets, left_diags), (right_offsets, right_diags) = left, right
    offsets = (np.asarray(left_offsets)[:, None] * right_diags.shape[-1] + np.asarray(right_offsets)).ravel()
    unique, inverse = np.unique(offsets, return_inverse=True)
    diags = jnp.zeros((unique.size, left_diags.shape[-1] * right_diags.shape[-1]), dtype=jnp.complex128)
    return tuple(int(offset) for offset in unique), diags.at[inverse].add(jnp.kron(left_diags, right_diags))


def _matmul_dia_traced(left: tuple[tuple[int, ...], Any],
                       right: tuple[tuple[int, ...], Any]) -> tuple[tuple[int, ...], Any]:
    """Product of two sparse-DIA operators whose diagonals may be traced, on dynamiqs' output offsets.

    Entry ``k`` of output diagonal ``p + q`` gathers ``left[p][k - q]·right[q][k]``
    for every pair of input offsets in one vectorized step.
    """
    (left_offsets, left_diags), (right_offsets, right_diags) = left, right
    n = left_diags.shape[-1]
    pairs = [(row, column, lo + ro) for row, lo in enumerate(left_offsets)
             for column, ro in enumerate(right_offsets) if abs(lo + ro) < n]
    if not pairs:
        return (), jnp.zeros((0, n), dtype=jnp.complex128)
    rows, columns, offsets = (np.asarray(values) for values in zip(*pairs))
    unique, inverse = np.unique(offsets, return_inverse=True)
    shifted = np.arange(n) - np.asarray(right_offsets)[columns][:, None]
    inside = (shifted >= 0) & (shifted < n)
    products = jnp.where(inside, left_diags[rows[:, None], np.clip(shifted, 0, n - 1)] * right_diags[columns], 0.0)
    diags = jnp.zeros((unique.size, n), dtype=jnp.complex128).at[inverse.ravel()].add(products)
    return tuple(int(offset) for offset in unique), diags


def _dense_linear_combination(terms: Sequence[tuple[Any, Any]], dims: tuple[int, ...]) -> QArray:
    """Dense ``Σ cᵢ·Aᵢ`` of unbatched qarrays, summed on the host when every term is concrete."""
    host_terms = []
    for raw, op in terms:
        coefficient, matrix = _concrete_coefficient(raw), _host_matrix(op)
        if coefficient is None or matrix is None:
            traced = sum(term.to_jax() if _is_unit(scale) else scale * term.to_jax() for scale, term in terms)
            return dq.asqarray(traced, dims=dims)
        host_terms.append(matrix if _is_unit(raw) else coefficient * matrix)
    return dq.asqarray(jnp.asarray(sum(host_terms)), dims=dims)


def _plain_operators(operators: Sequence[Any]) -> bool:
    """True when every operator is an unbatched, unvectorized qarray on one Hilbert space shape."""
    return all(isinstance(op, QArray) and op.ndim == 2 and not op.vectorized for op in operators)


def _host_dia(op: Any) -> tuple[tuple[int, ...], np.ndarray] | None:
    """Offsets and host diagonals of a concrete, unbatched sparse-DIA qarray, else ``None``."""
    if isinstance(op, QArray) and op.layout is dq.dia and op.ndim == 2 and not contains_tracer(op.data.diags):
        return tuple(int(offset) for offset in op.data.offsets), np.asarray(op.data.diags)
    return None


def _host_dense(op: Any) -> np.ndarray | None:
    """Host matrix of a concrete, unbatched dense qarray, else ``None``."""
    if isinstance(op, QArray) and op.layout is dq.dense and op.ndim == 2 and not contains_tracer(op.data):
        return np.asarray(op.to_jax())
    return None


def _dia_to_dense(offsets: Sequence[int], diags: np.ndarray) -> np.ndarray:
    """Dense host matrix of host sparse-DIA diagonals (column-aligned, as dynamiqs stores them)."""
    n = diags.shape[-1]
    dense = np.zeros((n, n), dtype=complex)
    cols = np.arange(n)
    for offset, row in zip(offsets, diags):
        valid = (cols - offset >= 0) & (cols - offset < n)
        dense[cols[valid] - offset, cols[valid]] += row[valid]
    return dense


def _host_matrix(op: Any) -> np.ndarray | None:
    """Dense host matrix of a concrete, unbatched qarray, else ``None``."""
    sparse = _host_dia(op)
    return _dia_to_dense(*sparse) if sparse is not None else _host_dense(op)


def _dense_to_dia(matrix: np.ndarray) -> tuple[tuple[int, ...], np.ndarray]:
    """Offsets and column-aligned diagonals holding a host matrix's nonzero entries."""
    rows, columns = np.nonzero(matrix)
    offsets = np.unique(columns - rows) if rows.size else np.zeros(1, dtype=int)
    n = matrix.shape[-1]
    cols = np.arange(n)
    diags = np.zeros((offsets.size, n), dtype=complex)
    for row, offset in enumerate(offsets):
        valid = (cols - offset >= 0) & (cols - offset < n)
        diags[row, valid] = matrix[cols[valid] - offset, cols[valid]]
    return tuple(int(offset) for offset in offsets), diags


def _kron_dia(left: tuple[tuple[int, ...], np.ndarray],
              right: tuple[tuple[int, ...], np.ndarray]) -> tuple[tuple[int, ...], np.ndarray]:
    """Kronecker product of two host sparse-DIA operators, adding diagonals that share an offset."""
    (left_offsets, left_diags), (right_offsets, right_diags) = left, right
    offsets = (np.asarray(left_offsets)[:, None] * right_diags.shape[-1] + np.asarray(right_offsets)).ravel()
    unique, inverse = np.unique(offsets, return_inverse=True)
    diags = np.zeros((unique.size, left_diags.shape[-1] * right_diags.shape[-1]), dtype=complex)
    np.add.at(diags, inverse, np.kron(left_diags, right_diags))
    return tuple(int(offset) for offset in unique), diags


def _layout_operator(n: int, offset: int, diagonal: np.ndarray) -> QArray:
    """One-diagonal operator built on the host in the active dynamiqs layout."""
    diags = np.asarray(diagonal, dtype=complex)[None, :]
    if get_layout() is dq.dia:
        return _dia_qarray((n,), (offset,), diags)
    return dq.asqarray(jnp.asarray(_dia_to_dense((offset,), diags)), dims=(n,))


def _signal_discontinuities(signal: Any) -> Any:
    """Flatten absolute pulse edges across native batch points for Diffrax."""
    edges = [jnp.ravel(edge) for bounds in signal_window_bounds(signal) for edge in bounds]
    return jnp.concatenate(edges) if edges else None


class _SignalCallable(eqx.Module):
    """Equinox wrapper: evaluates a signal-program AST under JAX tracing.

    Registering via ``eqx.Module`` makes the signal a pytree, so dynamiqs
    can vmap across batched parameters inside the carrier/envelope tree.
    """

    signal: Any

    def __call__(self, t: float) -> Any:
        return jnp.asarray(evaluate_signal_program(self.signal, t, xp=jnp))


class _SummedHamiltonian(eqx.Module):
    """``H(t) = H₀ + Σ_k c_k(t)·A_k`` assembled as one qarray per call.

    A sum of dynamiqs modulated terms scales, adds and validates one qarray
    per term at every right-hand-side evaluation. Here the diagonals (or
    dense blocks) of all terms are added into one array whose layout was
    validated once, when the Hamiltonian was assembled.
    """

    treedef: Any = eqx.field(static=True)
    rows: tuple[tuple[int, ...], ...] | None = eqx.field(static=True)
    base: Any
    terms: tuple[Any, ...]
    signals: tuple[_SignalCallable, ...]

    def __call__(self, t: float) -> Any:
        data = self.base
        for index, (term, signal) in enumerate(zip(self.terms, self.signals)):
            scaled = signal(t) * term
            if self.rows is None:
                data = data + scaled
            else:
                data = data.at[np.asarray(self.rows[index], dtype=int)].add(scaled)
        return jtu.tree_unflatten(self.treedef, [data])


def _summed_hamiltonian(static_ops: Sequence[Any], static_coeffs: Sequence[Any], dyn_ops: Sequence[Any],
                        dyn_signals: Sequence[Any]) -> Any:
    """Return a time-callable sum of unbatched terms, or ``None`` when an operator or signal is batched."""
    operators = [*static_ops, *dyn_ops]
    if any(op.ndim != 2 for op in operators):
        return None
    signals = tuple(_SignalCallable(signal) for signal in dyn_signals)
    if any(jax.eval_shape(signal, 0.0).shape for signal in signals):
        return None
    if all(op.layout is dq.dia for op in operators):
        offsets = tuple(sorted({int(offset) for op in operators for offset in op.data.offsets}))
        index = {offset: row for row, offset in enumerate(offsets)}
        n = operators[0].shape[-1]
        base = jnp.zeros((len(offsets), n), dtype=jnp.complex128)
        for op, coeff in zip(static_ops, static_coeffs):
            static_rows = np.asarray([index[int(o)] for o in op.data.offsets], dtype=int)
            base = base.at[static_rows].add(coeff * op.data.diags)
        rows: tuple[tuple[int, ...], ...] | None = tuple(
            tuple(index[int(o)] for o in op.data.offsets) for op in dyn_ops
        )
        terms = tuple(jnp.asarray(op.data.diags, dtype=jnp.complex128) for op in dyn_ops)
        template = _dia_qarray(operators[0].dims, offsets, base)
    else:
        n = operators[0].shape[-1]
        base = jnp.zeros((n, n), dtype=jnp.complex128)
        for op, coeff in zip(static_ops, static_coeffs):
            base = base + coeff * op.to_jax()
        rows = None
        terms = tuple(jnp.asarray(op.to_jax(), dtype=jnp.complex128) for op in dyn_ops)
        template = dq.asqarray(base, dims=operators[0].dims)
    leaves, treedef = jtu.tree_flatten(template)
    if len(leaves) != 1:
        return None
    hamiltonian = _SummedHamiltonian(treedef, rows, base, terms, signals)
    edges = [edge for edge in map(_signal_discontinuities, dyn_signals) if edge is not None]
    return dq.timecallable(hamiltonian, discontinuity_ts=jnp.concatenate(edges) if edges else None)


@dataclass(frozen=True)
class _DynamiqsBatchHamiltonian:
    """Stable native leaves used to assemble one vmapped RHS inside JIT."""

    static_operators: tuple[Any, ...]
    static_coefficients: tuple[Any, ...]
    dynamic_operators: tuple[Any, ...]
    dynamic_signals: tuple[Any, ...]


class DynamiqsBackend(Backend):
    """Concrete backend backed by dynamiqs + JAX. Operators/states are ``QArray``.

    Example
    -------
    >>> from quchip.backend.dynamiqs import DynamiqsBackend
    >>> backend = DynamiqsBackend()
    >>> a = backend.destroy(3)
    >>> n = backend.matmul(backend.dag(a), a)
    >>> float(backend.to_array(n)[2, 2].real)  # doctest: +SKIP
    2.0
    """

    # ------------------------------------------------------------------
    # Scalar / array surface
    # ------------------------------------------------------------------

    @property
    def array_module(self) -> Any:
        return jnp

    def to_array(self, op: Operator) -> Any:
        host = _host_dia(op)
        if host is not None:
            return jnp.asarray(_dia_to_dense(*host))
        if hasattr(op, "to_jax"):
            return jnp.asarray(op.to_jax(), dtype=jnp.complex128)
        if hasattr(op, "full"):
            return jnp.asarray(op.full(), dtype=jnp.complex128)
        return jnp.asarray(op, dtype=jnp.complex128)

    def overlap(self, a: State, b: State) -> complex:
        return dq.braket(a, b)

    def norm(self, state_or_op: State | Operator) -> Any:
        r"""Return the native dynamiqs norm with its default PSD assumption.

        Parameters
        ----------
        state_or_op : QArray
            Ket or bra for the Euclidean norm, or a positive-semidefinite matrix
            for its real trace. General indefinite operators are outside this hook's
            matrix convention.

        Returns
        -------
        jax.Array
            Native scalar; traced inputs retain their gradients.
        """
        return dq.norm(state_or_op)

    def trace(self, op: Operator) -> complex:
        return dq.trace(op)

    # ------------------------------------------------------------------
    # Operator / state factories
    # ------------------------------------------------------------------

    def eager_operators(self) -> Any:
        """Temporarily use dense Dynamiqs operators, then restore the global layout."""
        from contextlib import contextmanager

        from dynamiqs.qarrays.layout import get_layout, set_global_layout

        @contextmanager
        def dense_layout() -> Any:
            previous = get_layout()
            dq.set_layout("dense")
            try:
                yield
            finally:
                set_global_layout(previous)

        return dense_layout()

    def destroy(self, n: int) -> Operator:
        return _layout_operator(n, 1, np.sqrt(np.arange(n, dtype=float)))

    def create(self, n: int) -> Operator:
        return _layout_operator(n, -1, np.append(np.sqrt(np.arange(1, n, dtype=float)), 0.0))

    def number(self, n: int) -> Operator:
        return _layout_operator(n, 0, np.arange(n, dtype=float))

    def identity(self, n: int) -> Operator:
        return _layout_operator(n, 0, np.ones(n))

    def diag(self, values: Any, dims: list[list[int]] | None = None) -> Operator:
        r"""Build a backend-native sparse-DIA diagonal operator from main-diagonal *values*.

        Overrides the protocol default to keep the operator in sparse-DIA
        layout even when *values* is a JAX tracer — the offsets ``(0,)`` are
        concrete, so the sparse-DIA constructor accepts the traced
        values directly. Without this override, dynamiqs's
        ``from_canonical_operator`` densifies any traced DIA payload, which
        forces a dense ``H₀`` for circuit-style devices and triggers the
        sparse→dense warning during static-Hamiltonian assembly.

        Parameters
        ----------
        values : array_like
            Main-diagonal entries, flattened in input order.
        dims : list or None, default None
            Subsystem dimensions as in from_array; None uses one subsystem.

        See Also
        --------
        quchip.backend.protocol.Backend.diag
        """
        v = (jnp if contains_tracer(values) else np).asarray(values, dtype=complex).reshape(-1)
        n = v.shape[0]
        dim_tuple = self._coerce_dims(dims, (n, n))
        if dim_tuple is None:
            dim_tuple = (n,)
        return _dia_qarray(dim_tuple, (0,), v[None, :])

    def from_array(self, data: Any, dims: list[list[int]] | None = None) -> Operator:
        if isinstance(data, QArray):
            dims_tuple = self._coerce_dims(dims, data.shape)
            if dims_tuple is None or dims_tuple == data.dims:
                return data
            return replace(data, dims=dims_tuple)
        if hasattr(data, "to_jax"):
            data = data.to_jax()
        elif hasattr(data, "full"):
            data = data.full()
        array = _complex_payload(data)
        dims_tuple = self._coerce_dims(dims, array.shape)
        if dims_tuple is None:
            return dq.asqarray(array)
        return dq.asqarray(array, dims=dims_tuple)

    def to_canonical_operator(self, op: Operator) -> Any:
        from quchip.engine.ir import CanonicalOperator

        # getattr's default must stay lazy: to_array densifies a SparseDIA
        # operator (dim² allocation) just to read a shape that is discarded
        # whenever `.dims` exists — which it does for every dynamiqs qarray.
        dims_attr = getattr(op, "dims", None)
        dims = tuple(dims_attr) if dims_attr is not None else (self.to_array(op).shape[0],)
        labels = tuple(str(i) for i in range(len(dims)))
        if isinstance(op, QArray) and op.layout is dq.dia:
            return CanonicalOperator.from_dia(
                jnp.asarray(op.data.diags, dtype=jnp.complex128),
                np.asarray(op.data.offsets, dtype=int),
                shape=op.shape, dims=dims, basis="fock", subsystem_labels=labels,
            )
        if isinstance(op, QArray):
            return CanonicalOperator.from_dense(
                jnp.asarray(op.to_jax(), dtype=jnp.complex128),
                dims=dims, basis="fock", subsystem_labels=labels,
            )

        arr = jnp.asarray(op, dtype=jnp.complex128)
        return CanonicalOperator.from_dense(
            arr, dims=(arr.shape[0],), basis="fock", subsystem_labels=("0",),
        )

    def from_canonical_operator(self, canonical: Any) -> Operator:
        dims = tuple(canonical.dims)
        if canonical.layout == "dense":
            return dq.asqarray(_complex_payload(canonical.values), dims=dims)
        if canonical.layout == "dia":
            from quchip.engine.bands import canonical_to_dense_array

            if contains_tracer(canonical.offsets):
                # Sparse structure must be static. Value payloads may remain
                # traced because sparse-DIA diagonals are JAX leaves.
                dense = canonical_to_dense_array(canonical)
                return dq.asqarray(jnp.asarray(dense, dtype=jnp.complex128), dims=dims)
            return _dia_qarray(dims, tuple(int(x) for x in canonical.offsets), canonical.values)
        from quchip.engine.bands import canonical_to_coo

        rows, cols, values = canonical_to_coo(canonical)
        return self._sparse_qarray_from_coo(rows, cols, values, dims)

    def coerce_operator(self, op: Operator) -> Operator:
        return self.to_array(op)

    def dag(self, op: Operator) -> Operator:
        return dq.dag(op)

    def eigenenergies(self, op: Operator) -> Any:
        return jnp.linalg.eigvalsh(self.to_array(op))

    def eigensystem_data(self, op: Operator) -> EigensystemData:
        dense = self.to_array(op)
        evals, evecs = jnp.linalg.eigh(dense)
        dims = getattr(op, "dims", (dense.shape[0],))

        # Defer per-column ket construction: the dressing / sweep hot path reads
        # only evals + evecs + labeling, so the D backend-ket allocations (a
        # second O(D**2) densification) are pure waste there. evecs stays a plain
        # jnp array (never routed through a Python list) so vmap/grad are intact.
        def build_states() -> list[Any]:
            return [dq.asqarray(evecs[:, idx:idx + 1], dims=dims) for idx in range(evecs.shape[1])]

        return EigensystemData(
            eigenvalues=evals,
            eigenvector_matrix=evecs,
            _states_builder=build_states,
        )

    def expect(self, op: Operator, state: State) -> complex:
        return dq.expect(op, state)

    def ptrace(self, state: State, keep: int | list[int], dims: list[int]) -> State:
        keep_arg = tuple(keep) if isinstance(keep, list) else keep
        return dq.ptrace(state, keep_arg, dims=tuple(dims))

    # ------------------------------------------------------------------
    # Batched-over-time extraction (dq is batch-axis-native)
    # ------------------------------------------------------------------
    #
    # ``result.states`` is a single stacked ``QArray`` of shape ``(T, n, 1)``
    # (kets) or ``(T, n, n)`` (DMs). ``dq.expect`` / ``dq.ptrace`` /
    # ``dq.overlap`` broadcast over leading axes, so each extractor is one
    # batched call — no Python per-time loop, and the stacked ``QArray`` stays
    # a single jnp pytree (strictly more traceable than a list of tracers).

    def stack_states(self, states: Any) -> Any:
        if hasattr(states, "shape") and not isinstance(states, (list, tuple)):
            return states
        return self._stack_qarray_batch(list(states))

    def expect_over_time(self, op: Operator, stacked_states: Any) -> Any:
        # Native dq.expect fast path; the generic protocol overlap/populations
        # extractors are already array-namespace-parameterized (jnp here), so
        # they need no dynamiqs override — and overlap stays the complex
        # ⟨target|ψ⟩ amplitude (dq.overlap would square it and drop the phase).
        return jnp.asarray(dq.expect(op, stacked_states))

    def ptrace_over_time(self, stacked_states: Any, keep: int | list[int], dims: list[int]) -> Any:
        keep_arg = tuple(keep) if isinstance(keep, list) else keep
        reduced = dq.ptrace(stacked_states, keep_arg, dims=tuple(dims))
        return jnp.asarray(reduced.to_jax(), dtype=jnp.complex128)

    def matmul(self, a: Operator, b: Operator) -> Operator:
        """Multiply operators, keeping a product of sparse-DIA factors sparse.

        Concrete factors multiply on the host; traced sparse-DIA factors
        multiply their diagonals directly.
        """
        if not _plain_operators((a, b)) or tuple(a.dims) != tuple(b.dims):
            return super().matmul(a, b)
        left, right = _host_matrix(a), _host_matrix(b)
        if left is None or right is None:
            left_dia, right_dia = _dia_parts(a), _dia_parts(b)
            if left_dia is None or right_dia is None:
                return super().matmul(a, b)
            return _dia_qarray(tuple(a.dims), *_matmul_dia_traced(left_dia, right_dia))
        product = left @ right
        if a.layout is dq.dia and b.layout is dq.dia:
            return _dia_qarray(tuple(a.dims), *_dense_to_dia(product))
        return dq.asqarray(jnp.asarray(product), dims=tuple(a.dims))

    def linear_combination(self, terms: Sequence[tuple[Any, Operator]]) -> Operator:
        """Accumulate scalar multiples of operators into one qarray, on the host when every term is concrete.

        Sparse-DIA terms sum their diagonals into a sparse-DIA result; a sum
        that includes a dense term is dense, as in qarray arithmetic. Batched
        operators or coefficients use qarray arithmetic.
        """
        operators = [op for _, op in terms]
        dims = {tuple(op.dims) for op in operators if isinstance(op, QArray)}
        if not terms or len(dims) != 1 or not _plain_operators(operators) or any(np.ndim(c) for c, _ in terms):
            return super().linear_combination(terms)
        parts = [_dia_parts(op) for op in operators]
        if any(part is None for part in parts):
            return _dense_linear_combination(terms, dims.pop())
        coefficients = [_concrete_coefficient(coefficient) for coefficient, _ in terms]
        offsets = sorted({offset for term_offsets, _ in parts for offset in term_offsets})
        index = {offset: row for row, offset in enumerate(offsets)}
        shape = (len(offsets), parts[0][1].shape[-1])
        if all(c is not None for c in coefficients) and not contains_tracer([diags for _, diags in parts]):
            total = np.zeros(shape, dtype=complex)
            for (raw, _), coefficient, (term_offsets, term_diags) in zip(terms, coefficients, parts):
                rows = [index[offset] for offset in term_offsets]
                total[rows] += np.asarray(term_diags) if _is_unit(raw) else coefficient * np.asarray(term_diags)
        else:
            total = jnp.zeros(shape, dtype=jnp.complex128)
            for (raw, _), (term_offsets, term_diags) in zip(terms, parts):
                rows = np.asarray([index[offset] for offset in term_offsets], dtype=int)
                total = total.at[rows].add(term_diags if _is_unit(raw) else raw * term_diags)
        return _dia_qarray(dims.pop(), tuple(offsets), total)

    def tensor(self, *operators: Operator) -> Operator:
        if len(operators) == 1:
            return operators[0]
        sparse = [_host_dia(op) for op in operators]
        dense = [_host_dense(op) for op in operators]
        if not all(s is not None or d is not None for s, d in zip(sparse, dense)):
            traced = [_dia_parts(op) for op in operators]
            if not all(part is not None for part in traced):
                if any(part is not None for part in traced) and _plain_operators(operators):
                    # A dense factor makes the product dense, as in qarray arithmetic.
                    return dq.asqarray(reduce(jnp.kron, [op.to_jax() for op in operators]),
                                       dims=tuple(dim for op in operators for dim in op.dims))
                return dq.tensor(*operators)
            product = traced[0]
            for factor in traced[1:]:
                product = _kron_dia_traced(product, factor)
            return _dia_qarray(tuple(dim for op in operators for dim in op.dims), *product)
        # Concrete factors: the same product dynamiqs forms, computed on the host.
        dims = tuple(dim for op in operators for dim in op.dims)
        if all(factor is not None for factor in sparse):
            product = sparse[0]
            for factor in sparse[1:]:
                product = _kron_dia(product, factor)
            return _dia_qarray(dims, *product)
        matrices = [d if d is not None else _dia_to_dense(*s) for s, d in zip(sparse, dense)]
        return dq.asqarray(jnp.asarray(reduce(np.kron, matrices)), dims=dims)

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
        canonical = self.to_canonical_operator(op_ab)
        if canonical.is_sparse:
            return self._embed_two_body_sparse(canonical, index_a, index_b, dims)

        validate_two_body_indices(index_a, index_b, dims)
        expected_dim = dims[index_a] * dims[index_b]
        if canonical.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"Two-body operator dimension {canonical.shape[0]} does not match "
                f"dims[{index_a}]*dims[{index_b}] = {expected_dim}"
            )
        embedded = _embed_array(self.to_array(op_ab), (index_a, index_b), dims, jnp)
        return dq.asqarray(embedded, dims=tuple(dims))

    def basis(self, n: int, k: int) -> State:
        return dq.basis(n, k)

    def tensor_states(self, *states: State) -> State:
        if len(states) == 1:
            return states[0]
        return dq.tensor(*states)

    def coherent(self, n: int, alpha: complex) -> State:
        return dq.coherent(n, alpha)

    def state_to_dm(self, state: State) -> State:
        if not self.is_ket(state):
            return state
        column = self.coerce_state(state)
        return column @ dq.dag(column)

    def is_ket(self, state: State) -> bool:
        shape = tuple(state.shape)
        return len(shape) == 1 or (len(shape) == 2 and shape[1] == 1)

    def is_native_state(self, state: Any) -> bool:
        return isinstance(state, dq.QArray)

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
        r"""Validate dynamiqs integration options and fill the default step budget.

        Parameters
        ----------
        options : dict
            Supported keys: ``method`` (native deterministic dynamiqs method),
            ``gradient`` (native gradient configuration), ``max_steps`` (integer
            ceiling), ``progress_meter``, ``store_states``, and ``store_final_state``.
            ``nsteps`` aliases ``max_steps``; ``progress_bar`` aliases
            ``progress_meter``. Do not supply an alias and its canonical key together.
            Set tolerances on the native ``method`` object. With an explicit method,
            set its step limit there instead of also passing ``max_steps`` here.
            Monte Carlo methods and unrecognized keys are rejected.
        metadata : dict
            Lowering hints, including ordinary-GHz spectral/carrier bounds.
        tlist : array_like
            Save-time grid in ns used to estimate integration budgets.

        Returns
        -------
        dict
            Copied options with missing integration defaults filled.
        """
        resolved = self._normalize_dq_options(options)
        if "max_steps" not in resolved and resolved.get("method") is None:
            default = default_solver_steps(metadata, tlist)
            if default is not None:
                resolved["max_steps"] = default
        return resolved

    @staticmethod
    def _normalize_dq_options(options: dict[str, Any] | None) -> dict[str, Any]:
        """Map quchip option aliases onto dynamiqs' canonical key names.

        * ``progress_bar`` → ``progress_meter``
        * ``nsteps`` → ``max_steps``

        Return a fresh validated dict. Supplying both an alias and its canonical
        name is ambiguous and raises. Normalized dictionaries can be checked
        again at the native options and method boundaries.
        """
        resolved = {} if options is None else dict(options)
        for alias, canonical in (("progress_bar", "progress_meter"), ("nsteps", "max_steps")):
            if alias in resolved:
                if canonical in resolved:
                    raise ValueError(f"Supply only one of {alias!r} and {canonical!r}.")
                resolved[canonical] = resolved.pop(alias)
        supported = {"store_states", "store_final_state", "progress_meter", "max_steps", "method", "gradient"}
        unknown = set(resolved) - supported
        if unknown:
            raise ValueError(
                f"Unsupported Dynamiqs options: {sorted(unknown)}. "
                "Configure tolerances and integration controls on a native dynamiqs.method object."
            )
        method = resolved.get("method")
        if method is not None:
            if not isinstance(method, dq.method.Method):
                raise ValueError("Dynamiqs method must be a dynamiqs.method.Method instance.")
            if isinstance(method, (dq.method.JumpMonteCarlo, dq.method.DiffusiveMonteCarlo)):
                raise ValueError("quchip supports deterministic evolution methods; Monte Carlo trajectories "
                                 "and their termination diagnostics are not supported.")
            if resolved.get("max_steps") is not None:
                raise ValueError("Set max_steps on the explicit Dynamiqs method, not in both places.")
        return resolved

    def coerce_state(self, state: State, dims: tuple[int, ...] | None = None) -> State:
        if hasattr(state, "full"):  # qutip.Qobj duck-type; no qutip import needed
            return dq.asqarray(state, dims=dims)
        if len(tuple(state.shape)) == 1:  # flat native ket: dynamiqs solvers need a column
            column = jnp.asarray(state).reshape(-1, 1)
            return dq.asqarray(column, dims=dims) if dims else column
        return state

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
        result = dq.sesolve(
            H, psi0, jnp.asarray(tlist, dtype=float),
            **self._solve_kwargs(e_ops, options),
        )
        return self._wrap_result(result, solver="sesolve", options=options)

    def mesolve(
        self,
        H: Any,
        rho0: State,
        tlist: Any,
        c_ops: list[Operator] | None = None,
        e_ops: list[Operator] | None = None,
        options: dict[str, Any] | None = None,
    ) -> SolverResult:
        result = dq.mesolve(
            H, [] if c_ops is None else c_ops, rho0, jnp.asarray(tlist, dtype=float),
            **self._solve_kwargs(e_ops, options),
        )
        return self._wrap_result(result, solver="mesolve", options=options)

    def _stationary_liouvillian(self, engine_result: Any) -> Any:
        """Lower one static engine description to Dynamiqs' JAX Liouvillian."""
        if engine_result.dynamic_terms:
            raise ValueError("Stationary analysis requires a static resolved Hamiltonian.")
        hamiltonian = self.prepare_hamiltonian(engine_result).rhs
        collapse_ops = self._collapse_operators(engine_result)
        dimension = math.prod(engine_result.dims)
        # The dense D²×D² generator alone is a lower bound on the solve's memory.
        require_memory(
            16 * dimension**4,
            task=f"The dense D²×D² stationary Liouvillian at Hilbert dimension D = {dimension}",
            remedy="Reduce the device cutoffs.",
        )
        return self.to_array(dq.slindbladian(hamiltonian, collapse_ops))

    def steadystate(self, problem: Any, *, prepared: PreparedStationary | None = None) -> SteadyStateSolverResult:
        r"""Solve a static Lindblad generator by a trace-constrained JAX solve.

        Parameters
        ----------
        problem : SteadyStateProblem
            Captured static model, observables, and stationary solver options.
        prepared : PreparedStationary or None, default None
            Matching prepared generator; ``None`` builds it.

        Returns
        -------
        SteadyStateSolverResult
            Stationary density matrix and convergence diagnostics.

        See Also
        --------
        quchip.backend.protocol.Backend.steadystate

        Notes
        -----
        ``problem.options`` accepts only ``method="direct"`` and
        ``rank_tolerance`` (singular-value cutoff, default automatic). A nonunique
        stationary kernel produces a NaN state; inspect the returned nullity.
        """
        options = dict(problem.options)
        method = options.pop("method", "direct")
        if method != "direct":
            raise ValueError("Dynamiqs steady states support only method='direct'.")
        rank_tolerance = options.pop("rank_tolerance", None)
        if options:
            raise ValueError(
                "Unknown Dynamiqs steady-state options: " + ", ".join(sorted(options))
            )

        liouvillian = self.prepare_stationary(problem.engine_result, prepared=prepared).liouvillian
        dimension = math.prod(problem.engine_result.dims)

        trace_row = jnp.eye(dimension, dtype=jnp.complex128).reshape(-1)
        constrained = liouvillian.at[-1, :].set(trace_row)
        target = jnp.zeros((dimension * dimension,), dtype=jnp.complex128)
        target = target.at[-1].set(1.0)
        state_vector = jnp.linalg.solve(constrained, target)

        singular_values = jnp.linalg.svd(liouvillian, compute_uv=False)
        if rank_tolerance is None:
            rank_tolerance = (
                max(liouvillian.shape)
                * jnp.finfo(jnp.float64).eps
                * singular_values[0]
            )
        nullity = jnp.sum(singular_values <= rank_tolerance)
        state_array = jnp.reshape(state_vector, (dimension, dimension)).T
        state_array = jnp.where(
            nullity == 1,
            state_array,
            jnp.full_like(state_array, jnp.nan + 0.0j),
        )
        state = dq.asqarray(state_array, dims=tuple(problem.engine_result.dims))
        residual = jnp.linalg.norm(liouvillian @ state_vector)
        expectations = None
        if isinstance(problem.e_ops, list):
            expectations = [dq.expect(operator, state) for operator in problem.e_ops]

        return SteadyStateSolverResult(
            state=state,
            expect=expectations,
            stats={"method": method, "uniqueness_enforced": True},
            residual=residual,
            nullity=nullity,
            _condition_number=DeferredValue(partial(self._stationary_condition_number, problem.engine_result)),
        )

    def _stationary_condition_number(self, engine_result: Any) -> Any:
        liouvillian = self._stationary_liouvillian(engine_result)
        return stationary_condition_number(liouvillian, math.prod(engine_result.dims), xp=jnp)

    def linear_response(self, problem: Any) -> LinearResponseSolverResult:
        return linear_response(problem, xp=jnp)

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
        dimension = math.prod(engine_result.dims)
        targets = jnp.stack(
            [
                jnp.asarray(operator.to_dense(), dtype=jnp.complex128).T.reshape(-1)
                for _, operator in sources
            ],
            axis=1,
        )
        targets = targets.at[-1, :].set(0.0)
        trace_row = jnp.eye(dimension, dtype=jnp.complex128).reshape(-1)
        identity = jnp.eye(dimension * dimension, dtype=jnp.complex128)
        native_observables = tuple(
            (label, jnp.asarray(operator.to_dense(), dtype=jnp.complex128))
            for label, operator in observables
        )
        values: dict[tuple[str, str], list[Any]] = {
            (source_label, label): [] for source_label, _ in sources for label, _ in observables
        }

        for frequency in jnp.atleast_1d(jnp.asarray(frequencies, dtype=float)):
            shifted = liouvillian + 1j * (2.0 * jnp.pi) * frequency * identity
            constrained = shifted.at[-1, :].set(trace_row)
            solutions = jnp.linalg.solve(constrained, targets)
            for column, (source_label, _) in enumerate(sources):
                response = solutions[:, column].reshape((dimension, dimension)).T
                for label, observable in native_observables:
                    values[(source_label, label)].append(jnp.einsum("ij,ji->", observable, response))

        return {key: jnp.asarray(items) for key, items in values.items()}

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
        dimension = math.prod(engine_result.dims)
        initial_vector = jnp.asarray(initial.to_dense(), dtype=jnp.complex128).T.reshape(-1)
        native_observables = tuple(
            (label, jnp.asarray(operator.to_dense(), dtype=jnp.complex128))
            for label, operator in observables
        )
        values: dict[str, list[Any]] = {label: [] for label, _ in observables}

        for time in jnp.atleast_1d(jnp.asarray(times, dtype=float)):
            evolved = (jsp_linalg.expm(liouvillian * time) @ initial_vector).reshape(
                (dimension, dimension)
            ).T
            for label, observable in native_observables:
                values[label].append(jnp.einsum("ij,ji->", observable, evolved))

        return {label: jnp.asarray(items) for label, items in values.items()}

    # Build H inside jit from stable quchip pytrees, avoiding ephemeral Dynamiqs closure keys.
    # Cache only callables and static solver configuration; pass all physics data (operators,
    # coefficients, states, times and observables) as traced arguments. JAX then reuses matching
    # structures without capturing stale values or tracers, preserving grad/vmap across sweeps.

    _jit_solve_cache: dict[Any, Any]

    def _get_jit_solve_cache(self) -> dict[Any, Any]:
        cache = getattr(self, "_jit_solve_cache", None)
        if cache is None:
            cache = {}
            self._jit_solve_cache = cache
        return cache

    def solve_problem(self, problem: Any) -> SolverResult:
        if problem.stochastic:
            inputs = self._trajectory_inputs(problem)
            return SolverResult(times=problem.tlist, solver=problem.solver,
                                native=getattr(dq, problem.solver)(**inputs))
        engine_result = problem.engine_result
        if not self._engine_result_is_cacheable(engine_result):
            return super().solve_problem(problem)

        tlist_arr, c_ops, solver_name, opts, e_ops_arg = self._resolve_solve_config(
            problem, engine_result
        )
        native_options = self._options_from_dict(opts)
        method_obj = self._method_from_dict(opts)
        gradient = opts.get("gradient")

        # Decompose the engine result into clean, traced pytree leaves. Operator
        # *values* are rebuilt here (cheap, ~1 ms) and flow into the jit as
        # traced args (never closed over) so distinct skeletons with the same
        # structure never collide on a stale artifact.
        static_ops = [self.from_canonical_operator(t.operator) for t in engine_result.static_terms]
        static_coeffs = [
            jnp.asarray(t.coefficient, dtype=jnp.complex128)
            for t in engine_result.static_terms
        ]
        groups = [[engine_result.dynamic_terms[slot] for slot in group]
                  for group in _shared_operator_slots((engine_result,))]
        dyn_ops = [self.from_canonical_operator(terms[0].operator) for terms in groups]
        dyn_mods = [_summed_modulation([t.time_dependence for t in terms]) for terms in groups]

        solve_fn = self._cached_jit_solve(
            solver_name=solver_name,
            native_options=native_options,
            method_obj=method_obj,
            gradient=gradient,
            has_e_ops=e_ops_arg is not None,
            n_static=len(static_ops),
            n_dynamic=len(dyn_ops),
            n_c_ops=len(c_ops),
        )

        result = solve_fn(
            static_ops,
            static_coeffs,
            dyn_ops,
            dyn_mods,
            c_ops,
            e_ops_arg if e_ops_arg is not None else [],
            self.coerce_state(problem.initial_state, dims=problem.engine_result.dims),
            tlist_arr,
        )
        return self._wrap_result(result, solver=solver_name, options=opts)

    def _trajectory_inputs(self, problem: Any) -> dict[str, Any]:
        if problem.solver not in ("jssesolve", "dssesolve", "dsmesolve"):
            raise ValueError(f"Dynamiqs does not provide {problem.solver!r}.")
        options = dict(problem.options)
        if problem.states is not None:
            if "save_states" in options:
                raise ValueError("Conflicting states and native save_states.")
            options["save_states"] = problem.states == "all"
        kwargs = dict(H=self.prepare_hamiltonian(problem.engine_result, problem.tlist).rhs,
                      tsave=problem.tlist, exp_ops=problem.e_ops, **options, **problem.run_args)
        state = self.coerce_state(problem.initial_state, dims=problem.engine_result.dims)
        kwargs["rho0" if problem.solver == "dsmesolve" else "psi0"] = state
        if problem.solver in ("dssesolve", "dsmesolve") and (
            problem.monitoring is not None or problem.solver == "dssesolve"
        ):
            from quchip.engine.monitoring import monitored_operators

            loss, monitored, etas = monitored_operators(problem)
            kwargs["jump_ops"] = loss + monitored
            if problem.solver == "dsmesolve":
                if "etas" in problem.run_args:
                    raise ValueError("Choose with_monitoring or native etas, not both.")
                kwargs["etas"] = jnp.asarray([0.0] * len(loss) + etas)
        else:
            kwargs["jump_ops"] = self._collapse_operators(problem.engine_result)
        return kwargs

    @staticmethod
    def _engine_result_is_cacheable(engine_result: Any) -> bool:
        """Accept static results and results whose dynamic terms are all rebuildable ScalarModulations."""
        terms = getattr(engine_result, "dynamic_terms", None)
        if terms is None:
            return False
        return all(isinstance(t.time_dependence, ScalarModulation) for t in terms)

    def _cached_jit_solve(
        self,
        *,
        batched: bool = False,
        point_e_ops: bool = False,
        solver_name: str,
        native_options: dict[str, Any],
        method_obj: Any,
        gradient: Any,
        has_e_ops: bool,
        n_static: int,
        n_dynamic: int,
        n_c_ops: int,
    ) -> Any:
        """Cache single or vmapped solves by static solver configuration and term counts.
        JAX keys numerical arguments by pytree structure and shape; no physics values enter this key."""
        key = (
            "batch" if batched else "single",
            point_e_ops,
            solver_name,
            tuple(sorted(native_options.items())),
            method_obj,
            gradient,
            bool(has_e_ops),
            int(n_static),
            int(n_dynamic),
            int(n_c_ops),
        )
        cache = self._get_jit_solve_cache()
        fn = cache.get(key)
        if fn is not None:
            return fn

        kwargs: dict[str, Any] = {**native_options, "method": method_obj}
        if gradient is not None:
            kwargs["gradient"] = gradient

        def _solve(
            static_ops,
            static_coeffs,
            dynamic_ops,
            dynamic_payloads,
            c_ops,
            e_ops,
            state0,
            tarr,
        ):
            def native_solve(hamiltonian, jumps, observables, state):
                if solver_name == "mesolve":
                    return dq.mesolve(hamiltonian, list(jumps), state, tarr, exp_ops=observables, **kwargs)
                return dq.sesolve(hamiltonian, state, tarr, exp_ops=observables, **kwargs)

            if batched:
                from quchip.backend._dynamiqs_status import solve_with_status

                # Build single-point callables after mapping their numerical leaves.
                def solve_point(sops, coeffs, dops, signals, jumps, observables, state):
                    rhs = self._assemble_modulated_rhs(sops, coeffs, dops, signals)
                    return solve_with_status(rhs, list(jumps), state, tarr,
                        list(observables) if has_e_ops else None, solver=solver_name, **kwargs)

                axes = (tuple(0 if op.ndim > 2 else None for op in static_ops),
                        tuple(0 if jnp.ndim(coeff) else None for coeff in static_coeffs),
                        tuple(0 if op.ndim > 2 else None for op in dynamic_ops),
                        0 if state0.shape[0] > 1 else None, 0, 0 if point_e_ops else None, 0)
                result_type = dq.MESolveResult if solver_name == "mesolve" else dq.SESolveResult
                return jax.vmap(solve_point, in_axes=axes, out_axes=(result_type.out_axes(), 0))(
                    static_ops, static_coeffs, dynamic_ops, dynamic_payloads, c_ops, e_ops, state0)
            rhs = self._assemble_modulated_rhs(
                static_ops, static_coeffs, dynamic_ops,
                [modulation.signal for modulation in dynamic_payloads],
            )
            return native_solve(rhs, c_ops, list(e_ops) if has_e_ops else None, state0)

        jitted = jax.jit(_solve)
        cache[key] = jitted
        return jitted

    def _solve_kwargs(
        self, e_ops: list[Operator] | None, options: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Build the common dynamiqs solver kwargs from quchip's option dict."""
        kwargs: dict[str, Any] = {
            "exp_ops": e_ops,
            **self._options_from_dict(options),
            "method": self._method_from_dict(options),
        }
        # Opt-in differentiation mode. Default (no key) leaves dynamiqs'
        # checkpointed reverse-mode adjoint untouched. ``dq.gradient.Forward()``
        # (paired with ``jax.jacfwd``/``jvp``, never ``jax.grad``) is ~2.5x
        # faster for few-input pulse optimization; the user owns the choice
        # since forward-mode cannot be a blanket default (it breaks reverse-mode
        # ``jax.grad`` on the integrator's ``lax.while_loop``).
        if options is not None and options.get("gradient") is not None:
            kwargs["gradient"] = options["gradient"]
        return kwargs

    # ------------------------------------------------------------------
    # Batched IR lowering / dispatch
    # ------------------------------------------------------------------

    def prepare_hamiltonian(
        self,
        engine_result: Any,
        tlist: Any | None = None,
    ) -> PreparedHamiltonian:
        r"""Convert a :class:`EngineResult` into a dynamiqs native RHS.

        Static terms are summed as qarrays; dynamic terms with
        ``ScalarModulation`` time-dependence are wrapped via
        ``dynamiqs.modulated`` with a JAX-traceable :class:`_SignalCallable`.
        ``tlist`` is passed through in metadata but not used for sampling —
        dynamiqs evaluates callables on the integrator's adaptive grid.

        Parameters
        ----------
        engine_result : EngineResult
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
        static_ops = [
            self.from_canonical_operator(term.operator)
            for term in engine_result.static_terms
        ]
        static_coeffs = [term.coefficient for term in engine_result.static_terms]
        dyn_ops: list[Any] = []
        dyn_signals: list[Any] = []
        slots: dict[int, int] = {}
        for operator, signal in self._scalar_dynamic_terms(engine_result):
            slot = slots.setdefault(id(operator), len(dyn_ops))
            if slot == len(dyn_ops):
                dyn_ops.append(self.from_canonical_operator(operator))
                dyn_signals.append([])
            dyn_signals[slot].append(signal)
        dyn_signals = [signals[0] if len(signals) == 1 else Add(tuple(signals)) for signals in dyn_signals]

        rhs = self._assemble_modulated_rhs(static_ops, static_coeffs, dyn_ops, dyn_signals)
        if rhs is None:
            raise ValueError("EngineResult must contain at least one static or dynamic term.")

        return PreparedHamiltonian(rhs=rhs, metadata=dict(engine_result.metadata))

    def _assemble_modulated_rhs(
        self,
        static_ops: Sequence[Any],
        static_coeffs: Sequence[Any],
        dyn_ops: Sequence[Any],
        dyn_signals: Sequence[Any],
    ) -> Any:
        """Sum static ``coeff·op`` terms and modulated dynamic terms into one RHS.

        Static terms accumulate as ``Σ coeff·op``; each dynamic term wraps its
        signal-program AST in a JAX-traceable :class:`_SignalCallable` via
        ``dynamiqs.modulated``. Shared by :meth:`prepare_hamiltonian` and the
        cached-jit single-solve so the static/dynamic split lives in one place
        (the cached path passes traced operators/coefficients as jit arguments,
        so this stays fully traceable). The time-dependent terms form one sum:
        dynamiqs' ``+`` re-broadcasts every earlier term, so adding N terms one
        at a time nests their callables up to N deep and traces each signal
        O(N) times.
        """
        summed = _summed_hamiltonian(static_ops, static_coeffs, dyn_ops, dyn_signals) if dyn_ops else None
        if summed is not None:
            return summed
        rhs = self.linear_combination(list(zip(static_coeffs, static_ops))) if static_ops else None
        if not dyn_ops:
            return rhs
        dynamic = [
            dq.modulated(_SignalCallable(signal), op, discontinuity_ts=_signal_discontinuities(signal))
            for op, signal in zip(dyn_ops, dyn_signals)
        ]
        terms = dynamic if rhs is None else [dq.constant(rhs), *dynamic]
        return terms[0] if len(terms) == 1 else SummedTimeQArray(terms)

    def prepare_batch(self, batch: Any) -> DeferredBatch:
        r"""Lower compatible problems into stable leaves for one vmapped solve.

        Shared operators (static + per-slot dynamic) are converted exactly
        once via an id-keyed cache. For each dynamic slot, the per-element
        :class:`ScalarModulation` signals are stacked leaf-by-leaf along a
        leading batch axis (``jnp.stack`` on matching pytree leaves) and
        retained as engine pytrees. RHS callables are built later inside the
        cached JIT, so repeated objectives reuse compilation without caching
        value-bearing Hamiltonians.

        Raises
        ------
        ValueError
            If a slot contains heterogeneous pytree structures (cannot be
            stacked) or a non-``ScalarModulation`` time dependence.

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
        engine_results = tuple(problem.engine_result for problem in batch.problems)
        cached_native = self._make_op_cache()
        static_term_ids = engine_results[0]._static_term_ids
        if all(result._static_term_ids == static_term_ids for result in engine_results[1:]):
            static_operators = [
                cached_native(term.operator)
                for term in engine_results[0].static_terms
            ]
            static_coefficients = [
                jnp.asarray(term.coefficient, dtype=jnp.complex128)
                for term in engine_results[0].static_terms
            ]
        else:
            static_rhs = [
                self._sum_terms(result.static_terms, cached_native)
                for result in engine_results
            ]
            reference_rhs = next(value for value in static_rhs if value is not None)
            static_operators = [self._stack_qarray_batch([
                value if value is not None else 0 * reference_rhs for value in static_rhs
            ])]
            static_coefficients = [jnp.asarray(1.0, dtype=jnp.complex128)]

        dynamic_operators: list[Any] = []
        dynamic_signals: list[Any] = []
        for group in _shared_operator_slots(engine_results):
            slot = group[0]
            terms = tuple(result.dynamic_terms[slot] for result in engine_results)
            if all(term.operator is terms[0].operator for term in terms[1:]):
                op = cached_native(terms[0].operator)
            else:
                op = self._stack_qarray_batch(
                    [cached_native(term.operator) for term in terms]
                )
            slot_signals = tuple(map(_summed_modulation, zip(*(batch.signals_for(member) for member in group))))
            ref_td = slot_signals[0]
            if not isinstance(ref_td, ScalarModulation):
                raise ValueError(f"dynamiqs prepare_batch only supports ScalarModulation (slot {slot}).")
            ref_struct = jtu.tree_structure(ref_td)
            for td in slot_signals[1:]:
                # PyTreeDef defines runtime __eq__/__ne__ (pybind11 value equality);
                # jax's stubs omit a mypy-visible signature for it.
                if jtu.tree_structure(td) != ref_struct:  # type: ignore[operator]
                    raise ValueError(f"Heterogeneous signal pytree at slot {slot}; cannot stack.")
            stacked_td = (
                ref_td if len(slot_signals) == 1
                else jtu.tree_map(lambda *xs: jnp.stack(xs), *slot_signals)
            )
            dynamic_operators.append(op)
            dynamic_signals.append(stacked_td.signal)

        return DeferredBatch(
            shared=_DynamiqsBatchHamiltonian(
                static_operators=tuple(static_operators),
                static_coefficients=tuple(static_coefficients),
                dynamic_operators=tuple(dynamic_operators),
                dynamic_signals=tuple(dynamic_signals),
            ),
            batch_size=batch.batch_size,
            metadata=_aggregate_batch_metadata(list(engine_results)),
            tlist=batch.tlist,
        )

    def solve_batch(self, batch: Any, *, progress: bool = True) -> list[SolverResult]:
        r"""Solve a :class:`SolveBatch` via a single native dynamiqs vmap.

        Raises :class:`RuntimeError` when the batch is not structurally
        homogeneous — no silent fallback. Callers with heterogeneous inputs
        should regroup through :func:`quchip.engine.solve_many`.

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
            solver = batch.problems[0].solver
            if any(problem.solver != solver for problem in batch.problems):
                raise ValueError("A native trajectory batch requires one solver.")
            inputs = [self._trajectory_inputs(problem) for problem in batch.problems]
            shared = {"tsave": inputs[0].pop("tsave")}
            for kwargs in inputs[1:]:
                kwargs.pop("tsave")
            if solver == "dsmesolve":
                # Native 0.3.4 validates and partitions efficiencies in Python.
                shared["etas"] = inputs[0].pop("etas")
                for kwargs in inputs[1:]:
                    if not np.array_equal(kwargs.pop("etas"), shared["etas"]):
                        raise ValueError(
                            "Native SME batching requires shared efficiencies; use solve_many(list(batch))."
                        )
            static = [eqx.partition(kwargs, eqx.is_array)[1] for kwargs in inputs]
            if any(not eqx.tree_equal(static[0], value) for value in static[1:]):
                raise ValueError("Native trajectory batches require matching methods and run options.")
            stacked = jtu.tree_map(lambda *values: jnp.stack(values) if eqx.is_array(values[0]) else values[0],
                                   *inputs)
            native = eqx.filter_vmap(lambda kwargs: getattr(dq, solver)(**kwargs, **shared))(stacked)
            return [SolverResult(times=problem.tlist, solver=solver,
                                 native=jtu.tree_map(lambda value: value[i] if eqx.is_array(value) else value, native))
                    for i, problem in enumerate(batch.problems)]
        if batch.batch_size == 0:
            return []

        prepared = self.prepare_batch(batch)
        reference = batch.problems[0]
        tlist_arr = self.array_module.asarray(batch.tlist, dtype=float)
        c_ops = self._stack_batch_collapse_operators(batch)
        solver_names = {problem.solver_name(self) for problem in batch.problems}
        if len(solver_names) != 1:
            raise ValueError("Every SolveBatch point must resolve to the same solver.")
        solver_name = solver_names.pop()
        opts = self._resolve_problem_options(
            reference, metadata=prepared.metadata, tlist=tlist_arr, solver_name=solver_name,
        )
        e_ops, point_e_ops = self._batch_e_ops(batch)
        if point_e_ops is not None:
            e_ops = point_e_ops
        options = self._options_from_dict(opts, cartesian_batching=False)
        method = self._method_from_dict(opts)
        gradient = opts.get("gradient")

        hamiltonian = prepared.shared
        if not isinstance(hamiltonian, _DynamiqsBatchHamiltonian):
            raise TypeError("Dynamiqs prepare_batch returned an invalid native batch payload.")
        chip_dims = reference.engine_result.dims
        native_states = [self.coerce_state(s, dims=chip_dims) for s in batch.initial_states]
        stacked_state = self._stack_qarray_batch(native_states)
        solve_fn = self._cached_jit_solve(
            batched=True,
            point_e_ops=point_e_ops is not None,
            solver_name=solver_name,
            native_options=options,
            method_obj=method,
            gradient=gradient,
            has_e_ops=e_ops is not None,
            n_static=len(hamiltonian.static_operators),
            n_dynamic=len(hamiltonian.dynamic_operators),
            n_c_ops=len(c_ops),
        )

        batched_result, status = solve_fn(
            hamiltonian.static_operators,
            hamiltonian.static_coefficients,
            hamiltonian.dynamic_operators,
            hamiltonian.dynamic_signals,
            c_ops,
            e_ops if e_ops is not None else [],
            stacked_state,
            tlist_arr,
        )

        from quchip.backend._dynamiqs_status import require_success

        batched_result = require_success(batched_result, status,
            tuple(batch.params_at(i) for i in range(batch.batch_size)), traced_context=batch._failure_context)
        return self._split_batched_result(batched_result, solver=solver_name, count=batch.batch_size, options=opts)

    # ------------------------------------------------------------------
    # Internal: dims / shape coercion
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_dims(dims: list[list[int]] | None, shape: Sequence[int]) -> tuple[int, ...] | None:
        """Normalize quchip's ``[[row], [col]]`` dims metadata to a flat dynamiqs tuple."""
        if dims is not None:
            return normalize_dims_from_list(dims)
        if len(shape) == 2:
            if shape[1] == 1 or shape[0] == shape[1]:
                return (shape[0],)
        return None

    def _embed_two_body_sparse(
        self,
        canonical: Any,
        index_a: int,
        index_b: int,
        dims: Sequence[int],
    ) -> Operator:
        """Embed a sparse two-body operator without densifying the full matrix.

        Decomposes each nonzero's ``(row, col)`` into subsystem indices
        ``(row_A, row_B)`` × ``(col_A, col_B)`` via ``divmod``, fans each out
        across every spectator basis state using the stride array, and
        reassembles in sparse-DIA layout. Keeps complexity linear
        in ``nnz × n_spectators`` rather than ``nnz × total_dim``.
        """
        from quchip.engine.bands import canonical_to_coo

        n_devices = len(dims)
        validate_two_body_indices(index_a, index_b, dims)

        d_a = dims[index_a]
        d_b = dims[index_b]
        expected_dim = d_a * d_b
        if canonical.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"Two-body operator dimension {canonical.shape[0]} does not match "
                f"dims[{index_a}]*dims[{index_b}] = {expected_dim}"
            )

        rows, cols, values = canonical_to_coo(canonical)
        if len(rows) == 0:
            return dq.zeros(*dims)

        other_indices = [idx for idx in range(n_devices) if idx not in (index_a, index_b)]
        strides = np.array(
            [int(np.prod(dims[idx + 1:], dtype=int)) for idx in range(n_devices)],
            dtype=int,
        )

        row_a, row_b = np.divmod(rows, d_b)
        col_a, col_b = np.divmod(cols, d_b)

        if other_indices:
            spectator_arr = np.array(
                list(itertools.product(*(range(dims[idx]) for idx in other_indices))), dtype=int,
            )
            spectator_offsets = spectator_arr @ strides[other_indices]
        else:
            spectator_offsets = np.zeros(1, dtype=int)
        n_spectators = spectator_offsets.size

        row_ab_offset = row_a * strides[index_a] + row_b * strides[index_b]
        col_ab_offset = col_a * strides[index_a] + col_b * strides[index_b]

        full_rows = (row_ab_offset[:, None] + spectator_offsets[None, :]).ravel().astype(int)
        full_cols = (col_ab_offset[:, None] + spectator_offsets[None, :]).ravel().astype(int)
        # Each source value fans out over n_spectators row/col pairs.
        xp = jnp if contains_tracer(values) else np
        full_values = xp.repeat(xp.asarray(values, dtype=complex), n_spectators)

        return self._sparse_qarray_from_coo(full_rows, full_cols, full_values, tuple(dims))

    @staticmethod
    def _sparse_qarray_from_coo(
        rows: np.ndarray,
        cols: np.ndarray,
        values: Any,
        dims: tuple[int, ...],
    ) -> Operator:
        """Build the smaller Dynamiqs layout from COO-format indices and values."""
        total_dim = int(np.prod(dims, dtype=int))
        if values.size == 0:
            return dq.zeros(*dims)

        rows_np = np.asarray(rows, dtype=int)
        cols_np = np.asarray(cols, dtype=int)
        offsets = np.unique(cols_np - rows_np)
        n_diagonals = len(offsets)
        # Integer index structure lives on NumPy; traced values stay in JAX, concrete ones on the host.
        if n_diagonals >= total_dim:
            shape, index = (total_dim, total_dim), (rows_np, cols_np)
        else:
            shape, index = (n_diagonals, total_dim), (np.searchsorted(offsets, cols_np - rows_np), cols_np)
        if contains_tracer(values):
            data = jnp.zeros(shape, dtype=jnp.complex128).at[index].add(jnp.asarray(values, dtype=jnp.complex128))
        else:
            data = np.zeros(shape, dtype=complex)
            np.add.at(data, index, np.asarray(values, dtype=complex))
        if n_diagonals >= total_dim:
            return dq.asqarray(jnp.asarray(data), dims=dims)
        return _dia_qarray(dims, tuple(int(x) for x in offsets.tolist()), data)

    # ------------------------------------------------------------------
    # Internal: options / method builders
    # ------------------------------------------------------------------

    @staticmethod
    def _options_from_dict(
        options: dict[str, Any] | None,
        *,
        cartesian_batching: bool = True,
    ) -> dict[str, Any]:
        """Map a quchip solver-options dict onto dynamiqs solver keyword arguments."""
        options = DynamiqsBackend._normalize_dq_options(options)
        return {
            "save_states": bool(options.get("store_states", True)),
            "cartesian_batching": cartesian_batching,
            "progress_meter": options.get("progress_meter", False),
        }

    @staticmethod
    def _method_from_dict(options: dict[str, Any] | None) -> Any:
        """Return the explicit ``method`` or the default ``Dopri8`` integrator.

        The default tolerances keep pulse-parameter gradients converged, not
        only the states. ``max_steps`` is an abort ceiling (see
        :meth:`resolve_solver_options` for the step-size-bound limitation this
        implies for finite-support pulses in long idle spans).
        """
        options = DynamiqsBackend._normalize_dq_options(options)
        method = options.get("method")
        if method is not None:
            return method
        max_steps = options.get("max_steps")
        steps = {} if max_steps is None else {"max_steps": int(max_steps)}
        return dq.method.Dopri8(rtol=_DEFAULT_RTOL, atol=_DEFAULT_ATOL, **steps)

    # ------------------------------------------------------------------
    # Internal: pytree / operator stacking
    # ------------------------------------------------------------------

    def _stack_qarray_batch(self, values: list[Any]) -> Operator:
        """Stack one homogeneous native batch without changing its layout."""
        try:
            return dq.stack(values)
        except (ValueError, NotImplementedError) as exc:
            raise ValueError(
                "Dynamiqs batches require one native layout, shape, dims, and DIA offset structure."
            ) from exc

    def _stack_batch_collapse_operators(self, batch: Any) -> list[Operator]:
        """Lower and align each collapse slot across batch points."""
        results = tuple(problem.engine_result for problem in batch.problems)
        counts = {len(result.collapse_terms) for result in results}
        if len(counts) != 1:
            raise ValueError("Every SolveBatch point must have the same collapse-term structure.")
        count = len(results[0].collapse_terms)
        return [
            self._stack_qarray_batch(
                [
                    self.array_module.sqrt(result.collapse_terms[slot].rate)
                    * self.from_canonical_operator(result.collapse_terms[slot].operator)
                    for result in results
                ]
            )
            for slot in range(count)
        ]

    def _batch_e_ops(self, batch: Any) -> tuple[list[Operator] | None, list[Operator] | None]:
        """Return shared observables or operators aligned with the native solve axis."""
        per_point = [problem.e_ops for problem in batch.problems]
        if all(ops is None for ops in per_point):
            return None, None
        if not all(isinstance(ops, list) for ops in per_point):
            raise ValueError("Every SolveBatch point must share the same expectation-operator structure.")
        if all(ops is per_point[0] for ops in per_point[1:]):
            return per_point[0], None
        from quchip.utils.values import value_fingerprint

        try:
            keys = [tuple(value_fingerprint(self.to_array(op)) for op in ops) for ops in per_point]
        except ValueError:
            pass  # Traced operator values must remain point-specific.
        else:
            if all(key == keys[0] for key in keys[1:]):
                return per_point[0], None
        counts = {len(ops) for ops in per_point}
        if len(counts) != 1:
            raise ValueError("Every SolveBatch point must have the same number of expectation operators.")
        count = len(per_point[0])
        point_e_ops = [
            self._stack_qarray_batch([ops[slot] for ops in per_point])
            for slot in range(count)
        ]
        return None, point_e_ops

    # ------------------------------------------------------------------
    # Internal: dynamiqs Result -> SolverResult
    # ------------------------------------------------------------------

    @staticmethod
    def _native_final_state(result: Any, solver: str) -> Any:
        """Restore vectorized final-only ME states, preserving subsystem dimensions."""
        state = result.final_state
        if solver == "mesolve" and state.vectorized:
            # Native Expm (0.3.4 to 0.3.6) unvectorizes sampled saves but retains ylast as a vector.
            state = dq.asqarray(dq.unvectorize(state).to_jax(), dims=state.dims)
        return state

    @staticmethod
    def _split_batched_result(
        result: Any, *, solver: str, count: int, options: dict[str, Any],
    ) -> list[SolverResult]:
        """Unpack a dynamiqs flat-batched solve into per-problem :class:`SolverResult`s."""
        has_states = options.get("store_states", True) and result.states is not None
        has_expects = getattr(result, "expects", None) is not None
        final_state = (DynamiqsBackend._native_final_state(result, solver)
                       if options.get("store_final_state", True) else None)

        return [
            SolverResult(
                times=result.tsave,
                # Keep the per-element trajectory stacked: ``result.states[i]``
                # is a ``(T, n, 1/n)`` QArray. NOT unstacked into a Python list
                # — the over-time extractors consume the stacked form directly,
                # and indexing/iterating a stacked QArray still yields per-time
                # states for viz.
                states=result.states[i] if has_states else None,
                expect=list(result.expects[i]) if has_expects else None,
                final_state=(
                    final_state[i] if options.get("store_final_state", True) and final_state is not None else None
                ),
                stats={"batched": True, "batch_index": i, "options": DynamiqsBackend._effective_options(result)},
                solver=solver,
            )
            for i in range(count)
        ]

    @staticmethod
    def _wrap_result(
        result: Any, solver: str, *, options: dict[str, Any] | None = None,
    ) -> SolverResult:
        """Convert a single dynamiqs ``Result`` into a :class:`SolverResult`."""
        if isinstance(result.method, dq.method.Expm):
            from quchip.backend._dynamiqs_status import expm_status, require_success

            status = jtu.tree_map(jnp.atleast_1d, expm_status(result))
            result = require_success(result, status, ({},), single=True)
        # Keep the stacked ``QArray`` (shape ``(T, n, 1/n)``) intact rather than
        # exploding it into ``T`` separate states: the over-time extractors read
        # the stacked form directly and a stacked QArray still indexes/iterates
        # per-time for viz.
        states = result.states if (options or {}).get("store_states", True) else None
        expect = list(result.expects) if getattr(result, "expects", None) is not None else None

        stats: dict[str, Any] = {"options": DynamiqsBackend._effective_options(result)}
        infos = getattr(result, "infos", None)
        if infos is not None:
            stats["infos"] = str(infos)
            nsteps = DynamiqsBackend._extract_nsteps(infos)
            if nsteps is not None:
                stats["nsteps"] = nsteps

        return SolverResult(
            times=result.tsave,
            states=states,
            expect=expect,
            final_state=(
                DynamiqsBackend._native_final_state(result, solver)
                if (options or {}).get("store_final_state", True) else None
            ),
            stats=stats,
            solver=solver,
        )

    @staticmethod
    def _effective_options(result: Any) -> dict[str, Any]:
        """Capture the native solver settings, including resolved method defaults."""
        return {**vars(result.options), "method": result.method, "gradient": result.gradient}

    @staticmethod
    def _extract_nsteps(infos: Any) -> int | None:
        """Return the maximum step count from a dynamiqs solver ``infos`` object; ``None`` if absent.

        Returns ``None`` when ``nsteps`` is a JAX tracer (i.e., inside a JIT-traced
        loss function) — extracting a concrete integer from a traced value would break
        JAX traceability.
        """
        raw = getattr(infos, "nsteps", None)
        if raw is None:
            return None
        # Guard against traced JAX values: np.asarray on a JAX tracer raises
        # TracerArrayConversionError and breaks @jax.jit / jax.grad.
        try:
            values = np.asarray(raw)
        except Exception:
            return None
        if values.size == 0:
            return None
        return int(values.max())
