"""Band decomposition by excitation-change weight.

This module splits an operator ``O``, written in a number-basis product
representation, into disjoint bands that connect Fock states with a fixed
excitation difference. Let ``|m⟩`` denote a single-mode number state. The entry
``O_{nm} = ⟨n|O|m⟩`` contributes to the band with weight

.. math:: w \\;=\\; m - n \\;=\\; \\text{col} - \\text{row}.

For a bilinear two-body operator on modes ``a`` and ``b`` the weight is
the pair ``(Δa, Δb)``.

Physics use
-----------
* In a rotating frame, assembly attaches a carrier ``exp(−i w ω t)`` to each
  band of a drive or coupling operator, and the RWA drops the counter-rotating
  bands (Jaynes & Cummings 1963). For the structured cQED treatment, see
  Gambetta et al., *PRA* **74**, 042318 (2006). For cross-resonance
  specifically, see Rigetti & Devoret, *PRB* **81**, 134507 (2010), and Magesan
  & Gambetta, *PRA* **101**, 052308 (2020).
* Observable reconstruction uses the same weights to demodulate expectations
  back into the control frame.

Implementation
--------------
The canonical entry points keep sparse layouts (CSR / DIA) when their
structural metadata is concrete. Differentiable values can stay JAX-traced,
because the declared indices and offsets keep the sparsity pattern statically
known under ``jit``. Dense traced payloads keep every candidate band, because
they carry no equivalent structural declaration.
"""

from __future__ import annotations

from itertools import product
from math import prod
from typing import Any, cast

import numpy as np
import jax

from quchip.engine.ir import CanonicalOperator
from quchip.utils.jax_utils import (
    array_namespace as _array_namespace,
    contains_tracer,
    is_jax_array as _is_jax_array,
)

__all__ = [
    "decompose_bands",
    "decompose_canonical_bands",
    "decompose_two_body_canonical_bands",
    "canonical_to_coo",
    "canonical_to_dense_array",
    "concrete_excitation_changes",
    "local_mode_bands",
    "embed_single_mode_bands",
    "prune_zero_diagonals",
]

# A concrete band is dropped when its own Frobenius norm is at most this
# fraction of its parent operator's Frobenius norm. Scales the drop
# decision to the operator's own magnitude instead of an absolute cutoff.
# canonical_to_coo (entry-level extraction) uses exact nonzero comparison
# instead: an absolute cutoff there would strip every entry of an operator
# whose entire physical scale sits below the cutoff, defeating this
# relative test before it ever sees a band.
_BAND_NORM_RTOL = 1e-12


def _canonical_has_nonconcrete_structure(canonical: CanonicalOperator) -> bool:
    """Whether sparse indices, pointers, or offsets depend on traced values."""
    if canonical.layout == "csr":
        return contains_tracer(canonical.indices) or contains_tracer(canonical.indptr)
    if canonical.layout == "dia":
        return contains_tracer(canonical.offsets)
    return False


def _frobenius_norm(values: Any) -> float:
    """Return the concrete Frobenius (L2) norm of *values* as a Python float."""
    xp = _array_namespace(values)
    return float(np.asarray(xp.linalg.norm(values)))


def _concrete_parent_norm(values: Any) -> float | None:
    """Return a concrete pruning norm, or ``None`` for traced values."""
    if contains_tracer(values):
        return None
    if not values.size:
        return 0.0
    return _frobenius_norm(values)


def canonical_to_dense_array(canonical: CanonicalOperator) -> Any:
    """Materialize *canonical* densely, preserving its array namespace (JAX-safe).

    Alias for :meth:`CanonicalOperator.to_dense`.
    """
    # CanonicalOperator.to_dense owns the vectorized densification logic.
    return canonical.to_dense()


# Diagonals whose max absolute value falls at or below this floor are
# treated as exact algebraic cancellation (roundoff dust from assembly
# subtracting and re-adding the same coupling band), not physical
# structure. This is the module's only remaining absolute-threshold site;
# every band-drop decision elsewhere uses the relative _BAND_NORM_RTOL.
_DIAGONAL_PRUNE_THRESHOLD = 1e-15


def prune_zero_diagonals(canonical: CanonicalOperator) -> CanonicalOperator:
    """Drop concretely all-zero stored diagonals from a DIA canonical operator.

    Operator algebra that cancels terms exactly keeps the union of the
    operands' diagonal offsets, so the cancelled diagonals stay stored as
    explicit zeros. The solver applies these dead zeros at every integration
    step, and this function drops them from concrete (tracer-free) payloads.
    Traced payloads pass through unchanged, so the stored structure stays
    statically known under ``jit``.

    Non-DIA layouts pass through unchanged. CSR structure is the layout's
    sparsity declaration, and dense has no structural metadata to prune. If
    every diagonal is zero, the function keeps the first one, so the operator
    stays constructible.
    """
    # For example, assembly subtracts the lab-frame coupling from `H₀` and then
    # adds it again band-by-band as dynamic terms, which leaves cancelled
    # diagonals stored as explicit zeros.
    #
    # prune_zero_diagonals uses the absolute `_DIAGONAL_PRUNE_THRESHOLD`, not
    # the relative `_BAND_NORM_RTOL` that other band-drop sites use.
    # prune_zero_diagonals removes structure that cancelled exactly to the
    # roundoff floor, which is a fixed noise floor, not a fraction of the
    # operator's scale.
    if (
        canonical.layout != "dia"
        or contains_tracer(canonical.values)
        or _canonical_has_nonconcrete_structure(canonical)
    ):
        return canonical
    values = np.asarray(canonical.values)
    if values.shape[0] == 0:
        return canonical
    keep = [idx for idx in range(values.shape[0]) if np.abs(values[idx]).max() > _DIAGONAL_PRUNE_THRESHOLD]
    if len(keep) == values.shape[0]:
        return canonical
    if not keep:
        keep = [0]
    keep_idx = np.asarray(keep, dtype=int)
    return CanonicalOperator.from_dia(
        canonical.values[keep_idx],
        np.asarray(canonical.offsets, dtype=int)[keep_idx],
        shape=canonical.shape,
        dims=canonical.dims,
        basis=canonical.basis,
        subsystem_labels=canonical.subsystem_labels,
        tag=canonical.tag,
    )


def concrete_excitation_changes(
    matrix: Any,
    dims: tuple[int, ...],
    energy_vectors: Any | None = None,
) -> frozenset[int] | None:
    """Return the total level changes carried by a concrete operator.

    ``matrix`` acts on a product basis whose energy-ordered states are the
    columns of ``energy_vectors``; ``None`` means the product basis is already
    energy ordered. ``dims`` counts the energy levels per subsystem. Exact
    nonzero comparison can add round-off weights but never omits a band.
    Traced payloads return ``None``: their bands are not statically known.
    """
    if contains_tracer((matrix, energy_vectors)):
        return None
    values = np.asarray(matrix, dtype=complex)
    if energy_vectors is not None:
        vectors = np.asarray(energy_vectors, dtype=complex)
        values = vectors.conj().T @ values @ vectors
    levels = np.indices(tuple(dims)).reshape(len(dims), -1).sum(axis=0)
    rows, cols = np.nonzero(values)
    return frozenset(int(change) for change in np.unique(levels[cols] - levels[rows]))


def _decompose_dense_bands(
    matrix: Any,
    dims: tuple[int, ...],
    total_changes: frozenset[int] | None = None,
) -> dict[tuple[int, ...], Any]:
    """Partition by product-basis level changes; prune only concrete relative norms.

    ``total_changes`` declares which total level changes the operator can
    carry; candidate bands outside it are structurally zero and are skipped
    even when the payload is traced.
    """
    if not contains_tracer(matrix):
        matrix = np.asarray(matrix)
    xp = _array_namespace(matrix)
    states = np.stack(np.unravel_index(np.arange(prod(dims)), dims), axis=-1)
    changes = states[None, :, :] - states[:, None, :]
    zero = xp.zeros_like(matrix)
    parent_norm = _concrete_parent_norm(matrix)
    bands: dict[tuple[int, ...], Any] = {}
    for weights in product(*(range(-(dim - 1), dim) for dim in dims)):
        if total_changes is not None and sum(weights) not in total_changes:
            continue
        mask = np.all(changes == weights, axis=-1)
        band = np.where(mask, matrix, zero) if xp is np else matrix * mask
        if parent_norm is not None and _frobenius_norm(band) <= _BAND_NORM_RTOL * parent_norm:
            continue
        bands[weights] = band
    return bands


@jax.ensure_compile_time_eval()
def decompose_bands(
    op_matrix: Any,
    dim: int,
) -> dict[int, Any]:
    """Decompose a single-mode dense operator by weight ``w = col − row``.

    Returns a dict keyed by integer weight ``w ∈ [−(dim−1), dim−1]``. Each
    value is a full ``dim×dim`` matrix that holds only the entries on that
    diagonal band, with zeros elsewhere. Concrete arrays drop zero-norm bands.
    JAX-traced arrays keep every band, so the set of keys is statically known
    across traces.
    """
    if dim < 1:
        raise ValueError(f"dim must be positive, got {dim}")
    if op_matrix.shape != (dim, dim):
        raise ValueError(f"op_matrix shape {op_matrix.shape} does not match ({dim}, {dim})")

    matrix = _array_namespace(op_matrix).asarray(op_matrix)
    return {weights[0]: band for weights, band in _decompose_dense_bands(matrix, (dim,)).items()}


def _all_entries_coo(canonical: CanonicalOperator) -> tuple[Any, Any, Any]:
    """Densify and emit every ``(row, col)`` — fallback when no structural
    metadata is available to keep the COO sparsity pattern static under jit."""
    dense = canonical_to_dense_array(canonical)
    n_rows, n_cols = dense.shape
    rows = np.repeat(np.arange(n_rows, dtype=int), n_cols)
    cols = np.tile(np.arange(n_cols, dtype=int), n_rows)
    return rows, cols, dense.reshape(-1)


def canonical_to_coo(canonical: CanonicalOperator) -> tuple[Any, Any, Any]:
    """Flatten a canonical payload to ``(rows, cols, values)`` COO arrays.

    Dispatches on layout. Concrete dense and DIA payloads drop only
    exactly-zero entries (``value != 0``, no tolerance).

    CSR is already explicitly sparse and keeps its stored ``nnz``. Traced
    payloads keep their layout-native sparsity (CSR through indices/indptr, DIA
    through offsets), so the COO size stays static under jit. The
    fanout-everything fallback fires only when the layout's *structural*
    metadata is traced, or for dense, which has no structural metadata.
    """
    # Downstream band-drop decisions use the relative `_BAND_NORM_RTOL`, and
    # that test must see every surviving entry. An absolute per-entry cutoff
    # here would strip an operator whose full physical scale is below that
    # cutoff, before the relative test sees a band.
    #
    # The CSR layout itself is the sparsity declaration, so dropping stored
    # values would discard structurally meaningful zeros.
    if canonical.layout == "dense":
        if contains_tracer(canonical.values):
            return _all_entries_coo(canonical)
        dense = np.asarray(canonical.values, dtype=complex)
        rows, cols = np.nonzero(dense != 0)
        return rows.astype(int), cols.astype(int), dense[rows, cols]

    if canonical.layout == "csr":
        values = canonical.values if _is_jax_array(canonical.values) else np.asarray(canonical.values, dtype=complex)
        indices = np.asarray(canonical.indices, dtype=int)
        indptr = np.asarray(canonical.indptr, dtype=int)
        rows = np.repeat(np.arange(canonical.shape[0], dtype=int), np.diff(indptr))
        return rows, indices, values

    # DIA layout: needs concrete offsets to iterate the diagonals.
    if contains_tracer(canonical.offsets):
        return _all_entries_coo(canonical)

    offsets = np.asarray(canonical.offsets, dtype=int)
    payload = canonical.values
    traced = contains_tracer(payload)
    if not traced:
        # Slice concrete diagonals on the host; each slice of a device array is a new XLA program.
        payload = np.asarray(payload)
    xp = _array_namespace(payload)
    n_rows, n_cols = canonical.shape

    all_rows: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    all_vals: list[Any] = []
    all_diags: list[np.ndarray] = []

    for diag_idx, offset in enumerate(offsets):
        col_range = np.arange(n_cols, dtype=int)
        row_range = col_range - offset
        valid = (row_range >= 0) & (row_range < n_rows)
        valid_cols = col_range[valid]
        valid_rows = row_range[valid]

        if traced:
            all_rows.append(valid_rows)
            all_cols.append(valid_cols)
            all_diags.append(np.full(valid_cols.size, diag_idx, dtype=int))
        else:
            vals = np.asarray(payload[diag_idx, valid_cols], dtype=complex)
            nonzero = vals != 0
            if np.any(nonzero):
                all_rows.append(valid_rows[nonzero])
                all_cols.append(valid_cols[nonzero])
                all_vals.append(vals[nonzero])

    if not all_rows:
        if traced:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=int), xp.zeros((0,), dtype=payload.dtype)
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0, dtype=complex)

    rows_out = np.concatenate(all_rows)
    cols_out = np.concatenate(all_cols)
    if traced:
        # One gather for every diagonal keeps the traced program small.
        values = payload[np.concatenate(all_diags), cols_out]
    else:
        values = np.concatenate(all_vals)
    return rows_out, cols_out, values


def _canonical_from_csr(
    rows: np.ndarray,
    cols: np.ndarray,
    values: Any,
    *,
    shape: tuple[int, int],
    dims: tuple[int, ...],
    basis: str,
    subsystem_labels: tuple[str, ...],
    tag: str | None,
) -> CanonicalOperator:
    """Build a CSR :class:`CanonicalOperator` from COO-style ``(rows, cols, values)`` arrays."""
    order = np.lexsort((cols, rows))
    rows_sorted = rows[order].astype(int, copy=False)
    cols_sorted = cols[order].astype(int, copy=False)
    if _is_jax_array(values):
        values_sorted = values if np.array_equal(order, np.arange(order.size)) else values[order]
    else:
        values_sorted = np.asarray(values, dtype=complex)[order]
    counts = np.bincount(rows_sorted, minlength=shape[0])
    indptr = np.zeros(shape[0] + 1, dtype=int)
    indptr[1:] = np.cumsum(counts, dtype=int)
    return CanonicalOperator.from_csr(
        values_sorted,
        cols_sorted,
        indptr,
        shape=shape,
        dims=dims,
        basis=basis,
        subsystem_labels=subsystem_labels,
        tag=tag,
    )


def _canonical_band_from_single_weight(
    weight: int,
    cols: np.ndarray,
    values: Any,
    *,
    shape: tuple[int, int],
    dims: tuple[int, ...],
    basis: str,
    subsystem_labels: tuple[str, ...],
    tag: str | None,
) -> CanonicalOperator:
    # A single-weight band occupies exactly one diagonal, so DIA is the most
    # compact representation.
    xp = _array_namespace(values)
    diag_values = xp.zeros((1, shape[1]), dtype=complex)
    if _is_jax_array(values):
        diag_values = diag_values.at[0, cols].set(values)
    else:
        diag_values[0, cols] = values
    return CanonicalOperator.from_dia(
        diag_values,
        np.asarray([weight], dtype=int),
        shape=shape,
        dims=dims,
        basis=basis,
        subsystem_labels=subsystem_labels,
        tag=tag,
    )


def _sandwich_canonical(canonical: CanonicalOperator, left: Any, right: Any) -> CanonicalOperator:
    """Densely sandwich an operator while preserving its canonical metadata."""
    return CanonicalOperator.from_dense(
        left @ canonical_to_dense_array(canonical) @ right,
        dims=canonical.dims,
        basis=canonical.basis,
        subsystem_labels=canonical.subsystem_labels,
        tag=canonical.tag,
    )


@jax.ensure_compile_time_eval()
def decompose_canonical_bands(
    canonical: CanonicalOperator,
    dim: int,
    *,
    semantic_to_solver: Any | None = None,
    total_changes: frozenset[int] | None = None,
) -> dict[int, CanonicalOperator]:
    """Decompose a canonical single-mode operator by weight ``w = col − row``.

    Chooses the most compact representation for each band:

    * Sparse payloads with concrete structure (CSR/DIA) go through the COO
      path and emit sparse bands, including when their values are traced.
    * Dense payloads, or sparse payloads with traced structure, take
      :func:`decompose_bands` and emit dense bands.

    Copies the subsystem metadata (``dims``, ``basis``, ``subsystem_labels``,
    ``tag``) onto every band, so downstream engine operations can still find each
    band's subsystem. ``total_changes`` optionally declares the weights the
    operator can carry, and the function skips other weights as structural zeros.
    """
    if dim < 1:
        raise ValueError(f"dim must be positive, got {dim}")
    if canonical.shape != (dim, dim):
        raise ValueError(f"canonical shape {canonical.shape} does not match ({dim}, {dim})")

    return {
        weights[0]: band for weights, band in _decompose_product_canonical_bands(
            canonical, (dim,), semantic_to_solver=semantic_to_solver, total_changes=total_changes,
        ).items()
    }


def decompose_two_body_canonical_bands(
    canonical: CanonicalOperator,
    dims: list[int],
    *,
    semantic_to_solver: Any | None = None,
) -> dict[tuple[int, int], CanonicalOperator]:
    """Decompose a canonical two-body operator by ``(Δa, Δb)`` per-subsystem change.

    *canonical* is in the product basis ``|i_a⟩ ⊗ |i_b⟩``, with mode ``b`` as
    the fast index. ``dims`` is ``[d_a, d_b]``. Each band has a definite
    excitation change on each mode, so assembly attaches the carrier
    ``exp(−i (Δa · ω_a + Δb · ω_b) t)``. This is the standard rotating-frame
    form for a bilinear coupling (see e.g. Magesan & Gambetta, *PRA* **101**,
    052308 (2020), Eq. (2)).
    """
    if len(dims) != 2:
        raise ValueError(f"dims must have exactly 2 entries, got {len(dims)}")
    return cast(
        dict[tuple[int, int], CanonicalOperator],
        _decompose_product_canonical_bands(
            canonical,
            dims,
            semantic_to_solver=semantic_to_solver,
        ),
    )


@jax.ensure_compile_time_eval()
def _decompose_product_canonical_bands(
    canonical: CanonicalOperator,
    dims: list[int] | tuple[int, ...],
    *,
    semantic_to_solver: Any | None = None,
    total_changes: frozenset[int] | None = None,
) -> dict[tuple[int, ...], CanonicalOperator]:
    """Decompose an N-subsystem product operator by per-subsystem level change.

    ``total_changes`` declares the total level changes the operator can carry;
    bands with any other total weight are skipped as structural zeros.
    """
    if not dims or any(dim < 1 for dim in dims):
        raise ValueError(f"dims must contain positive subsystem dimensions, got {dims}")
    total_dim = prod(dims)
    if canonical.shape != (total_dim, total_dim):
        raise ValueError(
            f"canonical shape {canonical.shape} does not match dims product "
            f"({total_dim}, {total_dim})"
        )

    if semantic_to_solver is not None:
        transform = semantic_to_solver
        semantic = _sandwich_canonical(canonical, transform.conj().T, transform)
        return {
            weights: _sandwich_canonical(band, transform, transform.conj().T)
            for weights, band in _decompose_product_canonical_bands(
                semantic,
                dims,
                total_changes=total_changes,
            ).items()
        }

    if canonical.layout == "dense" or _canonical_has_nonconcrete_structure(canonical):
        return {
            weights: CanonicalOperator.from_dense(
                values, dims=canonical.dims, basis=canonical.basis,
                subsystem_labels=canonical.subsystem_labels, tag=canonical.tag,
            )
            for weights, values in _decompose_dense_bands(
                canonical.to_dense(), tuple(dims), total_changes,
            ).items()
        }

    rows, cols, values = canonical_to_coo(canonical)
    if not contains_tracer(values):
        values = np.asarray(values, dtype=complex)
    parent_norm = _concrete_parent_norm(values)
    row_states = np.stack(np.unravel_index(rows, dims), axis=-1)
    column_states = np.stack(np.unravel_index(cols, dims), axis=-1)
    changes = column_states - row_states

    bands: dict[tuple[int, ...], CanonicalOperator] = {}
    metadata: dict[str, Any] = dict(shape=canonical.shape, dims=canonical.dims, basis=canonical.basis,
                                    subsystem_labels=canonical.subsystem_labels, tag=canonical.tag)
    groups: list[tuple[tuple[int, ...], np.ndarray]] = []
    for weights in sorted({tuple(int(value) for value in change) for change in changes}):
        if total_changes is not None and sum(weights) not in total_changes:
            continue
        positions = np.flatnonzero(np.all(changes == weights, axis=1))
        groups.append((weights, positions[np.lexsort((cols[positions], rows[positions]))]))
    if not groups:
        return bands
    # Gather every band's values at once; each band is then a static slice.
    ordered = values[np.concatenate([positions for _, positions in groups])]
    start = 0
    for weights, positions in groups:
        band_values = ordered[start:start + positions.size]
        start += positions.size
        if parent_norm is not None and _frobenius_norm(band_values) <= _BAND_NORM_RTOL * parent_norm:
            continue
        bands[weights] = (
            _canonical_band_from_single_weight(weights[0], cols[positions], band_values, **metadata)
            if len(dims) == 1 else
            _canonical_from_csr(rows[positions], cols[positions], band_values, **metadata)
        )
    return bands


# ── Local-operator → excitation-band helpers ───────────────
#
# Assembly and observable reconstruction repeatedly turn a *local* operator (on one device's
# truncated space) into its excitation-change bands. These two helpers
# capture that shared "canonicalize → decompose → sorted-by-weight"
# skeleton. ``backend`` is any object satisfying the Backend protocol
# (``to_canonical_operator`` / ``from_canonical_operator`` / ``embed``);
# the helpers stay backend-agnostic. Neither applies the ``2π`` boundary
# (assembly owns that) — they return lab-frame, ordinary-GHz band operators.


def local_mode_bands(
    backend: Any,
    local_op: Any,
    *,
    dim: int,
    label: str,
    semantic_to_solver: Any | None = None,
) -> list[tuple[int, Any]]:
    """Decompose *local_op* into ascending excitation-change bands.

    Returns ``[(weight, band_op), ...]`` ordered by ascending weight. Each
    ``band_op`` is a backend operator on the local ``dim``-sized space. It is
    *not* embedded into the full chip space and is *without* the ``2π`` factor,
    so callers add their own embedding / scaling / wrapping.
    """
    canonical = backend.to_canonical_operator(local_op).with_metadata(
        dims=(dim,),
        subsystem_labels=(label,),
    )
    bands = decompose_canonical_bands(
        canonical,
        dim,
        semantic_to_solver=semantic_to_solver,
    )
    return [
        (weight, backend.from_canonical_operator(band))
        for weight, band in sorted(bands.items(), key=lambda kv: kv[0])
    ]


def embed_single_mode_bands(
    backend: Any,
    local_op: Any,
    *,
    device_index: int,
    dim: int,
    label: str,
    dims: tuple[int, ...],
    semantic_to_solver: Any | None = None,
) -> list[tuple[int, Any]]:
    """Like :func:`local_mode_bands`, but each band is embedded into *dims*.

    Returns ``[(weight, embedded_op), ...]``, where ``embedded_op`` acts on the
    full chip Hilbert space, still in the lab frame and ordinary GHz (no
    ``2π``).
    """
    return [
        (weight, backend.embed(band_op, device_index, dims))
        for weight, band_op in local_mode_bands(
            backend,
            local_op,
            dim=dim,
            label=label,
            semantic_to_solver=semantic_to_solver,
        )
    ]


def embed_on_support(backend: Any, op: Any, support: tuple[int, ...], dims: Any) -> Any:
    """Embed a component-local operator into the full space by support arity.

    ``support`` names the device indices the operator acts on. An empty tuple
    passes an already-embedded operator through. One index dispatches to
    :meth:`Backend.embed`, and two dispatch to :meth:`Backend.embed_two_body`.
    """
    # The engine uses this single arity dispatch for chip component
    # contributions (`Chip.dynamic_contributions` /
    # `Chip.collapse_contributions`).
    if len(support) == 0:
        return op
    if len(support) == 1:
        return backend.embed(op, support[0], dims)
    if len(support) == 2:
        return backend.embed_two_body(op, support[0], support[1], dims)
    if len(set(support)) != len(support):
        raise ValueError(f"Operator support repeats a subsystem: {support!r}.")
    from quchip.backend._dims import _embed_array

    dimensions = tuple(int(value) for value in dims)
    local = backend.to_array(op)
    xp = _array_namespace(local)
    embedded = _embed_array(xp.asarray(local, dtype=complex), support, dimensions, xp)
    return backend.from_array(embedded, dims=[list(dimensions), list(dimensions)])
