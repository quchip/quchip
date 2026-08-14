"""Band decomposition by excitation-change weight.

Given an operator ``O`` written in a number-basis product representation,
this module splits it into the disjoint bands that connect Fock states
differing by a fixed excitation change. Let ``|m⟩`` denote a number
state on a single mode; the entry ``O_{nm} = ⟨n|O|m⟩`` contributes to
the band with weight

.. math:: w \\;=\\; m - n \\;=\\; \\text{col} - \\text{row}.

For a bilinear two-body operator on modes ``a`` and ``b`` the weight is
the pair ``(Δa, Δb)``.

Physics use
-----------
* Assembly attaches a carrier ``exp(−i w ω t)`` to each band when it
  assembles a drive or coupling operator in a rotating frame, and drops
  counter-rotating bands under the RWA (Jaynes & Cummings 1963; for the
  structured cQED treatment see Gambetta et al., *PRA* **74**, 042318
  (2006); for cross-resonance specifically, Rigetti & Devoret, *PRB*
  **81**, 134507 (2010), and Magesan & Gambetta, *PRA* **101**, 052308
  (2020)).
* Observable reconstruction uses the same weights to demodulate expectations
  back into the control frame.

Implementation
--------------
The canonical entry points preserve sparse layouts (CSR / DIA) whenever
their structural metadata is concrete. Differentiable values may remain
JAX-traced because the declared indices and offsets keep the sparsity pattern
statically known under ``jit``. Dense traced payloads retain every candidate
band because they carry no equivalent structural declaration.
"""

from __future__ import annotations

from typing import Any

import numpy as np

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

    Alias for :meth:`CanonicalOperator.to_dense`, which owns the
    vectorized, array-namespace-preserving densification logic.
    """
    return canonical.to_dense()


# Diagonals whose max absolute value falls at or below this floor are
# treated as exact algebraic cancellation (roundoff dust from assembly
# subtracting and re-adding the same coupling band), not physical
# structure. This is the module's only remaining absolute-threshold site;
# every band-drop decision elsewhere uses the relative _BAND_NORM_RTOL.
_DIAGONAL_PRUNE_THRESHOLD = 1e-15


def prune_zero_diagonals(canonical: CanonicalOperator) -> CanonicalOperator:
    """Drop concretely all-zero stored diagonals from a DIA canonical operator.

    Operator algebra that cancels terms exactly (e.g. assembly subtracting the
    lab-frame coupling from ``H₀`` before re-adding it band-by-band as dynamic
    terms) leaves the union of the operands' diagonal offsets in the sum, with
    the cancelled diagonals stored as explicit zeros — dead payload the solver
    applies at every integration step. Concrete (tracer-free) payloads
    drop those diagonals here; traced payloads pass through untouched so the
    stored structure stays statically known under ``jit``. Non-DIA layouts
    pass through unchanged (CSR structure is the layout's own sparsity
    declaration; dense has no structural metadata to prune). If every diagonal
    is zero, the first one is kept so the operator stays constructible.

    Uses the absolute ``_DIAGONAL_PRUNE_THRESHOLD``, not the relative
    ``_BAND_NORM_RTOL`` other band-drop sites use: this removes structure
    that cancelled exactly to the roundoff floor, a fixed noise floor
    rather than a fraction of the operator's own scale.
    """
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


def _build_weighted_bands(
    matrix: Any,
    weights: Any,
    band_values: range,
) -> dict[int, Any]:
    """Mask *matrix* by each weight in *band_values*; drop low-norm bands for concrete arrays only.

    A band is dropped when its Frobenius norm is at most ``_BAND_NORM_RTOL``
    of *matrix*'s own Frobenius norm -- relative to the parent operator, not
    an absolute cutoff, so the test scales with the operator's own
    magnitude. Traced JAX arrays retain every band to keep the set of bands
    statically known across jit traces. Concrete (tracer-free) JAX arrays
    are inspectable and drop low-norm bands exactly like NumPy payloads.
    """
    xp = _array_namespace(matrix)
    zero = xp.zeros_like(matrix)
    parent_norm = _concrete_parent_norm(matrix)

    bands: dict[int, Any] = {}
    for weight in band_values:
        band = xp.where(weights == weight, matrix, zero)
        if parent_norm is not None and _frobenius_norm(band) <= _BAND_NORM_RTOL * parent_norm:
            continue
        bands[weight] = band
    return bands


def decompose_bands(
    op_matrix: Any,
    dim: int,
) -> dict[int, Any]:
    """Decompose a single-mode dense operator by weight ``w = col − row``.

    Returns a dict keyed by integer weight ``w ∈ [−(dim−1), dim−1]``;
    each value is a full ``dim×dim`` matrix containing only the entries
    on that diagonal band (everything else is zero). Concrete arrays
    drop zero-norm bands; JAX-traced arrays retain every band so the
    set of keys is statically known across traces.
    """
    if dim < 1:
        raise ValueError(f"dim must be positive, got {dim}")
    if op_matrix.shape != (dim, dim):
        raise ValueError(f"op_matrix shape {op_matrix.shape} does not match ({dim}, {dim})")

    xp = _array_namespace(op_matrix)
    matrix = xp.asarray(op_matrix)
    row_idx = xp.arange(dim, dtype=int)[:, None]
    col_idx = xp.arange(dim, dtype=int)[None, :]
    weights = col_idx - row_idx
    return _build_weighted_bands(matrix, weights, range(-(dim - 1), dim))


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
    exactly-zero entries (``value != 0``, no tolerance): whole-band drop
    decisions downstream use the relative ``_BAND_NORM_RTOL``, which needs
    every surviving entry -- an absolute per-entry cutoff here would strip
    an operator whose entire physical scale sits below that cutoff before
    the relative test ever sees a band. CSR is already explicitly sparse
    and its stored ``nnz`` is preserved as-is (the layout itself is the
    sparsity declaration, so dropping stored values would discard
    structurally meaningful zeros). Traced payloads keep their
    layout-native sparsity (CSR via indices/indptr, DIA via offsets)
    so the COO size stays static under jit. The fanout-everything
    fallback only fires when the layout's *structural* metadata itself
    is traced (or for dense, which has no structural metadata at all).
    """
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
    xp = _array_namespace(payload)
    n_rows, n_cols = canonical.shape

    all_rows: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    all_vals: list[Any] = []

    for diag_idx, offset in enumerate(offsets):
        col_range = np.arange(n_cols, dtype=int)
        row_range = col_range - offset
        valid = (row_range >= 0) & (row_range < n_rows)
        valid_cols = col_range[valid]
        valid_rows = row_range[valid]

        if traced:
            vals = payload[diag_idx, valid_cols]
            all_rows.append(valid_rows)
            all_cols.append(valid_cols)
            all_vals.append(vals)
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
        values = xp.concatenate(all_vals)
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
        values_sorted = values[order]
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


def decompose_canonical_bands(
    canonical: CanonicalOperator,
    dim: int,
    *,
    semantic_to_solver: Any | None = None,
) -> dict[int, CanonicalOperator]:
    """Decompose a canonical single-mode operator by weight ``w = col − row``.

    Chooses the most compact representation for each band:

    * Sparse payloads with concrete structure (CSR/DIA) go through the COO
      path and emit sparse bands, including when their values are traced.
    * Dense payloads, or sparse payloads with traced structure, take
      :func:`decompose_bands` and emit dense bands.

    Subsystem metadata (``dims``, ``basis``, ``subsystem_labels``,
    ``tag``) is copied onto every band so downstream engine operations can continue
    to reason about which subsystem each band lives on.
    """
    if dim < 1:
        raise ValueError(f"dim must be positive, got {dim}")
    if canonical.shape != (dim, dim):
        raise ValueError(f"canonical shape {canonical.shape} does not match ({dim}, {dim})")

    if semantic_to_solver is not None:
        transform = semantic_to_solver
        semantic_canonical = _sandwich_canonical(canonical, transform.conj().T, transform)
        return {
            weight: _sandwich_canonical(band, transform, transform.conj().T)
            for weight, band in decompose_canonical_bands(
                semantic_canonical,
                dim,
            ).items()
        }

    if canonical.layout == "dense" or _canonical_has_nonconcrete_structure(canonical):
        dense_bands = decompose_bands(canonical_to_dense_array(canonical), dim)
        return {
            weight: CanonicalOperator.from_dense(
                band,
                dims=canonical.dims,
                basis=canonical.basis,
                subsystem_labels=canonical.subsystem_labels,
                tag=canonical.tag,
            )
            for weight, band in dense_bands.items()
        }

    rows, cols, values = canonical_to_coo(canonical)
    parent_norm = _concrete_parent_norm(values)
    bands: dict[int, CanonicalOperator] = {}
    for weight in range(-(dim - 1), dim):
        mask = (cols - rows) == weight
        if not np.any(mask):
            continue
        positions = np.flatnonzero(mask)
        band_values = values[positions]
        if parent_norm is not None and _frobenius_norm(band_values) <= _BAND_NORM_RTOL * parent_norm:
            continue
        bands[weight] = _canonical_band_from_single_weight(
            weight,
            cols[positions],
            band_values,
            shape=canonical.shape,
            dims=canonical.dims,
            basis=canonical.basis,
            subsystem_labels=canonical.subsystem_labels,
            tag=canonical.tag,
        )
    return bands


def decompose_two_body_canonical_bands(
    canonical: CanonicalOperator,
    dims: list[int],
    *,
    semantic_to_solver: Any | None = None,
) -> dict[tuple[int, int], CanonicalOperator]:
    """Decompose a canonical two-body operator by ``(Δa, Δb)`` per-subsystem change.

    *canonical* is in the product basis ``|i_a⟩ ⊗ |i_b⟩`` with mode
    ``b`` as the fast index; ``dims`` is ``[d_a, d_b]``. Each band has
    a definite excitation change on each mode, so the carrier attached
    during assembly is ``exp(−i (Δa · ω_a + Δb · ω_b) t)`` — the standard
    rotating-frame form for a bilinear coupling (see e.g. Magesan &
    Gambetta, *PRA* **101**, 052308 (2020), Eq. (2)).
    """
    if len(dims) != 2:
        raise ValueError(f"dims must have exactly 2 entries, got {len(dims)}")
    d_a, d_b = dims
    d_total = d_a * d_b
    if canonical.shape != (d_total, d_total):
        raise ValueError(f"canonical shape {canonical.shape} does not match dims product ({d_total}, {d_total})")

    if semantic_to_solver is not None:
        transform = semantic_to_solver
        semantic_canonical = _sandwich_canonical(canonical, transform.conj().T, transform)
        return {
            weights: _sandwich_canonical(band, transform, transform.conj().T)
            for weights, band in decompose_two_body_canonical_bands(
                semantic_canonical,
                dims,
            ).items()
        }

    if canonical.layout == "dense" or _canonical_has_nonconcrete_structure(canonical):
        dense_bands = _decompose_coupling_dense(canonical_to_dense_array(canonical), dims)
        return {
            band: CanonicalOperator.from_dense(
                matrix,
                dims=canonical.dims,
                basis=canonical.basis,
                subsystem_labels=canonical.subsystem_labels,
                tag=canonical.tag,
            )
            for band, matrix in dense_bands.items()
        }

    rows, cols, values = canonical_to_coo(canonical)
    parent_norm = _concrete_parent_norm(values)
    delta_a = (cols // d_b) - (rows // d_b)
    delta_b = (cols % d_b) - (rows % d_b)

    # Single pass: group nonzero entries by the (Δa, Δb) band they belong to,
    # building only the bands that are actually populated rather than scanning
    # all (2·d_a−1)·(2·d_b−1) candidate pairs (a bilinear coupling populates a
    # handful). ``sorted`` keeps the band order identical to the old nested
    # ``range`` loop, so downstream order-sensitive consumers are unaffected.
    bands: dict[tuple[int, int], CanonicalOperator] = {}
    for band in sorted({(int(a), int(b)) for a, b in zip(delta_a.tolist(), delta_b.tolist())}):
        mask = (delta_a == band[0]) & (delta_b == band[1])
        positions = np.flatnonzero(mask)
        band_values = values[positions]
        if parent_norm is not None and _frobenius_norm(band_values) <= _BAND_NORM_RTOL * parent_norm:
            continue
        bands[band] = _canonical_from_csr(
            rows[positions],
            cols[positions],
            band_values,
            shape=canonical.shape,
            dims=canonical.dims,
            basis=canonical.basis,
            subsystem_labels=canonical.subsystem_labels,
            tag=canonical.tag,
        )
    return bands


def _decompose_coupling_dense(
    matrix: Any,
    dims: list[int],
) -> dict[tuple[int, int], Any]:
    """Dense ``(delta_a, delta_b)`` decomposition; used when the payload is dense-layout or JAX-traced.

    Drops low-norm bands by the same relative ``_BAND_NORM_RTOL`` test as
    :func:`_build_weighted_bands`, for concrete payloads only.
    """
    d_a, d_b = dims
    total_dim = d_a * d_b
    xp = _array_namespace(matrix)
    parent_norm = _concrete_parent_norm(matrix)
    flat_idx = xp.arange(total_dim, dtype=int)
    row_a = (flat_idx // d_b)[:, None]
    row_b = (flat_idx % d_b)[:, None]
    col_a = (flat_idx // d_b)[None, :]
    col_b = (flat_idx % d_b)[None, :]
    delta_a_grid = col_a - row_a
    delta_b_grid = col_b - row_b
    zero = xp.zeros_like(matrix)

    bands: dict[tuple[int, int], Any] = {}
    for delta_a in range(-(d_a - 1), d_a):
        for delta_b in range(-(d_b - 1), d_b):
            band = xp.where((delta_a_grid == delta_a) & (delta_b_grid == delta_b), matrix, zero)
            if parent_norm is not None and _frobenius_norm(band) <= _BAND_NORM_RTOL * parent_norm:
                continue
            bands[(delta_a, delta_b)] = band
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

    Returns ``[(weight, band_op), ...]`` ordered by ascending weight,
    where ``band_op`` is a backend operator on the local ``dim``-sized
    space — *not* embedded into the full chip space and *without* the
    ``2π`` factor. Callers layer their own embedding / scaling / wrapping.
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

    Returns ``[(weight, embedded_op), ...]`` where ``embedded_op`` acts on
    the full chip Hilbert space. Still in the lab frame and ordinary GHz
    (no ``2π``).
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

    ``support`` names the device indices the operator acts on: an empty
    tuple passes an already-embedded operator through, one index dispatches
    to :meth:`Backend.embed`, two to :meth:`Backend.embed_two_body`. This is
    the single arity dispatch the engine uses for chip component
    contributions (:meth:`Chip.dynamic_contributions` /
    :meth:`Chip.collapse_contributions`).
    """
    if len(support) == 0:
        return op
    if len(support) == 1:
        return backend.embed(op, support[0], dims)
    if len(support) == 2:
        return backend.embed_two_body(op, support[0], support[1], dims)
    raise ValueError(
        f"Unsupported operator support arity {len(support)}; "
        "the engine embeds 0-, 1-, and 2-body component operators."
    )
