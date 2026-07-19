"""JAX-traceable dressed-state labeling primitives.

Three layers, separated cleanly:

1. **Pure-JAX kernel.** :func:`label_eigensystem` takes ``(evals, evecs,
   reference, policy)`` and returns a :class:`Labeling`. Knows nothing
   about :class:`Chip`, devices, caching, warnings, or backends.
2. **References as data.** :class:`BareProductReference` and
   :class:`EigenstateReference` produce reference vectors and label keys.
   Overlaps are computed by one shared kernel, :func:`compute_overlaps`,
   so subspace generalization is a tensor reshape rather than a new
   subclass.
3. **Façade.** :class:`~quchip.chip.analysis.ChipAnalysis` calls the
   kernel and builds the existing :class:`DressedResult`.

Inspired by SuperGrad's ``compute_energy_map``: one-line ``jnp.argmax``
for greedy assignment, ``lax.scan`` argmax-with-masking for global
greedy, and continuation along a parameter path. Generalized so
references are pluggable as data and the path is a stacked eigvec
tensor rather than a list of chip configurations.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import lax


@dataclass(frozen=True)
class Labeling:
    """A labeled assignment of dressed eigenstates to user-meaningful keys.

    Array fields only — no embedded :class:`Reference` object — so the
    dataclass is a clean pytree under :func:`jax.jit`, :func:`jax.vmap`,
    and :func:`jax.grad`.

    ``indices`` is the integer assignment (non-differentiable, conceptually
    ``stop_gradient``'d). ``overlaps`` and ``margins`` are differentiable.
    Energies indexed *via* ``indices`` flow gradients through the gathered
    values: ``eigvals[labeling.indices[k]]`` is differentiable w.r.t.
    parameters that affect ``eigvals``.
    """

    keys: tuple[Any, ...]
    indices: jnp.ndarray
    overlaps: jnp.ndarray
    margins: jnp.ndarray
    duplicates: jnp.ndarray


@dataclass(frozen=True)
class LabelingPath:
    """Stacked labelings along a parameter path.

    Each array field carries a leading ``n_steps`` axis vs. :class:`Labeling`,
    plus a ``swap_events`` array surfacing where labels actually changed
    dressed-index between adjacent steps. A swap by itself is just a
    reassignment; combine it with a margin collapse (``margins`` near
    zero) at the same step to identify avoided crossings vs. routine
    mode mixing.
    """

    keys: tuple[Any, ...]
    indices: jnp.ndarray         # (n_steps, n_labels) int
    overlaps: jnp.ndarray        # (n_steps, n_labels) float
    margins: jnp.ndarray         # (n_steps, n_labels) float
    duplicates: jnp.ndarray      # (n_steps, n_labels) bool
    swap_events: jnp.ndarray     # (n_steps - 1, n_labels) bool

    @property
    def final(self) -> Labeling:
        """The labeling at the end of the path."""
        return Labeling(
            keys=self.keys,
            indices=self.indices[-1],
            overlaps=self.overlaps[-1],
            margins=self.margins[-1],
            duplicates=self.duplicates[-1],
        )


@dataclass(frozen=True)
class BareProductReference:
    """Bare product basis states in Kronecker order.

    Vectors are not materialized: bare products are the standard basis
    of the eigvec matrix in Kronecker order, so overlaps reduce to
    ``|evecs|**2`` directly (see :func:`compute_overlaps`).
    """

    dims: tuple[int, ...]

    @property
    def keys(self) -> tuple[tuple[int, ...], ...]:
        return tuple(itertools.product(*(range(d) for d in self.dims)))


@dataclass(frozen=True)
class EigenstateReference:
    """Reference vectors are explicit eigenvectors with attached label keys.

    Used for path continuation: the previous step's selected eigvecs become
    the next step's reference. ``vectors`` is row-major: each row is one
    reference state in the Kronecker basis.
    """

    vectors: jnp.ndarray   # complex (n_labels, dim)
    keys: tuple[Any, ...]


def compute_overlaps(reference: Any, evecs: jnp.ndarray) -> jnp.ndarray:
    """``|<ref_k | psi_j>|**2`` matrix, shape ``(n_labels, n_dressed)``.

    Specialized for :class:`BareProductReference` to skip materializing an
    identity matrix: bare products are the Kronecker standard basis, so
    ``|<k|psi_j>|**2 = |evecs[k, j]|**2``.
    """
    if isinstance(reference, BareProductReference):
        return jnp.abs(evecs) ** 2
    return jnp.abs(reference.vectors.conj() @ evecs) ** 2


def _top2_margins(overlaps: jnp.ndarray) -> jnp.ndarray:
    """``top1 - top2`` overlap per row — the shared assignment-confidence margin.

    Extracts the two largest per-row overlaps via :func:`jax.lax.top_k` and
    returns their difference. All three assignment policies report this exact
    margin from the *unmasked* overlap matrix, so it is independent of the
    assignment order. For ``n_dressed == 1`` there is no runner-up, so the
    margin is the sole (max) overlap.
    """
    if overlaps.shape[1] > 1:
        top2 = lax.top_k(overlaps, 2)[0]  # (n_labels, 2), descending
        return top2[:, 0] - top2[:, 1]
    return overlaps[:, 0]


def assign_argmax(
    overlaps: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One-shot ``argmax`` per row. Allows duplicate dressed-index assignments.

    Returns ``(indices, chosen, margins)``. Cheap and ``jit``-clean — matches
    SuperGrad's default greedy mode. Use when the chip is in a weak-coupling
    regime where each bare label has a clearly-dominant dressed eigenstate.
    """
    indices = jnp.argmax(overlaps, axis=1)
    chosen = jnp.take_along_axis(overlaps, indices[:, None], axis=1).squeeze(-1)
    margins = _top2_margins(overlaps)
    return indices, chosen, margins


def assign_global_greedy(
    overlaps: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Global descending-overlap greedy via ``lax.scan`` with masking.

    At each step, finds the ``(label, dressed_idx)`` pair with the highest
    remaining overlap, assigns it, and masks out that label's row and that
    column so neither is reused. Equivalent to the legacy Python
    ``sorted+set+dict`` path but pure JAX.

    Margins are computed from the *unmasked* overlap matrix, so they
    describe per-label confidence independent of assignment order.

    Requires ``n_labels <= n_dressed``.
    """
    n_labels, n_dressed = overlaps.shape
    if n_labels > n_dressed:
        raise ValueError(
            f"global-greedy requires n_labels ({n_labels}) <= n_dressed ({n_dressed})"
        )

    # Per-label confidence margin from the unmasked overlap matrix.
    margins = _top2_margins(overlaps)

    neg_inf = jnp.asarray(-jnp.inf, dtype=overlaps.dtype)

    def step(carry, _):
        masked, indices, chosens = carry
        flat_idx = jnp.argmax(masked.reshape(-1))
        label_idx = flat_idx // n_dressed
        dressed_idx = flat_idx % n_dressed
        chosen = masked[label_idx, dressed_idx]
        indices = indices.at[label_idx].set(dressed_idx)
        chosens = chosens.at[label_idx].set(chosen)
        masked = masked.at[label_idx, :].set(neg_inf)
        masked = masked.at[:, dressed_idx].set(neg_inf)
        return (masked, indices, chosens), None

    # ``argmax`` returns int64 under ``jax_enable_x64`` and int32 otherwise.
    # Match the carry to that dtype so the ``.at[].set`` scatter does not
    # tear (JAX has begun rejecting silent int64→int32 downcasts).
    index_dtype = jnp.argmax(jnp.zeros((1,), dtype=overlaps.dtype)).dtype
    init_indices = jnp.zeros((n_labels,), dtype=index_dtype)
    init_chosens = jnp.zeros((n_labels,), dtype=overlaps.dtype)
    (_, indices, chosens), _ = lax.scan(
        step, (overlaps, init_indices, init_chosens), xs=None, length=n_labels
    )
    return indices, chosens, margins


def assign_rowwise_greedy(
    overlaps: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Confidence-ordered row-greedy assignment — ``O(n_labels * n_dressed)``.

    Labels (rows) are visited in descending row-max-overlap order; each takes its
    highest-overlap *still-available* dressed index, then that column is masked so
    the result is a permutation. Each scan step is a single masked-row ``argmax``
    over ``n_dressed`` columns, so the whole assignment is ``O(n_labels *
    n_dressed)`` — for a square ``D = L**k`` overlap matrix that is ``O(L**(2k))``
    versus :func:`assign_global_greedy`'s ``O(L**(3k))`` (a full ``L**(2k)``-entry
    ``argmax`` at each of ``L**k`` steps). This is the SuperGrad / scqubits
    confidence-ordered greedy variant.

    **Semantics vs. global-greedy.** Identical to :func:`assign_global_greedy` in
    the near-permutation regime (dispersive / weak hybridization), where each bare
    label has a clearly dominant dressed eigenstate. They can diverge on
    strongly-hybridized overlap matrices: global-greedy always commits the single
    largest *remaining* entry, whereas this visits rows in (original) row-max
    order. In exactly that regime the assignment is intrinsically ambiguous and
    ``Chip.dress`` already warns that bare labels are approximate.
    Pass ``policy=assign_global_greedy`` to :func:`label_eigensystem` for exact
    legacy semantics. Row-max ties are broken by ascending index (``argsort``).

    Margins (``top1 - top2`` per label) come from the *unmasked* matrix via
    ``top_k``, so they match :func:`assign_global_greedy`'s margins and avoid a
    full per-row sort. Requires ``n_labels <= n_dressed``.
    """
    n_labels, n_dressed = overlaps.shape
    if n_labels > n_dressed:
        raise ValueError(
            f"rowwise-greedy requires n_labels ({n_labels}) <= n_dressed ({n_dressed})"
        )

    margins = _top2_margins(overlaps)

    # Most-confident labels first. ``argsort`` is stable, so equal row maxima are
    # resolved by ascending label index.
    order = jnp.argsort(-jnp.max(overlaps, axis=1))

    neg_inf = jnp.asarray(-jnp.inf, dtype=overlaps.dtype)
    index_dtype = jnp.argmax(jnp.zeros((1,), dtype=overlaps.dtype)).dtype

    def step(carry, label_idx):
        col_avail, indices, chosens = carry
        row = jnp.where(col_avail, overlaps[label_idx], neg_inf)
        dressed_idx = jnp.argmax(row)
        indices = indices.at[label_idx].set(dressed_idx.astype(index_dtype))
        chosens = chosens.at[label_idx].set(overlaps[label_idx, dressed_idx])
        col_avail = col_avail.at[dressed_idx].set(False)
        return (col_avail, indices, chosens), None

    init = (
        jnp.ones((n_dressed,), dtype=bool),
        jnp.zeros((n_labels,), dtype=index_dtype),
        jnp.zeros((n_labels,), dtype=overlaps.dtype),
    )
    (_, indices, chosens), _ = lax.scan(step, init, xs=order)
    return indices, chosens, margins


@partial(jax.jit, static_argnums=1)
def _labeling_arrays(
    overlaps: jnp.ndarray, policy: Callable[..., Any]
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JIT-compiled numeric core: ``overlaps -> (indices, chosen, margins, duplicates)``.

    The assignment policy runs a ``lax.scan`` (global-greedy) or sort
    (argmax). Executed **eagerly** — as it was when this lived inline in
    :func:`label_eigensystem` — every primitive of that scan dispatches
    one-at-a-time through JAX, costing ~30 ms for a modest Hilbert space
    even though the arithmetic is trivial. Wrapping the core in
    :func:`jax.jit` fuses the whole scan into a single compiled XLA call
    (compiled once per ``(shape, policy)`` and cached for the session),
    which is the dominant cost of the eager dressing path
    (``Chip.state``/``Chip.freq``/``Chip.dress``).

    ``jax.jit`` is transparent to ``grad``/``vmap``: under an outer trace
    this inlines, so differentiability and batching are unchanged.
    ``policy`` is a static argument (a function,
    hashable by identity), so distinct policies get distinct caches.
    """
    indices, chosen_overlaps, margins = policy(overlaps)
    n_dressed = overlaps.shape[1]
    one_hot = jnp.eye(n_dressed, dtype=jnp.int32)[indices]   # (n_labels, n_dressed)
    counts = one_hot.sum(axis=0)                             # (n_dressed,)
    duplicates = counts[indices] > 1
    return indices, chosen_overlaps, margins, duplicates


def label_eigensystem(
    evecs: jnp.ndarray,
    reference: Any,
    policy: Callable[
        [jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ] = assign_rowwise_greedy,
) -> Labeling:
    """Pure-JAX labeling kernel.

    Parameters
    ----------
    evecs : jnp.ndarray
        ``(dim, n_dressed)`` eigenvectors as columns.
    reference : object
        Data-only object (:class:`BareProductReference` or
        :class:`EigenstateReference`) defining what label keys mean.
    policy : callable
        Assignment function ``overlaps -> (indices, chosen, margins)``.
        Default :func:`assign_rowwise_greedy` — the confidence-ordered
        row-greedy policy that ``Chip.dress`` runs (its sole live caller
        passes it explicitly).

    Notes
    -----
    Energies are looked up by the caller as ``evals[labeling.indices[k]]``.
    ``policy`` is a static Python callable, resolved once at trace time, not
    a traced argument — under ``jax.jit`` it must be closed over or passed as
    a ``static_argnames`` entry, never as a dynamic (traced) argument.
    """
    overlaps = compute_overlaps(reference, evecs)
    indices, chosen_overlaps, margins, duplicates = _labeling_arrays(overlaps, policy)
    return Labeling(
        keys=tuple(reference.keys),
        indices=indices,
        overlaps=chosen_overlaps,
        margins=margins,
        duplicates=duplicates,
    )


def track_path(
    evals_path: jnp.ndarray,
    evecs_path: jnp.ndarray,
    initial_reference: Any,
    policy: Callable[
        [jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ] = assign_global_greedy,
) -> LabelingPath:
    """Propagate a labeling along a parameter path of stacked eigensystems.

    Step 0 is labeled against ``initial_reference``. For each subsequent
    step, the reference becomes the *previous step's selected eigvecs*
    (an :class:`EigenstateReference`). This is SuperGrad's continuation
    trick generalized: the path is a stacked eigvec tensor (typically the
    output of ``vmap(eigh)`` over a parameter grid), so it composes with
    any JAX-side parameter sweep.

    Parameters
    ----------
    evals_path : jnp.ndarray
        ``(n_steps, n_dressed)``.
    evecs_path : jnp.ndarray
        ``(n_steps, dim, n_dressed)``.
    initial_reference : object
        Bootstrap reference at step 0.
    policy : callable
        Per-step assignment function. Defaults to
        :func:`assign_global_greedy`.

    Returns
    -------
    LabelingPath
        Stacked diagnostics including ``swap_events`` per step boundary.
    """
    del evals_path  # energies are not needed for labeling; caller indexes them later.
    labeling_0 = label_eigensystem(evecs_path[0], initial_reference, policy)
    keys = labeling_0.keys

    def select_refs(evecs: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
        # evecs: (dim, n_dressed) -> reference_vectors: (n_labels, dim).
        return evecs[:, indices].T

    def step(carry_refs, evecs_k):
        ref = EigenstateReference(vectors=carry_refs, keys=keys)
        labeling_k = label_eigensystem(evecs_k, ref, policy)
        next_refs = select_refs(evecs_k, labeling_k.indices)
        return next_refs, (
            labeling_k.indices,
            labeling_k.overlaps,
            labeling_k.margins,
            labeling_k.duplicates,
        )

    refs_0 = select_refs(evecs_path[0], labeling_0.indices)
    _, (indices_rest, overlaps_rest, margins_rest, duplicates_rest) = lax.scan(
        step, refs_0, evecs_path[1:],
    )

    def stack(first: jnp.ndarray, rest: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([first[None, :], rest], axis=0)

    indices_path = stack(labeling_0.indices, indices_rest)
    overlaps_path = stack(labeling_0.overlaps, overlaps_rest)
    margins_path = stack(labeling_0.margins, margins_rest)
    duplicates_path = stack(labeling_0.duplicates, duplicates_rest)

    swap_events = indices_path[1:] != indices_path[:-1]

    return LabelingPath(
        keys=keys,
        indices=indices_path,
        overlaps=overlaps_path,
        margins=margins_path,
        duplicates=duplicates_path,
        swap_events=swap_events,
    )


def phase_fixed_transform(labeling: Labeling, evecs: jnp.ndarray) -> jnp.ndarray:
    """Phase-fixed transform from Kronecker basis to labeled dressed basis.

    Returns a complex ``(dim, n_labels)`` matrix whose columns are the
    dressed eigenstates assigned to ``labeling.keys``, with each column
    phase-rotated so its largest-magnitude component is real-positive.
    Eigh's default phase convention is gauge-arbitrary; this fix makes
    ``U`` a smooth function of physical parameters under :func:`vmap` and
    therefore safe for ``U.conj().T @ O @ U``-style dressed-basis
    transforms (see :meth:`ChipAnalysis.operator_in_dressed_basis`).

    Only meaningful when ``labeling.indices`` is a permutation. If any
    ``labeling.duplicates`` is True, two columns of ``U`` will be
    identical — caller's responsibility to guard.
    """
    U = evecs[:, labeling.indices]
    abs_U = jnp.abs(U)
    pivot_rows = jnp.argmax(abs_U, axis=0)
    n_labels = U.shape[1]
    pivots = U[pivot_rows, jnp.arange(n_labels)]
    phases = jnp.exp(-1j * jnp.angle(pivots))
    return U * phases[None, :]
