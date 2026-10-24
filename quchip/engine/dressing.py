"""Assign dressed eigenvectors to bare or eigenstate references, and build dressed results.

:func:`label_eigensystem` supports greedy, global-greedy, and continuation
assignments. :class:`BareProductReference` and :class:`EigenstateReference`
supply vectors and labels, and :func:`compute_overlaps` computes their
overlaps. The assignment kernels are inspired by SuperGrad's
``compute_energy_map``. :class:`DressedResult` is the eager, label-keyed view
of one dressed eigensystem that ``Chip.dress()`` and ``EngineResult.dress()``
return.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from quchip.backend import EigensystemData
from quchip.utils.jax_utils import concrete_array_module, contains_tracer
from quchip.utils.stack import user_stacklevel

if TYPE_CHECKING:
    from quchip.engine.ir import EngineResult


@dataclass(frozen=True)
class Labeling:
    """A labeled assignment of dressed eigenstates to user-meaningful keys.

    The dataclass has only array fields and no embedded :class:`Reference`
    object, so it is a clean pytree under :func:`jax.jit`, :func:`jax.vmap`,
    and :func:`jax.grad`.

    ``indices`` is the integer assignment (non-differentiable, conceptually
    ``stop_gradient``'d). ``overlaps`` and ``margins`` are differentiable.
    Energies indexed *via* ``indices`` flow gradients through the gathered
    values, so ``eigvals[labeling.indices[k]]`` is differentiable with respect
    to parameters that affect ``eigvals``.
    """

    keys: tuple[Any, ...]
    indices: jnp.ndarray
    overlaps: jnp.ndarray
    margins: jnp.ndarray
    duplicates: jnp.ndarray


@dataclass(frozen=True)
class LabelingPath:
    """Stacked labelings along a parameter path.

    Compared with :class:`Labeling`, each array field has a leading ``n_steps``
    axis. A ``swap_events`` array shows where labels changed dressed-index
    between adjacent steps. A swap alone is only a reassignment. To distinguish
    avoided crossings from routine mode mixing, combine a swap with a margin
    collapse (``margins`` near zero) at the same step.
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

    Optional local vectors map each component's energy states into its solver
    basis, and None denotes an identity factor. The full product matrix is
    never materialized; identity references reduce directly to ``|evecs|**2``.
    """

    dims: tuple[int, ...]
    local_vectors: tuple[Any | None, ...] = ()

    def __post_init__(self) -> None:
        if self.local_vectors and len(self.local_vectors) != len(self.dims):
            raise ValueError("Product references require one local factor per subsystem.")
        for dimension, vectors in zip(self.dims, self.local_vectors):
            if vectors is not None and vectors.shape != (dimension, dimension):
                raise ValueError("Local reference factors must match their subsystem dimensions.")

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


def reference_amplitudes(reference: Any, evecs: jnp.ndarray) -> jnp.ndarray:
    """Express solver eigenvectors in the captured reference coordinates.

    Concrete inputs combine on the host, where they compile no device programs.
    """
    if isinstance(reference, BareProductReference):
        xp = concrete_array_module(evecs, reference.local_vectors)
        amplitudes = xp.asarray(evecs).reshape(*reference.dims, evecs.shape[-1])
        for axis, vectors in enumerate(reference.local_vectors):
            if vectors is not None:
                amplitudes = xp.tensordot(xp.asarray(vectors).conj().T, amplitudes, axes=(1, axis))
                amplitudes = xp.moveaxis(amplitudes, 0, axis)
        return amplitudes.reshape(evecs.shape)
    xp = concrete_array_module(evecs, reference.vectors)
    return xp.asarray(reference.vectors).conj() @ xp.asarray(evecs)


def compute_overlaps(reference: Any, evecs: jnp.ndarray) -> jnp.ndarray:
    """``|<ref_k | psi_j>|**2`` matrix, shape ``(n_labels, n_dressed)``."""
    amplitudes = reference_amplitudes(reference, evecs)
    return concrete_array_module(amplitudes).abs(amplitudes) ** 2


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

    Returns ``(indices, chosen, margins)``. This policy is cheap and
    ``jit``-clean, and matches SuperGrad's default greedy mode. Use it in a
    weak-coupling regime where each bare label has a clearly dominant dressed
    eigenstate.
    """
    indices = jnp.argmax(overlaps, axis=1)
    chosen = jnp.take_along_axis(overlaps, indices[:, None], axis=1).squeeze(-1)
    margins = _top2_margins(overlaps)
    return indices, chosen, margins


def assign_global_greedy(
    overlaps: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Global descending-overlap greedy with ``lax.scan`` and masking.

    At each step, the function assigns the ``(label, dressed_idx)`` pair with
    the highest remaining overlap. It then masks out that label's row and that
    column, so this pure-JAX implementation assigns each label and dressed
    index at most once.

    Margins are computed from the *unmasked* overlap matrix, so they describe
    per-label confidence independent of assignment order.

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

    The function visits labels (rows) in descending row-max-overlap order. Each
    label takes its highest-overlap *still-available* dressed index, and the
    function masks that column, so the result is a permutation. Each scan step is
    a single masked-row ``argmax`` over ``n_dressed`` columns, so the whole
    assignment is ``O(n_labels * n_dressed)``. For a square ``D = L**k`` overlap
    matrix, this is ``O(L**(2k))``. :func:`assign_global_greedy` is ``O(L**(3k))``
    (a full ``L**(2k)``-entry ``argmax`` at each of ``L**k`` steps). This is the
    SuperGrad / scqubits confidence-ordered greedy variant.

    **Semantics vs. global-greedy.** This function is identical to
    :func:`assign_global_greedy` in the near-permutation regime (dispersive / weak
    hybridization), where each bare label has a clearly dominant dressed
    eigenstate. The two can diverge on strongly-hybridized overlap matrices.
    Global-greedy always commits the single largest *remaining* entry, whereas
    this function visits rows in (original) row-max order. In exactly that regime
    the assignment is intrinsically ambiguous, and ``Chip.dress`` already warns
    that bare labels are approximate. For global-greedy ordering, pass
    ``policy=assign_global_greedy`` to :func:`label_eigensystem`. Row-max ties are
    broken by ascending index (``argsort``).

    Margins (``top1 - top2`` per label) come from the *unmasked* matrix via
    ``top_k``, so they match the margins of :func:`assign_global_greedy`.
    Requires ``n_labels <= n_dressed``.
    """
    # Computing margins via top_k avoids a full per-row sort.
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
    (argmax). Without JIT, every primitive of that scan dispatches separately
    through JAX, costing ~30 ms for a modest Hilbert space. Wrapping the core
    in :func:`jax.jit` fuses the whole scan into a single compiled XLA call
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
    counts = jnp.bincount(indices, length=n_dressed)
    duplicates = counts[indices] > 1
    return indices, chosen_overlaps, margins, duplicates


STATE_OVERLAP_WARNING = 0.9


def warn_weak_assignment(label: tuple[int, ...], overlap: Any) -> None:
    """Warn when the dressed state assigned to a label keeps less than 0.9 of its bare product.

    Every query that selects the dressed state of a label calls this. A traced
    overlap does not warn. The warning names the first caller outside quchip.

    Parameters
    ----------
    label : tuple of int
        Product label of local energy levels.
    overlap : scalar
        Squared overlap of the assigned dressed state with the bare product.
    """
    from quchip.utils.jax_utils import maybe_concrete_scalar

    value = maybe_concrete_scalar(overlap)
    if value is None or value >= STATE_OVERLAP_WARNING:
        return
    warnings.warn(
        f"Dressed state label {tuple(label)} has assignment overlap {value:.3f} (< {STATE_OVERLAP_WARNING:.3f}); "
        "the label names the assigned dressed eigenstate. Use chip.bare_state(...) for the product state.",
        UserWarning,
        stacklevel=user_stacklevel(),
    )


def gauge_phases(amplitudes: Any, indices: Any) -> Any:
    """Return the phase factor that makes each column's assigned reference amplitude real and nonnegative.

    Parameters
    ----------
    amplitudes : array_like
        ``(n_labels, n_dressed)`` amplitudes ``<ref_k|psi_j>`` of the dressed columns.
    indices : array_like
        Column assigned to each reference label.

    Notes
    -----
    The anchor of a column is its amplitude on the label assigned to it. A
    one-to-one assignment can give a column whose assigned amplitude is zero.
    When that amplitude is not above a floor, the square root of the
    floating-point epsilon, and for an unassigned column, the anchor is the
    first amplitude within a relative
    ``1e-6`` of the column's largest magnitude. The factor of a column is
    ``conj(anchor) / |anchor|``. The rule is piecewise: the phase can jump
    where the assigned amplitude crosses the floor or where the fallback
    selects another amplitude. A phase cannot fix a rotation, so an exactly
    degenerate eigenspace keeps the basis of the eigensolver.
    """
    xp = concrete_array_module(amplitudes, indices)
    amplitudes = xp.asarray(amplitudes)
    columns = amplitudes.shape[-1]
    span = xp.arange(columns)
    magnitudes = xp.abs(amplitudes)
    peak = xp.max(magnitudes, axis=0)
    fallback = amplitudes[xp.argmax(magnitudes >= (1.0 - 1e-6) * peak, axis=0), span]
    labels = xp.arange(len(indices))
    rows: Any
    assigned: Any
    if xp is np:
        rows, assigned = np.zeros(columns, dtype=int), np.zeros(columns, dtype=bool)
        rows[np.asarray(indices)] = labels
        assigned[np.asarray(indices)] = True
    else:
        rows = jnp.zeros(columns, dtype=int).at[indices].set(labels)
        assigned = jnp.zeros(columns, dtype=bool).at[indices].set(True)
    own = amplitudes[rows, span]
    floor = float(np.sqrt(np.finfo(magnitudes.dtype).eps))
    anchor = xp.where(assigned & (xp.abs(own) > floor), own, fallback)
    return xp.conj(anchor) / xp.abs(anchor)


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
        Assignment function ``overlaps -> (indices, chosen, margins)``. Default
        :func:`assign_rowwise_greedy`, the confidence-ordered row-greedy policy
        that ``Chip.dress`` runs (its sole live caller passes it explicitly).

    Notes
    -----
    The caller gets energies as ``evals[labeling.indices[k]]``. ``policy`` is a
    static Python callable resolved once at trace time, not a traced argument.
    Under ``jax.jit``, close over ``policy`` or pass it as a
    ``static_argnames`` entry. Never pass it as a dynamic (traced) argument.
    """
    overlaps = compute_overlaps(reference, evecs)
    arrays = _labeling_arrays(overlaps, policy)
    if not contains_tracer(arrays):
        # Host copies of a concrete labeling: looking labels up compiles nothing.
        arrays = tuple(np.asarray(array) for array in arrays)
    indices, chosen_overlaps, margins, duplicates = arrays
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

    Step 0 is labeled against ``initial_reference``. At each later step, the
    reference becomes the *previous step's selected eigvecs* (an
    :class:`EigenstateReference`). This continuation method accepts a stacked
    eigvec tensor (usually the output of ``vmap(eigh)`` over a parameter grid),
    so it composes with any JAX-side parameter sweep.

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


DRESS_TRACING_ERROR = (
    "Dressing returns a concrete dict-keyed DressedResult and is not "
    "traceable under jax.jit/grad/vmap. Use Chip.energy(), Chip.freq(), "
    "Chip.dispersive_shift(), or Chip.state() inside transforms — they "
    "route through the array-only kernel in quchip.engine.dressing and stay "
    "differentiable."
)


@dataclass
class DressedResult:
    """Store a frozen snapshot of a dressed-state diagonalization.

    Attributes
    ----------
    eigenvalues : array-like
        Sorted dressed eigenvalues in GHz.
    eigenstates : array-like
        Backend eigenstate objects in the same order as ``eigenvalues``, with
        the phases of ``eigenvector_matrix``. They are materialized lazily on
        first access from the underlying
        :class:`~quchip.backend.containers.EigensystemData`. The dressing and
        sweep hot path never touches them, so the backend builds per-column
        kets only on request.
    state_map : dict[tuple[int, ...], int]
        Map from each bare product-basis label (one int per device) to the
        dressed eigenstate index with the largest overlap.
    dressed_eigenvalues : dict[tuple[int, ...], Any]
        Dressed eigenvalue for each assigned bare label, the direct lookup path
        for :meth:`Chip.energy`.
    assignment_overlaps : dict[tuple[int, ...], float]
        ``|⟨bare|dressed⟩|²`` of each assignment. Values below
        ``overlap_threshold`` flag hybridization.
    hybridized_labels : tuple[tuple[int, ...], ...]
        Bare labels whose assignment quality is below ``overlap_threshold``. A
        non-empty tuple triggers a user warning at dress time.
    bare_labels : tuple[tuple[int, ...], ...]
        Labels of the product energy levels exposed by the chip's resolved
        local dimensions, in chip order.
    bare_labels_by_dressed_index : dict[int, tuple[int, ...]]
        Inverse of :attr:`state_map`: dressed index → assigned bare label.
    eigenvector_matrix : array-like or None
        Dressed eigenvectors as columns, in the resolved solver basis, used by
        :meth:`ChipAnalysis.operator_in_dressed_basis` and
        :meth:`ChipAnalysis.state_components`. Each column has a real,
        nonnegative amplitude on its assigned bare state
        (:func:`~quchip.engine.dressing.gauge_phases`).
    overlap_threshold : float
        Minimum overlap for a confident assignment.
    labeling : str
        Algorithm id (currently only ``"DE"``). It resolves to the confidence-ordered
        row-greedy overlap-matching policy run by :meth:`ChipAnalysis.dress`
        (:func:`~quchip.engine.dressing.assign_rowwise_greedy`).
    """

    eigenvalues: Any
    state_map: dict[tuple[int, ...], int]
    dressed_eigenvalues: dict[tuple[int, ...], Any]
    assignment_overlaps: dict[tuple[int, ...], float]
    hybridized_labels: tuple[tuple[int, ...], ...]
    bare_labels: tuple[tuple[int, ...], ...]
    bare_labels_by_dressed_index: dict[int, tuple[int, ...]]
    eigenvector_matrix: Any = None
    overlap_threshold: float = 0.5
    labeling: str = "DE"
    _eigensystem: EigensystemData | None = None

    @property
    def eigenstates(self) -> Any:
        """Backend eigenstate kets, materialized lazily from the eigensystem."""
        if self._eigensystem is None:
            raise RuntimeError(
                "DressedResult was constructed without an EigensystemData; "
                "eigenstates are unavailable."
            )
        return self._eigensystem.eigenstates


def materialize_dressed_result(
    eigenvalues: Any,
    eigenvector_matrix: Any,
    eigensystem: EigensystemData,
    kernel_labeling: Labeling,
    *,
    overlap_threshold: float,
    labeling: str,
    warning_stacklevel: int,
) -> DressedResult:
    """Build the eager label-keyed result shared by chip and resolved dressing."""
    if contains_tracer((kernel_labeling.indices, kernel_labeling.overlaps)):
        raise RuntimeError(DRESS_TRACING_ERROR)

    bare_labels = kernel_labeling.keys
    indices_np = np.asarray(kernel_labeling.indices)
    overlaps_np = np.asarray(kernel_labeling.overlaps)
    state_map: dict[tuple[int, ...], int] = {}
    bare_labels_by_dressed_index: dict[int, tuple[int, ...]] = {}
    assignment_overlaps: dict[tuple[int, ...], float] = {}
    dressed_eigenvalues: dict[tuple[int, ...], Any] = {}
    for k, bare_label in enumerate(bare_labels):
        index = int(indices_np[k])
        state_map[bare_label] = index
        bare_labels_by_dressed_index[index] = bare_label
        assignment_overlaps[bare_label] = float(overlaps_np[k])
        dressed_eigenvalues[bare_label] = eigenvalues[index]

    hybridized_labels = tuple(
        bare_label
        for bare_label in bare_labels
        if assignment_overlaps[bare_label] < overlap_threshold
    )
    if hybridized_labels:
        # Fixed text, so the default warning filter shows it once per call
        # site instead of once per sweep point.
        warnings.warn(
            "Strong hybridization detected during dressed-state assignment; "
            f"bare labels with overlap below {overlap_threshold} are approximate. "
            "Inspect DressedResult.hybridized_labels and "
            "DressedResult.assignment_overlaps for the affected states.",
            UserWarning,
            stacklevel=warning_stacklevel,
        )

    return DressedResult(
        eigenvalues=eigenvalues,
        state_map=state_map,
        dressed_eigenvalues=dressed_eigenvalues,
        assignment_overlaps=assignment_overlaps,
        hybridized_labels=hybridized_labels,
        bare_labels=bare_labels,
        bare_labels_by_dressed_index=bare_labels_by_dressed_index,
        eigenvector_matrix=eigenvector_matrix,
        overlap_threshold=float(overlap_threshold),
        labeling=labeling,
        _eigensystem=eigensystem,
    )


def dress_engine_result(
    result: "EngineResult",
    *,
    at_time: Any | None = None,
    overlap_threshold: float = 0.5,
    labeling: str = "DE",
) -> DressedResult:
    """Materialize the instantaneous dressed eigensystem of an engine snapshot."""
    if labeling != "DE":
        raise ValueError(f"Unsupported labeling {labeling!r}. Only 'DE' is implemented.")
    if result._dressing_context is None:
        raise RuntimeError(
            "EngineResult has no resolved dressing context; obtain it from Chip.resolve() "
            "or a built solve problem."
        )
    if result.dynamic_terms and at_time is None:
        raise ValueError(
            "A dynamic EngineResult requires dress(at_time=...); this is an "
            "instantaneous eigensystem, not a Floquet analysis."
        )
    context = result._dressing_context
    backend = context.backend
    _require_eigensystem_memory(result.dims)
    hamiltonian = result.hamiltonian().matrix(t=at_time, backend=backend)
    if contains_tracer((hamiltonian, context.reference.local_vectors)):
        raise RuntimeError(DRESS_TRACING_ERROR)
    native_hamiltonian = backend.from_array(
        hamiltonian,
        dims=[list(result.dims), list(result.dims)],
    )
    eigensystem = backend.eigensystem_data(native_hamiltonian)
    kernel_labeling = label_eigensystem(
        eigensystem.eigenvector_matrix,
        context.reference,
        policy=assign_rowwise_greedy,
    )
    eigensystem = _gauge_fixed_eigensystem(eigensystem, context.reference, kernel_labeling)
    eigenvalues = eigensystem.eigenvalues
    eigenvector_matrix = eigensystem.eigenvector_matrix
    if result.excitation_space is not None:
        from dataclasses import replace
        kernel_labeling = replace(kernel_labeling, keys=tuple(map(tuple, result.excitation_space.levels.tolist())))
    return materialize_dressed_result(
        eigenvalues,
        eigenvector_matrix,
        eigensystem,
        kernel_labeling,
        overlap_threshold=float(overlap_threshold),
        labeling=labeling,
        warning_stacklevel=4,
    )


def _gauge_fixed_eigensystem(eigensystem: EigensystemData, reference: Any, labeling: Labeling) -> EigensystemData:
    """Rephase each eigenvector so that its assigned bare amplitude is real and nonnegative.

    The kets of the backend take the same phases, so a degenerate eigenspace
    keeps the eigensolver's basis.
    """
    raw = eigensystem.eigenvector_matrix
    phases = gauge_phases(reference_amplitudes(reference, raw), labeling.indices)
    xp = concrete_array_module(raw, phases)
    vectors = xp.asarray(raw) * phases
    if isinstance(raw, jax.Array) and xp is np:
        vectors = jnp.asarray(vectors)
    factors = [complex(factor) for factor in phases] if xp is np else list(phases)

    def build_states() -> list[Any]:
        return [state * factor for state, factor in zip(eigensystem.eigenstates, factors, strict=True)]

    return EigensystemData(eigenvalues=eigensystem.eigenvalues, eigenvector_matrix=vectors,
                           _states_builder=build_states)


def _require_eigensystem_memory(dims: Sequence[int]) -> None:
    """Raise :class:`MemoryError` before a full static eigensystem that cannot fit."""
    from quchip.utils.memory import eigensystem_bytes, require_memory

    dimension = int(np.prod(dims))
    require_memory(eigensystem_bytes(dimension), task=f"Full static eigensystem (dimension {dimension})",
                   remedy="Use a conserving model with labeled sector queries or reduce the model.")


def labeled_eigensystem(engine_result: "EngineResult", backend: Any) -> tuple[Any, Any, Any, Labeling]:
    """Diagonalize a static analysis result and assign its bare product labels.

    Returns ``(eigenvalues, eigenvector_matrix, eigensystem, labeling)``;
    traced Hamiltonians stay traced. Each eigenvector follows the phase
    convention of :func:`~quchip.engine.dressing.gauge_phases`.
    """
    from quchip.engine.assembly import _analysis_matrix_ghz

    context = engine_result._dressing_context
    if context is None:
        raise RuntimeError("Resolved analysis is missing its captured dressing reference.")
    dims = list(engine_result.dims)
    _require_eigensystem_memory(dims)
    eigensystem = backend.eigensystem_data(
        backend.from_array(_analysis_matrix_ghz(engine_result), dims=[dims, dims])
    )
    labeling = label_eigensystem(eigensystem.eigenvector_matrix, context.reference, policy=assign_rowwise_greedy)
    eigensystem = _gauge_fixed_eigensystem(eigensystem, context.reference, labeling)
    return eigensystem.eigenvalues, eigensystem.eigenvector_matrix, eigensystem, labeling
