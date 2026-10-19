"""Schrieffer-Wolff reduction kernels (2nd order) on bare chip blocks.

Kernels follow their inputs: a concrete chip reduces with NumPy on the host,
and a traced chip reduces with ``jax.numpy``. A host check reports coupled
singularities, also under tracing. No traced physics is coerced to Python scalars. ``H`` is
the chip's bare Hamiltonian in the C-order product basis, in ordinary GHz.

The partition eliminates one mode: P is the mode in its ground state, and Q is
everything else. The generator solves the Sylvester condition
``[S, H₀] = -V_offdiag`` on the P↔Q blocks, which gives the standard 2nd-order
effective Hamiltonian ``H_eff = P (H + ½[S, V]) P``.

References: Bravyi, DiVincenzo & Loss, Ann. Phys. 326, 2793 (2011)
(Schrieffer-Wolff); F. Yan et al., Phys. Rev. Applied 10, 054062 (2018)
(tunable-coupler exchange J); Koch et al., PRA 76, 042319 (2007), §IV
(dispersive shift); Krantz et al., Appl. Phys. Rev. 6, 021318 (2019), §V
(Purcell decay, dispersive readout).
"""
# Block masks are static NumPy booleans (dims are static).
#
# The caller (the elimination handlers in `quchip.chip.transformations`) owns
# cloning, folding, and control-plane concerns.

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from quchip.engine.excitation import sector_eigh
from quchip.engine.dressing import BareProductReference, Labeling, assign_rowwise_greedy, label_eigensystem
from quchip.utils.jax_utils import concrete_array_module, contains_tracer, raise_if

if TYPE_CHECKING:
    from quchip.approximations import Approximation
    from quchip.chip.chip import Chip

#: Matrix elements at or below this magnitude are treated as structural zeros
#: when selecting *which* entries feed a diagnostic (never on the value path).
_WORKING_PRECISION = 1e-12
#: Smallest largest-weight that gives a diagnostic label its majority dressed state.
_MAJORITY_WEIGHT = .5 + 1e-6
#: Largest residual ``‖X G X − I‖`` of the symmetric orthonormalization ``X = G^(-1/2)``.
_ORTHONORMALIZATION_RESIDUAL = 1e-9
_DEGENERACY_MESSAGE = ("Schrieffer-Wolff reduction is undefined for degenerate levels with nonzero coupling. "
                       "Keep the coupled levels together or change the operating point.")
_GRAM_MESSAGE = ("The projected dressed-state Gram matrix is singular or too ill-conditioned "
                 "for stable symmetric orthonormalization.")


class EliminationError(ValueError):
    """A check of :func:`~quchip.chip.transformations.eliminate` failed.

    The string form is the message of the check followed by its values. Inside
    ``jax.jit`` the check runs in a host callback, so the caller receives a JAX
    runtime error with the same text instead of this type.

    Parameters
    ----------
    message : str
        The message of the check followed by its values.
    route, check, devices, label, weight, levels, coupling, residual, threshold : optional
        Keywords that set the attributes below.

    Attributes
    ----------
    route : str
        ``"sw"`` or ``"exact"``.
    check : str
        ``"degeneracy"`` when two coupled levels have the same bare energy,
        ``"labeling"`` when a diagnostic label has no distinct majority dressed
        state, or ``"orthonormalization"`` when the kept overlap matrix cannot
        be orthonormalized stably.
    devices : tuple of str
        Device order of the occupations in ``label`` and ``levels``.
    label : tuple of int or None
        The diagnostic label that failed the labeling check.
    weight : float or None
        Its largest dressed weight.
    levels : tuple of tuple of int, or None
        The two coupled levels that failed the degeneracy check.
    coupling : complex or None
        Their coupling element in GHz.
    residual : float or None
        The orthonormalization residual.
    threshold : float or None
        The bound of the check: the smallest weight of a label, or the
        largest residual.
    """

    def __init__(self, message: str, *, route: str | None = None, check: str | None = None,
                 devices: tuple[str, ...] = (), label: tuple[int, ...] | None = None, weight: float | None = None,
                 levels: tuple[tuple[int, ...], ...] | None = None, coupling: complex | None = None,
                 residual: float | None = None, threshold: float | None = None) -> None:
        super().__init__(message)
        self.route, self.check, self.devices = route, check, devices
        self.label, self.weight = label, weight
        self.levels, self.coupling = levels, coupling
        self.residual, self.threshold = residual, threshold


def bare_hamiltonian(
    chip: "Chip",
    *,
    approximation: "Approximation | None" = None,
    include_network: bool = True,
) -> tuple[Any, list[str], tuple[int, ...]]:
    """Full bare Hamiltonian as a dense array in GHz, with labels and dims.

    This path is for analysis only. It applies the chip's approximation
    strategy, does not change the authored Hamiltonian, and intentionally
    materializes a dense matrix. Set ``include_network=False`` to omit terms
    that the network generates from its ports.
    """
    from quchip.engine.assembly import _analysis_matrix_ghz

    labels = [dev.label for dev in chip.devices]
    # Dressed-state analysis retains the complete authored Hamiltonian.
    # Reduction acts on the model selected for engine use.
    result = chip.resolve(frame="lab", approximation=approximation)
    h = _analysis_matrix_ghz(result, include_network=include_network)
    records = [result.bases[device.label] for device in chip.devices]
    transforms = [record.energy_to_solver() for record in records]
    xp = concrete_array_module(h, transforms)
    h = xp.asarray(h, dtype=complex)
    semantic_to_solver: Any | None = None
    for record, local_transform in zip(records, transforms):
        if local_transform is None:
            local_transform = xp.eye(record.resolved_dim, dtype=complex)
        semantic_to_solver = (
            xp.asarray(local_transform)
            if semantic_to_solver is None
            else xp.kron(semantic_to_solver, local_transform)
        )
    if semantic_to_solver is not None:
        h = semantic_to_solver.conj().T @ h @ semantic_to_solver
    return h, labels, result.dims


def mode_blocks(dims: tuple[int, ...], labels: list[str], mode_label: str) -> tuple[Any, Any]:
    """``(p_mask, q_mask)`` boolean arrays over the product basis.

    P is the eliminated mode in its ground state, and Q is everything else. The
    masks are static NumPy arrays (dims are static), so they can index and
    slice without affecting the trace.
    """
    mode_index = labels.index(mode_label)
    occupations = np.indices(dims).reshape(len(dims), -1)
    p_mask = occupations[mode_index] == 0
    return p_mask, ~p_mask


def cross_block_gap(h: Any, p_mask: Any) -> Any:
    """Smallest coupled P/Q bare-energy gap at diagnostic working precision."""
    xp = concrete_array_module(h)
    energies = xp.real(xp.diagonal(h))
    cross = p_mask[:, None] ^ p_mask[None, :]
    active = cross & (xp.abs(h) > _WORKING_PRECISION)
    return xp.min(xp.where(active, xp.abs(energies[:, None] - energies[None, :]), xp.inf))


def _coupled_degeneracy(row: Any, column: Any, coupling: Any, energy: Any) -> Exception:
    return ValueError(_DEGENERACY_MESSAGE)


def degeneracy_error(labels: Any, dims: tuple[int, ...]) -> Any:
    """Return the error of a coupled degeneracy that names its levels in the product basis of ``labels``."""
    devices = tuple(labels)

    def error(row: Any, column: Any, coupling: Any, energy: Any) -> Exception:
        levels = tuple(tuple(int(level) for level in np.unravel_index(int(index), dims)) for index in (row, column))
        element = complex(coupling)
        return EliminationError(
            f"{_DEGENERACY_MESSAGE} Levels {levels[0]} and {levels[1]} of {devices} have the same bare energy, "
            f"{float(energy):.6g} GHz, and a coupling element of magnitude {abs(element):.3g} GHz.",
            route="sw", check="degeneracy", devices=devices, levels=levels, coupling=element,
        )

    return error


def sylvester_generator(h: Any, p_mask: Any, *, error: Any = None) -> tuple[Any, Any]:
    """Generator ``S`` that solves the P↔Q Sylvester condition, and the block-gap diagnostic.

    ``E = diag(H)`` are the bare energies and ``V = H − diag(E)``.
    ``S_ij = V_ij / (E_i − E_j)`` on the cross blocks only. The division has a
    double-``where`` guard, so an exactly degenerate cross pair with no matrix
    element between it contributes zero with a finite gradient. It does not
    propagate a ``NaN`` backward through the unselected branch. A coupled
    degeneracy raises ``error(row, column, coupling, energy)``, by default a
    ``ValueError``; :func:`degeneracy_error` names the levels.

    Traced execution checks through a host callback that also survives callers
    requesting only a gradient, and marks invalid outputs NaN.

    Returns
    -------
    tuple
        ``(s, min_gap)``: the (anti-Hermitian) generator, and the smallest
        ``|E_i − E_j|`` over cross entries that carry a nonzero ``V`` at
        working precision. The gap is a traced scalar for diagnostics only
        (``jnp.inf`` when no cross entry couples).
    """
    xp = concrete_array_module(h)
    energies = xp.real(xp.diagonal(h))
    v = h - xp.diag(xp.diagonal(h))
    cross = p_mask[:, None] ^ p_mask[None, :]
    return interaction_generator(energies, xp.where(cross, v, 0.0), error=error), cross_block_gap(h, p_mask)


def interaction_generator(energies: Any, interaction: Any, *, error: Any = None) -> Any:
    """First-order anti-Hermitian generator that removes the off-diagonal part of an interaction.

    A coupled degeneracy raises ``error(row, column, coupling, energy)`` for its first pair, by default a
    ``ValueError`` (:func:`sylvester_generator`).
    """
    xp = concrete_array_module(energies, interaction)
    denom = energies[:, None] - energies[None, :]
    v = interaction - xp.diag(xp.diagonal(interaction))
    degenerate = (denom == 0.0) & (v != 0.0)
    invalid = xp.any(degenerate)
    first = xp.argmax(degenerate.ravel())
    row, column = first // v.shape[1], first % v.shape[1]
    raise_if(invalid, error or _coupled_degeneracy, row, column, v.ravel()[first], energies[row])
    s = v / xp.where(denom != 0.0, denom, 1.0)
    return xp.where(invalid, xp.full_like(s, xp.nan), s)


def h_effective_second_order(h: Any, s: Any, p_mask: Any) -> Any:
    """``H_eff = P (H + ½[S, V]) P`` restricted to the P block (dense, GHz)."""
    xp = concrete_array_module(h, s)
    v = h - xp.diag(xp.diagonal(h))
    h_eff_full = h + 0.5 * (s @ v - v @ s)
    p_index = np.flatnonzero(p_mask)
    return h_eff_full[np.ix_(p_index, p_index)]


def basis_row(p_index: Any, labels: list[str], dims: tuple[int, ...], excited_label: str | None = None) -> int:
    """Row in the P-block ordering for the ground state, or for the ``n=1`` occupation of one label.

    Shared basis bookkeeping between :func:`extract_pair_parameters` and any
    caller that reads a matching row of a separately transformed P-block
    operator.
    """
    occupations = np.array(np.unravel_index(np.asarray(p_index), dims))
    occ = [0] * len(dims)
    if excited_label is not None:
        occ[labels.index(excited_label)] = 1
    target = tuple(occ)
    for row in range(occupations.shape[1]):
        if tuple(occupations[:, row]) == target:
            return row
    raise KeyError(target)


def bare_index(labels: list[str], dims: tuple[int, ...], excited_label: str | None = None) -> int:
    """Full bare product-basis index for the ground state, or for the ``n=1`` occupation of one label."""
    occ = [0] * len(dims)
    if excited_label is not None:
        occ[labels.index(excited_label)] = 1
    return int(np.ravel_multi_index(tuple(occ), dims))


def extract_pair_parameters(
    h_eff: Any,
    p_index: Any,
    labels: list[str],
    dims: tuple[int, ...],
    mode_label: str,
) -> dict:
    """Read survivor parameters from the P-block matrix by pure indexing, with no physics choices.

    Returns ``{survivor: {"freq_after": E(1_s) − E(0)}}`` for every survivor,
    and ``("J", a, b): h_eff[<1_a|, |1_b>]`` for every survivor pair, the
    effective exchange between the two single-excitation states.
    """
    xp = concrete_array_module(h_eff)
    survivors = [lab for lab in labels if lab != mode_label]
    ground = basis_row(p_index, labels, dims)
    e_0 = xp.real(h_eff[ground, ground])

    params: dict[Any, Any] = {}
    for surv in survivors:
        row = basis_row(p_index, labels, dims, surv)
        params[surv] = {"freq_after": xp.real(h_eff[row, row]) - e_0}
    for i, a in enumerate(survivors):
        for b in survivors[i + 1:]:
            params[("J", a, b)] = h_eff[basis_row(p_index, labels, dims, a), basis_row(p_index, labels, dims, b)]
    return params


def _eigh(h: Any) -> tuple[Any, Any]:
    """Ascending eigenpairs of a Hermitian matrix, computed on the host when it is concrete."""
    return jnp.linalg.eigh(h) if contains_tracer(h) else np.linalg.eigh(np.asarray(h))


def _exact_eigensystem(h: Any, dims: tuple[int, ...], sectors: np.ndarray | None = None) -> tuple[Any, Any, Labeling]:
    """Diagonalize and label one semantic-basis Hamiltonian.

    ``sectors`` gives each product state's total excitation number when the
    Hamiltonian is known to conserve it; each sector is then diagonalized
    separately.
    """
    eigenvalues, eigenvectors = _eigh(h) if sectors is None else sector_eigh(h, sectors)
    labeling = label_eigensystem(
        eigenvectors,
        BareProductReference(dims),
        policy=assign_rowwise_greedy,
    )
    return eigenvalues, eigenvectors, labeling


def _ill_conditioned_gram(residual: Any) -> Exception:
    return ValueError(_GRAM_MESSAGE)


def _orthonormalization_error(residual: Any) -> Exception:
    value = float(residual)
    detail = (f"Its residual is {value:.3g}, above the threshold {_ORTHONORMALIZATION_RESIDUAL:.0e}."
              if np.isfinite(value) else "Its residual is not finite.")
    return EliminationError(
        f"{_GRAM_MESSAGE} {detail}",
        route="exact", check="orthonormalization", residual=value, threshold=_ORTHONORMALIZATION_RESIDUAL,
    )


def _inverse_sqrt_hermitian(matrix: Any, iterations: int = 64, *, error: Any = None) -> tuple[Any, Any]:
    """Return a differentiable inverse square root of a positive-definite Hermitian matrix, and its residual.

    The coupled Newton-Schulz iteration evaluates the matrix function through
    products and sums. It therefore avoids eigenvector derivatives, which are
    undefined when the matrix has repeated eigenvalues even though its inverse
    square root remains smooth. Eager and traced calls verify the defining
    residual; the fixed iteration count keeps the traced path compatible with reverse-
    mode differentiation and resolves condition numbers through ``1e8`` in
    double precision. A residual ``‖X M X − I‖`` above ``1e-9``, or a
    non-finite one, raises ``error(residual)``, by default a ``ValueError``.
    """
    xp = concrete_array_module(matrix)
    matrix = 0.5 * (matrix + matrix.conj().T)
    scale = xp.linalg.norm(matrix, ord="fro")
    identity = xp.eye(matrix.shape[0], dtype=matrix.dtype)
    y = matrix / scale
    z = identity
    for _ in range(iterations):
        correction = 0.5 * (3.0 * identity - z @ y)
        y = y @ correction
        z = correction @ z
    inverse_sqrt = z / xp.sqrt(scale)
    inverse_sqrt = 0.5 * (inverse_sqrt + inverse_sqrt.conj().T)
    residual = xp.linalg.norm(inverse_sqrt @ matrix @ inverse_sqrt - identity)
    raise_if(~xp.isfinite(residual) | (residual > _ORTHONORMALIZATION_RESIDUAL), error or _ill_conditioned_gram,
             residual)
    return inverse_sqrt, residual


@dataclass(frozen=True)
class ExactSubspace:
    """One orthonormal kept coordinate map and its exact Hamiltonian.

    ``gram`` is the overlap matrix of the kept dressed states on the kept bare
    states, and ``residual`` the error of its symmetric orthonormalization.
    ``label_margins`` holds the labeling report of :func:`exact_mode_subspace`.
    """

    hamiltonian: Any
    embedding: Any
    energies: Any
    kept_indices: Any
    gram: Any = None
    residual: Any = None
    label_margins: dict | None = None

    def transform_operator(self, operator: Any) -> Any:
        xp = concrete_array_module(self.embedding, operator)
        return self.embedding.conj().T @ xp.asarray(operator) @ self.embedding


def exact_subspace(eigenvalues: Any, eigenvectors: Any, kept_indices: Any, dressed_indices: Any, *,
                   error: Any = None) -> ExactSubspace:
    """Use one Lowdin map for a complete kept Hamiltonian and every operator.

    An unstable orthonormalization raises ``error(residual)``, by default a ``ValueError``.
    """
    kept = np.array(kept_indices, dtype=int, copy=True)
    kept.flags.writeable = False
    xp = concrete_array_module(eigenvalues, eigenvectors, dressed_indices)
    selected = xp.asarray(eigenvectors)[:, xp.asarray(dressed_indices)]
    energies = xp.asarray(eigenvalues)[xp.asarray(dressed_indices)]
    w = selected[kept]
    gram = w @ w.conj().T
    inverse_sqrt, residual = _inverse_sqrt_hermitian(gram, error=error)
    unitary = inverse_sqrt @ w
    embedding = selected @ unitary.conj().T
    hamiltonian = (unitary * energies) @ unitary.conj().T
    hamiltonian = .5 * (hamiltonian + hamiltonian.conj().T)
    return ExactSubspace(hamiltonian, embedding, energies, kept, gram, residual)


def exact_mode_subspace(h: Any, labels: list[str], dims: tuple[int, ...], mode_label: str,
                        survivor_labels: list[str], sectors: np.ndarray | None = None, *,
                        eigensystem: tuple[Any, Any, Labeling] | None = None) -> ExactSubspace:
    """Diagonalize one model and validate its computational label assignment.

    With ``sectors``, the Hamiltonian conserves total excitation number and the
    kept map keeps every sector separate.
    Reuse ``eigensystem`` when the caller already diagonalized and labeled ``h``.

    The diagnostic labels are the ground state and the single and pair
    excitations of the survivors. Each needs a distinct majority dressed state,
    with a largest weight of at least ``0.5 + 1e-6``, and the kept overlap
    matrix needs a stable orthonormalization. A failed check raises
    :class:`EliminationError`. ``label_margins`` of the result reports, for
    each diagnostic label in the occupation order of ``labels``, its largest
    weight, the runner-up and the margin to that threshold, and the smallest
    eigenvalue of the overlap matrix, the residual and its margin to ``1e-9``.
    """
    eigenvalues, eigenvectors, labeling = _exact_eigensystem(h, dims, sectors) if eigensystem is None else eigensystem
    p_mask, _ = mode_blocks(dims, labels, mode_label)
    kept = np.flatnonzero(p_mask)
    ground = [0] * len(labels)
    diagnostics = [tuple(ground)]
    for index, label in enumerate(survivor_labels):
        single = ground.copy()
        single[labels.index(label)] = 1
        diagnostics.append(tuple(single))
        for other in survivor_labels[index + 1:]:
            pair = single.copy()
            pair[labels.index(other)] = 1
            diagnostics.append(tuple(pair))
    rows = np.array([np.ravel_multi_index(occupation, dims) for occupation in diagnostics])
    xp = concrete_array_module(eigenvectors, labeling.indices)
    weights = xp.abs(eigenvectors[rows]) ** 2
    best = xp.argmax(weights, axis=1)
    duplicate = xp.any(xp.triu(best[:, None] == best[None, :], k=1))
    ranked = xp.sort(weights, axis=1)
    largest, runner_up = ranked[:, -1], ranked[:, -2]
    raise_if(duplicate | xp.any(largest < _MAJORITY_WEIGHT),
             partial(_labeling_error, mode_label, tuple(labels), diagnostics), largest, best)
    subspace = exact_subspace(eigenvalues, eigenvectors, kept, xp.asarray(labeling.indices)[kept],
                              error=_orthonormalization_error)
    margins = largest - _MAJORITY_WEIGHT
    residual_margin = _ORTHONORMALIZATION_RESIDUAL - subspace.residual
    report = {
        "labels": {occupation: {"weight": largest[row], "runner_up": runner_up[row], "margin": margins[row]}
                   for row, occupation in enumerate(diagnostics)},
        "threshold": _MAJORITY_WEIGHT,
        "margin": xp.min(margins),
        "gram_min_eigenvalue": xp.linalg.eigvalsh(subspace.gram)[0],
        "residual": subspace.residual,
        "residual_threshold": _ORTHONORMALIZATION_RESIDUAL,
        "residual_margin": residual_margin,
        "is_valid": (xp.min(margins) >= 0) & (residual_margin >= 0) & ~duplicate,
    }
    return replace(subspace, label_margins=report)


def _labeling_error(mode_label: str, devices: tuple[str, ...], diagnostics: list[tuple[int, ...]],
                    largest: Any, best: Any) -> Exception:
    largest, best = np.asarray(largest), np.asarray(best)
    row = int(np.argmin(largest))
    if largest[row] >= _MAJORITY_WEIGHT:
        # Every label has a majority, so two of them share one dressed state.
        row = next(index for index in range(len(best)) if np.count_nonzero(best[:index] == best[index]))
        detail = f"Label {diagnostics[row]} of {devices} shares its majority dressed state with another label."
    else:
        detail = (f"Label {diagnostics[row]} of {devices} has largest dressed weight {largest[row]:.7g}, "
                  f"below the threshold {_MAJORITY_WEIGHT:.7g}.")
    return EliminationError(
        f"Exact reduction of {mode_label!r} cannot label the kept block: Near-degenerate dressed states "
        "leave computational bare labels without distinct majority eigenstates. Shift the operating point "
        f"or use an appropriate SW reduction. {detail}",
        route="exact", check="labeling", devices=devices, label=diagnostics[row], weight=float(largest[row]),
        threshold=_MAJORITY_WEIGHT,
    )


def exact_pair_parameters(subspace: ExactSubspace, labels: list[str], dims: tuple[int, ...], mode_label: str,
                          survivor_labels: list[str]) -> dict:
    """Report labeled energies and exchange entries from the complete kept model."""
    xp = concrete_array_module(subspace.energies)
    params = extract_pair_parameters(subspace.hamiltonian, subspace.kept_indices, labels, dims, mode_label)
    rows = {int(index): row for row, index in enumerate(subspace.kept_indices)}

    def energy(*excited: str) -> Any:
        occupation = [int(label in excited) for label in labels]
        return subspace.energies[rows[int(np.ravel_multi_index(tuple(occupation), dims))]]

    ground = energy()
    for index, label in enumerate(survivor_labels):
        params[label]["freq_after"] = energy(label) - ground
        for other in survivor_labels[index + 1:]:
            params[("zz", label, other)] = xp.real(energy(label, other) - energy(label) - energy(other) + ground)
    return params


def _generator_entries(couplings: Any, gaps: Any) -> Any:
    """Return ``V / ΔE`` with the guard of :func:`interaction_generator`: zero for an uncoupled zero gap, NaN for a
    coupled one."""
    xp = concrete_array_module(couplings, gaps)
    ratio = couplings / xp.where(gaps != 0.0, gaps, 1.0)
    return xp.where((gaps == 0.0) & (couplings != 0.0), xp.nan, ratio)


def pathway_attribution(h: Any, s: Any, p_mask: Any, i_idx: int, j_idx: int) -> list[tuple[int, Any]]:
    """Virtual-state attribution for one ``H_eff`` matrix element.

    Intermediate ``|k⟩``'s contribution to ``(½[S, V])_ij`` is
    ``½ V_ik V_kj (1/(E_i − E_k) + 1/(E_j − E_k))``, evaluated from the supplied generator and its
    commutator. With ``s=None``, the two generator rows come from ``h``. Returns ``(k, amount)``
    pairs for every Q-block state with a nonzero contribution, so the amounts sum to the second-order
    part ``(H_eff − PHP)_ij``. Under tracing, the nonzero filter cannot run, so every Q state is
    returned, and the extra entries are exact zeros.
    """
    xp = concrete_array_module(h, s)
    v = h - xp.diag(xp.diagonal(h))
    q_index = np.flatnonzero(~np.asarray(p_mask))
    if s is None:
        energies = xp.real(xp.diagonal(h))
        left = _generator_entries(v[i_idx, q_index], energies[i_idx] - energies[q_index])
        right = _generator_entries(v[q_index, j_idx], energies[q_index] - energies[j_idx])
    else:
        left, right = s[i_idx, q_index], s[q_index, j_idx]
    amounts = 0.5 * (left * v[q_index, j_idx] - v[i_idx, q_index] * right)
    if contains_tracer(amounts):
        return [(int(k), amounts[row]) for row, k in enumerate(q_index)]
    return [(int(q_index[row]), amounts[row]) for row in np.flatnonzero(np.asarray(amounts) != 0)]
