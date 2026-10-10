"""Schrieffer-Wolff reduction kernels (2nd order) on bare chip blocks.

Kernels follow their inputs: a concrete chip reduces with NumPy on the host,
and a traced chip reduces with ``jax.numpy``. A conditional host check reports
coupled singularities. No traced physics is coerced to Python scalars. ``H`` is
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

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from quchip.chip.dressing import BareProductReference, Labeling, assign_rowwise_greedy, label_eigensystem
from quchip.utils.jax_utils import concrete_array_module, contains_tracer

if TYPE_CHECKING:
    from quchip.approximations import Approximation
    from quchip.chip.chip import Chip

#: Matrix elements at or below this magnitude are treated as structural zeros
#: when selecting *which* entries feed a diagnostic (never on the value path).
_WORKING_PRECISION = 1e-12


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


def _reject_coupled_degeneracy(invalid: Any) -> None:
    if bool(invalid):
        raise ValueError(
            "Schrieffer-Wolff reduction is undefined for degenerate levels with nonzero coupling. "
            "Keep the coupled levels together or change the operating point."
        )


def sylvester_generator(h: Any, p_mask: Any) -> tuple[Any, Any]:
    """Generator ``S`` that solves the P↔Q Sylvester condition, and the block-gap diagnostic.

    ``E = diag(H)`` are the bare energies and ``V = H − diag(E)``.
    ``S_ij = V_ij / (E_i − E_j)`` on the cross blocks only. The division has a
    double-``where`` guard, so an exactly degenerate cross pair with no matrix
    element between it contributes zero with a finite gradient. It does not
    propagate a ``NaN`` backward through the unselected branch. A coupled
    degeneracy raises.

    Traced execution uses a conditional error callback and marks invalid
    outputs NaN, so an elided debug effect cannot return a plausible generator.
    Valid vmapped calls can still dispatch the predicate check to the host
    under JAX's conditional batching rule.

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
    return interaction_generator(energies, xp.where(cross, v, 0.0)), cross_block_gap(h, p_mask)


def interaction_generator(energies: Any, interaction: Any) -> Any:
    """First-order anti-Hermitian generator that removes the off-diagonal part of an interaction."""
    xp = concrete_array_module(energies, interaction)
    denom = energies[:, None] - energies[None, :]
    v = interaction - xp.diag(xp.diagonal(interaction))
    invalid = xp.any((denom == 0.0) & (v != 0.0))
    if xp is jnp:
        jax.lax.cond(
            invalid,
            lambda flag: jax.debug.callback(_reject_coupled_degeneracy, flag),
            lambda flag: None,
            invalid,
        )
    else:
        _reject_coupled_degeneracy(invalid)
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


def excitation_sectors(dims: tuple[int, ...]) -> np.ndarray:
    """Total energy-level index of every C-order product state."""
    return np.indices(tuple(dims)).reshape(len(dims), -1).sum(axis=0)


def _sector_eigh(h: Any, sectors: np.ndarray) -> tuple[Any, Any]:
    """Diagonalize each total-excitation block; return ascending eigenpairs.

    Eigenvectors have exact zeros outside their own sector, so round-off can
    neither mix sectors nor make the result depend on near-degeneracies
    between them. A concrete Hamiltonian is diagonalized on the host, where
    the differently sized blocks compile no device programs.
    """
    if not contains_tracer(h):
        h = np.asarray(h)
        eigenvalues: Any = np.zeros(h.shape[0], dtype=h.real.dtype)
        eigenvectors: Any = np.zeros(h.shape, dtype=h.dtype)
        start = 0
        for sector in np.unique(sectors):
            rows = np.flatnonzero(sectors == sector)
            columns = np.arange(start, start + rows.size)
            eigenvalues[columns], eigenvectors[np.ix_(rows, columns)] = np.linalg.eigh(h[np.ix_(rows, rows)])
            start += rows.size
        order: Any = np.argsort(eigenvalues, kind="stable")
        return eigenvalues[order], eigenvectors[:, order]
    eigenvalues = jnp.zeros(h.shape[0], dtype=h.real.dtype)
    eigenvectors = jnp.zeros(h.shape, dtype=h.dtype)
    start = 0
    for sector in np.unique(sectors):
        rows = np.flatnonzero(sectors == sector)
        columns = np.arange(start, start + rows.size)
        values, vectors = jnp.linalg.eigh(h[np.ix_(rows, rows)])
        eigenvalues = eigenvalues.at[columns].set(values)
        eigenvectors = eigenvectors.at[np.ix_(rows, columns)].set(vectors)
        start += rows.size
    order = jnp.argsort(eigenvalues)
    return eigenvalues[order], eigenvectors[:, order]


def _eigh(h: Any) -> tuple[Any, Any]:
    """Ascending eigenpairs of a Hermitian matrix, computed on the host when it is concrete."""
    return jnp.linalg.eigh(h) if contains_tracer(h) else np.linalg.eigh(np.asarray(h))


def _exact_eigensystem(h: Any, dims: tuple[int, ...], sectors: np.ndarray | None = None) -> tuple[Any, Any, Labeling]:
    """Diagonalize and label one semantic-basis Hamiltonian.

    ``sectors`` gives each product state's total excitation number when the
    Hamiltonian is known to conserve it; each sector is then diagonalized
    separately.
    """
    eigenvalues, eigenvectors = _eigh(h) if sectors is None else _sector_eigh(h, sectors)
    labeling = label_eigensystem(
        eigenvectors,
        BareProductReference(dims),
        policy=assign_rowwise_greedy,
    )
    return eigenvalues, eigenvectors, labeling


def _raise_exact_condition(invalid: Any, *, message: str) -> None:
    if bool(invalid):
        raise ValueError(message)


def _check_exact_condition(invalid: Any, message: str) -> None:
    if contains_tracer(invalid):
        # An effectful check also survives callers requesting only a gradient.
        jax.experimental.io_callback(
            partial(_raise_exact_condition, message=message), None, jax.lax.stop_gradient(invalid),
        )
    else:
        _raise_exact_condition(invalid, message=message)


def _inverse_sqrt_hermitian(matrix: Any, iterations: int = 64) -> Any:
    """Return a differentiable inverse square root of a positive-definite Hermitian matrix.

    The coupled Newton-Schulz iteration evaluates the matrix function through
    products and sums. It therefore avoids eigenvector derivatives, which are
    undefined when the matrix has repeated eigenvalues even though its inverse
    square root remains smooth. Eager and traced calls verify the defining
    residual; the fixed iteration count keeps the traced path compatible with reverse-
    mode differentiation and resolves condition numbers through ``1e8`` in
    double precision.
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
    _check_exact_condition(
        ~xp.isfinite(residual) | (residual > 1e-9),
        "The projected dressed-state Gram matrix is singular or too ill-conditioned "
        "for stable symmetric orthonormalization.",
    )
    return inverse_sqrt


@dataclass(frozen=True)
class ExactSubspace:
    """One orthonormal kept coordinate map and its exact Hamiltonian."""

    hamiltonian: Any
    embedding: Any
    energies: Any
    kept_indices: Any

    def transform_operator(self, operator: Any) -> Any:
        xp = concrete_array_module(self.embedding, operator)
        return self.embedding.conj().T @ xp.asarray(operator) @ self.embedding


def exact_subspace(eigenvalues: Any, eigenvectors: Any, kept_indices: Any, dressed_indices: Any) -> ExactSubspace:
    """Use one Lowdin map for a complete kept Hamiltonian and every operator."""
    kept = np.array(kept_indices, dtype=int, copy=True)
    kept.flags.writeable = False
    xp = concrete_array_module(eigenvalues, eigenvectors, dressed_indices)
    selected = xp.asarray(eigenvectors)[:, xp.asarray(dressed_indices)]
    energies = xp.asarray(eigenvalues)[xp.asarray(dressed_indices)]
    w = selected[kept]
    inverse_sqrt = _inverse_sqrt_hermitian(w @ w.conj().T)
    unitary = inverse_sqrt @ w
    embedding = selected @ unitary.conj().T
    hamiltonian = (unitary * energies) @ unitary.conj().T
    hamiltonian = .5 * (hamiltonian + hamiltonian.conj().T)
    return ExactSubspace(hamiltonian, embedding, energies, kept)


def exact_mode_subspace(h: Any, labels: list[str], dims: tuple[int, ...], mode_label: str,
                        survivor_labels: list[str], sectors: np.ndarray | None = None, *,
                        eigensystem: tuple[Any, Any, Labeling] | None = None) -> ExactSubspace:
    """Diagonalize one model and validate its computational label assignment.

    With ``sectors``, the Hamiltonian conserves total excitation number and the
    kept map keeps every sector separate.
    Reuse ``eigensystem`` when the caller already diagonalized and labeled ``h``.
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
    _check_exact_condition(
        duplicate | xp.any(xp.max(weights, axis=1) < .5 + 1e-6),
        f"Exact reduction of {mode_label!r} cannot label the kept block: Near-degenerate dressed states "
        "leave computational bare labels without distinct majority eigenstates. Shift the operating point "
        "or use an appropriate SW reduction.",
    )
    return exact_subspace(eigenvalues, eigenvectors, kept, xp.asarray(labeling.indices)[kept])


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


def pathway_attribution(h: Any, s: Any, p_mask: Any, i_idx: int, j_idx: int) -> list[tuple[int, Any]]:
    """Virtual-state attribution for one ``H_eff`` matrix element.

    Intermediate ``|k⟩``'s contribution to ``(½[S, V])_ij`` is
    ``½ V_ik V_kj (1/(E_i − E_k) + 1/(E_j − E_k))``, evaluated directly from the supplied generator
    and its commutator. Returns ``(k, amount)`` pairs for the Q-block states that carry a nonzero
    path at working precision. Under tracing, the nonzero filter cannot run, so every Q state is
    returned. Either way the diagnostics stay complete, because extra entries are exact zeros.
    """
    xp = concrete_array_module(h, s)
    v = h - xp.diag(xp.diagonal(h))
    amounts = 0.5 * (s[i_idx, :] * v[:, j_idx] - v[i_idx, :] * s[:, j_idx])

    q_index = np.flatnonzero(~np.asarray(p_mask))
    if contains_tracer(amounts):
        return [(int(k), amounts[int(k)]) for k in q_index]
    path_strength = np.abs(np.asarray(v[i_idx, :])) * np.abs(np.asarray(v[:, j_idx]))
    return [(int(k), amounts[int(k)]) for k in q_index if path_strength[int(k)] > _WORKING_PRECISION]
