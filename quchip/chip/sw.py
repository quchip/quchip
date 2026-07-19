"""Schrieffer-Wolff reduction kernels (2nd order) on bare chip blocks.

All functions are pure, ``jax.numpy``-only on the value path, and traced-safe:
no ``float()``, no Python branch on a traced value. ``H`` is the chip's bare
Hamiltonian in the C-order product basis, ordinary GHz; block masks are static
NumPy booleans (dims are static). The caller (the elimination handlers in
``quchip.chip.transformations``) owns cloning, folding, and control-plane
concerns.

The partition eliminates one mode: P = the mode in its ground state, Q =
everything else. The generator solves the Sylvester condition
``[S, H₀] = -V_offdiag`` on the P↔Q blocks, giving the standard 2nd-order
effective Hamiltonian ``H_eff = P (H + ½[S, V]) P``.

References: Bravyi, DiVincenzo & Loss, Ann. Phys. 326, 2793 (2011)
(Schrieffer-Wolff); F. Yan et al., Phys. Rev. Applied 10, 054062 (2018)
(tunable-coupler exchange J); Koch et al., PRA 76, 042319 (2007), §IV
(dispersive shift); Krantz et al., Appl. Phys. Rev. 6, 021318 (2019), §V
(Purcell decay, dispersive readout).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from quchip.utils.jax_utils import contains_tracer

if TYPE_CHECKING:
    from quchip.chip.chip import Chip

#: Matrix elements at or below this magnitude are treated as structural zeros
#: when selecting *which* entries feed a diagnostic (never on the value path).
_WORKING_PRECISION = 1e-12


def bare_hamiltonian(chip: "Chip", backend: Any) -> tuple[Any, list[str], tuple[int, ...]]:
    """Full bare Hamiltonian as a dense ``jnp`` array in GHz, with labels and dims.

    Delegates assembly to :meth:`Chip.hamiltonian` — devices plus couplings at
    the chip's RWA policy, pre-2π — and converts dense once via
    ``backend.to_array``. This is an analysis kernel, not a solver path; the
    dense conversion is the point, not a cost to avoid.
    """
    h = jnp.asarray(backend.to_array(chip.hamiltonian()), dtype=complex)
    labels = [dev.label for dev in chip.devices]
    return h, labels, tuple(chip.dims)


def mode_blocks(dims: tuple[int, ...], labels: list[str], mode_label: str) -> tuple[Any, Any]:
    """``(p_mask, q_mask)`` boolean arrays over the product basis.

    P is the eliminated mode in its ground state, Q everything else. The masks
    are static NumPy arrays (dims are static), so they can index and slice
    without touching the trace.
    """
    mode_index = labels.index(mode_label)
    occupations = np.indices(dims).reshape(len(dims), -1)
    p_mask = occupations[mode_index] == 0
    return p_mask, ~p_mask


def sylvester_generator(h: Any, p_mask: Any) -> tuple[Any, Any]:
    """Generator ``S`` solving the P↔Q Sylvester condition, plus the block-gap diagnostic.

    ``E = diag(H)`` are the bare energies and ``V = H − diag(E)``;
    ``S_ij = V_ij / (E_i − E_j)`` on the cross blocks only. The division is
    double-``where`` guarded so an exactly degenerate cross pair with no
    matrix element between it contributes zero — with a finite gradient, not
    a ``NaN`` propagated backward through the unselected branch.

    Returns
    -------
    tuple
        ``(s, min_gap)``: the (anti-Hermitian) generator, and the smallest
        ``|E_i − E_j|`` over cross entries carrying a nonzero ``V`` at
        working precision — a traced scalar, diagnostics only
        (``jnp.inf`` when no cross entry couples).
    """
    energies = jnp.real(jnp.diagonal(h))
    v = h - jnp.diag(jnp.diagonal(h))
    denom = energies[:, None] - energies[None, :]
    cross = p_mask[:, None] ^ p_mask[None, :]
    safe = jnp.where(cross & (jnp.abs(denom) > 0.0), denom, 1.0)
    s = jnp.where(cross, v / safe, 0.0)
    active = cross & (jnp.abs(v) > _WORKING_PRECISION)
    min_gap = jnp.min(jnp.where(active, jnp.abs(denom), jnp.inf))
    return s, min_gap


def h_effective_second_order(h: Any, s: Any, p_mask: Any) -> Any:
    """``H_eff = P (H + ½[S, V]) P`` restricted to the P block (dense, GHz)."""
    v = h - jnp.diag(jnp.diagonal(h))
    h_eff_full = h + 0.5 * (s @ v - v @ s)
    p_index = np.flatnonzero(p_mask)
    return h_eff_full[np.ix_(p_index, p_index)]


def basis_row(p_index: Any, labels: list[str], dims: tuple[int, ...], excited_label: str | None = None) -> int:
    """Row within the P-block ordering for the ground state, or one label's ``n=1`` occupation.

    Shared basis bookkeeping between :func:`extract_pair_parameters` and any
    caller reading out a matching row of a separately transformed P-block
    operator (e.g. a collapse operator carried through :func:`transform_collapse`).
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
    """Full bare product-basis index for the ground state, or one label's ``n=1`` occupation."""
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
    """Read survivor parameters from the P-block matrix. Pure indexing, no physics choices.

    Returns ``{survivor: {"freq_after": E(1_s) − E(0)}}`` for every survivor,
    plus ``("J", a, b): h_eff[<1_a|, |1_b>]`` for every survivor pair — the
    effective exchange between the two single-excitation states.
    """
    survivors = [lab for lab in labels if lab != mode_label]
    ground = basis_row(p_index, labels, dims)
    e_0 = jnp.real(h_eff[ground, ground])

    params: dict[Any, Any] = {}
    for surv in survivors:
        row = basis_row(p_index, labels, dims, surv)
        params[surv] = {"freq_after": jnp.real(h_eff[row, row]) - e_0}
    for i, a in enumerate(survivors):
        for b in survivors[i + 1:]:
            params[("J", a, b)] = h_eff[basis_row(p_index, labels, dims, a), basis_row(p_index, labels, dims, b)]
    return params


def transform_collapse(c_full: Any, s: Any, p_mask: Any) -> Any:
    """``c_eff = P (c + [S, c]) P`` — the 2nd-order jump-operator transform (dense).

    The same rotation that block-diagonalizes ``H`` carries the jump
    operators into the reduced frame; truncating at first order in ``S``
    matches the Hamiltonian's 2nd-order accuracy. The projection is exact
    for the spectrum but approximate for dissipation — the caller records
    that honesty note (spec §6.2). Pass the *unit* jump operator and fold
    the rate back in via :func:`purcell_rate_from`.
    """
    c_rotated = c_full + s @ c_full - c_full @ s
    p_index = np.flatnonzero(p_mask)
    return c_rotated[np.ix_(p_index, p_index)]


def purcell_rate_from(c_eff_survivor_lowering_amplitude: Any, kappa: Any) -> Any:
    """``rate = |amplitude|² · κ`` — the mediated decay a survivor inherits.

    ``amplitude`` is the survivor-lowering matrix element of the transformed
    *unit* jump operator (dimensionless, ``≈ g/Δ`` in the dispersive case);
    ``κ`` is the eliminated mode's own rate in 1/ns, so the result is
    Lindblad-ready without any further unit conversion.
    """
    return jnp.abs(c_eff_survivor_lowering_amplitude) ** 2 * kappa


def exact_reduction(chip: "Chip", mode_label: str, survivor_labels: list[str]) -> dict:
    """Exact-from-dressing reduction: labeled dressed energies instead of perturbation theory.

    Diagonalizes once through the chip's traced-safe array path and reads the
    reduced parameters off the *labeled* spectrum, so kept-block energies are
    exact to all orders — which is what ZZ needs. This is the des-Cloizeaux
    caveat in reverse: energies are exact, but the effective basis is the
    overlap-projected one, not the canonical SW rotation, so off-diagonal
    reads (``J``) agree with the perturbative route only through 2nd order.

    Returns the same parameter shape as the perturbative extraction —
    ``{survivor: {"freq_after": E(1_s) − E(0)}}`` and ``("J", a, b)`` — plus
    ``("zz", a, b) = E₁₁ − E₁₀ − E₀₁ + E₀₀`` per survivor pair (identical
    convention to :meth:`Chip.dispersive_shift`).

    Raises
    ------
    ValueError
        When two kept computational labels are assigned the same dressed
        state (concrete path only; under tracing the guard is skipped —
        labeling indices are best-effort diagnostics there, never a traced
        branch).
    """
    analysis = chip._analysis
    eigenvalues, evecs, _, labeling = analysis._compute_array_labeled()
    precomputed = (eigenvalues, labeling)
    labels = [dev.label for dev in chip.devices]

    def occupation(excited: dict[str, int]) -> tuple[int, ...]:
        occ = [0] * len(labels)
        for lab, n in excited.items():
            occ[labels.index(lab)] = n
        return tuple(occ)

    kept_tuples = [occupation({})]
    kept_tuples += [occupation({s: 1}) for s in survivor_labels]
    for i, a in enumerate(survivor_labels):
        kept_tuples += [occupation({a: 1, b: 1}) for b in survivor_labels[i + 1:]]

    if not contains_tracer(evecs):
        # Collision check independent of the assignment policy: the row-greedy
        # policy excludes taken columns, so its `duplicates` diagnostic never
        # fires — but two kept labels whose *best-overlap* dressed state
        # coincides means the bare labels have stopped meaning anything.
        evecs_np = np.asarray(evecs)
        claimed: dict[int, tuple[int, ...]] = {}
        colliding: list[tuple[int, ...]] = []
        for kept in kept_tuples:
            weights = np.abs(evecs_np[analysis._bare_label_index(kept), :]) ** 2
            best = int(np.argmax(weights))
            # No majority: the bare label's plurality dressed state holds at
            # most half the label — a 50/50 hybrid with something outside the
            # kept block (the 1e-6 absorbs eigensolver noise on exact ties).
            if weights[best] < 0.5 + 1e-6:
                colliding.append(kept)
            elif best in claimed:
                colliding += [claimed[best], kept]
            else:
                claimed[best] = kept
        if colliding:
            raise ValueError(
                f"Exact reduction of '{mode_label}' cannot label the kept block: bare states "
                f"{sorted(set(colliding))} have no majority dressed eigenstate. Near-degenerate "
                "dressed states straddle the bare labels — exactly the regime near a coupler "
                "idle point; method='sw' remains available, or shift the operating point."
            )

    def energy(excited: dict[str, int]) -> Any:
        return analysis._eigenvalue_of_label(occupation(excited), precomputed=precomputed)

    e_0 = energy({})
    params: dict[Any, Any] = {}
    for surv in survivor_labels:
        params[surv] = {"freq_after": energy({surv: 1}) - e_0}

    evecs = jnp.asarray(evecs)
    eigenvalues = jnp.asarray(eigenvalues)
    for i, a in enumerate(survivor_labels):
        for b in survivor_labels[i + 1:]:
            bare = jnp.array([analysis._bare_label_index(occupation({a: 1})),
                              analysis._bare_label_index(occupation({b: 1}))])
            dressed = jnp.stack([labeling.indices[int(bare[0])], labeling.indices[int(bare[1])]])
            # des-Cloizeaux read: the projected dressed vectors are not
            # orthonormal in the 2-dim bare subspace, so symmetric (Löwdin)
            # orthonormalization S^{-1/2} (W E W†) S^{-1/2} is required —
            # its eigenvalues are exactly the two dressed energies, and the
            # off-diagonal is the effective exchange.
            w = evecs[bare[:, None], dressed[None, :]]
            gram = w @ w.conj().T
            gram_evals, gram_evecs = jnp.linalg.eigh(gram)
            inv_sqrt = gram_evecs @ jnp.diag(gram_evals ** -0.5) @ gram_evecs.conj().T
            h_sub = inv_sqrt @ (w @ jnp.diag(eigenvalues[dressed]) @ w.conj().T) @ inv_sqrt
            h_sub = 0.5 * (h_sub + h_sub.conj().T)
            params[("J", a, b)] = h_sub[0, 1]
            params[("zz", a, b)] = jnp.real(
                energy({a: 1, b: 1}) - energy({a: 1}) - energy({b: 1}) + e_0
            )
    return params


def exact_transform_collapse(c_full: Any, evecs: Any, kept_dressed_indices: Any) -> Any:
    """``c_eff = P U† c U P`` with ``U`` the labeled eigenvector matrix (dense).

    Rotates the jump operator into the dressed basis and keeps the rows and
    columns of the kept block's assigned dressed states. Exact counterpart of
    :func:`transform_collapse`; the spectrum-vs-dissipation honesty note is
    the caller's to record either way.
    """
    u = jnp.asarray(evecs)
    c_dressed = u.conj().T @ jnp.asarray(c_full) @ u
    kept = jnp.asarray(kept_dressed_indices)
    return c_dressed[kept[:, None], kept[None, :]]


def pathway_attribution(h: Any, s: Any, p_mask: Any, i_idx: int, j_idx: int) -> list[tuple[int, Any]]:
    """Virtual-state attribution for one ``H_eff`` matrix element.

    The contribution of intermediate ``|k⟩`` to ``(½[S, V])_ij`` is
    ``½ V_ik V_kj (1/(E_i − E_k) + 1/(E_j − E_k))``, with the same
    double-``where`` guard as the generator. Returns ``(k, amount)`` pairs
    for the Q-block states carrying a nonzero path at working precision;
    under tracing the nonzero filter cannot run, so every Q state is
    returned (diagnostics remain complete either way — extra entries are
    exact zeros).
    """
    energies = jnp.real(jnp.diagonal(h))
    v = h - jnp.diag(jnp.diagonal(h))

    def guarded_inverse(gap: Any) -> Any:
        safe = jnp.where(jnp.abs(gap) > 0.0, gap, 1.0)
        return jnp.where(jnp.abs(gap) > 0.0, 1.0 / safe, 0.0)

    gap_i = energies[i_idx] - energies
    gap_j = energies[j_idx] - energies
    amounts = 0.5 * v[i_idx, :] * v[:, j_idx] * (guarded_inverse(gap_i) + guarded_inverse(gap_j))

    q_index = np.flatnonzero(~np.asarray(p_mask))
    if contains_tracer(amounts):
        return [(int(k), amounts[int(k)]) for k in q_index]
    path_strength = np.abs(np.asarray(v[i_idx, :])) * np.abs(np.asarray(v[:, j_idx]))
    return [(int(k), amounts[int(k)]) for k in q_index if path_strength[int(k)] > _WORKING_PRECISION]
