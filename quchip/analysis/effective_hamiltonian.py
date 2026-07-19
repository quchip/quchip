"""Static-ZZ pathway attribution and des-Cloizeaux effective-Hamiltonian extraction.

Two analysis primitives built on the dressed-state and Schrieffer-Wolff (SW)
machinery of :mod:`quchip.chip.sw` and :mod:`quchip.chip.analysis`:

analyze_static_zz
    :func:`analyze_static_zz` — the exact residual ZZ between two devices
    (:meth:`~quchip.chip.chip.Chip.dispersive_shift`, unchanged), plus a
    2nd-order SW attribution of the virtual states mediating it. The exact
    ``zz`` value is the "ZZ = 0 while J stays on target" loss primitive of a
    calibration sweep; ``pathways`` is a diagnostic for which virtual
    transition dominates it.
effective_hamiltonian
    :func:`effective_hamiltonian` — the des-Cloizeaux effective Hamiltonian on
    a user-chosen computational subspace: dense, GHz, with eigenvalues that
    are exactly the labeled dressed energies of the requested bare states.

Both routes are algebra on the chip's exact dressed spectrum (no perturbative
truncation for the reported ``zz`` or ``effective_hamiltonian`` eigenvalues),
so they stay differentiable end-to-end whenever the chip's backend supports
differentiating through its eigensolver.

References
----------
Bravyi, DiVincenzo & Loss, *Schrieffer-Wolff transformation for quantum
many-body systems*, Ann. Phys. 326, 2793 (2011).
Blais et al., *Circuit quantum electrodynamics*, RMP 93, 025005 (2021), §IV.C
(static ZZ) and the des-Cloizeaux perturbation-theory appendix convention for
projecting onto a computational subspace.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from quchip.chip.sw import bare_hamiltonian, pathway_attribution, sylvester_generator
from quchip.devices.base import BaseDevice
from quchip.utils.jax_utils import contains_tracer, maybe_concrete_scalar

if TYPE_CHECKING:
    from quchip.chip.chip import Chip


# ---------------------------------------------------------------------------
# analyze_static_zz
# ---------------------------------------------------------------------------


def _computational_p_mask(dims: tuple[int, ...], idx_a: int, idx_b: int) -> np.ndarray:
    """Boolean mask over the product basis selecting the four computational states of ``(a, b)``.

    ``P`` is ``{|0,0>, |0,1>, |1,0>, |1,1>}`` of the ``(a, b)`` pair with
    every other device grounded; ``Q`` is everything else. This is the
    partition :func:`~quchip.chip.sw.sylvester_generator` needs to attribute
    the ZZ-carrying ``(1_a, 1_b)`` diagonal element — unlike
    :func:`~quchip.chip.sw.mode_blocks`, no single mode is eliminated here.
    """
    occupations = np.indices(dims).reshape(len(dims), -1)
    spectators_grounded = np.ones(occupations.shape[1], dtype=bool)
    for i in range(len(dims)):
        if i not in (idx_a, idx_b):
            spectators_grounded &= occupations[i] == 0
    computational = (occupations[idx_a] <= 1) & (occupations[idx_b] <= 1)
    return spectators_grounded & computational


def _bare_index_pair(dims: tuple[int, ...], idx_a: int, idx_b: int, level_a: int, level_b: int) -> int:
    """Full product-basis index for ``(a, b)`` at the given levels, spectators grounded."""
    occ = [0] * len(dims)
    occ[idx_a] = level_a
    occ[idx_b] = level_b
    return int(np.ravel_multi_index(tuple(occ), dims))


@dataclass(frozen=True)
class StaticZZResult:
    """Store exact static ZZ between two devices, plus its 2nd-order SW pathway attribution.

    Attributes
    ----------
    zz
        ``E(1,1) - E(1,0) - E(0,1) + E(0,0)``, identical to
        :meth:`~quchip.chip.chip.Chip.dispersive_shift(device_a, device_b)` —
        exact, not perturbative.
    pathways
        ``(bare_occupation, amount)`` pairs: the contribution of each virtual
        intermediate state to the 2nd-order SW correction of the ``(1_a,
        1_b)`` diagonal matrix element, ``amount = 1/2 * V_ik*V_ki*(1/(E_i -
        E_k) + 1/(E_i - E_k))`` for ``i`` the ``(1_a, 1_b)`` bare index —
        a decomposition of that one energy correction, not of ``zz`` itself
        (which combines four dressed energies exactly). ``bare_occupation``
        is a full chip-length Fock tuple, in device order.
    device_a, device_b
        Resolved device labels.
    device_labels
        Chip device labels in tensor-product order, for reading
        ``pathways``' occupation tuples.

    Amounts stay in the array namespace of the chip's parameters (JAX in, JAX
    out) — traceable and differentiable, precision-filtered only on the
    concrete path (:func:`~quchip.chip.sw.pathway_attribution`).
    """

    zz: Any
    pathways: list[tuple[tuple[int, ...], Any]]
    device_a: str
    device_b: str
    device_labels: tuple[str, ...]

    def describe(self) -> str:
        """Print and return the exact ZZ plus the leading virtual pathways.

        Concrete values only — call outside ``jax.jit``/``grad`` regions;
        traced amounts render as ``<traced>``.
        """
        zz_value = maybe_concrete_scalar(self.zz)
        zz_text = f"{zz_value * 1e3:+.4g} MHz" if zz_value is not None else "<traced>"
        lines = [f"Static ZZ({self.device_a}, {self.device_b}) = {zz_text}"]
        lines.append("  leading virtual pathways (2nd-order SW correction to E(1,1)):")
        if contains_tracer([amount for _, amount in self.pathways]):
            lines.append("    <traced>")
        else:
            # Each amount is |V_ik|^2 * (...) for the diagonal element attributed here — real-valued,
            # but carried in the complex dtype of the bare Hamiltonian.
            ranked = sorted(self.pathways, key=lambda kv: -abs(complex(kv[1]).real))
            for occupation, amount in ranked[:5]:
                ket = ",".join(f"{lab}={n}" for lab, n in zip(self.device_labels, occupation))
                lines.append(f"    |{ket}>: {complex(amount).real * 1e3:+.4g} MHz")
        text = "\n".join(lines)
        print(text)
        return text


def analyze_static_zz(chip: "Chip", device_a: str | BaseDevice, device_b: str | BaseDevice) -> StaticZZResult:
    """Compute exact static ZZ between two devices, plus its 2nd-order SW pathway attribution.

    ``zz`` is :meth:`~quchip.chip.chip.Chip.dispersive_shift`, unchanged —
    the exact, all-orders residual coupling. ``pathways`` decomposes the
    2nd-order SW correction to the ``(1_a, 1_b)`` diagonal energy into its
    virtual-intermediate-state contributions
    (:func:`~quchip.chip.sw.pathway_attribution`), read off a partition that
    keeps only the four computational states of ``(a, b)`` (every other
    device grounded) — the natural loss primitive for a calibration sweep
    that holds ``zz`` near zero while some other exchange (e.g. a
    :func:`~quchip.chip.transformations.eliminate`-mediated ``J``) stays on
    target.

    Parameters
    ----------
    chip : Chip
    device_a, device_b : str or BaseDevice
        The two devices whose static ZZ is analyzed.

    Returns
    -------
    StaticZZResult

    Examples
    --------
    >>> from quchip import Capacitive, Chip, DuffingTransmon
    >>> from quchip.analysis import analyze_static_zz
    >>> q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    >>> q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    >>> chip = Chip([q0, q1], couplings=[Capacitive(q0, q1, g=0.01)])
    >>> result = analyze_static_zz(chip, "q0", "q1")
    >>> zz = result.zz  # exact residual ZZ, GHz
    """
    zz = chip.dispersive_shift(device_a, device_b)

    idx_a, dev_a = chip._resolve_device_index(device_a)
    idx_b, dev_b = chip._resolve_device_index(device_b)
    device_labels = tuple(dev.label for dev in chip.devices)

    h, _, dims = bare_hamiltonian(chip, chip.backend)
    p_mask = _computational_p_mask(dims, idx_a, idx_b)
    s, _ = sylvester_generator(h, p_mask)
    i_idx = _bare_index_pair(dims, idx_a, idx_b, 1, 1)
    raw_pathways = pathway_attribution(h, s, p_mask, i_idx, i_idx)
    pathways = [(tuple(int(x) for x in np.unravel_index(k, dims)), amount) for k, amount in raw_pathways]

    return StaticZZResult(
        zz=zz,
        pathways=pathways,
        device_a=dev_a.label,
        device_b=dev_b.label,
        device_labels=device_labels,
    )


# ---------------------------------------------------------------------------
# effective_hamiltonian
# ---------------------------------------------------------------------------


def _normalize_subspace(
    chip: "Chip",
    subspace: Mapping[str | BaseDevice, int] | Sequence[str | BaseDevice],
) -> list[tuple[int, ...]]:
    """Full chip-length bare-occupation tuples spanning *subspace*.

    A ``{device: levels}`` mapping keeps ``range(levels)`` of each named
    device; a bare sequence of devices keeps each one's full qubit (Fock 0/1)
    subspace. Devices not named are grounded (Fock 0). Both forms accept a
    device label string or the object itself.
    """
    n = len(chip.devices)
    per_device_range: dict[int, range] = {}
    if isinstance(subspace, Mapping):
        for device_key, levels in subspace.items():
            idx, _ = chip._resolve_device_index(device_key)
            per_device_range[idx] = range(int(levels))
    else:
        for device_key in subspace:
            idx, _ = chip._resolve_device_index(device_key)
            per_device_range[idx] = range(2)

    ranges = [per_device_range.get(i, range(1)) for i in range(n)]
    return list(itertools.product(*ranges))


def _inverse_sqrt_hermitian(matrix: Any, iterations: int = 64) -> Any:
    """Return a differentiable inverse square root of a positive-definite Hermitian matrix.

    The coupled Newton-Schulz iteration evaluates the matrix function through
    products and sums. It therefore avoids eigenvector derivatives, which are
    undefined when the matrix has repeated eigenvalues even though its inverse
    square root remains smooth. Concrete calls verify the defining residual;
    the fixed iteration count keeps the traced path compatible with reverse-
    mode differentiation and resolves condition numbers through ``1e8`` in
    double precision.
    """
    matrix = 0.5 * (matrix + matrix.conj().T)
    scale = jnp.linalg.norm(matrix, ord="fro")
    identity = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    y = matrix / scale
    z = identity
    for _ in range(iterations):
        correction = 0.5 * (3.0 * identity - z @ y)
        y = y @ correction
        z = correction @ z
    inverse_sqrt = z / jnp.sqrt(scale)
    inverse_sqrt = 0.5 * (inverse_sqrt + inverse_sqrt.conj().T)
    residual = inverse_sqrt @ matrix @ inverse_sqrt - identity
    if not contains_tracer(residual):
        residual_norm = float(np.linalg.norm(np.asarray(residual), ord=2))
        if not np.isfinite(residual_norm) or residual_norm > 1e-9:
            raise ValueError(
                "The projected dressed-state Gram matrix is singular or too ill-conditioned "
                f"for stable symmetric orthonormalization (residual={residual_norm:.3e})"
            )
    return inverse_sqrt


@dataclass(frozen=True)
class EffectiveHamiltonianResult:
    """Store the des-Cloizeaux effective Hamiltonian on a labeled computational subspace.

    ``h_eff`` is built as ``S^-1/2 (W E W^dagger) S^-1/2`` with ``W`` the
    overlap block between the requested bare states and their assigned
    dressed states, ``E`` the labeled dressed energies, and ``S = W W^dagger``
    the (generally non-orthonormal) overlap Gram matrix — the symmetric
    (Löwdin) orthonormalization of :func:`~quchip.chip.sw.exact_reduction`'s
    pairwise construction, generalized to an arbitrary number of kept states.
    ``S^-1/2 W`` is unitary by construction, so ``h_eff`` is unitarily similar
    to ``diag(E)``: its eigenvalues are exactly the labeled dressed energies,
    to numerical precision, regardless of how strongly the kept states
    hybridize with the rest of the chip. Off-diagonal entries carry the
    effective couplings between kept states; the diagonal is not, in
    general, individually equal to any one dressed energy once couplings mix
    the kept states.

    Attributes
    ----------
    h_eff
        Dense Hermitian matrix, GHz, ordered as :attr:`basis`.
    basis
        Full chip-length bare-occupation tuples spanning the subspace, in
        :attr:`h_eff` row/column order.
    device_labels
        Chip device labels in tensor-product order, for reading
        :attr:`basis` tuples.
    """

    h_eff: Any
    basis: tuple[tuple[int, ...], ...]
    device_labels: tuple[str, ...]

    def describe(self) -> str:
        """Print and return the matrix with labeled row/column kets.

        Concrete values only — call outside ``jax.jit``/``grad`` regions;
        a traced matrix renders as ``<traced>``.
        """
        lines = ["Effective Hamiltonian (GHz):"]
        for i, occupation in enumerate(self.basis):
            ket = ",".join(f"{lab}={n}" for lab, n in zip(self.device_labels, occupation))
            lines.append(f"  [{i}] |{ket}>")
        if contains_tracer(self.h_eff):
            lines.append("  <traced>")
        else:
            matrix = np.asarray(self.h_eff)
            for row in matrix:
                lines.append("  " + "  ".join(f"{v.real:+.4f}{v.imag:+.4f}j" for v in row))
        text = "\n".join(lines)
        print(text)
        return text


def _h_eff_on_basis(chip: "Chip", basis: Sequence[tuple[int, ...]]) -> Any:
    """Löwdin-orthonormalized effective Hamiltonian projected onto explicit bare states.

    Shared core behind :func:`effective_hamiltonian` (an arbitrary subspace,
    built from a ``{device: levels}``/bare-sequence spec via
    :func:`_normalize_subspace`) and
    :func:`effective_hamiltonian_between_states` (exactly two explicit
    states) — see :class:`EffectiveHamiltonianResult` for the construction.
    """
    dims = tuple(dev.levels for dev in chip.devices)
    eigenvalues, evecs, _, labeling = chip._analysis._compute_array_labeled()
    evecs = jnp.asarray(evecs)
    eigenvalues = jnp.asarray(eigenvalues)

    bare_idx_list = [int(np.ravel_multi_index(state, dims)) for state in basis]
    bare_idx = jnp.array(bare_idx_list)
    dressed_idx = jnp.stack([labeling.indices[i] for i in bare_idx_list])

    w = evecs[bare_idx[:, None], dressed_idx[None, :]]
    gram = w @ w.conj().T
    gram = 0.5 * (gram + gram.conj().T)
    inv_sqrt = _inverse_sqrt_hermitian(gram)
    h_eff = inv_sqrt @ (w @ jnp.diag(eigenvalues[dressed_idx]) @ w.conj().T) @ inv_sqrt
    return 0.5 * (h_eff + h_eff.conj().T)


def effective_hamiltonian(
    chip: "Chip",
    subspace: Mapping[str | BaseDevice, int] | Sequence[str | BaseDevice],
) -> EffectiveHamiltonianResult:
    """Compute the des-Cloizeaux effective Hamiltonian on a user-chosen computational subspace.

    Reuses the chip's dressed spectrum
    (:meth:`~quchip.chip.analysis.ChipAnalysis._compute_array_labeled`) rather
    than re-diagonalizing: one full-chip diagonalization drives both this and
    :meth:`~quchip.chip.chip.Chip.dispersive_shift`. See
    :class:`EffectiveHamiltonianResult` for the construction and its exactness
    guarantee.

    Parameters
    ----------
    chip : Chip
    subspace : mapping or sequence
        ``{device: levels}`` keeps ``range(levels)`` of each named device
        (spectators grounded); a bare sequence of devices keeps each one's
        full qubit (Fock 0/1) subspace (spectators grounded). Both accept a
        device label string or the object itself.

    Returns
    -------
    EffectiveHamiltonianResult

    Examples
    --------
    >>> from quchip import Capacitive, Chip, DuffingTransmon
    >>> from quchip.analysis import effective_hamiltonian
    >>> q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    >>> q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    >>> chip = Chip([q0, q1], couplings=[Capacitive(q0, q1, g=0.01)])
    >>> result = effective_hamiltonian(chip, ["q0", "q1"])
    >>> result.h_eff.shape
    (4, 4)
    """
    device_labels = tuple(dev.label for dev in chip.devices)
    basis = _normalize_subspace(chip, subspace)
    h_eff = _h_eff_on_basis(chip, basis)
    return EffectiveHamiltonianResult(h_eff=h_eff, basis=tuple(basis), device_labels=device_labels)


def effective_hamiltonian_between_states(
    chip: "Chip", state_a: tuple[int, ...], state_b: tuple[int, ...]
) -> Any:
    r"""Compute the 2x2 Löwdin-orthonormalized effective Hamiltonian between two explicit bare states.

    The same des-Cloizeaux construction :func:`effective_hamiltonian` uses
    (see :class:`EffectiveHamiltonianResult`), specialized to exactly the
    two-state subspace spanned by *state_a* and *state_b* — not the
    four-state product subspace a ``["a", "b"]`` bare-sequence spec to
    :func:`effective_hamiltonian` would build (each device's full qubit
    subspace independently), which is a different projection. This is the
    natural primitive for a static exchange rate between two
    single-excitation bare states :math:`|1_a, 0_b\rangle` and
    :math:`|0_a, 1_b\rangle`: the returned matrix's off-diagonal entry is
    that exchange rate, in GHz.

    Parameters
    ----------
    chip : Chip
    state_a, state_b : tuple[int, ...]
        Full chip-length bare-occupation tuples (one entry per device, in
        :attr:`~quchip.chip.chip.Chip.devices` order).

    Returns
    -------
    Any
        ``(2, 2)`` Hermitian matrix, GHz, in the array namespace of the
        chip's backend. Eigenvalues are exactly *state_a* and *state_b*'s
        labeled dressed energies, to numerical precision.

    Examples
    --------
    >>> from quchip import Capacitive, Chip, DuffingTransmon
    >>> from quchip.analysis import effective_hamiltonian_between_states
    >>> q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    >>> q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    >>> chip = Chip([q0, q1], couplings=[Capacitive(q0, q1, g=0.01)])
    >>> h_eff = effective_hamiltonian_between_states(chip, (1, 0), (0, 1))
    >>> h_eff.shape
    (2, 2)
    """
    return _h_eff_on_basis(chip, (state_a, state_b))
