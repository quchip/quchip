"""Static-ZZ pathways and des-Cloizeaux effective Hamiltonians.

:func:`analyze_static_zz` reports the exact dressed ZZ and a second-order
Schrieffer-Wolff attribution to virtual transitions. The pathway estimates are
diagnostics, and the reported ZZ comes from the dressed spectrum.

:func:`effective_hamiltonian` returns a dense GHz matrix on a selected
computational subspace, whose eigenvalues equal its labeled dressed energies.
Differentiation requires backend support for the eigensolver and a fixed
assignment.

References: Bravyi, DiVincenzo & Loss, *Schrieffer-Wolff transformation for
quantum many-body systems*, Ann. Phys. 326, 2793 (2011); Blais et al.,
*Circuit quantum electrodynamics*, RMP 93, 025005 (2021), §IV.C.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from quchip.chip.sw import bare_hamiltonian, exact_subspace, pathway_attribution, sylvester_generator
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
    """Store the exact static ZZ between two devices and its 2nd-order SW pathway attribution.

    Attributes
    ----------
    zz
        ``E(1,1) - E(1,0) - E(0,1) + E(0,0)``, identical to
        :meth:`~quchip.chip.chip.Chip.dispersive_shift(device_a, device_b)`.
        This value is exact, not perturbative.
    pathways
        ``(bare_occupation, amount)`` pairs. Each pair gives one virtual
        intermediate state's contribution to the 2nd-order SW correction of the
        ``(1_a, 1_b)`` diagonal matrix element. The contribution is
        ``amount = 1/2 * V_ik*V_ki*(1/(E_i - E_k) + 1/(E_i - E_k))``, where
        ``i`` is the ``(1_a, 1_b)`` bare index. This decomposes that one energy
        correction, not ``zz`` itself, which combines four dressed energies
        exactly. ``bare_occupation`` is a full chip-length Fock tuple, in
        device order.
    device_a, device_b
        Resolved device labels.
    device_labels
        Chip device labels in tensor-product order, for reading the occupation
        tuples of ``pathways``.

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
        """Print and return the exact ZZ and the leading virtual pathways.

        Concrete values only. Call it outside ``jax.jit``/``grad`` regions.
        Traced amounts show as ``<traced>``.
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
    """Calculate the exact static ZZ between two devices and its 2nd-order SW pathway attribution.

    ``zz`` is :meth:`~quchip.chip.chip.Chip.dispersive_shift`, unchanged: the
    exact, all-orders residual coupling. ``pathways`` decomposes the 2nd-order
    SW correction to the ``(1_a, 1_b)`` diagonal energy into virtual
    intermediate-state contributions
    (:func:`~quchip.chip.sw.pathway_attribution`). The pathways come from a
    partition that keeps only the four computational states of ``(a, b)``, with
    all other devices grounded.

    This is the natural loss primitive for a calibration sweep that holds
    ``zz`` near zero while a different exchange, e.g. a
    :func:`~quchip.chip.transformations.eliminate`-mediated ``J``, stays on
    target.

    Parameters
    ----------
    chip : Chip
        Chip whose dressed spectrum and couplings define the static model.
    device_a, device_b : str or BaseDevice
        The two devices to analyze.

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

    h, _, dims = bare_hamiltonian(chip)
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


@dataclass(frozen=True)
class EffectiveHamiltonianResult:
    """Store the des-Cloizeaux effective Hamiltonian on a labeled computational subspace.

    The function builds ``h_eff`` as ``S^-1/2 (W E W^dagger) S^-1/2``. Here
    ``W`` is the overlap block between the requested bare states and their
    assigned dressed states, and ``E`` is the labeled dressed energies.
    ``S = W W^dagger`` is the overlap Gram matrix, which in general is not
    orthonormal. Exact mode reductions use the same symmetric (Löwdin)
    orthonormalization.

    ``S^-1/2 W`` is unitary by construction, so ``h_eff`` is unitarily similar
    to ``diag(E)``. Its eigenvalues are therefore exactly the labeled dressed
    energies, to numerical precision, for any hybridization strength between
    the kept states and the rest of the chip. The off-diagonal entries hold the
    effective couplings between kept states. When couplings mix the kept
    states, a diagonal entry in general does not equal one specific dressed
    energy.

    Attributes
    ----------
    h_eff
        Dense Hermitian matrix, GHz, ordered as :attr:`basis`.
    basis
        Full chip-length bare-occupation tuples that span the subspace, in the
        row/column order of :attr:`h_eff`.
    device_labels
        Chip device labels in tensor-product order, for reading the
        :attr:`basis` tuples.
    """

    h_eff: Any
    basis: tuple[tuple[int, ...], ...]
    device_labels: tuple[str, ...]

    def describe(self) -> str:
        """Print and return the matrix with labeled row/column kets.

        Concrete values only. Call it outside ``jax.jit``/``grad`` regions. A
        traced matrix shows as ``<traced>``.
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
    analysis = chip._analysis
    sectors = analysis._sector_model()
    if sectors is not None:
        # A conserving chip needs only the sectors that the requested states occupy.
        labels = [tuple(state) for state in basis]
        if not labels:
            raise ValueError("Effective subspace must contain at least one state.")
        invalid = [label for label in labels if not analysis._is_bare_label(label)]
        if invalid:
            raise ValueError(f"Effective subspace state {invalid[0]} is not a valid bare product-basis label.")
        eigenvalues, evecs, kept, dressed_idx = sectors.labeled_subspace(labels)
        return exact_subspace(eigenvalues, evecs, kept, dressed_idx).hamiltonian

    engine = analysis.engine_result()
    eigenvalues, evecs, _, labeling = analysis._compute_array_labeled(engine)
    evecs = analysis._semantic_amplitudes(evecs, engine)

    bare_idx_list = [int(np.ravel_multi_index(state, engine.dims)) for state in basis]
    if not bare_idx_list:
        raise ValueError("Effective subspace must contain at least one state.")
    dressed_idx = jnp.stack([labeling.indices[i] for i in bare_idx_list])

    return exact_subspace(eigenvalues, evecs, bare_idx_list, dressed_idx).hamiltonian


def effective_hamiltonian(
    chip: "Chip",
    subspace: Mapping[str | BaseDevice, int] | Sequence[str | BaseDevice],
) -> EffectiveHamiltonianResult:
    """Calculate the des-Cloizeaux effective Hamiltonian on a user-selected computational subspace.

    See :class:`EffectiveHamiltonianResult` for the construction and its
    exactness guarantee.

    Parameters
    ----------
    chip : Chip
        Chip whose dressed spectrum defines the effective subspace.
    subspace : mapping or sequence
        ``{device: levels}`` keeps ``range(levels)`` of each named device. A
        bare sequence of devices keeps each device's full qubit (Fock 0/1)
        subspace. In both forms spectators are grounded, and a device can be a
        label string or the device object.

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
    # effective_hamiltonian reuses the chip's dressed spectrum instead of
    # diagonalizing again, so one eigensystem per sector, or one full-chip
    # eigensystem, supplies both this function and `dispersive_shift`.
    device_labels = tuple(dev.label for dev in chip.devices)
    basis = _normalize_subspace(chip, subspace)
    h_eff = _h_eff_on_basis(chip, basis)
    return EffectiveHamiltonianResult(h_eff=h_eff, basis=tuple(basis), device_labels=device_labels)


def effective_hamiltonian_between_states(
    chip: "Chip", state_a: tuple[int, ...], state_b: tuple[int, ...]
) -> Any:
    r"""Calculate the 2x2 Löwdin-orthonormalized effective Hamiltonian between two explicit bare states.

    This function uses the des-Cloizeaux construction of
    :func:`effective_hamiltonian` (see :class:`EffectiveHamiltonianResult`),
    but on exactly the two-state subspace that *state_a* and *state_b* span. It
    does not use the four-state product subspace that a ``["a", "b"]``
    bare-sequence spec to :func:`effective_hamiltonian` builds from each
    device's full qubit subspace. It is the natural primitive for a static
    exchange rate between the single-excitation bare states
    :math:`|1_a, 0_b\rangle` and :math:`|0_a, 1_b\rangle`. The returned
    matrix's off-diagonal entry is that exchange rate, in GHz.

    Parameters
    ----------
    chip : Chip
        Chip whose dressed spectrum defines the effective Hamiltonian.
    state_a, state_b : tuple[int, ...]
        Full chip-length bare-occupation tuples (one entry for each device, in
        :attr:`~quchip.chip.chip.Chip.devices` order).

    Returns
    -------
    Any
        ``(2, 2)`` Hermitian matrix, GHz, in the array namespace of the chip's
        backend. Its eigenvalues are exactly the labeled dressed energies of
        *state_a* and *state_b*, to numerical precision.

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
