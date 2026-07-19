"""Structural rotating-wave kernels for coupling RWA policy.

RWA for a *static* two-body coupling is a structural statement: in the
``(Δa, Δb)`` excitation-change decomposition of the local interaction,
the retained terms are exactly the bands the coupling's
:meth:`~quchip.chip.coupling_base.BaseCoupling.rwa_keeps_band` predicate
accepts (default: total-excitation-conserving, ``Δa + Δb == 0`` — the
beam-splitter selection every textbook RWA coupling encodes). The
predicate consumes integer band offsets derived from operator structure,
never frequency values, so the policy is JAX-safe by construction: the
mask is a concrete constant and multiplying a traced operator by it
preserves gradients.

Two consumers share the policy: :meth:`Chip.hamiltonian` masks each
coupling's full interaction when the chip resolves RWA for it, and
stage 2 filters the band decomposition with the same predicate. The two
views agree because the mask's equivalence classes are exactly the
bands of :func:`~quchip.engine.bands.decompose_two_body_canonical_bands`.

Band-offset convention (shared with :mod:`quchip.engine.bands`):
``Δ = col − row``, so ``Δ = +1`` is a *lowering* operator. Custom
predicates must be symmetric under joint sign flip
(``keeps_band(-Δa, -Δb) == keeps_band(Δa, Δb)``) so the retained
operator stays Hermitian.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from quchip.backend.protocol import Backend, Operator

BandPredicate = Callable[[int, int], bool]


def excitation_band_mask(d_a: int, d_b: int, keeps_band: BandPredicate) -> np.ndarray:
    """Return the 0/1 mask retaining matrix elements whose band passes *keeps_band*.

    Element ``(row, col)`` of a local operator on ``H_a ⊗ H_b`` (mode
    ``b`` fast) belongs to the excitation-change band
    ``(Δa, Δb) = (col_a − row_a, col_b − row_b)``. Dims and the
    predicate are static structure, so the mask is a concrete constant
    regardless of any tracing in the operator it multiplies.

    Parameters
    ----------
    d_a, d_b : int
        Endpoint Hilbert-space dimensions.
    keeps_band : callable
        ``(Δa, Δb) -> bool`` predicate; ``True`` retains the band.

    Returns
    -------
    numpy.ndarray
        Float 0/1 matrix of shape ``(d_a·d_b, d_a·d_b)``.
    """
    flat = np.arange(d_a * d_b)
    idx_a = flat // d_b
    idx_b = flat % d_b
    delta_a = idx_a[None, :] - idx_a[:, None]
    delta_b = idx_b[None, :] - idx_b[:, None]
    mask = np.zeros((d_a * d_b, d_a * d_b))
    for da in range(-(d_a - 1), d_a):
        for db in range(-(d_b - 1), d_b):
            if keeps_band(da, db):
                mask[(delta_a == da) & (delta_b == db)] = 1.0
    return mask


def apply_rwa_mask(
    h_local: "Operator",
    *,
    dims: tuple[int, int],
    labels: tuple[str, str],
    keeps_band: BandPredicate,
    backend: "Backend",
) -> "Operator | None":
    """Sum of the bands *keeps_band* retains from a local two-body operator.

    Built from the same
    :func:`~quchip.engine.bands.decompose_two_body_canonical_bands`
    stage 2 filters, so the mask and the band filter agree by shared
    construction, and each retained band keeps its canonical layout — a
    sparse interaction never densifies on the way into
    :meth:`Chip.hamiltonian`. Numerically the result equals multiplying
    the dense operator by :func:`excitation_band_mask`. A JAX-traced
    payload routes through the decomposition's dense path, so gradients
    survive.

    Parameters
    ----------
    h_local : Operator
        Backend-native operator on the local ``H_a ⊗ H_b`` space,
        ordinary GHz.
    dims : tuple[int, int]
        Endpoint dimensions ``(d_a, d_b)``.
    labels : tuple[str, str]
        Endpoint device labels, carried into the canonical metadata.
    keeps_band : callable
        ``(Δa, Δb) -> bool`` retention predicate.
    backend : Backend
        Backend used for the canonical round-trip.

    Returns
    -------
    Operator or None
        Backend-native masked operator on the same local space, or
        ``None`` when no retained band is populated (a concrete payload
        whose every band the predicate rejects) — callers skip embedding
        the vanished interaction rather than shipping a zero operator to
        the solver. A traced payload always returns an operator: band
        population cannot be inspected without concretizing, so the
        where-masked dense sum is kept (numerically zero where
        rejected).
    """
    from quchip.engine.bands import decompose_two_body_canonical_bands

    d_a, d_b = dims
    canonical = backend.to_canonical_operator(h_local).with_metadata(
        dims=(d_a, d_b),
        subsystem_labels=tuple(labels),
        tag="coupling_local",
    )
    masked: "Operator | None" = None
    for (delta_a, delta_b), band in decompose_two_body_canonical_bands(canonical, [d_a, d_b]).items():
        if not keeps_band(delta_a, delta_b):
            continue
        band_op = backend.from_canonical_operator(band)
        masked = band_op if masked is None else masked + band_op
    return masked
