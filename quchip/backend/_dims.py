"""Pure backend-free helpers for dims metadata, two-body embedding, and step budgets.

These are stateless free functions shared by the concrete backends (QuTiP,
dynamiqs) and by the :class:`~quchip.backend.protocol.Backend` ABC defaults.
They take no backend ``self`` and introduce no native type, so they live apart
from the contract (the ABC) and its payloads (:mod:`quchip.backend.containers`).

Unit convention: every frequency here is **ordinary** (not angular) GHz with
time in ns.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def validate_two_body_indices(index_a: int, index_b: int, dims: Sequence[int]) -> None:
    """Raise ``ValueError`` when two-body device indices are equal or out of range."""
    n_devices = len(dims)
    if index_a == index_b:
        raise ValueError("index_a and index_b must be different")
    for label, idx in (("index_a", index_a), ("index_b", index_b)):
        if idx < 0 or idx >= n_devices:
            raise ValueError(f"{label} = {idx} out of range for {n_devices} devices")


def compute_two_body_permutation(
    idx_a: int, idx_b: int, dims: Sequence[int]
) -> tuple[list[int], list[int]]:
    """Compute permutations moving two subsystems to the front of a tensor product.

    Returns ``(order, inverse)`` where *order* lists the original subsystem
    positions in their reordered layout (the two target subsystems first, in
    ascending index order, followed by spectators) and *inverse* maps each
    original position to its new slot so ``order[inverse[i]] == i``.

    Used by backends to embed a two-body operator acting on arbitrary device
    indices into the full tensor product without densifying spectators.

    Parameters
    ----------
    idx_a, idx_b
        Subsystem positions the two-body operator acts on, in any order.
    dims
        Subsystem dimensions of the full tensor product; only ``len(dims)``
        is read.

    Examples
    --------
    >>> from quchip.backend._dims import compute_two_body_permutation
    >>> compute_two_body_permutation(2, 0, [2, 2, 2, 2])
    ([0, 2, 1, 3], [0, 2, 1, 3])
    """
    first, second = sorted((idx_a, idx_b))
    n = len(dims)
    order = [first, second] + [i for i in range(n) if i not in (first, second)]
    inverse = [0] * n
    for new_pos, old_pos in enumerate(order):
        inverse[old_pos] = new_pos
    return order, inverse


def normalize_dims_from_list(
    dims: list[list[int]] | list[int] | None, fallback: int | None = None
) -> tuple[int, ...]:
    """Normalize quchip-style dims metadata to a flat subsystem-dim tuple.

    Accepts ``[[rows], [cols]]``, ``[[rows]]``, or a flat list. ``None`` means a
    single subsystem of size *fallback*.

    Parameters
    ----------
    dims
        ``[[rows], [cols]]``, ``[[rows]]``, a flat list of subsystem
        dimensions, or ``None``.
    fallback
        Single-subsystem dimension used when *dims* is ``None``.

    Examples
    --------
    >>> from quchip.backend._dims import normalize_dims_from_list
    >>> normalize_dims_from_list([[2, 3], [2, 3]])
    (2, 3)
    >>> normalize_dims_from_list(None, fallback=4)
    (4,)
    """
    if dims is None:
        if fallback is None:
            raise ValueError("fallback is required when dims is None")
        return (fallback,)
    if dims and isinstance(dims[0], list):
        return tuple(dims[0])
    return tuple(d for d in dims if isinstance(d, int))


def default_solver_steps(metadata: dict[str, Any], tlist: Any) -> int | None:
    """Return a heuristic step budget: ~100 steps per period of the fastest scale, min 200,000.

    ``nsteps`` is an *abort ceiling* for adaptive integrators, not a
    step-size choice — the integrator picks its own step and the ceiling
    only decides when to give up, so a generous value costs nothing on a
    converging solve. The previous tight budget (10/period, min 500) sat
    at the same order as the true step count of high-order methods at
    tight tolerance over long spans, which forced user code to carry
    ``{"nsteps": 200_000}`` overrides; the floor now bakes that in.

    The fastest oscillation is the larger of the static spectral span
    (``spectral_bound_ghz``) and the largest time-dependent carrier
    (``max_carrier_freq_ghz``). Both are ordinary GHz. Considering the
    carrier is essential in a rotating frame, where the static span is
    near-zero but inter-mode / drive carriers still drive fast dynamics.

    Parameters
    ----------
    metadata
        Hamiltonian metadata; reads ``spectral_bound_ghz`` and
        ``max_carrier_freq_ghz`` (both ordinary GHz).
    tlist
        Save-time grid; only its span ``tlist[-1] - tlist[0]`` (ns) is used.

    Returns ``None`` when neither scale is available, the tlist span is
    non-positive, or any input is a JAX tracer (which must not be
    concretized — the caller falls back to the solver library's default).
    """
    from quchip.utils.jax_utils import maybe_concrete_scalar

    scales = [
        s
        for s in (
            maybe_concrete_scalar(metadata.get("spectral_bound_ghz")),
            maybe_concrete_scalar(metadata.get("max_carrier_freq_ghz")),
        )
        if s is not None
    ]
    if not scales:
        return None
    fastest = max(scales)
    if fastest <= 0 or len(tlist) < 2:
        return None
    span = maybe_concrete_scalar(tlist[-1] - tlist[0])
    if span is None or span <= 0:
        return None
    return max(200_000, int(np.ceil(100 * fastest * span)))
