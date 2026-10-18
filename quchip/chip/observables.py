"""Observable construction for :class:`~quchip.chip.chip.Chip`.

These helpers turn an operator specification into a backend-native operator
embedded on the chip's tensor-product space. The specification is a short name
string (resolved off the device through :meth:`BaseDevice.local_operator`), a
raw local-space operator, or a raw full-space NumPy array.

The chip forwards its public observable surface (:meth:`Chip.observable`,
:meth:`Chip.e_ops`, :meth:`Chip.from_array`) here, and users usually call those
chip methods instead.
"""
# Module-level functions (with `chip` as the first argument) mirror
# `quchip.chip.serialization`, because this group has no per-chip state of its
# own.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from quchip.backend import _backend_context
from quchip.devices.base import BaseDevice

if TYPE_CHECKING:
    from quchip.chip.chip import Chip


def from_array(chip: "Chip", data: Any, device: str | BaseDevice | None = None) -> Any:
    """Build a backend operator from a raw NumPy array.

    With *device*, the array is interpreted as a local operator on
    that device's subspace and embedded into the full tensor-product
    space. With ``device=None`` the array must already span the full
    chip Hilbert space.
    """
    array = np.asarray(data, dtype=complex)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"Operator data must be a square matrix, got shape {array.shape}")

    if device is None:
        total_dim = chip.total_dim
        if array.shape != (total_dim, total_dim):
            raise ValueError(
                f"full-space operator shape must be {(total_dim, total_dim)}, got {array.shape}"
            )
        return chip.backend.from_array(array, dims=[list(chip.dims), list(chip.dims)])

    from quchip.engine.observables import prepare_local_op

    idx, dev = chip._resolve_device_index(device)
    basis = chip.resolve().bases[dev.label]
    local = prepare_local_op(dev, array, basis, chip.backend)
    return chip.backend.embed(local, idx, chip.dims)


def observable(chip: "Chip", device: str | BaseDevice, op: str | Any) -> Any:
    """Embed a device operator onto the full chip Hilbert space.

    Accepts an operator name (``"X"``, ``"Y"``, ``"Z"``, ``"n"``, ``"a"``,
    ``"a_dag"``, ``"I"``) or an already-built local-space operator. Returns the
    operator embedded on the chip's tensor-product space.

    This function is for manual full-space operator construction and analysis.
    It is the named-operator counterpart of :func:`from_array`, together with
    :meth:`~quchip.chip.analysis.ChipAnalysis.operator_in_dressed_basis`. It is
    *not* a solver ``e_op``. :func:`e_ops` (``Chip.e_ops``) is the solver
    surface, and it keeps operators *local* so the demodulation pipeline can
    band-decompose and embed them correctly. If you pass this embedded operator
    to ``simulate(e_ops=...)``, it is misread as a local device operator.
    """
    from quchip.engine.observables import prepare_local_op

    idx, dev = chip._resolve_device_index(device)
    basis = chip.resolve().bases[dev.label]
    with _backend_context(chip.backend):
        local_op = prepare_local_op(dev, op, basis, chip.backend)
    return chip.backend.embed(local_op, idx, chip.dims)


def e_ops(
    chip: "Chip",
    *,
    correlators: dict[
        tuple[str | BaseDevice, str | BaseDevice],
        tuple[str | Any, str | Any],
    ] | None = None,
    **specs: str | list | Any,
) -> dict[str | tuple[str, str], Any]:
    """Build a dict-form ``e_ops`` mapping for the solver pipeline.

    Each keyword maps a device label to an operator specification: a name
    string, a list of names, a raw local-space operator, or a mixed list of
    strings and operators. Specify two-device correlators (e.g. ``⟨Z₁⊗Z₂⟩``)
    with *correlators* as device-label pairs → operator pairs. Returns
    local-space operators (not embedded), which the demodulation pipeline
    embeds as needed.
    """
    result: dict[str | tuple[str, str], Any] = {}
    for label, spec in specs.items():
        _, dev = chip._resolve_device_index(label)
        result[dev.label] = spec

    if correlators is not None:
        for (key_a, key_b), pair in correlators.items():
            _, dev_a = chip._resolve_device_index(key_a)
            _, dev_b = chip._resolve_device_index(key_b)
            result[(dev_a.label, dev_b.label)] = pair

    return result
