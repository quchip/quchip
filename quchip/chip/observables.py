"""Observable construction for :class:`~quchip.chip.chip.Chip`.

These helpers turn an operator specification — a short name string
(resolved off the device via :meth:`BaseDevice.local_operator`), a raw
local-space operator, or a raw full-space NumPy array — into a
backend-native operator embedded on the chip's tensor-product space.

The chip forwards its public observable surface (:meth:`Chip.observable`,
:meth:`Chip.e_ops`, :meth:`Chip.from_array`) here; users normally call
the chip methods, not these functions directly. Module-level functions
(taking ``chip`` as the first argument) mirror
:mod:`quchip.chip.serialization`, since this group carries no per-chip
state of its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from quchip.backend import _backend_context
from quchip.declarative.expr import (
    as_operator_expr,
    materialize_expr,
)
from quchip.devices.base import BaseDevice
from quchip.devices.spaces import FockSpace

if TYPE_CHECKING:
    from quchip.chip.chip import Chip
    from quchip.engine.basis import BasisRecord


def prepare_local_op(
    dev: BaseDevice,
    spec: str | Any,
    basis: "BasisRecord",
    backend: Any,
) -> Any:
    """Return one observable in the resolved local solver basis."""
    names = {
        "X": "sigma_x",
        "Y": "sigma_y",
        "Z": "sigma_z",
        "n": "n",
        "a": "a",
        "a_dag": "adag",
        "I": "I",
    }
    if isinstance(spec, str) and spec in names:
        return FockSpace(basis.resolved_dim).operator(names[spec], backend)
    if isinstance(spec, str) and spec == "charge" and hasattr(dev, "charge_coupling_operator"):
        authored = dev.charge_coupling_operator()
    elif isinstance(spec, str) and spec in ("phase", "flux") and hasattr(dev, "phase_coupling_operator"):
        authored = dev.phase_coupling_operator()
    else:
        authored = dev.local_operator(spec) if isinstance(spec, str) else spec
    authored = as_operator_expr(
        authored,
        labels=(dev.label,),
        dims=(basis.native_dim,),
        name=rf"\hat O_{{{dev.label}}}",
        owner=dev,
        scope=dev.label,
    )
    local = materialize_expr(authored, backend)
    if local.shape != (basis.native_dim, basis.native_dim):
        raise ValueError(
            f"Authored operator for {dev.label!r} must have shape "
            f"{(basis.native_dim, basis.native_dim)}, got {local.shape}."
        )
    if basis.kind == "native":
        return local
    matrix = backend.to_array(local)
    return backend.from_array(
        basis.transform_operator(matrix),
        dims=[[basis.resolved_dim], [basis.resolved_dim]],
    )


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

    idx, dev = chip._resolve_device_index(device)
    basis = chip.resolve().bases[dev.label]
    local = prepare_local_op(dev, array, basis, chip.backend)
    return chip.backend.embed(local, idx, chip.dims)


def observable(chip: "Chip", device: str | BaseDevice, op: str | Any) -> Any:
    """Embed a device operator onto the full chip Hilbert space.

    Accepts either an operator name (``"X"``, ``"Y"``, ``"Z"``,
    ``"n"``, ``"a"``, ``"a_dag"``, ``"I"``) or an already-built
    local-space operator, and returns it embedded on the chip's
    tensor-product space.

    This is for manual full-space operator construction and analysis —
    the named-operator counterpart of :func:`from_array`, alongside
    :meth:`~quchip.chip.analysis.ChipAnalysis.operator_in_dressed_basis`.
    It is *not* a solver ``e_op``: :func:`e_ops` (``Chip.e_ops``) is the
    solver surface, and it keeps operators *local* so the demodulation
    pipeline can band-decompose and embed them correctly. Passing this
    embedded operator into ``simulate(e_ops=...)`` would be misread as a
    local device operator.
    """
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

    Each keyword maps a device label to an operator specification: a
    name string, a list of names, a raw local-space operator, or a
    mixed list of strings and operators. Two-device correlators (e.g.
    ``⟨Z₁⊗Z₂⟩``) are specified via *correlators* as device-label pairs →
    operator pairs. Returns local-space operators (not embedded) — the
    demodulation pipeline embeds as needed.
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
