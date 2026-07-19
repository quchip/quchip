"""Chip serialization, deserialization, and structural cloning.

These helpers turn a :class:`~quchip.chip.chip.Chip` into a JSON-safe
dict and back (via the device / coupling registries), and produce
isolated structural clones suitable for sweep evaluation.

Cloning is structural, not numerical: devices are copied fresh,
couplings are rebound to the cloned device instances, and control
equipment — when attached — is cloned and reconnected. Clones keep the
chip-specific backend selection so sweeps run on the same backend as
the original.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quchip.chip.coupling_base import BaseCoupling
from quchip.control.equipment import ControlEquipment
from quchip.devices.base import BaseDevice

if TYPE_CHECKING:
    from quchip.chip.chip import Chip


def _serialize_frame(raw_frame: Any) -> str | float | dict[str, float]:
    """Normalize a chip frame spec into a JSON-safe value."""
    if isinstance(raw_frame, dict):
        return {str(k): float(v) for k, v in raw_frame.items()}
    if isinstance(raw_frame, (int, float)):
        return float(raw_frame)
    return raw_frame


def serialize_chip(chip: "Chip") -> dict[str, Any]:
    """Serialize chip topology into a JSON-safe dictionary.

    Captures devices, couplings, baths, frame, RWA policy, and — if
    present — the control equipment wiring. The chip label (if any) is
    included verbatim. Backend identity is *not* serialized;
    deserialization uses the process default backend unless changed
    afterwards.
    """
    data: dict[str, Any] = {
        "label": chip.label,
        "frame": _serialize_frame(chip.frame),
        "rwa": chip.rwa,
        "devices": [device.to_dict() for device in chip.devices],
        "couplings": [coupling.to_dict() for coupling in chip.couplings],
    }
    if chip.baths:
        data["baths"] = [bath.to_dict() for bath in chip.baths]
    if chip.control_equipment is not None:
        data["control_equipment"] = chip.control_equipment.to_dict()
    return data


def deserialize_chip(data: dict[str, Any]) -> "Chip":
    """Reconstruct a chip from :func:`serialize_chip` output.

    Device and coupling classes are resolved through their shared
    :class:`~quchip.utils.registry.Registrable` registries (via
    :meth:`BaseDevice.from_dict` / :meth:`BaseCoupling.from_dict`), which
    are populated at subclass-definition time, so any extension module must
    be imported before deserialization.
    """
    from quchip.chip.chip import Chip

    devices = [BaseDevice.from_dict(d) for d in data.get("devices", [])]
    device_map = {device.label: device for device in devices}

    couplings: list[BaseCoupling] = []
    for cd in data.get("couplings", []):
        couplings.append(
            BaseCoupling.from_dict(cd, device_map[cd["device_a_label"]], device_map[cd["device_b_label"]])
        )
    coupling_map = {coupling.label: coupling for coupling in couplings}

    from quchip.chip.baths import Bath

    baths = [Bath.from_dict(bd) for bd in data.get("baths", [])]

    control_equipment = (
        ControlEquipment.from_dict(data["control_equipment"], device_map, coupling_map)
        if "control_equipment" in data
        else None
    )

    chip = Chip(
        devices=devices,
        couplings=couplings or None,
        control_equipment=None,
        label=data.get("label"),
        frame=data.get("frame", "lab"),
        rwa=bool(data.get("rwa", True)),
        baths=baths or None,
    )
    if control_equipment is not None:
        chip.connect(control_equipment)
    return chip


def clone_chip(chip: "Chip") -> "Chip":
    """Isolated structural clone suitable for sweep evaluation.

    Devices are copied and decoupled from their original drive wiring;
    couplings are rebound to the cloned device instances. Control
    equipment, when present, is cloned and reconnected so the clone's
    drives target the clone's devices — not the originals.
    """
    from quchip.chip.chip import Chip

    devices = [device.copy() for device in chip.devices]
    device_map = {device.label: device for device in devices}
    couplings = [coupling.copy(device_map) for coupling in chip.couplings]
    frame = dict(chip.frame) if isinstance(chip.frame, dict) else chip.frame
    cloned = Chip(
        devices=devices,
        couplings=couplings or None,
        control_equipment=None,
        label=chip.label,
        frame=frame,
        rwa=chip.rwa,
        backend=chip._backend,
        baths=[bath.copy() for bath in chip.baths] or None,
    )
    if chip.control_equipment is not None:
        cloned.connect(chip.control_equipment.copy(device_map, cloned.coupling_map))
    return cloned
