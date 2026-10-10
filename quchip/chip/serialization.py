"""Chip serialization, deserialization, and structural cloning.

These helpers turn a :class:`~quchip.chip.chip.Chip` into a JSON-safe dict and
back (through the device / coupling registries). They also make isolated
structural clones suitable for sweep evaluation.

Cloning is structural, not numerical: devices are copied fresh, and couplings
are rebound to the cloned device instances. If control equipment is attached,
it is cloned and reconnected. Clones keep the chip-specific backend selection
so sweeps run on the same backend as the original.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from quchip.approximations import Approximation
from quchip.chip.coupling_base import BaseCoupling
from quchip.chip.effective import EffectiveTerms
from quchip.chip.states import copy_state_configuration
from quchip.control.equipment import ControlEquipment
from quchip.devices.base import BaseDevice
from quchip.utils.values import copy_value

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

    Captures devices, couplings, baths, frame, approximation strategy, and, if
    present, the control equipment wiring. The chip label, if any, is included
    verbatim. Backend identity is *not* serialized, so deserialization uses the
    process default backend unless changed afterwards.
    """
    data: dict[str, Any] = {
        "format_version": 1,
        "label": chip.label,
        "frame": _serialize_frame(chip.frame),
        "approximation": chip.approximation.to_dict(),
        "basis": chip.basis,
        "devices": [device.to_dict() for device in chip.devices],
        "couplings": [coupling.to_dict() for coupling in chip.couplings],
    }
    if chip._state_order is not None:
        data["state_order"] = list(chip._state_order)
        data["level_symbols"] = dict(chip._level_symbols)
    if chip.effective_terms:
        data["effective_terms"] = [terms.to_dict() for terms in chip.effective_terms]
    if chip.baths:
        data["baths"] = [bath.to_dict() for bath in chip.baths]
    if chip.port_network is not None:
        data["port_network"] = chip.port_network.to_dict()
    if chip.control_equipment is not None:
        data["control_equipment"] = chip.control_equipment.to_dict()
    return data


def deserialize_chip(data: dict[str, Any]) -> "Chip":
    """Reconstruct a chip from :func:`serialize_chip` output.

    Device and coupling classes are resolved through their shared
    :class:`~quchip.utils.registry.Registrable` registries (through
    :meth:`BaseDevice.from_dict` / :meth:`BaseCoupling.from_dict`). These
    registries are populated at subclass-definition time, so import any
    extension module before deserialization.
    """
    from quchip.chip.chip import Chip

    if type(data.get("format_version")) is not int or data["format_version"] != 1:
        raise ValueError("Unsupported Chip format_version; recreate older models with quchip 0.3.")
    allowed = {
        "format_version",
        "label",
        "frame",
        "approximation",
        "basis",
        "devices",
        "couplings",
        "baths",
        "port_network",
        "control_equipment",
        "state_order",
        "level_symbols",
        "effective_terms",
    }
    unknown = set(data) - allowed
    if unknown:
        raise TypeError(f"Unsupported serialized Chip fields: {sorted(unknown)}")
    if "approximation" not in data:
        raise TypeError("Serialized Chip payload is missing required field 'approximation'.")
    approximation = Approximation.from_dict(data["approximation"])

    devices = [BaseDevice.from_dict(d) for d in data.get("devices", [])]
    device_map = {device.label: device for device in devices}
    if len(device_map) != len(devices):
        raise ValueError("Serialized Chip contains duplicate device labels.")

    couplings: list[BaseCoupling] = []
    for cd in data.get("couplings", []):
        if "rwa" in cd:
            raise TypeError("Unsupported serialized coupling fields: ['rwa']")
        endpoints = [cd.get(key) for key in ("device_a_label", "device_b_label")]
        if any(label not in device_map for label in endpoints):
            raise ValueError(f"Serialized coupling references unknown device endpoints: {endpoints}")
        couplings.append(BaseCoupling.from_dict(cd, *(device_map[label] for label in endpoints)))
    coupling_map = {coupling.label: coupling for coupling in couplings}

    from quchip.chip.baths import Bath

    baths = [Bath.from_dict(bd) for bd in data.get("baths", [])]
    from quchip.chip.port_network import PortNetwork
    port_network = (
        PortNetwork.from_dict(data["port_network"])
        if "port_network" in data
        else None
    )

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
        approximation=approximation,
        basis=cast(Literal["native", "eigen"], data.get("basis", "native")),
        baths=baths or None,
        effective_terms=[EffectiveTerms.from_dict(item) for item in data.get("effective_terms", [])],
        port_network=port_network,
    )
    if control_equipment is not None:
        chip.connect(control_equipment)
    if data.get("state_order") is not None:
        chip.set_state_order(*data["state_order"], levels=data.get("level_symbols"))
    return chip


def clone_chip(chip: "Chip") -> "Chip":
    """Isolated structural clone suitable for sweep evaluation.

    Devices are copied and decoupled from their original drive wiring.
    Couplings are rebound to the cloned device instances. If control equipment
    is present, it is cloned and reconnected, so the clone's drives target the
    clone's devices and not the originals.
    """
    from quchip.chip.chip import Chip

    devices = [device.copy() for device in chip.devices]
    device_map = {device.label: device for device in devices}
    couplings = [coupling.copy(device_map) for coupling in chip.couplings]
    cloned = Chip(
        devices=devices,
        couplings=couplings or None,
        control_equipment=None,
        label=chip.label,
        frame=copy_value(chip.frame),
        approximation=chip.approximation,
        basis=chip.basis,
        backend=chip._backend,
        baths=[bath.copy() for bath in chip.baths] or None,
        effective_terms=chip.effective_terms,
        port_network=None if chip.port_network is None else chip.port_network.copy(),
    )
    if chip.control_equipment is not None:
        cloned.connect(chip.control_equipment.copy(device_map, cloned.coupling_map))
    copy_state_configuration(chip, cloned)
    return cloned
