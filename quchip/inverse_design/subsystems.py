"""Explicit local approximation for fitted observables.

With ``evaluator="local"``, each target uses its device(s) and their directly
coupled neighbors. The approximation omits more distant devices and their
indirect effects. Hilbert-space limits do not select this approximation
automatically.
"""

from __future__ import annotations

from typing import Any

from quchip.chip import Chip
from quchip.utils.labeling import resolve_label


def build_local_subsystem(chip: Chip, labels: tuple[str, ...]) -> Chip:
    """Build a reduced ``Chip`` that holds only the given device labels.

    A coupling is kept only if both endpoint labels are in ``labels``. The
    reduced chip inherits the basis, frame, RWA, and backend settings from the
    parent. Keeping all devices clones the full model. Partial extraction of a
    PortNetwork model is unsupported, because the network can induce
    interactions.

    Parameters
    ----------
    chip : Chip
    labels : tuple[str, ...]
        Labels to keep.

    Returns
    -------
    Chip
        Reduced chip that holds only the kept devices and their mutual
        couplings.
    """
    keep = set(labels)
    unknown = keep - chip.device_map.keys()
    if unknown:
        raise ValueError(f"Unknown local subsystem devices: {sorted(unknown)}")
    if keep == chip.device_map.keys():
        return chip.clone()
    if chip.effective_terms:
        raise NotImplementedError(
            "Local fit extraction cannot discard retained effective terms; evaluate the full chip."
        )
    if chip.port_network is not None:
        raise NotImplementedError(
            "Local fit extraction of a partial PortNetwork model is not supported. "
            "Evaluate the full chip to retain its network interactions."
        )
    devices = [device.copy() for device in chip.devices if device.label in keep]
    device_map = {device.label: device for device in devices}
    couplings = [
        coupling.copy(device_map)
        for coupling in chip.couplings
        if coupling.device_a_label in keep and coupling.device_b_label in keep
    ]
    frame: Any = (
        {label: value for label, value in chip.frame.items() if label in keep}
        if isinstance(chip.frame, dict) else chip.frame
    )
    local = Chip(
        devices=devices,
        couplings=couplings,
        frame=frame,
        approximation=chip.approximation,
        basis=chip.basis,
        backend=chip.backend,
    )
    from quchip.chip.states import copy_state_configuration

    copy_state_configuration(chip, local)
    return local


def device_labels_for_local_eval(chip: Chip, label: Any) -> tuple[str, ...]:
    """Return the seed device(s) and every directly coupled neighbor.

    ``label`` can be a single device/label or a tuple of device/labels. All
    entries are normalized through
    :func:`~quchip.utils.labeling.resolve_label`. The returned tuple is sorted
    for determinism.

    Parameters
    ----------
    chip : Chip
    label : Any
        A single device/label or a tuple of devices/labels.

    Returns
    -------
    tuple[str, ...]
        Sorted device labels: the seed(s) and every directly coupled neighbor.

    Examples
    --------
    A q0 - q1 - q2 chain keeps the seed and its direct neighbors:

    >>> from quchip import Chip, DuffingTransmon, Capacitive
    >>> from quchip.inverse_design.subsystems import device_labels_for_local_eval
    >>> qs = [DuffingTransmon(freq=5.0 + 0.1 * i, anharmonicity=-0.3, levels=3,
    ...                       label=f"q{i}") for i in range(3)]
    >>> chip = Chip(devices=qs, couplings=[Capacitive(qs[0], qs[1], g=0.005),
    ...                                    Capacitive(qs[1], qs[2], g=0.005)])
    >>> device_labels_for_local_eval(chip, "q0")
    ('q0', 'q1')
    >>> device_labels_for_local_eval(chip, ("q0", "q1"))
    ('q0', 'q1', 'q2')
    """
    seeds = {resolve_label(part) for part in label} if isinstance(label, tuple) else {resolve_label(label)}
    labels = set(seeds)
    for coupling in chip.couplings:
        if coupling.device_a_label in seeds:
            labels.add(coupling.device_b_label)
        if coupling.device_b_label in seeds:
            labels.add(coupling.device_a_label)
    return tuple(sorted(labels))
