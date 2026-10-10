"""Explicit local approximation for fitted observables.

With ``evaluator="local"``, each target uses its device(s) and their directly
coupled neighbors. The approximation omits more distant devices and their
indirect effects. Hilbert-space limits do not select this approximation
automatically.
"""

from __future__ import annotations

from typing import Any

from quchip.chip import Chip, PortNetwork
from quchip.utils.labeling import resolve_label


def build_local_subsystem(chip: Chip, labels: tuple[str, ...]) -> Chip:
    """Build a reduced ``Chip`` that holds only the given device labels.

    A coupling is kept only if both endpoint labels are in ``labels``. The
    reduced chip inherits the basis, frame, RWA, and backend settings from the
    parent. Keeping all devices clones the full model. A port whose targets
    are all in ``labels`` is kept with its field subgraph, via
    :meth:`~quchip.chip.port_network.PortNetwork.restrict`.

    Parameters
    ----------
    chip : Chip
    labels : tuple[str, ...]
        Labels to keep.

    Returns
    -------
    Chip
        Reduced chip that holds only the kept devices, their mutual
        couplings, and their ports.

    Raises
    ------
    ValueError
        A label is not a device of ``chip``.
    NotImplementedError
        The chip has effective terms, or the network cannot be cut at the
        subsystem without changing its interactions. This happens when a port
        targets kept and discarded devices, or when a kept port's field
        subgraph or boundary scattering reaches a discarded port.
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
    network = _local_port_network(chip, keep)
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
        port_network=network,
    )
    from quchip.chip.states import copy_state_configuration

    copy_state_configuration(chip, local)
    return local


def _local_port_network(chip: Chip, keep: set[str]) -> PortNetwork | None:
    """Restrict the chip's network to the ports whose targets are all kept."""
    network = chip.port_network
    if network is None:
        return None
    ports = []
    for port in network.ports:
        targets = set(port.resolve_targets(chip))
        if targets <= keep:
            ports.append(port.label)
        elif targets & keep:
            raise NotImplementedError(
                f"Local fit extraction cannot keep port {port.label!r}, which also targets devices "
                f"outside {sorted(keep)}. Evaluate the full chip to retain its network interactions."
            )
    if not ports:
        return None
    try:
        return network.restrict(ports)
    except ValueError as error:
        raise NotImplementedError(
            f"Local fit extraction cannot cut the PortNetwork at {sorted(keep)}: {error} "
            "Evaluate the full chip to retain its network interactions."
        ) from error


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
