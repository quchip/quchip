"""Neighborhood extraction for large-chip observables.

When the full Hilbert space is too large to dress repeatedly inside a
least-squares loop, ``fit_a_dress`` evaluates each target on the
smallest sub-chip that still contains the relevant physics: the target
device(s) plus every directly coupled neighbor (one-hop closure).

This is a pragmatic truncation, not a controlled approximation — the
one-hop neighborhood captures all first-order dispersive effects for
the target but ignores second-order contributions from non-neighbors.
Use ``max_hilbert_dim`` in :func:`~quchip.inverse_design.fit.fit_a_dress`
to control when the fit switches from ``"full"`` to ``"local"``.
"""

from __future__ import annotations

from typing import Any

from quchip.chip import Chip
from quchip.utils.labeling import resolve_label


def choose_evaluator(chip: Chip, max_hilbert_dim: int) -> str:
    """Pick ``"full"`` or ``"local"`` based on total Hilbert-space size.

    Returns ``"full"`` when the product of all device ``levels`` is at
    most ``max_hilbert_dim``; otherwise ``"local"``.

    Parameters
    ----------
    chip : Chip
    max_hilbert_dim : int
        Total-Hilbert-space-size threshold.

    Returns
    -------
    str
        ``"full"`` or ``"local"``.
    """
    return "full" if chip.total_dim <= max_hilbert_dim else "local"


def build_local_subsystem(chip: Chip, labels: tuple[str, ...]) -> Chip:
    """Build a reduced ``Chip`` holding only the given device labels.

    Couplings are retained iff both endpoint labels are in ``labels``;
    frame, RWA, and backend settings are inherited from the parent.

    Parameters
    ----------
    chip : Chip
    labels : tuple[str, ...]
        Device labels to keep.

    Returns
    -------
    Chip
        Reduced chip holding only the kept devices and their mutual
        couplings.
    """
    keep = set(labels)
    devices = [device for device in chip.devices if device.label in keep]
    couplings = [
        coupling
        for coupling in chip.couplings
        if coupling.device_a_label in keep and coupling.device_b_label in keep
    ]
    return Chip(devices=devices, couplings=couplings, frame=chip.frame, rwa=chip.rwa, backend=chip.backend)


def device_labels_for_local_eval(chip: Chip, label: Any) -> tuple[str, ...]:
    """Return the seed device(s) plus every directly coupled neighbor.

    ``label`` may be a single device/label or a tuple of device/labels;
    all entries are normalized through
    :func:`~quchip.utils.labeling.resolve_label`. The returned tuple is
    sorted for determinism.

    Parameters
    ----------
    chip : Chip
    label : Any
        A single device/label or a tuple of devices/labels.

    Returns
    -------
    tuple[str, ...]
        Sorted device labels: the seed(s) plus every directly coupled
        neighbor.

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
