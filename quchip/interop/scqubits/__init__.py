"""scqubits interoperability with ``from_scqubits`` and ``to_scqubits`` dispatch.

Importing this subpackage registers the shipped scqubits device mappings (see
:mod:`quchip.interop.scqubits.devices`) with the library-agnostic
:mod:`quchip.interop.base` registry. The two public functions dispatch through
that registry after checking that scqubits is installed.
"""

from __future__ import annotations

from typing import Any

from quchip.interop.base import export_object, import_object

from . import devices  # noqa: F401  (import registers the shipped mappings)


def _require_scqubits() -> None:
    """Import scqubits on demand so quchip remains usable without it."""
    try:
        import scqubits  # noqa: F401
    except ModuleNotFoundError:
        raise ImportError(
            "scqubits is required for this feature. "
            "Install it with:  pip install quchip[scqubits]"
        ) from None


def from_scqubits(obj: Any, **opts: Any) -> Any:
    """Import a scqubits device or composite system.

    Parameters
    ----------
    obj : scqubits object
        A supported circuit, oscillator, or ``HilbertSpace``.
        :mod:`quchip.interop.scqubits.devices` lists the device mappings.
        Energies are read in the current scqubits unit,
        ``scqubits.get_units()``, and converted to GHz.
    **opts
        Device imports accept ``levels`` (default: source ``truncated_dim``) and
        ``label`` (default: source ``id_str``). They also accept the target
        device's noise options, such as ``T1``, ``T2`` and ``thermal_occupation``.
        Matrix-element relaxation can also require ``coupling_channel`` (see the
        target class). Composite imports accept ``frame`` (default ``"lab"``) and
        ``approximation`` (default ``Exact()``), and do not forward device
        options.

    Returns
    -------
    BaseDevice or Chip
        Converted device, or a composite of frozen eigenbasis snapshots. A
        snapshot is not differentiable through quchip with respect to its
        source parameters.

    Raises
    ------
    ImportError
        The optional ``quchip[scqubits]`` dependency is unavailable.
    LookupError
        No mapping exists for the source type.
    NotImplementedError
        A composite contains unsupported interactions.

    References
    ----------
    Groszkowski and Koch, *scqubits: a Python package for superconducting
    qubits*, Quantum 5, 583 (2021), https://doi.org/10.22331/q-2021-11-17-583.
    """
    _require_scqubits()

    import scqubits

    if isinstance(obj, scqubits.HilbertSpace):
        from .composite import import_hilbertspace

        return import_hilbertspace(obj, **opts)

    return import_object(obj, **opts)


def to_scqubits(device_or_chip: Any, **opts: Any) -> Any:
    """Export a quchip device or chip to scqubits.

    Parameters
    ----------
    device_or_chip : BaseDevice or Chip
        Model with concrete parameters. A chip exports its subsystems and
        supported interactions. Filtered couplings must use ``Exact()`` before
        export.
    **opts
        Mapping-specific options. ``DuffingTransmon`` export accepts ``ncut``
        (integer charge cutoff, default 30) to reconstruct the circuit energies.
        Other shipped device mappings currently ignore extra options. Composite
        export accepts no keyword options.

    Returns
    -------
    scqubits object
        Related device or ``HilbertSpace``, with energies in the current scqubits
        unit, ``scqubits.get_units()``. The export omits chip control equipment and
        baths with a warning. Port networks and effective terms are unsupported.
        See :func:`~quchip.interop.scqubits.composite.export_chip`.

    Raises
    ------
    ImportError
        The optional ``quchip[scqubits]`` dependency is unavailable.
    LookupError
        No export mapping exists for a device type.
    ValueError
        Parameters are traced, or approximation filtering would change the export.

    References
    ----------
    Groszkowski and Koch, Quantum 5, 583 (2021),
    https://doi.org/10.22331/q-2021-11-17-583.
    """
    _require_scqubits()

    from quchip.chip.chip import Chip

    if isinstance(device_or_chip, Chip):
        from .composite import export_chip

        return export_chip(device_or_chip, **opts)

    return export_object(device_or_chip, "scqubits", **opts)


__all__ = ["from_scqubits", "to_scqubits"]
