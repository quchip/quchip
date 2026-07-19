"""scqubits interoperability — ``from_scqubits`` / ``to_scqubits`` dispatch.

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
    """Convert a scqubits object into the matching quchip device.

    Dispatches on ``type(obj)`` through the :class:`ModelMapping` registry.
    Keyword options (``levels``, ``label``, noise parameters) are forwarded to
    the mapping's ``import_model``.

    Raises
    ------
    ImportError
        scqubits is not installed.
    LookupError
        No :class:`ModelMapping` is registered for ``type(obj)``.
    """
    _require_scqubits()

    import scqubits

    if isinstance(obj, scqubits.HilbertSpace):
        from .composite import import_hilbertspace

        return import_hilbertspace(obj, **opts)

    return import_object(obj, **opts)


def to_scqubits(device_or_chip: Any, **opts: Any) -> Any:
    """Convert a quchip device into the matching scqubits object.

    Dispatches on ``type(device_or_chip)`` through the export registry for the
    ``"scqubits"`` library. Parameters must be concrete: a device carrying JAX
    tracers (inside ``jit``/``grad``) raises :class:`ValueError`.

    A :class:`~quchip.chip.chip.Chip` exports to an scqubits ``HilbertSpace``
    (devices as subsystems, couplings as interaction terms); see
    :func:`~quchip.interop.scqubits.composite.export_chip`.

    Raises
    ------
    ImportError
        scqubits is not installed.
    LookupError
        No :class:`ModelMapping` exports this device type to scqubits.
    ValueError
        A device parameter is a JAX tracer rather than a concrete value.
    """
    _require_scqubits()

    from quchip.chip.chip import Chip

    if isinstance(device_or_chip, Chip):
        from .composite import export_chip

        return export_chip(device_or_chip, **opts)

    return export_object(device_or_chip, "scqubits", **opts)


__all__ = ["from_scqubits", "to_scqubits"]
