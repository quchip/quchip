"""Solve-time request for a complete accessible output field."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quchip.utils.labeling import resolve_label


@dataclass(frozen=True)
class OutputField:
    r"""Request the mean field and photon flux at an exposed output.

    You usually get it as ``network.expose(...).output`` and give it to
    ``e_ops`` when you build a simulation.

    Parameters
    ----------
    exposure : str or NetworkPort
        Exposed network port or its label. The calculation must contain it.

    Attributes
    ----------
    exposure : str
        Stored exposure label.

    See Also
    --------
    quchip.results.results.OutputFieldTrace : Output moments, units, and conventions.
    quchip.chip.port_network.PortNetwork : Input-output model and physics references.
    """

    exposure: str

    def __init__(self, exposure: str | Any) -> None:
        object.__setattr__(self, "exposure", resolve_label(exposure))


def is_output_field(value: Any) -> bool:
    """Return whether *value* requests a complete output-field trace."""
    return isinstance(value, OutputField)


__all__ = ["OutputField"]
