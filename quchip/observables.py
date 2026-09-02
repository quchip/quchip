"""Solve-time request for a complete accessible output field."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quchip.utils.labeling import resolve_label


@dataclass(frozen=True)
class OutputField:
    r"""Request ``<b_out>`` and ``<b_out dagger b_out>`` at one exposure."""

    exposure: str

    def __init__(self, exposure: str | Any) -> None:
        object.__setattr__(self, "exposure", resolve_label(exposure))


def is_output_field(value: Any) -> bool:
    """Return whether *value* requests a complete output-field trace."""
    return isinstance(value, OutputField)


__all__ = ["OutputField"]
