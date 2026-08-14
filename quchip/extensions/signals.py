"""Reference classical signal transform."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quchip.control.signal import SignalMap, SignalTransform
from quchip.declarative import qnp
from quchip.utils.labeling import resolve_label


@dataclass(frozen=True)
class CableLoss(SignalTransform):
    """Attenuate one control line by a power loss specified in dB."""

    line: str
    loss_db: Any
    _parameter_names = ("loss_db",)

    def __init__(self, line: str | Any, loss_db: Any) -> None:
        object.__setattr__(self, "line", resolve_label(line))
        object.__setattr__(self, "loss_db", loss_db)

    def apply(self, signals: SignalMap) -> SignalMap:
        factor = qnp.power(10.0, -self.loss_db / 20.0)
        return {
            key: signal.scaled(factor) if key[0] == self.line else signal
            for key, signal in signals.items()
        }

    def referenced_lines(self) -> tuple[str, ...]:
        return (self.line,)

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update(line=self.line, loss_db=float(self.loss_db))
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CableLoss":
        return cls(line=str(data["line"]), loss_db=float(data["loss_db"]))
