"""Reference classical signal transform."""

from __future__ import annotations

from typing import Any

from quchip.declarative.parameters import parameter, setting
from quchip.control.signal import SignalMap, SignalTransform
from quchip.declarative import qnp
from quchip.utils.labeling import resolve_label


class CableLoss(SignalTransform, serializable=True):
    """Attenuate one control line by a power loss specified in dB.

    Parameters
    ----------
    line : str or object with a label
        Control-line label to transform. Resolved once, at construction.
    loss_db : float
        Positive power-loss value in dB. The complex amplitude is multiplied
        by ``10**(-loss_db/20)``.
    """

    line: str = setting()
    loss_db: Any = parameter()

    def __init__(self, line: str | Any, loss_db: Any) -> None:
        object.__setattr__(self, "line", resolve_label(line))
        object.__setattr__(self, "loss_db", loss_db)

    def apply(self, signals: SignalMap) -> SignalMap:
        """Return signals with this line's amplitude attenuated.

        Parameters
        ----------
        signals : SignalMap
            Mapping keyed by ``(line, channel)`` signal identifiers.
        """
        factor = qnp.power(10.0, -self.loss_db / 20.0)
        return {
            key: signal.scaled(factor) if key[0] == self.line else signal
            for key, signal in signals.items()
        }

    def referenced_lines(self) -> tuple[str, ...]:
        return (self.line,)
