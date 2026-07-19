"""Signal-chain transforms for control equipment.

Transforms operate on a :data:`SignalMap` keyed by
``(drive_label, drive_index)`` and are owned by
:class:`~quchip.control.equipment.ControlEquipment` (not by individual
drives). The equipment applies them in order, after
drives produce their raw line signals and before the engine assembles
the Hamiltonian.

Available transforms
--------------------
- :class:`Delay` — per-line time shift.
- :class:`Gain` — per-line complex scaling (IQ imbalance, attenuation).
- :class:`Crosstalk` — linear leakage from a source line onto a victim
  line, parameterized by amplitude ``beta``, angle ``theta``, and
  relative ``delay``. This is the standard single-parameter crosstalk
  model used e.g. in Sheldon et al., PRA 93, 060302 (2016) for
  two-qubit gate calibration, and in Sarovar et al., Quantum 4, 321
  (2020) for crosstalk characterization.

Examples
--------
>>> from quchip import ChargeDrive, Crosstalk, Delay, Gain
>>> # Crosstalk between two already-constructed drives:
>>> # xt = Crosstalk(source=drive_a, victim=drive_b, beta=0.02, theta=0.1)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from quchip.engine.ir import Add, PolarScale, Scale, Shift, SignalProgram
from quchip.utils.labeling import resolve_label
from quchip.utils.registry import Registrable

SignalKey = tuple[str, int]  # (drive_label, drive_index)
SignalMap = dict[SignalKey, SignalProgram]


class SignalTransform(Registrable, ABC, registry_root=True):
    """Abstract base for signal-map transforms, auto-registered for serialization.

    The type registry, the ``{"type": ...}`` :meth:`to_dict` stamp, and the
    ``from_dict`` dispatch are owned by the shared
    :class:`~quchip.utils.registry.Registrable` mixin; the parameter-less
    default reconstruction (``cls()``) covers transforms that carry no
    persisted state, while payload-carrying transforms override
    :meth:`to_dict` / :meth:`from_dict`.
    """

    @abstractmethod
    def apply(self, signals: SignalMap) -> SignalMap:
        """Return the transformed signal map."""

    def referenced_lines(self) -> tuple[str, ...]:
        """Return control-line labels referenced by this transform."""
        return ()

    def without_line(self, line: str) -> "SignalTransform | None":
        """Return this transform without *line*, or ``None`` when it must be dropped."""
        return None if line in self.referenced_lines() else self


@dataclass(frozen=True)
class Delay(SignalTransform):
    """Shift every signal on *line* in time by ``delta_t`` ns."""

    line: str
    delta_t: float

    def __init__(self, line: str | Any, delta_t: float) -> None:
        object.__setattr__(self, "line", resolve_label(line))
        object.__setattr__(self, "delta_t", delta_t)

    def apply(self, signals: SignalMap) -> SignalMap:
        """Time-shift every signal on :attr:`line` by ``delta_t`` ns."""
        s = dict(signals)
        for key in list(s):
            if key[0] == self.line:
                s[key] = Shift(s[key], delta_t=self.delta_t)
        return s

    def referenced_lines(self) -> tuple[str, ...]:
        return (self.line,)

    def to_dict(self) -> dict[str, Any]:
        """Serialize into a JSON-safe dictionary."""
        data = super().to_dict()
        data["line"] = self.line
        data["delta_t"] = float(self.delta_t)
        return data

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Delay":
        return cls(line=str(d["line"]), delta_t=float(d["delta_t"]))


@dataclass(frozen=True)
class Gain(SignalTransform):
    """Scale every signal on *line* by a complex *factor*."""

    line: str
    factor: complex

    def __init__(self, line: str | Any, factor: complex) -> None:
        object.__setattr__(self, "line", resolve_label(line))
        object.__setattr__(self, "factor", factor)

    def apply(self, signals: SignalMap) -> SignalMap:
        """Scale every signal on :attr:`line` by the complex ``factor``."""
        s = dict(signals)
        for key in list(s):
            if key[0] == self.line:
                s[key] = Scale(s[key], factor=self.factor)
        return s

    def referenced_lines(self) -> tuple[str, ...]:
        return (self.line,)

    def to_dict(self) -> dict[str, Any]:
        """Serialize into a JSON-safe dictionary."""
        data = super().to_dict()
        data["line"] = self.line
        data["real"] = float(complex(self.factor).real)
        data["imag"] = float(complex(self.factor).imag)
        return data

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Gain":
        return cls(
            line=str(d["line"]),
            factor=complex(float(d.get("real", 0.0)), float(d.get("imag", 0.0))),
        )


@dataclass(frozen=True)
class Crosstalk(SignalTransform):
    r"""Linear crosstalk from a source drive line onto a victim line.

    For each scheduled operation on the source line, adds

    .. math::

       \beta\, e^{i\theta}\, s_\mathrm{src}(t - \Delta t)

    onto the victim line, where :math:`s_\mathrm{src}` is the source's
    raw line signal — the frame-agnostic, carrier-free envelope signal
    the signal chain operates on, before stage 2 attaches any carrier
    or RWA modulation. Fidelity to classical microwave crosstalk is
    preserved downstream instead: stage 2 builds the victim's
    Hamiltonian term using the *source* drive's own carrier frequency
    for the leaked entry, not the victim's, so a leaked pulse still
    lands at the source's frequency (Balewski et al., arXiv:2502.05362,
    Eq. (3); Sheldon et al., PRA 93, 060302 (2016); Sarovar et al.,
    Quantum 4, 321 (2020)). The ``delay`` shifts the *baseband
    envelope*; the carrier-phase part of a physical path delay
    (:math:`2\pi f \tau`) belongs in ``theta``.

    Parameters
    ----------
    source : str | BaseDrive
        Source drive or its label.
    victim : str | BaseDrive
        Victim drive or its label.
    beta : float
        Leakage amplitude (dimensionless).
    theta : float
        Phase shift applied to the leaked signal, radians.
    delay : float
        Time shift of the leaked signal relative to the source, ns.
    """

    source: str
    victim: str
    beta: float
    theta: float = 0.0
    delay: float = 0.0

    def __init__(
        self,
        source: str | Any,
        victim: str | Any,
        beta: float,
        theta: float = 0.0,
        delay: float = 0.0,
    ) -> None:
        object.__setattr__(self, "source", resolve_label(source))
        object.__setattr__(self, "victim", resolve_label(victim))
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "delay", delay)

    def apply(self, signals: SignalMap) -> SignalMap:
        """Add the phase-rotated, delayed source signal onto the victim line."""
        output = dict(signals)
        for key, signal in signals.items():
            if key[0] != self.source:
                continue
            leaked: SignalProgram = PolarScale(
                child=Shift(signal, delta_t=self.delay),
                amplitude=self.beta,
                theta=self.theta,
            )
            victim_key = (self.victim, key[1])
            existing = output.get(victim_key)
            output[victim_key] = leaked if existing is None else Add((existing, leaked))
        return output

    def referenced_lines(self) -> tuple[str, ...]:
        return (self.source, self.victim)

    def to_dict(self) -> dict[str, Any]:
        """Serialize into a JSON-safe dictionary."""
        data = super().to_dict()
        data["source"] = self.source
        data["victim"] = self.victim
        data["beta"] = float(self.beta)
        data["theta"] = float(self.theta)
        data["delay"] = float(self.delay)
        return data

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Crosstalk":
        return cls(
            source=str(d["source"]),
            victim=str(d["victim"]),
            beta=float(d["beta"]),
            theta=float(d.get("theta", 0.0)),
            delay=float(d.get("delay", 0.0)),
        )
