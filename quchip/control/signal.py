"""Signal-chain transforms for control equipment.

Transforms operate on complete analytic signals keyed by
``(line_label, source_index)`` and are owned by
:class:`~quchip.control.equipment.ControlEquipment` (not by individual
drives). The equipment applies them after scheduling and before each
destination drive maps physical I/Q quadratures into the Hamiltonian.

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

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from quchip.declarative.expr import PhysicsExpr
from quchip.engine.ir import (
    Add,
    Carrier,
    EnvelopeRef,
    ImagPart,
    Multiply,
    PolarScale,
    RealPart,
    Scale,
    Shift,
    SignalProgram,
    Window,
    evaluate_signal_program,
)
from quchip.utils.constants import TWO_PI
from quchip.utils.labeling import resolve_label
from quchip.utils.registry import Registrable

SignalKey = tuple[str, int]  # (line_label, source_index)


def _reject_field_endpoint(value: Any, *, transform: str) -> None:
    """Keep field propagation in PortNetwork rather than control equipment."""
    from quchip.control.field import CoherentInput

    if isinstance(value, CoherentInput):
        raise TypeError(
            f"{transform} does not accept an external-plane input. Use PortNetwork "
            "scattering or a reference-plane delay for field propagation."
        )


@dataclass(frozen=True)
class AnalyticSignal:
    """Complete complex classical signal delivered on one control line.

    ``program`` includes the envelope, schedule timing and phase, and any
    carrier. Classical equipment transforms this complete value before a
    drive maps its physical quadratures into the quantum Hamiltonian.
    """

    program: SignalProgram
    carrier: Any | None = None
    phase_reference: Any | None = None

    @classmethod
    def from_pulse(cls, pulse: Any) -> "AnalyticSignal":
        """Build the complete scheduled signal for one pulse record."""
        local = Window(
            child=EnvelopeRef(pulse.envelope),
            start=0.0,
            stop=pulse.envelope.duration,
        )
        scheduled: SignalProgram = PolarScale(
            child=Shift(local, delta_t=pulse.start_time),
            amplitude=1.0,
            theta=pulse.phase_offset,
        )
        if pulse.freq is not None:
            scheduled = Multiply(
                (scheduled, Carrier(freq=TWO_PI * pulse.freq, sign=-1))
            )
        return cls(program=scheduled, carrier=pulse.freq)

    @property
    def i(self) -> PhysicsExpr:
        """In-phase physical quadrature of the delivered signal."""
        return PhysicsExpr.from_signal(RealPart(self.program), name="I")

    @property
    def q(self) -> PhysicsExpr:
        """Quadrature-phase physical component of the delivered signal."""
        return PhysicsExpr.from_signal(ImagPart(self.program), name="Q")

    def evaluate(self, t: Any, *, xp: Any | None = None) -> Any:
        """Evaluate the complete complex signal at time *t*."""
        return evaluate_signal_program(self.program, t, xp=xp)

    def shifted(self, delta_t: Any) -> "AnalyticSignal":
        """Return the signal delayed by ``delta_t`` ns."""
        return type(self)(
            program=Shift(self.program, delta_t=delta_t),
            carrier=self.carrier,
            phase_reference=self.phase_reference,
        )

    def scaled(self, factor: Any) -> "AnalyticSignal":
        """Return the signal multiplied by a complex factor."""
        return type(self)(
            program=Scale(self.program, factor=factor),
            carrier=self.carrier,
            phase_reference=self.phase_reference,
        )

    def polar_scaled(self, amplitude: Any, theta: Any) -> "AnalyticSignal":
        """Return the signal multiplied by ``amplitude * exp(i theta)``."""
        return type(self)(
            program=PolarScale(self.program, amplitude=amplitude, theta=theta),
            carrier=self.carrier,
            phase_reference=self.phase_reference,
        )

    def __add__(self, other: "AnalyticSignal") -> "AnalyticSignal":
        carrier = self.carrier if self.carrier is other.carrier else None
        phase_reference = (
            self.phase_reference
            if self.phase_reference is other.phase_reference
            else None
        )
        return type(self)(
            program=Add((self.program, other.program)),
            carrier=carrier,
            phase_reference=phase_reference,
        )


SignalMap = dict[SignalKey, AnalyticSignal]


class SignalTransform(Registrable, ABC, registry_root=True):
    """Abstract base for signal-map transforms, auto-registered for serialization.

    The type registry, the ``{"type": ...}`` :meth:`to_dict` stamp, and the
    ``from_dict`` dispatch are owned by the shared
    :class:`~quchip.utils.registry.Registrable` mixin; the parameter-less
    default reconstruction (``cls()``) covers transforms that carry no
    persisted state, while payload-carrying transforms override
    :meth:`to_dict` / :meth:`from_dict`.
    """

    _parameter_names: tuple[str, ...] = ()

    def parameter_values(self) -> dict[str, Any]:
        """Return transform-owned bindable values declared by the subclass."""
        return {name: getattr(self, name) for name in self._parameter_names}

    def with_parameter_value(self, name: str, value: Any) -> "SignalTransform":
        """Return this transform with one declared numerical value replaced."""
        if name not in self._parameter_names:
            raise KeyError(name)
        rebound = copy.copy(self)
        object.__setattr__(rebound, name, value)
        return rebound

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
    _parameter_names = ("delta_t",)

    def __init__(self, line: str | Any, delta_t: float) -> None:
        _reject_field_endpoint(line, transform="ControlEquipment.Delay")
        object.__setattr__(self, "line", resolve_label(line))
        object.__setattr__(self, "delta_t", delta_t)

    def apply(self, signals: SignalMap) -> SignalMap:
        """Time-shift every signal on :attr:`line` by ``delta_t`` ns."""
        s = dict(signals)
        for key in list(s):
            if key[0] == self.line:
                s[key] = s[key].shifted(self.delta_t)
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
    _parameter_names = ("factor",)

    def __init__(self, line: str | Any, factor: complex) -> None:
        _reject_field_endpoint(line, transform="ControlEquipment.Gain")
        object.__setattr__(self, "line", resolve_label(line))
        object.__setattr__(self, "factor", factor)

    def apply(self, signals: SignalMap) -> SignalMap:
        """Scale every signal on :attr:`line` by the complex ``factor``."""
        s = dict(signals)
        for key in list(s):
            if key[0] == self.line:
                s[key] = s[key].scaled(self.factor)
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

    onto the victim line. :math:`s_\mathrm{src}` is the complete source
    signal, including its carrier, phase, and both quadratures. Delaying it
    therefore includes the carrier phase :math:`2\pi f\Delta t` without a
    separate correction (Balewski et al., arXiv:2502.05362; Sheldon et al.,
    PRA 93, 060302 (2016); Sarovar et al., Quantum 4, 321 (2020)).

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
    _parameter_names = ("beta", "theta", "delay")

    def __init__(
        self,
        source: str | Any,
        victim: str | Any,
        beta: float,
        theta: float = 0.0,
        delay: float = 0.0,
    ) -> None:
        _reject_field_endpoint(source, transform="ControlEquipment.Crosstalk")
        _reject_field_endpoint(victim, transform="ControlEquipment.Crosstalk")
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
            leaked = signal.shifted(self.delay).polar_scaled(self.beta, self.theta)
            victim_key = (self.victim, key[1])
            existing = output.get(victim_key)
            output[victim_key] = leaked if existing is None else existing + leaked
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
