"""Signal-chain transforms for control equipment.

Transforms operate on complete analytic signals keyed by
``(line_label, source_index)`` and are owned by
:class:`~quchip.control.equipment.ControlEquipment` (not by individual
drives). The equipment applies them after scheduling and before each
destination drive maps physical I/Q quadratures into the Hamiltonian.

Available transforms
--------------------
- :class:`Delay`: per-line time shift.
- :class:`Gain`: per-line complex scaling (IQ imbalance, attenuation).
- :class:`Crosstalk`: linear leakage from a source line onto a victim line,
  parameterized by amplitude ``beta``, angle ``theta``, and relative ``delay``.
  It is the standard single-parameter model, used by Sheldon et al., PRA 93,
  060302 (2016) for two-qubit gate calibration. Sarovar et al., Quantum 4, 321
  (2020) used it for crosstalk characterization.

Examples
--------
>>> from quchip import ChargeDrive, Crosstalk, Delay, Gain
>>> # Crosstalk between two already-constructed drives:
>>> # xt = Crosstalk(source=drive_a, victim=drive_b, beta=0.02, theta=0.1)
"""

from __future__ import annotations

import numpy as np
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
from quchip.declarative.parameters import (
    KeywordOnlyDeclarativeMeta, parameter, setting, parameter_fields, setting_fields,
    resolve_declared_params, resolve_declared_settings, validate_sign, UNBOUND,
)
from quchip.utils.values import copy_value
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

    Parameters
    ----------
    program : SignalProgram
        Complete envelope, timing, phase, and carrier program.
    carrier : Any or None, default=None
        Carrier frequency in GHz, when uniquely defined.
    phase_reference : Any or None, default=None
        Reference used to track coherent phase alignment.
    """

    program: SignalProgram
    carrier: Any | None = None
    phase_reference: Any | None = None

    @classmethod
    def from_pulse(cls, pulse: Any) -> "AnalyticSignal":
        """Build the complete scheduled signal for one pulse record.

        A ``detuning`` multiplies the envelope by ``exp(-2πi detuning τ)`` at
        local time ``τ = t - start_time``. Its phase is therefore zero at the
        pulse start, and the delivered carrier frequency is
        ``freq + detuning``.

        Parameters
        ----------
        pulse : object
            Scheduled pulse entry with envelope, timing, phase, carrier, and an
            optional ``detuning`` in GHz.
        """
        detuning = getattr(pulse, "detuning", None)
        shape: SignalProgram = EnvelopeRef(pulse.envelope)
        if detuning is not None:
            shape = Multiply((shape, Carrier(freq=TWO_PI * detuning, sign=-1)))
        local = Window(
            child=shape,
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
        carrier = pulse.freq if detuning is None or pulse.freq is None else pulse.freq + detuning
        return cls(program=scheduled, carrier=carrier)

    @property
    def i(self) -> PhysicsExpr:
        """In-phase physical quadrature of the delivered signal."""
        return PhysicsExpr.from_signal(RealPart(self.program), name="I")

    @property
    def q(self) -> PhysicsExpr:
        """Quadrature-phase physical component of the delivered signal."""
        return PhysicsExpr.from_signal(ImagPart(self.program), name="Q")

    def evaluate(self, t: Any, *, xp: Any | None = None) -> Any:
        """Evaluate the complete complex signal.

        Parameters
        ----------
        t : array-like
            Evaluation times in ns.
        xp : array namespace or None, default=None
            Numerical array module; inferred when omitted.
        """
        return evaluate_signal_program(self.program, t, xp=xp)

    def shifted(self, delta_t: Any) -> "AnalyticSignal":
        """Return a delayed signal.

        Parameters
        ----------
        delta_t : Any
            Delay in ns.
        """
        return type(self)(
            program=Shift(self.program, delta_t=delta_t),
            carrier=self.carrier,
            phase_reference=self.phase_reference,
        )

    def scaled(self, factor: Any) -> "AnalyticSignal":
        """Return a signal multiplied by a complex factor.

        Parameters
        ----------
        factor : Any
            Dimensionless complex scale factor.
        """
        return type(self)(
            program=Scale(self.program, factor=factor),
            carrier=self.carrier,
            phase_reference=self.phase_reference,
        )

    def polar_scaled(self, amplitude: Any, theta: Any) -> "AnalyticSignal":
        """Return a signal multiplied by ``amplitude * exp(i theta)``.

        Parameters
        ----------
        amplitude : Any
            Dimensionless amplitude scale.
        theta : Any
            Phase rotation in radians.
        """
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


def _encode_field(value: Any) -> Any:
    if isinstance(value, complex):
        return {"complex": [value.real, value.imag]}
    if isinstance(value, dict):
        return {"dict": [[_encode_field(key), _encode_field(item)] for key, item in value.items()]}
    if isinstance(value, tuple):
        return {"tuple": [_encode_field(item) for item in value]}
    if isinstance(value, list):
        return [_encode_field(item) for item in value]
    if hasattr(value, "shape"):
        return {"array": _encode_field(np.asarray(value).tolist())}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Cannot serialize transform field of type {type(value).__name__}.")


def _decode_field(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"complex"}:
            return complex(*value["complex"])
        if set(value) == {"array"}:
            return np.asarray(_decode_field(value["array"]))
        if set(value) == {"dict"}:
            return {_decode_field(key): _decode_field(item) for key, item in value["dict"]}
        if set(value) == {"tuple"}:
            return tuple(_decode_field(item) for item in value["tuple"])
        raise ValueError("Invalid serialized transform field.")
    return [_decode_field(item) for item in value] if isinstance(value, list) else value


class SignalTransform(Registrable, ABC, registry_root=True, metaclass=KeywordOnlyDeclarativeMeta):
    """A signal-map transform with shared parameter and setting declarations.

    Implement ``apply`` and declare ``serializable=True`` on classes whose
    fields can be saved. Import the extension before loading its instances.

    Parameters
    ----------
    **values : Any
        Settings and numerical parameters declared by the concrete transform.
    """

    _serializable = False

    def __init_subclass__(cls, *, serializable: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._serializable = serializable
        if not serializable:
            cls._registry.pop(cls._type_key(), None)

    def __init__(self, **values: Any) -> None:
        settings = resolve_declared_settings(type(self), values)
        parameters = resolve_declared_params(type(self), values)
        for name, value in {**settings, **parameters}.items():
            if value is UNBOUND:
                raise TypeError(f"Missing required transform field: {name}")
            setattr(self, name, copy_value(value))
        self.validate()

    def validate(self) -> None:
        """Validate relationships between declared fields after construction or rebinding."""

    def __setattr__(self, name: str, value: Any) -> None:
        spec = parameter_fields(type(self)).get(name)
        if spec is not None:
            validate_sign(name, spec, value)
        super().__setattr__(name, value)

    def parameter_values(self) -> dict[str, Any]:
        """Return the transform's declared numerical values."""
        return {name: getattr(self, name) for name in parameter_fields(type(self))}

    def copy(self) -> "SignalTransform":
        """Copy authored fields while preserving native differentiation leaves."""
        copied = copy.copy(self)
        copied.__dict__ = copy_value(vars(self))
        return copied

    def with_parameter_value(self, name: str, value: Any) -> "SignalTransform":
        """Validate and rebind one numerical field on an independent transform.

        Parameters
        ----------
        name : str
            Declared numerical field name.
        value : Any
            Replacement value in the field's declared units.
        """
        if name not in parameter_fields(type(self)):
            raise KeyError(name)
        rebound = self.copy()
        setattr(rebound, name, copy_value(value))
        rebound.validate()
        return rebound

    def to_dict(self) -> dict[str, Any]:
        """Serialize declared fields for a class that opted into persistence."""
        if not self._serializable:
            raise TypeError(f"{type(self).__name__} requires serializable=True to save its instances.")
        fields: dict[str, Any] = {**parameter_fields(type(self)), **setting_fields(type(self))}
        return {**super().to_dict(), **{name: _encode_field(getattr(self, name))
                                       for name, spec in fields.items() if spec.serialize}}

    @classmethod
    def _from_dict_payload(cls, data: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        return cls(**{key: _decode_field(value) for key, value in data.items() if key != "type"})

    @abstractmethod
    def apply(self, signals: SignalMap) -> SignalMap:
        """Return the transformed signal map.

        Parameters
        ----------
        signals : SignalMap
            Signals keyed by control line and source index.
        """

    def referenced_lines(self) -> tuple[str, ...]:
        """Return control-line labels referenced by this transform."""
        return ()

    def without_line(self, line: str) -> "SignalTransform | None":
        """Return this transform without a line, or ``None`` if it must be dropped.

        Parameters
        ----------
        line : str
            Removed control-line label.
        """
        return None if line in self.referenced_lines() else self


class Delay(SignalTransform, serializable=True):
    """Shift every signal on one control line by ``delta_t`` ns.

    Parameters
    ----------
    line : str or object
        Drive label or drive object.
    delta_t : float
        Delay in ns; positive values shift the signal later.
    """

    line: str = setting()
    delta_t: float = parameter()

    def __init__(self, line: str | Any, delta_t: float) -> None:
        _reject_field_endpoint(line, transform="ControlEquipment.Delay")
        object.__setattr__(self, "line", resolve_label(line))
        object.__setattr__(self, "delta_t", delta_t)

    def apply(self, signals: SignalMap) -> SignalMap:
        """Time-shift every signal on :attr:`line` by ``delta_t`` ns.

        Parameters
        ----------
        signals : SignalMap
            Signals keyed by ``(line, source_index)``.
        """
        s = dict(signals)
        for key in list(s):
            if key[0] == self.line:
                s[key] = s[key].shifted(self.delta_t)
        return s

    def referenced_lines(self) -> tuple[str, ...]:
        """Return the line label affected by this delay."""
        return (self.line,)

class Gain(SignalTransform, serializable=True):
    """Scale every signal on one control line by a complex factor.

    Parameters
    ----------
    line : str or object
        Drive label or drive object.
    factor : complex
        Dimensionless amplitude and phase multiplier.
    """

    line: str = setting()
    factor: complex = parameter()

    def __init__(self, line: str | Any, factor: complex) -> None:
        _reject_field_endpoint(line, transform="ControlEquipment.Gain")
        object.__setattr__(self, "line", resolve_label(line))
        object.__setattr__(self, "factor", factor)

    def apply(self, signals: SignalMap) -> SignalMap:
        """Scale every signal on :attr:`line` by the complex ``factor``.

        Parameters
        ----------
        signals : SignalMap
            Signals keyed by ``(line, source_index)``.
        """
        s = dict(signals)
        for key in list(s):
            if key[0] == self.line:
                s[key] = s[key].scaled(self.factor)
        return s

    def referenced_lines(self) -> tuple[str, ...]:
        """Return the line label affected by this gain."""
        return (self.line,)

class Crosstalk(SignalTransform, serializable=True):
    r"""Linear crosstalk from a source drive line onto a victim line.

    For each scheduled operation on the source line, adds

    .. math::

       \beta\, e^{i\theta}\, s_\mathrm{src}(t - \Delta t)

    onto the victim line. :math:`s_\mathrm{src}` is the complete source signal,
    including its carrier, phase, and both quadratures, so delaying it includes
    the carrier phase :math:`2\pi f\Delta t` without a separate correction. See
    Balewski et al., arXiv:2502.05362; Sheldon et al., PRA 93, 060302 (2016);
    Sarovar et al., Quantum 4, 321 (2020).

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

    source: str = setting()
    victim: str = setting()
    beta: Any = parameter()
    theta: Any = parameter(default=0.0)
    delay: Any = parameter(default=0.0)

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
        """Add the phase-rotated, delayed source signal onto the victim line.

        Parameters
        ----------
        signals : SignalMap
            Signals keyed by ``(line, source_index)``.
        """
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
        """Return source and victim labels referenced by this crosstalk edge."""
        return (self.source, self.victim)
