"""Classical control lines and their quantum Hamiltonian couplings.

A drive builds the complete scheduled analytic signal, then maps its physical
I/Q quadratures to target-local quantum operators. Control equipment may alter
that signal before the Hamiltonian mapping. Projection, frames, approximation,
embedding, unit conversion, and backend lowering remain engine responsibilities.

Conventions:

- Frequencies are GHz; times are ns.
- Operators are returned in the device's authored local basis — embedding
  into the full chip Hilbert space is the engine's job.

References
----------
- Krantz et al., *A quantum engineer's guide to superconducting qubits*,
  APR 6, 021318 (2019) — microwave control of transmons (Sec. IV).
- Koch et al., PRA 76, 042319 (2007) — charge vs flux noise and drives
  in the transmon regime.
"""

from __future__ import annotations

import copy
import inspect
from typing import Any, ClassVar

from quchip.control.signal import AnalyticSignal
from quchip.declarative.dissipation import CollapseChannel, normalize_dissipation
from quchip.declarative.expr import ParameterNamespace, as_operator_expr
from quchip.declarative.ops import LocalOps
from quchip.declarative.parameters import (
    DriveDeclarativeMeta,
    Parameter,
    constructor_field,
    parameter_fields,
    resolve_declared_params,
    resolve_declared_settings,
    serializable_value,
    setting_fields,
    validate_declared_fields,
    validate_sign,
)
from quchip.devices.base import BaseDevice
from quchip.devices.protocols import ChargeCoupled, FluxCoupled, PhaseCoupled
from quchip.utils.labeling import auto_label
from quchip.utils.registry import Registrable


def _device_operator_expr(device: Any, value: Any, *, name: str) -> Any:
    """Normalize a device capability into its authored local expression."""
    dimension = (
        device.local_space().dimension
        if hasattr(device, "local_space")
        else int(device.levels)
    )
    return as_operator_expr(
        value,
        labels=(device.label,),
        dims=(dimension,),
        name=name,
    )

def _synthesize_drive_init(cls: type["BaseDrive"]) -> Any:
    """Build a constructor from a target and declared drive fields."""
    trailing = tuple(
        inspect.Parameter(
            name,
            inspect.Parameter.KEYWORD_ONLY,
            default=inspect.Parameter.empty if spec.required else spec.default,
        )
        for name, spec in parameter_fields(cls).items()
    ) + tuple(
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=spec.default)
        for name, spec in setting_fields(cls).items()
    ) + (
        inspect.Parameter("label", inspect.Parameter.KEYWORD_ONLY, default=None),
    )
    signature = inspect.Signature(
        (
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("target", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None),
            *trailing,
        )
    )

    def __init__(self: BaseDrive, *args: Any, **kwargs: Any) -> None:
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        arguments.pop("self")
        target = arguments.pop("target")
        label = arguments.pop("label")
        BaseDrive.__init__(self, target=target, label=label, **arguments)

    __init__.__signature__ = signature  # type: ignore[attr-defined]
    __init__.__qualname__ = f"{cls.__qualname__}.__init__"
    __init__.__doc__ = f"Initialize {cls.__name__} from its target and declared fields."
    return __init__


class BaseDrive(Registrable, registry_root=True, metaclass=DriveDeclarativeMeta):
    """Base class for classical control lines attached to one quantum target.

    Drives own their local Hamiltonian contribution and
    are auto-labelled from their ``_type_prefix`` (e.g. ``charge_0``,
    ``flux_0``) unless *label* is given. Subclasses are auto-registered
    for serialization via the shared
    :class:`~quchip.utils.registry.Registrable` mixin.

    Parameters
    ----------
    target : BaseDevice, BaseCoupling, str, or None
        Target accepted by the concrete drive. A :class:`DeviceDrive` targets
        a device; a :class:`CouplingDrive` targets a coupling. The target may
        be connected later or resolved by label through :class:`Chip`.
    label : str | None
        Optional explicit label; otherwise auto-generated.
    Examples
    --------
    >>> from quchip import DuffingTransmon, ChargeDrive
    >>> q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3)
    >>> drive = ChargeDrive(target=q)
    >>> drive.device_label == q.label
    True
    """

    _type_prefix: ClassVar[str] = "drive"

    target: Any = constructor_field(default=None, kw_only=False)
    label: Any = constructor_field(default=None, kw_only=True)

    __quchip_param_fields__: ClassVar[dict[str, Parameter]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        validate_declared_fields(cls)
        cls.__quchip_param_fields__ = parameter_fields(cls)
        if "__init__" not in cls.__dict__:
            cls.__init__ = _synthesize_drive_init(cls)  # type: ignore[method-assign]

    def __init__(
        self,
        target: Any | str | None = None,
        *,
        label: str | None = None,
        **params: Any,
    ) -> None:
        params = dict(params)
        settings = resolve_declared_settings(type(self), params)
        values = resolve_declared_params(
            type(self), params, fields=type(self).__quchip_param_fields__
        )
        self.label = label if label is not None else auto_label(type(self)._type_prefix)
        self._target: Any | str | None = None
        for name, value in settings.items():
            setattr(self, name, value)
        for name, value in values.items():
            setattr(self, name, value)
        if target is not None:
            if isinstance(target, str):
                self._target = target
            else:
                self.connect(target)

    def connect(self, target: Any) -> None:
        """Attach this device-drive implementation to *target*.

        If previously attached, the drive is removed from the old device's
        ``_connected_drives`` list. :class:`CouplingDrive` overrides this
        handshake because couplings do not own connected-drive lists.
        """
        old_target = self._target
        if old_target is not None and not isinstance(old_target, str) and old_target is not target:
            old_target._connected_drives = [d for d in old_target._connected_drives if d is not self]
        self._target = target
        target.connect(self)

    def parameter_values(self) -> dict[str, Any]:
        """Return drive-owned bindable values declared by the subclass."""
        return {name: getattr(self, name) for name in type(self).__quchip_param_fields__}

    def set_parameter_value(self, name: str, value: Any) -> None:
        """Apply one drive-owned value on an isolated drive copy."""
        spec = type(self).__quchip_param_fields__.get(name)
        if spec is None:
            raise KeyError(name)
        validate_sign(name, spec, value)
        setattr(self, name, value)

    @property
    def device_label(self) -> str | None:
        """Label of the connected device, or ``None`` if unconnected."""
        target = self._target
        if target is None:
            return None
        return target if isinstance(target, str) else target.label

    @property
    def target_label(self) -> str | None:
        """Label of this drive's target, or ``None`` if unconnected.

        Device-target drives alias :attr:`device_label`;
        :class:`ParametricDrive` resolves its coupling target instead.
        """
        return self.device_label

    def dissipation(
        self,
        target: BaseDevice,
        op: LocalOps,
        p: ParameterNamespace,
    ) -> tuple[CollapseChannel, ...]:
        """Return target-local Lindblad channels contributed by this line."""
        _ = (target, op, p)
        return ()

    def _collapse_channels_with_paths(
        self,
        target: BaseDevice,
    ) -> tuple[tuple[CollapseChannel, tuple[str, ...]], ...]:
        """Normalize authored line dissipation and infer dependencies."""
        fields = type(self).__quchip_param_fields__
        op = LocalOps(target.label, target.local_space(), device=target)
        p = ParameterNamespace(f"drive.{self.label}", fields)
        bindings = {
            f"drive.{self.label}.{name}": getattr(self, name)
            for name in fields
        }
        return normalize_dissipation(
            self.dissipation(target, op, p),
            labels=(target.label,),
            dims=(target.local_space().dimension,),
            owner=self,
            scope=f"drive.{self.label}",
            allowed=fields,
            bindings=bindings,
        )

    def signal(self, pulse: Any, target: Any) -> AnalyticSignal:
        """Build the complete scheduled analytic signal for one pulse."""
        _ = target
        return AnalyticSignal.from_pulse(pulse)

    def hamiltonian(self, target: Any, signal: AnalyticSignal) -> Any:
        """Map a delivered classical signal to target-local quantum physics."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement hamiltonian(target, signal)"
        )

    def physics_notes(self) -> list[str]:
        """Return human-readable declarations of this drive's approximations.

        Subclasses append their physical coupling details to the shared target
        line. Aggregated by :meth:`Chip.physics_notes`.
        """
        target = self.target_label if self.target_label is not None else "<unconnected>"
        return [f"Target: '{target}'"]

    def copy(self, *, target: BaseDevice | None = None) -> "BaseDrive":
        """Return a shallow copy, optionally rebound to a new target."""
        cloned = copy.copy(self)
        cloned._target = None
        if target is not None:
            cloned.connect(target)
        return cloned

    def to_dict(self) -> dict[str, Any]:
        """Serialize into a JSON-safe dictionary."""
        data = super().to_dict()
        for name, spec in type(self).__quchip_param_fields__.items():
            if spec.serialize:
                data[name] = serializable_value(getattr(self, name))
        for name, setting_spec in setting_fields(type(self)).items():
            if setting_spec.serialize:
                data[name] = getattr(self, name)
        data["target_label"] = self.target_label
        data["label"] = self.label
        return data

    @classmethod
    def _from_dict_payload(
        cls,
        d: dict[str, Any],
        target: Any | str | None = None,
    ) -> "BaseDrive":
        """Reconstruct a concrete drive, rebinding it to *target*.

        Shared by every standard drive whose only persisted state is its
        label and declared fields; the registry root's :meth:`from_dict` resolves
        the concrete class from the serialized ``type`` and forwards here.
        Subclasses with extra serialized state override :meth:`from_dict`.
        """
        allowed = {"type", "target_label", "label"}
        allowed.update(
            name
            for name, spec in cls.__quchip_param_fields__.items()
            if spec.serialize
        )
        allowed.update(
            name
            for name, spec in setting_fields(cls).items()
            if spec.serialize
        )
        unexpected = set(d) - allowed
        if unexpected:
            raise TypeError(
                f"Unsupported serialized fields for {cls.__name__}: "
                + ", ".join(sorted(unexpected))
            )
        params = {
            name: d[name]
            for name, spec in cls.__quchip_param_fields__.items()
            if spec.serialize and name in d
        }
        settings = {
            name: d[name]
            for name, spec in setting_fields(cls).items()
            if spec.serialize and name in d
        }
        return cls(target=target, label=d.get("label"), **params, **settings)

    def __repr__(self) -> str:
        attrs = []
        if self.target_label is not None:
            attrs.append(f"target={self.target_label!r}")
        return f"{type(self).__name__}({', '.join(attrs)})"


class DeviceDrive(BaseDrive):
    """Drive authoring base for a device-local Hamiltonian."""


class CouplingDrive(BaseDrive):
    """Drive authoring base for a two-endpoint coupling Hamiltonian.

    Subclasses implement :meth:`hamiltonian` for the coupling physics they
    accept. The base class imposes no parametric-interaction requirement.
    """

    @property
    def device_label(self) -> None:
        """Return ``None`` because a coupling drive has no device target."""
        return None

    @property
    def target_label(self) -> str | None:
        """Label of the connected coupling, if any."""
        target = self._target
        if target is None:
            return None
        return target if isinstance(target, str) else target.label

    def connect(self, target: Any) -> None:
        """Attach this line to a coupling without a device-side handshake."""
        self._target = target


class ChargeDrive(DeviceDrive):
    r"""Microwave charge drive on a transmon-like device.

    Contributes the standard charge-coupling Hamiltonian

    .. math::

       H_d(t) = \epsilon(t)\, i(\hat a - \hat a^\dagger)

    with :math:`\epsilon(t)` the in-phase quadrature of the complete
    delivered classical signal. This is the canonical transmon microwave drive
    (Koch et al., PRA 76, 042319 (2007); Krantz et al., APR 6, 021318
    (2019), Eq. 90).

    Examples
    --------
    >>> from quchip import DuffingTransmon, ChargeDrive
    >>> q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3)
    >>> drive = ChargeDrive(target=q)
    >>> drive.target_label == q.label
    True
    """

    _type_prefix: ClassVar[str] = "charge"

    def hamiltonian(self, device: Any, signal: AnalyticSignal) -> Any:
        if not isinstance(device, ChargeCoupled):
            raise TypeError(
                f"ChargeDrive requires {type(device).__name__} to define "
                "charge_coupling_operator()."
            )
        return signal.i * _device_operator_expr(
            device,
            device.charge_coupling_operator(),
            name=rf"\hat H_{{charge,{getattr(device, 'label')}}}",
        )

    def physics_notes(self) -> list[str]:
        return super().physics_notes() + [
            "Drive coupling: delivered in-phase signal times the device charge operator"
        ]


class PhaseDrive(DeviceDrive):
    r"""Microwave phase drive coupling to :math:`\hat a + \hat a^\dagger`.

    Same carrier machinery as :class:`ChargeDrive` but with an
    in-phase (rather than quadrature) coupling. Useful when modelling
    phase-noise channels or drives whose physical coupling is already
    referenced to the field quadrature. See Krantz et al. 2019, Sec.
    IV.A for the two conventions.
    """

    _type_prefix: ClassVar[str] = "phase"
    def hamiltonian(self, device: Any, signal: AnalyticSignal) -> Any:
        if not isinstance(device, PhaseCoupled):
            raise TypeError(
                f"PhaseDrive requires {type(device).__name__} to define "
                "phase_coupling_operator()."
            )
        return signal.i * _device_operator_expr(
            device,
            device.phase_coupling_operator(),
            name=rf"\hat H_{{phase,{getattr(device, 'label')}}}",
        )

    def physics_notes(self) -> list[str]:
        return super().physics_notes() + [
            "Drive coupling: delivered in-phase signal times the device phase operator"
        ]


class FluxDrive(DeviceDrive):
    r"""Real-valued flux drive coupling to :math:`\hat n`.

    The delivered signal's in-phase quadrature modulates the device frequency
    through its flux-coupling operator
    (Koch et al. 2007, Sec. II; Krantz et al. 2019, Sec. V.A on flux
    tunability).

    Examples
    --------
    >>> from quchip import DuffingTransmon, FluxDrive
    >>> q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3)
    >>> flux = FluxDrive(target=q)
    >>> flux.target_label == q.label
    True
    """

    _type_prefix: ClassVar[str] = "flux"

    def hamiltonian(self, device: Any, signal: AnalyticSignal) -> Any:
        if not isinstance(device, FluxCoupled):
            raise TypeError(
                f"FluxDrive requires {type(device).__name__} to define "
                "flux_coupling_operator()."
            )
        return signal.i * _device_operator_expr(
            device,
            device.flux_coupling_operator(),
            name=rf"\hat H_{{flux,{getattr(device, 'label')}}}",
        )

    def physics_notes(self) -> list[str]:
        return super().physics_notes() + [
            "Drive coupling: delivered in-phase signal times the device flux operator"
        ]


class ParametricDrive(CouplingDrive):
    """Control line pumping a modulable coupling's strength δ(t) in GHz.

    Targets a coupling (object or label string; labels late-bind via
    :meth:`Chip.connect`). The scheduled envelope is the *real amplitude*
    ``A(t)``: with an explicit carrier the pump is
    ``δ(t) = A(t)·cos(2π·freq·t - phase)``; with ``freq`` omitted the pump is
    carrier-free, ``δ(t) = A(t)`` directly. Approximation belongs to the
    chip's selected engine strategy, not to the drive.

    Accepted couplings implement
    :meth:`~quchip.declarative.models.CouplingModel.parametric_interaction`;
    a static coupling raises ``TypeError`` naming the hook.

    Parameters
    ----------
    coupling : BaseCoupling | str
        Modulable coupling to pump, given as the coupling object or its
        label. A string label late-binds to the coupling instance via
        :meth:`Chip.connect`.
    label : str | None
        Optional explicit label; otherwise auto-generated from
        ``"parametric"``.

    Raises
    ------
    TypeError
        *coupling* does not implement
        :meth:`~quchip.declarative.models.CouplingModel.parametric_interaction`
        (a static coupling), or an unexpected keyword argument is passed.
    """

    _type_prefix: ClassVar[str] = "parametric"

    def __repr__(self) -> str:
        """Return a compact pump-line summary naming the coupling."""
        return f"{type(self).__name__}(label='{self.label}', coupling='{self.target_label}')"

    def connect(self, coupling: Any) -> None:
        """Attach this line after confirming that the coupling is modulable."""
        _probe_modulable(coupling)
        self._target = coupling

    def hamiltonian(self, coupling: Any, signal: AnalyticSignal) -> Any:
        operator = coupling._bind_parametric_interaction()
        if operator is None:
            raise TypeError(
                f"{type(coupling).__name__} is not modulable: its "
                "parametric_interaction() hook returns None."
            )
        return signal.i * operator

    def physics_notes(self) -> list[str]:
        return super().physics_notes() + [
            "Edge pump: delivered in-phase signal multiplies the coupling's parametric structure"
        ]

def _probe_modulable(coupling: Any) -> None:
    """Raise the teaching TypeError when *coupling* declines the parametric hook."""
    from quchip.declarative.models import _symbolic_parameters
    from quchip.declarative.ops import EndpointOps
    probe = getattr(coupling, "parametric_interaction", None)
    expr = None
    if probe is not None:
        if getattr(coupling, "is_resolved", False):
            expr = coupling._bind_parametric_interaction()
        else:
            from quchip.devices.spaces import FockSpace

            expr = probe(
                EndpointOps(label=coupling.device_a_label, space=FockSpace(2)),
                EndpointOps(label=coupling.device_b_label, space=FockSpace(2)),
                _symbolic_parameters(coupling),
            )
    if expr is None:
        raise TypeError(
            f"{type(coupling).__name__} is not modulable: its parametric_interaction() hook "
            "returns None. Implement parametric_interaction() on "
            "the coupling (see CouplingModel), or use a modulable coupling such as TunableCapacitive."
        )
