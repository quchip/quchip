"""Drive classes and Hamiltonian channel definitions.

Drives model the classical control lines that couple a quantum device to
external fields — microwave charge/phase drives on transmons, real-valued
flux drives on tunable devices, and so on. Each drive owns its local
Hamiltonian contribution: it exposes an operator and a
:class:`~quchip.control.signal_spec.DriveModulation` tag that together
describe how its line signal enters the system Hamiltonian. Signal
transforms (delays, gains, crosstalk) live separately on
:class:`~quchip.control.equipment.ControlEquipment`.

Drives are frame-agnostic: they emit a
:class:`~quchip.control.signal_spec.DriveSignalSpec` in ordinary GHz
and never construct engine IR nodes. Stage 2 of the engine is the sole
place where the spec is composed with the resolved rotating frame into a
:class:`~quchip.engine.ir.SignalProgram`.

Conventions:

- Frequencies are GHz; times are ns.
- Operators are returned in the device's computational basis — embedding
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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from quchip.backend.protocol import Operator
from quchip.control.signal_spec import DriveSignalSpec, DriveModulation
from quchip.devices.base import BaseDevice
from quchip.devices.protocols import ChargeCoupled, FluxCoupled, PhaseCoupled
from quchip.utils.labeling import auto_label
from quchip.utils.registry import Registrable

# Note: ``DriveOp`` appears only as a string annotation on
# :meth:`BaseDrive.signal_spec` — the class lives in
# :mod:`quchip.engine.ir` but must not be imported here, even under
# ``TYPE_CHECKING``, because drives are strictly frame-agnostic and
# cannot depend on engine IR.


@dataclass(frozen=True)
class DriveChannel:
    """One Hamiltonian channel exposed by a drive on a target device.

    A channel pairs a local-basis operator with a
    :class:`~quchip.control.signal_spec.DriveModulation` tag that tells the
    engine how the channel's coefficient is built from the drive's line
    signal. A drive may expose multiple channels if, for example, it
    contributes both in-phase and quadrature couplings.
    """

    operator: Operator
    modulation: DriveModulation


class BaseDrive(Registrable, registry_root=True):
    """Base class for classical control lines driving a single device.

    Drives own their local Hamiltonian contribution and
    are auto-labelled from their ``_type_prefix`` (e.g. ``charge_0``,
    ``flux_0``) unless *label* is given. Subclasses are auto-registered
    for serialization via the shared
    :class:`~quchip.utils.registry.Registrable` mixin.

    Parameters
    ----------
    target : BaseDevice | None
        Device to attach this drive to. May be connected later via
        :meth:`connect`.
    label : str | None
        Optional explicit label; otherwise auto-generated.
    rwa : bool | None
        Per-drive RWA override. ``None`` means follow the chip-level
        setting.

    Examples
    --------
    >>> from quchip import DuffingTransmon, ChargeDrive
    >>> q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3)
    >>> drive = ChargeDrive(target=q)
    >>> drive.device_label == q.label
    True
    """

    _type_prefix: str = "drive"

    #: Structural dispatch key for equipment/chip/sequence code: ``"device"``
    #: for drives targeting a device (the default), ``"edge"`` for drives
    #: targeting a coupling (see :class:`ParametricDrive`).
    target_kind: ClassVar[str] = "device"

    #: True for microwave-style drives whose carrier frequency must be supplied
    #: explicitly when scheduling (cannot default to the device drive frequency).
    _carrier_required: bool = False

    #: True for microwave-style drives whose carrier frequency defaults to the
    #: target device's :attr:`~BaseDevice.drive_freq` when not given.
    _defaults_to_device_drive_freq: bool = False

    #: True for drives that participate in software-Z (virtual-Z) tracking —
    #: charge/phase microwave drives. Baseband flux drives leave virtual-Z
    #: phases unchanged.
    _accepts_virtual_z: bool = False

    # ------------------------------------------------------------------
    # Declarative local-channel dispatch.
    #
    # A simple single-channel drive declares the five attributes below and
    # inherits :meth:`local_channels` / :meth:`physics_notes` from the base
    # — no copy-pasted protocol-dispatch body. Drives with a non-standard
    # decomposition (e.g. two-photon parametric drives) leave the protocol
    # unset and override :meth:`local_channels` directly.
    # ------------------------------------------------------------------

    #: Device Protocol (ChargeCoupled / PhaseCoupled / FluxCoupled) whose
    #: physical coupling operator this drive prefers. ``None`` means the drive
    #: implements :meth:`local_channels` itself.
    _coupling_protocol: type | None = None
    #: Name of the device method returning the physical coupling operator when
    #: the target conforms to :attr:`_coupling_protocol`.
    _coupling_accessor: str | None = None
    #: Structural fallback building the coupling operator from the device's
    #: ``a`` / ``a†`` / ``n̂`` when it does not conform to the Protocol.
    _fallback: Callable[[BaseDevice], Operator]
    #: :class:`~quchip.control.signal_spec.DriveModulation` tag for the single
    #: channel exposed by :meth:`local_channels`.
    _modulation: DriveModulation | None = None
    #: ``physics_notes`` line declaring the coupling operator and modulation.
    _coupling_note: str | None = None

    def __init__(
        self,
        target: BaseDevice | None = None,
        *,
        label: str | None = None,
        rwa: bool | None = None,
    ) -> None:
        self.label = label if label is not None else auto_label(type(self)._type_prefix)
        self.rwa = rwa
        self._target: BaseDevice | None = None
        if target is not None:
            self.connect(target)

    def connect(self, device: BaseDevice) -> None:
        """Attach (or re-attach) this drive to *device*.

        If the drive was previously attached to another device, it is
        detached from that device's ``_connected_drives`` list so that
        only the new attachment is visible on the chip.
        """
        old_target = self._target
        if old_target is not None and old_target is not device:
            old_target._connected_drives = [d for d in old_target._connected_drives if d is not self]
        self._target = device
        device.connect(self)

    @property
    def device_label(self) -> str | None:
        """Label of the connected device, or ``None`` if unconnected."""
        return self._target.label if self._target is not None else None

    @property
    def target_label(self) -> str | None:
        """Label of this drive's target, or ``None`` if unconnected.

        Device-target drives (``target_kind == "device"``) alias
        :attr:`device_label`; :class:`ParametricDrive` overrides this to
        resolve its coupling target instead.
        """
        return self.device_label

    def collapse_operators(self, device: BaseDevice) -> list[Operator]:
        """Lindblad collapse operators contributed by this drive.

        Default: none. Override to model drive-induced dissipation
        (e.g. photon loss on a drive line).
        """
        _ = device
        return []

    def local_channels(self, device: BaseDevice) -> list[DriveChannel]:
        """Return the Hamiltonian channels this drive exposes on *device*.

        A simple single-channel drive declares :attr:`_coupling_protocol`,
        :attr:`_coupling_accessor`, :attr:`_fallback`, and :attr:`_modulation`;
        this method then dispatches generically — the device's physical
        coupling operator when it conforms to the Protocol, the structural
        fallback otherwise. Returned operators are in the device's local
        basis (no chip-wide embedding). Drives with a non-standard
        decomposition override this method instead.
        """
        if self._coupling_accessor is None or self._modulation is None:
            raise NotImplementedError(f"{type(self).__name__} must implement local_channels()")
        if self._coupling_protocol is not None and isinstance(device, self._coupling_protocol):
            op = getattr(device, self._coupling_accessor)()
        else:
            op = self._fallback(device)
        return [DriveChannel(operator=op, modulation=self._modulation)]

    def physics_notes(self) -> list[str]:
        """Return human-readable declarations of this drive's approximations.

        Returns the shared target/RWA lines plus this drive's
        :attr:`_coupling_note` when declared. Subclasses with extra notes
        ``super().physics_notes()`` and append their own. Aggregated by
        :meth:`Chip.physics_notes`.
        """
        rwa_note = "inherits chip default" if self.rwa is None else str(self.rwa)
        target = self.device_label if self.device_label is not None else "<unconnected>"
        notes = [
            f"Target device: '{target}'",
            f"RWA policy: {rwa_note}",
        ]
        if self._coupling_note is not None:
            notes.append(self._coupling_note)
        return notes

    def signal_spec(self, drive_op: Any, device: BaseDevice) -> DriveSignalSpec:
        """Build the frame-agnostic signal spec for one scheduled pulse.

        Parameters
        ----------
        drive_op : DriveOp
            Scheduled pulse with envelope, start time, phase offset,
            and (for microwave drives) carrier frequency. Typed as
            :class:`~typing.Any` because :class:`~quchip.engine.ir.DriveOp`
            lives in the engine IR, which drives must not import.
        device : BaseDevice
            Target device. Unused in the default implementation, kept
            as an extension hook so subclasses can emit device-aware
            specs (e.g. flux-tunable drives that query the device's
            tuning curve).

        Returns
        -------
        DriveSignalSpec
            Frame-agnostic description of the pulse — envelope, start
            time, duration, phase offset, and carrier frequency (or
            ``None`` for baseband drives) — built directly from
            *drive_op*, with no frame or IR commitment.

        Notes
        -----
        The default implementation produces a
        :class:`~quchip.control.signal_spec.DriveSignalSpec` describing
        a phase-rotated, time-shifted, finite-duration envelope. Stage 2
        of the engine turns the spec into the raw
        :class:`~quchip.engine.ir.SignalProgram`
        ``Scale(Shift(Window(env, 0, duration), start), exp(i·phi))``
        before applying the modulation dispatch.
        """
        _ = device
        return DriveSignalSpec(
            envelope=drive_op.envelope,
            start_time=drive_op.start_time,
            duration=drive_op.envelope.duration,
            phase_offset=drive_op.phase_offset,
            drive_freq=drive_op.freq,
        )

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
        data["target_label"] = self.target_label
        data["label"] = self.label
        data["rwa"] = self.rwa
        return data

    @classmethod
    def _from_dict_payload(
        cls,
        d: dict[str, Any],
        target: BaseDevice | None = None,
    ) -> "BaseDrive":
        """Reconstruct a concrete drive, rebinding it to *target*.

        Shared by every standard drive whose only persisted state is
        ``label`` / ``rwa``; the registry root's :meth:`from_dict` resolves
        the concrete class from the serialized ``type`` and forwards here.
        Subclasses with extra serialized state override :meth:`from_dict`.
        """
        return cls(
            target=target,
            label=d.get("label"),
            rwa=d.get("rwa"),
        )

    def __repr__(self) -> str:
        attrs = []
        if self._target is not None:
            attrs.append(f"target={self._target.label!r}")
        if self.rwa is not None:
            attrs.append(f"rwa={self.rwa}")
        return f"{type(self).__name__}({', '.join(attrs)})"


def _charge_fallback(device: BaseDevice) -> Operator:
    """Structural charge coupling ``i(a − a†)`` when the device is not ChargeCoupled."""
    a = device.lowering_operator()
    a_dag = device.raising_operator()
    return 1j * (a - a_dag)


def _phase_fallback(device: BaseDevice) -> Operator:
    """Structural phase coupling ``a + a†`` when the device is not PhaseCoupled."""
    a = device.lowering_operator()
    a_dag = device.raising_operator()
    return a + a_dag


def _flux_fallback(device: BaseDevice) -> Operator:
    """Structural flux coupling ``n̂`` when the device is not FluxCoupled."""
    return device.number_operator()


class ChargeDrive(BaseDrive):
    r"""Microwave charge drive on a transmon-like device.

    Contributes the standard charge-coupling Hamiltonian

    .. math::

       H_d(t) = \epsilon(t)\, i(\hat a - \hat a^\dagger)

    with :math:`\epsilon(t)` the (real-projected) mixed signal built by
    the :attr:`~quchip.control.signal_spec.DriveModulation.SINGLE_TONE`
    dispatch in stage 2. This is the canonical transmon microwave drive
    (Koch et al., PRA 76, 042319 (2007); Krantz et al., APR 6, 021318
    (2019), Eq. 90).

    Examples
    --------
    >>> from quchip import DuffingTransmon, ChargeDrive, Gaussian
    >>> q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3)
    >>> drive = ChargeDrive(target=q)
    >>> channels = drive.local_channels(q)
    >>> len(channels) == 1
    True
    """

    _type_prefix: str = "charge"
    _accepts_virtual_z: bool = True
    _defaults_to_device_drive_freq: bool = True

    _coupling_protocol = ChargeCoupled
    _coupling_accessor = "charge_coupling_operator"
    _fallback = staticmethod(_charge_fallback)
    _modulation = DriveModulation.SINGLE_TONE
    _coupling_note = "Drive coupling: charge operator n̂ (or i(a − a†) fallback); single-tone modulation"


class PhaseDrive(BaseDrive):
    r"""Microwave phase drive coupling to :math:`\hat a + \hat a^\dagger`.

    Same carrier machinery as :class:`ChargeDrive` but with an
    in-phase (rather than quadrature) coupling. Useful when modeling
    phase-noise channels or drives whose physical coupling is already
    referenced to the field quadrature. See Krantz et al. 2019, Sec.
    IV.A for the two conventions.
    """

    _type_prefix: str = "phase"
    _accepts_virtual_z: bool = True
    _carrier_required: bool = True

    _coupling_protocol = PhaseCoupled
    _coupling_accessor = "phase_coupling_operator"
    _fallback = staticmethod(_phase_fallback)
    _modulation = DriveModulation.SINGLE_TONE
    _coupling_note = "Drive coupling: phase operator φ̂ (or a + a† fallback); single-tone modulation"


class FluxDrive(BaseDrive):
    r"""Real-valued flux drive coupling to :math:`\hat n`.

    Uses :attr:`~quchip.control.signal_spec.DriveModulation.DIRECT_REAL` —
    no carrier, no RWA — as appropriate for a baseband flux line that
    modulates the device frequency through its number operator
    (Koch et al. 2007, Sec. II; Krantz et al. 2019, Sec. V.A on flux
    tunability).

    The ``rwa`` kwarg is rejected: applying RWA to a baseband drive
    is ill-defined.

    Examples
    --------
    >>> from quchip import DuffingTransmon, FluxDrive
    >>> q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3)
    >>> flux = FluxDrive(target=q)
    >>> flux.rwa is None
    True
    """

    _type_prefix: str = "flux"

    def __init__(
        self,
        target: BaseDevice | None = None,
        *,
        label: str | None = None,
        rwa: bool | None = None,
    ) -> None:
        if rwa is not None:
            raise ValueError("FluxDrive does not support an explicit rwa override.")
        super().__init__(target=target, label=label, rwa=None)

    _coupling_protocol = FluxCoupled
    _coupling_accessor = "flux_coupling_operator"
    _fallback = staticmethod(_flux_fallback)
    _modulation = DriveModulation.DIRECT_REAL
    _coupling_note = (
        "Drive coupling: flux/φ̂ operator (or n̂ fallback); "
        "baseband direct-real modulation (no carrier, no RWA)"
    )


class ParametricDrive(BaseDrive):
    """Control line pumping a modulable coupling's strength δ(t) in GHz.

    Targets a coupling (object or label string; labels late-bind via
    :meth:`Chip.connect`). The scheduled envelope is the *real amplitude*
    ``A(t)``: with an explicit ``freq`` the pump is
    ``δ(t) = A(t)·cos(2π·freq·t - phase)``; with ``freq`` omitted the pump is
    baseband, ``δ(t) = A(t)`` directly. The pump tone is never RWA-split —
    the coupling's RWA policy selects the *operator structure* via its
    parametric hook; the tone keeps both sidebands (declared, the solver
    integrates the fast one).

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
    ValueError
        An explicit ``rwa`` keyword argument is passed — the RWA policy
        is fixed by the coupling's parametric hook, not per-drive.
    """

    target_kind = "edge"
    _type_prefix: str = "parametric"
    _accepts_virtual_z: bool = False
    _modulation = DriveModulation.EDGE_PUMP
    _coupling_note: str | None = (
        "Edge pump: δ(t) in GHz multiplies the coupling's parametric structure; tone kept full (no RWA split)"
    )

    def __init__(self, coupling: "Any | str", *, label: str | None = None, **kwargs: Any) -> None:
        if "rwa" in kwargs:
            raise ValueError(
                "ParametricDrive does not take an RWA flag: the coupling's RWA policy "
                "selects the pump's operator structure."
            )
        if kwargs:
            raise TypeError(f"Unexpected arguments: {sorted(kwargs)}")
        self.label = label if label is not None else auto_label(type(self)._type_prefix)
        self.rwa = None
        self._target: Any = coupling
        if not isinstance(coupling, str):
            _probe_modulable(coupling)

    @property
    def device_label(self) -> None:
        """Always ``None``: an edge pump has no device target.

        Device-keyed paths (drive-noise collection, device-line
        bookkeeping) treat ``None`` as "skip this line"; edge code reads
        :attr:`target_label` instead.
        """
        return None

    @property
    def target_label(self) -> str:
        """Label of the pumped coupling (resolves objects and label strings)."""
        target = self._target
        return target if isinstance(target, str) else target.label

    def __repr__(self) -> str:
        """Return a compact pump-line summary naming the coupling."""
        return f"{type(self).__name__}(label='{self.label}', coupling='{self.target_label}')"

    def connect(self, coupling: Any) -> None:
        """Rebind to *coupling* (no device-side handshake — couplings hold no drive list)."""
        _probe_modulable(coupling)
        self._target = coupling

    def local_channels(self, device: Any) -> list[DriveChannel]:
        """Edge pumps have no device channel; stage 2 compiles them from the coupling's hook."""
        raise NotImplementedError(
            "ParametricDrive exposes no device channel: the pumped operator comes from the "
            "coupling's parametric_interaction() at assembly time."
        )

    @classmethod
    def _from_dict_payload(cls, d: dict[str, Any], target: Any = None) -> "ParametricDrive":
        """Reconstruct a pump line, rebinding it to *target* (a resolved coupling).

        Overrides :meth:`BaseDrive._from_dict_payload`: the base version calls
        ``cls(target=..., rwa=...)``, but :meth:`__init__` takes the coupling
        positionally and rejects an ``rwa`` kwarg outright.
        """
        return cls(target, label=d.get("label"))


def _probe_modulable(coupling: Any) -> None:
    """Raise the teaching TypeError when *coupling* declines the parametric hook."""
    from quchip.declarative.ops import EndpointOps

    probe = getattr(coupling, "parametric_interaction", None)
    expr = None
    if probe is not None:
        expr = probe(
            EndpointOps(label=coupling.device_a_label, levels=2),
            EndpointOps(label=coupling.device_b_label, levels=2),
        )
    if expr is None:
        raise TypeError(
            f"{type(coupling).__name__} is not modulable: its parametric_interaction() hook "
            "returns None. Implement parametric_interaction()/rwa_parametric_interaction() on "
            "the coupling (see CouplingModel), or use a modulable coupling such as TunableCapacitive."
        )
