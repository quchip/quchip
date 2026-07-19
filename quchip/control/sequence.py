"""Declarative pulse programming for a :class:`~quchip.chip.chip.Chip`.

:class:`QuantumSequence` is the user-facing scheduling API. It tracks
per-``(device, drive)`` timing cursors, per-device virtual-Z phase
frames, and cross-device barriers, then materializes the schedule into
:class:`~quchip.engine.ir.DriveOp` objects for the engine.

Conventions
-----------
- Times are ns, frequencies GHz (ordinary, not angular).
- Virtual-Z shifts accumulate into later *microwave* pulses on the same
  device (charge and phase drives); baseband flux pulses are unaffected,
  matching the lab-frame semantics of software-Z (McKay et al., PRA
  96, 022330 (2017)).
- All sweep axes stay JAX-traceable: pulse parameters, delays, and
  envelope fields flow into ``SolveProblem`` without Python-side
  concretization.

Examples
--------
>>> from quchip import (
...     DuffingTransmon, ChargeDrive, Chip, QuantumSequence, Gaussian
... )
>>> q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3)
>>> drive = ChargeDrive(target=q)
>>> chip = Chip([q])
>>> seq = QuantumSequence(chip)
>>> _ = seq.schedule(q, envelope=Gaussian(duration=20.0, amplitude=0.05))
>>> seq.total_duration
20.0
"""

from __future__ import annotations

import copy
import functools
from dataclasses import dataclass
from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from quchip.chip.chip import Chip
from quchip.chip.coupling_base import BaseCoupling
from quchip.control.batch import (
    BatchAxis,
    DelayHandle,
    ProblemBatch,
    PulseHandle,
    ZippedBatchAxis,
    _axis_metadata,
    _expand_axis_overrides,
)
from quchip.control.drive import BaseDrive, ChargeDrive, FluxDrive, PhaseDrive
from quchip.control.envelopes import BaseEnvelope
from quchip.devices.base import BaseDevice
from quchip.engine.ir import DriveOp, HamiltonianTemplate
from quchip.engine.stage4_problem import (
    build_solve_batch_from_descriptions,
    build_solve_problem,
    prepare_solve_problem_context,
    validate_drive_ops_window,
)
from quchip.engine.stage2_assembly import (
    compile_hamiltonian_template,
    instantiate_hamiltonian_description,
)
from quchip.utils.jax_utils import maybe_concrete_scalar
from quchip.utils.jax_utils import array_namespace, is_jax_namespace
from quchip.utils.jax_utils import is_jax_array as _is_traced
from quchip.utils.jax_utils import select_array_module as _select_array_module
from quchip.utils.labeling import resolve_label

if TYPE_CHECKING:
    from quchip.chip.transformations.active_patch import ActivePatchResult
    from quchip.engine.ir import SolveProblem
    from quchip.results.results import SimulationBatchResult, SimulationResult


@dataclass(frozen=True)
class _PulseEntry:
    target_label: str
    drive_label: str
    envelope: BaseEnvelope
    freq: float | None
    requested_start_time: float | None
    phase: float


@dataclass(frozen=True)
class _DelayEntry:
    device_label: str
    duration: Any


@dataclass(frozen=True)
class _BarrierEntry:
    device_labels: tuple[str, ...]


@dataclass(frozen=True)
class _FrameShiftEntry:
    device_label: str
    angle: Any


def _require_positive_duration(duration: Any) -> None:
    """Reject a non-positive *concrete* delay duration; traced values pass through.

    Shared by :meth:`QuantumSequence.delay` (validates the scheduled value)
    and the replay path (validates per-variant batch overrides) — the two
    sites check different values, so both must gate.
    """
    duration_val = maybe_concrete_scalar(duration)
    if duration_val is not None and duration_val <= 0:
        raise ValueError(f"Delay duration must be > 0, got {duration}")


def _cursor_max(values: Collection[Any]) -> Any:
    """Return the maximum of cursor *values*, tracing-safe under ``jax.jit``.

    Python's ``max()`` branches on the result of ``>``/``<``, which raises
    under a JAX trace once any operand is a tracer. Folds with
    ``jnp.maximum`` instead whenever any value is a JAX array; *values*
    must be non-empty.
    """
    xp = _select_array_module(any(_is_traced(v) for v in values))
    return functools.reduce(xp.maximum, values)


class QuantumSequence:
    """Declarative pulse sequence builder for a :class:`~quchip.chip.chip.Chip`.

    Tracks per-``(device, drive)`` timing cursors and per-device
    virtual-Z phase frames. Append pulses with :meth:`schedule` (or the
    conveniences :meth:`charge`, :meth:`phase`, :meth:`flux`);
    synchronize channels with :meth:`barrier` and :meth:`delay`. The
    schedule is materialized lazily into
    :class:`~quchip.engine.ir.DriveOp` objects by :meth:`build_problem`
    and :meth:`build_batch`.

    Wherever a device/drive is expected, either the object itself or its
    string label works, and examples should prefer object references.

    Simulation runs through one consistent verb — ``simulate`` — across
    three tiers, from most ergonomic to most explicit:

    1. :meth:`simulate` / :meth:`simulate_batch` — schedule *and* solve in
       one call (the example-facing path).
    2. :meth:`~quchip.chip.chip.Chip.solve` /
       :meth:`~quchip.chip.chip.Chip.solve_many` — solve a
       :class:`~quchip.engine.ir.SolveProblem` / batch you already hold.
    3. The module-level :func:`~quchip.engine.simulate` /
       :func:`~quchip.engine.solve_problem` / :func:`~quchip.engine.solve_many`
       — the low-level "I already have ``drive_ops`` / a ``SolveProblem``"
       tier.

    Parameters
    ----------
    chip : Chip
        Chip this sequence schedules against. Supplies the device and
        coupling maps used to resolve scheduling targets, the wired
        :class:`~quchip.control.equipment.ControlEquipment` lines, and
        the frame/backend settings used by :meth:`build_problem` and
        :meth:`simulate`.

    Examples
    --------
    >>> from quchip import (
    ...     DuffingTransmon, ChargeDrive, Chip, QuantumSequence, Gaussian
    ... )
    >>> q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3)
    >>> _ = ChargeDrive(target=q)
    >>> chip = Chip([q])
    >>> seq = QuantumSequence(chip)
    >>> _ = seq.charge(q, envelope=Gaussian(duration=20.0, amplitude=0.05))
    >>> seq.vz(q, angle=0.5)
    >>> _ = seq.charge(q, envelope=Gaussian(duration=10.0, amplitude=0.03))
    """

    def __init__(self, chip: Chip) -> None:
        self._chip = chip
        # ``_entries`` is the single source of truth for all timing. Cursors,
        # floors, and virtual-Z phase frames are reconstructed on demand by
        # ``_replay`` — there is no live cursor state to drift from the replay.
        self._entries: list[_PulseEntry | _DelayEntry | _BarrierEntry | _FrameShiftEntry] = []

    @staticmethod
    def _clone_envelope_with_overrides(envelope: BaseEnvelope, overrides: Mapping[str, Any]) -> BaseEnvelope:
        cloned = copy.copy(envelope)
        for field, value in overrides.items():
            if not hasattr(cloned, field):
                raise ValueError(f"Envelope '{type(cloned).__name__}' has no field '{field}'")
            setattr(cloned, field, value)
        return cloned

    def _find_drive_by_type(self, device_label: str, drive_type: type) -> BaseDrive:
        device = self._chip.device_map[device_label]
        matches = [d for d in device.connected_drives if isinstance(d, drive_type)]
        if len(matches) == 0:
            raise ValueError(f"No {drive_type.__name__} on device '{device_label}'.")
        if len(matches) > 1:
            raise ValueError(
                f"Multiple {drive_type.__name__} on '{device_label}': "
                f"{[d.label for d in matches]}. Use schedule(drive, ...) instead."
            )
        return matches[0]

    def _find_default_drive(self, device_label: str) -> BaseDrive:
        drives = self._chip.device_map[device_label].connected_drives
        if not drives:
            raise ValueError(
                f"Device '{device_label}' has no connected drives. "
                "Connect a drive to the device before scheduling."
            )
        return drives[0]

    def _find_pump_line(self, coupling_label: str) -> BaseDrive:
        """Return the unique ParametricDrive line pumping *coupling_label*."""
        equipment = self._chip.control_equipment
        lines = [] if equipment is None else [
            line for line in equipment.lines if line.target_kind == "edge" and line.target_label == coupling_label
        ]
        if len(lines) == 1:
            return lines[0]
        if not lines:
            raise ValueError(
                f"No ParametricDrive line pumps coupling '{coupling_label}'. Wire one into the chip's "
                "ControlEquipment (e.g. ParametricDrive(coupling)) before scheduling."
            )
        raise ValueError(
            f"Multiple pump lines target coupling '{coupling_label}': {[line.label for line in lines]}. "
            "Schedule on the drive object or its label instead."
        )

    def _find_drive_line(self, label: str) -> BaseDrive:
        """Return the control-equipment line named *label* (schedule()'s final string fallback).

        Resolves a drive's own label directly to that line, independent of
        its device or coupling target — the only name that survives when a
        line's original target has been eliminated (a retarget rule
        preserves the line's label; see
        :func:`~quchip.chip.transformations.eliminate`).
        """
        equipment = self._chip.control_equipment
        lines = [] if equipment is None else [line for line in equipment.lines if line.label == label]
        if lines:
            return lines[0]
        line_labels = [] if equipment is None else [line.label for line in equipment.lines]
        raise ValueError(
            f"Label '{label}' not found on chip as a device, coupling, or control-line label. "
            f"Available devices: {list(self._chip.device_map.keys())}. "
            f"Available couplings: {list(self._chip.coupling_map.keys())}. "
            f"Available control lines: {line_labels}."
        )

    def _schedule_on_drive(
        self,
        drive: BaseDrive,
        *,
        envelope: BaseEnvelope,
        freq: float | None,
        start_time: float | None = None,
        phase: float = 0.0,
    ) -> PulseHandle:
        if drive._target is None:
            raise ValueError(
                "Cannot schedule a drive that is not connected to a device. "
                "Connect it first with drive.connect(device)."
            )
        if drive.target_kind == "edge":
            target_label = drive.target_label
            if target_label not in self._chip.coupling_map:
                raise ValueError(
                    f"Pump line '{drive.label}' targets coupling '{target_label}', which is not on this chip. "
                    f"Available couplings: {list(self._chip.coupling_map.keys())}"
                )
        else:
            target_label = drive._target.label
            if target_label not in self._chip.device_map:
                raise ValueError(
                    f"Drive is connected to device '{target_label}', which is not on this chip. "
                    f"Available devices: {list(self._chip.device_map.keys())}"
                )
        self._entries.append(
            _PulseEntry(
                target_label=target_label,
                drive_label=drive.label,
                envelope=envelope,
                freq=freq,
                requested_start_time=start_time,
                phase=phase,
            )
        )
        return PulseHandle(self, len(self._entries) - 1)

    def _resolve_carrier_freq(self, drive: BaseDrive, freq: float | None) -> float | None:
        """Resolve a pulse carrier frequency from the drive's carrier policy.

        Single source of the carrier rule shared by :meth:`schedule` and the
        :meth:`charge` / :meth:`phase` / :meth:`flux` / :meth:`flux_to`
        conveniences. The drive declares its policy via ``_carrier_required``
        and ``_defaults_to_device_drive_freq``:

        * carrier-required drives (:class:`PhaseDrive`,
          :class:`~quchip.control.drives_two_photon.TwoPhotonDrive`) raise
          when *freq* is missing;
        * device-default drives (:class:`ChargeDrive`) fall back to the target
          device's ``drive_freq`` when *freq* is missing;
        * baseband drives (:class:`FluxDrive`) ignore the carrier — *freq* is
          ``None`` and stays ``None``.

        The device default (``drive._target.drive_freq``) is only read on the
        defaulting branch, so a disconnected drive with an explicit *freq*
        still reaches the clean "not connected" error in
        :meth:`_schedule_on_drive`. The returned frequency stays
        JAX-traceable (no concretization).
        """
        if freq is None and drive._carrier_required:
            raise ValueError(f"{type(drive).__name__} drives require an explicit freq argument")
        if freq is None and drive._defaults_to_device_drive_freq:
            if drive._target is None:
                raise ValueError(
                    "Cannot schedule a drive that is not connected to a device. "
                    "Connect it first with drive.connect(device)."
                )
            return self._chip.device_map[drive._target.label].drive_freq
        return freq

    def schedule(
        self,
        target: str | BaseDrive | BaseDevice | BaseCoupling,
        *,
        envelope: BaseEnvelope,
        freq: float | None = None,
        start_time: float | None = None,
        phase: float = 0.0,
    ) -> PulseHandle:
        """Schedule a pulse on *target*.

        Parameters
        ----------
        target : str | BaseDrive | BaseDevice | BaseCoupling
            Accepted forms, resolved in this order:

            * :class:`BaseDrive` — scheduled directly on that drive
              (a :class:`~quchip.control.drive.ParametricDrive` pumps
              its bound coupling).
            * :class:`BaseDevice` — uses the device's first connected
              drive; pass the drive object explicitly when a device
              has multiple drives.
            * :class:`~quchip.chip.coupling_base.BaseCoupling` — uses
              the unique :class:`~quchip.control.drive.ParametricDrive`
              line pumping that coupling; pass the drive object
              explicitly when a coupling has multiple pump lines.
            * ``str`` — a label, resolved in order: a device label; then,
              only when absent from the device map, a coupling label
              (the two label spaces are disjoint); then, only when
              absent from both, a control-equipment line label,
              scheduling directly on that line. This third fallback is
              what lets a caller schedule by a drive's own label after
              its device or coupling target has been eliminated — see
              :func:`~quchip.chip.transformations.eliminate`'s retarget
              registry, which preserves a converted line's label. A
              control-line label that collides with a device/coupling
              label is shadowed by the device/coupling resolution.

        envelope : BaseEnvelope
            Pulse envelope.
        freq : float, optional
            Carrier frequency in GHz for microwave drives. Defaults to
            the target device's ``drive_freq`` for
            :class:`ChargeDrive`; always required (no device-frequency
            fallback) for :class:`PhaseDrive` and
            :class:`~quchip.control.drives_two_photon.TwoPhotonDrive`;
            ignored for :class:`FluxDrive` and baseband
            :class:`ParametricDrive` pumps.
        start_time : float, optional
            Pulse start time in ns. Defaults to the current cursor;
            earlier times are rejected.
        phase : float
            Per-pulse phase offset, composed with any accumulated
            virtual-Z.
        """
        if isinstance(target, BaseDrive):
            drive = target
        else:
            label = resolve_label(target)
            if label in self._chip.device_map:
                drive = self._find_default_drive(label)
            elif label in self._chip.coupling_map:
                drive = self._find_pump_line(label)
            else:
                drive = self._find_drive_line(label)

        freq = self._resolve_carrier_freq(drive, freq)
        return self._schedule_on_drive(drive, envelope=envelope, freq=freq, start_time=start_time, phase=phase)

    def charge(
        self,
        target: str | BaseDevice,
        *,
        envelope: BaseEnvelope,
        freq: float | None = None,
        phase: float = 0.0,
    ) -> PulseHandle:
        """Schedule a :class:`ChargeDrive` pulse; *freq* defaults to ``device.drive_freq``."""
        label = resolve_label(target)
        self._validate_target(label)
        drive = self._find_drive_by_type(label, ChargeDrive)
        freq = self._resolve_carrier_freq(drive, freq)
        return self._schedule_on_drive(drive, envelope=envelope, freq=freq, phase=phase)

    def phase(
        self,
        target: str | BaseDevice,
        *,
        envelope: BaseEnvelope,
        freq: float,
        phase: float = 0.0,
    ) -> PulseHandle:
        """Schedule a :class:`PhaseDrive` pulse; *freq* is required."""
        label = resolve_label(target)
        self._validate_target(label)
        drive = self._find_drive_by_type(label, PhaseDrive)
        # PhaseDrive requires a carrier, so the helper either returns *freq*
        # unchanged or raises; route through it for one shared carrier rule.
        resolved_freq = self._resolve_carrier_freq(drive, freq)
        return self._schedule_on_drive(drive, envelope=envelope, freq=resolved_freq, phase=phase)

    def flux(
        self,
        target: str | BaseDevice,
        *,
        envelope: BaseEnvelope,
    ) -> PulseHandle:
        """Schedule a :class:`FluxDrive` pulse (no carrier frequency)."""
        label = resolve_label(target)
        self._validate_target(label)
        drive = self._find_drive_by_type(label, FluxDrive)
        freq = self._resolve_carrier_freq(drive, None)
        return self._schedule_on_drive(drive, envelope=envelope, freq=freq)

    def pump(
        self,
        coupling: str | BaseCoupling,
        *,
        envelope: BaseEnvelope,
        freq: float | None = None,
        start_time: float | None = None,
        phase: float = 0.0,
    ) -> PulseHandle:
        """Schedule an edge pump: baseband δ(t)=A(t) when ``freq`` is None, else A(t)·cos(2πft-φ)."""
        label = resolve_label(coupling)
        drive = self._find_pump_line(label)
        return self._schedule_on_drive(drive, envelope=envelope, freq=freq, start_time=start_time, phase=phase)

    def flux_to(
        self,
        target: str | BaseDevice,
        *,
        target_freq: Any,
        envelope: BaseEnvelope,
    ) -> PulseHandle:
        """Schedule a flux-drive frequency-shift pulse to ``target_freq``.

        This is not an inverse-SQUID flux calibration. It computes
        ``δω = target_freq − chip.freq(device)`` and schedules a
        :class:`FluxDrive` pulse whose envelope amplitude is that frequency
        shift in GHz. The resulting Hamiltonian contribution is the linear
        detuning term ``δω(t) n̂``. ``envelope`` should be passed with
        ``amplitude=None`` (or any placeholder) — this method replaces the
        amplitude with the computed δω.

        ``target_freq`` may be a JAX tracer (e.g. ``chip.freq(other_device)``).

        Parameters
        ----------
        target : str | BaseDevice
            Device to flux-pulse.
        target_freq : float
            Target ``0 → 1`` frequency in GHz.
        envelope : BaseEnvelope
            Envelope template with all timing parameters set. Its
            ``amplitude`` attribute is replaced by the computed δω; any
            value passed as ``amplitude`` is ignored.

        Returns
        -------
        PulseHandle
            Handle to the scheduled entry, usable with ``.vary()``.
        """
        label = resolve_label(target)
        self._validate_target(label)
        device = self._chip.device_map[label]
        current_freq = self._chip.freq(device)
        delta_omega = target_freq - current_freq
        pulse = copy.copy(envelope)
        pulse.amplitude = delta_omega
        drive = self._find_drive_by_type(label, FluxDrive)
        freq = self._resolve_carrier_freq(drive, None)
        return self._schedule_on_drive(drive, envelope=pulse, freq=freq)

    def vz(self, target: str | BaseDevice, angle: float) -> None:
        """Apply a virtual-Z frame shift of *angle* rad on *target*.

        The shift is free (no pulse is emitted) and accumulates into
        every subsequent microwave pulse on the device via its
        ``phase_offset``. This is the standard software-Z trick for
        transmons (McKay et al., PRA 96, 022330 (2017)). Baseband flux
        pulses are unaffected.
        """
        label = resolve_label(target)
        self._validate_target(label)
        self._entries.append(_FrameShiftEntry(device_label=label, angle=angle))

    def delay(self, scope: str | BaseDevice, duration: float) -> DelayHandle:
        """Insert an idle delay on all drive channels of *scope*."""
        target = resolve_label(scope)
        self._validate_target(target)
        _require_positive_duration(duration)
        self._entries.append(_DelayEntry(device_label=target, duration=duration))
        return DelayHandle(self, len(self._entries) - 1)

    def barrier(self, *labels: str | BaseDevice) -> None:
        """Synchronize channel cursors.

        With no arguments, every ``(device, drive)`` cursor on the
        chip is advanced to the current maximum. With explicit
        device targets, only those devices' cursors are aligned. Use
        this to guarantee that pulses scheduled after the barrier
        start no earlier than any pulse scheduled before it.
        """
        if not labels:
            self._entries.append(_BarrierEntry(device_labels=()))
            return

        resolved = tuple(resolve_label(lbl) for lbl in labels)
        for lbl in resolved:
            self._validate_target(lbl)
        self._entries.append(_BarrierEntry(device_labels=resolved))

    @property
    def total_duration(self) -> Any:
        """Total sequence duration in ns (maximum across all cursors)."""
        cursors, _ = self._replay_cursors()
        return _cursor_max(cursors.values()) if cursors else 0.0

    @property
    def scheduled_ops(self) -> tuple[DriveOp, ...]:
        """Materialize and return the scheduled :class:`DriveOp` tuple."""
        return tuple(self._materialize_drive_ops())

    @property
    def channel_cursors(self) -> dict[tuple[str, str], Any]:
        """Current time cursors keyed by ``(target_label, drive_label)``."""
        return self._replay_cursors()[0]

    def _resolve_tlist(self, tlist: Any | None) -> Any:
        """Return *tlist* unchanged, or synthesize the default save grid from :attr:`total_duration`.

        Synthesizes at 10 points/ns with a 100-point floor. This grid sets
        where expectation values and states are saved and returned; it is
        not a solver step-size or carrier-resolution guarantee. dynamiqs
        evaluates its Hamiltonian coefficients on an adaptive grid of its
        own, while QuTiP keeps carriers analytic and interpolates only
        carrier-free envelopes on this grid. Simulating lab-frame or other
        raw-carrier oscillations requires passing an explicitly dense
        ``tlist``.
        """
        if tlist is not None:
            if is_jax_namespace(array_namespace(tlist)):
                return tlist
            return np.asarray(tlist)

        dur = self.total_duration
        dur_value = maybe_concrete_scalar(dur)
        if dur_value is None:
            raise ValueError(
                "QuantumSequence cannot infer a default tlist from a traced or symbolic total_duration. "
                "Pass an explicit tlist when sequence timing depends on traced parameters."
            )
        if dur_value <= 0:
            dur = 1.0
            dur_value = 1.0
        n_points = max(int(dur_value * 10), 100)
        return np.linspace(0, dur, n_points)

    def _resolve_initial_state_spec(self, state_spec: Any) -> Any:
        if isinstance(state_spec, Mapping):
            return self._chip.state(state_spec)
        return state_spec

    def _drive_lookup(self) -> dict[str, BaseDrive]:
        """Return a ``{drive_label: drive}`` map across the chip.

        Consolidates the two sources of drives — the control
        equipment's line list and every device's ``connected_drives``
        — into a single dictionary so callers don't double-walk. The
        equipment takes priority when a label is present on both.
        """
        lookup: dict[str, BaseDrive] = {}
        for dev in self._chip.devices:
            for drv in dev.connected_drives:
                lookup[drv.label] = drv
        ce = self._chip.control_equipment
        if ce is not None:
            for drv in ce.lines:
                lookup[drv.label] = drv
        return lookup

    def _is_microwave_drive(self, drive_label: str, lookup: dict[str, BaseDrive]) -> bool:
        drv = lookup.get(drive_label)
        return drv is not None and drv._accepts_virtual_z

    def _replay(
        self,
        overrides: Mapping[tuple[int, str], Any] | None = None,
        *,
        collect_ops: bool,
    ) -> tuple[dict[tuple[str, str], Any], dict[str, Any], list[DriveOp]]:
        """Replay ``_entries`` into final cursors, floors, and (optionally) DriveOps.

        This is the single implementation of the delay-shift, barrier-alignment,
        floor, default-start, and virtual-Z phase semantics. Both the cursor
        accessors (:meth:`_replay_cursors`) and materialization
        (:meth:`_materialize_drive_ops`) run through it, so there is no way for a
        cursor read and a materialized schedule to disagree.

        Replaying the whole entry list on each call is O(n) per invocation
        (O(n^2) across a full build); this is an intentional trade for a single
        source of truth, and sequences carry only modest pulse counts.

        ``overrides`` maps ``(entry_index, field) -> value``; only ``duration``
        (on a delay entry or pulse envelope) and ``start_time``/``freq``/``phase``
        (on a pulse) affect the replay. ``collect_ops`` gates DriveOp building
        and the per-device drive lookup needed for virtual-Z routing.
        """
        overrides = {} if overrides is None else dict(overrides)
        cursors: dict[tuple[str, str], Any] = {}
        for dev in self._chip.devices:
            for drv in dev.connected_drives:
                cursors[(dev.label, drv.label)] = 0.0
        equipment = self._chip.control_equipment
        if equipment is not None:
            for line in equipment.lines:
                if line.target_kind == "edge":
                    assert line.target_label is not None
                    cursors[(line.target_label, line.label)] = 0.0
        floors: dict[str, Any] = {}
        phases: dict[str, Any] = {dev.label: 0.0 for dev in self._chip.devices}
        drive_lookup = self._drive_lookup() if collect_ops else {}
        drive_ops: list[DriveOp] = []

        for entry_index, entry in enumerate(self._entries):
            if isinstance(entry, _FrameShiftEntry):
                phases[entry.device_label] = phases[entry.device_label] + entry.angle
                continue

            if isinstance(entry, _DelayEntry):
                duration = overrides.get((entry_index, "duration"), entry.duration)
                _require_positive_duration(duration)
                for key in list(cursors.keys()):
                    if key[0] == entry.device_label:
                        cursors[key] += duration
                floors[entry.device_label] = floors.get(entry.device_label, 0.0) + duration
                continue

            if isinstance(entry, _BarrierEntry):
                if not entry.device_labels:
                    max_time = _cursor_max(cursors.values()) if cursors else 0.0
                    for key in cursors:
                        cursors[key] = max_time
                    for dev in self._chip.devices:
                        floors[dev.label] = max_time
                    continue

                matching_keys = [key for key in cursors if key[0] in entry.device_labels]
                if matching_keys:
                    max_time = _cursor_max([cursors[key] for key in matching_keys])
                    for key in matching_keys:
                        cursors[key] = max_time
                    for label in entry.device_labels:
                        floors[label] = max_time
                continue

            if not isinstance(entry, _PulseEntry):
                continue

            envelope_updates = {
                field: value
                for (idx, field), value in overrides.items()
                if idx == entry_index and field not in {"freq", "phase", "start_time"}
            }
            envelope = (
                self._clone_envelope_with_overrides(entry.envelope, envelope_updates)
                if envelope_updates
                else entry.envelope
            )
            requested_start_time = overrides.get((entry_index, "start_time"), entry.requested_start_time)
            key = (entry.target_label, entry.drive_label)
            if key not in cursors:
                cursors[key] = floors.get(entry.target_label, 0.0)
            current_cursor = cursors[key]
            start_val = maybe_concrete_scalar(requested_start_time)
            cursor_val = maybe_concrete_scalar(current_cursor)
            if (
                requested_start_time is not None
                and start_val is not None
                and cursor_val is not None
                and start_val < cursor_val
            ):
                raise ValueError(
                    "Explicit start_time cannot be earlier than current cursor "
                    f"for '{entry.target_label}:{entry.drive_label}'. start_time={requested_start_time}, "
                    f"current_cursor={current_cursor}."
                )
            start_time = current_cursor if requested_start_time is None else requested_start_time

            if collect_ops:
                freq = overrides.get((entry_index, "freq"), entry.freq)
                phase = overrides.get((entry_index, "phase"), entry.phase)
                phase_offset = phase
                if self._is_microwave_drive(entry.drive_label, drive_lookup):
                    phase_offset = phase_offset + phases.get(entry.target_label, 0.0)
                drive_ops.append(
                    DriveOp(
                        target_label=entry.target_label,
                        envelope=envelope,
                        freq=freq,
                        start_time=start_time,
                        phase_offset=phase_offset,
                        drive_label=entry.drive_label,
                    )
                )
            cursors[key] = start_time + envelope.duration
        return cursors, floors, drive_ops

    def _replay_cursors(
        self, overrides: Mapping[tuple[int, str], Any] | None = None
    ) -> tuple[dict[tuple[str, str], Any], dict[str, Any]]:
        """Replay ``_entries`` to their final ``(cursors, floors)`` without building DriveOps."""
        cursors, floors, _ = self._replay(overrides, collect_ops=False)
        return cursors, floors

    def _materialize_drive_ops(self, overrides: Mapping[tuple[int, str], Any] | None = None) -> list[DriveOp]:
        """Replay ``_entries`` into the scheduled :class:`DriveOp` list."""
        return self._replay(overrides, collect_ops=True)[2]

    def build_problem(
        self,
        tlist: Any | None = None,
        solver: str | None = None,
        options: dict | None = None,
        e_ops: dict | None = None,
        initial_state: Any | None = None,
    ) -> "SolveProblem":
        """Build a single :class:`~quchip.engine.ir.SolveProblem` from this sequence.

        Parameters
        ----------
        tlist : array-like, optional
            Save/output time grid in ns. Defaults to the grid built by
            :meth:`_resolve_tlist` from :attr:`total_duration` (10
            points/ns, 100-point floor).
        solver : str, optional
            Backend solver name. Defaults to the backend's own default
            solver when omitted.
        options : dict, optional
            Solver options, merged on top of the defaults
            ``{"store_states": True, "store_final_state": True}``.
        e_ops : dict, optional
            Expectation operators keyed by device label (or a 2-tuple of
            device labels for a two-body operator), mapping to a local
            operator (or a pair of local operators). Decomposed into
            per-band terms before reaching the solver.
        initial_state : Any, optional
            A ket, density matrix, or mapping for
            :meth:`~quchip.chip.chip.Chip.state`. Defaults to the chip's
            default initial state when omitted.

        Returns
        -------
        SolveProblem
            Frozen problem — chip, compiled Hamiltonian description,
            initial state, ``tlist``, collapse operators, and
            expectation operators — ready for
            :meth:`~quchip.chip.chip.Chip.solve` or
            :func:`~quchip.engine.solve_problem`.
        """
        actual_tlist = self._resolve_tlist(tlist)
        drive_ops = self._materialize_drive_ops()
        resolved_state = self._resolve_initial_state_spec(initial_state) if initial_state is not None else None
        return build_solve_problem(
            self._chip,
            drive_ops,
            actual_tlist,
            solver=solver,
            options=options,
            e_ops=e_ops,
            initial_state=resolved_state,
        )

    def vary(self, field: str, values: Any, *, name: str | None = None) -> BatchAxis:
        """Create a :class:`BatchAxis` over a sequence-level field.

        Parameters
        ----------
        field : str
            Sequence-level field to sweep. Only ``"initial_state"`` is
            supported; per-pulse fields (``freq``, ``phase``,
            ``amplitude``, ...) are swept via :meth:`PulseHandle.vary`
            on the handle returned by :meth:`schedule` instead.
        values : array-like
            Sequence of values for ``field``, one per batch point.
        name : str, optional
            Axis name recorded in :attr:`ProblemBatch.axes` and
            :meth:`ProblemBatch.params_at`. Defaults to *field*.

        Returns
        -------
        BatchAxis
            Sequence-level axis, consumable by :meth:`build_batch` or
            :meth:`zip`.
        """
        if field != "initial_state":
            raise ValueError("QuantumSequence only supports varying the sequence-level field 'initial_state'")
        return BatchAxis(owner=self, target_kind="sequence", field=field, values=values, name=name or field)

    def zip(self, *axes: BatchAxis) -> ZippedBatchAxis:
        """Zip axes into one pairwise dimension.

        Parameters
        ----------
        *axes : BatchAxis
            Two or more axes to zip, each created by this sequence
            (:meth:`vary`, :meth:`PulseHandle.vary`, or
            :meth:`DelayHandle.vary`). Every axis must have the same
            length.

        Returns
        -------
        ZippedBatchAxis
            Combined axis that :meth:`build_batch` treats as a single
            dimension: point ``i`` supplies point ``i`` from every
            zipped axis simultaneously, rather than the outer product.
        """
        if not axes:
            raise ValueError("zip() requires at least one axis")
        for axis in axes:
            if axis.owner is not self:
                raise ValueError("All axes passed to QuantumSequence.zip() must belong to this sequence")
        sizes = {axis.size for axis in axes}
        if len(sizes) != 1:
            raise ValueError(f"Zipped axes must have equal lengths, got {sorted(sizes)}")
        return ZippedBatchAxis(axes=tuple(axes))

    def _validate_axes(self, axes: Sequence[BatchAxis | ZippedBatchAxis]) -> None:
        seen_names: set[str] = set()
        for axis in axes:
            members = axis.axes if isinstance(axis, ZippedBatchAxis) else (axis,)
            for member in members:
                if member.owner is not self:
                    raise ValueError("All batch axes must belong to this QuantumSequence")
                if member.name in seen_names:
                    raise ValueError(
                        f"Duplicate batch axis name {member.name!r}; give each axis a "
                        "unique name via vary(..., name=...)."
                    )
                seen_names.add(member.name)
                if member.entry_index is not None and (
                    member.entry_index >= len(self._entries)
                    or self._entries[member.entry_index] is not member.entry
                ):
                    raise ValueError(
                        f"Batch axis {member.name!r} no longer matches its scheduled entry — "
                        "the sequence was modified after vary() was called. Re-schedule and "
                        "create the axis from a fresh handle."
                    )

    def _build_batch_point_description(
        self,
        template: HamiltonianTemplate,
        reference_description: Any,
        overrides: dict[tuple[int | None, str], Any],
        *,
        tlist: Any,
        shared_initial_state: Any | None,
    ) -> tuple[Any, Any]:
        """Return ``(HamiltonianDescription, initial_state)`` for one batch point."""
        axis_initial_state = overrides.get((None, "initial_state"))
        entry_overrides: dict[tuple[int, str], Any] = {
            (idx, field): value
            for (idx, field), value in overrides.items()
            if idx is not None
        }

        if axis_initial_state is not None and shared_initial_state is not None:
            raise ValueError("initial_state may be provided either as a shared scalar or as a batch axis, not both")

        if entry_overrides:
            # Entry overrides only mutate scalar fields (envelope params, freq,
            # phase, start_time), never the operator skeleton, so re-instantiating
            # the compiled template cannot fail structural validation. Call it
            # directly — a ValueError here is a real engine error, not a fallback
            # trigger.
            drive_ops = self._materialize_drive_ops(entry_overrides)
            # A per-point override may move a pulse window off the solve
            # interval (e.g. sweeping start_time or duration); the reference
            # schedule's window was already checked when the batch's shared
            # context was built, but this variant needs its own check.
            validate_drive_ops_window(drive_ops, tlist)
            description = instantiate_hamiltonian_description(template, drive_ops, self._chip)
        else:
            description = reference_description

        initial_state = axis_initial_state if axis_initial_state is not None else shared_initial_state
        resolved_initial_state = (
            self._resolve_initial_state_spec(initial_state) if initial_state is not None else None
        )
        return description, resolved_initial_state

    def build_batch(
        self,
        *axes: BatchAxis | ZippedBatchAxis,
        tlist: Any | None = None,
        solver: str | None = None,
        options: dict | None = None,
        e_ops: dict | None = None,
        initial_state: Any | None = None,
    ) -> ProblemBatch:
        """Build a batched solve request from sweep axes without solving.

        The returned :class:`ProblemBatch` wraps a
        :class:`~quchip.engine.ir.SolveBatch`: shared static/dynamic operators
        are collected exactly once and only per-element scalar modulations
        (and per-element initial states, when varied) change across the batch.
        """
        self._validate_axes(axes)
        actual_tlist = self._resolve_tlist(tlist)
        reference_drive_ops = self._materialize_drive_ops()
        context = prepare_solve_problem_context(
            self._chip,
            actual_tlist,
            solver=solver,
            options=options,
            e_ops=e_ops,
            drive_ops=reference_drive_ops,
        )
        template = compile_hamiltonian_template(self._chip, reference_drive_ops, resolved_frame=context.resolved_frame)
        reference_description = instantiate_hamiltonian_description(template, reference_drive_ops, self._chip)

        shape, expanded = _expand_axis_overrides(axes)
        params_store = np.empty(shape if shape else (), dtype=object)

        descriptions: list[Any] = []
        initial_states: list[Any] = []
        for coord, overrides in expanded:
            description, resolved_initial_state = self._build_batch_point_description(
                template,
                reference_description,
                overrides,
                tlist=context.tlist,
                shared_initial_state=initial_state,
            )
            descriptions.append(description)
            initial_states.append(resolved_initial_state)

            point_params: dict[str, Any] = {}
            for dim, axis in enumerate(axes):
                members = axis.axes if isinstance(axis, ZippedBatchAxis) else (axis,)
                for member in members:
                    # Name uniqueness is validated upfront in _validate_axes.
                    point_params[member.name] = member.values[coord[dim]]
            params_store[coord] = point_params

        batch = build_solve_batch_from_descriptions(context, descriptions, initial_states=initial_states)
        return ProblemBatch(
            batch=batch,
            params=params_store,
            shape=shape,
            axes=tuple(_axis_metadata(axis) for axis in axes),
        )

    def simulate(
        self,
        tlist: Any | None = None,
        solver: str | None = None,
        options: dict | None = None,
        e_ops: dict | None = None,
        initial_state: Any | None = None,
        *,
        backend: Any | None = None,
        check_truncation: bool = True,
        truncation_threshold: float = 1e-3,
        partition: bool = True,
    ) -> "SimulationResult":
        """Build and solve a single simulation, routed through :func:`~quchip.engine.simulate`.

        ``backend`` — name (``"qutip"``/``"dynamiqs"``) or instance — scopes
        this one call, outranking the chip's and the process default (so a
        gradient solve can run on dynamiqs while everything around it stays
        on QuTiP). Foreign-native initial states are coerced at the solve
        boundary. Caveat: Python evaluates arguments *before* the scope
        opens, so an ``initial_state=chip.state(...)`` expression inline in
        the call is built under the surrounding backend — fine for concrete
        chips (the state is coerced), but a chip carrying JAX tracers needs
        a JAX-capable surrounding backend (chip-level or process default)
        for that construction itself.

        Inherits the Hilbert-truncation safety net from :func:`~quchip.engine.solve_problem`;
        pass ``check_truncation=False`` to opt out or retune
        ``truncation_threshold``.

        ``partition``, default ``True``, forwards to
        :func:`~quchip.engine.simulate`: when the chip splits into
        independent sub-chips (:meth:`Chip.partition`), each component is
        dispatched separately and combined into a
        :class:`~quchip.results.partitioned.PartitionedSimulationResult`.
        This only engages when ``initial_state`` is ``None`` or a
        ``Mapping`` — a string shorthand or a concrete state is resolved
        via :meth:`_resolve_initial_state_spec` first and always takes the
        joint path. Pass ``partition=False`` to force the joint solve
        unconditionally.
        """
        from collections.abc import Mapping as _Mapping

        from quchip.engine import simulate as _engine_simulate

        with self._scoped_backend(backend):
            actual_tlist = self._resolve_tlist(tlist)
            drive_ops = self._materialize_drive_ops()
            if initial_state is None or isinstance(initial_state, _Mapping):
                resolved_state: Any = initial_state
            else:
                resolved_state = self._resolve_initial_state_spec(initial_state)
            return _engine_simulate(
                self._chip, drive_ops, actual_tlist,
                solver=solver, options=options, e_ops=e_ops,
                initial_state=resolved_state,
                check_truncation=check_truncation,
                truncation_threshold=truncation_threshold,
                partition=partition,
            )

    def simulate_batch(
        self,
        *axes: BatchAxis | ZippedBatchAxis,
        tlist: Any | None = None,
        solver: str | None = None,
        options: dict | None = None,
        e_ops: dict | None = None,
        initial_state: Any | None = None,
        backend: Any | None = None,
        progress: bool = True,
        check_truncation: bool = True,
        truncation_threshold: float = 1e-3,
    ) -> "SimulationBatchResult":
        """Build and solve a batched sweep. Equivalent to ``chip.solve_many(seq.build_batch(...))``.

        ``backend`` scopes this one call exactly as in :meth:`simulate`.

        The batched solve does not pass through :func:`~quchip.engine.solve_problem`,
        so the Hilbert-truncation safety net is applied per batch element here
        (default on); pass ``check_truncation=False`` to opt out or retune
        ``truncation_threshold``.
        """
        with self._scoped_backend(backend):
            problem_batch = self.build_batch(
                *axes,
                tlist=tlist,
                solver=solver,
                options=options,
                e_ops=e_ops,
                initial_state=initial_state,
            )
            result = self._chip.solve_many(problem_batch, progress=progress)
        if check_truncation:
            for element in result:
                element.check_truncation(threshold=truncation_threshold)
        return result

    def active_patch(self, *, hops: int = 1, method: str = "sw") -> "ActivePatchResult":
        """Reduce the chip to this schedule's active patch (spectators eliminated).

        Convenience for :func:`quchip.chip.transformations.active_patch`;
        see it for the activity rule, elimination order, and validity
        reporting.
        """
        from quchip.chip.transformations import active_patch as _active_patch

        return _active_patch(self, hops=hops, method=method)

    @staticmethod
    def _scoped_backend(backend: Any | None):
        """Context scoping a per-call backend override; no-op for ``None``."""
        from contextlib import nullcontext

        if backend is None:
            return nullcontext()
        from quchip.backend import _backend_context, _coerce_backend

        return _backend_context(_coerce_backend(backend))

    def clone(self) -> "QuantumSequence":
        """Return a deep copy.

        ``_entries`` is the single source of truth, so only it is deep-copied;
        the chip is shared, matching the previous shallow-copy behavior.
        """
        cloned = copy.copy(self)
        cloned._entries = copy.deepcopy(self._entries)
        return cloned

    def _validate_target(self, target: str) -> None:
        if target in self._chip.device_map:
            return
        available = list(self._chip.device_map.keys())
        hint = ""
        ce = self._chip.control_equipment
        if ce is not None:
            drive_labels = [d.label for d in ce.lines if d.label is not None]
            if target in drive_labels:
                hint = f" Hint: '{target}' is a drive label, not a device label. Pass the device label instead."
        raise ValueError(f"Device label '{target}' not found on chip. Available device labels: {available}.{hint}")

    def describe(self) -> str:
        """Plain-text timeline of the scheduled pulses.

        One row per pulse — resolved time window, drive → device,
        envelope with its declared parameters, carrier frequency — plus
        a trailing count of delays, barriers, and virtual-Z entries.
        Returns a string; ``print(seq.describe())``.
        """
        from quchip.chip.describe import describe_sequence

        return describe_sequence(self)

    def __repr__(self) -> str:
        return f"QuantumSequence({self._chip!r}, ops={len(self._entries)}, duration={self.total_duration!r} ns)"
