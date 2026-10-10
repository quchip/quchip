"""Declarative pulse programming for a :class:`~quchip.chip.chip.Chip`.

:class:`QuantumSequence` is the user-facing scheduling API. It tracks
per-endpoint timing cursors, per-device virtual-Z phase frames, and
cross-device barriers, then materializes the schedule into classical
:class:`~quchip.engine.ir.DriveOp` and field
:class:`~quchip.engine.ir.CoherentOp` records for the engine.

Conventions
-----------
- Times are in ns. Frequencies are in GHz (ordinary, not angular).
- Each carrier pulse follows the virtual-Z frame of one device. By default,
  this is the device that its line drives. Virtual-Z shifts accumulate into
  later carrier pulses in that frame on any line, but not into baseband flux
  pulses. This matches the lab-frame semantics of software-Z (McKay et al.,
  PRA 96, 022330 (2017)).
- All sweep axes stay JAX-traceable. Pulse parameters, delays, and envelope
  fields go into ``SolveProblem`` without Python-side concretization.

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
from dataclasses import dataclass, replace
from collections.abc import Collection, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from quchip.approximations import Approximation, require_approximation
from quchip.chip.chip import Chip
from quchip.chip.coupling_base import BaseCoupling
from quchip.control.batch import (
    BatchAxis,
    DelayHandle,
    PulseHandle,
    ZippedBatchAxis,
    _PULSE_FIELDS,
    _axis_metadata,
    _expand_axis_overrides,
)
from quchip.control.drive import (
    BaseDrive,
    ChargeDrive,
    CouplingDrive,
    FluxDrive,
    PhaseDrive,
)
from quchip.control.envelopes import Envelope
from quchip.control.field import CoherentInput, ControlEndpoint
from quchip.devices.base import BaseDevice
from quchip.engine.ir import StateStorage, CoherentOp, ControlOp, DriveOp, EngineResult
from quchip.engine.assembly import (
    build_engine_result,
    compile_hamiltonian_template,
    instantiate_engine_result,
)
from quchip.engine.frames import resolve_for_operations
from quchip.engine.sampling import AutomaticTimeGrid, interval_bounds, sample_problems
from quchip.engine.problem import (
    build_solve_batch_from_results,
    build_solve_problem,
    prepare_solve_problem_context,
)
from quchip.utils.jax_utils import (
    array_namespace,
    is_jax_array as _is_traced,
    is_jax_namespace,
    maybe_concrete_scalar,
    select_array_module as _select_array_module,
)
from quchip.utils.labeling import resolve_label
from quchip.utils.values import copy_value

if TYPE_CHECKING:
    from quchip.chip.transformations.active_patch import ActivePatchResult
    from quchip.declarative.expr import PhysicsExpr
    from quchip.engine.ir import FrameSpec, SolveBatch, SolveProblem
    from quchip.results.results import SimulationBatchResult, SimulationResult


@dataclass(frozen=True)
class _PulseEntry:
    target_label: str
    drive_label: str
    envelope: Envelope
    freq: float | None
    requested_start_time: float | None
    phase: float
    coherent_input: CoherentInput | None = None
    frame: str | None = None
    detuning: Any = None


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

    Tracks per-``(device, drive)`` timing cursors and per-device virtual-Z
    phase frames. Append pulses with :meth:`schedule` (or the conveniences
    :meth:`charge`, :meth:`phase`, :meth:`flux`). Synchronize channels with
    :meth:`barrier` and :meth:`delay`. :meth:`build_problem` and
    :meth:`build_batch` materialize the schedule lazily into engine
    control-operation records.

    Where a device/drive is expected, you can use the object or its string
    label.

    Simulation uses one consistent verb, ``simulate``, across three tiers, from
    most ergonomic to most explicit:

    1. :meth:`simulate` / :meth:`simulate_batch`: schedule *and* solve in one
       call (the example-facing path).
    2. :meth:`~quchip.chip.chip.Chip.solve` /
       :meth:`~quchip.chip.chip.Chip.solve_many`: solve a
       :class:`~quchip.engine.ir.SolveProblem` / batch you already hold.
    3. The module-level :func:`~quchip.engine.simulate` /
       :func:`~quchip.engine.solve_problem` /
       :func:`~quchip.engine.solve_many`: the low-level "I already have
       ``drive_ops`` / a ``SolveProblem``" tier.

    Parameters
    ----------
    chip : Chip
        Chip that this sequence schedules against. It supplies the device and
        coupling maps used to resolve scheduling targets, the wired
        :class:`~quchip.control.equipment.ControlEquipment` lines, and the
        frame/backend settings used by :meth:`build_problem` and
        :meth:`simulate`.

    Examples
    --------
    >>> from quchip import (
    ...     DuffingTransmon, ChargeDrive, Chip, QuantumSequence, Gaussian
    ... )
    >>> q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3)
    >>> drive = ChargeDrive(target=q)
    >>> chip = Chip([q])
    >>> chip.wire(drive)
    >>> seq = QuantumSequence(chip)
    >>> _ = seq.charge(q, envelope=Gaussian(duration=20.0, amplitude=0.05))
    >>> seq.vz(q, angle=0.5)
    >>> _ = seq.charge(q, envelope=Gaussian(duration=10.0, amplitude=0.03))
    """
    # Examples should use object references.

    def __init__(self, chip: Chip) -> None:
        ambiguous = [
            path
            for path in chip.parameters
            if len((parts := path.split(".", 2))) == 3
            and parts[0] == "pulse"
            and parts[1].isdigit()
        ]
        if ambiguous:
            raise ValueError(
                "Component labels matching 'pulse.<integer>' are reserved for "
                f"scheduled-pulse parameters; conflicting paths: {ambiguous}"
            )
        self._chip = chip
        # ``_entries`` is the single source of truth for all timing. Cursors,
        # floors, and virtual-Z phase frames are reconstructed on demand by
        # ``_replay`` — there is no live cursor state to drift from the replay.
        self._entries: list[_PulseEntry | _DelayEntry | _BarrierEntry | _FrameShiftEntry] = []

    def _device_drives(self, device_label: str) -> list[BaseDrive]:
        self._chip.device_map[device_label]
        equipment = self._chip.control_equipment
        return [] if equipment is None else [
            line for line in equipment.lines
            if not isinstance(line, CouplingDrive) and line.target_label == device_label
        ]

    def _find_drive_by_type(self, device_label: str, drive_type: type) -> BaseDrive:
        matches = [d for d in self._device_drives(device_label) if isinstance(d, drive_type)]
        if len(matches) == 0:
            raise ValueError(f"No {drive_type.__name__} on device '{device_label}'.")
        if len(matches) > 1:
            raise ValueError(
                f"Multiple {drive_type.__name__} on '{device_label}': "
                f"{[d.label for d in matches]}. Use schedule(drive, ...) instead."
            )
        return matches[0]

    def _find_default_drive(self, device_label: str) -> BaseDrive:
        drives = self._device_drives(device_label)
        if not drives:
            raise ValueError(
                f"Device '{device_label}' has no connected drives. "
                "Connect a drive to the device before scheduling."
            )
        return drives[0]

    def _find_coupling_drive(self, coupling_label: str) -> BaseDrive:
        """Return the unique control line targeting *coupling_label*."""
        equipment = self._chip.control_equipment
        lines = [] if equipment is None else [
            line
            for line in equipment.lines
            if isinstance(line, CouplingDrive)
            and line.target_label == coupling_label
        ]
        if len(lines) == 1:
            return lines[0]
        if not lines:
            raise ValueError(
                f"No coupling drive targets '{coupling_label}'. Wire a CouplingDrive, "
                "such as ParametricDrive(coupling), before scheduling."
            )
        raise ValueError(
            f"Multiple coupling drives target '{coupling_label}': {[line.label for line in lines]}. "
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

    def _pulse_frame(self, frame: str | BaseDevice | None, freq: Any, default: str | None) -> str | None:
        """Return the device whose virtual-Z phase a scheduled carrier follows."""
        if frame is None:
            return default
        if freq is None:
            raise ValueError("A frame requires a carrier: a pulse without freq has no carrier phase for vz() to shift.")
        label = resolve_label(frame)
        if label not in self._chip.device_map:
            raise ValueError(
                f"Frame '{label}' is not a device on this chip. "
                f"Available devices: {list(self._chip.device_map.keys())}"
            )
        return label

    @staticmethod
    def _pulse_detuning(detuning: Any, freq: Any) -> Any:
        """Return a start-referenced carrier offset after checking that the pulse has a carrier."""
        if detuning is not None and freq is None:
            raise ValueError("A detuning requires a carrier: pass freq, the frame frequency, with detuning.")
        return detuning

    def _schedule_on_drive(
        self,
        drive: BaseDrive,
        *,
        envelope: Envelope,
        freq: float | None,
        start_time: float | None = None,
        phase: float = 0.0,
        frame: str | BaseDevice | None = None,
        detuning: float | None = None,
    ) -> PulseHandle:
        if drive._target is None:
            raise ValueError(
                "Cannot schedule an unconnected drive. Connect it to its device "
                "or coupling first."
            )
        if isinstance(drive, CouplingDrive):
            target_label = drive.target_label
            if target_label not in self._chip.coupling_map:
                raise ValueError(
                    f"Coupling drive '{drive.label}' targets '{target_label}', which is not on this chip. "
                    f"Available couplings: {list(self._chip.coupling_map.keys())}"
                )
            default_frame = None
        else:
            target_label = drive.target_label
            assert target_label is not None
            if target_label not in self._chip.device_map:
                raise ValueError(
                    f"Drive is connected to device '{target_label}', which is not on this chip. "
                    f"Available devices: {list(self._chip.device_map.keys())}"
                )
            default_frame = target_label
        self._entries.append(
            _PulseEntry(
                target_label=target_label,
                drive_label=drive.label,
                envelope=envelope,
                freq=freq,
                requested_start_time=start_time,
                phase=phase,
                frame=self._pulse_frame(frame, freq, default_frame),
                detuning=self._pulse_detuning(detuning, freq),
            )
        )
        return PulseHandle(self, len(self._entries) - 1)

    def _schedule_on_coherent_input(
        self,
        coherent_input: CoherentInput,
        *,
        envelope: Envelope,
        freq: float | None,
        start_time: float | None = None,
        phase: float = 0.0,
        frame: str | BaseDevice | None = None,
        detuning: float | None = None,
    ) -> PulseHandle:
        exposures = [channel.key for channel in self._chip.resolve().slh.external_channels]
        if coherent_input.exposure not in exposures:
            raise ValueError(
                f"Unknown coherent-input exposure {coherent_input.exposure!r}. "
                f"Available exposures: {exposures}."
            )
        self._entries.append(
            _PulseEntry(
                target_label=coherent_input.exposure,
                drive_label=coherent_input.label,
                envelope=envelope,
                freq=freq,
                requested_start_time=start_time,
                phase=phase,
                coherent_input=coherent_input,
                frame=self._pulse_frame(frame, freq, None),
                detuning=self._pulse_detuning(detuning, freq),
            )
        )
        return PulseHandle(self, len(self._entries) - 1)

    def schedule(
        self,
        target: str | ControlEndpoint | BaseDevice | BaseCoupling,
        *,
        envelope: Envelope,
        freq: float | None = None,
        start_time: float | None = None,
        phase: float = 0.0,
        frame: str | BaseDevice | None = None,
        detuning: float | None = None,
    ) -> PulseHandle:
        """Schedule a pulse on *target*.

        Parameters
        ----------
        target : str | ControlEndpoint | BaseDevice | BaseCoupling
            Accepted forms, resolved in this order:

            * ``network.expose(...).input``: scheduled at that external
              reference plane without entering ``ControlEquipment``.
            * :class:`BaseDrive`: scheduled directly on that drive.
            * :class:`BaseDevice`: uses the device's first connected drive. If
              a device has multiple drives, pass the drive object explicitly.
            * :class:`~quchip.chip.coupling_base.BaseCoupling` — uses the
              unique :class:`~quchip.control.drive.CouplingDrive` that targets
              that coupling. If a coupling has multiple control lines, pass the
              drive object explicitly.
            * ``str``: a label, resolved in this order. First, a device label.
              Then, only when the label is absent from the device map, a
              coupling label (the two label spaces are disjoint). Then, only
              when the label is absent from both, a control-equipment line
              label, scheduled directly on that line. This third fallback lets
              a caller schedule by a drive's label after its device or coupling
              target is eliminated. See the retarget registry of
              :func:`~quchip.chip.transformations.eliminate`, which keeps a
              converted line's label. A control-line label that collides with a
              device/coupling label is shadowed by the device/coupling
              resolution.

        envelope : Envelope
            Pulse envelope.
        freq : float, optional
            Optional carrier frequency in GHz. Omitting it leaves the signal
            carrier-free. Drive-specific conveniences can supply their own
            explicit default, such as :meth:`charge` using ``chip.freq(device)``.
        start_time : float, optional
            Pulse start time in ns. Defaults to the current cursor; earlier
            times are rejected.
        phase : float
            Per-pulse phase offset, composed with any accumulated
            virtual-Z.
        frame : str or BaseDevice, optional
            Device whose virtual-Z frame the carrier follows. :meth:`vz` on
            that device shifts this pulse's phase. Defaults to the device that
            the drive line targets. Coupling-drive and coherent-input pulses
            have no default frame. A cross-resonance tone on the control's line
            names the target device, because its phase sets the axis of the
            target's conditional rotation. A frame requires ``freq``.
        detuning : float, optional
            Carrier offset ``δ`` in GHz, referenced to the pulse start ``t0``.
            The pulse delivers ``E(t - t0) exp(i phase) exp(-2πi freq t)
            exp(-2πi δ (t - t0))``. ``freq`` stays the frame frequency, so the
            pulse is the same in that frame at every ``t0``. A carrier at
            ``freq = f + δ`` without ``detuning`` is a fixed oscillator
            instead, with phase ``phase - 2π δ t0`` relative to the frame at
            ``f``. ``None`` adds no offset term. ``pulse.<index>.detuning``
            rebinds, sweeps, and differentiates like ``freq``. A detuning
            requires ``freq``.
        """
        if isinstance(target, CoherentInput):
            return self._schedule_on_coherent_input(
                target,
                envelope=envelope,
                freq=freq,
                start_time=start_time,
                phase=phase,
                frame=frame,
                detuning=detuning,
            )
        if isinstance(target, BaseDrive):
            drive = target
        else:
            label = resolve_label(target)
            if label in self._chip.device_map:
                drive = self._find_default_drive(label)
            elif label in self._chip.coupling_map:
                drive = self._find_coupling_drive(label)
            else:
                drive = self._find_drive_line(label)

        return self._schedule_on_drive(
            drive, envelope=envelope, freq=freq, start_time=start_time, phase=phase, frame=frame, detuning=detuning,
        )

    def charge(
        self,
        target: str | BaseDevice,
        *,
        envelope: Envelope,
        freq: float | None = None,
        phase: float = 0.0,
        frame: str | BaseDevice | None = None,
        detuning: float | None = None,
    ) -> PulseHandle:
        """Schedule a charge-drive pulse.

        Parameters
        ----------
        target : str or BaseDevice
            Driven device.
        envelope : Envelope
            Pulse envelope; time parameters are in ns and amplitudes in GHz.
        freq : float or None, default=None
            Carrier frequency in GHz; ``None`` uses ``chip.freq(target)``.
        phase : float, default=0.0
            Carrier phase in radians.
        frame : str or BaseDevice or None, default=None
            Device whose virtual-Z frame the carrier follows. ``None`` uses
            *target*. See :meth:`schedule`.
        detuning : float or None, default=None
            Carrier offset in GHz, referenced to the pulse start. See
            :meth:`schedule`.
        """
        label = resolve_label(target)
        self._validate_target(label)
        drive = self._find_drive_by_type(label, ChargeDrive)
        if freq is None:
            freq = self._chip.freq(label)
        return self._schedule_on_drive(
            drive, envelope=envelope, freq=freq, phase=phase, frame=frame, detuning=detuning,
        )

    def phase(
        self,
        target: str | BaseDevice,
        *,
        envelope: Envelope,
        freq: float,
        phase: float = 0.0,
        frame: str | BaseDevice | None = None,
        detuning: float | None = None,
    ) -> PulseHandle:
        """Schedule a phase-drive pulse.

        Parameters
        ----------
        target : str or BaseDevice
            Driven device.
        envelope : Envelope
            Pulse envelope; time parameters are in ns and amplitudes in GHz.
        freq : float
            Carrier frequency in GHz.
        phase : float, default=0.0
            Carrier phase in radians.
        frame : str or BaseDevice or None, default=None
            Device whose virtual-Z frame the carrier follows. ``None`` uses
            *target*. See :meth:`schedule`.
        detuning : float or None, default=None
            Carrier offset in GHz, referenced to the pulse start. See
            :meth:`schedule`.
        """
        label = resolve_label(target)
        self._validate_target(label)
        drive = self._find_drive_by_type(label, PhaseDrive)
        return self._schedule_on_drive(
            drive, envelope=envelope, freq=freq, phase=phase, frame=frame, detuning=detuning,
        )

    def flux(
        self,
        target: str | BaseDevice,
        *,
        envelope: Envelope,
    ) -> PulseHandle:
        """Schedule a baseband flux-drive pulse.

        Parameters
        ----------
        target : str or BaseDevice
            Driven flux-tunable device.
        envelope : Envelope
            Baseband pulse envelope in GHz versus time in ns.
        """
        label = resolve_label(target)
        self._validate_target(label)
        drive = self._find_drive_by_type(label, FluxDrive)
        return self._schedule_on_drive(drive, envelope=envelope, freq=None)

    def pump(
        self,
        coupling: str | BaseCoupling,
        *,
        envelope: Envelope,
        freq: float | None = None,
        start_time: float | None = None,
        phase: float = 0.0,
    ) -> PulseHandle:
        """Schedule an edge pump.

        Parameters
        ----------
        coupling : str or BaseCoupling
            Modulable coupling to pump.
        envelope : Envelope
            Pump-amplitude envelope in GHz versus time in ns.
        freq : float or None, default=None
            Carrier frequency in GHz; ``None`` applies the envelope at baseband.
        start_time : float or None, default=None
            Absolute start time in ns; ``None`` uses the channel cursor.
        phase : float, default=0.0
            Carrier phase in radians.
        """
        label = resolve_label(coupling)
        drive = self._find_coupling_drive(label)
        return self._schedule_on_drive(drive, envelope=envelope, freq=freq, start_time=start_time, phase=phase)

    def flux_to(
        self,
        target: str | BaseDevice,
        *,
        target_freq: Any,
        envelope: Envelope,
    ) -> PulseHandle:
        """Schedule a flux-drive frequency-shift pulse to ``target_freq``.

        This is not an inverse-SQUID flux calibration: it calculates
        ``δω = target_freq − chip.freq(device)`` and schedules a
        :class:`FluxDrive` pulse whose envelope amplitude is that frequency
        shift in GHz. The resulting Hamiltonian contribution is the linear
        detuning term ``δω(t) n̂``. Pass ``envelope`` with ``amplitude=None``
        (or any placeholder). This method replaces the amplitude with the
        calculated δω.

        ``target_freq`` can be a JAX tracer (for example ``chip.freq(other_device)``).

        Parameters
        ----------
        target : str | BaseDevice
            Device to flux-pulse.
        target_freq : float
            Target ``0 → 1`` frequency in GHz.
        envelope : Envelope
            Envelope template with all timing parameters set. Its ``amplitude``
            is replaced by the calculated δω, so any passed value is ignored.

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
        pulse = envelope.with_params({"amplitude": delta_omega})
        drive = self._find_drive_by_type(label, FluxDrive)
        return self._schedule_on_drive(drive, envelope=pulse, freq=None)

    def vz(self, target: str | BaseDevice, angle: float) -> None:
        """Apply a virtual-Z frame shift of *angle* rad on *target*.

        The shift is free (no pulse is emitted). It accumulates into the
        ``phase_offset`` of every later carrier pulse whose frame is *target*,
        on any line. A pulse's frame defaults to the device that its line
        drives, and ``schedule(..., frame=...)`` names another device. This
        matches the standard software-Z trick for transmons (McKay et al., PRA
        96, 022330 (2017)). Baseband flux pulses are not affected.

        Parameters
        ----------
        target : str or BaseDevice
            Device receiving the virtual frame shift.
        angle : float
            Z-frame shift in radians.
        """
        label = resolve_label(target)
        self._validate_target(label)
        self._entries.append(_FrameShiftEntry(device_label=label, angle=angle))

    def delay(self, scope: str | BaseDevice, duration: float) -> DelayHandle:
        """Insert an idle delay on all drive channels of a target.

        Parameters
        ----------
        scope : str or BaseDevice
            Device label or object whose channels advance.
        duration : float
            Delay in ns.
        """
        target = resolve_label(scope)
        self._validate_target(target)
        _require_positive_duration(duration)
        self._entries.append(_DelayEntry(device_label=target, duration=duration))
        return DelayHandle(self, len(self._entries) - 1)

    def barrier(self, *labels: str | BaseDevice) -> None:
        """Synchronize channel cursors.

        With no arguments, every ``(device, drive)`` cursor on the chip is
        advanced to the current maximum. With explicit device targets, only
        those devices' cursors are aligned. This guarantees that pulses
        scheduled after the barrier start no earlier than any pulse scheduled
        before it.

        Parameters
        ----------
        *labels : str or BaseDevice
            Devices to synchronize. If omitted, all device-drive cursors are synchronized.
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
    def scheduled_ops(self) -> tuple[ControlOp, ...]:
        """Materialize scheduled classical-drive and coherent-field operations."""
        return tuple(self._materialize_drive_ops())

    @property
    def channel_cursors(self) -> dict[tuple[str, str], Any]:
        """Current time cursors keyed by ``(target_label, drive_label)``."""
        return self._replay_cursors()[0]

    def _resolve_tlist(
        self, tlist: Any | None, *, duration: Any | None = None,
        overrides: Mapping[tuple[int, str], Any] | None = None,
    ) -> Any:
        """Preserve an explicit grid, or sample an interval starting at zero.

        A duration may extend the schedule. Explicit grids may select any
        partial interval; their first time is the initial-state time.
        """
        if tlist is not None:
            if duration is not None:
                raise ValueError("duration and tlist are mutually exclusive.")
            return tlist if isinstance(tlist, tuple) or is_jax_namespace(array_namespace(tlist)) else np.asarray(tlist)

        cursors, _ = self._replay_cursors(overrides)
        end = maybe_concrete_scalar(_cursor_max(cursors.values()) if cursors else 0.0)
        if end is None:
            raise ValueError(
                "Automatic sampling requires concrete sequence timing. "
                "Pass an explicit tlist when timing depends on traced parameters."
            )
        value = end if duration is None else maybe_concrete_scalar(duration)
        if value is None:
            raise ValueError("Automatic sampling requires a concrete duration; pass an explicit tlist.")
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Provide a finite, positive duration or an explicit tlist.")
        if not np.isfinite(end) or value < end:
            raise ValueError(f"duration must reach the schedule end at {end:g} ns.")
        return AutomaticTimeGrid(value)

    def _replay(
        self,
        overrides: Mapping[tuple[int, str], Any] | None = None,
        *,
        collect_ops: bool,
    ) -> tuple[dict[tuple[str, str], Any], dict[str, Any], list[ControlOp]]:
        """Replay ``_entries`` into final cursors, floors, and optional control ops.

        This is the single implementation of the delay-shift, barrier-alignment,
        floor, default-start, and virtual-Z phase semantics. Both the cursor
        accessors (:meth:`_replay_cursors`) and materialization
        (:meth:`_materialize_drive_ops`) run through it, so there is no way for a
        cursor read and a materialized schedule to disagree.

        Replaying the whole entry list on each call is O(n) per invocation
        (O(n^2) across a full build); this is an intentional trade for a single
        source of truth, and sequences carry only modest pulse counts.

        ``overrides`` maps ``(entry_index, field) -> value``; only ``duration``
        (on a delay entry or pulse envelope) and the pulse fields
        ``start_time``/``freq``/``phase``/``detuning`` affect the replay.
        ``collect_ops`` gates operation building and the virtual-Z phase of
        each pulse's frame.
        """
        overrides = {} if overrides is None else dict(overrides)
        cursors: dict[tuple[str, str], Any] = {}
        equipment = self._chip.control_equipment
        if equipment is not None:
            for line in equipment.lines:
                assert line.target_label is not None
                cursors[(line.target_label, line.label)] = 0.0
        for entry in self._entries:
            if isinstance(entry, _PulseEntry) and entry.coherent_input is not None:
                cursors[(entry.target_label, entry.drive_label)] = 0.0
        floors: dict[str, Any] = {}
        phases: dict[str, Any] = {dev.label: 0.0 for dev in self._chip.devices}
        drive_ops: list[ControlOp] = []

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
                field.removeprefix("envelope."): value
                for (idx, field), value in overrides.items()
                if idx == entry_index
                and (field.startswith("envelope.") or field not in _PULSE_FIELDS)
            }
            envelope = (
                entry.envelope.with_params(envelope_updates)
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
                detuning = self._pulse_detuning(overrides.get((entry_index, "detuning"), entry.detuning), freq)
                # A carrier follows its frame's virtual-Z phase. A baseband pulse has no carrier phase.
                frame = None if freq is None else entry.frame
                phase_offset = phase if frame is None else phase + phases[frame]
                if entry.coherent_input is None:
                    drive_ops.append(
                        DriveOp(
                            target_label=entry.target_label,
                            envelope=envelope,
                            freq=freq,
                            start_time=start_time,
                            phase_offset=phase_offset,
                            drive_label=entry.drive_label,
                            frame=frame,
                            detuning=detuning,
                        )
                    )
                else:
                    drive_ops.append(
                        CoherentOp(
                            coherent_input=entry.coherent_input,
                            envelope=envelope,
                            freq=freq,
                            start_time=start_time,
                            phase_offset=phase_offset,
                            frame=frame,
                            detuning=detuning,
                        )
                    )
            cursors[key] = start_time + envelope.duration
        return cursors, floors, drive_ops

    def _replay_cursors(
        self, overrides: Mapping[tuple[int, str], Any] | None = None
    ) -> tuple[dict[tuple[str, str], Any], dict[str, Any]]:
        """Replay ``_entries`` to final ``(cursors, floors)`` without building operations."""
        cursors, floors, _ = self._replay(overrides, collect_ops=False)
        return cursors, floors

    def _materialize_drive_ops(self, overrides: Mapping[tuple[int, str], Any] | None = None) -> list[ControlOp]:
        """Replay ``_entries`` into the scheduled control-operation list."""
        return self._replay(overrides, collect_ops=True)[2]

    def build_problem(
        self,
        tlist: Any | None = None,
        solver: str | None = None,
        options: dict | None = None,
        e_ops: dict | None = None,
        initial_state: Any | None = None,
        approximation: Approximation | None = None,
        frame: FrameSpec | None = None,
        states: StateStorage | None = None,
        dissipation: bool = True,
        run_args: dict | None = None,
        *,
        duration: Any | None = None,
    ) -> "SolveProblem":
        """Build a single :class:`~quchip.engine.ir.SolveProblem` from this sequence.

        Parameters
        ----------
        tlist : array-like, optional
            Grid passed to the numerical solver, in ns. Its first time is the
            initial-state time and its last time ends the calculation.
            Scheduled signals keep their absolute times in a partial interval.
        duration : float, optional
            Interval from zero in ns, with automatic sampling. It can extend
            the schedule but cannot cut it short, and cannot be combined with
            ``tlist``. Without either argument, the scheduled duration is used.
        solver : str, optional
            Backend solver name; defaults to the backend's own default solver.
        options : dict, optional
            Backend numerical options. Use ``states`` to select state retention.
        states : {"all", "final", "none"} or None, default None
            None saves all deterministic states and preserves native stochastic defaults.
            Explicit values keep the full state history, only the final state, or neither.
            Requested observable traces are kept independently.
        run_args : dict or None, default None
            Native trajectory call keywords: QuTiP seeds, ntraj, heterodyne,
            target_tol, timeout; Dynamiqs keys, method, gradient, etas.
            Assembled physics and native options cannot be overridden here.
        e_ops : dict, optional
            Expectation operators keyed by device label (or a 2-tuple of device
            labels for a two-body operator), mapped to a local operator (or a
            pair of local operators). They are decomposed into per-band terms
            before reaching the solver.
        initial_state : Any, optional
            Initial state; see :meth:`simulate` for accepted forms, coordinate
            conventions, and the meaning of ``None``.
        approximation : Approximation, optional
            Engine approximation override. ``None`` uses the chip default.
        frame : FrameSpec, optional
            Integration-frame override; defaults to the chip's declared frame.
            ``"auto"`` derives constraints from delivered scheduled signals and
            weights them over ``tlist[-1] - tlist[0]``.
        dissipation : bool, default=True
            Include authored collapse channels when true.

        Returns
        -------
        SolveProblem
            Frozen problem (chip, compiled engine result, initial state,
            ``tlist``, collapse operators, and expectation operators), ready
            for :meth:`~quchip.chip.chip.Chip.solve` or
            :func:`~quchip.engine.solve_problem`.
        """
        actual_tlist = self._resolve_tlist(tlist, duration=duration)
        drive_ops = self._materialize_drive_ops()
        return build_solve_problem(
            self._chip,
            drive_ops,
            actual_tlist,
            solver=solver,
            options=options, run_args=run_args,
            e_ops=e_ops,
            initial_state=initial_state,
            approximation=approximation,
            states=states,
            dissipation=dissipation,
            frame=frame,
        )
    def resolve(
        self,
        *,
        frame: FrameSpec | None = None,
        approximation: Approximation | None = None,
    ) -> EngineResult:
        """Resolve the backend-neutral Hamiltonian and noise description.

        Parameters
        ----------
        frame : FrameSpec or None, default=None
            Resolution frame, or ``None`` to use the chip default.
        approximation : Approximation or None, default=None
            Engine approximation, or ``None`` to use the chip default.
        """
        drive_ops = self._materialize_drive_ops()
        base_result = resolve_for_operations(self._chip, drive_ops, frame=frame, approximation=approximation)
        return build_engine_result(
            self._chip,
            drive_ops,
            resolved_frame=base_result.resolved_frame,
            approximation=base_result.approximation,
            _base_result=base_result,
        )

    def hamiltonian(self) -> PhysicsExpr:
        """Return this sequence's canonical time-dependent Hamiltonian."""
        return self.resolve().hamiltonian()

    def vary(self, field: str, values: Any, *, name: str | None = None) -> BatchAxis:
        """Create a :class:`BatchAxis` over a public parameter path or state.

        Parameters
        ----------
        field : str
            ``"initial_state"`` or any dotted path exposed by
            :attr:`parameters`. Pulse handles remain the concise way to vary
            a pulse field immediately after scheduling it.
        values : array-like
            Sequence of values for ``field``, one per batch point.
        name : str, optional
            Axis name recorded in :attr:`SolveBatch.axes` and
            :meth:`SolveBatch.params_at`. Defaults to *field*.

        Returns
        -------
        BatchAxis
            Sequence-level axis, consumable by :meth:`build_batch` or
            :meth:`zip`.
        """
        if field == "initial_state":
            return BatchAxis(owner=self, target_kind="sequence", field=field, values=values, name=name or field)
        if field not in self.parameters:
            raise ValueError(
                f"Parameter path {field!r} is not batchable. Available: {list(self.parameters)}"
            )

        parts = field.split(".", 2)
        if len(parts) == 3 and parts[0] == "pulse" and parts[1].isdigit():
            index = int(parts[1])
            entry = self._entries[index]
            entry_field = parts[2]
            return BatchAxis(
                owner=self,
                target_kind="entry",
                field=field,
                values=values,
                name=name or field,
                entry_index=index,
                entry_field=entry_field,
                entry=entry,
            )
        return BatchAxis(owner=self, target_kind="parameter", field=field, values=values, name=name or field)

    def zip(self, *axes: BatchAxis) -> ZippedBatchAxis:
        """Zip axes into one pairwise dimension.

        Parameters
        ----------
        *axes : BatchAxis
            Two or more axes to zip, each created by this sequence
            (:meth:`vary`, :meth:`PulseHandle.vary`, or
            :meth:`DelayHandle.vary`). All axes need the same length.

        Returns
        -------
        ZippedBatchAxis
            Combined axis that :meth:`build_batch` treats as a single
            dimension: point ``i`` supplies point ``i`` from every zipped axis
            together, not the outer product.
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
        seen_targets: set[tuple[int | None, str]] = set()
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
                if member.override_key in seen_targets:
                    raise ValueError(
                        f"Batch axis {member.name!r} binds the same parameter as another axis; "
                        "each parameter can be varied only once."
                    )
                seen_targets.add(member.override_key)
                if member.entry_index is not None and (
                    member.entry_index >= len(self._entries)
                    or self._entries[member.entry_index] is not member.entry
                ):
                    raise ValueError(
                        f"Batch axis {member.name!r} no longer matches its scheduled entry — "
                        "the sequence was modified after vary() was called. Re-schedule and "
                        "create the axis from a fresh handle."
                    )

    def build_batch(
        self,
        *axes: BatchAxis | ZippedBatchAxis,
        tlist: Any | None = None,
        solver: str | None = None,
        options: dict | None = None,
        e_ops: dict | None = None,
        initial_state: Any | None = None,
        approximation: Approximation | None = None,
        states: StateStorage | None = None,
        dissipation: bool = True,
        run_args: dict | None = None,
        duration: Any | None = None,
    ) -> "SolveBatch":
        """Build a batched solve request from explicit sweep axes.

        Parameters
        ----------
        *axes : BatchAxis or ZippedBatchAxis
            Independent or pairwise-zipped sweep axes.
        tlist : array-like or None, default=None
            Solver time grid in ns.
        solver : str or None, default=None
            Backend solver name; ``None`` uses its default.
        options : dict or None, default=None
            Backend numerical options.
        run_args : dict or None, default None
            Native trajectory call keywords: QuTiP seeds, ntraj, heterodyne,
            target_tol, timeout; Dynamiqs keys, method, gradient, etas.
            Assembled physics and native options cannot be overridden here.
        e_ops : dict or None, default=None
            Local expectation-operator specifications.
        initial_state : Any or None, default=None
            State reused at every batch point; see :meth:`simulate` for accepted
            forms and coordinate conventions. An ``initial_state`` axis supplies
            one state per point.
        approximation : Approximation or None, default=None
            Engine approximation override.
        states : {"all", "final", "none"} or None, default=None
            None saves all deterministic states and preserves native stochastic defaults.
            Explicit values select state retention; native storage conflicts raise.
        dissipation : bool, default=True
            Include authored collapse channels when true.
        duration : float or None, default=None
            Automatically sampled interval from zero, in ns.
        """
        self._validate_axes(axes)
        shape, expanded = _expand_axis_overrides(axes)
        params_store = np.empty(shape if shape else (), dtype=object)

        parameter_axes = any(
            member.target_kind == "parameter"
            for axis in axes
            for member in (axis.axes if isinstance(axis, ZippedBatchAxis) else (axis,))
        )
        point_tlists = [
            self._resolve_tlist(tlist, duration=duration, overrides=self._entry_overrides(overrides))
            for _, overrides in expanded
        ]
        actual_tlist = point_tlists[0] if point_tlists else self._resolve_tlist(tlist, duration=duration)
        different_intervals = tlist is None and any(
            grid != actual_tlist for grid in point_tlists
        )
        if parameter_axes or different_intervals or self._auto_frames_differ(expanded, actual_tlist, approximation):
            problems: list[Any] = []
            for (coord, overrides), point_tlist in zip(expanded, point_tlists):
                parameter_bindings = {
                    field: value
                    for (index, field), value in overrides.items()
                    if index is None and field != "initial_state"
                }
                entry_overrides = self._entry_overrides(overrides)
                axis_initial_state = overrides.get((None, "initial_state"))
                if axis_initial_state is not None and initial_state is not None:
                    raise ValueError(
                        "initial_state may be provided either as a shared scalar or as a batch axis, not both"
                    )

                variant = self.with_params(parameter_bindings)
                drive_ops = variant._materialize_drive_ops(entry_overrides)
                problems.append(
                    build_solve_problem(
                        variant._chip,
                        drive_ops,
                        point_tlist.bounds if isinstance(point_tlist, AutomaticTimeGrid) else point_tlist,
                        solver=solver,
                        options=options, run_args=run_args,
                        e_ops=e_ops,
                        initial_state=(axis_initial_state if axis_initial_state is not None else initial_state),
                        approximation=approximation,
                        states=states,
                        dissipation=dissipation,
                    )
                )
                params_store[coord] = self._point_params(axes, coord)

            from quchip.engine.ir import SolveBatch

            if tlist is None:
                problems = sample_problems(problems)
            from quchip.engine.problem import assign_point_noise

            return SolveBatch(
                chip=self._chip,
                problems=tuple(assign_point_noise(problems, split_keys=True)),
                params=params_store,
                shape=shape,
                axes=tuple(_axis_metadata(axis) for axis in axes),
            )

        reference_drive_ops = self._materialize_drive_ops()
        context = prepare_solve_problem_context(
            self._chip,
            actual_tlist,
            solver=solver,
            options=options, run_args=run_args,
            e_ops=e_ops,
            drive_ops=reference_drive_ops,
            approximation=approximation,
            states=states,
            dissipation=dissipation,
        )
        template = compile_hamiltonian_template(
            self._chip,
            reference_drive_ops,
            resolved_frame=context.resolved_frame,
            approximation=context.approximation,
            _base_result=context._base_result,
        )
        reference_result = instantiate_engine_result(template, reference_drive_ops, self._chip)

        engine_results: list[Any] = []
        initial_states: list[Any] = []
        for coord, overrides in expanded:
            axis_initial_state = overrides.get((None, "initial_state"))
            entry_overrides = self._entry_overrides(overrides)
            if axis_initial_state is not None and initial_state is not None:
                raise ValueError("initial_state may be provided either as a shared scalar or as a batch axis, not both")
            engine_result = reference_result
            if entry_overrides:
                # Scalar-only entry overrides keep the skeleton; engine errors must propagate.
                drive_ops = self._materialize_drive_ops(entry_overrides)
                engine_result = instantiate_engine_result(template, drive_ops, self._chip)
            engine_results.append(engine_result)
            initial_states.append(axis_initial_state if axis_initial_state is not None else initial_state)

            params_store[coord] = self._point_params(axes, coord)

        batch = build_solve_batch_from_results(
            context, engine_results, initial_states=initial_states
        )
        from quchip.engine.problem import assign_point_noise

        return replace(
            batch,
            problems=tuple(assign_point_noise(list(batch.problems), split_keys=True)),
            params=params_store,
            shape=shape,
            axes=tuple(_axis_metadata(axis) for axis in axes),
        )
    def _auto_frames_differ(self, expanded: Any, tlist: Any, approximation: Approximation | None) -> bool:
        """Return whether entry-axis values produce distinct or traced ``"auto"`` frames.

        A true result sends the batch through per-point problem construction.
        """
        from quchip.engine.frames import plan_for_operations

        if not (isinstance(self._chip.frame, str) and self._chip.frame == "auto"):
            return False
        strategy = self._chip.approximation if approximation is None else require_approximation(approximation)
        window = interval_bounds(tlist)
        reference = plan_for_operations(self._chip, "auto", self._materialize_drive_ops(),
                                        approximation=strategy, solve_window=window)
        try:
            keys: set[Any] = {reference.concrete_key()}
        except ValueError:
            return True
        for _, overrides in expanded:
            drive_ops = self._materialize_drive_ops(self._entry_overrides(overrides))
            plan = plan_for_operations(self._chip, "auto", drive_ops, approximation=strategy, solve_window=window)
            try:
                keys.add(plan.concrete_key())
            except ValueError:
                return True
            if len(keys) > 1:
                return True
        return False

    @staticmethod
    def _entry_overrides(overrides: Mapping[tuple[int | None, str], Any]) -> dict[tuple[int, str], Any]:
        """Return only the per-entry (non-shared) overrides of one batch point."""
        return {(index, field): value for (index, field), value in overrides.items() if index is not None}

    @staticmethod
    def _point_params(
        axes: Sequence[BatchAxis | ZippedBatchAxis], coord: tuple[int, ...]
    ) -> dict[str, Any]:
        """Return public axis values at one Cartesian coordinate."""
        point: dict[str, Any] = {}
        for dim, axis in enumerate(axes):
            members = axis.axes if isinstance(axis, ZippedBatchAxis) else (axis,)
            for member in members:
                point[member.name] = member.values[coord[dim]]
        return point

    def simulate(
        self,
        tlist: Any | None = None,
        solver: str | None = None,
        options: dict | None = None,
        e_ops: dict | None = None,
        initial_state: Any | None = None,
        *,
        backend: Any | None = None,
        partition: bool = True,
        approximation: Approximation | None = None,
        states: StateStorage | None = None,
        dissipation: bool = True,
        run_args: dict | None = None,
        duration: Any | None = None,
    ) -> "SimulationResult":
        """Build and solve one scheduled simulation.

        Parameters
        ----------
        tlist : array-like, optional
            Sample times in ns. Do not use with ``duration``. If omitted,
            quchip builds an automatic grid through the schedule end.
        solver : str, optional
            Backend solver name, such as ``"mesolve"`` or ``"sesolve"``.
        options : dict, optional
            Backend-specific solver options.
        run_args : dict or None, default None
            Native trajectory call keywords: QuTiP seeds, ntraj, heterodyne,
            target_tol, timeout; Dynamiqs keys, method, gradient, etas.
            Assembled physics and native options cannot be overridden here.
        e_ops : dict, optional
            Named observables to evaluate.
        initial_state : object or mapping, optional
            ``None`` uses the eigenstate of the solve's undriven static lab-frame Hamiltonian, as
            kept by its approximation, that is assigned to the all-ground label. Its overlap with
            the bare product is made real and nonnegative before it is expressed in the solve frame
            at ``tlist[0]``. With ordinary couplings, this is the bare product under the default
            RWA. When the solve uses the chip's approximation, it is the same physical state as
            ``chip.state()`` before the frame transform. Kept bands, effective terms, or network
            Hamiltonian terms that couple the vacuum can also dress an RWA start. At a nonzero
            rotating-frame start time, the frame transform is applied at ``tlist[0]``, so the state
            need not match the lab-frame vector returned by ``chip.state()``. Mappings and
            configured string shorthand give product states in the resolved local bases. QuTiP
            ``Qobj`` and dynamiqs ``QArray`` kets or density matrices use resolved solver
            coordinates. Arrays, symbolic expressions, and callables are authored-space kets
            projected onto kept levels.
        backend : object or {"qutip", "dynamiqs"}, optional
            Per-call backend override.
        partition : bool, default=True
            Solve independent chip components separately when the initial state
            permits it.
        approximation : Approximation, optional
            Override the chip's approximation for this solve.
        states : {"all", "final", "none"} or None, default=None
            None saves all deterministic states and preserves native stochastic defaults.
            State storage policy for the returned result.
        dissipation : bool, default=True
            Include declared collapse channels when ``True``.
        duration : float, optional
            Positive simulation end time in ns when ``tlist`` is omitted.

        Returns
        -------
        SimulationResult
            Time samples, requested observables, and states according to
            ``states``.

        Raises
        ------
        ValueError
            If ``tlist`` and ``duration`` are both supplied, timing is invalid,
            or automatic sampling lacks the concrete timing it requires.

        Notes
        -----
        A per-call ``backend`` outranks the chip and process defaults, and
        foreign-native initial states are coerced at the solve boundary. Python
        evaluates an inline ``initial_state=chip.state(...)`` before this scope
        opens, so a traced state still requires a JAX-capable surrounding
        backend.

        Partitioning returns a
        :class:`~quchip.results.partitioned.PartitionedSimulationResult` and
        requires ``initial_state`` to be ``None`` or a mapping; string
        shorthand and concrete states take the joint path.
        """
        from quchip.engine import simulate as _engine_simulate

        with self._scoped_backend(backend):
            actual_tlist = self._resolve_tlist(tlist, duration=duration)
            drive_ops = self._materialize_drive_ops()
            return _engine_simulate(
                self._chip, drive_ops, actual_tlist,
                solver=solver, options=options, run_args=run_args, e_ops=e_ops,
                initial_state=initial_state,
                partition=partition,
                approximation=approximation,
                states=states,
                dissipation=dissipation,
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
        approximation: Approximation | None = None,
        states: StateStorage | None = None,
        dissipation: bool = True,
        run_args: dict | None = None,
        duration: Any | None = None,
    ) -> "SimulationBatchResult":
        """Build and solve a batched sweep over pulse or sequence parameters.

        Parameters
        ----------
        *axes : BatchAxis or ZippedBatchAxis
            Cartesian sweep axes. A :class:`ZippedBatchAxis` pairs member axes
            pointwise and contributes one batch dimension.
        tlist, solver, options, run_args, e_ops, initial_state, backend
            As for :meth:`simulate`; one initial state is reused for every
            batch point unless it is a supported mapping.
        progress : bool, default=True
            Show backend batch progress when supported.
        approximation : Approximation, optional
            Per-batch approximation override.
        states : {"all", "final", "none"} or None, default=None
            None saves all deterministic states and preserves native stochastic defaults.
            State storage policy for each batch result.
        dissipation : bool, default=True
            Include declared collapse channels.
        duration : float, optional
            Positive end time in ns when ``tlist`` is omitted.

        Returns
        -------
        SimulationBatchResult
            Results indexed by the Cartesian or zipped axis shape.

        Raises
        ------
        ValueError
            If an axis is invalid, timing is inconsistent, or a sweep changes a
            structural quantity that cannot vary in one batch.

        """
        with self._scoped_backend(backend):
            problem_batch = self.build_batch(
                *axes,
                tlist=tlist,
                duration=duration,
                solver=solver,
                options=options, run_args=run_args,
                e_ops=e_ops,
                initial_state=initial_state,
                approximation=approximation,
                states=states,
                dissipation=dissipation,
            )
            result = self._chip.solve_many(
                problem_batch, progress=progress,
            )
        return result
    def active_patch(self, *, hops: int = 1, method: str = "sw") -> "ActivePatchResult":
        """Reduce the chip to this schedule's active patch (spectators eliminated).

        Convenience for :func:`quchip.chip.transformations.active_patch`;
        see it for the activity rule, elimination order, and validity
        reporting.

        Parameters
        ----------
        hops : int, default=1
            Coupling-graph expansion beyond scheduled targets.
        method : {"sw", "exact"}, default="sw"
            Reduction method forwarded to each elimination.
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
        """Return an independently editable model and pulse schedule."""
        return self._copy_with_chip(self._chip.clone())

    def _copy_with_chip(self, chip: Chip) -> "QuantumSequence":
        """Copy authored schedule entries onto an already independent chip."""
        cloned = copy.copy(self)
        cloned._chip = chip
        cloned._entries = copy.deepcopy(self._entries)
        return cloned

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Chip and scheduled-pulse values available to :meth:`with_params`."""
        values = dict(self._chip.parameters)
        for index, entry in enumerate(self._entries):
            if not isinstance(entry, _PulseEntry):
                continue
            prefix = f"pulse.{index}"
            pulse_values = {
                f"{prefix}.freq": entry.freq,
                f"{prefix}.phase": entry.phase,
                f"{prefix}.start_time": entry.requested_start_time,
                f"{prefix}.detuning": entry.detuning,
            }
            collision = values.keys() & pulse_values.keys()
            if collision:
                raise ValueError(f"Sequence parameter paths are ambiguous: {sorted(collision)}")
            values.update(pulse_values)
            for name, value in entry.envelope.parameter_values().items():
                # Carrier phase and an envelope's own phase are distinct.
                field = (
                    f"envelope.{name}"
                    if name in _PULSE_FIELDS
                    else name
                )
                path = f"{prefix}.{field}"
                if path in values:
                    raise ValueError(f"Sequence parameter path {path!r} is ambiguous")
                values[path] = value
        return MappingProxyType(values)

    @property
    def settings(self) -> Mapping[str, Any]:
        """Read-only sequence structure, separate from numerical parameters."""
        return MappingProxyType(
            {
                "chip": self._chip.settings,
                "entries": tuple(type(entry).__name__.removeprefix("_") for entry in self._entries),
            }
        )

    def with_params(self, bindings: Mapping[str, Any]) -> "QuantumSequence":
        """Return a cloned sequence with values rebound.

        Parameters
        ----------
        bindings : mapping[str, Any]
            Chip parameter paths or ``pulse.<index>`` fields and new values.
        """
        available = self.parameters
        unknown = set(bindings) - set(available)
        if unknown:
            raise KeyError(
                f"Unknown sequence parameter paths: {sorted(unknown)}. "
                f"Available: {list(available)}"
            )

        pulse_bindings: dict[int, dict[str, Any]] = {}
        chip_bindings: dict[str, Any] = {}
        for path, value in bindings.items():
            parts = path.split(".", 2)
            if len(parts) == 3 and parts[0] == "pulse" and parts[1].isdigit():
                pulse_bindings.setdefault(int(parts[1]), {})[parts[2]] = value
            else:
                chip_bindings[path] = value
        cloned = self._copy_with_chip(self._chip.with_params(chip_bindings))

        for index, updates in pulse_bindings.items():
            entry = cloned._entries[index]
            if not isinstance(entry, _PulseEntry):  # pragma: no cover - inventory excludes it
                raise KeyError(f"pulse.{index}")
            envelope_updates = {
                field.removeprefix("envelope."): value
                for field, value in updates.items() if field not in _PULSE_FIELDS
            }
            entry_updates = copy_value({
                "requested_start_time" if field == "start_time" else field: value
                for field, value in updates.items() if field in _PULSE_FIELDS
            })
            if envelope_updates:
                entry_updates["envelope"] = entry.envelope.with_params(envelope_updates)
            cloned._entries[index] = replace(entry, **entry_updates)
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

        One row per pulse shows the resolved time window, drive → device,
        envelope with its declared parameters, and carrier frequency, followed
        by a count of delays, barriers, and virtual-Z entries. A pulse whose
        frame differs from its device also names that frame. Returns a
        string: ``print(seq.describe())``.
        """
        from quchip.chip.describe import describe_sequence

        return describe_sequence(self)

    def __repr__(self) -> str:
        return f"QuantumSequence({self._chip!r}, ops={len(self._entries)}, duration={self.total_duration!r} ns)"
