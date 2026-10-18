"""Native gates: a pulse template, its calibration knobs, and its ideal unitary."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from quchip.control.batch import _PULSE_FIELDS, PulseHandle
from quchip.declarative.parameters import parameter_fields
from quchip.gateset.ideals import canonical_name, ideal_unitary
from quchip.utils.labeling import resolve_label

if TYPE_CHECKING:
    from quchip.control.sequence import QuantumSequence


@dataclass(frozen=True, eq=False)
class Knob:
    """A calibration parameter that moves one pulse field or several tied ones.

    Parameters
    ----------
    ref : PulseHandle, str, or sequence of (ref, factor)
        What the knob moves: a pulse handle with ``field``, a
        ``pulse.<index>.<field>`` path from ``sequence.parameters``, or a tie.
        A tie lists ``(ref, factor)`` pairs, where each ref is a path or a
        ``(handle, field)`` pair. Each tied value moves from its template
        value by its factor times the knob's change, so a fixed offset such as
        a π phase stays.
    field : str or None, default None
        Pulse field of a handle, such as ``"amplitude"``, ``"beta"``,
        ``"phase"``, or ``"detuning"``.
    bounds : (float or None, float or None) or None, default None
        Bounds of the knob value. The default bounds follow the field: a
        positive envelope field such as ``duration`` keeps a lower bound of 0.
    scale : float or None, default None
        Typical change of the knob, which scales the fit. The default is
        1 MHz for ``freq`` and ``detuning``, 0.1 rad for ``phase``,
        ``max(|x0|, 0.1 ns)`` for envelope times, and ``max(|x0|, 1e-3)``
        otherwise.
    fixed : bool, default False
        Keep the template value during calibration.
    """

    ref: Any
    field: str | None = None
    bounds: tuple[float | None, float | None] | None = None
    scale: float | None = None
    fixed: bool = False


@dataclass(frozen=True)
class ResolvedKnob:
    """A knob resolved against one template sequence.

    Attributes
    ----------
    name : str
        Knob name in the native gate.
    targets : tuple of (str, float)
        ``pulse.<index>.<field>`` paths and their tie factors.
    template : tuple of float
        Template value of each target.
    initial : float
        Knob value of the template.
    scale : float
        Fit scale.
    lower, upper : float
        Bounds, ``-inf`` or ``inf`` when open.
    fixed : bool
        Whether calibration keeps the template value.
    """

    name: str
    targets: tuple[tuple[str, float], ...]
    template: tuple[float, ...]
    initial: float
    scale: float
    lower: float
    upper: float
    fixed: bool

    def bindings(self, value: float) -> dict[str, float]:
        """Return the pulse-field values for one knob value."""
        delta = value - self.initial
        return {path: base + factor * delta for (path, factor), base in zip(self.targets, self.template)}


def _target_path(ref: Any, field: str | None, sequence: "QuantumSequence") -> str:
    """Resolve a handle or a path to a ``pulse.<index>.<field>`` path of *sequence*."""
    if isinstance(ref, PulseHandle):
        if ref._sequence is not sequence:
            raise ValueError("A knob's pulse handle belongs to another sequence than the native's template.")
        if field is None:
            raise ValueError("A knob on a pulse handle needs a field, such as 'amplitude'.")
        ref._resolve_entry()
        return f"pulse.{ref._entry_index}.{ref._normalize_field(field)}"
    if isinstance(ref, str):
        path = ref if field is None else f"{ref}.{field}"
        parts = path.split(".", 2)
        available = [key for key in sequence.parameters if key.startswith("pulse.")]
        if not (len(parts) == 3 and parts[0] == "pulse" and parts[1].isdigit()):
            raise ValueError(
                f"Gate knobs move pulse fields, and {path!r} is not one. Rebind chip parameters with "
                f"GateSet.with_chip(). Pulse paths: {available}."
            )
        if path not in sequence.parameters:
            raise KeyError(f"Unknown sequence parameter path {path!r}. Pulse paths: {available}.")
        return path
    raise TypeError(f"A knob ref is a PulseHandle, a path, or a tie list; got {type(ref).__name__}.")


def _default_scale_and_lower(sequence: "QuantumSequence", path: str, value: float) -> tuple[float, float]:
    """Return the default fit scale and lower bound of one pulse field."""
    _, index, field = path.split(".", 2)
    if field in ("freq", "detuning"):
        return 1e-3, -np.inf
    if field == "phase":
        return 0.1, -np.inf
    if field == "start_time":
        return max(abs(value), 0.1), 0.0
    envelope = sequence._entries[int(index)].envelope
    declaration = parameter_fields(type(envelope)).get(field.removeprefix("envelope."))
    floor = 0.1 if declaration is not None and declaration.unit == "ns" else 1e-3
    lower = 0.0 if declaration is not None and (declaration.positive or declaration.nonnegative) else -np.inf
    return max(abs(value), floor), lower


def resolve_knob(name: str, knob: Knob, sequence: "QuantumSequence") -> ResolvedKnob:
    """Resolve a :class:`Knob` against a template sequence."""
    if isinstance(knob.ref, (list, tuple)) and not isinstance(knob.ref, str):
        if knob.field is not None:
            raise ValueError(f"Knob {name!r}: a tie gives each field in its own (ref, factor) pair.")
        pairs = []
        for item in knob.ref:
            target, factor = item
            if isinstance(target, tuple):
                target_ref, target_field = target
            else:
                target_ref, target_field = target, None
            pairs.append((_target_path(target_ref, target_field, sequence), float(factor)))
        if not pairs:
            raise ValueError(f"Knob {name!r} ties no fields.")
    else:
        pairs = [(_target_path(knob.ref, knob.field, sequence), 1.0)]
    if any(factor == 0.0 for _, factor in pairs):
        raise ValueError(f"Knob {name!r} has a zero tie factor.")
    parameters = sequence.parameters
    template = tuple(0.0 if parameters[path] is None else float(parameters[path]) for path, _ in pairs)
    initial = template[0] / pairs[0][1]
    scale, lower = _default_scale_and_lower(sequence, pairs[0][0], initial)
    upper = np.inf
    if pairs[0][1] < 0:
        lower = -np.inf
    if knob.bounds is not None:
        lower = -np.inf if knob.bounds[0] is None else float(knob.bounds[0])
        upper = np.inf if knob.bounds[1] is None else float(knob.bounds[1])
    if knob.scale is not None:
        scale = float(knob.scale)
    if not scale > 0:
        raise ValueError(f"Knob {name!r} needs a positive scale.")
    if not lower <= initial <= upper:
        raise ValueError(f"Knob {name!r} starts at {initial:g}, outside its bounds [{lower:g}, {upper:g}].")
    return ResolvedKnob(name, tuple(pairs), template, initial, scale, lower, upper, bool(knob.fixed))


def bind_pulses(sequence: "QuantumSequence", bindings: Mapping[str, Any]) -> "QuantumSequence":
    """Return a copy of *sequence* on the same chip with pulse fields rebound.

    :meth:`QuantumSequence.with_params` clones the chip. Gate templates keep
    the gate set's chip, so their copies compose with ``extend()``.
    """
    from quchip.control.sequence import _PulseEntry

    bound = sequence._copy_with_chip(sequence._chip)
    grouped: dict[int, dict[str, Any]] = {}
    for path, value in bindings.items():
        _, index, field = path.split(".", 2)
        grouped.setdefault(int(index), {})[field] = value
    for index, updates in grouped.items():
        entry = bound._entries[index]
        if not isinstance(entry, _PulseEntry):
            raise KeyError(f"pulse.{index} is not a pulse entry.")
        envelope_updates = {
            field.removeprefix("envelope."): value for field, value in updates.items() if field not in _PULSE_FIELDS
        }
        entry_updates: dict[str, Any] = {
            "requested_start_time" if field == "start_time" else field: value
            for field, value in updates.items()
            if field in _PULSE_FIELDS
        }
        if envelope_updates:
            entry_updates["envelope"] = entry.envelope.with_params(envelope_updates)
        bound._entries[index] = replace(entry, **entry_updates)
    return bound


@dataclass(frozen=True)
class Seed:
    """A starting point of a calibration fit.

    Attributes
    ----------
    values : mapping of str to float
        Knob values. Knobs that are not listed start at their template value.
    provenance : str
        Where the values come from, for the report.
    """

    values: Mapping[str, float]
    provenance: str = "user"


class NativeGate:
    """A calibrated gate: one application's pulses, the knobs, and the ideal unitary.

    Parameters
    ----------
    name : str
        Gate name, such as ``"SX"`` or ``"ZX90"``.
    qubits : sequence of str or device
        Ordered gate qubits. A two-qubit gate lists the control first.
    sequence : QuantumSequence
        Pulses of one application on the gate set's chip, from ``t = 0``.
        Internal ``vz`` entries are allowed.
    knobs : mapping of str to Knob, optional
        Calibration parameters.
    ideal : str or array_like, optional
        Ideal unitary on the gate qubits, or a registered name. The default
        is ``name``.
    free_z : sequence of str or device, default ()
        Qubits whose post-gate Z phases are calibrated. Each phase ``φ_q``
        becomes a trailing ``vz(q, -φ_q)``.
    seeds : sequence of Seed or mapping, default ()
        Extra fit starts.
    explore : bool, default False
        Run a Nelder-Mead search before the least-squares fits.
    duration : float or None, default None
        Gate duration in ns. The default is the sequence's total duration.
    """

    def __init__(
        self,
        name: str,
        qubits: Sequence[Any],
        sequence: "QuantumSequence | None",
        *,
        knobs: Mapping[str, Knob] | None = None,
        ideal: Any = None,
        free_z: Sequence[Any] = (),
        seeds: Sequence[Seed | Mapping[str, float]] = (),
        explore: bool = False,
        duration: float | None = None,
    ) -> None:
        self.name = str(name)
        self.qubits = tuple(resolve_label(qubit) for qubit in qubits)
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError(f"Gate qubits must be distinct, got {list(self.qubits)}.")
        self.sequence = sequence
        self.knobs = dict(knobs or {})
        self.ideal = ideal if ideal is not None else self.name
        self.free_z = tuple(resolve_label(qubit) for qubit in free_z)
        outside = [label for label in self.free_z if label not in self.qubits]
        if outside:
            raise ValueError(f"free_z qubits {outside} are not gate qubits {list(self.qubits)}.")
        self.seeds = tuple(seed if isinstance(seed, Seed) else Seed(dict(seed), f"seed {i}")
                           for i, seed in enumerate(seeds))
        self.explore = bool(explore)
        self._duration = duration
        self._build: Callable[[Any], tuple[Any, Mapping[str, Knob]]] | None = None
        self.ideal_matrix()

    @classmethod
    def composite(
        cls,
        name: str,
        qubits: Sequence[Any],
        build: Callable[[Any], tuple["QuantumSequence", Mapping[str, Knob]]],
        *,
        ideal: Any = None,
        free_z: Sequence[Any] = (),
        seeds: Sequence[Seed | Mapping[str, float]] = (),
        explore: bool = False,
    ) -> "NativeGate":
        """Create a native whose pulses include other calibrated natives.

        ``build(gates)`` returns the sequence and its knobs. It inserts
        calibrated natives with ``seq.extend(gates.schedule([...]))``. The gate
        set calibrates those natives first, and their pulses stay fixed while
        this gate is calibrated.

        Parameters
        ----------
        name, qubits, ideal, free_z, seeds, explore
            As for :class:`NativeGate`.
        build : callable
            Function of the gate set that returns ``(sequence, knobs)``.
        """
        gate = cls(name, qubits, None, ideal=ideal, free_z=free_z, seeds=seeds, explore=explore)
        gate._build = build
        return gate

    @property
    def is_composite(self) -> bool:
        """True when the gate is built from other natives."""
        return self._build is not None

    @property
    def key(self) -> tuple[str, tuple[str, ...]]:
        """``(name, qubits)`` key in the gate set."""
        return (canonical_name(self.name), self.qubits)

    @property
    def duration(self) -> float:
        """Gate duration in ns."""
        if self._duration is not None:
            return float(self._duration)
        if self.sequence is None:
            raise ValueError(f"Composite native {self.name} has no sequence before it is built.")
        return float(self.sequence.total_duration)

    def ideal_matrix(self) -> np.ndarray:
        """Return the ideal unitary on the gate qubits."""
        matrix = ideal_unitary(self.ideal)
        if matrix.shape[0] != 2 ** len(self.qubits):
            raise ValueError(
                f"The ideal of {self.name} acts on {int(np.log2(matrix.shape[0]))} qubits; the gate has "
                f"{len(self.qubits)}."
            )
        return matrix

    def resolved_knobs(self) -> list[ResolvedKnob]:
        """Resolve the knobs against the template sequence."""
        if self.sequence is None:
            raise ValueError(f"Composite native {self.name} has no sequence before it is built.")
        return [resolve_knob(name, knob, self.sequence) for name, knob in self.knobs.items()]

    def copy(self) -> "NativeGate":
        """Return a copy that shares the template sequence."""
        return copy.copy(self)

    def __repr__(self) -> str:
        kind = "composite " if self.is_composite else ""
        return f"NativeGate({kind}{self.name!r}, qubits={self.qubits}, knobs={list(self.knobs)})"
