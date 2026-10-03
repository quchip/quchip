"""Resolve integration frames and observable-demodulation frequencies.

Supported :data:`FrameSpec` values:

* ``"lab"``: zero integration frequencies.
* ``"rotating"``: each device's ``reference_freq``, defaulting to its dressed
  drive frequency. An off-transition reference leaves idle Ramsey detuning.
* ``"auto"``: :func:`plan_frame` ranks constraints from retained coupling bands,
  cascade couplings, and scheduled tones by integrated strength. Consistent
  constraints set the frame; the rest remain time dependent and are recorded
  as residuals. Unconstrained frequencies use device references in declaration
  order.
* A scalar: one common frequency for all devices.
* A device-or-label mapping: per-device frequencies, with missing entries zero.

Assembly subtracts ``Σᵢ ω_frame,ᵢ nᵢ`` from the Hamiltonian and decides which
coupling bands become static. Stored states stay in the integration frame.
Transverse observables are demodulated by ``ω_ref - ω_frame`` into the readout
reference frame; this phase vanishes when the two frames coincide.

See Gambetta et al., *PRA* 74, 042318 (2006), and Krantz et al., *Appl. Phys.
Rev.* 6, 021318 (2019), for rotating-frame descriptions of driven qubits.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import jax
import numpy as np

from quchip.approximations import require_approximation
from quchip.control.signal import AnalyticSignal
from quchip.engine.ir import (
    CoherentOp,
    ResolvedFrame,
    _is_scalar_like,
    decompose_carrier_bands,
    evaluate_signal_program,
)
from quchip.utils.constants import TWO_PI
from quchip.utils.jax_utils import contains_tracer, maybe_concrete_scalar
from quchip.utils.labeling import resolve_label


def same_frequency(first: Any, second: Any) -> bool:
    """Return whether two frequencies are provably identical without concretizing tracers.

    Identical objects compare equal. Distinct objects compare equal only when
    both values are concrete and exactly equal; no tolerance is applied.
    """
    if first is second:
        return True
    first_value = maybe_concrete_scalar(first)
    second_value = maybe_concrete_scalar(second)
    return first_value is not None and second_value is not None and first_value == second_value

if TYPE_CHECKING:
    from quchip.chip.chip import Chip
    from quchip.engine.ir import ControlOp, FrameSpec

@dataclass(frozen=True)
class FrameTone:
    """Describe a tone constraint ``Σ_d k_d ω_d = freq`` and its weight.

    ``coefficients`` pairs each device label with its excitation-number change
    ``k_d``. ``weight`` is ``|h_band|² ∫|envelope|² dt`` when available.

    Attributes
    ----------
    coefficients : tuple of (str, int)
        Device labels and excitation-number changes.
    freq : scalar
        Constraint frequency in GHz.
    weight : float or None
        Integrated constraint weight, when available.
    source : str
        Description of the originating term or tone.
    """

    coefficients: tuple[tuple[str, int], ...]
    freq: Any
    weight: float | None
    source: str


@dataclass(frozen=True)
class FrameResidual:
    """Describe a rejected constraint that remains time dependent.

    ``frequency`` is its residual oscillation frequency, ``Σ_d k_d ω_d`` for
    a static coupling or ``Σ_d k_d ω_d - freq`` for a tone. ``source`` and
    ``devices`` identify its origin, and ``weight`` records its
    integrated strength when available.

    Attributes
    ----------
    source : str
        Description of the rejected constraint.
    devices : tuple of str
        Devices participating in the constraint.
    frequency : scalar
        Residual frequency in GHz.
    weight : float or None
        Integrated constraint weight, when available.
    """

    source: str
    devices: tuple[str, ...]
    frequency: Any
    weight: float | None


@dataclass(frozen=True)
class FramePlan:
    """Record the frequencies and constraints chosen by :func:`plan_frame`.

    ``frequencies`` maps every device to its frame frequency. ``clusters``
    groups devices linked by accepted constraints, ``pins`` records the
    reference frequencies used to determine free variables, and ``residuals``
    lists rejected constraints that remain time dependent. ``tones`` contains
    the accepted tone constraints that fixed driven clusters, including each
    tone's source, charge coefficients, frequency, and weight.

    Attributes
    ----------
    frequencies : mapping
        Frame frequency in GHz for each device label.
    clusters : tuple of tuple of str
        Device groups linked by accepted constraints.
    pins : tuple of (str, scalar)
        Reference frequencies in GHz used for free variables.
    residuals : tuple of FrameResidual
        Rejected constraints that remain time dependent.
    tones : tuple of FrameTone
        Accepted driven constraints.
    """

    frequencies: Mapping[str, Any]
    clusters: tuple[tuple[str, ...], ...]
    pins: tuple[tuple[str, Any], ...]
    residuals: tuple[FrameResidual, ...]
    tones: tuple[FrameTone, ...] = ()

    def concrete_key(self) -> tuple[Any, ...]:
        """Return a hashable key for concrete frame content.

        Raises
        ------
        ValueError
            If a frame, pin, residual, or tone frequency is traced.
        """

        def scalar(value: Any) -> float:
            concrete = maybe_concrete_scalar(value)
            if concrete is None:
                raise ValueError("Frame plan contains traced values.")
            return concrete

        return (
            tuple((label, scalar(freq)) for label, freq in self.frequencies.items()),
            self.clusters,
            tuple((label, scalar(freq)) for label, freq in self.pins),
            tuple((r.source, r.devices, scalar(r.frequency), r.weight) for r in self.residuals),
            tuple((t.source, t.coefficients, scalar(t.freq), t.weight) for t in self.tones),
        )


class FrameConflict(ValueError):
    """Report a mandatory tone that conflicts with earlier strict constraints.

    ``kind`` is ``"port"`` when an accepted tone already addresses the same
    devices and ``"exchange"`` otherwise. ``cluster`` lists the devices linked
    by accepted constraints.
    """

    def __init__(self, source: str, devices: tuple[str, ...], *, kind: str, cluster: tuple[str, ...]) -> None:
        self.source = source
        self.devices = devices
        self.kind = kind
        self.cluster = cluster
        super().__init__(f"Tone {source!r} cannot be made static with the other tones on {list(cluster)}.")


# ── Symbolic affine values ───────────────────────────────────────────


class _Leaves:
    """Intern traced frequency values by identity so structural comparisons stay JAX-safe."""

    def __init__(self) -> None:
        self.objects: dict[int, Any] = {}

    def affine(self, value: Any) -> "_Affine":
        concrete = maybe_concrete_scalar(value)
        if concrete is not None:
            return _Affine(float(concrete), {})
        self.objects.setdefault(id(value), value)
        return _Affine(0.0, {id(value): Fraction(1)})

    def evaluate(self, affine: "_Affine") -> Any:
        """Rebuild a value, returning a lone traced leaf itself so identity checks still hold."""
        value: Any = None if affine.terms and affine.constant == 0.0 else affine.constant
        for key, coefficient in affine.terms.items():
            term = self.objects[key] if coefficient == 1 else float(coefficient) * self.objects[key]
            value = term if value is None else value + term
        return value


@dataclass(frozen=True)
class _Affine:
    constant: float
    terms: Mapping[int, Fraction]

    def scaled(self, factor: Fraction) -> "_Affine":
        return _Affine(self.constant * float(factor), {key: value * factor for key, value in self.terms.items()})

    def minus(self, other: "_Affine") -> "_Affine":
        terms = dict(self.terms)
        for key, value in other.terms.items():
            terms[key] = terms.get(key, Fraction(0)) - value
        return _Affine(self.constant - other.constant, {key: value for key, value in terms.items() if value != 0})

    @property
    def is_zero(self) -> bool:
        return not self.terms and same_frequency(self.constant, 0.0)


@dataclass(frozen=True)
class _Row:
    coefficients: Mapping[str, Fraction]
    rhs: _Affine
    weight: float | None
    source: str
    devices: tuple[str, ...]
    order: int
    is_tone: bool


class _System:
    """Incremental exact elimination over device-frequency unknowns."""

    def __init__(self) -> None:
        self.pivots: dict[str, tuple[dict[str, Fraction], _Affine]] = {}

    def reduce(self, row: _Row) -> tuple[dict[str, Fraction], _Affine]:
        coefficients = dict(row.coefficients)
        rhs = row.rhs
        for variable, (pivot_row, pivot_rhs) in self.pivots.items():
            factor = coefficients.pop(variable, Fraction(0))
            if not factor:
                continue
            for other, value in pivot_row.items():
                coefficients[other] = coefficients.get(other, Fraction(0)) - factor * value
            rhs = rhs.minus(pivot_rhs.scaled(factor))
        return {key: value for key, value in coefficients.items() if value != 0}, rhs

    def consistent(self, row: _Row) -> bool:
        coefficients, rhs = self.reduce(row)
        return bool(coefficients) or rhs.is_zero

    def add(self, row: _Row) -> bool:
        """Add ``row``; return whether it raised the rank."""
        coefficients, rhs = self.reduce(row)
        if not coefficients:
            return False
        variable = sorted(coefficients)[0]
        lead = coefficients.pop(variable)
        normalized = {key: value / lead for key, value in coefficients.items()}
        rhs = rhs.scaled(1 / lead)
        for other, (pivot_row, pivot_rhs) in list(self.pivots.items()):
            factor = pivot_row.get(variable)
            if not factor:
                continue
            updated = dict(pivot_row)
            del updated[variable]
            for key, value in normalized.items():
                updated[key] = updated.get(key, Fraction(0)) - factor * value
            self.pivots[other] = ({k: v for k, v in updated.items() if v != 0}, pivot_rhs.minus(rhs.scaled(factor)))
        self.pivots[variable] = (normalized, rhs)
        return True

    def copy(self) -> "_System":
        clone = _System()
        clone.pivots = {key: (dict(row), rhs) for key, (row, rhs) in self.pivots.items()}
        return clone

    def solve(self, variable: str) -> _Affine:
        pivot = self.pivots[variable]
        if pivot[0]:
            raise ValueError(f"Frame frequency of {variable!r} is not determined.")
        return pivot[1]


# ── Planning ────────────────────────────────────────────────────────


def _compile_time(function: Any) -> Any:
    """Run frame analysis in JAX's compile-time evaluation context.

    The wrapped function receives the chip first and also enters its backend's
    eager-operator context. Constant operator amplitudes stay concrete under
    ``jit``; values that depend on traced inputs stay traced.
    """

    @functools.wraps(function)
    def wrapper(chip: Any, *args: Any, **kwargs: Any) -> Any:
        with jax.ensure_compile_time_eval(), chip.backend.eager_operators():
            return function(chip, *args, **kwargs)

    return wrapper


def resolve_reference_frequencies(chip: Any, *, local_resolution: Any = None) -> dict[str, Any]:
    """Resolve explicit overrides or dressed references in one chip context."""
    dressed = (
        chip.analysis._dressed_frequencies(chip.analysis.engine_result(_local_resolution=local_resolution))
        if any(device.reference_freq is None for device in chip.devices)
        else {}
    )
    return {
        device.label: dressed[device.label] if device.reference_freq is None else device.reference_freq
        for device in chip.devices
    }


@_compile_time
def plan_frame(
    chip: "Chip",
    tones: Sequence[FrameTone],
    *,
    approximation: Any,
    strict: bool,
    solve_duration: float | None = None,
    reference_frequencies: Mapping[str, Any] | None = None,
    local_resolution: Any | None = None,
) -> FramePlan:
    """Choose one frame frequency per device from charge-vector constraints.

    A retained static coupling band imposes ``Σ_d k_d ω_d = 0``. Each entry in
    ``tones`` imposes ``Σ_d k_d ω_d = freq``. Cascade-generated network
    couplings add equality constraints between their device frequencies. The
    non-strict planner keeps the consistent subset with the largest total
    integrated strength; declaration order breaks ties. Rejected constraints
    become residual oscillations. Free variables are pinned to device reference
    frequencies in declaration order.

    Parameters
    ----------
    chip : Chip
        The chip whose devices receive frame frequencies.
    tones : sequence of FrameTone
        Charge-vector constraints requested by scheduled or stationary tones.
    approximation : Approximation
        Decides which coupling bands are retained.
    strict : bool
        Apply stationary ordering when true. Require cascade constraints,
        exchange coupling bands with nonzero charge vectors summing to zero,
        and tones, in that order. Raise :class:`FrameConflict` when a tone
        conflicts; only non-exchange coupling bands may remain as residuals.
    solve_duration : float, optional
        Integration-window duration used for the static-coupling weight
        ``|h_band|² solve_duration``. If omitted or traced, static-coupling
        weights remain unknown.
    reference_frequencies : mapping, optional
        Per-device reference frequencies for pins; defaults to the devices'.
    local_resolution : optional
        Resolved local bases from assembly, reused when already computed.

    Returns
    -------
    FramePlan
        Frequencies, clusters, pins, and residuals.

    Raises
    ------
    FrameConflict
        In strict mode, when a tone cannot be made static.
    """
    from quchip.engine.assembly import _resolve_system, coupling_band_records

    resolution = local_resolution if local_resolution is not None else _resolve_system(chip, chip.backend)
    references = (
        resolve_reference_frequencies(chip, local_resolution=resolution)
        if reference_frequencies is None
        else dict(reference_frequencies)
    )
    leaves = _Leaves()
    duration = None if solve_duration is None else maybe_concrete_scalar(solve_duration)
    zero = _Affine(0.0, {})

    hard: list[_Row] = []
    soft: list[_Row] = []
    network = chip.port_network
    for group in () if network is None else network.dynamical_supports(chip, _compiled=resolution.network):
        for first, second in zip(group, group[1:], strict=False):
            hard.append(_Row({first: Fraction(1), second: Fraction(-1)}, zero, None, "cascade", group, -1, False))
    for record in coupling_band_records(chip, resolution, chip.backend):
        if (not record.retained and not approximation.keeps_operator_band(record.charges)) or not any(record.charges):
            continue
        coefficients = _normalized(zip(record.devices, record.charges, strict=True))
        weight = None if record.amplitude is None or duration is None else record.amplitude**2 * duration
        row = _Row(coefficients, zero, weight, record.source, record.devices, len(soft), False)
        (hard if strict and sum(record.charges) == 0 else soft).append(row)
    for tone in tones:
        coefficients = _normalized(tone.coefficients)
        if not coefficients:
            continue
        devices = tuple(dict.fromkeys(label for label, _ in tone.coefficients))
        row = _Row(coefficients, leaves.affine(tone.freq), tone.weight, tone.source, devices, len(soft), True)
        (hard if strict else soft).append(row)

    system = _System()
    accepted: list[_Row] = []
    for row in hard:
        if system.consistent(row):
            system.add(row)
            accepted.append(row)
            continue
        if not row.is_tone:
            continue
        same_target = any(other.is_tone and set(other.devices) == set(row.devices) for other in accepted)
        cluster = next(
            (group for group in _clusters(chip, accepted) if set(row.devices) & set(group)),
            row.devices,
        )
        raise FrameConflict(row.source, row.devices, kind="port" if same_target else "exchange", cluster=cluster)

    candidates = _aggregate(soft)
    selected = _select(system, candidates)
    rejected = [row for index, row in enumerate(candidates) if index not in selected]
    for index in sorted(selected):
        system.add(candidates[index])
        accepted.append(candidates[index])

    pins: list[tuple[str, Any]] = []
    for device in chip.devices:
        reference = references[device.label]
        pin = _Row({device.label: Fraction(1)}, leaves.affine(reference), None, "reference", (device.label,), -1, False)
        if system.add(pin):
            pins.append((device.label, reference))

    frequencies = {device.label: leaves.evaluate(system.solve(device.label)) for device in chip.devices}
    residuals = tuple(
        FrameResidual(row.source, row.devices, leaves.evaluate(_residual(row, system)), row.weight) for row in rejected
    )
    static_tones = tuple(
        FrameTone(
            tuple((label, int(value)) for label, value in row.coefficients.items()),
            leaves.evaluate(row.rhs),
            row.weight,
            row.source,
        )
        for row in accepted
        if row.is_tone
    )
    return FramePlan(
        frequencies=frequencies,
        clusters=_clusters(chip, accepted),
        pins=tuple(pins),
        residuals=residuals,
        tones=static_tones,
    )


def _normalized(coefficients: Any) -> dict[str, Fraction]:
    """Merge duplicate labels and flip signs so the first nonzero coefficient is positive.

    A band and its Hermitian conjugate carry opposite charge vectors but are
    made static by the same frame, so both reduce to one constraint.
    """
    merged: dict[str, Fraction] = {}
    for label, charge in coefficients:
        merged[label] = merged.get(label, Fraction(0)) + Fraction(int(charge))
    merged = {label: value for label, value in merged.items() if value != 0}
    if not merged:
        return {}
    sign = 1 if next(iter(merged.values())) > 0 else -1
    return {label: value * sign for label, value in merged.items()}


def _aggregate(rows: list[_Row]) -> list[_Row]:
    """Merge identical constraints and order them by weight, then declaration.

    If any contribution has an unknown weight, the merged weight remains
    unknown, and equal or unknown weights fall back to declaration order.
    """
    merged: dict[tuple[Any, ...], _Row] = {}
    for row in rows:
        key = (tuple(sorted(row.coefficients.items())), row.rhs.constant, tuple(sorted(row.rhs.terms.items())))
        previous = merged.get(key)
        if previous is None:
            merged[key] = row
            continue
        weight = None if previous.weight is None or row.weight is None else previous.weight + row.weight
        merged[key] = _Row(
            previous.coefficients,
            previous.rhs,
            weight,
            "; ".join(dict.fromkeys((*previous.source.split("; "), row.source))),
            previous.devices,
            previous.order,
            previous.is_tone,
        )
    return sorted(merged.values(), key=lambda row: (-(row.weight or 0.0), row.order))


def _select(system: _System, candidates: list[_Row]) -> set[int]:
    """Return the indices of the consistent subset of maximum total weight, ties to earlier rows."""
    weights = [row.weight or 0.0 for row in candidates]
    suffix = [0.0] * (len(candidates) + 1)
    for index in range(len(candidates) - 1, -1, -1):
        suffix[index] = suffix[index + 1] + weights[index]
    best: tuple[float, list[int]] = (-1.0, [])

    def search(index: int, state: _System, chosen: list[int], total: float) -> None:
        nonlocal best
        if total + suffix[index] <= best[0]:
            return
        if index == len(candidates):
            best = (total, list(chosen))
            return
        row = candidates[index]
        if state.consistent(row):
            branch = state.copy()
            branch.add(row)
            chosen.append(index)
            search(index + 1, branch, chosen, total + weights[index])
            chosen.pop()
        search(index + 1, state, chosen, total)

    search(0, system, [], 0.0)
    return set(best[1])


def _residual(row: _Row, system: _System) -> _Affine:
    """Return ``Σ k_d ω_d − rhs`` for a rejected row in the fully determined system."""
    return _Affine(0.0, {}).minus(system.reduce(row)[1])


def _clusters(chip: "Chip", accepted: list[_Row]) -> tuple[tuple[str, ...], ...]:
    parent = {device.label: device.label for device in chip.devices}

    def find(label: str) -> str:
        while parent[label] != label:
            label = parent[label]
        return label

    for row in accepted:
        labels = list(row.coefficients)
        for other in labels[1:]:
            parent[find(other)] = find(labels[0])
    groups: dict[str, list[str]] = {}
    for device in chip.devices:
        groups.setdefault(find(device.label), []).append(device.label)
    return tuple(tuple(group) for group in groups.values())


# ── Tones from scheduled operations ────────────────────────────────


@_compile_time
def frame_tones(
    chip: "Chip",
    operations: Sequence["ControlOp"],
    *,
    approximation: Any,
    resolution: Any,
    solve_window: tuple[float, float] | None = None,
) -> list[FrameTone]:
    """Build weighted frame constraints from scheduled drive and coherent-input signals.

    Drive signals are read after control equipment applies gain, attenuation,
    delay, and crosstalk. Each nonzero carrier band constrains every retained
    operator band of its destination drive; a zero-frequency band adds no
    constraint. A crosstalk destination therefore receives its own tone, and
    gain changes the tone's weight. Coherent inputs use their complete
    scheduled signal, including its pulse window, start time, phase, and
    carrier, while network scattering scales the port amplitude. Both paths
    use the same carrier decomposition. Each signal's energy is integrated over
    its own support intersected with ``solve_window``; when omitted, that outer
    window is ``[0, last pulse end]``. Traced values leave the corresponding
    weight unknown.
    """
    from quchip.engine.assembly import _build_delivered_signals, drive_bands

    span = _window(operations, solve_window)
    tones: list[FrameTone] = []
    for delivered in _build_delivered_signals(chip, list(operations)).values():
        drive, target = delivered.drive, delivered.target
        bands = drive_bands(chip, drive, target, resolution.bases, resolution.dims, chip.backend, approximation)
        records = [record for record, _, _, _ in bands]
        source = f"{drive.label} → {target.label}" + (" (crosstalk)" if delivered.origin == "crosstalk" else "")
        tones += _signal_tones(delivered.signal.program, records, span, source)
    for operation in operations:
        if isinstance(operation, CoherentOp):
            records = _fed_port_records(chip, operation.exposure, resolution)
            source = f"{operation.drive_label} → {operation.exposure}"
            tones += _signal_tones(AnalyticSignal.from_pulse(operation).program, records, span, source, signed=True)
    return tones


def _signal_tones(
    program: Any, records: Sequence[Any], span: tuple[float, float] | None, source: str, *, signed: bool = False,
) -> list[FrameTone]:
    """One tone per nonzero carrier band of a scheduled signal, weighted by its windowed envelope energy."""
    tones: list[FrameTone] = []
    for band in decompose_carrier_bands(program):
        if maybe_concrete_scalar(band.freq) == 0.0:
            continue
        energy = _program_energy(band.envelope, span)
        tones += _tones_from_records(records, -band.freq / TWO_PI, energy, source, signed=signed)
    return tones


@_compile_time
def stationary_tones(chip: "Chip", port_frequencies: Sequence[tuple[str, Any]], *, resolution: Any) -> list[FrameTone]:
    """Build unweighted frame constraints for stationary port tones.

    Strict stationary planning uses these constraints to reject tone sets that
    no single time-independent frame can satisfy.
    """
    tones: list[FrameTone] = []
    for exposure, freq in port_frequencies:
        tones += _tones_from_records(_fed_port_records(chip, exposure, resolution), freq, None, exposure,
                                     signed=True)
    return tones


def _fed_port_records(chip: "Chip", exposure: str, resolution: Any) -> list[Any]:
    """Return reached port bands with scattering-scaled amplitudes in GHz.

    For a coherent field ``β`` entering ``exposure``, each reached channel has
    ``c = S β`` and drives ``i(c* L − c L†)``. Each returned amplitude is
    ``|S| |L_band| / 2π``; integrating ``|β|²`` gives the same weight as the
    instantaneous ordinary-GHz amplitude ``|c| |L_band| / 2π``.
    """
    from quchip.engine.assembly import port_band_records

    network = chip.port_network
    fed = (
        [(chip.port(exposure), 1.0)] if network is None
        else network.fed_ports(chip, exposure, _compiled=resolution.network)
    )
    records: list[Any] = []
    for port, coefficient in fed:
        gain = None if contains_tracer(coefficient) else float(np.abs(np.asarray(coefficient)))
        for record in port_band_records(chip, port, resolution, chip.backend):
            amplitude = None if gain is None or record.amplitude is None else gain * record.amplitude / TWO_PI
            records.append(replace(record, amplitude=amplitude))
    return records


def _tones_from_records(
    records: Sequence[Any], freq: Any, energy: float | None, source: str, *, signed: bool = False,
) -> list[FrameTone]:
    """One tone per charged band record, fundamental bands declared before multiphoton ones.

    Drive records list each band with its Hermitian partner, so a tone row only
    fixes ``|k·ω| = |freq|``. A port's coupling operator ``L`` has no partner in
    its records: band ``k`` of ``L`` is static only when ``k·ω = freq``. For
    ``signed`` records, a band whose first charge is negative is therefore stated
    as ``(-k)·ω = -freq`` before frame planning normalizes its sign.
    """
    ordered = sorted((record for record in records if any(record.charges)), key=lambda r: sum(map(abs, r.charges)))
    tones = []
    for record in ordered:
        charges = tuple(record.charges)
        tone_freq = freq
        if signed and next(charge for charge in charges if charge) < 0:
            charges, tone_freq = tuple(-charge for charge in charges), -freq
        tones.append(FrameTone(
            tuple(zip(record.devices, charges, strict=True)),
            tone_freq,
            None if energy is None or record.amplitude is None else record.amplitude**2 * energy,
            source,
        ))
    return tones


def _window(operations: Sequence["ControlOp"], solve_window: Any) -> tuple[float, float] | None:
    """Return the concrete outer window used to integrate frame weights.

    Use ``(tlist[0], tlist[-1])`` when callers supply ``solve_window``;
    otherwise use ``(0, last pulse end)``. Return ``None`` when a required
    bound is traced or no operation supplies an end time.
    """
    if solve_window is not None:
        start, stop = (maybe_concrete_scalar(bound) for bound in solve_window)
        return None if start is None or stop is None else (start, stop)
    ends: list[float] = []
    for op in operations:
        end = maybe_concrete_scalar(op.start_time + op.envelope.duration)
        if end is None:
            return None
        ends.append(end)
    return (0.0, max(ends)) if ends else None


def _support(program: Any) -> tuple[float, float] | None:
    """Return a concrete interval outside which ``program`` is zero.

    Read bounds from ``Window`` and ``Shift`` nodes, intersect child supports
    for ``Multiply``, and take their union for ``Add``. Return ``None`` when no
    finite support can be established.
    """
    from quchip.engine.ir import Add, Multiply, Shift, Window, signal_children

    if isinstance(program, Window):
        bounds = _support(program.child)
        start, stop = maybe_concrete_scalar(program.start), maybe_concrete_scalar(program.stop)
        if start is None or stop is None:
            return bounds
        return (start, stop) if bounds is None else (max(start, bounds[0]), min(stop, bounds[1]))
    if isinstance(program, Shift):
        bounds = _support(program.child)
        shift = maybe_concrete_scalar(program.delta_t)
        return None if bounds is None or shift is None else (bounds[0] + shift, bounds[1] + shift)
    if isinstance(program, (Multiply, Add)):
        supports = [_support(child) for child in program.children]
        if isinstance(program, Multiply):
            bounded = [item for item in supports if item is not None]
            if not bounded:
                return None
            return (max(item[0] for item in bounded), min(item[1] for item in bounded))
        bounded = [item for item in supports if item is not None]
        if len(bounded) != len(supports):
            return None
        return (min(item[0] for item in bounded), max(item[1] for item in bounded))
    children = signal_children(program)
    return _support(children[0]) if len(children) == 1 else None


def _program_energy(program: Any, span: tuple[float, float] | None) -> float | None:
    """Integrate ``|program(t)|²`` over its support inside ``span``.

    Intersect the known support with ``span`` and sample 1024 points in that
    interval. Return zero for an empty intersection and ``None`` when the span
    or program is traced.
    """
    if span is None or contains_tracer(program):
        return None
    start, stop = span
    support = _support(program)
    if support is not None:
        start, stop = max(start, support[0]), min(stop, support[1])
    if stop <= start:
        return 0.0
    times = np.linspace(start, stop, 1024)
    values = evaluate_signal_program(program, times)
    if contains_tracer(values):
        return None
    return float(np.trapezoid(np.abs(np.asarray(values)) ** 2, times))


@_compile_time
def planning_resolution(chip: "Chip") -> Any:
    """Resolve local solver bases during JAX compile-time evaluation.

    Bases from constant device matrices stay concrete inside ``jit``. Bases
    that depend on traced inputs remain traced.
    """
    from quchip.engine.assembly import _resolve_system

    return _resolve_system(chip, chip.backend)


def resolve_for_operations(
    chip: "Chip",
    operations: Sequence["ControlOp"],
    *,
    frame: Any = None,
    approximation: Any = None,
    solve_window: Any = None,
) -> Any:
    """Resolve ``chip`` with an operation-aware ``"auto"`` frame when requested.

    ``approximation`` selects the retained bands. ``solve_window`` supplies the
    solve's ``(start, stop)`` bounds for signal-energy weights; when omitted,
    the bounds are ``(0, last pulse end)``.
    """
    strategy = chip.approximation if approximation is None else require_approximation(approximation)
    frame_spec = chip.frame if frame is None else frame
    if not (isinstance(frame_spec, str) and frame_spec == "auto"):
        return chip.resolve(frame=frame_spec, approximation=strategy)
    resolution = planning_resolution(chip)
    spec = plan_for_operations(
        chip, frame_spec, operations, approximation=strategy, solve_window=solve_window, _resolution=resolution,
    )
    return chip._resolve(frame=spec, approximation=strategy, _resolution=resolution)


def plan_for_operations(
    chip: "Chip",
    frame_spec: Any,
    operations: Sequence["ControlOp"],
    *,
    approximation: Any,
    solve_window: Any = None,
    _resolution: Any = None,
) -> Any:
    """Build an operation-aware :class:`FramePlan` for ``"auto"``.

    Return any other ``frame_spec`` unchanged. ``approximation`` selects the
    retained bands. ``solve_window`` bounds each signal-energy integral, and
    its length sets the static-coupling weight.
    """
    if not (isinstance(frame_spec, str) and frame_spec == "auto"):
        return frame_spec
    resolution = planning_resolution(chip) if _resolution is None else _resolution
    span = _window(operations, solve_window)
    tones = frame_tones(chip, operations, approximation=approximation, resolution=resolution, solve_window=span)
    duration = None if span is None else span[1] - span[0]
    return plan_frame(
        chip, tones, approximation=approximation, strict=False, solve_duration=duration, local_resolution=resolution
    )


# ── Resolution ──────────────────────────────────────────────────────


def resolve_frame(
    chip: Chip,
    frame_spec: FrameSpec,
    *,
    reference_frequencies: Mapping[str, Any] | None = None,
    local_resolution: Any | None = None,
    approximation: Any | None = None,
) -> ResolvedFrame:
    """Resolve *frame_spec* into a :class:`ResolvedFrame`.

    Dispatches on ``frame_spec`` shape (``str`` / scalar-like / dict / plan),
    fills a per-device ``frequencies`` dict, and computes the
    demodulation frequencies ``reference_freq − ω_frame``. See the
    module docstring for the physical meaning of each mode. A bare ``"auto"``
    spec plans from coupling and network constraints only. Operation-aware
    callers pass a :class:`FramePlan` that also contains their tone
    constraints.

    Missing references resolve from this chip's dressed transitions. Explicit
    references bypass that calculation; reading a device setting does no work.
    """
    devices = chip.devices
    labels = [dev.label for dev in devices]
    references = (
        resolve_reference_frequencies(chip, local_resolution=local_resolution)
        if reference_frequencies is None
        else {dev.label: reference_frequencies[dev.label] for dev in devices}
    )

    mode: str
    frequencies: dict[str, Any]
    plan: FramePlan | None = None

    # Dispatch by spec shape. Keep the isinstance(str) check ahead of any
    # equality comparison so no JAX tracer is compared to a string
    # (which would yield a traced bool and force concretization).
    if isinstance(frame_spec, FramePlan):
        mode = "auto"
        plan = frame_spec
        frequencies = dict(plan.frequencies)
    elif isinstance(frame_spec, str):
        if frame_spec == "lab":
            mode = "lab"
            frequencies = {label: 0.0 for label in labels}
        elif frame_spec == "rotating":
            mode = "rotating"
            frequencies = dict(references)
        elif frame_spec == "auto":
            mode = "auto"
            plan = plan_frame(
                chip,
                (),
                approximation=chip.approximation if approximation is None else approximation,
                strict=False,
                reference_frequencies=references,
                local_resolution=local_resolution,
            )
            frequencies = dict(plan.frequencies)
        else:
            raise ValueError(f"Unknown frame string. Expected 'lab', 'rotating', or 'auto', got {frame_spec!r}")
    elif _is_scalar_like(frame_spec):
        mode = "float"
        frequencies = {label: frame_spec for label in labels}
    elif isinstance(frame_spec, dict):
        mode = "dict"
        normalized = {resolve_label(key): value for key, value in frame_spec.items()}
        frequencies = {label: normalized.get(label, 0.0) for label in labels}
    else:
        raise TypeError(
            "frame_spec must be 'lab', 'rotating', 'auto', a scalar-like frequency, or a "
            f"dict[str|BaseDevice, scalar-like], got {type(frame_spec).__name__}"
        )

    demod_freqs = {label: references[label] - frequencies[label] for label in labels}

    return ResolvedFrame(
        frequencies=frequencies,
        demod_freqs=demod_freqs,
        mode=mode,
        plan=plan,
    )
