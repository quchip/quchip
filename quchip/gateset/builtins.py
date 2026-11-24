"""Built-in natives with physics seeds: DRAG SX and cross resonance."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from quchip.gateset.native import Knob, NativeGate, Seed
from quchip.utils.labeling import resolve_label
from quchip.utils.stack import user_stacklevel
from quchip.utils.warning_categories import AccuracyWarning, AutoChoiceWarning

if TYPE_CHECKING:
    from quchip.chip.chip import Chip


def _charge_line(chip: "Chip", device: str, line: Any) -> str:
    """Return the label of a device's only charge line, or the given one."""
    from quchip.control.drive import ChargeDrive

    if line is not None:
        return resolve_label(line)
    equipment = chip.control_equipment
    lines = [] if equipment is None else [
        item.label for item in equipment.lines if item.device_label == device and isinstance(item, ChargeDrive)
    ]
    if len(lines) != 1:
        raise ValueError(f"{device} has charge lines {lines}; pass line= to choose one.")
    return lines[0]


def _unit_area(envelope: Any) -> float:
    """Return ``∫ Re E(t) dt`` of an envelope scaled to unit amplitude, in ns."""
    return float(np.real(envelope.with_params({"amplitude": 1.0}).area()))


def drag_sx(
    chip: "Chip",
    qubit: Any,
    *,
    line: Any = None,
    duration: float = 40.0,
    sigmas: float = 3.0,
    name: str = "SX",
    angle: float = np.pi / 2,
    detuning: bool = False,
) -> NativeGate:
    """Return a single-qubit rotation native with one ``GaussianDRAG`` pulse.

    The pulse runs at ``freq=chip.freq(qubit)`` with ``phase=-arg(m)``, where
    ``m`` is the qubit's dressed ``0 -> 1`` drive matrix element on the line,
    so it rotates about +X. The knobs are the amplitude and, for devices with
    more than two levels, the DRAG coefficient β. The amplitude seed comes
    from :meth:`~quchip.chip.chip.Chip.rotation_amplitude`, and the β seed is
    ``1 / (4π |α|)``, with α the dressed anharmonicity.

    Parameters
    ----------
    chip : Chip
        Chip with the qubit's charge line wired.
    qubit : str or device
        Driven qubit.
    line : str or drive, optional
        Charge line. The default is the qubit's only charge line.
    duration : float, default 40.0
        Pulse duration in ns.
    sigmas : float, default 3.0
        Gaussian half-window in standard deviations. A narrower pulse has more
        spectral weight at a neighbor's frequency. For an X fit with
        ``spectators="none"`` next to a qubit 100 MHz away with g = 3 MHz,
        scored with that qubit in ``|0>``, a 30 ns pulse at ±4σ has
        ``1 - F = 1.5e-4``, mostly population left in the neighbor, and a
        40 ns pulse at ±3σ has ``1.9e-6``.
    name : str, default "SX"
        Gate name.
    angle : float, default π/2
        Rotation angle; π/2 gives SX and π gives X.
    detuning : bool, default False
        Add the pulse detuning as a knob, bounded to ±5 MHz. A detuning can
        lower the error, for example by centering the carrier between the
        frequencies that a spectator's ZZ shift gives. Near β's optimum,
        however, a detuning and a change of β have almost the same effect,
        and a fit with both knobs can move far along that direction for a
        small gain. The default keeps the fit to the amplitude and β.
    """
    from quchip.control.envelopes import GaussianDRAG
    from quchip.control.sequence import QuantumSequence

    label = resolve_label(qubit)
    line_label = _charge_line(chip, label, line)
    element = complex(chip.drive_matrix_elements(label)[line_label])
    levels = int(chip.device_map[label].levels) if hasattr(chip.device_map[label], "levels") else 2
    beta = 0.0
    if levels > 2:
        beta = 1.0 / (4 * np.pi * abs(float(chip.dressed_anharmonicity(label))))
    envelope = GaussianDRAG(duration=duration, sigmas=sigmas, amplitude=1.0, beta=0.0)
    amplitude = float(chip.rotation_amplitude(label, line_label, envelope, angle))
    sequence = QuantumSequence(chip)
    pulse = sequence.schedule(
        line_label,
        envelope=GaussianDRAG(duration=duration, sigmas=sigmas, amplitude=amplitude, beta=beta),
        freq=chip.freq(label),
        phase=-float(np.angle(element)),
        detuning=0.0,
    )
    knobs = {"amplitude": Knob(pulse, "amplitude")}
    if levels > 2:
        knobs["beta"] = Knob(pulse, "beta", scale=0.1)
    if detuning:
        knobs["detuning"] = Knob(pulse, "detuning", bounds=(-0.005, 0.005))
    ideal: Any
    if np.isclose(angle, np.pi / 2):
        ideal = "SX"
    elif np.isclose(angle, np.pi):
        ideal = "X"
    else:
        ideal = np.cos(angle / 2) * np.eye(2) - 1j * np.sin(angle / 2) * np.array([[0, 1], [1, 0]])
    gate = NativeGate(name, (label,), sequence, knobs=knobs, ideal=ideal, free_z=(label,))
    gate._declare = lambda changed: drag_sx(changed, label, line=line_label, duration=duration, sigmas=sigmas,
                                             name=name, angle=angle, detuning=detuning)
    return gate


def _conditional_elements(chip: "Chip", control: str, target: str, line_label: str) -> tuple[complex, complex]:
    """Return a line's dressed target ``0 -> 1`` elements with the control in 0 and in 1."""
    equipment = chip.control_equipment
    assert equipment is not None
    line = next(line for line in equipment.lines if line.label == line_label)
    control_device, target_device = chip.device_map[control], chip.device_map[target]
    t0 = chip.drive_matrix_elements(({}, {target_device: 1}), drives=[line])[line]
    t1 = chip.drive_matrix_elements(({control_device: 1}, {control_device: 1, target_device: 1}), drives=[line])[line]
    return complex(t0), complex(t1)


def _rotary_turns(envelope: Any, most: float) -> list[float]:
    """Return the turns per tone, up to ``most``, at which a static ``Z_t Z_s`` averages out at first order.

    A target tone ``(N/A) E(t)`` turns the target by ``θ(t) = 2π (N/A) ∫_0^t E``, with ``A`` the envelope area.
    Over one symmetric tone, the first-order part of a static ``Z_t Z_s`` is proportional to
    ``∫ cos(θ(t) - θ(T)/2) dt``. Its zeros are whole turns for a square envelope.
    """
    from scipy.integrate import cumulative_trapezoid, trapezoid
    from scipy.optimize import brentq

    times = np.linspace(0.0, float(envelope.duration), 4001)
    shape = np.real(np.asarray(envelope.with_params({"amplitude": 1.0}).sample(times)))
    running = cumulative_trapezoid(shape, times, initial=0.0)
    centred = running / running[-1] - 0.5

    def integral(turns: float) -> float:
        return float(trapezoid(np.cos(2 * np.pi * turns * centred), times))

    if most <= 0:
        return []
    # A grid of at most 0.05 turns that ends at `most`, so a zero just below `most` is kept.
    grid = np.linspace(0.0, most, int(np.ceil(most / 0.05)) + 1)
    values = [integral(float(turns)) for turns in grid]
    roots = [float(turns) for turns, value in zip(grid, values) if value == 0.0]
    roots += [float(brentq(integral, grid[i], grid[i + 1], xtol=1e-10))
              for i in range(len(grid) - 1) if values[i] * values[i + 1] < 0]
    return sorted(roots)


#: Largest ratio of a drive rate on the control to its detuning in the automatic CR duration.
CR_DRIVE_RATIO = 0.4
#: Detuning in GHz below which the automatic CR duration refuses a control transition.
_CR_GUARD = 0.005
#: Detuning in GHz below which a 0-1 or 1-2 transition of the control crowds the CR tone.
_CR_CROWDED = 0.060


class _CRPlan(NamedTuple):
    """Seed response, edge and total duration of a CR pair, with the warnings they need."""

    zx: complex
    ix: complex
    elements: tuple[complex, complex] | None
    duration: float
    edge: float
    reason: str | None
    notes: list[tuple[type[Warning], str]]


def _mhz(detuning: float) -> str:
    return f"{detuning * 1e3:.1f} MHz" if detuning >= 1e-4 else f"{detuning * 1e6:.3g} kHz"


def _grid_above(value: float) -> float:
    """Return the smallest multiple of 10 ns above ``value``."""
    return float(10.0 * (np.floor(value / 10.0) + 1.0))


def _cr_transitions(chip: "Chip", control: str, target: str) -> dict[str, float]:
    """Return the dressed detunings in GHz of the CR tone from the control's transitions.

    The tone runs at the target frequency: ``|Δ|`` from the control's 0-1
    transition, ``|Δ + α_c|`` from its 1-2 transition, and ``|2Δ + α_c|``
    from its two-photon 0-2 transition, with ``Δ = f_c - f_t``. A two-level
    control has the 0-1 transition only.
    """
    delta = float(chip.freq(control)) - float(chip.freq(target))
    try:
        alpha = float(chip.dressed_anharmonicity(control))
    except ValueError:
        alpha = float("nan")
    transitions = {"0-1": abs(delta)}
    if np.isfinite(alpha):
        transitions["1-2"] = abs(delta + alpha)
        transitions["two-photon 0-2"] = abs(2 * delta + alpha)
    return transitions


def _cr_edge(detuning: float, sigmas: float) -> float:
    """Return the CR edge for a transition at detuning ``d``: Gaussian width ``σ = 6 / (2π d)``.

    This is an empirical default that the validation tests check. The flat top's joins and the two-photon drive
    leave more weight at ``d`` than an infinite Gaussian edge would.
    """
    return 2 * sigmas * 6.0 / (2 * np.pi * detuning) if detuning > 0 else float("inf")


def _automatic_cr_duration(chip: "Chip", control: str, cr_line: str, transitions: dict[str, float], unit: float,
                           halves: int, edge: float, sigmas: float) -> tuple[float, str]:
    """Return the shortest total CR duration on a 10 ns grid that keeps the drive weak and fits the edges, and why.

    ``unit`` is the seed CR peak amplitude times the area of one tone at unit
    amplitude. A peak amplitude A drives the control's 0-1 and 1-2
    transitions at ``Ω01 = A |<1|D|0>|`` and ``Ω12 = A |<2|D|1>|`` (cyclic GHz,
    dressed elements of the CR line). The duration keeps ``Ω01/|Δ|``,
    ``Ω12/|Δ + α_c|``, and the two-photon ``Ω01 Ω12 / (2 |Δ| |2Δ + α_c|)`` at
    most ``CR_DRIVE_RATIO``.
    """
    from quchip.control.envelopes import GaussianEdge

    equipment = chip.control_equipment
    assert equipment is not None
    line = next(item for item in equipment.lines if item.label == cr_line)
    device = chip.device_map[control]
    m01 = abs(complex(chip.drive_matrix_elements(({}, {device: 1}), drives=[line])[line]))
    limits = {"0-1": CR_DRIVE_RATIO * transitions["0-1"] / m01}
    if "1-2" in transitions:
        m12 = abs(complex(chip.drive_matrix_elements(({device: 1}, {device: 2}), drives=[line])[line]))
        limits["1-2"] = CR_DRIVE_RATIO * transitions["1-2"] / m12
        limits["two-photon 0-2"] = float(np.sqrt(
            2 * CR_DRIVE_RATIO * transitions["0-1"] * transitions["two-photon 0-2"] / (m01 * m12)))
    binding = min(limits, key=limits.__getitem__)
    nearest = min(transitions, key=transitions.__getitem__)
    by_edges = (f"{2 * halves} edges of {edge:.1f} ns for the control's {nearest} transition, "
                f"{_mhz(transitions[nearest])} away")
    shortest = _grid_above(2 * halves * edge)
    if not np.isfinite(shortest):
        return float("inf"), by_edges

    def area(total: float) -> float:
        return _unit_area(GaussianEdge(duration=total / halves, edge_duration=edge, sigmas=sigmas, amplitude=1.0))

    # Above two edges, the area of a tone grows one to one with its length.
    offset = 4 * edge - area(4 * edge * halves)
    duration = max(shortest, float(10.0 * np.ceil(halves * (unit / limits[binding] + offset) / 10.0 - 1e-9)))
    while unit / area(duration) > limits[binding] * (1 + 1e-9):
        duration += 10.0
    if duration > shortest:
        return duration, (f"the drive on the control's {binding} transition at {CR_DRIVE_RATIO:g} of its "
                          f"{_mhz(transitions[binding])} detuning")
    return duration, by_edges


def _cr_plan(chip: "Chip", control: str, target: str, cr_line: str, target_line: str | None, *, echo: bool,
             edge: float | None, sigmas: float, duration: float | None) -> _CRPlan:
    """Return the seed response, the edge, and the total duration of a CR pair, as :func:`cross_resonance` sets them.

    A ``duration`` of None chooses it, and refuses a pair whose tone is too
    close to a control transition for the choice.
    """
    from quchip.analysis.cross_resonance import analyze_cr_susceptibility

    response = analyze_cr_susceptibility(chip, control, target, drive=cr_line)
    zx, ix = complex(response.ZX_per_amplitude), complex(response.IX_per_amplitude)
    halves = 2 if echo else 1
    pair = f"{control}->{target}"
    elements = None
    if target_line is not None:
        elements = _conditional_elements(chip, control, target, target_line)
        t0, t1 = elements
        coefficients = np.array([[zx, (t0 - t1) / 2], [ix, (t0 + t1) / 2]])
        condition = float(np.linalg.cond(coefficients))
        if not condition < 1e12:
            raise ValueError(f"The CR and cancellation tones on {pair} give no independent ZX and IX, since their "
                             f"joint response is singular (condition number {condition:.1e}). Check the coupling "
                             f"and the cancellation line {target_line}, or pass cancel=False.")
        # The seed CR amplitude after the joint solve with the cancellation tone, times the area of one tone.
        unit = abs(complex(np.linalg.solve(coefficients, np.array([1 / (4 * halves), 0.0], dtype=complex))[0]))
    else:
        equipment = chip.control_equipment
        assert equipment is not None
        line = next(item for item in equipment.lines if item.label == cr_line)
        drive = abs(complex(chip.drive_matrix_elements(({}, {chip.device_map[control]: 1}), drives=[line])[line]))
        # Without the cancellation tone the CR tone alone makes the ZX, so a ZX at roundoff level has no seed.
        if not abs(zx) > 64 * np.finfo(float).eps * drive:
            raise ValueError(f"The CR drive on {cr_line} gives {pair} no ZX: {abs(zx):.1e} per unit amplitude, "
                             f"against {drive:.2e} on the control's 0-1 transition. Check the coupling and the "
                             "drive line.")
        unit = 1.0 / (4 * halves * abs(zx))
    transitions = _cr_transitions(chip, control, target)
    nearest = min(transitions, key=transitions.__getitem__)
    close = transitions[nearest] < _CR_GUARD
    notes: list[tuple[type[Warning], str]] = []
    if edge is None:
        edge = _cr_edge(transitions[nearest], sigmas)
        if duration is not None and duration / halves <= 2 * edge and nearest != "0-1":
            fallback = _cr_edge(transitions["0-1"], sigmas)
            if duration / halves > 2 * fallback:
                notes.append((AccuracyWarning,
                              f"A {duration:g} ns CR on {pair} is too short for the {edge:.1f} ns edges that the "
                              f"control's {nearest} transition, {_mhz(transitions[nearest])} from the tone, needs. It "
                              f"uses {fallback:.1f} ns edges, which can excite that transition; "
                              f"{_grid_above(2 * halves * edge):g} ns fits the slower edges."))
            edge = fallback
    if duration is not None and duration / halves <= 2 * edge:
        raise ValueError(f"A {duration:g} ns CR on {pair} is too short for {edge:.1f} ns edges, since each tone must "
                         f"last more than two edges. Give at least {_grid_above(2 * halves * edge):g} ns, or pass a "
                         "shorter edge=.")
    reason = None
    if duration is None:
        if close:
            would = ""
            if transitions[nearest] > 0:
                chosen, _ = _automatic_cr_duration(chip, control, cr_line, transitions, unit, halves, edge, sigmas)
                would = f" The rule would choose {chosen:g} ns." if np.isfinite(chosen) else ""
            raise ValueError(f"No automatic CR duration for {pair}: the tone is {_mhz(transitions[nearest])} from the "
                             f"control's {nearest} transition.{would} Give the duration by hand to build the gate "
                             "anyway, or move the pair away from this transition.")
        duration, reason = _automatic_cr_duration(chip, control, cr_line, transitions, unit, halves, edge, sigmas)
    elif close:
        notes.append((AccuracyWarning, f"The CR tone on {pair} is {_mhz(transitions[nearest])} from the control's "
                                       f"{nearest} transition, closer than the automatic choice allows, so the CR "
                                       "drive can excite it."))
    if transitions.get("1-2", np.inf) < _CR_CROWDED and not (close and nearest == "1-2"):
        notes.append((AccuracyWarning, f"The CR tone on {pair} is {_mhz(transitions['1-2'])} from the control's 1-2 "
                                       "transition, near the avoided crossing of |11> and |20>, so the gates can "
                                       "leak. How much depends on the coupling."))
    if transitions["0-1"] < _CR_CROWDED:
        notes.append((AccuracyWarning, f"The qubits of {pair} are {_mhz(transitions['0-1'])} apart, so SX and X "
                                       "pulses on one can drive the other. Longer SX and X pulses help."))
    return _CRPlan(zx, ix, elements, float(duration), float(edge), reason, notes)


def cross_resonance(
    chip: "Chip",
    control: Any,
    target: Any,
    *,
    duration: float | None = None,
    line: Any = None,
    cancel: bool = True,
    cancel_line: Any = None,
    echo: bool = True,
    edge: float | None = None,
    sigmas: float = 2.5,
    rotary: bool = False,
    name: str = "ZX90",
) -> NativeGate:
    """Return a ZX90 native driven by cross resonance.

    The CR tone runs on the control's charge line at ``freq=chip.freq(target)``
    in the target's frame. A cancellation tone on the target's line runs with
    it. Both are ``GaussianEdge`` flat tops. The knobs are the CR amplitude and
    phase and the cancellation amplitude and phase. The seeds solve two linear
    equations for the complex amplitudes of the two tones: the ZX term turns
    the target by π/2 over the gate, and the IX term is zero. The CR line's
    coefficients come from
    :func:`~quchip.analysis.cross_resonance.analyze_cr_susceptibility`, and the
    target line's come from the same dressed elements, with the control in 0
    and 1.

    With ``echo=True`` the native is composite: CR+, X on the control, CR-,
    and X on the control, where CR- reverses the phases of both tones. The X
    gates come from the gate set's calibrated natives. ``duration`` is the
    total CR tone time of both halves.

    Parameters
    ----------
    chip : Chip
        Chip with the charge lines wired.
    control, target : str or device
        Directed CR pair.
    duration : float or None, default None
        Total CR tone duration in ns. The default is the shortest multiple of
        10 ns at which each tone lasts longer than two edges and the seed CR
        amplitude A keeps the drive on the control weak: ``Ω01/|Δ|``,
        ``Ω12/|Δ + α_c|`` and the two-photon ``Ω01 Ω12 / (2 |Δ| |2Δ + α_c|)``
        stay at most 0.4, with dressed ``Δ = f_c - f_t``, ``Ω01 = A |<1|D|0>|``
        and ``Ω12 = A |<2|D|1>|`` on the control. A two-level control keeps
        the first ratio only. An ``AutoChoiceWarning`` gives the duration and
        what set it. The choice refuses a tone less than 5 MHz from a control
        transition, where a given duration builds with an
        ``AccuracyWarning``. An ``AccuracyWarning`` also flags a tone less than
        60 MHz from the control's 1-2 transition, near the avoided crossing of
        ``|11>`` and ``|20>``, and qubits less than 60 MHz apart. A duration
        chosen here stays when the chip changes in place, with a warning that
        gives the duration this rule picks for the changed chip. A change that
        makes the edges too long for the kept duration raises an error.
    line : str or drive, optional
        CR line on the control. The default is the control's only charge line.
    cancel : bool, default True
        Add the cancellation tone.
    cancel_line : str or drive, optional
        Target line of the cancellation tone.
    echo : bool, default True
        Build the echoed CR.
    edge : float or None, default None
        Rise and fall time of each tone in ns. The CR tone is detuned from the
        control's transitions, so a fast edge excites the control. The
        default sets the edge Gaussian's width to ``σ = 6 / (2π d)`` and the
        edge to ``2 sigmas σ``, with d the smallest of ``|Δ|``,
        ``|Δ + α_c|`` and ``|2Δ + α_c|``. The Gaussian part of the spectrum
        is then negligible at d; the joints between the edges and the plateau
        leave a tail that falls as d⁻³, about 7e-4 of the zero-frequency value
        for a 200 ns tone at d = 100 MHz. The two-photon transition sees the
        squared envelope, which keeps about 1e-3 at ``|2Δ + α_c|`` = 70 MHz for
        a 195 ns tone. The validation tests check the leakage of this default.
        A given duration too short for these edges uses the edges for ``|Δ|``,
        with an ``AccuracyWarning``, and raises when those do not fit either.
    sigmas : float, default 2.5
        Half-width of each edge in Gaussian standard deviations
        (``σ = edge / (2 sigmas)``).
    rotary : bool, default False
        Add rotary seeds, which the gate set screens (see
        :class:`~quchip.gateset.Seed`). In each, the target tone also turns
        the target by N turns per half about the ZX axis, along it or against
        it, and the echo turns it back. N takes the values at which a static
        ZZ between the target and a spectator would average out at first order
        over a half if only the rotary angle θ turned the target,
        ``∫ cos(θ(t) - θ(T)/2) dt = 0``. These are whole turns for a square
        tone and lie between whole turns for shaped edges. The ZX term turns
        the two control branches by ±π/4 more per half, so part of the ZZ
        stays. The target tone's peak Rabi rate stays below ``|α|/8``
        of the target, in the seeds and in the fit. Needs ``echo`` and
        ``cancel``.
    name : str, default "ZX90"
        Gate name.
    """
    from quchip.control.envelopes import GaussianEdge
    from quchip.control.sequence import QuantumSequence

    if rotary and not (echo and cancel):
        raise ValueError("rotary needs echo=True and cancel=True: the echo turns the rotary tone back.")
    control_label, target_label = resolve_label(control), resolve_label(target)
    cr_line = _charge_line(chip, control_label, line)
    target_line = _charge_line(chip, target_label, cancel_line) if cancel else None
    plan = _cr_plan(chip, control_label, target_label, cr_line, target_line, echo=echo, edge=edge, sigmas=sigmas,
                    duration=duration)
    for category, message in plan.notes:
        warnings.warn(message, category, stacklevel=user_stacklevel())
    if plan.reason is not None:
        warnings.warn(f"cross_resonance chose duration={plan.duration!r} for {control_label}->{target_label}, set by "
                      f"{plan.reason}. Pass duration= to keep it.", AutoChoiceWarning, stacklevel=user_stacklevel())
    duration_request, edge_request = duration, edge
    zx, ix, duration, edge = plan.zx, plan.ix, plan.duration, plan.edge
    halves = 2 if echo else 1
    tone = duration / halves
    shape = GaussianEdge(duration=tone, edge_duration=edge, sigmas=sigmas, amplitude=1.0)
    area = _unit_area(shape)
    seeds: list[Seed] = []
    upper = None
    if not cancel:
        cr_amplitude, cr_phase = 1.0 / (4 * halves * abs(zx) * area), -float(np.angle(zx))
    else:
        assert target_line is not None and plan.elements is not None
        t0, t1 = plan.elements
        coefficients = np.array([[zx, (t0 - t1) / 2], [ix, (t0 + t1) / 2]])

        def amplitudes(turns: float) -> np.ndarray:
            # Complex CR and target-tone amplitudes: ZX turns π/4 per half and IX turns the target by `turns`.
            return np.linalg.solve(coefficients, np.array([1 / (4 * halves * area), turns / area], dtype=complex))

        cr, cancellation = amplitudes(0.0)
        cr_amplitude, cr_phase = float(abs(cr)), float(np.angle(cr))
        cancel_amplitude, cancel_phase = float(abs(cancellation)), float(np.angle(cancellation))
        if rotary:
            rate = abs(float(chip.dressed_anharmonicity(target_label)))
            if not np.isfinite(rate):
                rate = abs(float(getattr(chip.device_map[target_label], "anharmonicity", np.nan)))
            if not np.isfinite(rate) or rate == 0:
                raise ValueError(f"rotary bounds the target tone by |α|/8, and {target_label} has no anharmonicity.")
            rate /= 8
            peak = max(abs(t0), abs(t1))
            upper = rate / peak
            for turns in _rotary_turns(shape, (rate - abs(ix * cr_amplitude)) * area):
                for sign in (1, -1):
                    cr, cancellation = amplitudes(sign * turns)
                    if abs(cancellation) < upper:
                        seeds.append(Seed({"cr_amplitude": float(abs(cr)), "cr_phase": float(np.angle(cr)),
                                           "cancel_amplitude": float(abs(cancellation)),
                                           "cancel_phase": float(np.angle(cancellation))},
                                          f"rotary {turns:.2f} turns {'along' if sign > 0 else 'against'} ZX",
                                          screen=True))
    frequency = chip.freq(target_label)

    def tones(sequence: QuantumSequence, sign: float) -> tuple[Any, Any]:
        offset = 0.0 if sign > 0 else np.pi
        cr = sequence.schedule(cr_line, envelope=shape.with_params({"amplitude": cr_amplitude}), freq=frequency,
                               frame=target_label, phase=cr_phase + offset)
        cancellation = None
        if cancel:
            assert target_line is not None
            cancellation = sequence.schedule(
                target_line, envelope=shape.with_params({"amplitude": cancel_amplitude}), freq=frequency,
                frame=target_label, phase=cancel_phase + offset)
        return cr, cancellation

    def knobs_for(pairs: list[tuple[Any, Any]]) -> dict[str, Knob]:
        knobs = {
            "cr_amplitude": Knob([((cr, "amplitude"), 1.0) for cr, _ in pairs]),
            "cr_phase": Knob([((cr, "phase"), 1.0) for cr, _ in pairs]),
        }
        if cancel:
            knobs["cancel_amplitude"] = Knob([((c, "amplitude"), 1.0) for _, c in pairs], scale=max(
                cancel_amplitude, 1e-3 * cr_amplitude), bounds=None if upper is None else (0.0, upper))
            knobs["cancel_phase"] = Knob([((c, "phase"), 1.0) for _, c in pairs])
        return knobs

    qubits = (control_label, target_label)
    if not echo:
        sequence = QuantumSequence(chip)
        pair = tones(sequence, +1.0)
        sequence.barrier(control_label, target_label)
        gate = NativeGate(name, qubits, sequence, knobs=knobs_for([pair]), ideal="ZX90", free_z=qubits)
    else:
        def build(gateset: Any) -> tuple[QuantumSequence, dict[str, Knob]]:
            sequence = QuantumSequence(chip)
            plus = tones(sequence, +1.0)
            sequence.extend(gateset.schedule([("X", control_label)]))
            sequence.barrier(control_label, target_label)
            minus = tones(sequence, -1.0)
            sequence.extend(gateset.schedule([("X", control_label)]))
            sequence.barrier(control_label, target_label)
            return sequence, knobs_for([plus, minus])

        gate = NativeGate.composite(name, qubits, build, ideal="ZX90", free_z=qubits, seeds=seeds)
    def declare(changed: "Chip") -> NativeGate:
        # The gate keeps its duration when the chip changes in place, so a sweep keeps one gate length.
        try:
            native = cross_resonance(changed, control_label, target_label, duration=duration, line=cr_line,
                                     cancel=cancel, cancel_line=target_line, echo=echo, edge=edge_request,
                                     sigmas=sigmas, rotary=rotary, name=name)
        except ValueError as error:
            if duration_request is None:
                error.add_note(f"{name} on {control_label}->{target_label} keeps the {duration:g} ns that "
                               "cross_resonance chose for the chip before the change.")
            raise
        if duration_request is None:
            # Each later change keeps this duration too, and compares it with the choice for that chip.
            native._declare = declare
            try:
                now = _cr_plan(changed, control_label, target_label, cr_line, target_line, echo=echo,
                               edge=edge_request, sigmas=sigmas, duration=None)
            except ValueError:
                # The choice refuses the changed chip, and the kept duration built with an AccuracyWarning.
                return native
            if now.duration != duration:
                category = AccuracyWarning if now.duration > duration else AutoChoiceWarning
                warnings.warn(f"{name} on {control_label}->{target_label} keeps duration={duration!r} after the "
                              f"chip change. For the changed chip, cross_resonance would choose "
                              f"{now.duration!r}, set by {now.reason}. Build the gate set again to use it.",
                              category, stacklevel=user_stacklevel())
        return native

    gate._declare = declare
    return gate
