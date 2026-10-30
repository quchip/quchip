"""Built-in natives with physics seeds: DRAG SX and cross resonance."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from quchip.gateset.native import Knob, NativeGate, Seed
from quchip.utils.labeling import resolve_label

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


def cross_resonance(
    chip: "Chip",
    control: Any,
    target: Any,
    *,
    duration: float,
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
    duration : float
        Total CR tone duration in ns.
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
        control by the qubit detuning Δ, so a fast edge excites the control.
        The default sets the edge Gaussian's width to ``σ = 6 / (2π|Δ|)`` and
        the edge to ``2 sigmas σ``. The Gaussian part of the spectrum is then
        negligible at Δ; the joints between the edges and the plateau leave a
        tail that falls as Δ⁻³, about 7e-4 of the zero-frequency value for a
        200 ns tone at Δ = 100 MHz.
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
    from quchip.analysis.cross_resonance import analyze_cr_susceptibility
    from quchip.control.envelopes import GaussianEdge
    from quchip.control.sequence import QuantumSequence

    if rotary and not (echo and cancel):
        raise ValueError("rotary needs echo=True and cancel=True: the echo turns the rotary tone back.")
    control_label, target_label = resolve_label(control), resolve_label(target)
    cr_line = _charge_line(chip, control_label, line)
    target_line = _charge_line(chip, target_label, cancel_line) if cancel else None
    response = analyze_cr_susceptibility(chip, control_label, target_label, drive=cr_line)
    zx, ix = complex(response.ZX_per_amplitude), complex(response.IX_per_amplitude)
    halves = 2 if echo else 1
    tone = duration / halves
    edge_request = edge
    if edge is None:
        detuning = abs(float(chip.freq(control_label)) - float(chip.freq(target_label)))
        edge = 2 * sigmas * 6.0 / (2 * np.pi * detuning)
    if tone <= 2 * edge:
        raise ValueError(f"Each CR tone lasts {tone:g} ns, which does not exceed twice the edge ({edge:g} ns).")
    shape = GaussianEdge(duration=tone, edge_duration=edge, sigmas=sigmas, amplitude=1.0)
    area = _unit_area(shape)
    cr_amplitude = 1.0 / (4 * halves * abs(zx) * area)
    cr_phase = -float(np.angle(zx))
    seeds: list[Seed] = []
    upper = None
    if cancel:
        assert target_line is not None
        t0, t1 = _conditional_elements(chip, control_label, target_label, target_line)
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
        def build(gates: Any) -> tuple[QuantumSequence, dict[str, Knob]]:
            sequence = QuantumSequence(chip)
            plus = tones(sequence, +1.0)
            sequence.extend(gates.schedule([("X", control_label)]))
            sequence.barrier(control_label, target_label)
            minus = tones(sequence, -1.0)
            sequence.extend(gates.schedule([("X", control_label)]))
            sequence.barrier(control_label, target_label)
            return sequence, knobs_for([plus, minus])

        gate = NativeGate.composite(name, qubits, build, ideal="ZX90", free_z=qubits, seeds=seeds)
    gate._declare = lambda changed: cross_resonance(
        changed, control_label, target_label, duration=duration, line=cr_line, cancel=cancel, cancel_line=target_line,
        echo=echo, edge=edge_request, sigmas=sigmas, rotary=rotary, name=name)
    return gate
