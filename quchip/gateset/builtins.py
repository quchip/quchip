"""Built-in natives with physics seeds: DRAG SX and cross resonance."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from quchip.gateset.native import Knob, NativeGate
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
    unit = envelope.with_params({"amplitude": 1.0})
    times = np.linspace(0.0, float(unit.duration), 20001)
    return float(np.trapezoid(np.real(np.asarray(unit.value(times))), times))


def drag_sx(
    chip: "Chip",
    qubit: Any,
    *,
    line: Any = None,
    duration: float = 40.0,
    sigmas: float = 3.0,
    name: str = "SX",
    angle: float = np.pi / 2,
) -> NativeGate:
    """Return a single-qubit rotation native with one ``GaussianDRAG`` pulse.

    The pulse runs at ``freq=chip.freq(qubit)`` with ``phase=-arg(m)``, where
    ``m`` is the qubit's dressed ``0 -> 1`` drive matrix element on the line,
    so it rotates about +X. The knobs are the amplitude, the DRAG coefficient
    β (for devices with more than two levels), and the detuning, bounded to
    ±5 MHz so that a wrong-axis solution ends at a bound. The seeds are the
    amplitude ``A = θ / (2π |m| ∫E dt)`` and ``β = 1 / (4π |α|)``, with α the
    dressed anharmonicity.

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
        spectral weight at a neighbor's frequency: next to a qubit 100 MHz
        away with g = 3 MHz, a 30 ns X at ±4σ leaves 1.5e-4 in the neighbor,
        where 40 ns at ±3σ leaves 2e-6.
    name : str, default "SX"
        Gate name.
    angle : float, default π/2
        Rotation angle; π/2 gives SX and π gives X.
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
    amplitude = angle / (2 * np.pi * abs(element) * _unit_area(envelope))
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
    knobs["detuning"] = Knob(pulse, "detuning", bounds=(-0.005, 0.005))
    ideal: Any
    if np.isclose(angle, np.pi / 2):
        ideal = "SX"
    elif np.isclose(angle, np.pi):
        ideal = "X"
    else:
        ideal = np.cos(angle / 2) * np.eye(2) - 1j * np.sin(angle / 2) * np.array([[0, 1], [1, 0]])
    return NativeGate(name, (label,), sequence, knobs=knobs, ideal=ideal, free_z=(label,))


def _conditional_ix(chip: "Chip", control: str, target: str, line_label: str) -> complex:
    """Return a line's IX coefficient on the target: its dressed ``0 -> 1`` element averaged over the control."""
    equipment = chip.control_equipment
    assert equipment is not None
    line = next(line for line in equipment.lines if line.label == line_label)
    control_device, target_device = chip.device_map[control], chip.device_map[target]
    t0 = chip.drive_matrix_elements(({}, {target_device: 1}), drives=[line])[line]
    t1 = chip.drive_matrix_elements(({control_device: 1}, {control_device: 1, target_device: 1}), drives=[line])[line]
    return complex((t0 + t1) / 2)


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
    name: str = "ZX90",
) -> NativeGate:
    """Return a ZX90 native driven by cross resonance.

    The CR tone runs on the control's charge line at ``freq=chip.freq(target)``
    in the target's frame. A cancellation tone on the target's line runs with
    it. Both are ``GaussianEdge`` flat tops. The knobs are the CR amplitude and
    phase and the cancellation amplitude and phase. The seeds come from
    :func:`~quchip.analysis.cross_resonance.analyze_cr_susceptibility`: the CR
    amplitude sets the ZX angle to π/2 over the envelope area, and the
    cancellation tone cancels the IX term with the target line's IX
    coefficient, the average of its elements with the control in 0 and 1.

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
    name : str, default "ZX90"
        Gate name.
    """
    from quchip.analysis.cross_resonance import analyze_cr_susceptibility
    from quchip.control.envelopes import GaussianEdge
    from quchip.control.sequence import QuantumSequence

    control_label, target_label = resolve_label(control), resolve_label(target)
    cr_line = _charge_line(chip, control_label, line)
    target_line = _charge_line(chip, target_label, cancel_line) if cancel else None
    response = analyze_cr_susceptibility(chip, control_label, target_label, drive=cr_line)
    zx, ix = complex(response.ZX_per_amplitude), complex(response.IX_per_amplitude)
    halves = 2 if echo else 1
    tone = duration / halves
    if edge is None:
        detuning = abs(float(chip.freq(control_label)) - float(chip.freq(target_label)))
        edge = 2 * sigmas * 6.0 / (2 * np.pi * detuning)
    if tone <= 2 * edge:
        raise ValueError(f"Each CR tone lasts {tone:g} ns, which does not exceed twice the edge ({edge:g} ns).")
    area = _unit_area(GaussianEdge(duration=tone, edge_duration=edge, sigmas=sigmas, amplitude=1.0))
    cr_amplitude = 1.0 / (4 * halves * abs(zx) * area)
    cr_phase = -float(np.angle(zx))
    if cancel:
        assert target_line is not None
        element = _conditional_ix(chip, control_label, target_label, target_line)
        wanted = -ix * cr_amplitude * np.exp(1j * cr_phase) / element
        cancel_amplitude, cancel_phase = abs(wanted), float(np.angle(wanted))
    frequency = chip.freq(target_label)

    def tones(sequence: QuantumSequence, sign: float) -> tuple[Any, Any]:
        offset = 0.0 if sign > 0 else np.pi
        cr = sequence.schedule(cr_line, envelope=GaussianEdge(duration=tone, edge_duration=edge, sigmas=sigmas,
                                                              amplitude=cr_amplitude),
                               freq=frequency, frame=target_label, phase=cr_phase + offset)
        cancellation = None
        if cancel:
            assert target_line is not None
            cancellation = sequence.schedule(
                target_line, envelope=GaussianEdge(duration=tone, edge_duration=edge, sigmas=sigmas,
                                                   amplitude=cancel_amplitude),
                freq=frequency, frame=target_label, phase=cancel_phase + offset)
        return cr, cancellation

    def knobs_for(pairs: list[tuple[Any, Any]]) -> dict[str, Knob]:
        knobs = {
            "cr_amplitude": Knob([((cr, "amplitude"), 1.0) for cr, _ in pairs]),
            "cr_phase": Knob([((cr, "phase"), 1.0) for cr, _ in pairs]),
        }
        if cancel:
            knobs["cancel_amplitude"] = Knob([((c, "amplitude"), 1.0) for _, c in pairs], scale=max(
                cancel_amplitude, 1e-3 * cr_amplitude))
            knobs["cancel_phase"] = Knob([((c, "phase"), 1.0) for _, c in pairs])
        return knobs

    qubits = (control_label, target_label)
    if not echo:
        sequence = QuantumSequence(chip)
        pair = tones(sequence, +1.0)
        sequence.barrier(control_label, target_label)
        return NativeGate(name, qubits, sequence, knobs=knobs_for([pair]), ideal="ZX90", free_z=qubits)

    def build(gates: Any) -> tuple[QuantumSequence, dict[str, Knob]]:
        sequence = QuantumSequence(chip)
        plus = tones(sequence, +1.0)
        sequence.extend(gates.schedule([("X", control_label)]))
        sequence.barrier(control_label, target_label)
        minus = tones(sequence, -1.0)
        sequence.extend(gates.schedule([("X", control_label)]))
        sequence.barrier(control_label, target_label)
        return sequence, knobs_for([plus, minus])

    return NativeGate.composite(name, qubits, build, ideal="ZX90", free_z=qubits)
