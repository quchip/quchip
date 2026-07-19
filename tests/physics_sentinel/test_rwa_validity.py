"""RWA validity: counter-rotating coupling and drive terms shift transitions by the Bloch-Siegert scale.

Pins the RWA and counter-rotating-term contract in ``PHYSICS.md`` §§6-7, exercised via the
user-facing ``rwa=False`` switch.
"""
import numpy as np
import pytest

from quchip import Capacitive, ChargeDrive, Chip, ControlEquipment, DuffingTransmon, Gaussian, QuantumSequence
from quchip.engine import solve_problem

F_A, F_B = 5.0, 5.1  # GHz


def _coupling_shift(g: float) -> float:
    """Exact 0→1 splitting change when the counter-rotating coupling is kept."""
    # Counter-rotating g(ab + a-dag b-dag) shifts 0->1 by +g^2/(f_a+f_b) at second order
    # (Bloch & Siegert 1940; Zueco et al., PRA 80, 033846 (2009)).
    splitting = {}
    for rwa in (True, False):
        qa = DuffingTransmon(freq=F_A, anharmonicity=-0.25, levels=2)
        qb = DuffingTransmon(freq=F_B, anharmonicity=-0.25, levels=2)
        chip = Chip([qa, qb], couplings=[Capacitive(qa, qb, g=g, rwa=rwa)])
        evals = np.sort(np.linalg.eigvalsh(chip.hamiltonian().full()))
        splitting[rwa] = evals[1] - evals[0]
    return splitting[False] - splitting[True]


def test_coupling_counter_rotating_shift_is_g_squared_over_sum_frequency():
    """The counter-rotating coupling shift matches the Bloch-Siegert prediction g^2/(f_a+f_b)."""
    for g in (0.01, 0.02):
        assert _coupling_shift(g) == pytest.approx(g**2 / (F_A + F_B), rel=1e-3)


def test_coupling_counter_rotating_shift_scales_quadratically():
    """Doubling the coupling strength g quadruples the counter-rotating splitting shift."""
    assert _coupling_shift(0.02) / _coupling_shift(0.01) == pytest.approx(4.0, rel=1e-3)


def _drive_cr_phase(amp: float, f_drive: float, duration: float = 30.0) -> tuple[float, float]:
    """Return (Δφ measured full-vs-RWA, predicted −2π·2∫(A/2)²dt/f_Σ)."""
    tlist = np.linspace(0.0, duration, 601)
    phase = {}
    for rwa in (True, False):
        q = DuffingTransmon(freq=F_A, anharmonicity=-0.25, levels=2)
        drive = ChargeDrive(target=q)
        chip = Chip([q], control_equipment=ControlEquipment(lines=[drive]), frame={q: F_A}, rwa=rwa)
        sequence = QuantumSequence(chip)
        sequence.schedule(drive, envelope=Gaussian(duration=duration, amplitude=amp, sigmas=4), freq=f_drive)
        problem = sequence.build_problem(
            tlist=tlist,
            e_ops=chip.e_ops(**{q.label: "a"}),
            initial_state=chip.superposition({q: 0}, {q: 1}),
            options={"atol": 1e-12, "rtol": 1e-10, "nsteps": 10_000_000},
        )
        result = solve_problem(problem, check_truncation=False)
        phase[rwa] = np.angle(np.asarray(result.expect(q.label))[-1])

    grid = np.linspace(0.0, duration, 4001)
    waveform = np.asarray(Gaussian(duration=duration, amplitude=amp, sigmas=4).waveform(grid))
    integral = np.trapezoid(np.abs(waveform / 2.0) ** 2, grid)
    # Generalized Bloch-Siegert phase: Δφ = -2π·2∫(A(t)/2)² dt / (f_q + f_d), first order in the shift.
    predicted = -2.0 * np.pi * 2.0 * integral / (F_A + f_drive)
    return phase[False] - phase[True], predicted


def test_drive_counter_rotating_phase_matches_bloch_siegert_scale():
    """At A = 0.1 GHz the measured counter-rotating phase matches A²/(2 f_Σ) to within 2%."""
    measured, predicted = _drive_cr_phase(amp=0.1, f_drive=4.0)
    assert measured == pytest.approx(predicted, rel=2e-2)
    assert measured < 0.0  # counter-rotating term shifts the resonance UP


def test_drive_counter_rotating_phase_scales_quadratically_with_amplitude():
    """Doubling the drive amplitude quadruples the counter-rotating Bloch-Siegert phase shift."""
    weak, _ = _drive_cr_phase(amp=0.1, f_drive=4.0)
    strong, _ = _drive_cr_phase(amp=0.2, f_drive=4.0)
    assert strong / weak == pytest.approx(4.0, rel=2.5e-2)
