"""Carrier of a pulse scheduled without ``freq``: the drive class chooses it, and ``freq=None`` is baseband."""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest

from quchip import (
    Capacitive,
    ChargeDrive,
    Chip,
    DuffingTransmon,
    Exact,
    FluxDrive,
    Gaussian,
    KerrCavity,
    ParametricDrive,
    PhaseDrive,
    PortNetwork,
    QuantumSequence,
    Resonator,
    RWA,
    Square,
    TunableCapacitive,
    TwoPhotonDrive,
)
from quchip.control.field import CoherentInput


CARRIER_WARNING = ("A pulse without freq gets a carrier at its device's dressed 0 -> 1 frequency, chip.freq(device) "
                   "(twice it on a two-photon line). Pass freq= to set it, or freq=None for no carrier.")


def _carrier_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [str(item.message) for item in caught if str(item.message) == CARRIER_WARNING]


@pytest.mark.unit
def test_a_pulse_without_freq_takes_the_carrier_its_drive_class_declares() -> None:
    """Charge and phase lines take chip.freq, a two-photon line twice it, and other lines stay at baseband."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=3, label="r")
    s = DuffingTransmon(freq=5.4, anharmonicity=-0.25, levels=3, label="s")
    cav = KerrCavity(freq=6.0, kerr=0.002, levels=4, label="cav")
    edge = TunableCapacitive(q, s, g_0=0.005, label="qs")
    network = PortNetwork(label="line")
    network.port("feed", target=r, rate=0.001)
    chip = Chip([q, r, s, cav], couplings=[Capacitive(q, r, g=0.08), edge], port_network=network)
    chip.wire(ChargeDrive(q, label="xy"), PhaseDrive(q, label="ph"), FluxDrive(s, label="z"),
              TwoPhotonDrive(cav, label="pump2"), ParametricDrive(edge, label="pump"))
    dressed, cavity = float(chip.freq(q)), float(chip.freq(cav))
    # The resonator pulls the qubit by about g²/Δ = 3.2 MHz, so the carrier is the dressed frequency.
    assert abs(dressed - 5.0) > 1e-3

    pulse = Square(duration=10.0, amplitude=0.01)
    seq = QuantumSequence(chip)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for line in ("xy", "ph", "pump2", "z", "pump", CoherentInput("feed"), "xy"):
            seq.schedule(line, envelope=pulse)
        seq.charge(q, envelope=pulse)
        seq.schedule("xy", envelope=pulse, freq=5.1)
    carriers = [op.freq for op in seq.scheduled_ops]
    np.testing.assert_allclose(carriers[:3] + carriers[6:8], [dressed, dressed, 2 * cavity, dressed, dressed],
                               rtol=0, atol=0)
    assert carriers[3:6] == [None, None, None]
    assert [op.frame for op in seq.scheduled_ops[:3]] == ["q", "q", "cav"]
    # One warning per line with an automatic carrier.
    assert _carrier_warnings(caught) == [CARRIER_WARNING] * 3
    # Each new sequence reports its own lines once.
    with pytest.warns(UserWarning, match=re.escape(CARRIER_WARNING)):
        QuantumSequence(chip).schedule("xy", envelope=pulse)
    # The two-photon carrier makes the ±2 bands static in the cavity's dressed frame.
    plan = seq.resolve(frame="auto").resolved_frame.plan
    np.testing.assert_allclose(plan.frequencies["cav"], cavity, rtol=1e-12, atol=0)


@pytest.mark.unit
def test_charge_takes_the_carrier_and_the_warning_of_schedule() -> None:
    """charge() without freq takes chip.freq(q) and warns once per line, and freq=None gives a baseband pulse."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=3, label="r")
    chip = Chip([q, r], couplings=[Capacitive(q, r, g=0.08)])
    chip.wire(ChargeDrive(q, label="xy"))
    pulse = Square(duration=10.0, amplitude=0.01)
    dressed = float(chip.freq(q))
    seq = QuantumSequence(chip)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        seq.charge(q, envelope=pulse)
        seq.charge(q, envelope=pulse)
        seq.charge(q, envelope=pulse, freq=None)
    carriers = [op.freq for op in seq.scheduled_ops]
    np.testing.assert_allclose(carriers[:2], [dressed, dressed], rtol=0, atol=0)
    assert carriers[2] is None
    assert _carrier_warnings(caught) == [CARRIER_WARNING]
    assert [item.filename for item in caught if str(item.message) == CARRIER_WARNING] == [__file__]
    with pytest.raises(ValueError, match="freq must be a carrier frequency in GHz, None, or 'auto'"):
        seq.charge(q, envelope=pulse, freq="dressed")


@pytest.mark.unit
def test_freq_none_schedules_the_carrier_free_pulse() -> None:
    """freq=None follows no frame and takes neither frame nor detuning; freq=0.0 stays a carrier in its frame."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([q])
    chip.wire(ChargeDrive(q, label="xy"))
    pulse = Square(duration=10.0, amplitude=0.01)
    seq = QuantumSequence(chip)
    seq.vz(q, 0.4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        seq.schedule("xy", envelope=pulse, freq=None)
        seq.schedule("xy", envelope=pulse, freq=0.0)
    baseband, zero = seq.scheduled_ops
    assert (baseband.freq, baseband.frame, baseband.phase_offset) == (None, None, 0.0)
    assert (zero.freq, zero.frame, zero.phase_offset) == (0.0, "q", 0.4)
    with pytest.raises(ValueError, match="A frame requires a carrier"):
        seq.schedule("xy", envelope=pulse, freq=None, frame=q)
    with pytest.raises(ValueError, match="A detuning requires a carrier"):
        seq.schedule("xy", envelope=pulse, freq=None, detuning=0.01)
    with pytest.raises(ValueError, match="freq must be a carrier frequency in GHz, None, or 'auto'"):
        seq.schedule("xy", envelope=pulse, freq="dressed")


@pytest.mark.unit
def test_a_pulse_without_freq_asks_for_freq_when_the_dressed_frequency_is_unreliable() -> None:
    """Identical coupled transmons have no reliable dressed 0 -> 1 label, so the automatic carrier asks for freq."""
    devices = [DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label=label) for label in ("a", "b")]
    chip = Chip(devices, [Capacitive(*devices, g=0.01, label="ab")])
    chip.wire(ChargeDrive(devices[0], label="xy"))
    seq = QuantumSequence(chip)
    pulse = Square(duration=10.0, amplitude=0.01)
    with pytest.raises(ValueError, match=r"line 'xy' sets no freq, and its carrier needs chip.freq\('a'\): Dressed"):
        seq.schedule("xy", envelope=pulse)
    seq.schedule("xy", envelope=pulse, freq=5.0)
    assert [op.freq for op in seq.scheduled_ops] == [5.0]


def test_a_pulse_without_freq_drives_the_dressed_transition() -> None:
    """Without freq, a calibrated π pulse matches freq=chip.freq(q). The previous baseband pulse leaves q idle."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=3, label="r")
    chip = Chip([q, r], couplings=[Capacitive(q, r, g=0.08)], frame="rotating")
    line = ChargeDrive(q, label="xy")
    chip.wire(line)
    envelope = Gaussian(duration=20.0, amplitude=1.0)
    amplitude = chip.rotation_amplitude(q, line, envelope)

    def excited(**carrier):
        seq = QuantumSequence(chip)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            seq.schedule(line, envelope=envelope.with_params({"amplitude": amplitude}), **carrier)
        return float(np.real(seq.simulate(tlist=np.linspace(0.0, 20.0, 3)).population(q, 1)[-1]))

    automatic, explicit, baseband = excited(), excited(freq=chip.freq(q)), excited(freq=None)
    # The same carrier builds the same problem, so the populations agree to roundoff.
    assert automatic == pytest.approx(explicit, rel=0, abs=1e-12)
    # Leakage to |2> and the AC Stark shift keep this first-guess 20 ns pulse at P1 = 0.988.
    assert automatic > 0.98
    # A baseband pulse 5 GHz below the transition excites |1> to second order in Ω/ω ≈ 1e-2.
    assert baseband < 1e-6


def test_baseband_pulse_keeps_the_complete_real_field_under_rwa() -> None:
    """Under RWA, a freq=None or freq=0.0 pulse on a 50 MHz mode evolves as under Exact()."""
    def excited(approximation, freq, phase=0.0):
        q = DuffingTransmon(freq=0.05, anharmonicity=0.0, levels=3, label="q")
        chip = Chip([q], approximation=approximation)
        chip.wire(ChargeDrive(q, label="xy"))
        seq = QuantumSequence(chip)
        seq.schedule("xy", envelope=Gaussian(duration=20.0, amplitude=0.02), freq=freq, phase=phase)
        return float(np.real(seq.simulate(tlist=np.linspace(0.0, 20.0, 3)).population(q, 1)[-1]))

    # Both solves integrate the same Hamiltonian, so they agree to the solver tolerance.
    baseband = excited(RWA(), None)
    np.testing.assert_allclose(baseband, excited(Exact(), None), rtol=1e-6)
    assert baseband == pytest.approx(0.2757, abs=1e-3)
    # Both partners of a zero carrier are resonant, so RWA() keeps the real drive A(t) cos φ of Exact().
    np.testing.assert_allclose(excited(RWA(), 0.0), baseband, rtol=1e-6)
    assert excited(RWA(), 0.0, phase=np.pi / 2) < 1e-12


@pytest.mark.optional_backend
def test_zero_freq_is_the_same_carrier_eagerly_and_under_jit() -> None:
    """freq=0.0 builds the same zero-frequency carrier eagerly, under jax.jit, and in a batch through zero."""
    pytest.importorskip("dynamiqs")
    import jax

    q = DuffingTransmon(freq=0.05, anharmonicity=0.0, levels=3, label="q")
    chip = Chip([q], backend="dynamiqs")
    chip.wire(ChargeDrive(q, label="xy"))

    def excited(freq):
        seq = QuantumSequence(chip)
        seq.schedule("xy", envelope=Gaussian(duration=20.0, amplitude=0.02), freq=freq)
        return seq.simulate(tlist=np.linspace(0.0, 20.0, 3)).population(q, 1)[-1]

    eager = float(np.real(excited(0.0)))
    np.testing.assert_allclose(float(np.real(jax.jit(excited)(0.0))), eager, rtol=0, atol=1e-10)
    # RWA() keeps the complete real drive of the zero carrier, as in the qutip check above.
    assert eager == pytest.approx(0.2757, abs=1e-3)


@pytest.mark.parametrize("backend", ["qutip", pytest.param("dynamiqs", marks=pytest.mark.optional_backend)])
def test_carrier_batch_through_zero_matches_single_solves(backend) -> None:
    """A carrier axis through 0.0 keeps both partners at zero and one elsewhere, as single solves do."""
    pytest.importorskip(backend)
    q = DuffingTransmon(freq=0.05, anharmonicity=0.0, levels=3, label="q")
    chip = Chip([q], backend=backend, frame="rotating", approximation=RWA())
    chip.wire(ChargeDrive(q, label="xy"))

    def sequence(freq):
        seq = QuantumSequence(chip)
        return seq, seq.schedule("xy", envelope=Gaussian(duration=20.0, amplitude=0.02), freq=freq)

    times = np.linspace(0.0, 20.0, 3)
    seq, pulse = sequence(0.0)
    batch = seq.simulate_batch(pulse.vary("freq", [-0.01, 0.0, 0.01], name="f"), tlist=times, progress=False,
                               max_workers=1)
    swept = np.real(np.asarray(batch.population(q, level=1, reduce="last")))
    single = [float(np.real(sequence(freq)[0].simulate(tlist=times).population(q, 1)[-1])) for freq in (0.0, 0.01)]
    # A real envelope at zero phase gives cos(2πft), even in f, so the ±0.01 GHz points agree.
    np.testing.assert_allclose(swept, [single[1], single[0], single[1]], rtol=0, atol=1e-8)
    assert single[0] == pytest.approx(0.2757, abs=1e-3)
