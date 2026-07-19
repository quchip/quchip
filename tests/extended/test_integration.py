"""Integration tests for the dispersive readout pipeline.

Full 3-device chain: transmon + readout resonator + Purcell filter.
Verifies that the backend-agnostic simulation produces physically correct
dispersive readout results.  All expected values come from dispersive
physics formulas, not from running the code first.

Physics:
    ω_q = 5.0 GHz, α = -0.3 GHz, 3 levels
    ω_r = 5.5 GHz, 6 levels
    ω_f = 6.0 GHz, 4 levels
    g_qr = 0.04 GHz (40 MHz)
    g_rf = 0.03 GHz (30 MHz)
    Readout drive on filter at ω_r = 5.5 GHz, amplitude 0.01 GHz, 100 ns Gaussian.

    Dispersive regime: Δ_qr = ω_q - ω_r = -0.5 GHz, g_qr/Δ_qr = 0.08 ≪ 1.
    Dispersive shift: χ = g²·α / [Δ·(Δ+α)] = 0.04²·(-0.3) / [(-0.5)·(-0.8)]
                        = -4.8e-4 / 0.4 = -1.2e-3 GHz.
    State-dependent cavity freq: ω_r ± χ → different steady-state amplitudes
    for |g⟩ vs |e⟩ → IQ separation ≠ 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from quchip import (
    Capacitive,
    ChargeDrive,
    Chip,
    ControlEquipment,
    DuffingTransmon,
    Gaussian,
    QuantumSequence,
    Resonator,
)
from quchip.utils.labeling import reset_label_counters


# ---------------------------------------------------------------------------
# Shared simulation results — run once per module
# ---------------------------------------------------------------------------


@dataclass
class DispersiveResults:
    """Container for the two simulation results and derived observables."""

    result_g: object  # SimulationResult for |g⟩
    result_e: object  # SimulationResult for |e⟩
    times: np.ndarray
    n_g: np.ndarray  # cavity photon number from |g⟩
    n_e: np.ndarray  # cavity photon number from |e⟩
    a_g: np.ndarray  # demodulated ⟨a⟩ from |g⟩ (complex)
    a_e: np.ndarray  # demodulated ⟨a⟩ from |e⟩ (complex)
    readout_freq: float


@pytest.fixture(scope="module")
def dispersive_results() -> DispersiveResults:
    """Run the 3-device dispersive readout simulation once and share results across the module."""
    reset_label_counters()

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3, label="q")
    r = Resonator(freq=5.5, levels=6, label="r")
    f = Resonator(freq=6.0, levels=4, label="f")

    drive_f = ChargeDrive(target=f, label="Df")

    chip = Chip(
        devices=[q, r, f],
        couplings=[
            Capacitive(q, r, g=0.04, rwa=False),
            Capacitive(r, f, g=0.03, rwa=False),
        ],
        control_equipment=ControlEquipment(lines=[drive_f]),
    )
    chip.set_frame("lab")

    readout_freq = r.freq  # 5.5 GHz
    readout_duration = 100.0  # ns
    readout_amplitude = 0.01  # GHz
    envelope = Gaussian(duration=readout_duration, amplitude=readout_amplitude, sigmas=3)

    tlist = np.linspace(0, readout_duration, 1000)

    # |g⟩ simulation
    seq_g = QuantumSequence(chip)
    seq_g.schedule(drive_f, envelope=envelope, freq=readout_freq)
    psi_g = chip.bare_state(q=0, r=0, f=0)
    result_g = seq_g.simulate(tlist=tlist, initial_state=psi_g, e_ops=chip.e_ops(r=["a", "n"]))

    # |e⟩ simulation
    seq_e = QuantumSequence(chip)
    seq_e.schedule(drive_f, envelope=envelope, freq=readout_freq)
    psi_e = chip.bare_state(q=1, r=0, f=0)
    result_e = seq_e.simulate(tlist=tlist, initial_state=psi_e, e_ops=chip.e_ops(r=["a", "n"]))

    t = result_g.times
    n_g = np.real(result_g._expect_data["r"][1].values)
    n_e = np.real(result_e._expect_data["r"][1].values)

    # Demodulate ⟨a⟩ to drive frame
    demod = np.exp(+1j * 2 * np.pi * readout_freq * t)
    a_g_demod = np.asarray(result_g._expect_data["r"][0].values) * demod
    a_e_demod = np.asarray(result_e._expect_data["r"][0].values) * demod

    reset_label_counters()

    return DispersiveResults(
        result_g=result_g,
        result_e=result_e,
        times=t,
        n_g=n_g,
        n_e=n_e,
        a_g=a_g_demod,
        a_e=a_e_demod,
        readout_freq=readout_freq,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDispersiveSimulationCompletes:
    """Both simulations complete without error and produce valid output."""

    def test_g_result_has_times(self, dispersive_results: DispersiveResults) -> None:
        """The |g⟩ simulation produces a time array of expected length."""
        assert len(dispersive_results.times) == 1000

    def test_g_result_has_expect_values(self, dispersive_results: DispersiveResults) -> None:
        """The |g⟩ result has one named observable with two traces."""
        assert len(dispersive_results.result_g._expect_data) == 1
        assert len(dispersive_results.result_g._expect_data["r"]) == 2

    def test_e_result_has_expect_values(self, dispersive_results: DispersiveResults) -> None:
        """The |e⟩ result has one named observable with two traces."""
        assert len(dispersive_results.result_e._expect_data) == 1
        assert len(dispersive_results.result_e._expect_data["r"]) == 2

    def test_times_monotonic(self, dispersive_results: DispersiveResults) -> None:
        """Time array is monotonically increasing."""
        dt = np.diff(dispersive_results.times)
        assert np.all(dt > 0), "Time array is not monotonically increasing"


class TestCavityPhotonsStateDependent:
    """Cavity photon numbers differ between |g⟩ and |e⟩ initial states.

    Physics: The dispersive shift χ shifts the resonator frequency depending
    on the qubit state.  When driven at bare ω_r, the cavity response differs
    for |g⟩ vs |e⟩, producing different photon numbers.
    """

    def test_g_photons_nonzero(self, dispersive_results: DispersiveResults) -> None:
        """Peak ⟨n⟩ from |g⟩ is > 0 — the cavity is being driven."""
        assert np.max(dispersive_results.n_g) > 0, "Cavity photon number from |g⟩ is zero — drive not working"

    def test_e_photons_nonzero(self, dispersive_results: DispersiveResults) -> None:
        """Peak ⟨n⟩ from |e⟩ is > 0 — the cavity is being driven."""
        assert np.max(dispersive_results.n_e) > 0, "Cavity photon number from |e⟩ is zero — drive not working"

    def test_photon_numbers_differ(self, dispersive_results: DispersiveResults) -> None:
        """Peak photon counts differ between |g⟩ and |e⟩ due to the dispersive shift."""
        peak_g = np.max(dispersive_results.n_g)
        peak_e = np.max(dispersive_results.n_e)
        assert peak_g != pytest.approx(peak_e, abs=1e-6), (
            f"Peak photons are identical (g={peak_g:.6f}, e={peak_e:.6f}) — no dispersive shift observed"
        )


class TestIQSeparationNonzero:
    """IQ separation is nonzero — the core dispersive readout observable.

    Physics: |a_g(t_final) - a_e(t_final)| > 0 confirms that the coherent
    state amplitude in the readout resonator is qubit-state-dependent,
    which is the physical basis of dispersive readout.
    """

    def test_iq_separation_positive(self, dispersive_results: DispersiveResults) -> None:
        """Final IQ separation is strictly positive."""
        iq_sep = np.abs(dispersive_results.a_g[-1] - dispersive_results.a_e[-1])
        assert iq_sep > 0, "IQ separation is zero — dispersive readout is not working"

    def test_iq_separation_meaningful(self, dispersive_results: DispersiveResults) -> None:
        """IQ separation is at least 0.01 (cavity-amplitude units) for χ ≈ 1.2 MHz, drive 10 MHz, 100 ns."""
        iq_sep = np.abs(dispersive_results.a_g[-1] - dispersive_results.a_e[-1])
        assert iq_sep > 0.01, f"IQ separation = {iq_sep:.6f} is too small to be physical"


class TestQubitPopulationsPhysical:
    """Qubit populations remain physical during readout.

    Physics:
    - Weak readout (Ω = 0.01 GHz ≪ Δ = 0.5 GHz) should not excite the qubit
      significantly.
    - From |g⟩: ground state population should stay > 0.9 throughout.
    - From |e⟩: excited state population stays > 0.5 (bounded T1 decay and
      readout-induced transitions in 100 ns window).
    """

    def test_g_stays_in_ground(self, dispersive_results: DispersiveResults) -> None:
        """Starting from |g⟩, qubit ground-state pop stays > 0.9."""
        pop_g0 = dispersive_results.result_g.population("q", 0)
        min_pop = np.min(pop_g0)
        assert min_pop > 0.9, (
            f"Qubit ground pop dropped to {min_pop:.4f} — weak readout should not excite qubit significantly"
        )

    def test_e_stays_mostly_excited(self, dispersive_results: DispersiveResults) -> None:
        """Starting from |e⟩, qubit excited-state pop stays > 0.5 (closed system, no T1)."""
        pop_e1 = dispersive_results.result_e.population("q", 1)
        min_pop = np.min(pop_e1)
        assert min_pop > 0.5, (
            f"Qubit excited pop dropped to {min_pop:.4f} — unexpected large readout-induced transition"
        )


class TestPhotonNumbersBounded:
    """Peak photon numbers are bounded — drive is in the dispersive regime.

    Physics: With Ω = 0.01 GHz (10 MHz) drive and dispersive shift
    χ ≈ 1.2 MHz, the steady-state photon number is
    n̄ ≈ (Ω/κ_eff)² or bounded by the transient fill.
    For these parameters, < 10 photons is expected.
    """

    def test_g_photons_bounded(self, dispersive_results: DispersiveResults) -> None:
        """Peak cavity photons from |g⟩ < 10."""
        peak = np.max(dispersive_results.n_g)
        assert peak < 10, f"Peak photons from |g⟩ = {peak:.4f} — exceeds dispersive bound"

    def test_e_photons_bounded(self, dispersive_results: DispersiveResults) -> None:
        """Peak cavity photons from |e⟩ < 10."""
        peak = np.max(dispersive_results.n_e)
        assert peak < 10, f"Peak photons from |e⟩ = {peak:.4f} — exceeds dispersive bound"
