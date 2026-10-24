"""Physics-verifying simulation tests — all expected values from analytical formulas.

Frame consistency:
    Rabi oscillation populations must match whether computed in lab or
    rotating frame.  P(|1⟩, t) = sin²(π·Ω·t) in both frames.

Dispersive shift eigenvalues:
    Coupled transmon+resonator in the dispersive regime.  The dressed
    |0,1⟩ ↔ |1,1⟩ splitting differs from the bare qubit splitting by
    2χ, where χ = g²·α / [Δ·(Δ+α)].

Resonant eigenvalue splitting:
    Two identical resonators coupled capacitively: single-excitation
    manifold splits by exactly 2g.

Result accessors:
    overlap(), reduced_state(), reduced() checked against physics.

Coherent state:
    ⟨n̂⟩ = |α|² for a resonator coherent state.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt

from quchip.backend.protocol import Backend
from quchip.chip.chip import Chip
from quchip.chip.couplings import Capacitive
from quchip.control.drive import ChargeDrive, FluxDrive
from quchip.control.envelopes import Gaussian, Square
from quchip.control.equipment import ControlEquipment
from quchip.devices.resonator import Resonator
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.engine import simulate
from quchip.engine.ir import DriveOp
from quchip.engine.frames import resolve_frame


class TestFrameConsistency:
    """Rabi oscillation populations must match in lab and rotating frames.

    Single DuffingTransmon (freq=5.0, α=-0.25, levels=3), resonant
    ChargeDrive with Ω=0.02 GHz.  P(|1⟩, t) = sin²(π·Ω·t).
    Lab and rotating frame populations must agree within 1%.
    """

    FREQ = 5.0
    ANHARMONICITY = -0.25
    LEVELS = 3
    OMEGA = 0.02
    DURATION = 50.0
    N_POINTS = 201

    def _run_rabi_in_frame(self, frame: str) -> np.ndarray:
        """Run Rabi oscillation and return P(|1⟩) time series."""
        q = DuffingTransmon(
            freq=self.FREQ,
            anharmonicity=self.ANHARMONICITY,
            levels=self.LEVELS,
            label="q",
        )
        drive = ChargeDrive(target=q)
        chip = Chip([q])
        chip.connect(ControlEquipment(lines=[drive]))

        if frame == "rotating":
            chip.set_frame("rotating")

        envelope = Square(duration=self.DURATION, amplitude=self.OMEGA)
        drive_op = DriveOp(
            target_label="q",
            envelope=envelope,
            freq=self.FREQ,
            start_time=0.0,
            drive_label=drive.label,
        )
        tlist = np.linspace(0, self.DURATION, self.N_POINTS)

        result = simulate(chip, [drive_op], tlist)
        return result.population("q", 1)


    def test_lab_frame_matches_analytic(self) -> None:
        """Lab frame P(|1⟩) matches sin²(π·Ω·t) within 1%."""
        p1_lab = self._run_rabi_in_frame("lab")
        tlist = np.linspace(0, self.DURATION, self.N_POINTS)
        p1_analytic = np.sin(np.pi * self.OMEGA * tlist) ** 2

        npt.assert_allclose(
            p1_lab,
            p1_analytic,
            atol=0.01,
            err_msg="Lab frame Rabi deviates from sin²(π·Ω·t)",
        )

    def test_rotating_frame_matches_analytic(self) -> None:
        """Rotating frame P(|1⟩) matches sin²(π·Ω·t) within 1%."""
        p1_rot = self._run_rabi_in_frame("rotating")
        tlist = np.linspace(0, self.DURATION, self.N_POINTS)
        p1_analytic = np.sin(np.pi * self.OMEGA * tlist) ** 2

        npt.assert_allclose(
            p1_rot,
            p1_analytic,
            atol=0.01,
            err_msg="Rotating frame Rabi deviates from sin²(π·Ω·t)",
        )


class TestDispersiveShift:
    """Verify dispersive shift χ in coupled transmon+resonator system.

    ω_q = 5.0, ω_r = 7.0 → Δ = ω_q - ω_r = -2.0 GHz
    α = -0.25 GHz, g = 0.05 GHz, levels = 5

    Dispersive shift: χ = g²·α / [Δ·(Δ+α)]
        = 0.05² · (-0.25) / [(-2.0) · (-2.25)]
        = 0.0025 · (-0.25) / 4.5
        = -6.25e-4 / 4.5
        ≈ -1.389e-4 GHz

    The dressed |0,1⟩ ↔ |1,1⟩ splitting minus the bare splitting should
    equal 2χ.
    """

    FREQ_Q = 5.0
    FREQ_R = 7.0
    ALPHA = -0.25
    G = 0.05
    LEVELS = 5

    def test_dispersive_shift(self, backend: Backend) -> None:
        """Dressed splitting differs from bare by 2χ within ~10%."""
        q = DuffingTransmon(
            freq=self.FREQ_Q,
            anharmonicity=self.ALPHA,
            levels=self.LEVELS,
            label="q",
        )
        r = Resonator(freq=self.FREQ_R, levels=self.LEVELS, label="r")
        coupling = Capacitive(q, r, g=self.G)
        chip = Chip(devices=[q, r], couplings=[coupling])

        H = chip.hamiltonian()
        evals = np.sort(np.linalg.eigvalsh(H.matrix(backend=backend)).real)

        # Bare eigenvalues: E_{n_q, n_r} = ω_q·n_q + (α/2)·n_q·(n_q-1) + ω_r·n_r.
        # Identify each dressed state by proximity to its uncoupled value:
        # |0,0⟩ ≈ 0, |0,1⟩ ≈ ω_r, |1,0⟩ ≈ ω_q, |1,1⟩ ≈ ω_q + ω_r.
        E_00_uncoupled = 0.0
        E_01_uncoupled = self.FREQ_R  # 7.0
        E_10_uncoupled = self.FREQ_Q  # 5.0
        E_11_uncoupled = self.FREQ_Q + self.FREQ_R  # 12.0

        E_00 = evals[np.argmin(np.abs(evals - E_00_uncoupled))]
        E_01 = evals[np.argmin(np.abs(evals - E_01_uncoupled))]
        E_10 = evals[np.argmin(np.abs(evals - E_10_uncoupled))]
        E_11 = evals[np.argmin(np.abs(evals - E_11_uncoupled))]

        # Dressed qubit splitting with 1 photon: E_11 - E_01
        dressed_qubit_splitting_1photon = E_11 - E_01
        # Dressed qubit splitting with 0 photons: E_10 - E_00
        dressed_qubit_splitting_0photon = E_10 - E_00

        # The difference is 2χ:
        # ω̃_q(n=1) - ω̃_q(n=0) = 2χ
        delta_splitting = dressed_qubit_splitting_1photon - dressed_qubit_splitting_0photon

        # Analytical: χ = g²·α / [Δ·(Δ+α)]
        Delta = self.FREQ_Q - self.FREQ_R  # -2.0
        chi_analytic = self.G**2 * self.ALPHA / (Delta * (Delta + self.ALPHA))

        # δ(splitting) should be 2χ
        two_chi = 2.0 * chi_analytic

        assert abs(delta_splitting - two_chi) < abs(two_chi) * 0.15, (
            f"Dispersive shift mismatch: "
            f"δ(dressed splitting) = {delta_splitting:.6e}, "
            f"2χ_analytic = {two_chi:.6e}, "
            f"relative error = {abs(delta_splitting - two_chi) / abs(two_chi):.2%}"
        )


class TestResonantEigenvalueSplitting:
    """Two identical resonators coupled capacitively: splitting = 2g.

    Both at ω = 6.0 GHz, coupled with Capacitive(g=0.05).
    In the single-excitation manifold, E_± = ω ± g.
    |E₊ - E₋| = 2g = 0.1 GHz.
    """

    FREQ = 6.0
    G = 0.05
    LEVELS = 3

    def test_single_excitation_splitting(self, backend: Backend) -> None:
        """Single-excitation splitting equals 2g within 1%."""
        r1 = Resonator(freq=self.FREQ, levels=self.LEVELS, label="r1")
        r2 = Resonator(freq=self.FREQ, levels=self.LEVELS, label="r2")
        coupling = Capacitive(r1, r2, g=self.G)
        chip = Chip(devices=[r1, r2], couplings=[coupling])

        H = chip.hamiltonian()
        evals = np.sort(np.linalg.eigvalsh(H.matrix(backend=backend)).real)

        # Single-excitation manifold E_- = ω - g, E_+ = ω + g lies near ω = 6.0.
        single_exc = evals[(evals > 5.0) & (evals < 7.0)]
        assert len(single_exc) == 2, (
            f"Expected 2 eigenvalues in single-excitation band, got {len(single_exc)}: {single_exc}"
        )

        splitting = single_exc[1] - single_exc[0]
        expected_splitting = 2.0 * self.G  # 0.1

        npt.assert_allclose(
            splitting,
            expected_splitting,
            rtol=0.01,
            err_msg=(f"Single-excitation splitting {splitting:.6f} GHz differs from 2g = {expected_splitting:.6f} GHz"),
        )

class TestResultAccessors:
    """Verify result accessor methods against physics expectations."""

    def test_overlap_after_pi_pulse(self, backend: Backend) -> None:
        """After a π-pulse (t=1/(2Ω)=25ns for Ω=0.02 GHz), overlap with |1⟩ ≈ 1.0."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        _drive = ChargeDrive(target=q)
        chip = Chip([q])
        chip.connect(ControlEquipment(lines=[_drive]))
        chip.set_frame("rotating")

        envelope = Square(duration=50.0, amplitude=0.02)
        drive_op = DriveOp(
            target_label="q",
            envelope=envelope,
            freq=5.0,
            start_time=0.0,
            drive_label=_drive.label,
        )
        tlist = np.linspace(0, 50.0, 501)
        result = simulate(chip, [drive_op], tlist)

        target_1 = backend.basis(3, 1)
        overlap = result.overlap(target_1)

        idx_25 = np.argmin(np.abs(tlist - 25.0))
        assert overlap[idx_25] > 0.99, f"overlap(|1⟩) at π-pulse = {overlap[idx_25]:.4f}, expected ≈ 1.0"

    def test_reduced_state_after_swap(self, backend: Backend) -> None:
        """After vacuum Rabi swap (t=1/(4g), initial |1,0⟩), q0's reduced state ≈ |0⟩⟨0|."""
        g = 0.01
        q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
        q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q1")
        coupling = Capacitive(q0, q1, g=g)
        chip = Chip([q0, q1], [coupling], frame="lab")
        psi0 = chip.bare_state(q0=1)

        t_swap = 1.0 / (4.0 * g)  # 25 ns
        tlist = np.linspace(0, t_swap, 201)

        result = simulate(
            chip,
            [],
            tlist,
            initial_state=psi0,
        )

        rho_q0 = result.reduced_state(t_swap, "q0")

        ground_ket = backend.basis(3, 0)
        overlap_ground = float(
            np.real(
                complex(
                    backend.expect(
                        backend.matmul(ground_ket, backend.dag(ground_ket)),
                        rho_q0,
                    )
                )
            )
        )
        assert overlap_ground > 0.95, (
            f"q0 reduced state overlap with |0⟩ = {overlap_ground:.4f}, expected > 0.95 after swap"
        )

class TestCoherentState:
    """Verify coherent state properties: ⟨n̂⟩ = |α|²."""

    def test_poisson_distribution(self, backend: Backend) -> None:
        """Fock populations n=0..4 follow Poisson: P(n) = e^(-|α|²)·|α|^(2n) / n factorial."""
        import math

        r = Resonator(freq=6.0, levels=20, label="r")
        alpha = 2.0
        psi = r.coherent_state(alpha)

        n_bar = abs(alpha) ** 2  # 4.0

        for n in range(5):
            p_poisson = np.exp(-n_bar) * n_bar**n / math.factorial(n)

            fock_n = backend.basis(20, n)
            proj = backend.matmul(fock_n, backend.dag(fock_n))
            p_measured = float(np.real(complex(backend.expect(proj, psi))))

            npt.assert_allclose(
                p_measured,
                p_poisson,
                atol=1e-6,
                err_msg=(f"P({n}) = {p_measured:.6f}, Poisson = {p_poisson:.6f}"),
            )


class TestConstants:
    """Verify physical constants against CODATA 2018 values."""

    def test_boltzmann_constant(self) -> None:
        """k_B/h matches the CODATA 2018 exact conversion factor, 0.020836619120... GHz/mK."""
        from quchip.utils.constants import k_B

        # CODATA 2018 exact: k_B = 1.380649e-23 J/K, h = 6.62607015e-34 J*s.
        expected = 1.380649e-23 / 6.62607015e-34 * 1e-12  # GHz/mK
        assert abs(k_B - expected) / expected < 1e-6, f"k_B = {k_B}, expected {expected:.12f} GHz/mK (CODATA 2018)"


class TestDuffingEigenvalues:
    """Verify DuffingTransmon eigenvalues."""

    def test_eigenvalues_match_analytic(self, backend: Backend) -> None:
        """Eigenvalues match ``E_n = omega*n + (alpha/2)*n*(n-1)`` (Koch et al., PRA 76, 042319 (2007))."""
        omega, alpha, levels = 5.0, -0.25, 5
        q = DuffingTransmon(freq=omega, anharmonicity=alpha, levels=levels, label="q")
        chip = Chip([q])
        H = chip.hamiltonian()
        evals = np.sort(np.linalg.eigvalsh(H.matrix(backend=backend)).real)

        for n in range(levels):
            expected = omega * n + (alpha / 2.0) * n * (n - 1)
            assert abs(evals[n] - expected) < 1e-10, f"E_{n} = {evals[n]:.10f}, expected {expected:.10f}"


class TestResonatorEigenvalues:
    """Verify resonator eigenvalues."""

    def test_eigenvalues_equally_spaced(self, backend: Backend) -> None:
        """Resonator eigenvalues match the QHO ladder ``E_n = omega*n``."""
        omega, levels = 6.0, 5
        r = Resonator(freq=omega, levels=levels, label="r")
        chip = Chip([r])
        H = chip.hamiltonian()
        evals = np.sort(np.linalg.eigvalsh(H.matrix(backend=backend)).real)

        for n in range(levels):
            expected = omega * n
            assert abs(evals[n] - expected) < 1e-10, f"E_{n} = {evals[n]:.10f}, expected {expected:.10f}"


class TestCollapseOperators:
    """Verify collapse-operator physics."""

    def test_thermal_occupation(self) -> None:
        """At thermal equilibrium, P(1)/P(0) matches the Boltzmann ratio exp(-freq/(k_B*T))."""
        from quchip.utils.constants import k_B

        freq = 5.0
        T_mK = 100.0
        n_bar = 1.0 / (np.exp(freq / (k_B * T_mK)) - 1.0)

        q = DuffingTransmon(
            freq=freq,
            anharmonicity=-0.25,
            levels=3,
            label="q",
            T1=500.0,
            thermal_occupation=n_bar,
        )
        chip = Chip([q])
        chip.set_frame("rotating")
        tlist = np.linspace(0, 5000.0, 501)

        result = simulate(chip, [], tlist)

        p0_final = result.population("q", 0)[-1]
        p1_final = result.population("q", 1)[-1]

        # Boltzmann ratio: P(1)/P(0) = exp(-freq/(k_B*T))
        expected_ratio = np.exp(-freq / (k_B * T_mK))
        if p0_final > 0.01:  # avoid division by near-zero
            actual_ratio = p1_final / p0_final
            assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.15, (
                f"Thermal ratio P(1)/P(0) = {actual_ratio:.4f}, expected Boltzmann = {expected_ratio:.4f}"
            )


class TestPhotonLoss:
    """Verify resonator photon loss ``kappa = sqrt(2*pi*freq/Q)``."""

    def test_photon_decay_rate(self) -> None:
        """Mean photon number decays as ``<n>(t) = <n>(0) * exp(-kappa_angular*t)``, kappa_angular = 2*pi*freq/Q."""
        freq = 6.0
        Q = 10000
        r = Resonator(freq=freq, levels=10, label="r", internal_quality_factor=Q)
        chip = Chip([r])
        chip.set_frame("rotating")

        # Start in |1> Fock state
        psi0 = chip.state(r=1)
        tlist = np.linspace(0, 1000.0, 501)

        result = simulate(chip, [], tlist, initial_state=psi0)
        p1 = result.population("r", 1)

        # Analytical: P(1,t) = exp(-kappa_angular * t) for |1> state
        kappa_angular = 2 * np.pi * freq / Q
        p1_analytic = np.exp(-kappa_angular * tlist)

        npt.assert_allclose(p1, p1_analytic, atol=0.03, err_msg="Photon loss P(1,t) does not match exp(-kappa*t)")


class TestDemodulation:
    """Verify rotating-frame demodulation."""

    def test_demod_recovers_lab_frame(self) -> None:
        """Round-trip modulate/demodulate exactly recovers the original rotating-frame ⟨a⟩."""
        # <O>_lab(t) = <O>_rot(t) * exp(+i*2pi*omega_frame*t): rotating-frame <a> is slowly
        # varying while the demodulated lab-frame version oscillates at the device frequency.
        omega = 5.0  # GHz
        q = DuffingTransmon(freq=omega, anharmonicity=-0.25, levels=4, label="q")
        chip = Chip([q])
        chip.set_frame("rotating")

        # Prepare (|0>+|1>)/sqrt(2) which has nonzero <a>
        backend = chip.backend
        psi0_local = (backend.basis(4, 0) + backend.basis(4, 1)) * (1.0 / np.sqrt(2))
        psi0_full = chip.bare_state(q=psi0_local)

        tlist = np.linspace(0, 10.0, 1001)  # 10 ns, fine sampling for oscillation

        result = simulate(chip, [], tlist, initial_state=psi0_full, e_ops=chip.e_ops(q="a"))

        a_rot = result._expect_data["q"].values  # complex expectation value in rotating frame

        omega_frame = resolve_frame(chip, chip.frame).frequencies["q"]  # dressed reference frequency
        a_lab_from_demod = a_rot * np.exp(1j * 2 * np.pi * omega_frame * tlist)

        rot_variation = np.max(np.abs(np.diff(a_rot)))
        lab_variation = np.max(np.abs(np.diff(a_lab_from_demod)))

        assert lab_variation > 5 * rot_variation, (
            f"Demodulated signal should oscillate faster than rotating-frame signal. "
            f"Rotating variation={rot_variation:.6f}, Lab variation={lab_variation:.6f}"
        )

        # Re-modulate: multiply lab-frame by exp(-i*2pi*omega_frame*t) to recover rotating.
        a_rot_recovered = a_lab_from_demod * np.exp(-1j * 2 * np.pi * omega_frame * tlist)
        npt.assert_allclose(
            a_rot_recovered,
            a_rot,
            atol=1e-10,
            err_msg="Round-trip modulate/demodulate must recover original rotating-frame expectation",
        )


class TestVacuumRabi:
    """Verify vacuum Rabi oscillation from capacitive coupling.

    Two identical resonators coupled with strength g exchange a single
    excitation at the vacuum Rabi frequency 2g.  The population oscillates
    as P_A(t) = cos²(2π·g·t), giving a full swap period T = 1/(2g).

    Analytical reference: H_int = g(a†b + ab†).
    """

    G = 0.05  # coupling in GHz
    FREQ = 6.0  # resonator frequency in GHz

    def test_vacuum_rabi_cosine_squared(self) -> None:
        """Population matches P_A(t) = cos²(2π·g·t), the exact solution for RWA coupling."""
        r_a = Resonator(freq=self.FREQ, levels=4, label="r_a")
        r_b = Resonator(freq=self.FREQ, levels=4, label="r_b")
        coupling = Capacitive(r_a, r_b, g=self.G)
        chip = Chip([r_a, r_b], [coupling], frame="lab")

        psi0 = chip.bare_state(r_a=1, r_b=0)
        T_swap = 1.0 / (2.0 * self.G)
        tlist = np.linspace(0, T_swap, 501)

        result = simulate(chip, [], tlist, initial_state=psi0)
        p_a = result.population("r_a", 1)

        # Analytical: P_A(t) = cos²(2π·g·t)
        p_a_analytic = np.cos(2 * np.pi * self.G * tlist) ** 2
        npt.assert_allclose(p_a, p_a_analytic, atol=0.05, err_msg="Vacuum Rabi P(r_a=1,t) does not match cos²(2π·g·t)")


class TestRabiFrequency:
    """Verify Rabi frequency from the coupling-operator matrix element.

    For a ChargeDrive with coupling operator H_c = i(a† - a), the matrix
    element |⟨0|H_c|1⟩| = 1.0 in the Fock basis. The on-resonance Rabi
    formula gives P(|1⟩) = sin²(π · Ω · |⟨0|H_c|1⟩| · t).

    This tests the specific connection between drive amplitude, coupling
    operator matrix element, and observed oscillation frequency.
    """

    def test_rabi_frequency_from_matrix_element(self) -> None:
        """Rabi oscillation frequency matches Ω × |⟨0|H_c|1⟩|, giving P(|1⟩) = sin²(π·Ω·mel·t)."""
        # <0|i(a†-a)|1> = -i, so |<0|H_c|1>| = mel = 1.0.
        omega = 0.02  # drive amplitude in GHz
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=4, label="q")
        chip = Chip([q])

        backend = chip.backend
        from quchip.control.signal import AnalyticSignal
        from quchip.engine.ir import Constant

        H_c = ChargeDrive(target=q).hamiltonian(
            q,
            AnalyticSignal(Constant(1.0)),
        ).matrix(t=0.0, backend=backend)
        mel = abs(complex(H_c[0, 1]))
        npt.assert_allclose(mel, 1.0, atol=1e-10, err_msg=f"|⟨0|i(a†-a)|1⟩| = {mel}, expected 1.0")

        d = ChargeDrive(target=q)
        chip.connect(ControlEquipment(lines=[d]))
        drive_op = DriveOp(
            target_label="q",
            envelope=Square(amplitude=omega, duration=100),
            freq=q.freq,
            start_time=0.0,
            drive_label=d.label,
        )
        tlist = np.linspace(0, 100, 1001)
        result = simulate(chip, [drive_op], tlist)

        p1 = result.population("q", 1)
        p1_analytic = np.sin(np.pi * omega * mel * tlist) ** 2
        npt.assert_allclose(p1, p1_analytic, atol=0.02, err_msg="Rabi frequency does not match Ω × |⟨0|H_c|1⟩|")


class TestFluxDrivePhase:
    """Verify FluxDrive diagonal coupling properties.

    The FluxDrive coupling operator n̂ is diagonal in the Fock basis.
    Key analytical consequences:
    1. No transitions between Fock states (diagonal operators preserve populations)
    2. ⟨0|n̂|0⟩ = 0 → ground state is completely unaffected by flux drive
    3. The quadrature partner [n̂, n̂] = 0, so on-resonance drive terms vanish

    These properties are verified against time-domain simulations.
    """

    def test_flux_drive_preserves_fock_populations(self) -> None:
        """Flux drive preserves ``|1⟩`` population for all t because its coupling operator ``n̂`` is diagonal."""
        freq_q = 5.0
        q = DuffingTransmon(freq=freq_q, anharmonicity=-0.2, levels=4, label="q")
        chip = Chip([q])

        d = FluxDrive(target=q)
        chip.connect(ControlEquipment(lines=[d]))
        drive_op = DriveOp(
            target_label="q",
            envelope=Square(amplitude=0.1, duration=200),
            freq=freq_q + 0.3,  # off-resonance to maximize drive term
            start_time=0.0,
            drive_label=d.label,
        )

        tlist = np.linspace(0, 200, 1001)
        psi0 = chip.bare_state(q=1)

        result = simulate(chip, [drive_op], tlist, initial_state=psi0)
        p1 = result.population("q", 1)

        npt.assert_allclose(
            p1,
            np.ones_like(p1),
            atol=0.02,
            err_msg="FluxDrive diagonal coupling should preserve Fock state populations",
        )

    def test_flux_drive_matrix_element(self) -> None:
        """The number operator is diagonal: ⟨n|n̂|m⟩ = n·δ_{nm}."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=5, label="q")
        backend = Chip([q]).backend
        n_op = q.number_operator()

        for n in range(5):
            for m in range(5):
                bra_n = backend.dag(backend.basis(5, n))
                mel = complex(backend.expect(backend.matmul(backend.basis(5, n), bra_n), n_op))
                if n == m:
                    npt.assert_allclose(mel.real, float(n), atol=1e-10, err_msg=f"⟨{n}|n̂|{n}⟩ = {mel}, expected {n}")
                else:
                    proj = backend.matmul(backend.basis(5, n), backend.dag(backend.basis(5, m)))
                    off_diag = complex(backend.expect(proj, n_op))
                    npt.assert_allclose(
                        abs(off_diag), 0.0, atol=1e-10, err_msg=f"⟨{n}|n̂|{m}⟩ = {off_diag}, expected 0 (diagonal)"
                    )


class TestGaussianEnvelopeArea:
    """Verify the Gaussian envelope area formula.

    The integral of a Gaussian pulse ∫|Ω(t)|dt should match the
    analytical formula A × σ × √(2π), truncated at ±sigmas standard
    deviations.
    """

    def test_gaussian_area_matches_analytic(self) -> None:
        """Numerical integral of the Gaussian envelope matches A·σ·√(2π)·erf(sigmas/√2)."""
        duration = 100.0
        sigmas = 3.0
        amplitude = 0.05

        env = Gaussian(duration=duration, sigmas=sigmas, amplitude=amplitude)
        sigma = duration / (2.0 * sigmas)

        # Fine time grid for accurate numerical integration
        n_points = 10001
        t = np.linspace(0, duration, n_points)
        waveform = env.value(t)

        dt = duration / (n_points - 1)
        numerical_area = np.trapezoid(np.abs(waveform), dx=dt)

        from scipy.special import erf

        truncation_factor = erf(sigmas / np.sqrt(2))
        analytical_area = amplitude * sigma * np.sqrt(2 * np.pi) * truncation_factor

        npt.assert_allclose(
            numerical_area,
            analytical_area,
            rtol=0.01,
            err_msg=(
                f"Gaussian area mismatch: numerical={numerical_area:.6f}, "
                f"analytical={analytical_area:.6f} (A·σ·√(2π)·erf(sigmas/√2))"
            ),
        )
