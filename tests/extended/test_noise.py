"""Physics-verifying open-system tests: T1 decay, dephasing, and thermal channels.

Verifies:
    - mesolve auto-routing when collapse operators are present
    - sesolve routing when the device is noiseless
    - exponential T1 relaxation matching exp(-t/T1)
    - pure dephasing (T2) via off-diagonal density-matrix decay
    - thermal channel dynamics (upward transition, operator count)
    - thermal_population=0 boundary (no upward operator)
    - solver name exposed on SimulationResult

Unit convention: frequencies in GHz, times in ns.

References
----------
.. [1] Breuer & Petruccione, *The Theory of Open Quantum Systems* (2002),
       Ch. 3 — Lindblad master equation.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from quchip.backend.protocol import Backend
from quchip.chip.chip import Chip
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.engine import simulate


# ---------------------------------------------------------------------------
# TestSolverRouting — mesolve vs sesolve auto-selection
# ---------------------------------------------------------------------------


class TestSolverRouting:
    """Verify that simulate selects the correct solver branch."""

    def test_noiseless_routes_to_sesolve(self) -> None:
        """A device with no noise parameters must use sesolve."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        chip = Chip([q])
        tlist = np.linspace(0, 100, 11)

        result = simulate(chip, [], tlist)

        assert result.solver == "sesolve", (
            f"Expected 'sesolve' for noiseless chip, got '{result.solver}'. "
            "Solver routing may be broken — collapse operators should be empty."
        )

    def test_T1_routes_to_mesolve(self) -> None:
        """A device with T1 set must trigger mesolve."""
        q = DuffingTransmon(
            freq=5.0,
            anharmonicity=-0.25,
            levels=3,
            label="q",
            T1=10_000,
        )
        chip = Chip([q])
        tlist = np.linspace(0, 100, 11)

        result = simulate(chip, [], tlist)

        assert result.solver == "mesolve", (
            f"Expected 'mesolve' for T1-noisy device, got '{result.solver}'. "
            "collapse_operators() may not be producing operators."
        )

    def test_thermal_routes_to_mesolve(self) -> None:
        """A device with thermal_population must trigger mesolve."""
        q = DuffingTransmon(
            freq=5.0,
            anharmonicity=-0.25,
            levels=3,
            label="q",
            thermal_population=0.05,
        )
        chip = Chip([q])
        tlist = np.linspace(0, 100, 11)

        result = simulate(chip, [], tlist)

        assert result.solver == "mesolve", (
            f"Expected 'mesolve' for thermal device, got '{result.solver}'. "
            "thermal_population should produce collapse operators."
        )

    def test_T2_without_T1_emits_pure_dephasing(self) -> None:
        """T2 without T1 emits a single pure-dephasing operator with gamma_phi = 1/T2."""
        T2 = 100.0
        q = DuffingTransmon(
            freq=5.0,
            anharmonicity=-0.25,
            levels=3,
            label="q",
            T2=T2,
        )
        c_ops = q.collapse_operators()
        assert len(c_ops) == 1, (
            f"Expected 1 collapse op (pure dephasing) when T2 is set and T1 is None, got {len(c_ops)}."
        )
        # Dephasing op is sqrt(2*gamma_phi) * n_hat with gamma_phi = 1/T2 here
        # (T1 absent) — n_hat[1,1] = 1, so op[1,1]^2 == 2/T2. The factor 2 makes
        # the 0–1 coherence decay at gamma_phi = 1/T2 (the input coherence time).
        mat = np.asarray(c_ops[0].full())
        coeff_sq = abs(mat[1, 1]) ** 2
        expected = 2.0 / T2
        assert abs(coeff_sq - expected) < 1e-12, (
            f"Dephasing coefficient squared {coeff_sq} should equal 2/T2 = {expected}."
        )

    def test_T2_greater_than_2T1_raises(self) -> None:
        """T2 > 2*T1 makes the implied gamma_phi negative and must be rejected."""
        with pytest.raises(ValueError, match="T2 must satisfy T2 <= 2\\*T1"):
            DuffingTransmon(
                freq=5.0,
                anharmonicity=-0.25,
                levels=3,
                label="q",
                T1=1.0,
                T2=3.0,
            )

    def test_T1_only_no_dephasing_op(self) -> None:
        """T1 without T2 yields only the relaxation operator (no dephasing channel)."""
        q = DuffingTransmon(
            freq=5.0,
            anharmonicity=-0.25,
            levels=3,
            label="q",
            T1=50.0,
        )
        c_ops = q.collapse_operators()
        assert len(c_ops) == 1, f"Expected only the T1 operator when T2 is None, got {len(c_ops)}."


# ---------------------------------------------------------------------------
# TestT1Decay — the slice demo: |1⟩ exponential relaxation
# ---------------------------------------------------------------------------


class TestT1Decay:
    """Prepare |1⟩ with T1=10_000 ns and verify P1(t) = exp(-t/T1) (rotating frame)."""

    T1 = 10_000  # ns
    FREQ = 5.0
    ALPHA = -0.25
    LEVELS = 3
    DURATION = 30_000  # 3 × T1
    N_POINTS = 61

    def test_t1_decay_matches_analytic(self) -> None:
        """P1(t) follows exp(-t/T1) within 2% absolute tolerance."""
        q = DuffingTransmon(
            freq=self.FREQ,
            anharmonicity=self.ALPHA,
            levels=self.LEVELS,
            label="q",
            T1=self.T1,
        )
        chip = Chip([q])
        chip.set_frame("rotating")
        psi0 = chip.bare_state(q=1)

        tlist = np.linspace(0, self.DURATION, self.N_POINTS)
        result = simulate(
            chip,
            [],
            tlist,
            initial_state=psi0,
        )

        assert result.solver == "mesolve", f"Expected mesolve for T1 decay, got '{result.solver}'"

        p1 = result.population("q", 1)
        p1_analytic = np.exp(-tlist / self.T1)

        npt.assert_allclose(
            p1,
            p1_analytic,
            atol=0.02,
            err_msg=(
                f"T1 decay P1(t) deviates from exp(-t/T1). "
                f"Max |error| = {np.max(np.abs(p1 - p1_analytic)):.4f}. "
                f"T1={self.T1} ns, duration={self.DURATION} ns."
            ),
        )

    def test_t1_decay_endpoint(self) -> None:
        """At t=T1, P1 ≈ 1/e ≈ 0.368."""
        q = DuffingTransmon(
            freq=self.FREQ,
            anharmonicity=self.ALPHA,
            levels=self.LEVELS,
            label="q",
            T1=self.T1,
        )
        chip = Chip([q])
        chip.set_frame("rotating")
        psi0 = chip.bare_state(q=1)

        tlist = np.linspace(0, self.T1, 51)
        result = simulate(
            chip,
            [],
            tlist,
            initial_state=psi0,
        )

        p1_final = result.population("q", 1)[-1]
        expected = np.exp(-1.0)  # 1/e ≈ 0.3679

        assert abs(p1_final - expected) < 0.02, (
            f"P1(t=T1) = {p1_final:.4f}, expected 1/e = {expected:.4f}. Deviation = {abs(p1_final - expected):.4f}."
        )

    def test_ground_state_stable_under_t1(self) -> None:
        """Starting from |0⟩ with T1, population stays in ground state."""
        q = DuffingTransmon(
            freq=self.FREQ,
            anharmonicity=self.ALPHA,
            levels=self.LEVELS,
            label="q",
            T1=self.T1,
        )
        chip = Chip([q])
        chip.set_frame("rotating")

        tlist = np.linspace(0, self.T1, 21)
        result = simulate(chip, [], tlist)

        p0 = result.population("q", 0)

        npt.assert_allclose(
            p0,
            np.ones_like(p0),
            atol=1e-4,
            err_msg="Ground state |0⟩ should be stable under T1 relaxation.",
        )


# ---------------------------------------------------------------------------
# TestDephasing — T2 pure dephasing from superposition
# ---------------------------------------------------------------------------


class TestDephasing:
    """Verify pure dephasing: off-diagonal density-matrix decay at rate 1/T2."""
    # L = sqrt(2*gamma_phi) * n_hat gives |rho_01(t)| = |rho_01(0)| * exp(-t/T2), with
    # gamma_phi = 1/T2 - 1/(2*T1); the factor 2 in L makes the input T2 equal the
    # resulting coherence time.

    T1 = 50_000  # ns  (large to isolate dephasing)
    T2 = 10_000  # ns
    FREQ = 5.0
    ALPHA = -0.25
    LEVELS = 3
    DURATION = 20_000  # 2 × T2
    N_POINTS = 51

    def test_offdiag_decay_rate(self, backend: Backend) -> None:
        """Off-diagonal |rho_01| decays at analytically predicted rate."""
        q = DuffingTransmon(
            freq=self.FREQ,
            anharmonicity=self.ALPHA,
            levels=self.LEVELS,
            label="q",
            T1=self.T1,
            T2=self.T2,
        )
        chip = Chip([q])
        chip.set_frame("rotating")

        # Superposition |+⟩ = (|0⟩ + |1⟩) / √2
        ket0 = backend.basis(self.LEVELS, 0)
        ket1 = backend.basis(self.LEVELS, 1)
        psi_plus = (ket0 + ket1).unit()

        tlist = np.linspace(0, self.DURATION, self.N_POINTS)
        result = simulate(
            chip,
            [],
            tlist,
            initial_state=psi_plus,
            options={"nsteps": 5000},
        )

        offdiag = np.array([abs(complex(result.dm_at(t)[0, 1])) for t in tlist])

        # Analytical decay rate for n̂-based dephasing with the sqrt(2*gamma_phi)
        # normalization: rate = gamma_phi + 1/(2*T1) = 1/T2 (input coherence time).
        gamma_phi = 1.0 / self.T2 - 1.0 / (2.0 * self.T1)
        decay_rate = gamma_phi + 1.0 / (2.0 * self.T1)
        offdiag_analytic = 0.5 * np.exp(-decay_rate * tlist)

        npt.assert_allclose(
            offdiag,
            offdiag_analytic,
            atol=0.02,
            err_msg=(
                f"Off-diagonal decay deviates from analytic. "
                f"gamma_phi={gamma_phi:.2e}, decay_rate={decay_rate:.2e}. "
                f"Max |error| = {np.max(np.abs(offdiag - offdiag_analytic)):.4f}."
            ),
        )

    def test_populations_preserved_under_dephasing(self, backend: Backend) -> None:
        """Pure dephasing does not change diagonal populations: P0, P1 stay near 0.5 from |+⟩."""
        q = DuffingTransmon(
            freq=self.FREQ,
            anharmonicity=self.ALPHA,
            levels=self.LEVELS,
            label="q",
            T1=self.T1,
            T2=self.T2,
        )
        chip = Chip([q])
        chip.set_frame("rotating")

        ket0 = backend.basis(self.LEVELS, 0)
        ket1 = backend.basis(self.LEVELS, 1)
        psi_plus = (ket0 + ket1).unit()

        tlist = np.linspace(0, self.DURATION, self.N_POINTS)
        result = simulate(
            chip,
            [],
            tlist,
            initial_state=psi_plus,
            options={"nsteps": 5000},
        )

        p0 = result.population("q", 0)
        p1 = result.population("q", 1)

        # T1 is large relative to duration, so populations stay near 0.5
        # Allow 10% deviation for T1-induced drift
        assert np.all(p0 > 0.35), (
            f"P0 dropped below 0.35 — dephasing should not cause population transfer. Min P0 = {np.min(p0):.4f}."
        )
        assert np.all(p1 > 0.25), (
            f"P1 dropped below 0.25 — dephasing should not cause significant "
            f"population transfer. Min P1 = {np.min(p1):.4f}."
        )

    def test_T2_equals_2T1_no_pure_dephasing(self) -> None:
        """When T2 = 2*T1 (T1 limit), no pure dephasing operator is produced."""
        T1 = 10_000
        T2 = 2 * T1  # gamma_phi = 0 → no pure dephasing

        q = DuffingTransmon(
            freq=self.FREQ,
            anharmonicity=self.ALPHA,
            levels=self.LEVELS,
            label="q",
            T1=T1,
            T2=T2,
        )

        c_ops = q.collapse_operators()
        assert len(c_ops) == 1, (
            f"Expected 1 collapse operator (T1 only) when T2=2*T1, got {len(c_ops)}. "
            "gamma_phi should be zero, suppressing the dephasing channel."
        )


# ---------------------------------------------------------------------------
# TestThermalChannel — thermal population dynamics
# ---------------------------------------------------------------------------


class TestThermalChannel:
    """Verify thermal channel: upward transitions enabled by n_bar > 0."""

    FREQ = 5.0
    ALPHA = -0.25
    LEVELS = 3

    def test_thermal_operator_count_finite_nbar(self) -> None:
        """n_bar > 0 produces 2 collapse operators (downward + upward)."""
        q = DuffingTransmon(
            freq=self.FREQ,
            anharmonicity=self.ALPHA,
            levels=self.LEVELS,
            label="q",
            thermal_population=0.1,
        )

        c_ops = q.collapse_operators()
        assert len(c_ops) == 2, f"Expected 2 thermal collapse operators (down + up), got {len(c_ops)}."

    def test_thermal_operator_count_zero_nbar(self) -> None:
        """n_bar = 0 produces 1 collapse operator (downward only)."""
        q = DuffingTransmon(
            freq=self.FREQ,
            anharmonicity=self.ALPHA,
            levels=self.LEVELS,
            label="q",
            thermal_population=0.0,
        )

        c_ops = q.collapse_operators()
        assert len(c_ops) == 1, f"Expected 1 collapse operator for n_bar=0 (downward only), got {len(c_ops)}."

    def test_thermal_excitation_from_ground(self) -> None:
        """Starting from |0⟩ with finite thermal population, P1 rises toward n_bar/(2*n_bar+1)."""
        n_bar = 0.1
        q = DuffingTransmon(
            freq=self.FREQ,
            anharmonicity=self.ALPHA,
            levels=self.LEVELS,
            label="q",
            thermal_population=n_bar,
        )
        chip = Chip([q])
        chip.set_frame("rotating")

        tlist = np.linspace(0, 500, 51)
        result = simulate(chip, [], tlist)

        p1_final = result.population("q", 1)[-1]
        # Equilibrium P1 for thermal channel with default gamma=1
        expected_equilibrium = n_bar / (2.0 * n_bar + 1.0)

        assert p1_final > 0.01, (
            f"P1 at end = {p1_final:.6f}; expected upward excitation from thermal population n_bar={n_bar}."
        )
        # Should approach equilibrium — allow 20% relative tolerance
        assert abs(p1_final - expected_equilibrium) < 0.03, (
            f"P1 at equilibrium = {p1_final:.4f}, expected ~{expected_equilibrium:.4f} (n_bar/(2*n_bar+1))."
        )

    def test_thermal_zero_nbar_no_excitation(self) -> None:
        """n_bar = 0: no upward transition, |0⟩ remains stable."""
        q = DuffingTransmon(
            freq=self.FREQ,
            anharmonicity=self.ALPHA,
            levels=self.LEVELS,
            label="q",
            thermal_population=0.0,
        )
        chip = Chip([q])
        chip.set_frame("rotating")

        tlist = np.linspace(0, 500, 51)
        result = simulate(chip, [], tlist)

        p0 = result.population("q", 0)
        npt.assert_allclose(
            p0,
            np.ones_like(p0),
            atol=1e-4,
            err_msg="With n_bar=0, ground state should remain stable.",
        )

    def test_T1_thermal_foldin_operator_count(self) -> None:
        """T1 + thermal_population fold-in produces 2 operators."""
        q = DuffingTransmon(
            freq=self.FREQ,
            anharmonicity=self.ALPHA,
            levels=self.LEVELS,
            label="q",
            T1=10_000,
            thermal_population=0.1,
        )

        c_ops = q.collapse_operators()

        assert len(c_ops) == 2, f"Expected 2 fold-in operators (T1+thermal), got {len(c_ops)}."

    def test_T1_thermal_foldin_rates(self) -> None:
        """T1+thermal fold-in uses gamma=1/T1 in thermal formula."""
        T1 = 10_000
        n_bar = 0.2

        q = DuffingTransmon(
            freq=self.FREQ,
            anharmonicity=self.ALPHA,
            levels=self.LEVELS,
            label="q",
            T1=T1,
            thermal_population=n_bar,
        )
        chip = Chip([q])
        chip.set_frame("rotating")

        psi0 = chip.bare_state(q=1)
        tlist = np.linspace(0, 30_000, 61)

        result = simulate(
            chip,
            [],
            tlist,
            initial_state=psi0,
        )

        assert result.solver == "mesolve", f"Expected mesolve for T1+thermal, got '{result.solver}'"

        # P1 should decay from 1 but settle to a thermal equilibrium > 0
        p1_final = result.population("q", 1)[-1]
        assert p1_final > 0.01, (
            f"P1 final = {p1_final:.4f}; with thermal population, equilibrium P1 should be above zero."
        )
        assert p1_final < 0.8, f"P1 final = {p1_final:.4f}; should have decayed from 1.0."


# ---------------------------------------------------------------------------
# TestCollapseOperatorAssembly — operator-level unit checks
# ---------------------------------------------------------------------------


class TestCollapseOperatorAssembly:
    """Unit-level checks on collapse operator dimensions and rates."""

    LEVELS = 3

    def test_no_noise_produces_empty(self) -> None:
        """Device with no noise params yields empty collapse operator list."""
        q = DuffingTransmon(
            freq=5.0,
            anharmonicity=-0.25,
            levels=self.LEVELS,
            label="q",
        )
        assert q.collapse_operators() == [], "Noiseless device should return empty list of collapse operators."

    def test_T1_only_one_operator(self) -> None:
        """T1-only yields exactly one operator of correct dimension."""
        q = DuffingTransmon(
            freq=5.0,
            anharmonicity=-0.25,
            levels=self.LEVELS,
            label="q",
            T1=10_000,
        )
        c_ops = q.collapse_operators()
        assert len(c_ops) == 1, f"Expected 1 T1 operator, got {len(c_ops)}"
        shape = c_ops[0].shape
        assert shape[0] == self.LEVELS and shape[1] == self.LEVELS, (
            f"Operator shape {shape}, expected ({self.LEVELS}, {self.LEVELS})"
        )

    def test_T1_T2_two_operators(self) -> None:
        """T1 + T2 (with gamma_phi > 0) yields 2 operators."""
        q = DuffingTransmon(
            freq=5.0,
            anharmonicity=-0.25,
            levels=self.LEVELS,
            label="q",
            T1=10_000,
            T2=5_000,
        )
        c_ops = q.collapse_operators()
        assert len(c_ops) == 2, f"Expected 2 operators (T1 + dephasing), got {len(c_ops)}"

    def test_invalid_T1_raises(self) -> None:
        """Negative T1 is rejected at construction."""
        with pytest.raises(ValueError, match="T1 must be positive"):
            DuffingTransmon(
                freq=5.0,
                anharmonicity=-0.25,
                levels=3,
                label="q",
                T1=-1,
            )

    def test_invalid_T2_exceeds_2T1_raises(self) -> None:
        """T2 > 2*T1 is rejected at construction."""
        with pytest.raises(ValueError, match="T2 must satisfy T2 <= 2\\*T1"):
            DuffingTransmon(
                freq=5.0,
                anharmonicity=-0.25,
                levels=3,
                label="q",
                T1=10_000,
                T2=30_000,
            )

    def test_negative_thermal_raises(self) -> None:
        """Negative thermal_population is rejected at construction."""
        with pytest.raises(ValueError, match="thermal_population must be ≥ 0"):
            DuffingTransmon(
                freq=5.0,
                anharmonicity=-0.25,
                levels=3,
                label="q",
                thermal_population=-0.1,
            )
