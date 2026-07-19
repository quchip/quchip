"""Domain model tests with analytical verification.

Every expected value is derived from physics formulas rather than running the
implementation first. Coverage includes device, coupling, and import-cleanliness
behavior.

Eigenvalue formulas:
    DuffingTransmon: E_n = ω·n + (α/2)·n·(n−1)
    Resonator:       E_n = ω·n
    Capacitive RWA:  ⟨1,0|H_int|0,1⟩ = g  (hopping matrix element)
"""

from __future__ import annotations

import numpy as np
import pytest

from quchip.backend.protocol import Backend
from quchip.chip.chip import Chip
from quchip.utils.labeling import reset_label_counters
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.devices.resonator import Resonator
from quchip.chip.couplings import Capacitive
from quchip.chip.rwa import apply_rwa_mask
from quchip.control.drive import ChargeDrive, FluxDrive
from quchip.utils.labeling import auto_label, resolve_label


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_labels():
    """Reset auto-label counters so tests are order-independent."""
    reset_label_counters()
    yield
    reset_label_counters()


# ---------------------------------------------------------------------------
# TestDuffingTransmon
# ---------------------------------------------------------------------------


class TestDuffingTransmon:
    """Analytical tests for the DuffingTransmon device."""

    def test_hamiltonian_shape(self) -> None:
        """hamiltonian() returns operator with correct dimensions."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4)
        H = q.hamiltonian()
        assert H.shape == (4, 4)

    def test_eigenvalues(self, backend: Backend) -> None:
        """Eigenvalues match E_n = ω·n + (α/2)·n·(n−1)."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4)
        H = q.hamiltonian()
        evals = backend.eigenenergies(H)
        # levels=4, freq=5.0, alpha=-0.25: E_0=0.0, E_1=5.0,
        # E_2=10.0+(-0.125)*2=9.75, E_3=15.0+(-0.125)*6=14.25
        expected = [0.0, 5.0, 9.75, 14.25]
        np.testing.assert_allclose(evals, expected, atol=1e-10)

    def test_lowering_operator(self, backend: Backend) -> None:
        """⟨0|a|1⟩ = 1.0 — standard Fock-space matrix element."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4)
        a = q.lowering_operator()
        a_full = a.full()
        assert abs(a_full[0, 1] - 1.0) < 1e-10

    def test_number_operator(self) -> None:
        """Diagonal elements of n̂ are [0, 1, 2, ...]."""
        levels = 5
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=levels)
        n = q.number_operator()
        diag = n.diag()
        expected = np.arange(levels, dtype=float)
        np.testing.assert_allclose(diag, expected, atol=1e-10)

    def test_backend_optional(self) -> None:
        """Calling hamiltonian() without backend resolves to default."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3)
        H = q.hamiltonian()
        assert H is not None
        assert H.shape == (3, 3)


# ---------------------------------------------------------------------------
# TestResonator
# ---------------------------------------------------------------------------


class TestResonator:
    """Analytical tests for the Resonator device."""

    def test_hamiltonian_eigenvalues(self, backend: Backend) -> None:
        """Eigenvalues match E_n = ω·n for freq=6.0, levels=5."""
        r = Resonator(freq=6.0, levels=5)
        H = r.hamiltonian()
        evals = backend.eigenenergies(H)
        expected = [0.0, 6.0, 12.0, 18.0, 24.0]
        np.testing.assert_allclose(evals, expected, atol=1e-10)

    def test_coherent_state(self, backend: Backend) -> None:
        """coherent_state(0.0) produces vacuum — overlap with |0⟩ ≈ 1.0."""
        r = Resonator(freq=6.0, levels=10)
        psi = r.coherent_state(0.0)
        ket_0 = backend.basis(10, 0)
        overlap = abs(backend.expect(ket_0 * backend.dag(ket_0), psi))
        assert abs(overlap - 1.0) < 1e-10

    def test_collapse_operators_with_quality_factor(self) -> None:
        """Q=10000 at f=5 GHz produces one collapse op with coefficient sqrt(2π·f/Q)."""
        # Convention lock-in (issue #66): Q is defined against the ordinary frequency, so
        # kappa = 2*pi*f/Q (rad/ns). The 2*pi is part of Q's physical definition, not a
        # units-boundary conversion, and must not move out of resonator.py.
        freq = 5.0
        Q = 10000.0
        r = Resonator(freq=freq, quality_factor=Q, levels=5)
        c_ops = r.collapse_operators()
        assert len(c_ops) == 1

        expected_sqrt_kappa = np.sqrt(2 * np.pi * freq / Q)
        c_full = c_ops[0].full()
        np.testing.assert_allclose(abs(c_full[0, 1]), expected_sqrt_kappa, atol=1e-10)

    def test_collapse_operators_without_quality_factor(self) -> None:
        """No quality_factor → empty list."""
        r = Resonator(freq=6.0, levels=5)
        c_ops = r.collapse_operators()
        assert c_ops == []


# ---------------------------------------------------------------------------
# TestCapacitive
# ---------------------------------------------------------------------------


class TestCapacitive:
    """Analytical tests for the Capacitive coupling."""

    def test_interaction_rwa(self, backend: Backend) -> None:
        """RWA: ⟨1,0|H_int|0,1⟩ = g (the hopping matrix element)."""
        # H_int^RWA = g*(a-dag⊗b + a⊗b-dag); <1,0|H_int|0,1> = g*<1|a-dag|0>*<0|b|1> = g*1*1 = g
        g = 0.02
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3)
        r = Resonator(freq=6.0, levels=5)
        c = Capacitive(q, r, g=g, rwa=True)
        H_int = apply_rwa_mask(
            c.interaction_hamiltonian(),
            dims=(q.levels, r.levels),
            labels=(q.label, r.label),
            keeps_band=c.rwa_keeps_band,
            backend=backend,
        )

        bra_10 = backend.tensor_states(backend.basis(3, 1), backend.basis(5, 0))
        ket_01 = backend.tensor_states(backend.basis(3, 0), backend.basis(5, 1))
        bra_00 = backend.tensor_states(backend.basis(3, 0), backend.basis(5, 0))
        ket_11 = backend.tensor_states(backend.basis(3, 1), backend.basis(5, 1))

        # bra†·H·ket collapses to a scalar in QuTiP.
        element = complex(backend.dag(bra_10) * H_int * ket_01)
        np.testing.assert_allclose(element.real, g, atol=1e-10)
        assert abs(element.imag) < 1e-10
        # The RWA mask removes the counter-rotating a b / a†b† band.
        assert abs(complex(backend.dag(bra_00) * H_int * ket_11)) < 1e-12

    def test_interaction_full(self, backend: Backend) -> None:
        """Full form: ⟨0,0|H_int|1,1⟩ ≠ 0 (counter-rotating ab term present)."""
        # H_int^full = g*(a+a-dag)⊗(b+b-dag); <0,0|H_int|1,1> = g*<0|(a+a-dag)|1>*<0|(b+b-dag)|1> = g*1*1 = g
        g = 0.02
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3)
        r = Resonator(freq=6.0, levels=5)
        c = Capacitive(q, r, g=g, rwa=False)
        H_int = c.interaction_hamiltonian()

        bra_00 = backend.tensor_states(backend.basis(3, 0), backend.basis(5, 0))
        ket_11 = backend.tensor_states(backend.basis(3, 1), backend.basis(5, 1))

        element = complex(backend.dag(bra_00) * H_int * ket_11)
        assert abs(element) > 1e-15, "Counter-rotating term should be non-zero"
        np.testing.assert_allclose(element.real, g, atol=1e-10)

    def test_coupling_strength_property(self) -> None:
        """c.coupling_strength == g."""
        g = 0.035
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3)
        r = Resonator(freq=6.0, levels=5)
        c = Capacitive(q, r, g=g)
        assert c.coupling_strength == g

    def test_chip_rwa_is_inherited_by_default(self) -> None:
        """Couplings and eligible drives inherit the chip-wide RWA policy."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=5, label="r")
        coupling = Capacitive(q, r, g=0.02)
        drive = ChargeDrive(target=q)
        chip = Chip([q, r], [coupling], rwa=False)

        assert chip.resolve_rwa(coupling) is False
        assert chip.resolve_rwa(drive) is False

    def test_object_rwa_override_wins_over_chip_default(self) -> None:
        """Explicit per-object RWA settings take precedence over the chip default."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=5, label="r")
        coupling = Capacitive(q, r, g=0.02, rwa=False)
        drive = ChargeDrive(target=q, rwa=False)
        chip = Chip([q, r], [coupling], rwa=True)

        assert chip.resolve_rwa(coupling) is False
        assert chip.resolve_rwa(drive) is False

    def test_accepts_label_strings_via_late_binding(self) -> None:
        """Capacitive(\"q0\", \"q1\", ...) resolves inside Chip and matches object form."""
        # Coupling constructors accept device objects or label strings; strings bind
        # to devices at Chip construction time.
        q_s = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
        r_s = Resonator(freq=6.0, levels=5, label="q1")
        c_str = Capacitive("q0", "q1", g=0.01)

        # Before the chip resolves, the coupling carries strings, not devices.
        assert c_str.is_resolved is False
        assert c_str.device_a_label == "q0"
        assert c_str.device_b_label == "q1"

        chip_str = Chip([q_s, r_s], couplings=[c_str])
        assert c_str.is_resolved is True
        assert c_str.device_a is q_s
        assert c_str.device_b is r_s

        q_o = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
        r_o = Resonator(freq=6.0, levels=5, label="q1")
        chip_obj = Chip([q_o, r_o], couplings=[Capacitive(q_o, r_o, g=0.01)])

        H_str = chip_str.hamiltonian()
        H_obj = chip_obj.hamiltonian()
        np.testing.assert_allclose(H_str.full(), H_obj.full(), atol=0.0)

    def test_accepts_mixed_string_and_object(self) -> None:
        """One label string + one device object resolves correctly."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
        r = Resonator(freq=6.0, levels=5, label="q1")
        c = Capacitive(q, "q1", g=0.01)

        assert c.is_resolved is False
        Chip([q, r], couplings=[c])
        assert c.is_resolved is True
        assert c.device_a is q
        assert c.device_b is r

    def test_rejects_non_label_non_device_types(self) -> None:
        """Integers, dicts, and other junk still get a clear TypeError."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
        with pytest.raises(TypeError, match="BaseDevice or label string"):
            Capacitive(5, q, g=0.01)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="BaseDevice or label string"):
            Capacitive(q, {"not": "a device"}, g=0.01)  # type: ignore[arg-type]

    def test_unknown_label_raises_at_chip_construction(self) -> None:
        """A label that matches no device on the chip is reported clearly."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
        r = Resonator(freq=6.0, levels=5, label="q1")
        c = Capacitive("q0", "nope", g=0.01)
        with pytest.raises(ValueError, match="nope"):
            Chip([q, r], couplings=[c])


# ---------------------------------------------------------------------------
# TestNoiseValidation — parameter guardrails
# ---------------------------------------------------------------------------


class TestNoiseValidation:
    """Verify constructor rejects invalid noise parameter combinations."""

    def test_T1_negative(self) -> None:
        """T1 <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="T1 must be positive"):
            DuffingTransmon(freq=5.0, anharmonicity=-0.25, T1=-1.0)

    def test_T1_zero(self) -> None:
        """T1 == 0 raises ValueError."""
        with pytest.raises(ValueError, match="T1 must be positive"):
            DuffingTransmon(freq=5.0, anharmonicity=-0.25, T1=0.0)

    def test_T2_negative(self) -> None:
        """T2 <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="T2 must be positive"):
            DuffingTransmon(freq=5.0, anharmonicity=-0.25, T1=1000.0, T2=-1.0)

    def test_T2_zero(self) -> None:
        """T2 == 0 raises ValueError."""
        with pytest.raises(ValueError, match="T2 must be positive"):
            DuffingTransmon(freq=5.0, anharmonicity=-0.25, T1=1000.0, T2=0.0)

    def test_T2_without_T1_accepted(self) -> None:
        """T2 without T1 is accepted — pure-dephasing collapse op uses gamma_phi = 1/T2."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, T2=500.0)
        assert q.T2 == 500.0
        assert q.T1 is None
        assert len(q.collapse_operators()) == 1

    def test_T2_exceeds_2T1(self) -> None:
        """T2 > 2·T1 raises ValueError."""
        with pytest.raises(ValueError, match="T2 must satisfy T2 <= 2\\*T1"):
            DuffingTransmon(freq=5.0, anharmonicity=-0.25, T1=1000.0, T2=2500.0)

    def test_thermal_population_negative(self) -> None:
        """thermal_population < 0 raises ValueError."""
        with pytest.raises(ValueError, match="thermal_population must be ≥ 0"):
            DuffingTransmon(freq=5.0, anharmonicity=-0.25, thermal_population=-0.1)

    def test_valid_noise_params_accepted(self) -> None:
        """Valid parameter combos are stored correctly."""
        q = DuffingTransmon(
            freq=5.0,
            anharmonicity=-0.25,
            T1=10_000.0,
            T2=8_000.0,
            thermal_population=0.02,
        )
        assert q.T1 == 10_000.0
        assert q.T2 == 8_000.0
        assert q.thermal_population == 0.02

    def test_noise_defaults_are_none(self) -> None:
        """Without noise kwargs, all noise fields are None."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25)
        assert q.T1 is None
        assert q.T2 is None
        assert q.thermal_population is None

    def test_resonator_accepts_noise_kwargs(self) -> None:
        """Resonator forwards noise kwargs to BaseDevice."""
        r = Resonator(freq=6.0, levels=5, T1=5000.0)
        assert r.T1 == 5000.0

    def test_T2_equals_2T1_accepted(self) -> None:
        """T2 == 2·T1 is the boundary — accepted, no pure dephasing."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, T1=1000.0, T2=2000.0)
        assert q.T2 == 2000.0

    def test_thermal_population_zero_accepted(self) -> None:
        """thermal_population == 0 is valid (zero-temp thermal channel)."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, thermal_population=0.0)
        assert q.thermal_population == 0.0


# ---------------------------------------------------------------------------
# TestCollapseOperators — channel assembly and rates
# ---------------------------------------------------------------------------


class TestCollapseOperators:
    """Verify collapse operator counts, rates, and formulas."""

    def test_no_noise_empty(self) -> None:
        """No noise params → empty collapse operator list."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3)
        assert q.collapse_operators() == []

    def test_T1_only_one_op(self) -> None:
        """T1 only → single relaxation operator √(1/T1)·a."""
        T1 = 10_000.0  # ns
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, T1=T1)
        c_ops = q.collapse_operators()
        assert len(c_ops) == 1

        # Check rate: (0,1) element = √(1/T1) · 1.0  (from lowering op)
        expected_rate = np.sqrt(1.0 / T1)
        c_full = c_ops[0].full()
        np.testing.assert_allclose(abs(c_full[0, 1]), expected_rate, atol=1e-12)

    def test_T1_T2_two_ops(self) -> None:
        """T1 + T2 (with γ_φ > 0) → relaxation + dephasing = 2 ops."""
        T1 = 10_000.0
        T2 = 5_000.0  # T2 < 2·T1, so γ_φ > 0
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, T1=T1, T2=T2)
        c_ops = q.collapse_operators()
        assert len(c_ops) == 2

        # Dephasing rate: γ_φ = 1/T2 - 1/(2·T1)
        gamma_phi = 1.0 / T2 - 1.0 / (2.0 * T1)
        assert gamma_phi > 0
        # Second op is √(2·γ_φ) · n̂ — diagonal with (1,1) element = √(2·γ_φ) · 1.
        # The factor 2 makes the 0–1 coherence decay at γ_φ (not γ_φ/2), so the
        # input T2 equals the resulting coherence time.
        dephasing_full = c_ops[1].full()
        np.testing.assert_allclose(abs(dephasing_full[1, 1]), np.sqrt(2.0 * gamma_phi), atol=1e-12)

    def test_T2_equals_2T1_no_dephasing(self) -> None:
        """T2 == 2·T1 → γ_φ = 0 → only T1 relaxation (1 op)."""
        T1 = 10_000.0
        T2 = 2 * T1
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, T1=T1, T2=T2)
        c_ops = q.collapse_operators()
        assert len(c_ops) == 1  # Only the T1 relaxation op

    def test_thermal_only_with_nbar(self) -> None:
        """thermal_population > 0 without T1 → downward + upward = 2 ops."""
        n_bar = 0.05
        q = DuffingTransmon(
            freq=5.0,
            anharmonicity=-0.25,
            levels=3,
            thermal_population=n_bar,
        )
        c_ops = q.collapse_operators()
        assert len(c_ops) == 2  # downward (a) + upward (a†)

        # Default γ=1.0: downward rate √(n̄+1), upward rate √n̄
        down_full = c_ops[0].full()
        np.testing.assert_allclose(
            abs(down_full[0, 1]),
            np.sqrt(n_bar + 1),
            atol=1e-12,
        )
        up_full = c_ops[1].full()
        np.testing.assert_allclose(
            abs(up_full[1, 0]),
            np.sqrt(n_bar),
            atol=1e-12,
        )

    def test_thermal_zero_nbar_one_op(self) -> None:
        """thermal_population == 0 → only downward channel (1 op)."""
        q = DuffingTransmon(
            freq=5.0,
            anharmonicity=-0.25,
            levels=3,
            thermal_population=0.0,
        )
        c_ops = q.collapse_operators()
        assert len(c_ops) == 1  # Only downward: √(γ·(0+1)) · a

    def test_T1_thermal_foldin(self) -> None:
        """T1 + thermal_population → fold-in with γ=1/T1."""
        T1 = 10_000.0
        n_bar = 0.05
        q = DuffingTransmon(
            freq=5.0,
            anharmonicity=-0.25,
            levels=3,
            T1=T1,
            thermal_population=n_bar,
        )
        c_ops = q.collapse_operators()

        # 2 ops: downward √((n̄+1)/T1)·a  +  upward √(n̄/T1)·a†
        assert len(c_ops) == 2
        gamma = 1.0 / T1
        down_full = c_ops[0].full()
        np.testing.assert_allclose(
            abs(down_full[0, 1]),
            np.sqrt(gamma * (n_bar + 1)),
            atol=1e-12,
        )
        up_full = c_ops[1].full()
        np.testing.assert_allclose(
            abs(up_full[1, 0]),
            np.sqrt(gamma * n_bar),
            atol=1e-12,
        )

    def test_T1_thermal_zero_nbar_foldin(self) -> None:
        """T1 + thermal_population=0 → fold-in, only downward (1 op)."""
        T1 = 10_000.0
        q = DuffingTransmon(
            freq=5.0,
            anharmonicity=-0.25,
            levels=3,
            T1=T1,
            thermal_population=0.0,
        )
        c_ops = q.collapse_operators()
        assert len(c_ops) == 1
        gamma = 1.0 / T1
        down_full = c_ops[0].full()
        np.testing.assert_allclose(
            abs(down_full[0, 1]),
            np.sqrt(gamma * 1.0),
            atol=1e-12,
        )

    def test_resonator_Q_plus_T1(self) -> None:
        """Resonator with quality_factor AND T1 → Q-loss + T1 relaxation."""
        freq = 6.0
        Q = 1e4
        T1 = 50_000.0
        r = Resonator(freq=freq, quality_factor=Q, levels=5, T1=T1)
        c_ops = r.collapse_operators()
        # BaseDevice T1 produces 1 op, Resonator Q produces 1 op → 2 total
        assert len(c_ops) == 2

        # First: T1 relaxation from BaseDevice
        expected_t1_rate = np.sqrt(1.0 / T1)
        c0_full = c_ops[0].full()
        np.testing.assert_allclose(abs(c0_full[0, 1]), expected_t1_rate, atol=1e-12)

        # Second: Q-based photon loss from Resonator (kappa = sqrt(2*pi*freq/Q))
        kappa = np.sqrt(2 * np.pi * freq / Q)
        c1_full = c_ops[1].full()
        np.testing.assert_allclose(abs(c1_full[0, 1]), kappa, atol=1e-12)

    def test_resonator_Q_no_noise(self) -> None:
        """Resonator Q-factor loss is produced without additional noise terms."""
        freq = 6.0
        Q = 1e4
        r = Resonator(freq=freq, quality_factor=Q, levels=5)
        c_ops = r.collapse_operators()
        assert len(c_ops) == 1  # Only Q-based loss


# ---------------------------------------------------------------------------
# TestLabeling — utils/labeling shared auto-label machinery
# ---------------------------------------------------------------------------


def test_auto_label_increments_per_prefix():
    """Each prefix keeps its own counter, independent of other prefixes."""
    reset_label_counters()
    assert auto_label("charge") == "charge_0"
    assert auto_label("charge") == "charge_1"
    assert auto_label("flux") == "flux_0"
    assert auto_label("charge") == "charge_2"


def test_resolve_label_passes_through_strings():
    """A string label resolves to itself."""
    assert resolve_label("q0") == "q0"


def test_resolve_label_extracts_dot_label():
    """An object with a ``.label`` attribute resolves to that label."""
    class FakeDevice:
        label = "q0"
    assert resolve_label(FakeDevice()) == "q0"


def test_resolve_label_rejects_unlabeled_objects():
    """An object without a label raises TypeError."""
    with pytest.raises(TypeError, match="label"):
        resolve_label(42)


def test_coupling_auto_labels():
    """Successive Capacitive couplings get incrementing auto-labels."""
    reset_label_counters()
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3)
    r = Resonator(freq=7.0, levels=5)
    c = Capacitive(q, r, g=0.02)
    assert c.label == "cap_0"
    c2 = Capacitive(q, r, g=0.01)
    assert c2.label == "cap_1"


def test_coupling_explicit_label():
    """An explicit label overrides auto-labeling."""
    reset_label_counters()
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3)
    r = Resonator(freq=7.0, levels=5)
    c = Capacitive(q, r, g=0.02, label="my_cap")
    assert c.label == "my_cap"


def test_drive_auto_labels():
    """Drive auto-labels increment per subclass prefix."""
    reset_label_counters()
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3)
    d1 = ChargeDrive(target=q)
    d2 = ChargeDrive(target=q)
    d3 = FluxDrive(target=q)
    assert d1.label == "charge_0"
    assert d2.label == "charge_1"
    assert d3.label == "flux_0"


def test_drive_explicit_label():
    """An explicit label overrides drive auto-labeling."""
    reset_label_counters()
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3)
    d = ChargeDrive(target=q, label="my_drive")
    assert d.label == "my_drive"
