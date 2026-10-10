"""Domain model tests with analytical verification.

The formulas below define the reference energies and coupling matrix element.

Eigenvalue formulas:
    DuffingTransmon: E_n = ω·n + (α/2)·n·(n−1)
    Resonator:       E_n = ω·n
    Capacitive RWA:  ⟨1,0|H_int|0,1⟩ = g  (hopping matrix element)
"""

from __future__ import annotations

from quchip.approximations import RWA

import numpy as np
import pytest

from quchip.backend.protocol import Backend
from quchip.utils.labeling import reset_label_counters
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.declarative.expr import materialize_expr
from quchip.devices.resonator import Resonator
from quchip.chip.couplings import Capacitive
from quchip.engine.approximations import apply_operator_band_filter
from quchip.utils.labeling import auto_label, resolve_label


pytestmark = pytest.mark.unit


class TestCapacitive:
    """Analytical tests for the Capacitive coupling."""

    def test_interaction_rwa(self, backend: Backend) -> None:
        """RWA: ⟨1,0|H_int|0,1⟩ = g (the hopping matrix element)."""
        # H_int^RWA = g*(a-dag⊗b + a⊗b-dag); <1,0|H_int|0,1> = g*<1|a-dag|0>*<0|b|1> = g*1*1 = g
        g = 0.02
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3)
        r = Resonator(freq=6.0, levels=5)
        c = Capacitive(q, r, g=g)
        H_int = apply_operator_band_filter(
            materialize_expr(c.interaction_hamiltonian(), backend),
            dims=(q.levels, r.levels),
            labels=(q.label, r.label),
            keeps_band=lambda first, second: RWA().keeps_operator_band((first, second)),
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
        """Full form: ⟨0,0|H_int|1,1⟩ = -g (counter-rotating ab term present)."""
        # H_int^full = g*i(a-a-dag)⊗i(b-b-dag); <0,0|H_int|1,1> = g*<0|i(a-a-dag)|1>*<0|i(b-b-dag)|1> = g*i*i = -g
        g = 0.02
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3)
        r = Resonator(freq=6.0, levels=5)
        c = Capacitive(q, r, g=g)
        H_int = materialize_expr(c.interaction_hamiltonian(), backend)

        bra_00 = backend.tensor_states(backend.basis(3, 0), backend.basis(5, 0))
        ket_11 = backend.tensor_states(backend.basis(3, 1), backend.basis(5, 1))

        element = complex(backend.dag(bra_00) * H_int * ket_11)
        assert abs(element) > 1e-15, "Counter-rotating term should be non-zero"
        np.testing.assert_allclose(element.real, -g, atol=1e-10)


class TestCollapseOperators:
    """Verify collapse operator counts, rates, and formulas."""


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


    def test_thermal_only_with_nbar(self) -> None:
        """thermal_occupation > 0 without T1 → downward + upward = 2 ops."""
        n_bar = 0.05
        q = DuffingTransmon(
            freq=5.0,
            anharmonicity=-0.25,
            levels=3,
            thermal_occupation=n_bar,
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


    def test_T1_thermal_foldin(self) -> None:
        """T1 + thermal_occupation → fold-in with γ=1/T1."""
        T1 = 10_000.0
        n_bar = 0.05
        q = DuffingTransmon(
            freq=5.0,
            anharmonicity=-0.25,
            levels=3,
            T1=T1,
            thermal_occupation=n_bar,
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


def test_auto_label_increments_per_prefix():
    """Each prefix keeps its own counter, independent of other prefixes."""
    reset_label_counters()
    assert auto_label("charge") == "charge_0"
    assert auto_label("charge") == "charge_1"
    assert auto_label("flux") == "flux_0"
    assert auto_label("charge") == "charge_2"


def test_resolve_label_rejects_unlabeled_objects():
    """An object without a label raises TypeError."""
    with pytest.raises(TypeError, match="label"):
        resolve_label(42)
