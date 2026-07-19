"""Tests for the generic Coupling class."""

from __future__ import annotations

import numpy as np
import pytest

from quchip import Capacitive, Chip, Coupling, DuffingTransmon, Resonator
from quchip.backend import get_default_backend
from quchip.chip.rwa import apply_rwa_mask


@pytest.fixture()
def q0():
    return DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")


@pytest.fixture()
def q1():
    return DuffingTransmon(freq=6.0, anharmonicity=-0.22, levels=3, label="q1")


@pytest.fixture()
def res():
    return Resonator(freq=7.0, levels=5, label="r0")


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestCouplingValidation:
    def test_product_form(self, q0, q1):
        """Product-form Coupling stores the coupling_strength g."""
        c = Coupling(
            q0, q1, g=0.01,
            op_a=lambda d: d.number_operator(),
            op_b=lambda d: d.number_operator(),
        )
        assert c.coupling_strength == 0.01

    def test_callable_form(self, q0, q1):
        """Callable-form Coupling stores the coupling_strength g."""
        c = Coupling(
            q0, q1, g=0.02,
            interaction=lambda a, b, bk: bk.tensor(a.number_operator(), b.number_operator()),
        )
        assert c.coupling_strength == 0.02

    def test_missing_both_raises(self, q0, q1):
        """Coupling with neither op_a/op_b nor interaction raises ValueError."""
        with pytest.raises(ValueError, match="Provide either"):
            Coupling(q0, q1, g=0.01)

    def test_both_modes_raises(self, q0, q1):
        """Coupling given both op_a/op_b and interaction raises ValueError."""
        with pytest.raises(ValueError, match="not both"):
            Coupling(
                q0, q1, g=0.01,
                op_a=lambda d: d.number_operator(),
                op_b=lambda d: d.number_operator(),
                interaction=lambda a, b, bk: bk.tensor(a.identity(), b.identity()),
            )

    def test_partial_ops_raises(self, q0, q1):
        """Coupling with only op_a set raises ValueError."""
        with pytest.raises(ValueError, match="Both op_a and op_b"):
            Coupling(q0, q1, g=0.01, op_a=lambda d: d.number_operator())

    def test_non_device_raises(self, q0):
        """A non-device, non-label first argument raises TypeError."""
        with pytest.raises(TypeError, match="BaseDevice or label string"):
            Coupling(42, q0, g=0.01, op_a=lambda d: d.identity(), op_b=lambda d: d.identity())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Hamiltonian generation
# ---------------------------------------------------------------------------


class TestCouplingHamiltonian:
    def test_product_form_cross_kerr(self, q0, q1):
        """g · n_a ⊗ n_b should produce the correct two-body operator."""
        c = Coupling(
            q0, q1, g=0.001,
            op_a=lambda d: d.number_operator(),
            op_b=lambda d: d.number_operator(),
        )
        H = c.interaction_hamiltonian()
        backend = get_default_backend()
        H_arr = np.array(backend.to_array(H), dtype=complex)

        n3 = np.diag([0.0, 1.0, 2.0])
        expected = 0.001 * np.kron(n3, n3)
        np.testing.assert_allclose(H_arr, expected, atol=1e-12)

    def test_callable_form_jc(self, q0, res):
        """Callable-form full form masked to RWA matches the masked Capacitive form."""
        def jc(dev_a, dev_b, bk):
            a = dev_a.lowering_operator()
            b = dev_b.lowering_operator()
            return (
                bk.tensor(bk.dag(a), b) + bk.tensor(a, bk.dag(b))
                + bk.tensor(a, b) + bk.tensor(bk.dag(a), bk.dag(b))
            )

        c = Coupling(q0, res, g=0.02, interaction=jc)
        cap = Capacitive(q0, res, g=0.02)

        backend = get_default_backend()
        H = apply_rwa_mask(
            c.interaction_hamiltonian(),
            dims=(q0.levels, res.levels),
            labels=(q0.label, res.label),
            keeps_band=c.rwa_keeps_band,
            backend=backend,
        )
        H_cap = apply_rwa_mask(
            cap.interaction_hamiltonian(),
            dims=(q0.levels, res.levels),
            labels=(q0.label, res.label),
            keeps_band=cap.rwa_keeps_band,
            backend=backend,
        )
        np.testing.assert_allclose(
            np.array(backend.to_array(H)),
            np.array(backend.to_array(H_cap)),
            atol=1e-12,
        )

    def test_interaction_hamiltonian_is_policy_free(self, q0, q1):
        """interaction_hamiltonian() returns the full form regardless of the rwa attribute."""
        backend = get_default_backend()
        c_default = Coupling(
            q0, q1, g=0.01,
            op_a=lambda d: d.number_operator(),
            op_b=lambda d: d.number_operator(),
        )
        c_full = Coupling(
            q0, q1, g=0.01, rwa=False,
            op_a=lambda d: d.number_operator(),
            op_b=lambda d: d.number_operator(),
        )
        H_default = np.array(backend.to_array(c_default.interaction_hamiltonian()))
        H_full = np.array(backend.to_array(c_full.interaction_hamiltonian()))
        np.testing.assert_allclose(H_default, H_full, atol=1e-12)


# ---------------------------------------------------------------------------
# Chip integration
# ---------------------------------------------------------------------------


class TestCouplingChipIntegration:
    def test_chip_hamiltonian_with_generic_coupling(self, q0, q1):
        """A generic Coupling contributes to chip.hamiltonian() at full composite dimension."""
        coupling = Coupling(
            q0, q1, g=0.001,
            op_a=lambda d: d.number_operator(),
            op_b=lambda d: d.number_operator(),
        )
        chip = Chip(devices=[q0, q1], couplings=[coupling])
        H = chip.hamiltonian()
        assert tuple(H.shape) == (9, 9)

    def test_chip_dress_with_generic_coupling(self, q0, q1):
        """chip.dress() succeeds with a generic Coupling present."""
        coupling = Coupling(
            q0, q1, g=0.001,
            op_a=lambda d: d.number_operator(),
            op_b=lambda d: d.number_operator(),
        )
        chip = Chip(devices=[q0, q1], couplings=[coupling])
        result = chip.dress()
        assert result is not None

    def test_mixed_coupling_types(self, q0, q1, res):
        """Chip with both Capacitive and generic Coupling."""
        chip = Chip(
            devices=[q0, q1, res],
            couplings=[
                Capacitive(q0, res, g=0.02),
                Coupling(q0, q1, g=0.001,
                    op_a=lambda d: d.number_operator(),
                    op_b=lambda d: d.number_operator()),
            ],
        )
        H = chip.hamiltonian()
        assert tuple(H.shape) == (45, 45)  # 3 * 3 * 5


# ---------------------------------------------------------------------------
# Serialization (not supported)
# ---------------------------------------------------------------------------


class TestCouplingSerialization:
    def test_to_dict_raises(self, q0, q1):
        """Coupling.to_dict raises NotImplementedError."""
        c = Coupling(q0, q1, g=0.01, op_a=lambda d: d.identity(), op_b=lambda d: d.identity())
        with pytest.raises(NotImplementedError, match="cannot be serialized"):
            c.to_dict()

    def test_from_dict_raises(self, q0, q1):
        """Coupling.from_dict raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="cannot be deserialized"):
            Coupling.from_dict({}, q0, q1)


# ---------------------------------------------------------------------------
# Properties and repr
# ---------------------------------------------------------------------------


class TestCouplingProperties:
    def test_device_labels(self, q0, q1):
        """Coupling exposes device_a_label and device_b_label."""
        c = Coupling(q0, q1, g=0.01, op_a=lambda d: d.identity(), op_b=lambda d: d.identity())
        assert c.device_a_label == "q0"
        assert c.device_b_label == "q1"

    def test_rwa_property_passthrough(self, q0, q1):
        """Coupling.rwa passes through the constructor value."""
        c = Coupling(q0, q1, g=0.01, rwa=True, op_a=lambda d: d.identity(), op_b=lambda d: d.identity())
        assert c.rwa is True

    def test_rwa_default_none(self, q0, q1):
        """Coupling.rwa defaults to None."""
        c = Coupling(q0, q1, g=0.01, op_a=lambda d: d.identity(), op_b=lambda d: d.identity())
        assert c.rwa is None

    def test_repr_product_mode(self, q0, q1):
        """repr() of a product-form Coupling names the mode and a device label."""
        c = Coupling(q0, q1, g=0.01, op_a=lambda d: d.identity(), op_b=lambda d: d.identity())
        r = repr(c)
        assert "product" in r
        assert "q0" in r

    def test_repr_interaction_mode(self, q0, q1):
        """repr() of an interaction-form Coupling names the mode."""
        c = Coupling(q0, q1, g=0.01, interaction=lambda a, b, bk: bk.tensor(a.identity(), b.identity()))
        r = repr(c)
        assert "interaction" in r
