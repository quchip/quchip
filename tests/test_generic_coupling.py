"""Tests for the generic Coupling class."""

from __future__ import annotations


import numpy as np
import pytest

from quchip import RWA, Capacitive, Chip, Coupling, DuffingTransmon, Exact, Resonator, eliminate
from quchip.backend import get_default_backend
from quchip.engine.approximations import apply_operator_band_filter
from quchip.declarative.expr import materialize_expr


pytestmark = pytest.mark.unit


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


# ---------------------------------------------------------------------------
# Hamiltonian generation
# ---------------------------------------------------------------------------


class TestCouplingHamiltonian:
    def test_product_form_cross_kerr(self, q0, q1):
        """g · n_a ⊗ n_b should produce the correct two-body operator."""
        c = Coupling(
            q0,
            q1,
            g=0.001,
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
            return bk.tensor(bk.dag(a), b) + bk.tensor(a, bk.dag(b)) + bk.tensor(a, b) + bk.tensor(bk.dag(a), bk.dag(b))

        c = Coupling(q0, res, g=0.02, interaction=jc)
        cap = Capacitive(q0, res, g=0.02)

        backend = get_default_backend()
        def keeps_band(first, second):
            return RWA().keeps_operator_band((first, second))
        H = apply_operator_band_filter(
            c.interaction_hamiltonian(),
            dims=(q0.levels, res.levels),
            labels=(q0.label, res.label),
            keeps_band=keeps_band,
            backend=backend,
        )
        H_cap = apply_operator_band_filter(
            materialize_expr(cap.interaction_hamiltonian(), backend),
            dims=(q0.levels, res.levels),
            labels=(q0.label, res.label),
            keeps_band=keeps_band,
            backend=backend,
        )
        np.testing.assert_allclose(
            np.array(backend.to_array(H)),
            np.array(backend.to_array(H_cap)),
            atol=1e-12,
        )

# ---------------------------------------------------------------------------
# Chip integration
# ---------------------------------------------------------------------------


class TestCouplingChipIntegration:
    def test_chip_hamiltonian_with_generic_coupling(self, q0, q1):
        """A generic Coupling contributes to chip.hamiltonian() at full composite dimension."""
        coupling = Coupling(
            q0,
            q1,
            g=0.001,
            op_a=lambda d: d.number_operator(),
            op_b=lambda d: d.number_operator(),
        )
        chip = Chip(devices=[q0, q1], couplings=[coupling])
        H = chip.hamiltonian()
        assert tuple(H.shape) == (9, 9)

    def test_chip_dress_with_generic_coupling(self, q0, q1):
        """chip.dress() succeeds with a generic Coupling present."""
        coupling = Coupling(
            q0,
            q1,
            g=0.001,
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
                Coupling(q0, q1, g=0.001, op_a=lambda d: d.number_operator(), op_b=lambda d: d.number_operator()),
            ],
        )
        H = chip.hamiltonian()
        assert tuple(H.shape) == (45, 45)  # 3 * 3 * 5

    @pytest.mark.optional_backend
    @pytest.mark.parametrize("form", ["product", "callable"])
    def test_generic_coupling_reads_the_chip_backend(self, form):
        """A generic coupling on a dynamiqs chip resolves, dresses and eliminates while QuTiP is the default."""
        pytest.importorskip("dynamiqs")
        received = set()

        def exchange(dev_a, dev_b, bk):
            received.add(type(bk).__name__)
            hop = bk.tensor(bk.dag(dev_a.lowering_operator()), dev_b.lowering_operator())
            return hop + bk.dag(hop)

        def number(device):
            return device.number_operator()

        def chip_on(backend):
            q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
            r = Resonator(freq=6.0, levels=3, label="r")
            operators = {"interaction": exchange} if form == "callable" else {"op_a": number, "op_b": number}
            return Chip([q, r], [Coupling(q, r, g=0.06, **operators)], approximation=Exact(), backend=backend)

        def results(chip):
            hamiltonian = np.asarray(chip.resolve(frame="lab").hamiltonian().matrix(backend=chip.backend))
            excited = np.asarray(chip.backend.to_array(chip.state({"q": 1}))).ravel()
            return hamiltonian, excited, float(eliminate(chip, "r").chip.freq("q"))

        expected = results(chip_on("qutip"))
        received.clear()
        hamiltonian, excited, freq = results(chip_on("dynamiqs"))
        assert received <= {"DynamiqsBackend"}
        np.testing.assert_allclose(hamiltonian, expected[0], rtol=0, atol=1e-12)
        assert abs(np.vdot(expected[1], excited)) == pytest.approx(1.0, abs=1e-12)
        assert freq == pytest.approx(expected[2], abs=1e-12)


# ---------------------------------------------------------------------------
# Serialization (not supported)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Properties and repr
# ---------------------------------------------------------------------------
