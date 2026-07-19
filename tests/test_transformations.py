"""Tests for the model-reduction transform ``eliminate()`` and the ``ChipTransform`` protocol."""

import numpy as np
import pytest

from quchip import Capacitive, Chip, CrossKerr, DuffingTransmon, Resonator, TunableCapacitive
from quchip.chip.transformations import ChipTransform, EliminationResult
from quchip.inverse_design.types import FitADressResult


def test_elimination_result_satisfies_chiptransform_protocol():
    """EliminationResult structurally satisfies the ChipTransform protocol."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    res = EliminationResult(chip=Chip([q]), effective_params={}, validity={}, notes=[])
    assert isinstance(res, ChipTransform)
    assert res.chip is not None


def test_fitadress_result_also_satisfies_protocol_without_changes():
    """FitADressResult also satisfies ChipTransform via its `chip` field, unmodified."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q")
    res = FitADressResult(
        chip=Chip([q]),
        loss=0.0,
        history=None,
        initial_targets=(),
        final_targets=(),
        initial_params={},
        final_params={},
        solver_info={},
    )
    assert isinstance(res, ChipTransform)


def test_eliminate_resonator_folds_lamb_shift_and_purcell():
    """Eliminating a resonator folds its Lamb shift and Purcell decay into the survivor."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, quality_factor=5000.0, levels=4, label="r")
    g = 0.08
    chip = Chip([q, r], couplings=[Capacitive(q, r, g=g)])
    from quchip.chip.transformations import eliminate

    res = eliminate(chip, "r")
    reduced = res.chip
    assert [d.label for d in reduced.devices] == ["q"]

    delta = 5.0 - 7.0
    lamb = g**2 / delta
    kappa = 2 * np.pi * 7.0 / 5000.0
    purcell = (g / delta) ** 2 * kappa

    assert reduced["q"].freq == pytest.approx(5.0 + lamb, rel=1e-6)
    assert 1.0 / reduced["q"].T1 == pytest.approx(purcell, rel=1e-6)
    assert res.validity["cap_0"]["g_over_delta"] == pytest.approx(abs(g / delta), rel=1e-6)


def _bridge_chip(direct_g=None):
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q0")
    q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.25, levels=2, label="q1")
    bus = Resonator(freq=7.0, levels=3, label="bus")
    couplings = [Capacitive(q0, bus, g=0.05, label="leg0"), Capacitive(q1, bus, g=0.05, label="leg1")]
    if direct_g is not None:
        couplings.append(Capacitive(q0, q1, g=direct_g, label="direct"))
    return Chip([q0, q1, bus], couplings=couplings)


def test_eliminate_bridge_derives_mediated_exchange():
    """Eliminating a bridging bus derives the mediated exchange J = (g1 g2/2)(1/Δ1 + 1/Δ2)."""
    from quchip.chip.transformations import eliminate

    res = eliminate(_bridge_chip(), "bus")
    reduced = res.chip

    j_expected = 0.05 * 0.05 / 2.0 * (1.0 / (5.0 - 7.0) + 1.0 / (5.1 - 7.0))
    assert [d.label for d in reduced.devices] == ["q0", "q1"]
    (mediated,) = reduced.couplings
    assert type(mediated) is Capacitive
    assert mediated.label == "elim_bus"
    assert float(mediated.g) == pytest.approx(j_expected, rel=1e-9)

    assert reduced["q0"].freq == pytest.approx(5.0 + 0.05**2 / (5.0 - 7.0), rel=1e-9)
    assert reduced["q1"].freq == pytest.approx(5.1 + 0.05**2 / (5.1 - 7.0), rel=1e-9)
    assert set(res.validity) == {"leg0", "leg1"}
    assert res.effective_params["exchange"]["between"] == ("q0", "q1")


def test_eliminate_bridge_folds_into_existing_direct_coupling():
    """A direct coupling tuned to cancel the mediated exchange nets to g=0 after elimination."""
    from quchip.chip.transformations import eliminate

    j_expected = 0.05 * 0.05 / 2.0 * (1.0 / (5.0 - 7.0) + 1.0 / (5.1 - 7.0))
    res = eliminate(_bridge_chip(direct_g=-j_expected), "bus")

    (direct,) = res.chip.couplings
    assert type(direct) is Capacitive
    assert direct.label == "direct"
    assert float(direct.g) == pytest.approx(0.0, abs=1e-12)
    assert res.effective_params["exchange"]["folded_into"] == "direct"


def test_eliminate_bridge_preserves_non_foldable_direct_edge_without_double_counting():
    """A direct edge that doesn't declare folds_exchange is preserved unchanged; no double-counted exchange."""
    from quchip.chip.sw import bare_hamiltonian, bare_index
    from quchip.chip.transformations import eliminate
    from quchip.declarative.models import CouplingModel
    from quchip.declarative.parameters import Scalar, parameter

    class CustomExchange(CouplingModel):
        """Minimal honest exchange coupling that does not declare folds_exchange."""

        j: Scalar = parameter(unit="GHz")

        def interaction(self, a, b):
            return self.j * (a.adag * b.a + a.a * b.adag)

    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q0")
    q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.25, levels=2, label="q1")
    bus = Resonator(freq=7.0, levels=3, label="bus")
    direct_j = 0.01
    chip = Chip(
        [q0, q1, bus],
        couplings=[
            Capacitive(q0, bus, g=0.05, label="leg0"),
            Capacitive(q1, bus, g=0.05, label="leg1"),
            CustomExchange(q0, q1, j=direct_j, label="direct"),
        ],
    )

    res = eliminate(chip, "bus")
    reduced = res.chip

    assert {c.label for c in reduced.couplings} == {"direct", "elim_bus"}
    assert float(reduced.coupling_map["direct"].j) == pytest.approx(direct_j)  # untouched

    j_mediated_expected = 0.05 * 0.05 / 2.0 * (1.0 / (5.0 - 7.0) + 1.0 / (5.1 - 7.0))
    parallel_edge = reduced.coupling_map["elim_bus"]
    assert float(parallel_edge.g) == pytest.approx(j_mediated_expected, rel=1e-6)

    # The reduced chip's own total exchange (read the same way the fold
    # itself measures it) must equal direct_j + j_mediated exactly once —
    # a double count would read 2*direct_j + j_mediated instead.
    h, labels, dims = bare_hamiltonian(reduced, reduced.backend)
    row = bare_index(labels, dims, "q0")
    col = bare_index(labels, dims, "q1")
    assert complex(h[row, col]).real == pytest.approx(direct_j + j_mediated_expected, rel=1e-6)


def test_eliminate_bridge_direct_edge_whose_rwa_rejects_exchange_contributes_nothing():
    """A direct edge whose resolved RWA rejects the exchange band is excluded from the fold's accounting."""
    import warnings

    from quchip.chip.sw import bare_hamiltonian, bare_index
    from quchip.chip.transformations import eliminate
    from quchip.declarative.models import CouplingModel
    from quchip.declarative.parameters import Scalar, parameter

    class RwaRejectsExchange(CouplingModel):
        """Exchange-only interaction whose RWA policy rejects its own (only) band."""

        j: Scalar = parameter(unit="GHz")

        def interaction(self, a, b):
            return self.j * (a.adag * b.a + a.a * b.adag)

        def rwa_keeps_band(self, delta_a, delta_b):
            return False

    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q0")
    q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.25, levels=2, label="q1")
    bus = Resonator(freq=7.0, levels=3, label="bus")
    chip = Chip(
        [q0, q1, bus],
        couplings=[
            Capacitive(q0, bus, g=0.05, label="leg0"),
            Capacitive(q1, bus, g=0.05, label="leg1"),
            RwaRejectsExchange(q0, q1, j=0.01, label="direct"),
        ],
    )

    with warnings.catch_warnings():
        # "direct" vanishes entirely under the chip's resolved RWA — expected
        # and irrelevant to what this test checks.
        warnings.simplefilter("ignore", UserWarning)
        res = eliminate(chip, "bus")
        reduced = res.chip

        j_mediated_expected = 0.05 * 0.05 / 2.0 * (1.0 / (5.0 - 7.0) + 1.0 / (5.1 - 7.0))
        parallel_edge = reduced.coupling_map["elim_bus"]
        assert float(parallel_edge.g) == pytest.approx(j_mediated_expected, rel=1e-6)

        h, labels, dims = bare_hamiltonian(reduced, reduced.backend)
        row = bare_index(labels, dims, "q0")
        col = bare_index(labels, dims, "q1")
        assert complex(h[row, col]).real == pytest.approx(j_mediated_expected, rel=1e-6)


def test_eliminate_bridge_fold_target_and_preserved_edge_are_each_counted_exactly_once():
    """A foldable edge and a preserved edge between the same pair each contribute exactly once."""
    from quchip.chip.sw import bare_hamiltonian, bare_index
    from quchip.chip.transformations import eliminate
    from quchip.declarative.models import CouplingModel
    from quchip.declarative.parameters import Scalar, parameter

    class CustomExchange(CouplingModel):
        """Minimal honest exchange coupling that does not declare folds_exchange."""

        j: Scalar = parameter(unit="GHz")

        def interaction(self, a, b):
            return self.j * (a.adag * b.a + a.a * b.adag)

    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q0")
    q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.25, levels=2, label="q1")
    bus = Resonator(freq=7.0, levels=3, label="bus")
    direct_j = 0.01
    foldable_g = 0.02
    chip = Chip(
        [q0, q1, bus],
        couplings=[
            Capacitive(q0, bus, g=0.05, label="leg0"),
            Capacitive(q1, bus, g=0.05, label="leg1"),
            Capacitive(q0, q1, g=foldable_g, label="foldable"),
            CustomExchange(q0, q1, j=direct_j, label="preserved"),
        ],
    )

    res = eliminate(chip, "bus")
    reduced = res.chip

    assert {c.label for c in reduced.couplings} == {"foldable", "preserved"}
    assert float(reduced.coupling_map["preserved"].j) == pytest.approx(direct_j)  # untouched

    j_mediated_expected = 0.05 * 0.05 / 2.0 * (1.0 / (5.0 - 7.0) + 1.0 / (5.1 - 7.0))
    total_expected = foldable_g + direct_j + j_mediated_expected

    h, labels, dims = bare_hamiltonian(reduced, reduced.backend)
    row = bare_index(labels, dims, "q0")
    col = bare_index(labels, dims, "q1")
    assert complex(h[row, col]).real == pytest.approx(total_expected, rel=1e-6)


def test_eliminate_bridge_purcell_from_mode_t1():
    """A dissipative bridge feeds a mediated-decay rate (g/Δ)²/T1_mode to both survivors."""
    from quchip.chip.transformations import eliminate

    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q0")
    q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.25, levels=2, label="q1")
    bus = Resonator(freq=7.0, levels=3, label="bus", T1=10_000.0)
    chip = Chip([q0, q1, bus], couplings=[Capacitive(q0, bus, g=0.05), Capacitive(q1, bus, g=0.05)])

    res = eliminate(chip, "bus")
    expected_rate = (0.05 / (5.0 - 7.0)) ** 2 / 10_000.0
    assert 1.0 / res.chip["q0"].T1 == pytest.approx(expected_rate, rel=1e-9)


def test_eliminate_purcell_survivor_without_thermal_population_folds_normally():
    """A Purcell fold onto a survivor with no thermal_population scales T1 exactly as before (no regression)."""
    from quchip.chip.transformations import eliminate

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q", T1=30_000.0)
    r = Resonator(freq=7.0, quality_factor=5000.0, levels=4, label="r")
    chip = Chip([q, r], couplings=[Capacitive(q, r, g=0.08, label="cap0")])

    res = eliminate(chip, "r")
    kappa = 2 * np.pi * 7.0 / 5000.0
    purcell_rate = (0.08 / (5.0 - 7.0)) ** 2 * kappa
    expected_rate = 1.0 / 30_000.0 + purcell_rate
    assert 1.0 / res.chip["q"].T1 == pytest.approx(expected_rate, rel=1e-6)


def test_eliminate_purcell_survivor_with_thermal_population_raises():
    """A Purcell fold onto a survivor that also carries thermal_population fails fast rather than mis-modeling."""
    from quchip.chip.transformations import eliminate

    q = DuffingTransmon(
        freq=5.0, anharmonicity=-0.25, levels=3, label="q", T1=30_000.0, thermal_population=0.02
    )
    r = Resonator(freq=7.0, quality_factor=5000.0, levels=4, label="r")
    chip = Chip([q, r], couplings=[Capacitive(q, r, g=0.08, label="cap0")])

    with pytest.raises(NotImplementedError, match="thermal_population"):
        eliminate(chip, "r")


def test_eliminate_three_survivors_emits_pairwise_edges_matching_yan_formula():
    """A mode touching three survivors emits one edge per pair, each carrying the Yan mediated J."""
    from quchip.chip.transformations import eliminate

    qs = [DuffingTransmon(freq=5.0 + 0.15 * i, anharmonicity=-0.25, levels=3, label=f"q{i}") for i in range(3)]
    bus = Resonator(freq=7.0, levels=4, label="bus")
    chip = Chip(qs + [bus], couplings=[Capacitive(q, bus, g=0.06, label=f"leg{i}") for i, q in enumerate(qs)])

    res = eliminate(chip, "bus")
    reduced = res.chip
    assert sorted(d.label for d in reduced.devices) == ["q0", "q1", "q2"]
    pairs = {frozenset((c.device_a_label, c.device_b_label)) for c in reduced.couplings}
    assert pairs == {frozenset({"q0", "q1"}), frozenset({"q0", "q2"}), frozenset({"q1", "q2"})}
    assert all(type(c) is Capacitive for c in reduced.couplings)

    exchange = res.effective_params["exchange"]
    assert set(exchange) == {("q0", "q1"), ("q0", "q2"), ("q1", "q2")}
    freqs = {q.label: q.freq for q in qs}
    for (a, b), entry in exchange.items():
        delta_a = freqs[a] - 7.0
        delta_b = freqs[b] - 7.0
        expected = 0.06 * 0.06 / 2.0 * (1.0 / delta_a + 1.0 / delta_b)
        assert float(entry["j_eff"]) == pytest.approx(expected, rel=1e-6)


def test_eliminate_unknown_method_raises_value_error():
    """eliminate() rejects any method other than 'sw'/'exact'."""
    from quchip.chip.transformations import eliminate

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    chip = Chip([q, r], couplings=[Capacitive(q, r, g=0.05)])
    with pytest.raises(ValueError, match="'sw'|'exact'"):
        eliminate(chip, "r", method="numeric")


def test_registry_apparatus_is_gone():
    """The analytic elimination-rule registry is deleted; no legacy shim remains."""
    import quchip

    assert not hasattr(quchip, "register_elimination_rule")
    assert not hasattr(quchip.chip.transformations, "register_elimination_rule")
    assert not hasattr(quchip.chip.transformations, "_ANALYTIC_RULES")


def test_sw_bridge_j_matches_the_analytic_yan_formula():
    """The default method='sw' bridge exchange matches the analytic Yan J to 1e-9 relative."""
    from quchip.chip.transformations import eliminate

    res = eliminate(_bridge_chip(), "bus")
    j_expected = 0.05 * 0.05 / 2.0 * (1.0 / (5.0 - 7.0) + 1.0 / (5.1 - 7.0))
    assert float(res.effective_params["exchange"]["j_eff"]) == pytest.approx(j_expected, rel=1e-9)


@pytest.mark.optional_backend
def test_eliminate_bridge_exchange_is_differentiable_in_leg_g():
    """The mediated exchange J is differentiable in a bridge leg's coupling strength g."""
    pytest.importorskip("dynamiqs")
    import jax
    from quchip.chip.transformations import eliminate

    def j_eff(g):
        q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q0")
        q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.25, levels=2, label="q1")
        bus = Resonator(freq=7.0, levels=3, label="bus")
        chip = Chip(
            [q0, q1, bus],
            couplings=[Capacitive(q0, bus, g=g), Capacitive(q1, bus, g=0.05)],
            backend="dynamiqs",
        )
        return eliminate(chip, "bus").effective_params["exchange"]["j_eff"]

    grad = jax.grad(j_eff)(0.05)
    # dJ/dg1 = (g2/2)(1/Δ1 + 1/Δ2)
    assert float(grad) == pytest.approx(0.05 / 2.0 * (1.0 / -2.0 + 1.0 / -1.9), rel=1e-6)


def test_eliminate_bridge_composes_sequentially_over_a_chain():
    """Eliminating both couplers of QB1-TC1-CR-TC2-QB2 leaves two mediated qubit-resonator couplings."""
    from quchip.chip.transformations import eliminate

    qb1 = DuffingTransmon(freq=4.7, anharmonicity=-0.2, levels=2, label="qb1")
    tc1 = DuffingTransmon(freq=6.0, anharmonicity=-0.1, levels=2, label="tc1")
    cr = Resonator(freq=4.2, levels=3, label="cr")
    tc2 = DuffingTransmon(freq=6.0, anharmonicity=-0.1, levels=2, label="tc2")
    qb2 = DuffingTransmon(freq=4.5, anharmonicity=-0.2, levels=2, label="qb2")
    chip = Chip(
        [qb1, tc1, cr, tc2, qb2],
        couplings=[
            Capacitive(qb1, tc1, g=0.09),
            Capacitive(tc1, cr, g=0.11),
            Capacitive(tc2, cr, g=0.11),
            Capacitive(qb2, tc2, g=0.10),
        ],
    )

    step1 = eliminate(chip, "tc1")
    step2 = eliminate(step1.chip, "tc2")
    reduced = step2.chip

    assert sorted(d.label for d in reduced.devices) == ["cr", "qb1", "qb2"]
    pairs = {frozenset((c.device_a_label, c.device_b_label)) for c in reduced.couplings}
    assert pairs == {frozenset({"qb1", "cr"}), frozenset({"qb2", "cr"})}


def test_eliminate_folds_through_a_fold_created_edge():
    """A fold-created coupling behaves as an ordinary edge later, driving the next survivor's g/Δ and Lamb shift."""
    from quchip.chip.transformations import eliminate

    step1 = eliminate(_bridge_chip(), "bus")
    mid = step1.chip
    (edge,) = mid.couplings
    assert type(edge) is Capacitive

    step2 = eliminate(mid, "q1")
    assert [d.label for d in step2.chip.devices] == ["q0"]

    delta = float(mid["q0"].freq - mid["q1"].freq)
    g = float(edge.g)
    assert step2.validity["elim_bus"]["g_over_delta"] == pytest.approx(abs(g / delta), rel=1e-9)
    # Deep in the dispersive regime (g0/Δ ≈ 0.013) the leaf fold's Lamb shift
    # is g0²/Δ to leading order; the counter-rotating correction is ~1%.
    lamb = float(step2.effective_params["q0"]["lamb_shift"])
    assert lamb == pytest.approx(g**2 / delta, rel=0.05)
    # The exact route reads the same g and must agree on the dressed frequency.
    exact = eliminate(mid, "q1", method="exact")
    assert float(exact.chip["q0"].freq) == pytest.approx(float(step2.chip["q0"].freq), abs=1e-6)


def test_eliminate_refuses_bath_explicitly_targeting_the_mode():
    """eliminate() raises when a chip-level bath explicitly targets the eliminated mode."""
    from quchip import Bath

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q")
    r = Resonator(freq=7.0, quality_factor=5000.0, levels=3, label="r")
    chip = Chip(
        [q, r],
        couplings=[Capacitive(q, r, g=0.05)],
        baths=[Bath("collective_decay", targets=[q, r], rate=0.01)],
    )
    from quchip.chip.transformations import eliminate

    with pytest.raises(ValueError, match="explicitly targets"):
        eliminate(chip, "r")


@pytest.mark.optional_backend
def test_eliminate_lamb_shift_is_differentiable_in_g():
    """The Lamb shift is differentiable in g on a JAX-native backend, since chi is lazy."""
    pytest.importorskip("dynamiqs")
    import jax
    from quchip.chip.transformations import eliminate

    def lamb_shift(g):
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q")
        r = Resonator(freq=7.0, levels=3, label="r")
        chip = Chip([q, r], couplings=[Capacitive(q, r, g=g)], backend="dynamiqs")
        return eliminate(chip, "r").effective_params["q"]["lamb_shift"]

    grad = jax.grad(lamb_shift)(0.08)
    # d/dg (g^2/Δ) = 2g/Δ, Δ = -2.0
    assert float(grad) == pytest.approx(2 * 0.08 / (5.0 - 7.0), rel=1e-4)


def test_public_api_exports():
    """Bath, eliminate, EliminationResult, and ChipTransform are exported from quchip."""
    import quchip

    for name in ("Bath", "eliminate", "EliminationResult", "ChipTransform"):
        assert hasattr(quchip, name), name


def test_eliminate_with_circuit_level_survivor_warns_instead_of_raising():
    """A survivor without a 'freq' tunable keeps its bare spectrum; the shift is reported."""
    from quchip import ChargeBasisTransmon
    from quchip.chip.transformations import eliminate

    q = ChargeBasisTransmon(E_C=0.25, E_J=12.0, n_g=0.0, levels=4, label="q")
    r = Resonator(freq=7.1, levels=5, label="r")
    chip = Chip([q, r], couplings=[Capacitive(q, r, g=0.08)])
    bare_freq = chip["q"].freq

    with pytest.warns(UserWarning, match="exposes no 'freq' tunable"):
        res = eliminate(chip, r)

    assert [d.label for d in res.chip.devices] == ["q"]
    assert res.chip["q"].freq == pytest.approx(bare_freq)  # spectrum not folded
    delta = bare_freq - 7.1
    lamb = 0.08**2 / delta
    assert float(res.effective_params["q"]["lamb_shift"]) == pytest.approx(lamb, rel=0.05)
    assert float(res.effective_params["q"]["freq_after"]) == pytest.approx(bare_freq + lamb, rel=1e-3)


def test_eliminate_coupling_target_capacitive_lamb_shifts_both_endpoints():
    """Coupling-target elimination of a Capacitive Lamb-shifts both endpoints and reports validity."""
    from quchip.chip.transformations import eliminate

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=5, label="r")
    chip = Chip([q, r], couplings=[Capacitive(q, r, g=0.05, label="c")])

    res = eliminate(chip, "c")
    reduced = res.chip

    assert sorted(d.label for d in reduced.devices) == ["q", "r"]
    (crosskerr,) = reduced.couplings
    assert type(crosskerr) is CrossKerr

    assert res.validity["c"]["g_over_delta"] == pytest.approx(abs(0.05 / (5.0 - 7.0)), rel=1e-6)
    assert float(res.effective_params["q"]["lamb_shift"]) != 0.0
    assert float(res.effective_params["r"]["lamb_shift"]) != 0.0
    assert any("uniform-chi" in note for note in res.notes)


def test_eliminate_coupling_target_tunable_capacitive_reports_validity():
    """Coupling-target elimination of a TunableCapacitive reports non-empty validity (previously empty)."""
    from quchip.chip.transformations import eliminate

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=5, label="r")
    chip = Chip([q, r], couplings=[TunableCapacitive(q, r, g_0=0.05, label="tc")])

    res = eliminate(chip, "tc")
    assert res.validity != {}
    assert res.validity["tc"]["g_over_delta"] == pytest.approx(abs(0.05 / (5.0 - 7.0)), rel=1e-6)


def test_eliminate_coupling_target_rejects_a_non_exchange_coupling():
    """A coupling that does not declare reduces_to_crosskerr is rejected with an explicit error."""
    from quchip.chip.transformations import eliminate

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=5, label="r")
    chip = Chip([q, r], couplings=[CrossKerr(q, r, chi=0.001, label="xk")])

    with pytest.raises(NotImplementedError, match="reduces_to_crosskerr"):
        eliminate(chip, "xk")


def test_eliminate_coupling_target_endpoint_without_freq_raises():
    """Coupling-target elimination raises when an endpoint exposes no 'freq' tunable."""
    from quchip import ChargeBasisTransmon
    from quchip.chip.transformations import eliminate

    q = ChargeBasisTransmon(E_C=0.25, E_J=12.0, n_g=0.0, levels=4, label="q")
    r = Resonator(freq=7.1, levels=5, label="r")
    chip = Chip([q, r], couplings=[Capacitive(q, r, g=0.08, label="c")])

    with pytest.raises(NotImplementedError, match="freq"):
        eliminate(chip, "c")
