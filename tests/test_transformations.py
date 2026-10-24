"""Tests for the model-reduction transform ``eliminate()`` and the ``ChipTransform`` protocol."""

import numpy as np
import pytest

from quchip import Capacitive, Chip, DuffingTransmon, Resonator


def test_eliminate_resonator_retains_lamb_shift_and_purcell():
    """Eliminating a resonator retains its Lamb shift and Purcell decay separately from authored parameters."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, internal_quality_factor=5000.0, levels=4, label="r")
    g = 0.08
    chip = Chip([q, r], couplings=[Capacitive(q, r, g=g)])
    from quchip.chip.transformations import eliminate

    res = eliminate(chip, "r")
    reduced = res.chip
    assert [d.label for d in reduced.devices] == ["q"]

    delta = 5.0 - 7.0
    lamb = g**2 / delta
    kappa = 2 * np.pi * 7.0 / 5000.0
    purcell = np.sin(g / delta) ** 2 * kappa

    assert reduced["q"].freq == 5.0
    assert reduced.freq("q") == pytest.approx(5.0 + lamb, rel=1e-6)
    jumps = reduced.backend._collapse_operators(reduced.resolve(frame="lab"))
    assert sum(abs(reduced.backend.to_array(op)[0, 1])**2 for op in jumps) == pytest.approx(purcell, rel=1e-6)
    assert reduced["q"].T1 is None
    assert res.validity["cap_0"]["g_over_delta"] == pytest.approx(abs(g / delta), rel=1e-6)


def test_eliminate_bridge_preserves_non_foldable_direct_edge_without_double_counting():
    """A direct edge that owns its authored interaction is preserved unchanged; no double-counted exchange."""
    from quchip.chip.sw import bare_hamiltonian, bare_index
    from quchip.chip.transformations import eliminate
    from quchip.declarative.models import CouplingModel
    from quchip.declarative.parameters import Scalar, parameter

    class CustomExchange(CouplingModel):
        """Minimal honest exchange coupling that owns its authored interaction."""

        j: Scalar = parameter(unit="GHz")

        def interaction(self, a, b, p):
            return p.j * (a.adag * b.a + a.a * b.adag)

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
    h, labels, dims = bare_hamiltonian(reduced)
    row = bare_index(labels, dims, "q0")
    col = bare_index(labels, dims, "q1")
    assert complex(h[row, col]).real == pytest.approx(direct_j + j_mediated_expected, rel=1e-6)


def test_eliminate_bridge_direct_exchange_is_counted_once():
    """An authored direct exchange edge contributes once to the reduced fold."""
    import warnings

    from quchip.chip.sw import bare_hamiltonian, bare_index
    from quchip.chip.transformations import eliminate
    from quchip.declarative.models import CouplingModel
    from quchip.declarative.parameters import Scalar, parameter

    class DirectExchange(CouplingModel):
        """Exchange-only authored interaction."""

        j: Scalar = parameter(unit="GHz")

        def interaction(self, a, b, p):
            return p.j * (a.adag * b.a + a.a * b.adag)

    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q0")
    q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.25, levels=2, label="q1")
    bus = Resonator(freq=7.0, levels=3, label="bus")
    chip = Chip(
        [q0, q1, bus],
        couplings=[
            Capacitive(q0, bus, g=0.05, label="leg0"),
            Capacitive(q1, bus, g=0.05, label="leg1"),
            DirectExchange(q0, q1, j=0.01, label="direct"),
        ],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = eliminate(chip, "bus")
        reduced = res.chip

        j_mediated_expected = 0.05 * 0.05 / 2.0 * (1.0 / (5.0 - 7.0) + 1.0 / (5.1 - 7.0))
        parallel_edge = reduced.coupling_map["elim_bus"]
        assert float(parallel_edge.g) == pytest.approx(j_mediated_expected, rel=1e-6)

        h, labels, dims = bare_hamiltonian(reduced)
        row = bare_index(labels, dims, "q0")
        col = bare_index(labels, dims, "q1")
        assert complex(h[row, col]).real == pytest.approx(0.01 + j_mediated_expected, rel=1e-6)


def test_eliminate_bridge_fold_target_and_preserved_edge_are_each_counted_exactly_once():
    """A foldable edge and a preserved edge between the same pair each contribute exactly once."""
    from quchip.chip.sw import bare_hamiltonian, bare_index
    from quchip.chip.transformations import eliminate
    from quchip.declarative.models import CouplingModel
    from quchip.declarative.parameters import Scalar, parameter

    class CustomExchange(CouplingModel):
        """Minimal honest exchange coupling that owns its authored interaction."""

        j: Scalar = parameter(unit="GHz")

        def interaction(self, a, b, p):
            return p.j * (a.adag * b.a + a.a * b.adag)

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

    assert {c.label for c in reduced.couplings} == {"foldable", "preserved", "elim_bus"}
    assert float(reduced.coupling_map["preserved"].j) == pytest.approx(direct_j)  # untouched

    j_mediated_expected = 0.05 * 0.05 / 2.0 * (1.0 / (5.0 - 7.0) + 1.0 / (5.1 - 7.0))
    total_expected = foldable_g + direct_j + j_mediated_expected

    h, labels, dims = bare_hamiltonian(reduced)
    row = bare_index(labels, dims, "q0")
    col = bare_index(labels, dims, "q1")
    assert complex(h[row, col]).real == pytest.approx(total_expected, rel=1e-6)


def test_eliminate_bridge_purcell_from_mode_t1():
    """A dissipative bridge carries its collective bright-state decay through the retained rotation."""
    from quchip.chip.transformations import eliminate

    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q0")
    q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.25, levels=2, label="q1")
    bus = Resonator(freq=7.0, levels=3, label="bus", T1=10_000.0)
    chip = Chip([q0, q1, bus], couplings=[Capacitive(q0, bus, g=0.05), Capacitive(q1, bus, g=0.05)])

    res = eliminate(chip, "bus")
    angles = np.array([0.05 / (5.0 - 7.0), 0.05 / (5.1 - 7.0)])
    expected_rate = (angles[0] * np.sinc(np.linalg.norm(angles) / np.pi)) ** 2 / 10_000.0
    jumps = res.chip.backend._collapse_operators(res.chip.resolve(frame="lab"))
    psi = res.chip.backend.to_array(res.chip.bare_state(q0=1))
    assert sum(np.linalg.norm(res.chip.backend.to_array(op) @ psi)**2 for op in jumps) == pytest.approx(
        expected_rate, rel=1e-9)


def test_eliminate_purcell_keeps_intrinsic_thermal_noise_separate():
    """Inherited emission and intrinsic thermal absorption remain distinct transformed channels."""
    from quchip.chip.transformations import eliminate

    q = DuffingTransmon(
        freq=5.0, anharmonicity=-0.25, levels=3, label="q", T1=30_000.0, thermal_occupation=0.02
    )
    r = Resonator(freq=7.0, internal_quality_factor=5000.0, levels=4, label="r")
    chip = Chip([q, r], couplings=[Capacitive(q, r, g=0.08, label="cap0")])

    reduced = eliminate(chip, "r").chip
    jumps = [reduced.backend.to_array(op) for op in reduced.backend._collapse_operators(reduced.resolve(frame="lab"))]
    angle = 0.08 / -2.0
    expected_down = 1.02 * np.cos(angle)**2 / 30_000.0 + np.sin(angle)**2 * (2 * np.pi * 7.0 / 5000.0)
    assert sum(abs(op[0, 1])**2 for op in jumps) == pytest.approx(expected_down, rel=1e-9)
    assert sum(abs(op[1, 0])**2 for op in jumps) == pytest.approx(.02 * np.cos(angle)**2 / 30_000.0, rel=1e-9)


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


@pytest.mark.validation
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


def test_eliminate_retains_thermal_bath_and_intrinsic_loss():
    """Independent thermal channels and intrinsic loss retain their mapped generator."""
    from quchip import Bath, eliminate
    from qutip import lindblad_dissipator

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q")
    r = Resonator(freq=7.0, internal_quality_factor=5000.0, levels=3, label="r")
    chip = Chip([q, r], [Capacitive(q, r, g=0.05)],
                baths=[Bath("thermal", targets=[q, r], temperature=20., rate=.01)])
    result = eliminate(chip, "r")
    before = chip.backend._collapse_operators(chip.resolve(frame="lab"))
    after = result.chip.backend._collapse_operators(result.chip.resolve(frame="lab"))
    expected = sum(lindblad_dissipator(result.mapping.project_operator(c)) for c in before)
    actual = sum(lindblad_dissipator(c) for c in after)
    np.testing.assert_allclose(actual.full(), expected.full(), atol=1e-12)


@pytest.mark.validation
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

    grad = jax.jit(jax.grad(lamb_shift))(0.08)
    # d/dg (g^2/Δ) = 2g/Δ, Δ = -2.0
    assert float(grad) == pytest.approx(2 * 0.08 / (5.0 - 7.0), rel=1e-4)
