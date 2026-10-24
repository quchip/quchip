"""Labeling margins and typed errors of eliminate() (#198), its virtual-state pathways (#199), and loss diagnostics."""

import itertools

import numpy as np
import pytest

from quchip import Capacitive, Chip, DuffingTransmon, EliminationError, Exact, Resonator, eliminate


def bus_chip(f_bus, backend="qutip", spectator=False):
    """The chip of #198 and #199: two transmons coupled through a bus, optionally with a spectator on qb."""
    a = DuffingTransmon(freq=5.05, anharmonicity=-0.22, levels=3, label="qa")
    b = DuffingTransmon(freq=5.35, anharmonicity=-0.24, levels=3, label="qb")
    bus = Resonator(freq=f_bus, levels=4, label="bus")
    devices = [a, b, bus]
    couplings = [Capacitive(a, bus, g=0.07, label="ga"), Capacitive(b, bus, g=0.07, label="gb")]
    if spectator:
        c = DuffingTransmon(freq=5.9, anharmonicity=-0.2, levels=2, label="qc")
        devices.append(c)
        couplings.append(Capacitive(b, c, g=0.01, label="gc"))
    return Chip(devices, couplings, approximation=Exact(), backend=backend)


def largest_weights(chip, occupation):
    """The two largest weights of a bare product state over the eigenvectors of the full Hamiltonian."""
    _, vectors = np.linalg.eigh(np.asarray(chip.hamiltonian().matrix()))
    weights = np.sort(np.abs(vectors[np.ravel_multi_index(occupation, chip.dims)]) ** 2)
    return weights[-1], weights[-2]


@pytest.mark.validation
def test_labeling_margin_closes_on_its_threshold_next_to_each_raise():
    """Along the bus sweep of #198, the margin of label (1, 1, 0) falls toward zero on both sides of each raise.
    The raise names that label, and its weight equals the largest weight of an independent diagonalization."""
    # The points avoid the exact degeneracies of the bus with qa (5.05 GHz) and qb (5.35 GHz).
    sides = [(5.055, 5.06, 5.07, 5.08, 5.085), (5.17, 5.16, 5.15, 5.145), (5.33, 5.34, 5.345), (5.38, 5.37, 5.365)]
    for side, raised in zip(sides, (5.09, 5.14, 5.355, 5.36)):
        margins = []
        for f_bus in side:
            entry = eliminate(bus_chip(f_bus), "bus", method="exact").validity["bus_labeling"]
            label = entry["labels"][(1, 1, 0)]
            weight, runner_up = largest_weights(bus_chip(f_bus), (1, 1, 0))
            np.testing.assert_allclose([label["weight"], label["runner_up"]], [weight, runner_up], rtol=1e-10)
            np.testing.assert_allclose(label["margin"], weight - (0.5 + 1e-6), rtol=0, atol=1e-15)
            assert bool(entry["is_valid"]) and entry["margin"] <= label["margin"]
            margins.append(float(label["margin"]))
        assert np.all(np.diff(margins) < 0) and 0 < margins[-1] < 0.02
        with pytest.raises(EliminationError) as caught:
            eliminate(bus_chip(raised), "bus", method="exact")
        error = caught.value
        assert (error.route, error.check, error.devices, error.label) == ("exact", "labeling", ("qa", "qb", "bus"),
                                                                          (1, 1, 0))
        np.testing.assert_allclose(error.weight, largest_weights(bus_chip(raised), (1, 1, 0))[0], rtol=1e-10)
        assert error.weight < error.threshold == 0.5 + 1e-6
        assert str(error).startswith("Exact reduction of 'bus' cannot label the kept block: Near-degenerate dressed "
                                     "states leave computational bare labels without distinct majority eigenstates.")


@pytest.mark.unit
def test_labeling_entry_reports_the_overlap_conditioning_and_residual():
    """The exact route reports the overlap matrix's smallest eigenvalue and its orthonormalization residual;
    the SW route has no labeling entry."""
    entry = eliminate(bus_chip(5.08), "bus", method="exact").validity["bus_labeling"]
    assert set(entry["labels"]) == {(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)}
    assert entry["threshold"] == 0.5 + 1e-6 and entry["residual_threshold"] == 1e-9
    assert 0 < entry["gram_min_eigenvalue"] < 1 and 0 <= entry["residual"] < 1e-12
    np.testing.assert_allclose(entry["residual_margin"], 1e-9 - entry["residual"], rtol=0, atol=1e-24)
    assert "bus_labeling" not in eliminate(bus_chip(5.08), "bus", method="sw").validity
    assert "bus_labeling weight margin=0.014 ✓" in eliminate(bus_chip(5.08), "bus", method="exact").describe()


@pytest.mark.unit
def test_sw_degeneracy_error_names_its_levels_and_coupling():
    """At 4.83 GHz, |qa=1, bus=1⟩ and |qa=2⟩ both lie at 9.88 GHz and couple through √2 · 0.07 GHz."""
    with pytest.raises(EliminationError) as caught:
        eliminate(bus_chip(4.83), "bus", method="sw")
    error = caught.value
    assert (error.route, error.check, error.levels) == ("sw", "degeneracy", ((1, 0, 1), (2, 0, 0)))
    np.testing.assert_allclose(abs(error.coupling), 0.07 * np.sqrt(2), rtol=1e-12)
    assert str(error).startswith("Schrieffer-Wolff reduction is undefined for degenerate levels with nonzero "
                                 "coupling. Keep the coupled levels together or change the operating point.")


@pytest.mark.unit
def test_exchange_pathways_are_keyed_by_occupation():
    """The exchange of #199 runs through |bus=1⟩ and, by counter-rotating terms, |qa=1, qb=1, bus=1⟩. Each amount is
    ½ g² (1/(E_i − E_k) + 1/(E_j − E_k)) with g = 0.07 GHz."""
    result = eliminate(bus_chip(6.40), "bus", method="sw")
    pathways = result.effective_params["exchange"]["pathways"]

    def amount(energy):
        return 0.5 * 0.07**2 * (1 / (5.05 - energy) + 1 / (5.35 - energy))

    assert [key for key, _ in pathways] == [(0, 0, 1), (1, 1, 1)]
    np.testing.assert_allclose([complex(value) for _, value in pathways], [amount(6.40), amount(16.80)], rtol=1e-12)
    assert result.pathways({"qa": 1}, {"qb": 1}) == pathways


@pytest.mark.validation
def test_element_pathways_sum_to_the_retained_second_order_part():
    """For every retained element, diagonal or not, the pathways sum to the reduced Hamiltonian minus its bare part."""
    chip = bus_chip(6.40)
    result = eliminate(chip, "bus", method="sw")
    bare = np.asarray(chip.hamiltonian().matrix())
    kept = [np.ravel_multi_index((a, b, 0), chip.dims) for a, b in itertools.product(range(3), range(3))]
    second = np.asarray(result.chip.hamiltonian().matrix()) - bare[np.ix_(kept, kept)]
    assert np.max(np.abs(second)) > 1e-2
    for (row, final), (column, initial) in itertools.product(enumerate(itertools.product(range(3), range(3))),
                                                             repeat=2):
        pathways = result.pathways(dict(zip(("qa", "qb"), final)), dict(zip(("qa", "qb"), initial)))
        np.testing.assert_allclose(sum(complex(value) for _, value in pathways), second[row, column],
                                   rtol=0, atol=1e-14)


@pytest.mark.unit
def test_pathways_keep_contributions_below_the_old_reporting_floor():
    """A 1e-7 GHz coupling across a 1e-5 GHz gap shifts the qubit by g²/Δ = −1e-9 GHz, through one pathway."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q")
    r = Resonator(freq=5.0 + 1e-5, levels=2, label="r")
    result = eliminate(Chip([q, r], [Capacitive(q, r, g=1e-7)], approximation=Exact()), "r", method="sw")
    [(key, value)] = result.pathways({"q": 1})
    assert key == (0, 1)
    np.testing.assert_allclose(complex(value), -1e-14 / 1e-5, rtol=1e-9)


@pytest.mark.e2e
def test_local_pathways_match_the_full_reduction_and_keep_spectator_levels():
    """A local reduction attributes the same elements; a spectator outside the patch keeps its level."""
    chip = bus_chip(6.40, spectator=True)
    full, local = (eliminate(chip, "bus", method="sw", local=flag) for flag in (False, True))
    for final, initial in [({"qa": 1}, {"qb": 1}), ({"qa": 1, "qc": 1}, {"qb": 1, "qc": 1}), ({"qb": 2}, None)]:
        expected = full.pathways(final, initial)
        actual = local.pathways(final, initial)
        assert [key for key, _ in actual] == [key for key, _ in expected]
        np.testing.assert_allclose([complex(v) for _, v in actual], [complex(v) for _, v in expected],
                                   rtol=1e-12, atol=1e-18)
    assert [key for key, _ in local.pathways({"qa": 1, "qc": 1}, {"qb": 1, "qc": 1})] == [(0, 0, 1, 1), (1, 1, 1, 1)]
    assert local.pathways({"qa": 1, "qc": 1}, {"qb": 1}) == []


@pytest.mark.unit
def test_pathways_reject_other_routes_and_states_outside_the_retained_space():
    result = eliminate(bus_chip(6.40), "bus", method="sw")
    with pytest.raises(ValueError, match="needs a device elimination with method='sw'"):
        eliminate(bus_chip(6.40), "bus", method="exact").pathways({"qa": 1})
    with pytest.raises(ValueError, match="'bus' is eliminated"):
        result.pathways({"bus": 1})
    with pytest.raises(ValueError, match="not a device of the source chip"):
        result.pathways({"qx": 1})
    with pytest.raises(ValueError, match="Level 3 of 'qa' is outside its 3 levels"):
        result.pathways({"qa": 3})


@pytest.mark.unit
def test_readout_loss_diagnostics_survive_an_earlier_exact_elimination():
    """After an exact bus step, the readout's Purcell rate and kappa agree with the values after an SW bus step."""
    from quchip import RWA

    q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q1")
    q2 = DuffingTransmon(freq=5.4, anharmonicity=-0.25, levels=3, label="q2")
    bus = Resonator(freq=6.4, levels=3, label="bus")
    readout = Resonator(freq=7.0, levels=3, internal_quality_factor=200_000, label="r1")
    chip = Chip([q1, q2, bus, readout],
                [Capacitive(q1, bus, g=0.07), Capacitive(q2, bus, g=0.07), Capacitive(q1, readout, g=0.08)],
                approximation=RWA())
    kappa = 2 * np.pi * 7.0 / 200_000
    for method, local in (("sw", False), ("exact", False), ("sw", True)):
        reference, actual = (
            eliminate(eliminate(chip, "bus", method=first).chip, "r1", method=method, local=local)
            .effective_params["q1"] for first in ("sw", "exact"))
        # The two bus routes give retained coordinates that differ at relative order (g/Delta)^2 = (0.07/1.4)^2
        # = 2.5e-3. The tolerance allows four times that.
        np.testing.assert_allclose(float(actual["kappa"]), kappa, rtol=1e-2)
        np.testing.assert_allclose(float(actual["purcell_rate"]), float(reference["purcell_rate"]), rtol=1e-2)
        # The Purcell rate is kappa (g/Delta)^2 = kappa (0.08/2.0)^2 at leading order. The bus shift of q1, about
        # 3.5 MHz, changes it by 3.5e-3, and the next order in g/Delta by 3 (0.08/2.0)^2 = 4.8e-3. The tolerance
        # allows about twice their sum.
        np.testing.assert_allclose(float(actual["purcell_rate"]), kappa * (0.08 / 2.0) ** 2, rtol=2e-2)


@pytest.mark.unit
def test_readout_loss_diagnostics_count_only_the_channels_the_readout_declares():
    """A retained channel on the readout alone stays out of kappa and the Purcell rate on every route."""
    from quchip import CollapseChannel, EffectiveTerms

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    readout = Resonator(freq=7.0, levels=3, internal_quality_factor=200_000, label="r")
    lowering = np.diag(np.sqrt([1.0, 2.0]), 1)
    extra = EffectiveTerms(("r",), (3,), np.zeros((3, 3)), channels=(CollapseChannel(lowering, 0.01, "extra"),))
    chip = Chip([q, readout], [Capacitive(q, readout, g=0.08)], effective_terms=[extra])
    kappa = 2 * np.pi * 7.0 / 200_000
    for method, local in (("sw", False), ("exact", False), ("sw", True)):
        params = eliminate(chip, "r", method=method, local=local).effective_params["q"]
        np.testing.assert_allclose(float(params["kappa"]), kappa, rtol=1e-12)
        # The Purcell rate is kappa (g/Delta)^2 = kappa (0.08/2.0)^2 at leading order; the next order in g/Delta
        # changes it by 3 (0.08/2.0)^2 = 4.8e-3.
        np.testing.assert_allclose(float(params["purcell_rate"]), kappa * (0.08 / 2.0) ** 2, rtol=1e-2)
