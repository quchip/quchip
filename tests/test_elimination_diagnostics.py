"""Labeling margins and typed errors of eliminate() (#198)."""

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


@pytest.mark.validation
@pytest.mark.optional_backend
def test_labeling_margin_derivative_and_compiled_error_text():
    """On dynamiqs, the derivative of a margin matches central differences, and a compiled failure carries the
    label and the weight in its text."""
    pytest.importorskip("dynamiqs")
    import jax

    chip = bus_chip(5.08, backend="dynamiqs")

    def margin(f_bus):
        reduction = eliminate(chip.with_params({"bus.freq": f_bus}), "bus", method="exact")
        return reduction.validity["bus_labeling"]["labels"][(1, 1, 0)]["margin"]

    step = 1e-5
    finite = (margin(5.08 + step) - margin(5.08 - step)) / (2 * step)
    np.testing.assert_allclose(jax.grad(margin)(5.08), finite, rtol=1e-6)
    with pytest.raises(Exception, match=r"Label \(1, 1, 0\) of \('qa', 'qb', 'bus'\) has largest dressed weight "
                                        r"0\.491756"):
        jax.block_until_ready(jax.jit(margin)(5.09))


