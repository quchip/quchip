"""EPR model contracts: input conventions, first-order parameters, chip wiring and loss.

Formulas are checked against independent expressions: the flux-quantum
conversion from SciPy constants, pyEPR's first-order expressions written as
explicit sums, and Koch et al.'s transmon limit.
"""

from __future__ import annotations

import warnings
from math import factorial

import numpy as np
import pytest
import scipy.linalg as sla
from scipy.constants import e, h, hbar

from quchip import CrossKerr, DuffingTransmon, EPRModel, Resonator

pytestmark = pytest.mark.unit

FREQS = [4.8, 5.3, 7.1]
PARTICIPATIONS = [[0.93, 0.004], [0.012, 0.91], [0.02, 0.03]]
SIGNS = [[1, -1], [1, 1], [-1, 1]]
INDUCTANCES = [12e-9, 11e-9]
# Structural tests use truncations far below convergence on purpose.
UNCONVERGED = "ignore:The cosine chip is not converged:UserWarning"


def _model(**changes):
    arguments = dict(junction_inductances=INDUCTANCES, signs=SIGNS, labels=("a", "b", "r"))
    arguments.update(changes)
    return EPRModel(FREQS, PARTICIPATIONS, **arguments)


def _pyepr_chi(freqs, participations, energies):
    """First-order chi_mn = f_m f_n sum_j p_mj p_nj / (4 E_j) as explicit sums (pyEPR, before halving)."""
    n_modes, n_junctions = np.shape(participations)
    return np.array([
        [sum(freqs[m] * freqs[n] * participations[m][j] * participations[n][j] / (4 * energies[j])
             for j in range(n_junctions)) for n in range(n_modes)]
        for m in range(n_modes)
    ])


def test_inductances_convert_with_the_reduced_flux_quantum():
    """E_J = (hbar/2e)^2/(h L_J) in GHz and phi_mj = s_mj sqrt(p_mj f_m / (2 E_J))."""
    model = _model()
    energies = (hbar / (2 * e)) ** 2 / (h * np.asarray(INDUCTANCES)) / 1e9

    np.testing.assert_allclose(model.junction_energies, energies, rtol=1e-12)
    expected = np.asarray(SIGNS) * np.sqrt(
        np.asarray(PARTICIPATIONS) * np.asarray(FREQS)[:, None] / (2 * energies[None, :])
    )
    np.testing.assert_allclose(model.phi_zpf, expected, rtol=1e-12)


def test_first_order_chip_carries_pyepr_first_order_parameters():
    """First-order devices hold f_m - chi_m/2 sums and -chi_mm/2; each pair holds -chi_mn."""
    model = _model()
    chi = _pyepr_chi(FREQS, PARTICIPATIONS, np.asarray(model.junction_energies))

    chip = model.chip(levels=3, nonlinearity="first_order")

    assert [device.label for device in chip.devices] == ["a", "b", "r"]
    assert all(isinstance(device, DuffingTransmon) for device in chip.devices)
    for m, device in enumerate(chip.devices):
        assert device.freq == pytest.approx(FREQS[m] - 0.5 * chi[m].sum(), rel=1e-12)
        assert device.anharmonicity == pytest.approx(-0.5 * chi[m, m], rel=1e-12)
    couplings = {coupling.label: coupling for coupling in chip.couplings}
    assert set(couplings) == {"a_b", "a_r", "b_r"}
    for label, (m, n) in {"a_b": (0, 1), "a_r": (0, 2), "b_r": (1, 2)}.items():
        assert isinstance(couplings[label], CrossKerr)
        assert couplings[label].chi == pytest.approx(-chi[m, n], rel=1e-12)


def test_single_junction_first_order_limit_is_the_transmon_formula():
    """With p = 1, first order gives f01 = sqrt(8 E_J E_C) - E_C and anharmonicity -E_C (Koch et al. 2007)."""
    charging, josephson = 0.25, 20.0
    model = EPRModel([np.sqrt(8 * josephson * charging)], [[1.0]], junction_energies=[josephson], labels=["t"])

    transmon = model.chip(levels=3, nonlinearity="first_order")["t"]

    assert transmon.freq == pytest.approx(np.sqrt(8 * josephson * charging) - charging, rel=1e-12)
    assert transmon.anharmonicity == pytest.approx(-charging, rel=1e-12)


@pytest.mark.filterwarnings(UNCONVERGED)
def test_cosine_chip_hamiltonian_is_the_black_box_hamiltonian():
    """The default cosine chip assembles sum_m f_m n_m - sum_j E_J (cos phi_j - 1 + phi_j^2/2) in model order."""
    model = _model()
    dims = (4, 3, 3)

    def embed(operator, mode):
        factors = [np.eye(dim) for dim in dims]
        factors[mode] = operator
        return np.kron(np.kron(factors[0], factors[1]), factors[2])

    lowering = [np.diag(np.sqrt(np.arange(1.0, dim)), 1) for dim in dims]
    quadrature = [embed(a + a.T, mode) for mode, a in enumerate(lowering)]
    linear = sum(FREQS[mode] * embed(a.T @ a, mode) for mode, a in enumerate(lowering))
    phases = [sum(model.phi_zpf[mode, j] * quadrature[mode] for mode in range(3)) for j in range(2)]
    identity = np.eye(np.prod(dims))
    energies = np.asarray(model.junction_energies)

    exact = linear - sum(energies[j] * (sla.cosm(phi) - identity + phi @ phi / 2) for j, phi in enumerate(phases))
    series = linear - sum(
        energies[j] * sum((-1) ** k * np.linalg.matrix_power(phi, 2 * k) / factorial(2 * k)
                          for k in range(2, 5))
        for j, phi in enumerate(phases)
    )

    chip = model.chip(levels=dims, frame="lab")
    truncated = model.chip(levels=dims, cos_trunc=4, frame="lab")

    assert all(isinstance(device, Resonator) for device in chip.devices)
    assert chip.effective_terms[0].label == "junctions"
    np.testing.assert_allclose(np.asarray(chip.hamiltonian().matrix()), exact, atol=1e-10)
    np.testing.assert_allclose(np.asarray(truncated.hamiltonian().matrix()), series, atol=1e-10)


@pytest.mark.filterwarnings(UNCONVERGED)
def test_junction_signs_matter_only_relative_to_other_junctions():
    """Reversing one mode's signs is a parity gauge; reversing one junction entry changes the spectrum."""
    model = _model()
    signs = np.asarray(SIGNS)

    def spectrum(candidate):
        chip = candidate.chip(levels=(4, 4, 3))
        return np.linalg.eigvalsh(np.asarray(chip.hamiltonian().matrix()))

    reference = spectrum(model)
    gauge = spectrum(model.replace(signs=signs * np.array([[1, 1], [-1, -1], [1, 1]])))
    physical = spectrum(model.replace(signs=signs * np.array([[1, 1], [1, -1], [1, 1]])))

    # The parity transformation is exact; eigvalsh roundoff on GHz-scale entries is below 1e-10.
    np.testing.assert_allclose(gauge, reference, atol=1e-10)
    assert np.max(np.abs(physical - reference)) > 1e-3


@pytest.mark.filterwarnings(UNCONVERGED)
def test_quality_factors_give_the_same_decay_rate_in_both_models():
    """Q_m gives kappa_m = 2 pi f_m / Q_m in both chips; inf marks a lossless mode."""
    model = _model(quality_factors=[2.0e6, np.inf, 8.0e3])

    first = model.chip(levels=3, nonlinearity="first_order")
    cosine = model.chip(levels=2)

    for label, freq, quality in (("a", FREQS[0], 2.0e6), ("r", FREQS[2], 8.0e3)):
        kappa = 2 * np.pi * freq / quality
        assert first[label].intrinsic_decay_rate() == pytest.approx(kappa, rel=1e-12)
        assert cosine[label].intrinsic_decay_rate() == pytest.approx(kappa, rel=1e-12)
    assert first["b"].T1 is None
    assert cosine["b"].internal_quality_factor is None


def test_levels_accept_one_value_one_per_mode_or_one_per_label():
    """levels may be a scalar, a sequence in model order, or a label mapping."""
    model = _model()

    assert model.chip(levels=3, nonlinearity="first_order").dims == (3, 3, 3)
    assert model.chip(levels=[4, 3, 5], nonlinearity="first_order").dims == (4, 3, 5)
    assert model.chip(levels={"r": 5, "a": 4, "b": 3}, nonlinearity="first_order").dims == (4, 3, 5)


def test_replace_returns_a_new_model_with_named_fields_changed():
    """replace() keeps other inputs, converts new inductances and can drop loss."""
    model = _model(quality_factors=[1e6, 1e6, 1e4])

    changed = model.replace(junction_inductances=[10e-9, 10e-9], quality_factors=None, labels=("q0", "q1", "c"))

    assert changed is not model and model.quality_factors is not None
    assert changed.quality_factors is None
    assert changed.labels == ("q0", "q1", "c")
    np.testing.assert_allclose(changed.junction_energies, model.junction_energies[0] * 1.2, rtol=1e-12)
    np.testing.assert_array_equal(changed.participations, model.participations)
    with pytest.raises(TypeError, match="Unknown EPRModel fields"):
        model.replace(frequency=[5.0, 6.0, 7.0])


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (dict(junction_energies=[13.0, 14.0]), "exactly one"),
        (dict(junction_inductances=None), "exactly one"),
        (dict(junction_inductances=[12e-9]), "Expected 2 junction"),
        (dict(junction_inductances=[12e-9, -1e-9]), "positive"),
        (dict(signs=[[1, 1], [1, 1]]), "signs must have shape"),
        (dict(signs=[[1, 0], [1, 1], [1, 1]]), r"\+1 or -1"),
        (dict(labels=("a", "a", "r")), "unique"),
        (dict(labels=("a", "b")), "Expected 3 labels"),
        (dict(quality_factors=[1e6, 0.0, 1e4]), "positive"),
        (dict(quality_factors=[1e6, 1e4]), "shape"),
    ],
)
def test_invalid_model_inputs_raise(arguments, message):
    """Inconsistent shapes, units and values are rejected when the model is built."""
    with pytest.raises(ValueError, match=message):
        _model(**arguments)


@pytest.mark.parametrize(
    ("freqs", "participations", "message"),
    [
        ([5.0, -7.0], [[0.9], [0.1]], "freqs"),
        ([5.0, 7.0], [[0.9], [1.2]], r"\[0, 1\]"),
        ([5.0, 7.0], [[0.9]], "shape"),
        ([5.0, 7.0], [0.9, 0.1], "2 dimension"),
    ],
)
def test_invalid_modes_or_participations_raise(freqs, participations, message):
    """Frequencies must be positive and participations an M x J array in [0, 1]."""
    with pytest.raises(ValueError, match=message):
        EPRModel(freqs, participations, junction_energies=[14.0])


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (dict(levels=3, nonlinearity="quartic"), "nonlinearity"),
        (dict(levels=3, nonlinearity="first_order", cos_trunc=4), "only to nonlinearity='cosine'"),
        (dict(levels=3, cos_trunc=1), "at least 2"),
        (dict(levels=1), "at least 2"),
        (dict(levels=True), "at least 2"),
        (dict(levels=[3, 3]), "Expected 3 levels"),
        (dict(levels={"a": 3, "b": 3}), "each mode once"),
    ],
)
def test_invalid_chip_options_raise(options, message):
    """Unknown nonlinearities, misplaced cos_trunc and invalid truncations are rejected."""
    with pytest.raises(ValueError, match=message):
        _model().chip(**options)


def test_cosine_chip_warns_when_a_nonlinear_mode_is_not_converged():
    """At phi_zpf = 0.41, 4 levels reverse the anharmonicity's sign; 10 levels and a weak mode pass."""
    model = EPRModel([5.0, 7.0], [[0.95], [0.002]], junction_energies=[14.0], labels=["q", "r"])

    with pytest.warns(UserWarning, match="mode 'q' at 4 levels"):
        model.chip(levels={"q": 4, "r": 3})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.chip(levels={"q": 10, "r": 3})
        model.chip(levels=3, nonlinearity="first_order")
