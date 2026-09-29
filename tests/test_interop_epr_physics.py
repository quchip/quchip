"""EPR chips against exact lumped-circuit models, the first-order limit and JAX derivatives.

A lumped circuit has an exact normal-mode decomposition, so its EPR inputs are
known without a field solver. With the exact junction cosine, the EPR chip is
the same Hamiltonian as the node-basis circuit model, written in the basis of
linear modes. The two agree up to Fock truncation of the modes and the charge
dispersion of the compact transmon coordinate, which is negligible for
E_J/E_C >= 70 at offset charge 1/4 (Koch et al., Phys. Rev. A 76, 042319 (2007)).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla
from scipy.constants import e, h, hbar

from quchip import Capacitive, ChargeBasisTransmon, Chip, EPRModel, Exact, Resonator


def _charging_energy(inverse_capacitance: float) -> float:
    """E_C = e^2 / (2C) in GHz for one diagonal element of C^-1."""
    return e**2 * inverse_capacitance / 2 / h / 1e9


def _josephson_energy(inductance: float) -> float:
    """E_J = (hbar / 2e)^2 / L in GHz."""
    return (hbar / (2 * e)) ** 2 / inductance / h / 1e9


def _lumped_epr(capacitance, inductances, junction_nodes):
    """Return (freqs in GHz, participations, signs) of a grounded lumped circuit.

    Solves L^-1 v = w^2 C v with C-orthonormal modes. Junction j's branch flux in
    mode m is v_m[node_j], so p_mj = v_m[node_j]^2 / (L_j w_m^2).
    """
    inverse_inductance = np.diag(1.0 / np.asarray(inductances))
    w2, vectors = sla.eigh(inverse_inductance, capacitance)
    flux = vectors[junction_nodes, :].T
    junction_inductances = np.asarray(inductances)[junction_nodes]
    participations = flux**2 / (junction_inductances[None, :] * w2[:, None])
    return np.sqrt(w2) / (2 * np.pi * 1e9), participations, np.sign(flux)


def _spectrum(chip: Chip, count: int) -> np.ndarray:
    energies = np.linalg.eigvalsh(np.asarray(chip.hamiltonian().matrix()))[:count]
    return energies - energies[0]


def test_single_junction_cosine_model_matches_the_charge_basis_transmon():
    """A p = 1 EPR mode with the exact cosine converges to the charge-basis transmon spectrum."""
    charging, josephson = 0.25, 20.0  # E_J / E_C = 80
    model = EPRModel([np.sqrt(8 * josephson * charging)], [[1.0]], junction_energies=[josephson], labels=["t"])
    transmon = ChargeBasisTransmon(E_C=charging, E_J=josephson, n_g=0.25, num_basis=41, levels=6, label="t")
    exact = _spectrum(Chip([transmon]), 4)

    coarse, fine = (_spectrum(model.chip(levels=n, nonlinearity="cosine"), 4) for n in (20, 30))

    # Fock truncation converges exponentially, so the last increment bounds the remaining
    # error (measured: 5e-7 GHz at 20 levels, 4e-9 at 30). Charge dispersion of the
    # compared levels cancels at n_g = 1/4 to first order and stays below 1e-8 GHz.
    assert np.max(np.abs(fine - exact)) < np.max(np.abs(fine - coarse))
    assert np.max(np.abs(fine - exact)) < 1e-7


@pytest.mark.validation
def test_dc_squid_enters_with_its_operating_point_josephson_energy():
    """A p = 1 mode at the SQUID's operating-point E_J reproduces the two-junction charge-basis spectrum."""
    charging, first, second, flux_phase = 0.25, 14.0, 9.0, 0.4 * np.pi
    josephson = np.sqrt(first**2 + second**2 + 2 * first * second * np.cos(flux_phase))  # E_J / E_C = 75
    # 4 E_C (n - n_g)^2 - E_J1 cos(phi) - E_J2 cos(phi - phi_ext), where e^{i phi} raises n by one.
    charge = np.arange(-20, 21)
    tunneling = np.full(charge.size - 1, -0.5 * (first + second * np.exp(-1j * flux_phase)))
    squid = np.diag(4 * charging * (charge - 0.25) ** 2) + np.diag(tunneling, -1) + np.diag(tunneling.conj(), 1)
    exact = np.linalg.eigvalsh(squid)[:4] - np.linalg.eigvalsh(squid)[0]
    model = EPRModel([np.sqrt(8 * josephson * charging)], [[1.0]], junction_energies=[josephson], labels=["t"])

    coarse, fine = (_spectrum(model.chip(levels=n), 4) for n in (20, 30))

    # As for the single junction: the last Fock increment bounds the truncation error, and
    # charge dispersion at n_g = 1/4 stays below 1e-8 GHz for E_J / E_C = 75.
    assert np.max(np.abs(fine - exact)) < np.max(np.abs(fine - coarse))
    assert np.max(np.abs(fine - exact)) < 1e-7


@pytest.mark.validation
def test_two_junction_normal_modes_match_exact_coupled_transmons():
    """Hybridized two-junction EPR data reproduce the node-basis spectrum through two excitations."""
    c1, c2, cg, l1, l2 = 110e-15, 118e-15, 5e-15, 9.5e-9, 9.0e-9
    capacitance = np.array([[c1 + cg, -cg], [-cg, c2 + cg]])
    freqs, participations, signs = _lumped_epr(capacitance, [l1, l2], [0, 1])
    assert np.min(participations) > 0.3  # the modes are strongly hybridized, not bare transmons
    model = EPRModel(freqs, participations, junction_inductances=[l1, l2], signs=signs, labels=["a", "b"])

    inverse = np.linalg.inv(capacitance)
    transmons = [
        ChargeBasisTransmon(E_C=_charging_energy(inverse[i, i]), E_J=_josephson_energy(inductance),
                            n_g=0.25, num_basis=41, levels=10, label=label)
        for i, (inductance, label) in enumerate(((l1, "a"), (l2, "b")))
    ]
    # Node charges couple through C^-1: H = C^-1_12 (2e)^2 n_a n_b.
    coupling = Capacitive(*transmons, g=inverse[0, 1] * (2 * e) ** 2 / h / 1e9, label="ab")
    exact = _spectrum(Chip(transmons, [coupling], approximation=Exact()), 6)

    coarse, fine = (_spectrum(model.chip(levels=n, nonlinearity="cosine"), 6) for n in (12, 14))

    # Measured errors: 1e-7 GHz at 12 levels and 7e-9 at 14; the last increment bounds the rest.
    assert np.max(np.abs(fine - exact)) < np.max(np.abs(fine - coarse))
    assert np.max(np.abs(fine - exact)) < 1e-6


def test_cosine_model_reproduces_the_dispersive_shift_of_a_transmon_resonator_circuit():
    """EPR data of a transmon coupled to an LC resonator give the exact f_q, anharmonicity and chi."""
    cq, cr, cg, lq, lr = 90e-15, 400e-15, 6e-15, 11e-9, 1.2e-9
    capacitance = np.array([[cq + cg, -cg], [-cg, cr + cg]])
    freqs, participations, signs = _lumped_epr(capacitance, [lq, lr], [0])
    model = EPRModel(freqs, participations, junction_inductances=[lq], signs=signs, labels=["q", "r"])

    inverse = np.linalg.inv(capacitance)
    resonator_angular = np.sqrt(inverse[1, 1] / lr)
    charge_zpf = np.sqrt(hbar * resonator_angular / (2 * inverse[1, 1]))
    transmon = ChargeBasisTransmon(E_C=_charging_energy(inverse[0, 0]), E_J=_josephson_energy(lq),
                                   n_g=0.25, num_basis=31, levels=8, label="q")
    resonator = Resonator(freq=resonator_angular / (2 * np.pi * 1e9), levels=8, label="r")
    # C^-1_qr Q_q Q_r with Q_q = 2e n and Q_r = i Q_zpf (a^dag - a); a -> i a maps it onto Capacitive's n (a + a^dag).
    coupling = Capacitive(transmon, resonator, g=inverse[0, 1] * 2 * e * charge_zpf / h / 1e9, label="qr")
    exact = Chip([transmon, resonator], [coupling], approximation=Exact())

    def observables(chip):
        kerr = chip.kerr_matrix()
        return np.array([chip.freq("q"), kerr["q", "q"], kerr["q", "r"]], dtype=float)

    target = observables(exact)
    coarse, fine = (observables(model.chip(levels=n, nonlinearity="cosine")) for n in (10, 12))

    # chi is -0.745 MHz here; the first-order EPR estimate is 16% larger in magnitude.
    assert target[2] == pytest.approx(-7.45e-4, rel=1e-2)
    # Measured errors at 12 levels: 3e-7 GHz (f_q), 1e-5 (anharmonicity), 2e-8 (chi);
    # each is below its 10 -> 12 level truncation increment.
    np.testing.assert_array_less(np.abs(fine - target), np.abs(fine - coarse))
    first_order = model.chip(levels=3, nonlinearity="first_order").kerr_matrix()["q", "r"]
    assert abs(first_order - target[2]) > 100 * abs(fine[2] - target[2])


@pytest.mark.validation
def test_cosine_model_reduces_to_the_first_order_model_as_phase_fluctuations_vanish():
    """Kerr coefficients of the two models differ at relative order phi_zpf^2, not at first order."""
    base = EPRModel([5.0, 7.0], [[0.95], [0.02]], junction_energies=[13.6], signs=[[1], [-1]], labels=["q", "r"])

    def relative_errors(scale):
        model = base.replace(junction_energies=base.junction_energies * scale)
        full = np.asarray(model.chip(levels=(16, 6), nonlinearity="cosine").kerr_matrix().values)
        first = np.asarray(model.chip(levels=3, nonlinearity="first_order").kerr_matrix().values)
        return np.abs(full / first - 1)[[0, 0], [0, 1]], float(model.phi_zpf[0, 0] ** 2)

    (errors_4, phi2_4), (errors_16, phi2_16) = relative_errors(4.0), relative_errors(16.0)

    # Quadrupling E_J quarters phi_zpf^2, so a leading O(phi_zpf^2) correction shrinks fourfold.
    # The O(phi_zpf^4) term shifts the ratio by b (phi2_4 - phi2_16) with |b| ~ 1.3 fitted at
    # E_J x1 -> x4; rtol = 3 phi2_4 = 0.13 allows |b| up to about 4.
    assert phi2_4 / phi2_16 == pytest.approx(4.0)
    np.testing.assert_allclose(errors_4 / errors_16, 4.0, rtol=3 * phi2_4)


@pytest.mark.validation
@pytest.mark.optional_backend
@pytest.mark.parametrize("nonlinearity", ["first_order", "cosine"])
def test_dressed_quantities_are_differentiable_in_the_junction_energy(nonlinearity):
    """jax.grad through EPR inputs matches central differences for f_q and chi_qr."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    def observables(josephson):
        model = EPRModel([5.0, 7.0], [[0.95], [0.02]], junction_energies=josephson, signs=[[1], [-1]],
                         labels=["q", "r"])
        chip = model.chip(levels=(8, 5), nonlinearity=nonlinearity, backend="dynamiqs")
        return jnp.stack([jnp.asarray(chip.freq("q")), chip.kerr_matrix()["q", "r"]])

    point = jnp.array([13.6])
    jacobian = np.asarray(jax.jacrev(observables)(point))[:, 0]
    step = 1e-3  # GHz; central-difference truncation ~ step^2 f''' / 6 is below 1e-8 relative here
    finite = (np.asarray(observables(point + step)) - np.asarray(observables(point - step))) / (2 * step)

    np.testing.assert_allclose(jacobian, finite, rtol=1e-5)
    assert np.all(jacobian != 0)


def _write_pyepr_data(path, freqs, participations, signs, inductances, quality_factors, junction_sums):
    """Write one variation in the layout pyEPR's DistributedAnalysis saves for QuantumAnalysis.

    With U_tot_cap = U_tot_ind = 1 and U_H = 1 - s_m, pyEPR renormalizes mode m's
    participations above 0.15 so that they sum to s_m (``config.epr.renorm_pj = 2``).
    """
    import pickle

    import pandas as pd

    modes = list(range(len(freqs)))
    junctions = [f"j{index}" for index in range(len(inductances))]
    energies = {m: {"U_tot_cap": 1.0, "U_tot_ind": 1.0, "U_H": 1.0 - s, "U_E": 1.0 - s, "U_norm": 1.0}
                for m, s in zip(modes, junction_sums)}
    result = {
        "Pm": pd.DataFrame(participations, index=modes, columns=junctions),
        "Pm_cap": pd.DataFrame(np.zeros_like(participations), index=modes, columns=junctions),
        "Sm": pd.DataFrame(signs, index=modes, columns=junctions),
        "Om": pd.DataFrame({m: pd.Series({"freq_GHz": f}) for m, f in zip(modes, freqs)}),
        "sols": pd.DataFrame({m: pd.Series({"U_H": 1.0, "U_E": 1.0}) for m in modes}).transpose(),
        "Qm_coupling": pd.DataFrame(np.full((len(modes), 1), np.inf), index=modes, columns=["port"]),
        "Ljs": pd.Series(inductances, index=junctions),
        "Cjs": pd.Series(np.full(len(junctions), 2e-15), index=junctions),
        "Qs": pd.Series(quality_factors, index=modes),
        "freqs_hfss_GHz": pd.Series(freqs, index=modes),
        "hfss_variables": pd.Series({"_Lj": "10nH"}),
        "modes": modes,
        "I_peak": pd.Series(dtype=float),
        "V_peak": pd.Series(dtype=float),
        "ansys_energies": energies,
        "mesh": None,
        "convergence": None,
        "convergence_f_pass": None,
    }
    with open(path, "wb") as handle:
        pickle.dump({"project_info": {}, "results": {"0": result}}, handle)


# The comparison uses pyEPR's own fock_trunc=7, which is below convergence on purpose.
@pytest.mark.filterwarnings("ignore:The cosine chip is not converged:UserWarning")
def test_imported_models_reproduce_pyepr_first_order_and_diagonalized_results(tmp_path):
    """from_pyepr chips give pyEPR's own chi_O1, f_1, chi_ND and f_ND after its renormalization."""
    pyepr = pytest.importorskip("pyEPR")
    pytest.importorskip("pandas")
    from quchip import from_pyepr

    path = tmp_path / "epr.pkl"
    _write_pyepr_data(
        path,
        freqs=[4.8, 5.3, 7.1],
        participations=np.array([[0.93, 0.004], [0.012, 0.91], [0.02, 0.03]]),
        signs=np.array([[1, -1], [1, 1], [-1, 1]]),
        inductances=[12e-9, 11e-9],
        quality_factors=[1.5e6, np.inf, 2e4],
        junction_sums=[0.96, 0.93, 0.05],
    )
    analysis = pyepr.QuantumAnalysis(str(path), do_print_info=False)
    reference = analysis.analyze_variation("0", cos_trunc=8, fock_trunc=7, print_result=False)
    model = from_pyepr(analysis)

    first = model.chip(levels=3, nonlinearity="first_order")
    full = model.chip(levels=7, cos_trunc=8)
    labels = model.labels

    # Both sides diagonalize the same double-precision Hamiltonian; pyEPR reports MHz with
    # the opposite sign, and chi_O1's diagonal is the anharmonicity magnitude.
    assert model.participations[0, 0] == pytest.approx(0.93 * 0.96 / 0.934)  # renormalized row
    np.testing.assert_allclose(-1e3 * np.asarray(first.kerr_matrix().values), np.asarray(reference["chi_O1"]),
                               rtol=1e-10)
    np.testing.assert_allclose([1e3 * first.freq(label) for label in labels], np.asarray(reference["f_1"]),
                               rtol=1e-12)
    np.testing.assert_allclose(-1e3 * np.asarray(full.kerr_matrix().values),
                               np.real(np.asarray(reference["chi_ND"])), rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose([1e3 * full.freq(label) for label in labels],
                               np.real(np.asarray(reference["f_ND"])), rtol=1e-11)
    np.testing.assert_array_equal(model.quality_factors, [1.5e6, np.inf, 2e4])


# pyEPR's own fock_trunc = 9 is below convergence on purpose, as in the comparison above.
@pytest.mark.filterwarnings("ignore:The cosine chip is not converged:UserWarning")
def test_imported_pyaedt_analysis_reproduces_its_diagonalization():
    """from_pyepr on pyEPR's PyAEDT analysis gives the f_ND and chi_ND of its own analyze()."""
    pytest.importorskip("pyEPR")
    from types import SimpleNamespace

    from pyEPR.ansys_pyaedt import PyaedtDistributedAnalysis

    from quchip import from_pyepr

    # The arrays do_EPR_analysis() stores for pyEPR's PyAEDT demo transmon.
    analysis = PyaedtDistributedAnalysis(SimpleNamespace(junctions={"j1": {}}))
    analysis.freqs_GHz = np.array([4.815509, 5.060004])
    analysis.Ljs = np.array([12.9201e-9])
    analysis.PJ = np.array([[0.9755], [0.0061]])
    analysis.SJ = np.array([[-1.0], [1.0]])
    f_nd, chi_nd = analysis.analyze(cos_trunc=8, fock_trunc=9)
    chip = from_pyepr(analysis).chip(levels=9, cos_trunc=8)

    # pyEPR returns f_ND in Hz and chi_ND in MHz with the opposite sign. Its PyAEDT path forms
    # phi_zpf with rounded constants whose (Phi_0/2pi)^2/h is 1.5e-8 below the value its
    # Hamiltonian uses, which moves chi by about twice that (measured: 3.6e-8) and the
    # frequencies by less (1.7e-9); quchip's constants match the Hamiltonian's to 1e-15.
    np.testing.assert_allclose([chip.freq(label) for label in ("mode_0", "mode_1")], np.real(f_nd) / 1e9,
                               rtol=1e-8)
    np.testing.assert_allclose(-1e3 * np.asarray(chip.kerr_matrix().values), np.real(chi_nd), rtol=1e-7, atol=1e-9)
