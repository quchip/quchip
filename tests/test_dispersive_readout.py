"""Dispersive pull, intrinsic decay diagnostics, and steady-state detected readout from a chip."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import quchip
from quchip import (
    RWA,
    VNA,
    Capacitive,
    Chip,
    DuffingTransmon,
    Exact,
    PortNetwork,
    Resonator,
    analyze_dispersive_readout,
)
from quchip.chip.transformations import eliminate

TWO_PI = 2 * np.pi


# ---------------------------------------------------------------------------
# Part A — χ/κ reported by eliminate()
# ---------------------------------------------------------------------------


def _readout_chip(g=0.05):
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    r = Resonator(freq=6.5, internal_quality_factor=5000.0, levels=6, label="r")
    return Chip([q, r], couplings=[Capacitive(q, r, g=g)]), q, r


@pytest.mark.validation
def test_eliminate_reports_kappa_matching_purcell_kappa():
    """eliminate() reports kappa = 2*pi*f_r/Q (1/ns), the same rate the Purcell fold uses."""
    chip, q, r = _readout_chip()
    res = eliminate(chip, r)
    assert float(res.effective_params["q"]["kappa"]) == pytest.approx(2 * np.pi * 6.5 / 5000.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Part B — steady-state pointer states from the chip's pulls and linewidth
# ---------------------------------------------------------------------------


def readout(g=0.05, *, rate=0.005, internal_q=None, qubit_freq=5.0, resonator_levels=10, T2=None,
            approximation=RWA(), backend="qutip", port=True):
    """A transmon dispersively coupled to a resonator with one reflection port ``p``."""
    q = DuffingTransmon(freq=qubit_freq, anharmonicity=-0.25, levels=5, label="q", T2=T2)
    loss = {} if internal_q is None else {"internal_quality_factor": internal_q}
    r = Resonator(freq=6.5, levels=resonator_levels, label="r", **loss)
    network = None
    if port:
        network = PortNetwork(label="line")
        reflection = network.port("r_port", target=r, rate=rate)
        network.expose("p", input=reflection.input, output=reflection.output)
    return Chip([q, r], [Capacitive(q, r, g=g)], port_network=network, approximation=approximation,
                backend=backend)


def test_pointer_states_coincide_without_a_pull():
    """Without coupling the pull is zero, so the pointer states coincide and SNR vanishes."""
    ro = analyze_dispersive_readout(readout(g=0.0), "q", "r", tau=500.0, n_photons=2.0)

    assert float(ro.chi) == 0.0
    np.testing.assert_allclose(ro.pointer_states[0], ro.pointer_states[1])
    assert float(ro.snr) == pytest.approx(0.0, abs=1e-12)
    assert float(ro.assignment_error) == pytest.approx(0.5, rel=1e-9)
    assert any("steady-state" in note.lower() for note in ro.notes)


def test_photon_number_reproduces_requested_n_photons():
    """The requested n_photons is reproduced whether driven by n_photons or the equivalent eps."""
    chip = readout()
    for delta_r in (0.0, 0.003):
        ro = analyze_dispersive_readout(chip, "q", "r", tau=500.0, n_photons=2.0, delta_r=delta_r)
        assert float(ro.photon_numbers[0]) == pytest.approx(2.0, rel=1e-9)

    eps = np.sqrt(2.0 * ((float(ro.kappa) / 2) ** 2 + (TWO_PI * 0.003) ** 2))
    driven = analyze_dispersive_readout(chip, "q", "r", tau=500.0, eps=eps, delta_r=0.003)
    assert float(driven.photon_numbers[0]) == pytest.approx(2.0, rel=1e-9)


def test_snr_grows_as_sqrt_tau():
    """SNR scales with sqrt(tau): a 4x integration time gives a 2x SNR."""
    chip = readout()
    snr_1 = float(analyze_dispersive_readout(chip, "q", "r", tau=400.0, n_photons=2.0).snr)
    snr_4 = float(analyze_dispersive_readout(chip, "q", "r", tau=1600.0, n_photons=2.0).snr)
    assert snr_4 / snr_1 == pytest.approx(2.0, rel=1e-9)


@pytest.mark.parametrize("delta_r", [0.0, 0.0005])
def test_dephasing_rate_matches_small_chi_limit(delta_r):
    """Small-pull dephasing includes the analytic detuning suppression."""
    ro = analyze_dispersive_readout(readout(g=5e-4), "q", "r", tau=500.0, n_photons=5.0, delta_r=delta_r)

    # Z has positive ground eigenvalue, so chi_Z,angular = -pi*chi_pull.
    chi, kappa = float(ro.chi), float(ro.kappa)
    assert 0 < abs(chi) < 1e-7
    expected = 8 * (np.pi * chi) ** 2 * 5.0 / kappa / (1 + (4 * np.pi * delta_r / kappa) ** 2)
    assert float(ro.dephasing_rate) == pytest.approx(expected, rel=1e-3)


def test_each_pointer_state_uses_the_dressed_pull_of_its_level():
    """Level j is pulled by f_r|j - f_r|0 from the dressed spectrum, not by j*chi."""
    chip = readout()
    ro = analyze_dispersive_readout(chip, "q", "r", tau=500.0, n_photons=2.0, levels=3)

    expected = [chip.transition_frequency("r", 0, 1, when={"q": j}) for j in range(3)]
    np.testing.assert_allclose(ro.resonator_frequencies, expected, rtol=1e-14)
    pull_2 = expected[2] - expected[0]
    assert abs(pull_2 - 2 * float(ro.chi)) > 0.1 * abs(float(ro.chi))
    eps = np.sqrt(2.0) * float(ro.kappa) / 2
    alpha_2 = -1j * eps / (float(ro.kappa) / 2 + 1j * TWO_PI * pull_2)
    assert complex(ro.pointer_states[2]) == pytest.approx(alpha_2, rel=1e-12)
    assert np.shape(ro.photon_numbers) == (3,)


def test_detected_loss_follows_the_output_port():
    """Internal loss lowers the detected part only: the pointer states and dephasing keep the total kappa."""
    chip = readout(internal_q=2e4)
    total = analyze_dispersive_readout(chip, "q", "r", tau=500.0, n_photons=2.0)
    port = analyze_dispersive_readout(chip, "q", "r", tau=500.0, n_photons=2.0, output="p")

    assert float(total.kappa_detected) == float(total.kappa)
    assert 0 < float(port.kappa_detected) < float(port.kappa)
    np.testing.assert_allclose(port.pointer_states, total.pointer_states, atol=1e-14)
    assert float(port.dephasing_rate) == pytest.approx(float(total.dephasing_rate))
    ratio = np.sqrt(float(port.kappa_detected) / float(port.kappa))
    assert float(port.snr) == pytest.approx(float(total.snr) * ratio, rel=1e-12)
    halved = analyze_dispersive_readout(chip, "q", "r", tau=500.0, n_photons=2.0, output="p", efficiency=0.25)
    assert float(halved.snr) == pytest.approx(float(port.snr) / 2, rel=1e-12)
    dark = analyze_dispersive_readout(chip, "q", "r", tau=500.0, n_photons=2.0, output="p", efficiency=0.0)
    assert float(dark.snr) == 0.0
    assert float(dark.assignment_error) == 0.5
    assert np.isinf(dark.sigma)


def test_nonlinearity_reports_the_pull_drift_over_the_pointer_photons():
    """A weak drive samples the Poisson support of each pointer state and reports small drifts."""
    ro = analyze_dispersive_readout(readout(resonator_levels=12), "q", "r", tau=500.0, n_photons=2.0)
    check = ro.nonlinearity

    assert check["status"] == "checked" and check["reason"] is None
    assert check["photon_ranges"][0] == (0, 8)
    assert np.all(np.asarray(check["drift_over_hwhm"]) < 0.1)
    np.testing.assert_allclose(check["drift_over_hwhm"],
                               np.asarray(check["frequency_drift"]) / (float(ro.kappa) / (4 * np.pi)), rtol=1e-12)
    assert 0 < check["pull_change"] < 0.05


@pytest.mark.unit
def test_nonlinearity_is_unchecked_when_the_resonator_is_truncated():
    """A truncation that cannot hold the occupied photon numbers leaves the check unchecked."""
    ro = analyze_dispersive_readout(readout(resonator_levels=4), "q", "r", tau=500.0, n_photons=2.0)
    assert ro.nonlinearity["status"] == "unchecked"
    assert "resonator keeps 4 levels" in ro.nonlinearity["reason"]


@pytest.mark.unit
def test_exactly_one_of_n_photons_and_eps_is_required():
    """Passing neither or both of n_photons/eps raises ValueError."""
    chip = readout(resonator_levels=4)
    with pytest.raises(ValueError, match="xactly one"):
        analyze_dispersive_readout(chip, "q", "r", tau=500.0)
    with pytest.raises(ValueError, match="xactly one"):
        analyze_dispersive_readout(chip, "q", "r", tau=500.0, n_photons=2.0, eps=0.01)


@pytest.mark.unit
@pytest.mark.parametrize("name,value", [
    ("tau", 0.0), ("tau", -1.0), ("n_photons", -1.0), ("levels", 1), ("levels", 2.5),
    ("efficiency", 1.1), ("efficiency", -0.1), ("delta_r", np.nan),
])
def test_readout_rejects_invalid_concrete_domains(name, value):
    """Concrete parameters outside the model's physical domains raise before calculation."""
    arguments = dict(tau=500.0, n_photons=2.0)
    arguments[name] = value
    with pytest.raises(ValueError, match=name):
        analyze_dispersive_readout(readout(resonator_levels=4), "q", "r", **arguments)


@pytest.mark.unit
def test_readout_needs_a_weak_probe_route_and_resonator_loss():
    """Counter-rotating terms leave no weak-probe route, and a lossless resonator has no linewidth."""
    with pytest.raises(ValueError, match="weak-probe route"):
        analyze_dispersive_readout(readout(resonator_levels=4, approximation=Exact()), "q", "r",
                                   tau=500.0, n_photons=2.0)
    with pytest.raises(ValueError, match="no energy loss"):
        analyze_dispersive_readout(readout(resonator_levels=4, port=False), "q", "r", tau=500.0, n_photons=2.0)
    with pytest.raises(ValueError, match="different devices"):
        analyze_dispersive_readout(readout(resonator_levels=4), "r", "r", tau=500.0, n_photons=2.0)


def test_readout_is_jax_traceable_and_differentiable():
    """SNR is differentiable and jit-compilable in a chip parameter; kappa_detected needs eigenvectors."""
    pytest.importorskip("dynamiqs")
    import jax

    def snr(g, output=None):
        chip = readout(g, resonator_levels=6, backend="dynamiqs")
        return analyze_dispersive_readout(chip, "q", "r", tau=500.0, n_photons=1.0, output=output).snr

    value, slope = jax.value_and_grad(snr)(0.05)
    step = 1e-5
    reference = (snr(0.05 + step) - snr(0.05 - step)) / (2 * step)
    assert float(slope) == pytest.approx(float(reference), rel=1e-6)
    assert float(jax.jit(snr)(0.05)) == pytest.approx(float(value), rel=1e-12)
    assert float(jax.jit(lambda g: snr(g, "p"))(0.05)) == pytest.approx(float(snr(0.05, "p")), rel=1e-12)
    with pytest.raises(NotImplementedError, match="eigenvector"):
        jax.grad(lambda g: snr(g, "p"))(0.05)


# ---------------------------------------------------------------------------
# Part C — validation against the dispersive limit and the weak-probe response
# ---------------------------------------------------------------------------


@pytest.mark.validation
def test_pull_approaches_the_duffing_dispersive_limit():
    """chi tends to 2g²α/[Δ(Δ+α)] for the RWA exchange g, with a correction of relative order (g/Δ)²."""
    errors = []
    for g in (0.04, 0.02, 0.01):
        ro = analyze_dispersive_readout(readout(g, resonator_levels=4), "q", "r", tau=500.0, n_photons=1.0)
        delta, alpha = 5.0 - 6.5, -0.25
        errors.append(abs(float(ro.chi) / (2 * g**2 * alpha / (delta * (delta + alpha))) - 1))
    assert errors[0] < 3e-3
    np.testing.assert_allclose(np.asarray(errors[:-1]) / np.asarray(errors[1:]), 4.0, rtol=0.02)


def _two_mode_pole(g, kappa):
    """Resonator-like eigenvalue of -i[[ω_q, g_ω], [g_ω, ω_r]] - diag(0, κ/2) and its qubit amplitude ratio."""
    omega_q, omega_r, coupling = TWO_PI * 5.0, TWO_PI * 6.5, TWO_PI * g
    mean = (-1j * (omega_q + omega_r) - kappa / 2) / 2
    half = (-1j * (omega_q - omega_r) + kappa / 2) / 2
    roots = [mean + np.sqrt(half**2 - coupling**2), mean - np.sqrt(half**2 - coupling**2)]
    pole = min(roots, key=lambda value: abs(-value.imag - omega_r))
    # The first row of (A - λ)v = 0 gives v_q = -i g_ω v_r / (λ + i ω_q).
    return pole, abs(coupling / (pole + 1j * omega_q)) ** 2


@pytest.mark.validation
def test_linewidths_match_the_two_mode_drift_for_reflection_hanger_and_reference_planes():
    """kappa is the exact pole linewidth, and kappa_detected is the port's share of the dressed resonator.

    A reflection port gets the dressed external rate, a symmetric hanger half of it per direction, and a reference
    filter with power transmission η scales it by η.
    """
    g, external, internal_q = 0.05, 0.005, 2e4
    kappa_r = external + TWO_PI * 6.5 / internal_q
    pole, qubit_ratio = _two_mode_pole(g, kappa_r)
    participation = 1 / (1 + qubit_ratio)

    chip = readout(g, rate=external, internal_q=internal_q, resonator_levels=4)
    ro = analyze_dispersive_readout(chip, "q", "r", tau=500.0, n_photons=1.0, output="p")
    assert float(ro.kappa) == pytest.approx(-2 * pole.real, rel=1e-11)
    assert float(ro.kappa_detected) == pytest.approx(external * participation, rel=1e-11)
    # The leading dispersive participation, with an O((g/Δ)²) error.
    theta = 0.5 * np.arctan2(2 * g, 6.5 - 5.0)
    assert float(ro.kappa) == pytest.approx(kappa_r * np.cos(theta) ** 2, rel=1e-6)

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=5, label="q")
    r = Resonator(freq=6.5, levels=4, internal_quality_factor=internal_q, label="r")
    hanger = PortNetwork(label="hanger")
    right, left = (hanger.port(f"r_{side}", target=r, rate=external / 2) for side in ("right", "left"))
    filtered = hanger.filter("cable", transfer=lambda frequency, eta: np.sqrt(eta) + 0 * frequency, eta=0.3)
    hanger.link(right, filtered)
    hanger.expose("right", at=filtered.port(2))
    hanger.expose("left", input=left.input, output=left.output)
    chip = Chip([q, r], [Capacitive(q, r, g=g)], port_network=hanger, approximation=RWA())
    for output, share in (("left", 0.5), ("right", 0.5 * 0.3)):
        ro = analyze_dispersive_readout(chip, "q", "r", tau=500.0, n_photons=1.0, output=output)
        assert float(ro.kappa) == pytest.approx(-2 * pole.real, rel=1e-11)
        assert float(ro.kappa_detected) == pytest.approx(share * external * participation, rel=1e-11)


@pytest.mark.validation
def test_pointer_separation_approaches_the_conditional_vna_response():
    """The conditional VNA response, which makes no dispersive expansion, approaches the pointer-state contrast.

    For a weak probe, b_out(j) = S(f; j) β, so |S(f;1) - S(f;0)| / |S(f;0) - D| tends to |α₁ - α₀| / |α₀|. On
    resonance, |S - D| = sqrt(κ_det κ_in)/(κ/2), which is 2 κ_det/κ for a reflection port.
    """
    for g, tolerance in ((0.05, 5e-3), (0.025, 1e-3)):
        chip = readout(g, internal_q=2e4, resonator_levels=4)
        vna = VNA(chip)
        direct = complex(vna.resonances().direct[0, 0])
        kappa = float(analyze_dispersive_readout(chip, "q", "r", tau=1.0, n_photons=1.0).kappa)
        for delta_r in (0.0, kappa / (4 * np.pi)):
            ro = analyze_dispersive_readout(chip, "q", "r", tau=1.0, n_photons=1.0, output="p", delta_r=delta_r)
            drive = np.asarray([float(ro.resonator_frequencies[0]) - delta_r])
            ground, excited = (complex(vna.sweep(drive, when={"q": j}).matrix[0, 0, 0]) for j in (0, 1))
            alpha_0, alpha_1 = (complex(value) for value in ro.pointer_states[:2])
            contrast = abs(excited - ground) / abs(ground - direct)
            assert contrast == pytest.approx(abs(alpha_1 - alpha_0) / abs(alpha_0), rel=tolerance)
            if delta_r == 0.0:
                expected = 2 * float(ro.kappa_detected) / float(ro.kappa)
                assert abs(ground - direct) == pytest.approx(expected, rel=tolerance)


@pytest.mark.validation
def test_kappa_is_the_energy_linewidth_under_qubit_dephasing():
    """Qubit dephasing broadens the resonator pole but not its energy decay, which kappa reports.

    At Δ = 1 GHz, g = 50 MHz, κ_r = 0.01/ns and γ_φ = 0.02/ns, the pole is 1% wider than the energy-loss rate. The
    initial decay of the total excitation number from the dressed resonator state follows kappa.
    """
    rates = {}
    for T2 in (None, 200.0, 50.0):
        q = DuffingTransmon(freq=5.5, anharmonicity=-0.25, levels=3, label="q", T2=T2)
        r = Resonator(freq=6.5, levels=3, label="r")
        network = PortNetwork(label="line")
        reflection = network.port("r_port", target=r, rate=0.01)
        network.expose("p", input=reflection.input, output=reflection.output)
        chip = Chip([q, r], [Capacitive(q, r, g=0.05)], port_network=network, approximation=RWA(), frame="rotating")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ro = analyze_dispersive_readout(chip, "q", "r", tau=1.0, n_photons=0.1, output="p")
        times = np.linspace(0.0, 2.0, 21)
        result = quchip.simulate(chip, [], times, initial_state=chip.state({"q": 0, "r": 1}),
                                 e_ops={q: q.number_operator(), r: r.number_operator()},
                                 options={"atol": 1e-12, "rtol": 1e-10})
        excitations = np.real(result.expect(q)) + np.real(result.expect(r))
        rates[T2] = (float(ro.kappa), float(ro.kappa_detected), float(VNA(chip).resonances()["r"].linewidth),
                     -np.polyfit(times, excitations, 3)[-2])

    kappa = rates[None][0]
    for T2, (value, detected, pole, decay) in rates.items():
        assert value == pytest.approx(kappa, rel=1e-13) and detected == pytest.approx(kappa, rel=1e-12)
        assert decay == pytest.approx(kappa, rel=1e-5)
    assert rates[None][2] == pytest.approx(kappa, rel=1e-12)
    assert rates[50.0][2] == pytest.approx(0.01007444, rel=1e-6)
    assert rates[50.0][2] / kappa - 1 > 100 * abs(rates[50.0][3] / kappa - 1)
