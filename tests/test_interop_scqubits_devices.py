"""Spectrum-identity and dispatch tests for the shipped scqubits mappings.

Each transcription mapping is checked against scqubits as the oracle: the
imported quchip device must reproduce the source object's lowest transition
energies. Export round-trips and the concrete-parameter guard are covered
alongside the registry dispatch (unmapped type -> ``LookupError``).
"""

from __future__ import annotations

import numpy as np
import pytest

scq = pytest.importorskip("scqubits")

from quchip import from_scqubits, to_scqubits  # noqa: E402
from quchip.backend import get_default_backend  # noqa: E402
from quchip.declarative.expr import materialize_expr  # noqa: E402
from quchip.devices import DuffingTransmon, Resonator  # noqa: E402
from quchip.devices.fluxonium import Fluxonium  # noqa: E402
from quchip.devices.transmon import ChargeBasisTransmon  # noqa: E402


def _ground_shifted(evals: np.ndarray) -> np.ndarray:
    """Return the real spectrum shifted so the ground state sits at zero."""
    evals = np.real(np.sort(np.asarray(evals)))
    return evals - evals[0]


def _device_spectrum(device, count: int) -> np.ndarray:
    """Return the device's lowest ``count`` transition energies (``E_0 = 0``)."""
    backend = get_default_backend()
    matrix = np.asarray(backend.to_array(materialize_expr(device.hamiltonian(), backend)))
    return _ground_shifted(np.linalg.eigvalsh(matrix)[:count])


def _oracle_spectrum(obj, count: int) -> np.ndarray:
    """Return the scqubits object's lowest ``count`` eigenvalues (``E_0 = 0``)."""
    return _ground_shifted(obj.eigenvals(evals_count=count))


# ---------------------------------------------------------------------------
# Transmon
# ---------------------------------------------------------------------------


def test_transmon_import_matches_scqubits_spectrum_exactly():
    """A Transmon's imported spectrum matches scqubits' to near machine precision."""
    tmon = scq.Transmon(EJ=30.0, EC=0.2, ng=0.25, ncut=31, truncated_dim=5)
    dev = from_scqubits(tmon)
    got = _device_spectrum(dev, 5)
    want = _oracle_spectrum(tmon, 5)
    assert np.allclose(got, want, atol=1e-9)


def test_transmon_roundtrip_export():
    """import(export(transmon)) reproduces the original scqubits Transmon's spectrum."""
    tmon = scq.Transmon(EJ=30.0, EC=0.2, ng=0.0, ncut=31, truncated_dim=4)
    back = to_scqubits(from_scqubits(tmon))
    assert isinstance(back, scq.Transmon)
    assert np.allclose(back.eigenvals(4), tmon.eigenvals(4), atol=1e-9)


# ---------------------------------------------------------------------------
# Label and noise-kwarg forwarding
# ---------------------------------------------------------------------------


def test_import_forwards_noise_kwargs():
    """T1/T2/thermal_occupation reach the imported device unchanged, given an explicit coupling_channel."""
    tmon = scq.Transmon(EJ=30.0, EC=0.2, ng=0.0, ncut=31, truncated_dim=3)
    dev = from_scqubits(tmon, T1=30000.0, T2=20000.0, thermal_occupation=0.01, coupling_channel="charge")
    assert dev.T1 == 30000.0
    assert dev.T2 == 20000.0
    assert dev.thermal_occupation == 0.01


# ---------------------------------------------------------------------------
# TunableTransmon (import-only)
# ---------------------------------------------------------------------------


def test_tunable_transmon_import_matches_effective_spectrum():
    """A TunableTransmon imports at its flux-evaluated effective E_J, matching scqubits' spectrum."""
    ttmon = scq.TunableTransmon(
        EJmax=30.0, EC=0.2, d=0.1, flux=0.25, ng=0.0, ncut=31, truncated_dim=5
    )
    dev = from_scqubits(ttmon)
    got = _device_spectrum(dev, 5)
    want = _oracle_spectrum(ttmon, 5)
    assert np.allclose(got, want, atol=1e-9)


# ---------------------------------------------------------------------------
# Fluxonium
# ---------------------------------------------------------------------------


def test_fluxonium_import_matches_scqubits_spectrum():
    """A Fluxonium's imported spectrum matches scqubits' at the cross-discretization tolerance."""
    flx = scq.Fluxonium(EJ=8.9, EC=2.5, EL=0.5, flux=0.5, cutoff=110, truncated_dim=5)
    dev = from_scqubits(flx)
    got = _device_spectrum(dev, 5)
    want = _oracle_spectrum(flx, 5)
    # quchip's plane-wave phase grid differs from scqubits' harmonic basis;
    # the physics is identical but the discretizations are not, so the two
    # spectra agree only to a relative tolerance (~1.5e-4), not to machine
    # precision.
    assert np.allclose(got, want, rtol=5e-4, atol=1e-6)


def test_fluxonium_roundtrip_export():
    """import(export(fluxonium)) reproduces the original scqubits Fluxonium's spectrum."""
    flx = scq.Fluxonium(EJ=8.9, EC=2.5, EL=0.5, flux=0.5, cutoff=110, truncated_dim=4)
    back = to_scqubits(from_scqubits(flx))
    assert isinstance(back, scq.Fluxonium)
    assert np.allclose(back.eigenvals(4), flx.eigenvals(4), atol=1e-6)


# ---------------------------------------------------------------------------
# Oscillator
# ---------------------------------------------------------------------------


def test_oscillator_import_matches_spectrum():
    """An Oscillator's imported spectrum matches scqubits' to near machine precision."""
    osc = scq.Oscillator(E_osc=5.0, truncated_dim=6)
    dev = from_scqubits(osc)
    got = _device_spectrum(dev, 6)
    want = _oracle_spectrum(osc, 6)
    assert np.allclose(got, want, atol=1e-9)


def test_oscillator_roundtrip_export():
    """import(export(oscillator)) reproduces the original scqubits Oscillator's spectrum."""
    osc = scq.Oscillator(E_osc=5.0, truncated_dim=6)
    back = to_scqubits(from_scqubits(osc))
    assert isinstance(back, scq.Oscillator)
    assert np.allclose(back.eigenvals(6), osc.eigenvals(6), atol=1e-9)


# ---------------------------------------------------------------------------
# KerrOscillator (the sign trap)
# ---------------------------------------------------------------------------


def test_kerr_oscillator_import_matches_spectrum():
    """A KerrOscillator's imported spectrum matches scqubits' K > 0 (self-focusing) convention."""
    kerr = scq.KerrOscillator(E_osc=5.0, K=0.05, truncated_dim=6)
    dev = from_scqubits(kerr)
    got = _device_spectrum(dev, 6)
    want = _oracle_spectrum(kerr, 6)
    assert np.allclose(got, want, atol=1e-9)


# ---------------------------------------------------------------------------
# DuffingTransmon (export-only)
# ---------------------------------------------------------------------------


def test_duffing_transmon_export_uses_consistent_ncut():
    """DuffingTransmonMapping.export_model uses the same ncut for E_J/E_C inversion and the built Transmon."""
    dev = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")

    default_export = to_scqubits(dev)
    assert default_export.ncut == 30

    custom_export = to_scqubits(dev, ncut=45)
    assert custom_export.ncut == 45
    e_j, e_c = scq.Transmon.find_EJ_EC(E01=5.0, anharmonicity=-0.25, ncut=45)
    assert np.isclose(custom_export.EJ, e_j)
    assert np.isclose(custom_export.EC, e_c)


# ---------------------------------------------------------------------------
# GenericQubit (import-only)
# ---------------------------------------------------------------------------


def test_generic_qubit_import_matches_splitting():
    """A GenericQubit imports as a two-level DuffingTransmon with the same level splitting."""
    gq = scq.GenericQubit(E=4.5)
    dev = from_scqubits(gq)
    got = _device_spectrum(dev, 2)
    want = _oracle_spectrum(gq, 2)
    assert np.allclose(got, want, atol=1e-9)


# ---------------------------------------------------------------------------
# scqubits energy units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("make_source", "count", "tolerance"),
    [
        (lambda: scq.Transmon(EJ=30e3, EC=200.0, ng=0.25, ncut=31, truncated_dim=5), 5, {"atol": 1e-9}),
        (lambda: scq.TunableTransmon(EJmax=30e3, EC=200.0, d=0.1, flux=0.25, ng=0.0, ncut=31, truncated_dim=5),
         5, {"atol": 1e-9}),
        # Cross-discretization tolerance, as in test_fluxonium_import_matches_scqubits_spectrum.
        (lambda: scq.Fluxonium(EJ=8.9e3, EC=2.5e3, EL=0.5e3, flux=0.5, cutoff=110, truncated_dim=5), 5,
         {"rtol": 5e-4, "atol": 1e-6}),
        (lambda: scq.Oscillator(E_osc=5e3, truncated_dim=6), 6, {"atol": 1e-9}),
        (lambda: scq.KerrOscillator(E_osc=5e3, K=50.0, truncated_dim=6), 6, {"atol": 1e-9}),
        (lambda: scq.GenericQubit(E=4.5e3), 2, {"atol": 1e-9}),
        pytest.param(lambda: _small_zero_pi(units_per_ghz=1e3), 4, {"atol": 1e-8}, marks=pytest.mark.validation),
    ],
    ids=["transmon", "tunable_transmon", "fluxonium", "oscillator", "kerr_oscillator", "generic_qubit", "zero_pi"],
)
def test_import_converts_scqubits_units_to_ghz(scqubits_mhz, make_source, count, tolerance):
    """A source built in MHz imports with its scqubits spectrum converted to GHz."""
    source = make_source()
    np.testing.assert_allclose(
        _device_spectrum(from_scqubits(source), count), _oracle_spectrum(source, count) / 1e3, **tolerance
    )


@pytest.mark.parametrize(
    ("device", "count", "tolerance"),
    [
        (ChargeBasisTransmon(E_C=0.2, E_J=30.0, n_g=0.25, num_basis=63, levels=5, basis="eigen"), 5,
         {"atol": 1e-9}),
        (Fluxonium(E_C=2.5, E_J=8.9, E_L=0.5, phi_ext=0.5, levels=5, basis="eigen"), 5,
         {"rtol": 5e-4, "atol": 1e-6}),
        (Resonator(freq=5.0, levels=6), 6, {"atol": 1e-9}),
    ],
    ids=["transmon", "fluxonium", "resonator"],
)
def test_export_writes_scqubits_units(scqubits_mhz, device, count, tolerance):
    """An export under MHz reproduces the device spectrum in MHz."""
    np.testing.assert_allclose(
        _oracle_spectrum(to_scqubits(device), count) / 1e3, _device_spectrum(device, count), **tolerance
    )


def test_duffing_export_inverts_in_scqubits_units(scqubits_mhz):
    """A Duffing export under MHz keeps its GHz frequency and anharmonicity."""
    exported = to_scqubits(DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3))
    levels = _oracle_spectrum(exported, 3) / 1e3
    np.testing.assert_allclose([levels[1], levels[2] - 2 * levels[1]], [5.0, -0.25], atol=1e-6)


# ---------------------------------------------------------------------------
# Dispatch and guards
# ---------------------------------------------------------------------------


def test_unmapped_type_raises_lookup_error():
    """A scqubits type with no registered mapping raises LookupError naming the missing mapping."""
    fq = scq.FluxQubit(**scq.FluxQubit.default_params())
    with pytest.raises(LookupError, match="ModelMapping"):
        from_scqubits(fq)


def test_export_of_traced_device_raises_value_error():
    """Exporting a device with a JAX-traced parameter raises ValueError rather than silently dropping it."""
    import jax

    @jax.jit
    def export_traced(e):
        return to_scqubits(ChargeBasisTransmon(E_C=0.2, E_J=e))

    with pytest.raises(ValueError):
        export_traced(30.0)


# ---------------------------------------------------------------------------
# ZeroPi (frozen snapshot, no native quchip model)
# ---------------------------------------------------------------------------


def _small_zero_pi(units_per_ghz: float = 1.0):
    """A deliberately small ZeroPi so the test stays a few seconds, with energies in the given unit."""
    grid = scq.Grid1d(-19.0, 19.0, 200)
    return scq.ZeroPi(
        grid=grid,
        EJ=10.0 * units_per_ghz,
        EL=0.04 * units_per_ghz,
        ECJ=20.0 * units_per_ghz,
        EC=0.04 * units_per_ghz,
        ng=0.1,
        flux=0.23,
        ncut=30,
        truncated_dim=4,
    )


@pytest.mark.validation
def test_zero_pi_import_matches_scqubits_spectrum_exactly():
    """One imported ZeroPi preserves both its independent spectrum and charge matrix elements."""
    zp = _small_zero_pi()
    dev = from_scqubits(zp)
    esys = zp.eigensys(evals_count=zp.truncated_dim)
    np.testing.assert_allclose(_device_spectrum(dev, 4), _ground_shifted(esys[0]), atol=1e-8)
    expected_charge = np.abs(np.asarray(zp.n_theta_operator(energy_esys=esys)))
    imported_charge = np.abs(np.asarray(get_default_backend().to_array(dev.charge_coupling_operator())))
    np.testing.assert_allclose(imported_charge, expected_charge, atol=1e-8)
