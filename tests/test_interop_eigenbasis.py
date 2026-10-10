import numpy as np
import pytest

from quchip.interop.eigenbasis import EigenbasisDevice


pytestmark = pytest.mark.unit


def _dev(**kw):
    E = np.array([0.0, 5.0, 9.8])
    n = np.array([[0, 1.0, 0], [1.0, 0, 1.3], [0, 1.3, 0]], dtype=complex)
    return EigenbasisDevice(E, charge_operator=n, **kw)


def test_spectrum_and_freq():
    """The stored energies are ground-shifted and freq reads the 0->1 gap."""
    d = _dev()
    assert np.allclose(np.asarray(d.eigenenergies()), [0.0, 5.0, 9.8])
    assert float(d.freq) == pytest.approx(5.0)


def test_charge_operator_projected_identically():
    """The stored charge operator's magnitude matches the supplied matrix elementwise."""
    d = _dev()
    got = np.asarray(d.charge_coupling_operator())
    assert np.allclose(np.abs(got), np.abs([[0, 1.0, 0], [1.0, 0, 1.3], [0, 1.3, 0]]))


def test_missing_phase_operator_raises_with_guidance():
    """Requesting a phase operator that was never supplied raises ValueError with guidance."""
    with pytest.raises(ValueError, match="phase_operator"):
        _dev().phase_coupling_operator()


def test_capacitive_coupling_without_charge_operator_raises_with_guidance():
    """A capacitive coupling to a model imported without a charge operator raises ValueError."""
    from quchip import Capacitive, Resonator

    bare = EigenbasisDevice(np.array([0.0, 5.0, 9.8]), label="bare")
    with pytest.raises(ValueError, match="capacitive couplings require"):
        Capacitive(bare, Resonator(freq=7.0, levels=2, label="r"), g=0.02).interaction_hamiltonian()


def test_roundtrip_serialization():
    """to_dict()/from_dict() round-trips the spectrum and charge operator unchanged."""
    d = _dev(label="zp", T1=50_000.0, coupling_channel="charge")
    d2 = EigenbasisDevice.from_dict(d.to_dict())
    assert np.allclose(np.asarray(d2.eigenenergies()), np.asarray(d.eigenenergies()))
    assert np.allclose(np.asarray(d2.charge_coupling_operator()), np.asarray(d.charge_coupling_operator()))


def test_tunable_param_names_pinned_empty():
    """EigenbasisDevice pins tunable_param_names empty: fitting an imported fixed spectrum is meaningless."""
    d = _dev()
    assert d.tunable_param_names == ()
    assert d.tunable_params() == {}
