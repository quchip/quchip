import numpy as np
import pytest

from quchip.interop.eigenbasis import EigenbasisDevice


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


def test_physics_notes_declare_frozen_import():
    """physics_notes() declares the frozen-at-import snapshot for a device with a source_type."""
    assert any("frozen at import" in n for n in _dev(source_type="scqubits.ZeroPi").physics_notes())


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
