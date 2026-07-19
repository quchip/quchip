"""Tests for ChargeBasisTransmon — charge-basis transmon model."""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

from quchip.utils.labeling import reset_label_counters
from quchip.devices.protocols import ChargeCoupled, FluxCoupled, PhaseCoupled
from quchip.devices.transmon.charge_basis import ChargeBasisTransmon


@pytest.fixture(autouse=True)
def _reset():
    reset_label_counters()
    yield


# ----------------------------------------------------------------------
# Construction & validation
# ----------------------------------------------------------------------


def test_constructor_accepts_energies():
    """Positive E_C and E_J construct a device with the requested level count."""
    q = ChargeBasisTransmon(E_C=0.25, E_J=20.0, levels=3)
    assert q.levels == 3


def test_invalid_energies_raise_on_concrete():
    """Non-positive E_C or E_J raises ValueError naming the invalid parameter."""
    with pytest.raises(ValueError, match="E_C"):
        ChargeBasisTransmon(E_C=-0.1, E_J=20.0)
    with pytest.raises(ValueError, match="E_J"):
        ChargeBasisTransmon(E_C=0.25, E_J=-1.0)


def test_num_basis_must_be_odd():
    """num_basis must be odd; an even charge-basis cutoff raises ValueError."""
    with pytest.raises(ValueError, match="num_basis"):
        ChargeBasisTransmon(E_C=0.25, E_J=20.0, num_basis=60)


def test_is_charge_coupled_and_phase_coupled():
    """ChargeBasisTransmon satisfies both the ChargeCoupled and PhaseCoupled protocols."""
    q = ChargeBasisTransmon(E_C=0.25, E_J=20.0)
    assert isinstance(q, ChargeCoupled)
    assert isinstance(q, PhaseCoupled)


def test_is_not_flux_coupled():
    """Fixed-frequency charge-basis transmon has no flux DOF."""
    q = ChargeBasisTransmon(E_C=0.25, E_J=20.0)
    assert not isinstance(q, FluxCoupled)


def test_computational_property():
    """ChargeBasisTransmon is classified as a computational qubit."""
    q = ChargeBasisTransmon(E_C=0.25, E_J=20.0)
    assert q.computational is True


def test_flux_coupling_channel_rejected_at_construction():
    """ChargeBasisTransmon has no flux bath model; coupling_channel='flux' raises at construction."""
    with pytest.raises(ValueError, match="flux bath model"):
        ChargeBasisTransmon(E_C=0.25, E_J=20.0, coupling_channel="flux")


def test_flux_coupling_channel_rejected_on_write():
    """coupling_channel='flux' also raises on a post-construction write."""
    q = ChargeBasisTransmon(E_C=0.25, E_J=20.0, coupling_channel="charge")
    with pytest.raises(ValueError, match="flux bath model"):
        q.coupling_channel = "flux"


# ----------------------------------------------------------------------
# Spectrum
# ----------------------------------------------------------------------


def test_koch_reference_spectrum():
    """Eigenspectrum in the deep transmon regime sits near sqrt(8 E_J E_C) - E_C."""
    q = ChargeBasisTransmon(E_C=0.2, E_J=20.0, n_g=0.0, levels=3, num_basis=61)
    freq = float(q.freq)
    # Asymptotic Duffing-regime estimate: sqrt(8*20*0.2) - 0.2 = sqrt(32) - 0.2 ≈ 5.46
    # Exact diagonalization gives close to this with a small negative correction.
    assert 5.0 < freq < 6.0


def test_jax_grad_through_E_J():
    """The 0→1 transition frequency is differentiable with respect to E_J through the diagonalization."""
    def f01(E_J):
        return ChargeBasisTransmon(E_C=0.2, E_J=E_J, levels=3).freq

    grad = jax.grad(f01)(20.0)
    assert np.isfinite(float(grad))


# ----------------------------------------------------------------------
# Serialization
# ----------------------------------------------------------------------


def test_serialization_roundtrip():
    """to_dict/from_dict round-trips E_C, E_J, n_g, levels, label, and T1 exactly."""
    q = ChargeBasisTransmon(
        E_C=0.25, E_J=20.0, n_g=0.1, levels=4, label="tr0", T1=30_000.0,
        coupling_channel="charge",
    )
    d = q.to_dict()
    q2 = ChargeBasisTransmon.from_dict(d)
    assert q2.E_C == q.E_C
    assert q2.E_J == q.E_J
    assert q2.n_g == q.n_g
    assert q2.levels == q.levels
    assert q2.label == q.label
    assert q2.T1 == q.T1


# ----------------------------------------------------------------------
# from_frequency behavior
# ----------------------------------------------------------------------


def test_from_frequency_roundtrip_in_transmon_regime():
    """In the transmon regime, from_frequency's Duffing inversion reproduces the requested 0→1 frequency within 1%."""
    freq_requested = 5.0
    anh_requested = -0.25
    q = ChargeBasisTransmon.from_frequency(freq=freq_requested, anharmonicity=anh_requested)
    freq_diagonalized = float(q.freq)
    assert abs(freq_diagonalized - freq_requested) / freq_requested < 0.01


def test_from_frequency_warns_below_charge_qubit_threshold():
    """from_frequency warns when the inverted E_J/E_C ratio falls below the transmon regime."""
    with pytest.warns(UserWarning, match="below transmon regime"):
        ChargeBasisTransmon.from_frequency(freq=2.0, anharmonicity=-0.5)


def test_from_frequency_no_warn_in_deep_transmon():
    """from_frequency stays silent when the inverted E_J/E_C ratio is well within the transmon regime."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ChargeBasisTransmon.from_frequency(freq=5.0, anharmonicity=-0.25)


def test_from_frequency_tracer_does_not_warn():
    """Tracing from_frequency under jax.jit skips the concrete-scalar regime warning and yields a finite frequency."""
    @jax.jit
    def build_and_return_freq(freq_val):
        return ChargeBasisTransmon.from_frequency(freq=freq_val, anharmonicity=-0.25).freq

    out = build_and_return_freq(5.0)
    assert np.isfinite(float(out))
