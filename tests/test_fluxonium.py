"""Tests for Fluxonium — phase-basis fluxonium model."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from quchip.utils.labeling import reset_label_counters
from quchip.devices.fluxonium import Fluxonium
from quchip.devices.protocols import ChargeCoupled, FluxCoupled, PhaseCoupled


@pytest.fixture(autouse=True)
def _reset():
    reset_label_counters()
    yield


def test_constructor_accepts_three_energies():
    """Fluxonium accepts E_C, E_J, E_L and stores the requested level count."""
    q = Fluxonium(E_C=1.0, E_J=4.0, E_L=1.0, phi_ext=0.5, levels=5)
    assert q.levels == 5


def test_invalid_energies_raise():
    """Non-positive E_C, E_J, or E_L raises ValueError."""
    with pytest.raises(ValueError, match="E_C"):
        Fluxonium(E_C=-1.0, E_J=4.0, E_L=1.0)
    with pytest.raises(ValueError, match="E_J"):
        Fluxonium(E_C=1.0, E_J=-4.0, E_L=1.0)
    with pytest.raises(ValueError, match="E_L"):
        Fluxonium(E_C=1.0, E_J=4.0, E_L=-1.0)


def test_conforms_to_all_three_protocols():
    """Fluxonium implements ChargeCoupled, PhaseCoupled, and FluxCoupled."""
    q = Fluxonium(E_C=1.0, E_J=4.0, E_L=1.0)
    assert isinstance(q, ChargeCoupled)
    assert isinstance(q, PhaseCoupled)
    assert isinstance(q, FluxCoupled)


def test_sweet_spot_is_lowest_01_gap():
    """0→1 gap is minimized at phi_ext=0.5 for standard fluxonium params."""
    q_half = Fluxonium(E_C=1.0, E_J=4.0, E_L=1.0, phi_ext=0.5, levels=3, num_basis=200)
    q_zero = Fluxonium(E_C=1.0, E_J=4.0, E_L=1.0, phi_ext=0.0, levels=3, num_basis=200)
    assert float(q_half.freq) < float(q_zero.freq)


def test_computational_property():
    """Fluxonium.computational is True."""
    q = Fluxonium(E_C=1.0, E_J=4.0, E_L=1.0)
    assert q.computational is True


def test_state_version_bumps_on_phi_ext_mutation():
    """Mutating phi_ext increments state_version."""
    q = Fluxonium(E_C=1.0, E_J=4.0, E_L=1.0, phi_ext=0.0)
    v0 = q.state_version
    q.phi_ext = 0.5
    assert q.state_version > v0


def test_jax_grad_through_E_J():
    """The 0->1 gap is differentiable with respect to E_J."""
    def f01(E_J):
        return Fluxonium(E_C=1.0, E_J=E_J, E_L=1.0, phi_ext=0.5, levels=3, num_basis=200).freq

    grad = jax.grad(f01)(4.0)
    assert np.isfinite(float(grad))


def test_serialization_roundtrip():
    """to_dict/from_dict round-trips constructor parameters and label."""
    q = Fluxonium(
        E_C=1.0, E_J=4.0, E_L=1.0, phi_ext=0.5, levels=6, label="f0",
        T1=30_000.0, num_basis=200, phi_max=4 * np.pi,
        coupling_channel="flux",
    )
    d = q.to_dict()
    q2 = Fluxonium.from_dict(d)
    assert q2.E_C == q.E_C
    assert q2.E_J == q.E_J
    assert q2.E_L == q.E_L
    assert q2.phi_ext == q.phi_ext
    assert q2.num_basis == q.num_basis
    assert q2.label == q.label


def test_flux_coupling_operator_equals_phase_coupling_operator():
    """On a fluxonium, flux couples through φ̂ — same as phase coupling."""
    q = Fluxonium(E_C=1.0, E_J=4.0, E_L=1.0, phi_ext=0.5, levels=3, num_basis=200)
    flux_op = np.asarray(q.flux_coupling_operator())
    phase_op = np.asarray(q.phase_coupling_operator())
    assert np.allclose(flux_op, phase_op)


def test_fermi_golden_requires_explicit_coupling_channel():
    """Constructing with T1 but no coupling_channel must fail fast (issue #62)."""
    with pytest.raises(ValueError, match="coupling_channel"):
        Fluxonium(E_C=1.0, E_J=4.0, E_L=1.0, phi_ext=0.5, levels=3, num_basis=200, T1=30_000.0)


def test_charge_and_flux_channels_give_different_rates_at_sweet_spot():
    """Charge and flux coupling channels share the T1 anchor rate but differ in cascade rates."""
    kwargs = dict(
        E_C=1.0, E_J=4.0, E_L=1.0, phi_ext=0.5, levels=4, num_basis=200, T1=30_000.0,
    )
    q_charge = Fluxonium(**kwargs, coupling_channel="charge")
    q_flux = Fluxonium(**kwargs, coupling_channel="flux")

    ops_charge = [np.asarray(op) for op in q_charge.collapse_operators()]
    ops_flux = [np.asarray(op) for op in q_flux.collapse_operators()]

    def _total_rate_matrix(ops: list[np.ndarray]) -> np.ndarray:
        rate = np.zeros((4, 4))
        for op in ops:
            rate += np.abs(op) ** 2
        return rate

    R_charge = _total_rate_matrix(ops_charge)
    R_flux = _total_rate_matrix(ops_flux)

    # |0⟩←|1⟩ is the FG normalization anchor: both channels fix it to 1/T1.
    assert np.isclose(R_charge[0, 1], 1.0 / 30_000.0, rtol=1e-10)
    assert np.isclose(R_flux[0, 1], 1.0 / 30_000.0, rtol=1e-10)
    # Off-anchor rates differ: φ̂ and n̂ have different parity structure at the sweet spot.
    assert not np.allclose(R_charge, R_flux, atol=1e-10)
