"""Tests for Fluxonium — phase-basis fluxonium model."""

from __future__ import annotations

import pytest

import jax
import numpy as np

from quchip import Chip
from quchip.devices.fluxonium import Fluxonium


def test_computational_property():
    """Fluxonium.computational is True."""
    q = Fluxonium(E_C=1.0, E_J=4.0, E_L=1.0)
    assert q.computational is True


@pytest.mark.validation
def test_jax_grad_through_E_J():
    """The 0->1 gap is differentiable with respect to E_J."""
    def f01(E_J):
        return Fluxonium(E_C=1.0, E_J=E_J, E_L=1.0, phi_ext=0.5, levels=3, num_basis=200).freq

    grad = jax.jit(jax.grad(f01))(4.0)
    assert np.isfinite(float(grad))


def test_flux_coupling_operator_equals_phase_coupling_operator():
    """On a fluxonium, flux couples through φ̂ — same as phase coupling."""
    q = Fluxonium(E_C=1.0, E_J=4.0, E_L=1.0, phi_ext=0.5, levels=3, num_basis=200)
    flux_op = np.asarray(q.flux_coupling_operator())
    phase_op = np.asarray(q.phase_coupling_operator())
    assert np.allclose(flux_op, phase_op)


@pytest.mark.validation
def test_charge_and_flux_channels_give_different_rates_at_sweet_spot():
    """Charge and flux coupling channels share the T1 anchor rate but differ in cascade rates."""
    kwargs = dict(
        E_C=1.0, E_J=4.0, E_L=1.0, phi_ext=0.5, levels=4, num_basis=200, T1=30_000.0,
    )
    q_charge = Fluxonium(**kwargs, coupling_channel="charge")
    q_flux = Fluxonium(**kwargs, coupling_channel="flux")

    def _resolved_operators(device: Fluxonium) -> list[np.ndarray]:
        result = Chip([device], basis="eigen").resolve()
        return [
            np.sqrt(np.asarray(term.rate)) * np.asarray(term.operator.to_dense())
            for term in result.collapse_terms
        ]

    ops_charge = _resolved_operators(q_charge)
    ops_flux = _resolved_operators(q_flux)

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
