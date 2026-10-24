"""Focused integration checks for projected native-basis devices."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from quchip.devices.fluxonium import Fluxonium


def test_dark_transition_is_rejected() -> None:
    """Matrix-element relaxation fails closed when its reference transition is dark."""

    class DarkFluxonium(Fluxonium):
        __init__ = Fluxonium.__init__

        def charge_coupling_operator(self):
            return jnp.zeros((self.num_basis, self.num_basis), dtype=jnp.complex128)

    device = DarkFluxonium(
        E_C=1.0,
        E_J=4.0,
        E_L=1.0,
        phi_ext=0.5,
        levels=3,
        num_basis=80,
        basis="eigen",
        T1=30_000.0,
        coupling_channel="charge",
    )
    with pytest.raises(ValueError, match="dark"):
        device.collapse_operators()
