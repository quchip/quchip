"""Focused integration checks for projected native-basis devices."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
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


@pytest.mark.optional_backend
def test_projected_fluxonium_two_backends_agree() -> None:
    """QuTiP and Dynamiqs agree on the same engine-materialized Hamiltonian."""
    dq = pytest.importorskip("dynamiqs")
    qutip = pytest.importorskip("qutip")

    device = Fluxonium(
        E_C=1.0,
        E_J=4.0,
        E_L=1.0,
        phi_ext=0.5,
        levels=4,
        num_basis=100,
        basis="eigen",
    )
    hamiltonian = device.resolve().hamiltonian().matrix()
    initial = np.zeros((4, 1), dtype=complex)
    initial[0, 0] = initial[1, 0] = 1.0 / np.sqrt(2.0)
    times = np.linspace(0.0, 1.0, 3)

    qutip_state = qutip.sesolve(
        qutip.Qobj(np.asarray(hamiltonian)),
        qutip.Qobj(initial),
        times,
        options={"store_states": True},
    ).states[-1].full()
    dynamiqs_state = dq.sesolve(
        jnp.asarray(hamiltonian),
        jnp.asarray(initial),
        jnp.asarray(times),
    ).states[-1]

    fidelity = np.abs(np.vdot(qutip_state.ravel(), np.asarray(dynamiqs_state).ravel())) ** 2
    assert fidelity > 1.0 - 1e-4
