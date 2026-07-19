"""Extended integration coverage for circuit-level devices.

Covers:

* Dark-transition safeguard: concrete zero matrix element raises;
  tracer path does not.
* Backend round-trip: same ``Fluxonium`` simulated on QuTiP and dynamiqs
  returns states that agree to high fidelity.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quchip.devices.fluxonium import Fluxonium


def test_dark_transition_concrete_raises():
    """When |N_01|² = 0 concretely, collapse_operators raises."""

    class _DarkFluxonium(Fluxonium):
        def _native_charge_operator(self):
            return jnp.zeros((self.num_basis, self.num_basis), dtype=jnp.complex128)

    q = _DarkFluxonium(
        E_C=1.0, E_J=4.0, E_L=1.0, phi_ext=0.5, levels=3, T1=30_000.0,
        coupling_channel="charge",
    )
    with pytest.raises(ValueError, match="n̂|1"):
        q.collapse_operators()


def test_dark_transition_tracer_does_not_raise():
    """JAX tracer path for |N_01| must pass through without raising."""

    def build_and_sum_ops(phi_ext):
        q = Fluxonium(
            E_C=1.0, E_J=4.0, E_L=1.0, phi_ext=phi_ext, levels=3,
            num_basis=100, T1=30_000.0, coupling_channel="charge",
        )
        ops = q.collapse_operators()
        return jnp.sum(jnp.stack(ops, axis=0))

    result = jax.jit(build_and_sum_ops)(0.5)
    assert np.isfinite(float(jnp.real(result)))


@pytest.mark.optional_backend
def test_fluxonium_two_backends_agree_closed_system():
    """QuTiP and dynamiqs produce states that agree under unitary evolution."""
    dynamiqs = pytest.importorskip("dynamiqs")  # noqa: F841

    import qutip
    from dynamiqs import sesolve as dqsesolve

    q = Fluxonium(E_C=1.0, E_J=4.0, E_L=1.0, phi_ext=0.5, levels=4, num_basis=200)
    # hamiltonian() returns a backend operator (Qobj under the default QuTiP
    # backend); extract the dense matrix backend-agnostically.
    h_op = q.hamiltonian()
    H = np.asarray(h_op.full() if hasattr(h_op, "full") else h_op)

    psi0 = np.zeros(q.levels, dtype=complex)
    psi0[0] = 1.0 / np.sqrt(2)
    psi0[1] = 1.0 / np.sqrt(2)

    tlist = np.linspace(0.0, 1.0, 3)

    psi_qutip = (
        qutip.sesolve(
            qutip.Qobj(H),
            qutip.Qobj(psi0),
            list(tlist),
            options={"store_states": True},
        )
        .states[-1]
        .full()
        .flatten()
    )

    # dynamiqs expects ket shape (n, 1) and Hamiltonian as (n, n).
    psi0_ket = jnp.asarray(psi0).reshape(-1, 1)
    H_jax = jnp.asarray(H)
    psi_dq = dqsesolve(H_jax, psi0_ket, jnp.asarray(tlist)).states[-1]
    psi_dq = np.asarray(psi_dq).flatten()

    fidelity = np.abs(np.vdot(psi_qutip, psi_dq)) ** 2
    assert fidelity > 1.0 - 1e-4
