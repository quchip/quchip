"""Regression coverage for dynamiqs-backend contract fixes: overlap, norm, and permute_state.

Every test requires the optional ``dynamiqs`` extra and is marked
``optional_backend`` accordingly; ``pytest.importorskip`` gives a clean skip
when it is not installed.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

# Establishes the quchip module import graph before quchip.results/quchip.chip.partition
# are imported directly — importing those first hits a circular-import ordering issue.
from quchip.chip.chip import Chip  # noqa: F401


pytestmark = pytest.mark.optional_backend


@pytest.fixture
def dynamiqs_backend():
    """A fresh DynamiqsBackend instance, skipping when dynamiqs is unavailable."""
    pytest.importorskip("dynamiqs")
    from quchip.backend.dynamiqs import DynamiqsBackend

    return DynamiqsBackend()


class TestDynamiqsOverlapContract:
    """DynamiqsBackend.overlap must return the complex ⟨a|b⟩, not dq.overlap's |.|**2."""

    def test_overlap_matches_qutip_for_nonorthogonal_phased_states(self, dynamiqs_backend) -> None:
        """A relative-phase overlap agrees with QuTiPBackend.overlap and is complex-valued."""
        from quchip.backend.qutip import QuTiPBackend

        # Two non-orthogonal, non-identical states with a nontrivial relative phase.
        a_coeffs = np.array([0.6, 0.8j], dtype=complex)
        b_raw = np.array([0.5, 0.5 * np.exp(1j * np.pi / 3)], dtype=complex)
        b_coeffs = b_raw / np.linalg.norm(b_raw)

        qutip_backend = QuTiPBackend()
        a_qutip = qutip_backend.from_array(a_coeffs.reshape(-1, 1))
        b_qutip = qutip_backend.from_array(b_coeffs.reshape(-1, 1))
        a_dq = dynamiqs_backend.from_array(a_coeffs.reshape(-1, 1))
        b_dq = dynamiqs_backend.from_array(b_coeffs.reshape(-1, 1))

        overlap_qutip = qutip_backend.overlap(a_qutip, b_qutip)
        overlap_dq = complex(dynamiqs_backend.overlap(a_dq, b_dq))

        # The pre-fix dq.overlap() returned the real-valued |<a|b>|**2, collapsing the phase.
        assert abs(overlap_qutip.imag) > 1e-6, "test fixture must carry a nontrivial phase"
        npt.assert_allclose(overlap_dq, overlap_qutip, atol=1e-10)


class TestDynamiqsNormTraceability:
    """DynamiqsBackend.norm returns a native scalar; states.superposition stays tracer-safe."""

    def test_norm_returns_native_array_not_python_float(self, dynamiqs_backend) -> None:
        """norm() no longer forces float(), so it survives inside a jax.jit trace."""
        import jax
        import jax.numpy as jnp

        @jax.jit
        def traced_norm(amp):
            psi = amp * dynamiqs_backend.basis(2, 0) + dynamiqs_backend.basis(2, 1)
            return dynamiqs_backend.norm(psi)

        value = traced_norm(jnp.asarray(0.6))
        npt.assert_allclose(value, np.sqrt(0.6**2 + 1.0), atol=1e-6)

    def test_superposition_traced_amplitude_jit_and_grad_through_norm(self, dynamiqs_backend) -> None:
        """A traced superposition amplitude raises no TracerBoolConversionError under jit/grad."""
        import jax
        import jax.numpy as jnp

        from quchip.devices.transmon.duffing import DuffingTransmon

        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        chip = Chip(devices=[q], backend=dynamiqs_backend)

        @jax.jit
        def normalized_norm(amp):
            psi = chip.superposition((amp, {"q": 0}), (1.0, {"q": 1}))
            return jnp.real(chip.backend.norm(psi))

        value = normalized_norm(jnp.asarray(0.6))
        npt.assert_allclose(value, 1.0, atol=1e-6)

        # A normalized state's own norm is identically 1 regardless of the (nonzero)
        # amplitude used to build it, so the correct gradient through it is exactly 0.
        grad_value = jax.grad(normalized_norm)(jnp.asarray(0.6))
        npt.assert_allclose(grad_value, 0.0, atol=1e-6)

    def test_superposition_zero_traced_amplitude_returns_zero_state_not_nan(self, dynamiqs_backend) -> None:
        """A traced amplitude of exactly zero returns the unnormalized zero state under jit, never NaN."""
        import jax
        import jax.numpy as jnp

        from quchip.devices.transmon.duffing import DuffingTransmon

        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        chip = Chip(devices=[q], backend=dynamiqs_backend)

        @jax.jit
        def zero_amplitude_state(amp):
            psi = chip.superposition((amp, {"q": 0}))
            return chip.backend.to_array(psi)

        value = zero_amplitude_state(jnp.asarray(0.0))
        assert not bool(jnp.any(jnp.isnan(value))), f"expected the zero state, got {value!r}"
        npt.assert_allclose(np.asarray(value), np.zeros_like(np.asarray(value)), atol=0.0)


class TestPermuteStateTraceability:
    """Backend.permute_state's default body stays traced end-to-end (used by dynamiqs)."""

    @staticmethod
    def _partitioned_result(dynamiqs_backend, amp, *, as_density_matrix: bool):
        """Two-component PartitionedSimulationResult needing a real (non-identity) permutation.

        Concatenated component order is (q0, q2, q1); ``chip_order`` is
        (q0, q1, q2) — exactly the component-order-vs-chip-order mismatch
        :meth:`~quchip.results.partitioned.PartitionedSimulationResult.final_state`
        resolves via :meth:`Backend.permute_state`.
        """
        import jax.numpy as jnp

        from quchip.backend.containers import SolverResult
        from quchip.chip.partition import PartitionComponent, PartitionResult
        from quchip.results.partitioned import PartitionedSimulationResult
        from quchip.results.results import SimulationResult

        backend = dynamiqs_backend
        q0_ket = jnp.cos(amp) * backend.basis(2, 0) + jnp.sin(amp) * backend.basis(2, 1)
        q2_ket = backend.basis(2, 1)
        joint_a = backend.tensor_states(q0_ket, q2_ket)
        q1_state = backend.basis(2, 0)
        if as_density_matrix:
            q1_state = backend.state_to_dm(q1_state)

        sr_a = SolverResult(times=jnp.array([0.0]), states=[joint_a], final_state=joint_a, solver="sesolve")
        sr_b = SolverResult(times=jnp.array([0.0]), states=[q1_state], final_state=q1_state, solver="sesolve")
        result_a = SimulationResult(sr_a, backend, dims=[2, 2], device_info=[("q0", True), ("q2", True)])
        result_b = SimulationResult(sr_b, backend, dims=[2], device_info=[("q1", True)])

        partition = PartitionResult(
            components=(
                PartitionComponent(labels=("q0", "q2"), chip=None),
                PartitionComponent(labels=("q1",), chip=None),
            ),
            chip_order=("q0", "q1", "q2"),
        )
        partitioned = PartitionedSimulationResult([result_a, result_b], partition, {})
        return backend.to_array(partitioned.final_state)

    def test_ket_final_state_permutes_correctly_under_jit(self, dynamiqs_backend) -> None:
        """A ket-valued partitioned final_state jits and permutes into chip order."""
        import jax
        import jax.numpy as jnp

        amp0 = 0.3
        jitted = jax.jit(lambda a: self._partitioned_result(dynamiqs_backend, a, as_density_matrix=False))
        value = jitted(jnp.asarray(amp0))

        q0_expected = np.array([np.cos(amp0), np.sin(amp0)])
        q1_expected = np.array([1.0, 0.0])
        q2_expected = np.array([0.0, 1.0])
        expected = np.kron(np.kron(q0_expected, q1_expected), q2_expected).reshape(-1, 1)
        npt.assert_allclose(np.asarray(value), expected, atol=1e-5)

    def test_ket_final_state_gradient_is_finite(self, dynamiqs_backend) -> None:
        """Gradient through the permuted ket final_state is finite (no concretization break)."""
        import jax
        import jax.numpy as jnp

        def summed(a):
            return jnp.real(jnp.sum(self._partitioned_result(dynamiqs_backend, a, as_density_matrix=False)))

        grad = jax.grad(summed)(jnp.asarray(0.3))
        assert np.isfinite(float(grad))

    def test_density_matrix_final_state_permutes_correctly_under_jit(self, dynamiqs_backend) -> None:
        """A mixed ket/DM component set promotes to DMs and permutes correctly under jit."""
        import jax
        import jax.numpy as jnp

        amp0 = 0.3
        jitted = jax.jit(lambda a: self._partitioned_result(dynamiqs_backend, a, as_density_matrix=True))
        value = jitted(jnp.asarray(amp0))

        q0_expected = np.array([np.cos(amp0), np.sin(amp0)])
        q1_expected = np.array([1.0, 0.0])
        q2_expected = np.array([0.0, 1.0])
        psi = np.kron(np.kron(q0_expected, q1_expected), q2_expected)
        expected_dm = np.outer(psi, psi.conj())
        npt.assert_allclose(np.asarray(value), expected_dm, atol=1e-5)
