"""Tests for Backend protocol and QuTiPBackend implementation."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from quchip.backend import compute_two_body_permutation
from quchip.backend.qutip import QuTiPBackend


pytestmark = pytest.mark.unit


class TestEmbedSingleBody:
    """Single-body operator embedding into composite Hilbert space."""

    def test_embed_trace_device_0(self, backend: QuTiPBackend) -> None:
        """embed(number(3), 0, [3, 5]) has trace = (0+1+2)×5 = 15."""
        op = backend.embed(backend.number(3), 0, [3, 5])
        assert op.shape == (15, 15)
        npt.assert_allclose(op.tr(), 15.0, atol=1e-10)

    def test_embed_trace_device_1(self, backend: QuTiPBackend) -> None:
        """embed(number(5), 1, [3, 5]) has trace = (0+1+2+3+4)×3 = 30."""
        op = backend.embed(backend.number(5), 1, [3, 5])
        assert op.shape == (15, 15)
        npt.assert_allclose(op.tr(), 30.0, atol=1e-10)


class TestEmbedTwoBody:
    """Two-body operator embedding, including reorder/permute logic."""

    def test_embed_two_body_adjacent(self, backend: QuTiPBackend) -> None:
        """In a 2-device system, adjacent embedding is just the operator itself."""
        a3 = backend.destroy(3)
        c5 = backend.create(5)
        op_ab = backend.tensor(a3, c5)
        embedded = backend.embed_two_body(op_ab, 0, 1, [3, 5])
        npt.assert_allclose(embedded.full(), op_ab.full(), atol=1e-12)

    def test_embed_two_body_non_adjacent(self, backend: QuTiPBackend) -> None:
        """3-device system [3, 4, 5], devices 0 and 2: correct placement."""
        dims = [3, 4, 5]
        a3 = backend.destroy(3)
        c5 = backend.create(5)
        op_02 = backend.tensor(a3, c5)  # acts on devices 0 ⊗ 2
        embedded = backend.embed_two_body(op_02, 0, 2, dims)

        total = 3 * 4 * 5
        assert embedded.shape == (total, total)

        # a|1> = |0>, c†|0> = |1>, so embedded maps |1,0,0> to |0,0,1> with unit amplitude.
        b = backend
        psi_in = b.tensor_states(b.basis(3, 1), b.basis(4, 0), b.basis(5, 0))
        psi_out = b.tensor_states(b.basis(3, 0), b.basis(4, 0), b.basis(5, 1))
        bra_out = psi_out.dag()
        mel = bra_out * embedded * psi_in
        val = mel.full()[0, 0] if hasattr(mel, "full") else complex(mel)
        npt.assert_allclose(val, 1.0, atol=1e-12)

    def test_embed_two_body_reversed_indices(self, backend: QuTiPBackend) -> None:
        """Reversed indices (index_a > index_b) SWAP-permute to match the natural-order embedding."""
        dims = [3, 5]
        a3 = backend.destroy(3)
        c5 = backend.create(5)
        op_ba = backend.tensor(c5, a3)  # device_b(5) ⊗ device_a(3)
        embedded_reversed = backend.embed_two_body(op_ba, 1, 0, dims)

        op_ab = backend.tensor(a3, c5)  # device_a(3) ⊗ device_b(5), natural order
        embedded_natural = backend.embed_two_body(op_ab, 0, 1, dims)

        npt.assert_allclose(embedded_reversed.full(), embedded_natural.full(), atol=1e-12)

    def test_embed_two_body_three_devices_adjacent(self, backend: QuTiPBackend) -> None:
        """3-device system, adjacent pair (1, 2): identity on device 0."""
        dims = [2, 3, 4]
        n3 = backend.number(3)
        n4 = backend.number(4)
        op_12 = backend.tensor(n3, n4)
        embedded = backend.embed_two_body(op_12, 1, 2, dims)
        total = 2 * 3 * 4
        assert embedded.shape == (total, total)
        # Trace = tr(I₂) × tr(n₃) × tr(n₄) = 2 × (0+1+2) × (0+1+2+3) = 2×3×6 = 36
        npt.assert_allclose(embedded.tr(), 36.0, atol=1e-10)


class TestEmbedErrors:
    """Error handling for embedding methods."""

    def test_embed_dimension_mismatch_raises(self, backend: QuTiPBackend) -> None:
        """embed raises ValueError when operator size mismatches dims."""
        op = backend.number(3)
        with pytest.raises(ValueError, match="dimension"):
            backend.embed(op, 0, [5, 5])  # op is 3×3, but dims[0]=5

    def test_embed_index_out_of_range_raises(self, backend: QuTiPBackend) -> None:
        """embed raises ValueError for out-of-range device_index."""
        op = backend.number(3)
        with pytest.raises(ValueError, match="out of range"):
            backend.embed(op, 2, [3, 5])  # only 2 devices, index 2 invalid

    def test_embed_two_body_same_index_raises(self, backend: QuTiPBackend) -> None:
        """embed_two_body raises ValueError when index_a == index_b."""
        op = backend.tensor(backend.number(3), backend.number(3))
        with pytest.raises(ValueError, match="different"):
            backend.embed_two_body(op, 0, 0, [3, 3])

    def test_embed_two_body_index_out_of_range_raises(self, backend: QuTiPBackend) -> None:
        """embed_two_body raises ValueError for out-of-range index."""
        op = backend.tensor(backend.number(3), backend.number(5))
        with pytest.raises(ValueError, match="out of range"):
            backend.embed_two_body(op, 0, 3, [3, 5, 4])


class TestComputeTwoBodyPermutation:
    """Tests for the shared two-body permutation helper in protocol.py."""

    def test_adjacent_indices_are_identity(self) -> None:
        """Adjacent indices (0, 1) in a 2-device system produce trivial permutations."""
        reorder, inverse = compute_two_body_permutation(0, 1, [3, 5])
        assert reorder == [0, 1]
        assert inverse == [0, 1]

    def test_adjacent_indices_three_devices(self) -> None:
        """Adjacent indices (0, 1) in a 3-device system put the third at the end."""
        reorder, inverse = compute_two_body_permutation(0, 1, [2, 3, 4])
        assert reorder == [0, 1, 2]
        assert inverse == [0, 1, 2]

    def test_nonadjacent_indices(self) -> None:
        """Non-adjacent indices (0, 2) in a 3-device system reorder correctly."""
        reorder, inverse = compute_two_body_permutation(0, 2, [2, 3, 4])
        assert sorted(reorder) == [0, 1, 2]
        assert sorted(inverse) == [0, 1, 2]
        # Forward: devices 0, 2 first, then 1
        assert reorder == [0, 2, 1]
        # Inverse: position 0 -> 0, position 1 -> 2, position 2 -> 1
        assert inverse == [0, 2, 1]

    def test_nonadjacent_four_devices(self) -> None:
        """Non-adjacent indices (1, 3) in a 4-device system."""
        reorder, inverse = compute_two_body_permutation(1, 3, [2, 3, 4, 5])
        assert reorder == [1, 3, 0, 2]
        # inverse[old_pos] = new_pos: 0->2, 1->0, 2->3, 3->1
        assert inverse == [2, 0, 3, 1]

    def test_inverse_undoes_forward(self) -> None:
        """Applying forward then inverse permutation recovers original ordering."""
        for idx_a, idx_b, dims in [
            (0, 2, [2, 3, 4]),
            (1, 3, [2, 3, 4, 5]),
            (0, 3, [2, 3, 4, 5]),
            (0, 4, [2, 3, 4, 5, 6]),
        ]:
            reorder, inverse = compute_two_body_permutation(idx_a, idx_b, dims)
            n = len(dims)
            for i in range(n):
                assert reorder[inverse[i]] == i


class TestCoerceOperator:
    """Array-like operands coerce to native form at composition entry points."""

    def test_coerce_operator_passthrough_for_native(self, backend: QuTiPBackend) -> None:
        """A Qobj passes through coerce_operator unchanged (same object)."""
        op = backend.destroy(3)
        assert backend.coerce_operator(op) is op

    def test_tensor_accepts_array_like_operand(self, backend: QuTiPBackend) -> None:
        """tensor(ndarray, Qobj) equals tensor(Qobj, Qobj) of the same matrices."""
        n_arr = np.diag(np.arange(3.0))  # number operator as a plain array
        a = backend.destroy(4)
        mixed = backend.tensor(n_arr, a + backend.dag(a))
        native = backend.tensor(backend.coerce_operator(n_arr), a + backend.dag(a))
        npt.assert_allclose(
            np.asarray(backend.to_array(mixed)), np.asarray(backend.to_array(native)), atol=1e-14
        )
        expected = np.kron(n_arr, np.asarray(backend.to_array(a)) + np.asarray(backend.to_array(a)).conj().T)
        npt.assert_allclose(np.asarray(backend.to_array(mixed)), expected, atol=1e-14)

    def test_dag_accepts_array_like(self, backend: QuTiPBackend) -> None:
        """dag(ndarray) returns the conjugate transpose as a native operator."""
        m = np.array([[0.0, 1.0 + 2.0j], [0.0, 0.0]])
        d = backend.dag(m)
        npt.assert_allclose(np.asarray(backend.to_array(d)), m.conj().T, atol=1e-14)


class TestSuperoperatorMemory:
    """QuTiP keeps structurally sparse operators sparse and refuses Liouvillians that cannot fit."""

    def test_mostly_zero_dense_payloads_are_stored_sparse(self) -> None:
        """Dense payloads at most a quarter nonzero become CSR without changing their values."""
        from quchip.engine.ir import CanonicalOperator

        def stored(values: np.ndarray) -> object:
            canonical = CanonicalOperator.from_dense(values, dims=(4,), basis="fock", subsystem_labels=("0",))
            return QuTiPBackend._canonical_to_qobj(canonical)

        sparse_values = np.zeros((4, 4), dtype=complex)
        sparse_values[0, 1] = 2.0
        sparse, dense = stored(sparse_values), stored(np.ones((4, 4), dtype=complex))
        assert type(sparse.data).__name__ == "CSR"
        assert type(dense.data).__name__ == "Dense"
        npt.assert_array_equal(sparse.full(), sparse_values)

    def test_estimate_counts_dense_copies_and_sparse_entries(self) -> None:
        """The peak estimate counts dense copies per Hamiltonian part and sparse entries otherwise."""
        import qutip

        from quchip.backend.qutip import _superoperator_peak_bytes

        dense_h = qutip.Qobj(np.ones((5, 5)), dtype="Dense")
        drives = [qutip.Qobj(np.eye(5, k=k), dtype="Dense") for k in (1, 2)]
        evolving = qutip.QobjEvo([dense_h, [drives[0], lambda t: np.cos(t)], [drives[1], lambda t: np.sin(t)]])
        assert _superoperator_peak_bytes(dense_h, [qutip.destroy(5)]) == 6 * 16 * 5**4
        assert _superoperator_peak_bytes(evolving, [qutip.destroy(5)]) == (6 + 2 * 4) * 16 * 5**4
        assert _superoperator_peak_bytes(qutip.num(30), [qutip.destroy(30)]) < 16 * 30**4

    @pytest.mark.parametrize("tiny", [0.0, 1e-9], ids=["local-loss", "tiny-entries"])
    def test_sparse_estimate_follows_the_entries_qutip_stores(self, tiny: float) -> None:
        """The sparse estimate stays within a small factor above the assembled Liouvillian.

        Local loss keeps c†c diagonal, and QuTiP drops products below its
        tidy-up tolerance, so tiny entries store no pairs among themselves.
        """
        import qutip

        from quchip.backend.qutip import _superoperator_peak_bytes

        dims = (4, 4, 4)
        modes = [
            qutip.tensor(*[qutip.destroy(d) if index == k else qutip.qeye(d) for index, d in enumerate(dims)])
            for k in range(len(dims))
        ]
        hamiltonian = sum(mode.dag() * mode for mode in modes) + 0.01 * (
            modes[0].dag() * modes[1] + modes[1].dag() * modes[0]
        )
        noise = qutip.Qobj(np.random.default_rng(7).normal(size=(64, 64)), dims=modes[0].dims)
        jumps = [(np.sqrt(0.01) * mode + tiny * noise).to("CSR") for mode in modes]

        entry_bytes = 16 + np.dtype(qutip.core.data.base.idxint_dtype).itemsize
        stored = qutip.liouvillian(hamiltonian, jumps).data_as("csr_matrix").nnz * entry_bytes
        assert stored <= _superoperator_peak_bytes(hamiltonian.to("CSR"), jumps) <= 6 * stored

    def test_mesolve_raises_before_building_a_liouvillian_that_cannot_fit(
        self, backend: QuTiPBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """mesolve raises MemoryError naming dynamiqs before assembling an oversized Liouvillian."""
        import qutip

        from quchip.backend import _memory

        monkeypatch.setattr(_memory, "available_memory_bytes", lambda: 10**6)
        hamiltonian = qutip.Qobj(np.ones((12, 12)), dtype="Dense")
        with pytest.raises(MemoryError, match="backend='dynamiqs'"):
            backend.mesolve(hamiltonian, qutip.basis(12, 0), [0.0, 1.0], c_ops=[qutip.destroy(12)])
        assert _memory.available_memory_bytes() == 10**6

    def test_mesolve_matches_qutip_for_driven_lossy_hamiltonians(self, backend: QuTiPBackend) -> None:
        """A driven lossy mode evolves as under QuTiP's own Liouvillian, with or without the drive's partner."""
        import qutip

        mode = qutip.destroy(5)
        drive = qutip.coefficient(lambda t: 0.3 * np.exp(-0.7j * t))
        hermitian = qutip.QobjEvo([0.2 * mode.dag() * mode, [mode, drive], [mode.dag(), drive.conj()]])
        one_sided = qutip.QobjEvo([0.2 * mode.dag() * mode, [mode.dag(), drive]])
        options = {"method": "vern9", "rtol": 1e-10, "atol": 1e-12}
        for hamiltonian in (hermitian, one_sided):
            expected = qutip.mesolve(hamiltonian, qutip.basis(5, 0), [0.0, 5.0], c_ops=[0.1 * mode],
                                     options=options).final_state
            result = backend.mesolve(hamiltonian, qutip.basis(5, 0), [0.0, 5.0], c_ops=[0.1 * mode],
                                     options=options)
            npt.assert_allclose(result.final_state.full(), expected.full(), atol=1e-9)
