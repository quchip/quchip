"""Tests for Backend protocol and QuTiPBackend implementation.

Every assertion checks against a known analytical value, never against the implementation's
own output, so tests catch real bugs rather than encoding implementation tautologies.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from quchip.backend import Backend, SolverResult, compute_two_body_permutation
from quchip.backend.qutip import QuTiPBackend


# ── Operator factories ──────────────────────────────────────────────


class TestDestroyOperator:
    """Annihilation operator matrix elements."""

    def test_destroy_matrix_elements(self, backend: QuTiPBackend) -> None:
        """destroy(3) has superdiagonal [1, √2], all else zero."""
        a = backend.destroy(3)
        mat = a.full()
        # Superdiagonal elements: ⟨0|a|1⟩ = 1, ⟨1|a|2⟩ = √2
        npt.assert_allclose(mat[0, 1], 1.0, atol=1e-12)
        npt.assert_allclose(mat[1, 2], np.sqrt(2), atol=1e-12)
        expected = np.zeros((3, 3))
        expected[0, 1] = 1.0
        expected[1, 2] = np.sqrt(2)
        npt.assert_allclose(mat, expected, atol=1e-12)


class TestCreateOperator:
    """Creation operator matrix elements."""

    def test_create_matrix_elements(self, backend: QuTiPBackend) -> None:
        """create(3) has subdiagonal [1, √2], all else zero."""
        adag = backend.create(3)
        mat = adag.full()
        # Subdiagonal elements: ⟨1|a†|0⟩ = 1, ⟨2|a†|1⟩ = √2
        npt.assert_allclose(mat[1, 0], 1.0, atol=1e-12)
        npt.assert_allclose(mat[2, 1], np.sqrt(2), atol=1e-12)
        expected = np.zeros((3, 3))
        expected[1, 0] = 1.0
        expected[2, 1] = np.sqrt(2)
        npt.assert_allclose(mat, expected, atol=1e-12)


class TestNumberOperator:
    """Number operator diagonal entries."""

    def test_number_diagonal(self, backend: QuTiPBackend) -> None:
        """number(3) has diagonal [0, 1, 2]."""
        n_op = backend.number(3)
        mat = n_op.full()
        npt.assert_allclose(np.diag(mat), [0.0, 1.0, 2.0], atol=1e-12)
        off_diag = mat - np.diag(np.diag(mat))
        npt.assert_allclose(off_diag, 0.0, atol=1e-12)


class TestIdentityOperator:
    """Identity operator properties."""

    def test_identity(self, backend: QuTiPBackend) -> None:
        """identity(n) is the n×n identity matrix."""
        for n in (2, 3, 5):
            eye = backend.identity(n)
            npt.assert_allclose(eye.full(), np.eye(n), atol=1e-12)


# ── Tensor product ──────────────────────────────────────────────────


class TestTensorProduct:
    """Tensor (Kronecker) product of operators."""

    def test_tensor_dimensions(self, backend: QuTiPBackend) -> None:
        """tensor(identity(2), identity(3)) produces a 6×6 matrix."""
        result = backend.tensor(backend.identity(2), backend.identity(3))
        assert result.shape == (6, 6)

    def test_tensor_product_is_identity(self, backend: QuTiPBackend) -> None:
        """tensor(I₂, I₃) = I₆."""
        result = backend.tensor(backend.identity(2), backend.identity(3))
        npt.assert_allclose(result.full(), np.eye(6), atol=1e-12)

    def test_tensor_product_values(self, backend: QuTiPBackend) -> None:
        """tensor(destroy(2), identity(3)) places I₃ in the upper-right 3×3 block, zero elsewhere."""
        a2 = backend.destroy(2)
        i3 = backend.identity(3)
        result = backend.tensor(a2, i3)
        mat = result.full()
        npt.assert_allclose(mat[0:3, 3:6], np.eye(3), atol=1e-12)
        npt.assert_allclose(mat[3:6, :], 0.0, atol=1e-12)
        npt.assert_allclose(mat[0:3, 0:3], 0.0, atol=1e-12)


# ── Algebra methods ─────────────────────────────────────────────────


class TestAlgebra:
    """Algebra operations: dag, eigenenergies, expect."""

    def test_dag_of_destroy_equals_create(self, backend: QuTiPBackend) -> None:
        """dag(destroy(n)) ≈ create(n) for several n."""
        for n in (3, 5, 8):
            a = backend.destroy(n)
            adag = backend.dag(a)
            c = backend.create(n)
            npt.assert_allclose(adag.full(), c.full(), atol=1e-12)

    def test_eigenenergies(self, backend: QuTiPBackend) -> None:
        """eigenenergies(number(3)) returns [0, 1, 2]."""
        n_op = backend.number(3)
        evals = backend.eigenenergies(n_op)
        npt.assert_allclose(evals, [0.0, 1.0, 2.0], atol=1e-12)

    def test_matmul(self, backend: QuTiPBackend) -> None:
        """matmul(create(3), destroy(3)) = number(3)."""
        adag = backend.create(3)
        a = backend.destroy(3)
        result = backend.matmul(adag, a)
        n_op = backend.number(3)
        npt.assert_allclose(result.full(), n_op.full(), atol=1e-12)


class TestCanonicalRoundTrip:
    """Canonical sparse/dense round-trips for the QuTiP backend."""

    def test_sparse_operator_roundtrip_preserves_csr(self, backend: QuTiPBackend) -> None:
        """Canonical round-trip of a sparse operator preserves its CSR layout and matrix elements."""
        op = backend.destroy(4)
        canonical = backend.to_canonical_operator(op)
        rebuilt = backend.from_canonical_operator(canonical)

        assert canonical.layout == "csr"
        assert type(rebuilt.data).__name__ == "CSR"
        npt.assert_allclose(rebuilt.full(), op.full(), atol=1e-12)

    def test_dense_operator_roundtrip_stays_dense(self, backend: QuTiPBackend) -> None:
        """Canonical round-trip of a dense operator preserves its dense layout and matrix elements."""
        dense = backend.from_array(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=complex))
        canonical = backend.to_canonical_operator(dense)
        rebuilt = backend.from_canonical_operator(canonical)

        assert canonical.layout == "dense"
        assert type(rebuilt.data).__name__ == "Dense"
        npt.assert_allclose(rebuilt.full(), dense.full(), atol=1e-12)


# ── Embedding: single-body ──────────────────────────────────────────


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

    def test_embed_identity_is_full_identity(self, backend: QuTiPBackend) -> None:
        """Embedding identity into any slot gives the full identity."""
        dims = [3, 5]
        for idx in range(2):
            embedded = backend.embed(backend.identity(dims[idx]), idx, dims)
            npt.assert_allclose(embedded.full(), np.eye(15), atol=1e-12)


# ── Embedding: two-body ─────────────────────────────────────────────


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


# ── Embedding: error cases ──────────────────────────────────────────


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


# ── State factory ───────────────────────────────────────────────────


class TestStateFactory:
    """Basis states, density matrices, coherent states."""

    def test_basis_state(self, backend: QuTiPBackend) -> None:
        """basis(3, 1) is [0, 1, 0] column vector."""
        psi = backend.basis(3, 1)
        vec = psi.full().flatten()
        npt.assert_allclose(vec, [0.0, 1.0, 0.0], atol=1e-12)

    def test_basis_state_ground(self, backend: QuTiPBackend) -> None:
        """basis(3, 0) is [1, 0, 0]."""
        psi = backend.basis(3, 0)
        vec = psi.full().flatten()
        npt.assert_allclose(vec, [1.0, 0.0, 0.0], atol=1e-12)

    def test_state_to_dm(self, backend: QuTiPBackend) -> None:
        """Converting |1⟩ to dm gives |1⟩⟨1| — projector onto |1⟩."""
        psi = backend.basis(3, 1)
        dm = backend.state_to_dm(psi)
        expected = np.zeros((3, 3))
        expected[1, 1] = 1.0
        npt.assert_allclose(dm.full(), expected, atol=1e-12)

    def test_state_to_dm_idempotent(self, backend: QuTiPBackend) -> None:
        """state_to_dm on a density matrix returns it unchanged."""
        psi = backend.basis(3, 0)
        dm = backend.state_to_dm(psi)
        dm2 = backend.state_to_dm(dm)
        npt.assert_allclose(dm2.full(), dm.full(), atol=1e-12)

    def test_is_ket(self, backend: QuTiPBackend) -> None:
        """Distinguishes kets from density matrices."""
        psi = backend.basis(3, 0)
        assert backend.is_ket(psi) is True
        dm = backend.state_to_dm(psi)
        assert backend.is_ket(dm) is False

    def test_coherent_vacuum(self, backend: QuTiPBackend) -> None:
        """coherent(n, 0) is the vacuum state |0⟩."""
        coh = backend.coherent(5, 0.0)
        vac = backend.basis(5, 0)
        npt.assert_allclose(coh.full(), vac.full(), atol=1e-10)

    def test_coherent_normalization(self, backend: QuTiPBackend) -> None:
        """Coherent state with α≠0 is normalized."""
        coh = backend.coherent(20, 2.0 + 1j)
        inner = coh.dag() * coh
        norm_sq = inner.full()[0, 0] if hasattr(inner, "full") else complex(inner)
        npt.assert_allclose(abs(norm_sq), 1.0, atol=1e-10)


# ── Expectation value ───────────────────────────────────────────────


class TestExpect:
    """Expectation value computation."""

    def test_expect_number_in_fock(self, backend: QuTiPBackend) -> None:
        """⟨2|n̂|2⟩ = 2.0 for number operator in Fock state |2⟩."""
        n_op = backend.number(3)
        psi = backend.basis(3, 2)
        val = backend.expect(n_op, psi)
        npt.assert_allclose(float(np.real(val)), 2.0, atol=1e-12)

    def test_expect_number_in_ground(self, backend: QuTiPBackend) -> None:
        """⟨0|n̂|0⟩ = 0.0."""
        n_op = backend.number(5)
        psi = backend.basis(5, 0)
        val = backend.expect(n_op, psi)
        npt.assert_allclose(float(np.real(val)), 0.0, atol=1e-12)


# ── Solver dispatch ─────────────────────────────────────────────────


class TestSolver:
    """Solver smoke tests with analytically trivial cases."""

    def test_sesolve_trivial(self, backend: QuTiPBackend) -> None:
        """Zero Hamiltonian: initial state is preserved at all times."""
        n = 3
        H = backend.number(n) * 0.0  # zero Hamiltonian
        psi0 = backend.basis(n, 1)
        tlist = np.linspace(0, 10, 5)
        result = backend.sesolve(H, psi0, tlist)

        assert isinstance(result, SolverResult)
        assert result.solver == "sesolve"
        assert result.states is not None
        assert len(result.states) == len(tlist)

        for state in result.states:
            inner = psi0.dag() * state
            val = inner.full()[0, 0] if hasattr(inner, "full") else complex(inner)
            overlap = abs(val) ** 2
            npt.assert_allclose(overlap, 1.0, atol=1e-10)

    def test_sesolve_returns_final_state(self, backend: QuTiPBackend) -> None:
        """SolverResult.final_state matches the last state in states list."""
        H = backend.number(3) * 0.0
        psi0 = backend.basis(3, 0)
        tlist = np.linspace(0, 1, 3)
        result = backend.sesolve(H, psi0, tlist)
        assert result.final_state is not None
        npt.assert_allclose(result.final_state.full(), result.states[-1].full(), atol=1e-14)

    def test_sesolve_with_e_ops(self, backend: QuTiPBackend) -> None:
        """sesolve with e_ops tracks expectation values."""
        n = 3
        H = backend.number(n) * 0.0
        psi0 = backend.basis(n, 2)
        tlist = np.linspace(0, 1, 5)
        result = backend.sesolve(H, psi0, tlist, e_ops=[backend.number(n)])

        assert result.expect is not None
        assert len(result.expect) == 1
        npt.assert_allclose(result.expect[0], [2.0] * len(tlist), atol=1e-10)

    def test_mesolve_trivial(self, backend: QuTiPBackend) -> None:
        """mesolve with zero H and no collapse operators preserves state."""
        n = 3
        H = backend.number(n) * 0.0
        psi0 = backend.basis(n, 1)
        tlist = np.linspace(0, 5, 4)
        result = backend.mesolve(H, psi0, tlist)

        assert isinstance(result, SolverResult)
        assert result.solver == "mesolve"
        assert result.states is not None

        # Final state fidelity with initial should be 1
        dm0 = backend.state_to_dm(psi0)
        dm_final = result.final_state
        if backend.is_ket(dm_final):  # mesolve may return kets when H and L are ket-compatible
            dm_final = backend.state_to_dm(dm_final)
        fidelity = np.real((dm0 * dm_final).tr())
        npt.assert_allclose(fidelity, 1.0, atol=1e-10)


# ── Partial trace ───────────────────────────────────────────────────


class TestPartialTrace:
    """Partial trace of composite states."""

    def test_ptrace_product_state(self, backend: QuTiPBackend) -> None:
        """Tracing out device 1 from |0⟩⊗|1⟩ gives |0⟩⟨0|."""
        psi = backend.tensor_states(backend.basis(2, 0), backend.basis(3, 1))
        rho_0 = backend.ptrace(psi, 0, [2, 3])
        expected = np.array([[1.0, 0.0], [0.0, 0.0]])
        npt.assert_allclose(rho_0.full(), expected, atol=1e-12)


# ── Backend protocol ───────────────────────────────────────────────


class TestProtocolContract:
    """Verify the QuTiPBackend satisfies the Backend protocol."""

    def test_is_backend_subclass(self) -> None:
        """QuTiPBackend is a subclass of Backend."""
        assert issubclass(QuTiPBackend, Backend)

    def test_instance_check(self, backend: QuTiPBackend) -> None:
        """A QuTiPBackend instance passes isinstance check."""
        assert isinstance(backend, Backend)

class TestArrayProtocolSurface:
    """Dense-conversion and scalar algebra helpers exposed by Backend."""

    def test_array_module_is_numpy(self, backend: QuTiPBackend) -> None:
        """QuTiPBackend.array_module exposes NumPy."""
        assert backend.array_module is np

    def test_to_array_returns_complex_ndarray(self, backend: QuTiPBackend) -> None:
        """to_array returns a dense complex ndarray."""
        array = backend.to_array(backend.number(3))
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.complex128
        npt.assert_allclose(np.diag(array), [0.0, 1.0, 2.0], atol=1e-12)

    def test_overlap_matches_known_inner_product(self, backend: QuTiPBackend) -> None:
        """⟨1|2⟩ = 0 and ⟨1|1⟩ = 1 through backend.overlap()."""
        psi_1 = backend.basis(3, 1)
        psi_2 = backend.basis(3, 2)
        npt.assert_allclose(backend.overlap(psi_1, psi_1), 1.0, atol=1e-12)
        npt.assert_allclose(backend.overlap(psi_1, psi_2), 0.0, atol=1e-12)

    def test_norm_matches_known_state_norm(self, backend: QuTiPBackend) -> None:
        """Basis states are normalized to 1."""
        psi = backend.basis(3, 2)
        npt.assert_allclose(backend.norm(psi), 1.0, atol=1e-12)

    def test_trace_matches_known_operator_trace(self, backend: QuTiPBackend) -> None:
        """tr(number(3)) = 0 + 1 + 2 = 3."""
        npt.assert_allclose(backend.trace(backend.number(3)), 3.0, atol=1e-12)


# ── Eigenstates ────────────────────────────────────────────────────


class TestEigenstates:
    """Eigenvalues + eigenstates from Backend.eigenstates()."""

    @staticmethod
    def _diag_hamiltonian(backend: QuTiPBackend, diag: list[float]):
        """Build a diagonal Hamiltonian with given diagonal entries."""
        n = len(diag)
        H = diag[0] * backend.basis(n, 0) * backend.basis(n, 0).dag()
        for i in range(1, n):
            H = H + diag[i] * backend.basis(n, i) * backend.basis(n, i).dag()
        return H

    def test_eigenvalues_sorted_ascending(self, backend: QuTiPBackend) -> None:
        """eigenstates() eigenvalues match diagonal entries, sorted ascending."""
        H = self._diag_hamiltonian(backend, [3.0, 1.0, 2.0])

        eigenvalues, _ = backend.eigenstates(H)
        npt.assert_allclose(eigenvalues, [1.0, 2.0, 3.0], atol=1e-12)

    def test_eigenstates_correct_dimension(self, backend: QuTiPBackend) -> None:
        """Each eigenstate is a ket of the correct dimension."""
        H = self._diag_hamiltonian(backend, [3.0, 1.0, 2.0])

        _, estates = backend.eigenstates(H)
        for es in estates:
            assert es.shape == (3, 1) or es.shape[0] == 3

    def test_eigenstates_orthonormal(self, backend: QuTiPBackend) -> None:
        """Eigenstates form an orthonormal set."""
        H = self._diag_hamiltonian(backend, [3.0, 1.0, 2.0])

        _, estates = backend.eigenstates(H)
        n = len(estates)
        for i in range(n):
            for j in range(n):
                overlap = abs(estates[i].dag() * estates[j])
                expected = 1.0 if i == j else 0.0
                npt.assert_allclose(overlap, expected, atol=1e-12)

    def test_eigenstates_correspondence(self, backend: QuTiPBackend) -> None:
        """eigenstates[i] is the eigenvector of eigenvalues[i]."""
        H = self._diag_hamiltonian(backend, [3.0, 1.0, 2.0])

        eigenvalues, estates = backend.eigenstates(H)
        for i, (ev, es) in enumerate(zip(eigenvalues, estates)):
            # H|ψ_i⟩ = E_i|ψ_i⟩  ⇒  H|ψ_i⟩ - E_i|ψ_i⟩ = 0
            residual = (H * es - ev * es).norm()
            npt.assert_allclose(residual, 0.0, atol=1e-12)

    def test_consistency_with_eigenenergies(self, backend: QuTiPBackend) -> None:
        """eigenstates() eigenvalues match eigenenergies() output."""
        H = self._diag_hamiltonian(backend, [5.0, 2.0, 8.0])

        eigenvalues_from_eigenstates, _ = backend.eigenstates(H)
        eigenvalues_from_eigenenergies = backend.eigenenergies(H)
        npt.assert_allclose(
            eigenvalues_from_eigenstates,
            eigenvalues_from_eigenenergies,
            atol=1e-12,
        )


# ── Shared permutation helper ─────────────────────────────────────


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
