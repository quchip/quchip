"""Structural RWA mask kernel: band conventions, masking, traceability."""

from __future__ import annotations

import numpy as np

from quchip.backend import get_default_backend
from quchip.chip.rwa import apply_rwa_mask, excitation_band_mask


def _number_conserving(da: int, db: int) -> bool:
    return da + db == 0


def _lowering(d: int) -> np.ndarray:
    return np.diag(np.sqrt(np.arange(1, d)), k=1)


class TestExcitationBandMask:
    def test_two_level_pair_keeps_exchange_kills_counter_rotating(self):
        """The number-conserving mask keeps the a†b/ab† exchange and diagonal elements, zeroing the ab/a†b† pair."""
        mask = excitation_band_mask(2, 2, _number_conserving)
        # Basis |a b> with b fast: 0=|00>, 1=|01>, 2=|10>, 3=|11>.
        assert mask[1, 2] == 1.0 and mask[2, 1] == 1.0  # a†b / a b† exchange
        assert mask[0, 3] == 0.0 and mask[3, 0] == 0.0  # a b / a†b† counter-rotating
        assert mask[0, 0] == 1.0 and mask[3, 3] == 1.0  # diagonal always (0, 0)

    def test_band_convention_matches_engine(self):
        """The mask's Δ = col − row band convention matches the engine's, keeping |00⟩⟨11| and rejecting its mirror."""
        # Engine convention (engine/bands.py): delta = col - row, so the pure
        # lowering operator a (row n, col n+1) sits in the delta = +1 band.
        mask = excitation_band_mask(2, 2, lambda da, db: (da, db) == (1, 1))
        assert mask[0, 3] == 1.0   # |00><11| : both lowered, (Δa, Δb) = (+1, +1)
        assert mask[3, 0] == 0.0

    def test_capacitive_full_masks_to_beam_splitter(self):
        """Masking the capacitive dipole-dipole term with the number-conserving band yields the beam-splitter term."""
        d = 3
        a = _lowering(d)
        x = a + a.T
        full = np.kron(x, x)
        expected = np.kron(a.T, a) + np.kron(a, a.T)
        masked = full * excitation_band_mask(d, d, _number_conserving)
        np.testing.assert_allclose(masked, expected, atol=1e-12)

    def test_diagonal_operator_invariant(self):
        """A diagonal number-operator term lies entirely in the Δ=0 band and passes the mask unchanged."""
        n = np.diag(np.arange(3.0))
        nn = np.kron(n, n)
        masked = nn * excitation_band_mask(3, 3, _number_conserving)
        np.testing.assert_allclose(masked, nn, atol=1e-12)


class TestApplyRwaMask:
    def test_backend_roundtrip_masks_capacitive(self):
        """The backend round trip through apply_rwa_mask matches the beam-splitter term from a dense mask multiply."""
        backend = get_default_backend()
        d = 3
        a = _lowering(d)
        full = np.kron(a + a.T, a + a.T)
        h = backend.from_array(full, dims=[[d, d], [d, d]])
        masked = apply_rwa_mask(
            h, dims=(d, d), labels=("qa", "qb"), keeps_band=_number_conserving, backend=backend
        )
        expected = np.kron(a.T, a) + np.kron(a, a.T)
        np.testing.assert_allclose(np.asarray(backend.to_array(masked)), expected, atol=1e-12)

    def test_band_sum_matches_mask_multiply_oracle(self):
        """The band-sum implementation equals the dense mask multiply on a many-band operator."""
        backend = get_default_backend()
        d = 3
        a = _lowering(d)
        n = np.diag(np.arange(float(d)))
        mixed = np.kron(a + a.T, a + a.T) + np.kron(n, n) + np.kron(a @ a + (a @ a).T, a + a.T)
        h = backend.from_array(mixed, dims=[[d, d], [d, d]])
        masked = apply_rwa_mask(
            h, dims=(d, d), labels=("qa", "qb"), keeps_band=_number_conserving, backend=backend
        )
        oracle = mixed * excitation_band_mask(d, d, _number_conserving)
        np.testing.assert_allclose(np.asarray(backend.to_array(masked)), oracle, atol=1e-12)

    def test_fully_rejected_operator_returns_none(self):
        """A longitudinal n̂ ⊗ (b + b†) has no number-conserving band: the mask reports None."""
        backend = get_default_backend()
        d = 3
        a = _lowering(d)
        longitudinal = np.kron(np.diag(np.arange(float(d))), a + a.T)
        h = backend.from_array(longitudinal, dims=[[d, d], [d, d]])
        assert (
            apply_rwa_mask(h, dims=(d, d), labels=("qa", "qb"), keeps_band=_number_conserving, backend=backend)
            is None
        )


class TestTraceability:
    def test_gradient_flows_through_mask(self):
        """The band mask is a concrete constant: gradients flow through kept elements, vanish through dropped ones."""
        import jax
        import jax.numpy as jnp

        mask = excitation_band_mask(2, 2, _number_conserving)
        a = _lowering(2)
        full = jnp.asarray(np.kron(a + a.T, a + a.T))

        def kept_element(g):
            return jnp.real((g * full * mask)[1, 2])

        def dropped_element(g):
            return jnp.real((g * full * mask)[0, 3])

        assert np.isclose(jax.grad(kept_element)(0.05), 1.0)
        assert np.isclose(jax.grad(dropped_element)(0.05), 0.0)


class TestPredicateHook:
    def test_base_coupling_default_is_number_conserving(self):
        """A coupling's default rwa_keeps_band accepts exchange/diagonal bands, rejecting counter-rotating ones."""
        from quchip.chip.couplings import Capacitive

        c = Capacitive("q0", "q1", g=0.01)
        assert c.rwa_keeps_band(1, -1) is True
        assert c.rwa_keeps_band(-1, 1) is True
        assert c.rwa_keeps_band(0, 0) is True
        assert c.rwa_keeps_band(1, 1) is False
        assert c.rwa_keeps_band(-1, -1) is False
