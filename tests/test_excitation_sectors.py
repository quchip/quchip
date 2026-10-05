"""Total-excitation sectors stay exact in reductions and in band decomposition."""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quchip.chip.effective import EffectiveTerms
from quchip.chip.sw import _exact_eigensystem, excitation_sectors
from quchip.declarative.dissipation import CollapseChannel
from quchip.declarative.expr import declared_excitation_changes
from quchip.engine.bands import _decompose_product_canonical_bands, concrete_excitation_changes
from quchip.engine.ir import CanonicalOperator

pytestmark = pytest.mark.unit


def _conserving_hamiltonian(dims, seed=0):
    sectors = excitation_sectors(dims)
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(sectors.size,) * 2) + 1j * rng.normal(size=(sectors.size,) * 2)
    matrix = matrix + matrix.conj().T
    return np.where(sectors[:, None] == sectors[None, :], matrix, 0.0), sectors


def test_sector_eigensystem_matches_full_diagonalization_with_exact_zeros_between_sectors():
    """Sector-wise diagonalization reproduces the full spectrum with eigenvectors confined to one sector."""
    dims = (3, 3, 2)
    h, sectors = _conserving_hamiltonian(dims)
    values, vectors, _ = _exact_eigensystem(h, dims, sectors)
    values, vectors = np.asarray(values), np.asarray(vectors)

    np.testing.assert_allclose(values, np.linalg.eigvalsh(h), atol=1e-12)
    np.testing.assert_allclose((vectors * values) @ vectors.conj().T, h, atol=1e-12)
    np.testing.assert_allclose(vectors.conj().T @ vectors, np.eye(sectors.size), atol=1e-12)
    vector_sectors = sectors[np.argmax(np.abs(vectors), axis=0)]
    assert np.all(vectors[sectors[:, None] != vector_sectors[None, :]] == 0.0)


def test_concrete_excitation_changes_use_energy_coordinates_and_skip_traced_payloads():
    """Excitation changes are read in energy coordinates from concrete operators and unknown for tracers."""
    lowering = np.diag(np.sqrt([1.0, 2.0]), k=1)
    assert concrete_excitation_changes(lowering, (3,)) == frozenset({1})
    assert concrete_excitation_changes(lowering + lowering.T, (3,)) == frozenset({-1, 1})

    # Authored level 1 is the energy ground state, so |0><1| raises the energy index.
    swap = np.eye(3)[:, [1, 0, 2]]
    assert concrete_excitation_changes(np.outer([1, 0, 0], [0, 1, 0]), (3,), swap) == frozenset({-1})

    seen = []
    jax.jit(lambda matrix: seen.append(concrete_excitation_changes(matrix, (3,))) or matrix)(jnp.asarray(lowering))
    assert seen == [None]


def test_declared_changes_drop_structural_zero_bands_of_traced_payloads():
    """Declared total changes limit the candidate bands of traced payloads to that total."""
    dims = (3, 2)
    lower_a = np.kron(np.diag(np.sqrt([1.0, 2.0]), k=1), np.eye(2))
    lower_b = np.kron(np.eye(3), np.diag([1.0], k=1))

    def band_weights(values, changes):
        canonical = CanonicalOperator.from_dense(values, dims=dims, basis="semantic", subsystem_labels=("a", "b"))
        return sorted(_decompose_product_canonical_bands(canonical, dims, total_changes=changes))

    traced = {}

    @jax.jit
    def decompose(values):
        traced["declared"] = band_weights(values, frozenset({1}))
        traced["undeclared"] = band_weights(values, None)
        return values

    decompose(jnp.asarray(lower_a + lower_b, dtype=complex))
    # A traced payload keeps every candidate with the declared total; nothing else.
    assert traced["declared"] == [(0, 1), (1, 0), (2, -1)]
    assert len(traced["undeclared"]) == 5 * 3
    assert band_weights(jnp.asarray(lower_a + lower_b, dtype=complex), frozenset({1})) == [(0, 1), (1, 0)]


def test_effective_terms_declare_channel_excitation_changes_and_round_trip():
    """Retained terms validate, expose, describe and serialize their declared excitation changes."""
    lowering = np.diag([1.0], k=1)
    channels = (CollapseChannel(lowering, 0.1, "loss"),)
    terms = EffectiveTerms(("a",), (2,), np.zeros((2, 2)), channels, excitation_changes={"loss": [1]})
    plain = EffectiveTerms(("a",), (2,), np.zeros((2, 2)), channels)

    assert terms.excitation_changes == {"loss": frozenset({1})}
    assert declared_excitation_changes(terms.expression()) == frozenset({0})
    assert declared_excitation_changes(terms.channel_expression(terms.channels[0])) == frozenset({1})
    assert declared_excitation_changes(plain.expression()) is None
    assert declared_excitation_changes(plain.channel_expression(plain.channels[0])) is None
    assert "total excitation number" in " ".join(terms.physics_notes())
    assert "excitation_changes" not in plain.to_dict()
    loaded = EffectiveTerms.from_dict(json.loads(json.dumps(terms.to_dict())))
    assert loaded.excitation_changes == terms.excitation_changes
    assert EffectiveTerms.from_dict(plain.to_dict()).excitation_changes is None
    with pytest.raises(ValueError, match="unknown effective channels"):
        EffectiveTerms(("a",), (2,), np.zeros((2, 2)), channels, excitation_changes={"gain": [-1]})
