"""Physical detunings and advisory checks for structurally discarded bands."""

import warnings

import numpy as np
import pytest

from quchip import Capacitive, Chip, DuffingTransmon, Exact, Fluxonium, RWA, Resonator


def _guide_chip(g=0.05, frequency=5.0, *, backend="qutip"):
    q = DuffingTransmon(freq=frequency, anharmonicity=-0.25, levels=4, label="q")
    r = Resonator(freq=7.0, levels=5, label="r")
    return Chip([q, r], [Capacitive(q, r, g=g, label="qr")], backend=backend)


def _drop(result, weights):
    return next(term for term in result.dropped_terms if term.band_weights == weights)


@pytest.mark.unit
@pytest.mark.parametrize("frame", ["lab", "rotating"])
def test_ladder_diagnostic_uses_bare_transition_detuning(frame):
    """The guide's largest discarded shift uses the 11.5 GHz transition in either frame."""
    with warnings.catch_warnings(record=True) as caught:
        result = _guide_chip().resolve(frame=frame)
    assert not caught
    term = _drop(result, (1, 1))
    assert term.amplitude == pytest.approx(0.05 * np.sqrt(12), rel=1e-12)
    assert term.frequency == pytest.approx(11.5, abs=1e-12)
    assert term.frame_frequency == pytest.approx(0 if frame == "lab" else 12, abs=1e-12)


@pytest.mark.parametrize("frame", ["lab", "rotating"])
def test_fluxonium_warning_and_diagnostic_match_independent_entries(frame):
    """The discarded near-resonant fluxonium exchange warns without changing the mask."""
    f = Fluxonium(E_C=0.95, E_J=4.2, E_L=0.85, phi_ext=0.45, levels=4, basis="eigen", label="f")
    r = Resonator(freq=6.4, levels=3, label="r")
    chip = Chip([f, r], [Capacitive(f, r, g=0.03, label="fr")])
    if frame == "lab":
        with pytest.warns(UserWarning, match=r"RWA\(\).*\(3, -1\)"):
            shift = chip.dispersive_shift("f", "r")
        # The advisory must preserve the reported structural-RWA result, 39.13 kHz to its printed precision.
        assert float(shift) == pytest.approx(39.13e-6, abs=0.01e-6)
    exact = Chip([f, r], [Capacitive(f, r, g=0.03)], approximation=Exact()).resolve()
    matrix = np.asarray(exact.hamiltonian().matrix())
    # |0,2><3,1| has the largest |V|²/dE in this band. Its bare energies
    # follow from isolated spectra, independently of the diagnostic fields.
    row, col = 2, 10
    amplitude = abs(matrix[row, col])
    detuning = abs(matrix[col, col] - matrix[row, row])
    with pytest.warns(UserWarning, match=r"RWA\(\).*\(3, -1\)"):
        result = chip.resolve(frame=frame)
    term = _drop(result, (3, -1))
    # Both assemblies read the same matrix entries, so only round-off separates them.
    assert term.amplitude == pytest.approx(amplitude, rel=1e-12)
    assert term.frequency == pytest.approx(detuning, abs=1e-12)
    # The expected scale is rounded to three significant figures.
    assert term.amplitude**2 / term.frequency == pytest.approx(0.000971, rel=1e-3)
    if frame == "lab":
        assert np.asarray(result.hamiltonian().matrix())[row, col] == 0
    with warnings.catch_warnings(record=True) as caught:
        kept = chip.resolve(frame=frame, approximation=RWA(keep_bands={(1, -1), (-1, 1), (3, -1), (-3, 1)}))
    assert not caught
    assert all(record.band_weights != (3, -1) for record in kept.dropped_terms)


@pytest.mark.optional_backend
def test_diagnostic_derivatives_include_coupling_and_energy_parameters():
    """JIT and autodiff retain the analytic derivatives of the entrywise shift."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    def diagnostic(theta):
        term = _drop(_guide_chip(theta[0], theta[1], backend="dynamiqs").resolve(frame="lab"), (1, 1))
        return jnp.stack((term.amplitude, term.frequency, term.amplitude**2 / term.frequency))

    point = jnp.array([0.05, 5.0])
    expected = np.array([0.05 * np.sqrt(12), 11.5, 12 * 0.05**2 / 11.5])
    jacobian = np.array([[np.sqrt(12), 0], [0, 1], [24 * 0.05 / 11.5, -12 * 0.05**2 / 11.5**2]])
    with warnings.catch_warnings(record=True) as caught:
        # The expected values and derivatives are analytic, so only round-off remains.
        np.testing.assert_allclose(jax.jit(diagnostic)(point), expected, rtol=1e-12)
        np.testing.assert_allclose(jax.jit(jax.jacrev(diagnostic))(point), jacobian, atol=1e-12)
    assert not caught
    step = 1e-5
    finite_difference = np.column_stack([
        (np.asarray(diagnostic(point + step * direction)) - np.asarray(diagnostic(point - step * direction)))
        / (2 * step) for direction in np.eye(2)
    ])
    # Round-off in central differences, about eps * 11.5 / step = 3e-10, sets the bound.
    np.testing.assert_allclose(finite_difference, jacobian, atol=1e-9)


@pytest.mark.unit
def test_largest_shift_can_have_a_smaller_matrix_element():
    """A weak transition near resonance outranks a stronger far-detuned entry."""
    from quchip.interop.eigenbasis import EigenbasisDevice

    q = EigenbasisDevice([0, 5, 5.01], charge_operator=[[0, 3, 0], [3, 0, 1], [0, 1, 0]], label="q")
    r = Resonator(freq=0.1, levels=2, label="r")
    term = _drop(Chip([q, r], [Capacitive(q, r, g=0.01)]).resolve(), (1, 1))
    # The inputs are exact, so the bounds cover round-off of one product and one energy sum.
    assert term.amplitude == pytest.approx(0.01, abs=1e-14)
    assert term.frequency == pytest.approx(0.11, abs=1e-14)


@pytest.mark.unit
@pytest.mark.parametrize("explicit_references", [False, True])
def test_resonant_discarded_entry_warns_once_and_retains_zero_detuning(explicit_references):
    """Discarding resonant exchange reports zero detuning and one advisory per coupling."""
    a = Resonator(freq=5, levels=2, label="a")
    b = Resonator(freq=5, levels=2, label="b")
    if explicit_references:
        a.reference_freq = b.reference_freq = 5.0
    chip = Chip([a, b], [Capacitive(a, b, g=0.01, label="ab")],
                approximation=RWA(keep_bands={(1, 1), (-1, -1)}))
    with pytest.warns(UserWarning, match="nonzero resonant entry") as caught:
        result = chip.resolve()
    assert len(caught) == 1
    term = _drop(result, (1, -1))
    assert term.amplitude == pytest.approx(0.01)
    assert term.frequency == 0


@pytest.mark.unit
@pytest.mark.parametrize("conserving", [False, True])
@pytest.mark.parametrize("exact", [False, True])
def test_cold_approximation_override_does_not_warn_for_reference_policy(conserving, exact):
    """Cold overrides retain resonant bands without advisories from implicit reference analysis."""
    from quchip.interop.eigenbasis import EigenbasisDevice

    a = EigenbasisDevice([0, 1, 5], charge_operator=[[0, 0, 1], [0, 0, 0.5], [1, 0.5, 0]], label="a")
    b = Resonator(freq=5, levels=2, label="b")
    default = RWA() if conserving else RWA(keep_bands={(1, 1), (-1, -1)})
    chip = Chip([a, b], [Capacitive(a, b, g=0.01, label="ab")], approximation=default)
    requested = Exact() if exact else RWA(keep_bands={(1, -1), (-1, 1), (2, -1), (-2, 1)})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = chip.resolve(frame="lab", approximation=requested)
    assert not caught
    assert all(term.band_weights not in {(2, -1), (-2, 1)} for term in result.dropped_terms)
    assert abs(np.asarray(result.hamiltonian().matrix())[1, 4]) == pytest.approx(0.01)
    # Public analysis uses the chip policy, so it reports the advisory that the reference lookup kept.
    with pytest.warns(UserWarning, match="nonzero resonant entry") as caught:
        frequencies = chip.freq()
    assert len(caught) == 1
    # Reference frequencies still follow the chip policy, not the snapshot override.
    # Both read the same cached dressed energies.
    for label, frequency in frequencies.items():
        assert result.resolved_frame.demod_freqs[label] == pytest.approx(frequency, abs=1e-12)
    with pytest.warns(UserWarning, match="nonzero resonant entry") as caught:
        chip.resolve(frame="lab")
    assert len(caught) == 1


def test_override_resolution_leaves_the_fluxonium_advisory_to_dispersive_shift():
    """A reference lookup under an override fills the analysis cache, and the first public query warns once."""
    f = Fluxonium(E_C=0.95, E_J=4.2, E_L=0.85, phi_ext=0.45, levels=4, basis="eigen", label="f")
    r = Resonator(freq=6.4, levels=3, label="r")
    chip = Chip([f, r], [Capacitive(f, r, g=0.03, label="fr")])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chip.resolve(approximation=Exact())
    assert not caught
    with pytest.warns(UserWarning, match=r"RWA\(\).*\(3, -1\)") as caught:
        chip.dispersive_shift("f", "r")
    assert len(caught) == 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chip.dispersive_shift("f", "r")
    assert not caught


@pytest.mark.unit
def test_cold_conserving_resolution_warns_once():
    """A cold compact reference calculation adds no advisory to the requested resolution."""
    from quchip.interop.eigenbasis import EigenbasisDevice

    a = EigenbasisDevice([0, 1, 5], charge_operator=[[0, 0, 1], [0, 0, 0.5], [1, 0.5, 0]], label="a")
    b = Resonator(freq=5, levels=2, label="b")
    chip = Chip([a, b], [Capacitive(a, b, g=0.01, label="ab")])
    with pytest.warns(UserWarning, match="nonzero resonant entry") as caught:
        result = chip.resolve(frame="lab")
    assert len(caught) == 1
    assert _drop(result, (2, -1)).frequency == 0


@pytest.mark.parametrize("levels, strength", [(4, 0.0), (4, 0.03), (400, 0.03)])
def test_sparse_compact_diagnostics_preserve_resonance_and_mask(levels, strength):
    """Sparse large supports preserve resonant-band warnings and the conserving operator."""
    from quchip.engine.assembly import _resolve_system, _resolved_energy_interactions
    from quchip.interop.eigenbasis import EigenbasisDevice

    charge = np.zeros((levels, levels))
    charge[0, 2] = charge[2, 0] = 1
    charge[1, 2] = charge[2, 1] = 0.5
    energies = np.concatenate(([0, 1, 5], np.arange(6, levels + 3)))
    a = EigenbasisDevice(energies, charge_operator=charge, label="a")
    b = Resonator(freq=5, levels=3, label="b")
    chip = Chip([a, b], [Capacitive(a, b, g=strength, label="ab")])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        terms = list(_resolved_energy_interactions(chip, _resolve_system(chip, chip.backend), chip.approximation))
    assert len(caught) == (1 if strength else 0)
    if strength:
        assert "nonzero resonant entry" in str(caught[0].message)
        assert "(2, -1)" in str(caught[0].message)
    lowering = np.diag(np.sqrt([1, 2]), k=1)
    full = strength * np.kron(charge, 1j * (lowering - lowering.T))
    excitations = (np.arange(levels)[:, None] + np.arange(3)[None, :]).reshape(-1)
    expected = full * (excitations[:, None] == excitations[None, :])
    assert len(terms) == 1
    operator, support = terms[0]
    assert support == (0, 1)
    np.testing.assert_allclose(operator, expected, atol=1e-12)


@pytest.mark.unit
def test_zero_entries_at_resonance_do_not_produce_nan_or_false_warnings():
    """Structural zeros at resonance have zero score in eager and traced diagnostics."""
    import jax
    import jax.numpy as jnp

    from quchip.engine.assembly import _band_diagnostics, _dropped_band_advisory
    from quchip.engine.bands import canonical_to_coo
    from quchip.engine.ir import CanonicalOperator

    def diagnostics(value):
        operator = CanonicalOperator.from_dense(
            jnp.zeros((4, 4)).at[1, 2].set(value), dims=(2, 2), basis="fock", subsystem_labels=("a", "b"),
        )
        return _band_diagnostics(*canonical_to_coo(operator), (2, 2), (jnp.array([0, 5]), jnp.array([0, 5])))

    # Traced payloads keep every entry. Concrete payloads keep only nonzero entries.
    traced = jax.jit(lambda value: jnp.stack(diagnostics(value)[(1, -1)]))
    np.testing.assert_array_equal(traced(0.0), [0, 0, 0])
    np.testing.assert_array_equal(traced(0.1), [0.1, 0, np.inf])
    assert (1, -1) not in diagnostics(0.0)
    np.testing.assert_array_equal(diagnostics(0.1)[(1, -1)], [0.1, 0, np.inf])
    assert _dropped_band_advisory("ab", [((1, -1), False, 0.0), ((1, 1), True, 0.0)]) is None


@pytest.mark.unit
def test_static_diagnostics_are_independent_of_solver_basis():
    """Native and full eigen coordinates give the same physical band diagnostics."""
    from quchip import ChargeBasisTransmon

    results = []
    for basis in ("native", "eigen"):
        q = ChargeBasisTransmon(E_C=0.3, E_J=12, n_g=0.17, num_basis=5,
                               levels=5 if basis == "eigen" else None, basis=basis, label="q")
        r = Resonator(freq=7, levels=2, label="r")
        # The retained exchange dominates in either basis, so neither resolution may warn.
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            results.append(Chip([q, r], [Capacitive(q, r, g=0.01)]).resolve())
    for native, eigen in zip(results[0].dropped_terms, results[1].dropped_terms, strict=True):
        assert native.band_weights == eigen.band_weights
        # The bases differ by one unitary change of coordinates, so only round-off separates them.
        np.testing.assert_allclose([native.amplitude, native.frequency], [eigen.amplitude, eigen.frequency], atol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("detuning", [None, 0.5])
def test_model_declared_drop_keeps_its_detuning_separate_from_the_carrier(detuning):
    """Assembly preserves an authored detuning, including unknown, while resolving its carrier."""
    from quchip.engine.ir import DroppedTerm

    class WithOmittedTerm(Capacitive):
        def dropped_terms(self):
            return [DroppedTerm(self.label, "omitted term", "model approximation",
                                band_weights=(1, 1), amplitude=0.01, frequency=detuning)]

    q = Resonator(freq=5, levels=2, label="q")
    r = Resonator(freq=7, levels=2, label="r")
    result = Chip([q, r], [WithOmittedTerm(q, r, g=0.01, label="qr")], approximation=Exact()).resolve(frame="rotating")
    term = result.dropped_terms[0]
    assert term.frequency == detuning
    # Dressed references move the 12 GHz carrier by about 2e-5 GHz.
    assert term.frame_frequency == pytest.approx(12, rel=1e-5)


@pytest.mark.unit
def test_energy_diagonal_interaction_does_not_warn_for_projection_roundoff():
    """A declared diagonal interaction has no discarded transition even after projection."""
    from quchip import ChargeBasisTransmon, CrossKerr

    q = ChargeBasisTransmon(E_C=0.2, E_J=30, n_g=0.25, levels=4, num_basis=63, basis="eigen", label="q")
    r = Resonator(freq=6, levels=3, label="r")
    chip = Chip([q, r], [CrossKerr(q, r, chi=0.01)])
    with warnings.catch_warnings(record=True) as caught:
        chip.freq()
        result = chip.resolve()
    assert not caught
    assert result.dropped_terms == ()
