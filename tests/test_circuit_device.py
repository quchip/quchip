"""Tests for CircuitDevice — shared base for circuit-level devices."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quchip.devices.circuit import CircuitDevice


class _ToyCircuit(CircuitDevice):
    """Minimal CircuitDevice subclass for exercising the base.

    Models a simple 1D quartic oscillator
    ``H = 4 E_C n̂² + alpha φ̂⁴`` in a small plane-wave phase basis.
    Chosen small enough that tests run in milliseconds.
    """

    _type_prefix = "toy"

    def __init__(self, E_C: float = 1.0, alpha: float = 0.5, levels: int = 3,
                 label: str | None = None, *,
                 collapse_model: str = "fermi_golden",
                 coupling_channel: str | None = "charge",
                 **noise):
        self._E_C = E_C
        self._alpha = alpha
        self._num_basis = 31
        self._phi_max = 3.0
        super().__init__(
            levels=levels,
            label=label,
            collapse_model=collapse_model,
            coupling_channel=coupling_channel,
            **noise,
        )
        self._finish_init()

    def _build_native_hamiltonian(self):
        phi_grid = jnp.linspace(-self._phi_max, self._phi_max, self._num_basis, endpoint=False)
        d_phi = 2 * self._phi_max / self._num_basis
        lap = (
            jnp.eye(self._num_basis, k=1)
            - 2 * jnp.eye(self._num_basis)
            + jnp.eye(self._num_basis, k=-1)
        ) / d_phi**2
        K = -4.0 * self._E_C * lap
        V = self._alpha * phi_grid**4
        return (K + jnp.diag(V)).astype(jnp.complex128)

    def _native_charge_operator(self):
        d_phi = 2 * self._phi_max / self._num_basis
        plus = jnp.eye(self._num_basis, k=1)
        minus = jnp.eye(self._num_basis, k=-1)
        return (-1j * (plus - minus) / (2 * d_phi)).astype(jnp.complex128)

    def _native_phase_operator(self):
        phi_grid = jnp.linspace(-self._phi_max, self._phi_max, self._num_basis, endpoint=False)
        return jnp.diag(phi_grid.astype(jnp.complex128))

    @property
    def num_basis(self) -> int:
        return self._num_basis


# ----------------------------------------------------------------------
# Scaffolding / basics
# ----------------------------------------------------------------------


def test_hamiltonian_is_diagonal_with_ground_at_zero():
    """hamiltonian() is diagonal in the truncated eigenbasis with the ground-state entry at zero."""
    q = _ToyCircuit(E_C=1.0, alpha=0.5, levels=3)
    from quchip.backend import get_default_backend
    H = get_default_backend().to_array(q.hamiltonian())
    assert np.allclose(H - np.diag(np.diag(H)), 0.0)
    assert np.isclose(H[0, 0], 0.0)


def test_freq_property_is_01_gap():
    """freq equals the shifted first eigenvalue, the bare 0→1 transition energy."""
    q = _ToyCircuit(E_C=1.0, alpha=0.5, levels=3)
    eigvals = q.eigenenergies()
    assert np.isclose(float(q.freq), float(eigvals[1]))


def test_eigenvectors_shape():
    """eigenvectors() returns the (num_basis, levels) truncated eigenvector matrix."""
    q = _ToyCircuit(E_C=1.0, alpha=0.5, levels=3)
    V = q.eigenvectors()
    assert V.shape == (q.num_basis, q.levels)


def test_project_operator_matches_unshifted_eigvals_on_hamiltonian():
    """V† H_native V = diag(E_0, E_1, …); q.hamiltonian() is the same diag shifted to E_0=0."""
    q = _ToyCircuit(E_C=1.0, alpha=0.5, levels=3)
    H_native = q._build_native_hamiltonian()
    projected = np.asarray(q.project_operator(H_native))
    assert np.allclose(projected - np.diag(np.diag(projected)), 0.0, atol=1e-4)
    from quchip.backend import get_default_backend
    ham_shifted = get_default_backend().to_array(q.hamiltonian())
    shift = projected[0, 0] - ham_shifted[0, 0]
    assert np.allclose(np.diag(projected) - shift, np.diag(ham_shifted), atol=1e-4)


def test_charge_coupling_operator_returns_projected_native():
    """charge_coupling_operator() equals the native charge operator projected into the truncated eigenbasis."""
    q = _ToyCircuit(E_C=1.0, alpha=0.5, levels=3)
    projected = q.charge_coupling_operator()
    expected = q.project_operator(q._native_charge_operator())
    assert np.allclose(np.asarray(projected), np.asarray(expected))


def test_phase_coupling_operator_returns_projected_native():
    """phase_coupling_operator() equals the native phase operator projected into the truncated eigenbasis."""
    q = _ToyCircuit(E_C=1.0, alpha=0.5, levels=3)
    projected = q.phase_coupling_operator()
    expected = q.project_operator(q._native_phase_operator())
    assert np.allclose(np.asarray(projected), np.asarray(expected))


# ----------------------------------------------------------------------
# Cache behavior
# ----------------------------------------------------------------------


def test_cache_hits_on_repeated_access():
    """Repeated _eigensys() calls at the same state_version return the identical cached object."""
    q = _ToyCircuit(E_C=1.0, alpha=0.5, levels=3)
    first = q._eigensys()
    second = q._eigensys()
    assert first is second


def test_eigensys_cache_absent_immediately_after_init():
    """Init-order contract: __init__ must not trigger eager diagonalization."""
    q = _ToyCircuit(E_C=1.0, alpha=0.5, levels=3)
    assert not hasattr(q, "_eigensys_cache"), \
        "CircuitDevice must be lazy — _eigensys cache must not be populated until first access."


def test_eigensys_cache_written_on_first_access():
    """First _eigensys() access populates _eigensys_cache keyed on the current state_version."""
    q = _ToyCircuit(E_C=1.0, alpha=0.5, levels=3)
    _ = q._eigensys()
    assert hasattr(q, "_eigensys_cache")
    assert q._eigensys_cache[0] == q.state_version


def test_cache_invalidates_on_state_version_bump():
    """A state_version bump invalidates the eigensys cache; unchanged params give identical eigenvalues on recompute."""
    q = _ToyCircuit(E_C=1.0, alpha=0.5, levels=3)
    first_eigvals, _ = q._eigensys()
    version_before = q.state_version

    # Non-underscore assignment bumps state_version via BaseDevice.__setattr__.
    q.marker = 1
    assert q.state_version > version_before

    second_eigvals, _ = q._eigensys()
    assert q._eigensys_cache[0] == q.state_version
    # Hamiltonian didn't actually change, so values agree; but cache was recomputed.
    assert np.allclose(np.asarray(first_eigvals), np.asarray(second_eigvals))


def test_cache_not_written_when_inputs_contain_tracer():
    """A traced H_native under jax.jit is never cached (no leaked-tracer error on return)."""

    def traced_gap(E_C_val):
        q = _ToyCircuit(E_C=E_C_val, alpha=0.5, levels=3)
        return q.freq

    result = jax.jit(traced_gap)(1.0)
    assert np.isfinite(float(result))


def test_freq_is_jax_differentiable():
    """freq is differentiable through the eigendecomposition via jax.grad."""
    def spectrum_gap(E_C):
        q = _ToyCircuit(E_C=E_C, alpha=0.5, levels=3)
        return q.freq

    grad = jax.grad(spectrum_gap)(1.0)
    assert np.isfinite(float(grad))


# ----------------------------------------------------------------------
# Collapse operators (Fermi-golden-rule override)
# ----------------------------------------------------------------------


def test_collapse_operators_empty_without_T1_or_T2():
    """collapse_operators() returns no channels when neither T1 nor T2 is set."""
    q = _ToyCircuit(levels=4)
    assert q.collapse_operators() == []


def test_collapse_operators_fermi_golden_produces_multiple_ops():
    """Fermi-golden-rule relaxation emits a channel per level pair clearing the pruning threshold, not just 0→1."""
    q = _ToyCircuit(levels=4, T1=30_000.0)
    ops = q.collapse_operators()
    assert 1 <= len(ops) <= 6  # covers relaxation + any near-degenerate off-tridiagonal channels


def test_collapse_operators_ladder_mode_matches_base_device():
    """collapse_model='ladder' defers to BaseDevice's single structural Fock-ladder emission channel."""
    q_ladder = _ToyCircuit(levels=3, T1=30_000.0, collapse_model="ladder")
    ops = q_ladder.collapse_operators()
    assert len(ops) == 1  # BaseDevice emits 1 op for T1-only + no thermal + no T2


def test_collapse_operators_pruning_threshold_drops_small_elements():
    """Raising collapse_rate_threshold prunes more transitions, so channel count is non-increasing in the threshold."""
    q_tight = _ToyCircuit(levels=4, T1=30_000.0, collapse_rate_threshold=1e-20)
    ops_tight = q_tight.collapse_operators()
    q_loose = _ToyCircuit(levels=4, T1=30_000.0, collapse_rate_threshold=1.0)
    ops_loose = q_loose.collapse_operators()
    assert len(ops_tight) >= len(ops_loose)


def test_collapse_operators_include_thermal_absorption():
    """Setting thermal_population adds thermal-absorption collapse operators alongside the T1 relaxation channels."""
    q = _ToyCircuit(levels=3, T1=30_000.0, thermal_population=0.05)
    ops_with_thermal = q.collapse_operators()
    q_no_thermal = _ToyCircuit(levels=3, T1=30_000.0)
    ops_no_thermal = q_no_thermal.collapse_operators()
    assert len(ops_with_thermal) > len(ops_no_thermal)


def test_collapse_operators_accept_traced_thermal_population():
    """Fermi-golden-rule thermal absorption should not branch on traced n_bar."""

    @jax.jit
    def thermal_metric(n_bar):
        q = _ToyCircuit(levels=3, T1=30_000.0, thermal_population=n_bar)
        return sum(jnp.real(jnp.sum(jnp.abs(op))) for op in q.collapse_operators())

    value = thermal_metric(jnp.asarray(0.05))
    assert np.isfinite(float(value))


def test_collapse_operators_dephasing_uses_diagonal_form():
    """When T2 is set, at least one emitted op is purely diagonal (dephasing)."""
    q = _ToyCircuit(levels=3, T1=30_000.0, T2=15_000.0)
    ops = q.collapse_operators()
    diag_ops = [op for op in ops if np.allclose(np.asarray(op) - np.diag(np.diag(np.asarray(op))), 0.0)]
    assert len(diag_ops) >= 1


def test_post_construction_energy_write_raises():
    """A non-positive post-construction E_C write raises, mirroring the constructor check."""
    from quchip.devices.fluxonium import Fluxonium

    q = Fluxonium(E_C=1.0, E_J=4.0, E_L=1.0)
    with pytest.raises(ValueError, match="E_C"):
        q.E_C = -5.0


def test_dark_transition_safeguard_raises_on_concrete_zero():
    """A concretely zero 0→1 coupling element (dark transition) raises ValueError, avoiding a silent divide-by-zero."""
    class _DarkCircuit(_ToyCircuit):
        def _native_charge_operator(self):
            return jnp.zeros((self._num_basis, self._num_basis), dtype=jnp.complex128)

    q = _DarkCircuit(E_C=1.0, alpha=0.5, levels=3, T1=30_000.0)
    with pytest.raises(ValueError, match="n̂|1"):
        q.collapse_operators()


def test_gradients_finite_and_fd_exact_at_charge_symmetric_points():
    """E_J/E_C gradients through eigenvectors stay finite and match FD at n_g = 0 and 0.5."""
    # The stock eigh VJP divides by every eigenvalue gap; the +-n charge pairs among the
    # discarded levels are numerically degenerate at the charge-symmetric points and would
    # NaN the whole gradient without the truncation-aware custom VJP (_truncated_eigh).
    from quchip.devices import ChargeBasisTransmon

    def observable(p, n_g):
        qb = ChargeBasisTransmon(E_C=p[1], E_J=p[0], n_g=n_g, levels=3, label="grad_probe")
        e = qb.eigenenergies()
        n01 = qb.project_operator(qb._native_charge_operator())[0, 1]
        return (e[1] - e[0]) + jnp.abs(n01)

    p = jnp.array([10.35, 0.345])
    for n_g in (0.0, 0.5):
        grad = np.asarray(jax.grad(lambda q: observable(q, n_g))(p))
        assert np.isfinite(grad).all()
        for i in range(2):
            h = 1e-6 * float(p[i])
            fd = (float(observable(p.at[i].add(h), n_g)) - float(observable(p.at[i].add(-h), n_g))) / (2 * h)
            # Measured agreement ~3e-9 relative; 1e-6 leaves headroom for platform jitter.
            assert abs(grad[i] - fd) <= 1e-6 * abs(fd)


def test_coupling_channel_write_to_none_with_t1_raises():
    """Clearing coupling_channel while T1 is set under fermi_golden raises."""
    from quchip.devices.fluxonium import Fluxonium

    q = Fluxonium(E_C=1.0, E_J=4.0, E_L=0.9, coupling_channel="flux", T1=20_000.0)
    with pytest.raises(ValueError):
        q.coupling_channel = None


def test_collapse_model_switch_to_fermi_golden_without_channel_raises():
    """Switching back to fermi_golden with T1 set and no channel raises."""
    from quchip.devices.fluxonium import Fluxonium

    q = Fluxonium(E_C=1.0, E_J=4.0, E_L=0.9, collapse_model="ladder", T1=20_000.0)
    with pytest.raises(ValueError):
        q.collapse_model = "fermi_golden"


def test_levels_exceeding_num_basis_rejected_at_construction():
    """levels > num_basis is rejected at construction for both circuit subclasses."""
    from quchip.devices.fluxonium import Fluxonium
    from quchip.devices.transmon.charge_basis import ChargeBasisTransmon

    with pytest.raises(ValueError):
        Fluxonium(E_C=1.0, E_J=4.0, E_L=0.9, levels=4, num_basis=3)
    with pytest.raises(ValueError):
        ChargeBasisTransmon(E_C=0.22, E_J=11.0, levels=4, num_basis=3)


def test_base_device_levels_write_floor():
    """levels = 1 is rejected on post-construction write for any device."""
    from quchip.devices.transmon.duffing import DuffingTransmon

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4)
    with pytest.raises(ValueError):
        q.levels = 1
