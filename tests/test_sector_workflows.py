"""Exact excitation sectors across static queries and inverse design."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quchip import ChargeDrive, Chip, DuffingTransmon, Exact, fit_a_dress
from quchip.chip.effective import conserves_excitation_number
from quchip.declarative import CouplingModel, Scalar, parameter


class Exchange(CouplingModel):
    g: Scalar = parameter(unit="GHz")

    def interaction(self, a, b, p):
        return p.g * (a.adag * b.a + a.a * b.adag)


def pair(*, backend="qutip"):
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = DuffingTransmon(freq=5.4, anharmonicity=-0.22, levels=3, label="r")
    chip = Chip([q, r], [Exchange(q, r, g=0.04, label="qr")], approximation=Exact(), backend=backend)
    chip.wire(ChargeDrive(q, label="xy"))
    return chip


def test_exact_exchange_uses_sectors_and_matches_independent_spectrum():
    """A custom exchange Hamiltonian needs no RWA to conserve the energy-level index."""
    chip = pair()
    assert conserves_excitation_number(chip, chip.approximation)
    assert chip.analysis._dressed_dimension(2) == 3
    expected = 5.2 - np.sqrt(0.2**2 + 0.04**2)
    assert chip.freq("q") == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("backend", ["qutip", pytest.param("dynamiqs", marks=pytest.mark.optional_backend)])
def test_exact_fit_uses_sector_dimension_and_recovers_bare_frequency(backend):
    """An Exact exchange fit respects the sector limit and reproduces its dressed target."""
    if backend == "dynamiqs":
        pytest.importorskip("dynamiqs")
    chip = pair(backend=backend)
    target = 4.95
    result = fit_a_dress(chip, constraints={"q": {"freq": target, "anharmonicity": None},
                                          "r": {"freq": None, "anharmonicity": None},
                                          "qr": {"coupling_strength": None}},
                         vary={"q": ["freq"]}, max_hilbert_dim=2)
    assert result.chip.freq("q") == pytest.approx(target, abs=1e-9)
    assert result.chip["q"].freq == pytest.approx(target + 0.04**2 / (5.4 - target), abs=1e-9)


@pytest.mark.optional_backend
@pytest.mark.validation
def test_exact_sector_frequency_and_drive_gradients_match_analytic_exchange():
    """JIT keeps the exact sector and matches derivatives of the two-mode mixing angle."""
    pytest.importorskip("dynamiqs")
    chip = pair(backend="dynamiqs")

    def outputs(g):
        bound = chip.with_params({"qr.g": g})
        assert bound.analysis._sector_model() is not None
        return jnp.stack([bound.freq("q"), jnp.abs(bound.drive_matrix_elements("q")["xy"])])

    def reference(g):
        gap = jnp.sqrt(0.2**2 + g**2)
        return jnp.stack([5.2 - gap, jnp.sqrt((1 + 0.2 / gap) / 2)])

    actual = jax.jit(lambda g: (outputs(g), jax.jacrev(outputs)(g)))
    def expected(g):
        return reference(g), jax.jacrev(reference)(g)
    for g in (0.02, 0.07):
        np.testing.assert_allclose(actual(g), expected(g), rtol=1e-10, atol=1e-12)


def test_counter_rotating_support_is_not_declared_conserving_at_zero_strength():
    """A zero coefficient does not freeze a nonconserving parameter direction into a sector."""
    from quchip import Capacitive

    source = pair()
    q, r = source.devices
    for g in (0.0, 1e-16, 0.04):
        chip = Chip([q, r], [Capacitive(q, r, g=g)], approximation=Exact())
        assert not conserves_excitation_number(chip, chip.approximation)


def test_undeclared_retained_matrix_is_checked_without_a_tolerance():
    """Concrete retained matrices can prove conservation, but any nonzero forbidden entry rejects it."""
    from quchip.chip.effective import EffectiveTerms

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    for offdiagonal in (0.0, 1e-16):
        h = np.diag([0.0, 0.01, 0.02])
        h[0, 1] = h[1, 0] = offdiagonal
        chip = Chip([q], effective_terms=[EffectiveTerms(("q",), (3,), h)], approximation=Exact())
        assert conserves_excitation_number(chip, chip.approximation) == (offdiagonal == 0)
        if offdiagonal == 0:
            assert chip.freq(q) == pytest.approx(5.01, abs=1e-12)


def test_sector_and_full_fallback_refuse_impossible_allocations(monkeypatch):
    """Static requests reject both an oversized sector and an oversized full fallback before assembly."""
    from quchip import Capacitive
    from quchip.backend import _memory

    chip = pair()
    model = chip.analysis._sector_model()
    monkeypatch.setattr(_memory, "available_memory_bytes", lambda: 1)
    with pytest.raises(MemoryError, match="Excitation sector"):
        model.eigensystem(2)
    q, r = chip.devices
    full = Chip([q, r], [Capacitive(q, r, g=0.04)], approximation=Exact())
    monkeypatch.setattr(_memory, "available_memory_bytes", lambda: 1000)
    with pytest.raises(MemoryError, match="Full static eigensystem"):
        full.freq(q)


@pytest.mark.parametrize("backend", ["qutip", pytest.param("dynamiqs", marks=pytest.mark.optional_backend)])
def test_large_exact_model_fits_and_reads_drive_elements_without_product_space(backend):
    """A 24-mode Exact exchange chain fits and reads drives using its independent 24-state model."""
    if backend == "dynamiqs":
        pytest.importorskip("dynamiqs")
    devices = [DuffingTransmon(freq=5 + 0.07*i, anharmonicity=-0.25, levels=3, label=f"q{i}")
               for i in range(24)]
    edges = [Exchange(a, b, g=0.01, label=f"edge{i}")
             for i, (a, b) in enumerate(zip(devices[:-1], devices[1:], strict=True))]
    chip = Chip(devices, edges, approximation=Exact(), backend=backend)
    chip.wire(ChargeDrive(devices[0], label="xy"))
    matrix = np.diag([q.freq for q in devices]) + np.diag([0.01]*23, 1) + np.diag([0.01]*23, -1)
    matrix[0, 1] = matrix[1, 0] = 0.015
    values, vectors = np.linalg.eigh(matrix)
    column = np.argmax(abs(vectors[0]))
    constraints = {q.label: {"freq": None, "anharmonicity": None} for q in devices}
    constraints.update({edge.label: {"coupling_strength": None} for edge in edges})
    constraints["q0"]["freq"] = values[column]
    result = fit_a_dress(chip, constraints=constraints, vary={"edge0": ["g"]}, max_hilbert_dim=24)
    assert result.solver_info["jacobian"] == ("jax" if backend == "dynamiqs" else "finite-difference")
    assert result.chip.coupling_map["edge0"].g == pytest.approx(0.015, abs=1e-9)
    assert abs(result.chip.drive_matrix_elements("q0")["xy"]) == pytest.approx(abs(vectors[0, column]), abs=1e-10)


def test_exact_vacuum_response_matches_stationary_response():
    """A conserving Exact nonlinear model keeps the vacuum VNA route and agrees with the full solver."""
    from quchip import PortNetwork, VNA

    source = pair()
    network = PortNetwork()
    network.port("p", target=source.devices[0], rate=0.02)
    chip = Chip(source.devices, source.couplings, approximation=Exact(), port_network=network)
    frequencies = np.array([4.97, 5.0, 5.42])
    compact = VNA(chip).sweep(frequencies)
    full = VNA(chip).sweep(frequencies, options={})
    assert compact.diagnostics[0]["solver"] == "vacuum_response"
    np.testing.assert_allclose(compact.matrix, full.matrix, atol=1e-9, rtol=0)


@pytest.mark.optional_backend
def test_fit_rechecks_dimension_when_tracing_cannot_prove_conservation():
    """Opaque traced matrix support cannot expand a fit beyond its requested dimension limit."""
    pytest.importorskip("dynamiqs")

    class MatrixExchange(Exchange):
        def interaction(self, a, b, p):
            lowering = np.diag(np.sqrt(np.arange(1, 3)), 1)
            return self.g * (np.kron(lowering.T, lowering) + np.kron(lowering, lowering.T))

    q, r = pair().devices
    chip = Chip([q, r], [MatrixExchange(q, r, g=.04, label="qr")], approximation=Exact(), backend="dynamiqs")
    assert chip.analysis._dressed_dimension(1) == 2
    with pytest.raises(ValueError, match="dimension 9, exceeding max_hilbert_dim=2"):
        fit_a_dress(chip, constraints={"q": {"freq": 4.95, "anharmonicity": None},
                                      "r": {"freq": None, "anharmonicity": None},
                                      "qr": {"coupling_strength": None}},
                     vary={"qr": ["g"]}, max_hilbert_dim=2)


def test_opaque_traced_operator_requires_declared_changes():
    """Undeclared traced matrix support is unknown; a supplied conservation declaration survives tracing."""
    from quchip.chip.effective import authored_excitation_changes
    from quchip.declarative.expr import PhysicsExpr
    from quchip.engine.assembly import _resolve_system

    chip = pair()
    bases = _resolve_system(chip, chip.backend).bases
    seen = []

    @jax.jit
    def check(matrix):
        for declaration in (None, (0,)):
            expr = PhysicsExpr.from_matrix(matrix, labels=("q",), dims=(3,), excitation_changes=declaration)
            seen.append(authored_excitation_changes(expr, ("q",), chip.backend, bases))
        seen.append(authored_excitation_changes(matrix, ("q",), chip.backend, bases))
        return matrix

    check(jnp.eye(3))
    assert seen == [None, frozenset({0}), None]


def test_direct_coupling_target_does_not_consume_an_eigensystem_budget():
    """A bare-strength-only fit uses no dressed matrix and therefore needs no Hilbert-space allowance."""
    from quchip import Capacitive

    q, r = pair().devices
    chip = Chip([q, r], [Capacitive(q, r, g=0.04, label="qr")], approximation=Exact())
    result = fit_a_dress(chip, constraints={"q": {"freq": None, "anharmonicity": None},
                                          "r": {"freq": None, "anharmonicity": None},
                                          "qr": {"cross_kerr": None, "coupling_strength": 0.05}},
                         vary={"qr": ["g"]}, max_hilbert_dim=1)
    assert result.chip.coupling_map["qr"].g == pytest.approx(0.05, abs=1e-10)
