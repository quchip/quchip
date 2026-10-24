"""Exact excitation sectors across static queries and inverse design."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quchip import ChargeDrive, Chip, DuffingTransmon, Exact, fit_a_dress
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
    assert chip.excitation_structure().conserves
    assert chip.excitation_structure().largest_block(2) == 3
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
        assert not chip.excitation_structure().conserves


@pytest.mark.optional_backend
def test_traced_counter_rotating_frequency_uses_the_full_eigensystem_as_eager():
    """A traced frequency keeps the full eigensystem that plain evaluation uses for counter-rotating terms."""
    pytest.importorskip("dynamiqs")
    from quchip import Capacitive, Resonator

    q = DuffingTransmon(freq=5., anharmonicity=-.25, levels=3, label="q")
    r = Resonator(freq=6., levels=3, label="r")
    chip = Chip([q, r], [Capacitive(q, r, g=.06)], approximation=Exact(), backend="dynamiqs")

    def frequency(x):
        bound = chip.with_params({"q.freq": x})
        assert bound.analysis._sector_model() is None
        return bound.freq("q")

    value, derivative = jax.jit(jax.value_and_grad(frequency))(5.)
    step = 1e-4
    difference = (frequency(5. + step) - frequency(5. - step)) / (2 * step)
    # Plain and compiled calls diagonalize the same 9 x 9 Hamiltonian, so only round-off differs.
    np.testing.assert_allclose(value, frequency(5.), atol=1e-12)
    np.testing.assert_allclose(jax.grad(frequency)(5.), derivative, atol=1e-12)
    # The central difference has O(step²) truncation error near 1e-10 and round-off divided by the step.
    np.testing.assert_allclose(derivative, difference, atol=1e-9)


def test_undeclared_retained_matrix_is_checked_without_a_tolerance():
    """Concrete retained matrices can prove conservation, but any nonzero forbidden entry rejects it."""
    from quchip.chip.effective import EffectiveTerms

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    for offdiagonal in (0.0, 1e-16):
        h = np.diag([0.0, 0.01, 0.02])
        h[0, 1] = h[1, 0] = offdiagonal
        chip = Chip([q], effective_terms=[EffectiveTerms(("q",), (3,), h)], approximation=Exact())
        assert chip.excitation_structure().conserves == (offdiagonal == 0)
        if offdiagonal == 0:
            assert chip.freq(q) == pytest.approx(5.01, abs=1e-12)


def test_projected_cascade_port_is_not_declared_conserving():
    """A retained map that mixes a cascaded port's levels defeats the proof, so the default state is dressed."""
    from quchip import EffectiveTerms, PortNetwork, QuantumSequence
    from quchip.chip.effective import OperatorProjection

    a = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="a")
    b = DuffingTransmon(freq=5.2, anharmonicity=-0.25, levels=2, label="b")
    network = PortNetwork(label="line")
    upstream, downstream = network.port("a_out", target=a, rate=1.0), network.port("b_out", target=b, rate=1.0)
    network.cascade(upstream, downstream)
    network.expose("line", input=upstream.input, output=downstream.output)
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)
    rotated = EffectiveTerms(("a",), (2,), np.zeros((2, 2)), label="rotated",
                             projection=OperatorProjection(("a",), (2,), ("a",), (2,), hadamard))
    chip = Chip([a, b], port_network=network, effective_terms=[rotated], approximation=Exact())
    # Both authored ports lower one level, but the projected upstream port also keeps and raises levels.
    structure = chip.excitation_structure()
    assert not structure.conserves
    assert [(term.label, term.changes) for term in structure.breaking] == [("a_out", frozenset({-1, 0, 1}))]
    default = QuantumSequence(chip).build_problem([0.0, 1.0]).initial_state
    default = np.ravel(np.asarray(chip.backend.to_array(default)))
    np.testing.assert_allclose(default, np.ravel(np.asarray(chip.backend.to_array(chip.state({})))), atol=1e-12)
    assert abs(default[0]) < 1 - 1e-6


def test_sector_and_full_fallback_refuse_impossible_allocations(monkeypatch):
    """Static requests reject an oversized sector before assembly and an oversized full eigensystem before solving."""
    from quchip import Capacitive
    from quchip.utils import memory as _memory

    chip = pair()
    model = chip.analysis._sector_model()
    monkeypatch.setattr(_memory, "available_memory_bytes", lambda: 1)
    with pytest.raises(MemoryError, match="Excitation sector"):
        model.eigensystem(2)
    q, r = chip.devices
    full = Chip([q, r], [Capacitive(q, r, g=0.04)], approximation=Exact())
    # The sparse static model of the 9 states fits in 4 kB, and its eigensystem needs 7.8 kB.
    monkeypatch.setattr(_memory, "available_memory_bytes", lambda: 4000)
    with pytest.raises(MemoryError, match="Full static eigensystem"):
        full.freq(q)


@pytest.mark.parametrize("backend", ["qutip", pytest.param("dynamiqs", marks=pytest.mark.optional_backend)])
def test_conserving_chip_state_reads_its_sector_and_matches_the_full_eigensystem(backend):
    """A sector state in solver coordinates equals the assigned column of the full eigensystem."""
    if backend == "dynamiqs":
        pytest.importorskip("dynamiqs")
    import warnings

    from quchip import Capacitive, RWA
    from quchip.devices.transmon.charge_basis import ChargeBasisTransmon

    # The charge-basis device makes the energy-to-solver map nontrivial.
    q = ChargeBasisTransmon(E_C=0.25, E_J=12.0, n_g=0.0, num_basis=7, basis="native", label="q")
    r = DuffingTransmon(freq=5.1, anharmonicity=-0.22, levels=3, label="r")
    chip = Chip([q, r], [Capacitive(q, r, g=0.03)], approximation=RWA(), backend=backend)
    assert chip.analysis._sector_model() is not None
    dressed = chip.dress()
    matrix = np.asarray(dressed.eigenvector_matrix)
    labels = [label for label in dressed.bare_labels if sum(label) <= 2]
    assert len(labels) == 6
    with warnings.catch_warnings():
        # The 2-excitation states hybridize; the comparison still covers them.
        warnings.filterwarnings("ignore", message="Dressed state label")
        for label in labels:
            state = np.ravel(np.asarray(chip.backend.to_array(chip.state(q=label[0], r=label[1]))))
            np.testing.assert_allclose(state, matrix[:, dressed.state_map[label]], atol=1e-10)


def test_reduced_chip_state_and_default_initial_state_need_no_full_eigensystem(monkeypatch):
    """A conserving reduced chip prepares its states from sectors within a budget that refuses a full eigensystem."""
    from quchip import Capacitive, QuantumSequence, RWA, Resonator, Square, eliminate
    from quchip.utils import memory as _memory

    q = DuffingTransmon(freq=5.3, anharmonicity=-0.26, levels=3, label="q")
    modes = [Resonator(freq=6.0 + 0.1 * i, levels=3, label=f"m{i}") for i in range(5)]
    readout = Resonator(freq=6.9, levels=3, label="readout")
    couplings = [Capacitive(q, modes[0], g=0.03), Capacitive(q, readout, g=0.05)]
    couplings += [Capacitive(a, b, g=0.02) for a, b in zip(modes, modes[1:])]
    source = Chip([q, *modes, readout], couplings, approximation=RWA())
    source.wire(ChargeDrive(q, label="xy"))
    chip = eliminate(source, "readout", local=True).chip
    sequence = QuantumSequence(chip)
    sequence.schedule("xy", envelope=Square(duration=1, amplitude=0.01), freq=5.3)
    # The full eigensystem of the 729 states needs 51 MB. The sectors and the sparse solve need less than 1 MB.
    monkeypatch.setattr(_memory, "available_memory_bytes", lambda: 10**7)
    with pytest.raises(MemoryError, match="Full static eigensystem"):
        chip.dress()
    problem = sequence.build_problem([0, 1])
    assert problem.engine_result.excitation_space is None
    np.testing.assert_array_equal(
        chip.backend.to_array(problem.initial_state), chip.backend.to_array(chip.bare_state({})),
    )
    state = np.ravel(np.asarray(chip.backend.to_array(chip.state(q=1))))
    assert abs(state[3**5]) > 0.99


@pytest.mark.optional_backend
def test_traced_sector_state_matches_the_eager_state_and_central_differences():
    """A traced sector state equals the eager state, and its derivative matches central differences."""
    pytest.importorskip("dynamiqs")
    chip = pair(backend="dynamiqs")

    def state(g):
        bound = chip.with_params({"qr.g": g})
        return jnp.ravel(bound.backend.to_array(bound.state(q=1)))

    np.testing.assert_allclose(jax.jit(state)(0.04), state(0.04), atol=1e-12)
    _, tangent = jax.jvp(jax.jit(state), (0.04,), (1.0,))
    central = (state(0.04 + 1e-6) - state(0.04 - 1e-6)) / 2e-6
    np.testing.assert_allclose(tangent, central, atol=1e-8)
    assert float(jnp.max(jnp.abs(tangent))) > 0.1


def test_large_exact_model_fits_and_reads_drive_elements_without_product_space():
    """A 24-mode Exact exchange chain fits and reads drives using its independent 24-state model."""
    devices = [DuffingTransmon(freq=5 + 0.07*i, anharmonicity=-0.25, levels=3, label=f"q{i}")
               for i in range(24)]
    edges = [Exchange(a, b, g=0.01, label=f"edge{i}")
             for i, (a, b) in enumerate(zip(devices[:-1], devices[1:], strict=True))]
    chip = Chip(devices, edges, approximation=Exact())
    chip.wire(ChargeDrive(devices[0], label="xy"))
    matrix = np.diag([q.freq for q in devices]) + np.diag([0.01]*23, 1) + np.diag([0.01]*23, -1)
    matrix[0, 1] = matrix[1, 0] = 0.015
    values, vectors = np.linalg.eigh(matrix)
    column = np.argmax(abs(vectors[0]))
    constraints = {q.label: {"freq": None, "anharmonicity": None} for q in devices}
    constraints.update({edge.label: {"coupling_strength": None} for edge in edges})
    constraints["q0"]["freq"] = values[column]
    result = fit_a_dress(chip, constraints=constraints, vary={"edge0": ["g"]}, max_hilbert_dim=24)
    assert result.solver_info["jacobian"] == "finite-difference"
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
    assert chip.excitation_structure().largest_block(1) == 2
    with pytest.raises(ValueError, match="dimension 9, exceeding max_hilbert_dim=2"):
        fit_a_dress(chip, constraints={"q": {"freq": 4.95, "anharmonicity": None},
                                      "r": {"freq": None, "anharmonicity": None},
                                      "qr": {"coupling_strength": None}},
                     vary={"qr": ["g"]}, max_hilbert_dim=2)


def test_opaque_traced_operator_requires_declared_changes():
    """Undeclared traced matrix support is unknown; a supplied conservation declaration survives tracing."""
    from quchip.engine.excitation import authored_excitation_changes
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
