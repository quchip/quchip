"""Local-energy observables keep their meaning across solver representations."""

import numpy as np
import pytest

from quchip import Chip, QuantumSequence
from quchip.declarative import DeviceModel, Scalar, parameter


class TiltedMode(DeviceModel):
    freq: Scalar = parameter(default=1.0)
    mixing: Scalar = parameter(default=0.3)

    def local_hamiltonian(self, op, p):
        return p.freq * op.n + p.mixing * op.sigma_x + 0.2 * op.sigma_y


@pytest.mark.parametrize("frequency", [1.0, -1.0])
def test_constant_level_expression_keeps_tilted_and_reordered_energy_states(frequency):
    import jax
    from quchip.declarative.expr import materialize_array
    from quchip.declarative.ops import LocalOps

    q = TiltedMode(freq=frequency, levels=3, label="q")
    _, vectors = np.linalg.eigh(np.asarray(materialize_array(q.unresolved_hamiltonian())))
    expected = vectors @ np.diag(np.arange(3)) @ vectors.conj().T

    def operator():
        return materialize_array(LocalOps(label="q", space=q.local_space(), device=q).level)

    np.testing.assert_allclose(operator(), expected, atol=1e-12)
    np.testing.assert_allclose(jax.jit(operator)(), expected, atol=1e-12)


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
@pytest.mark.parametrize("frame", ["lab", "rotating"])
def test_saved_level_expression_keeps_its_hamiltonian_after_source_changes(backend, frame):
    from quchip import Exact, Resonator
    from quchip.declarative import CouplingModel
    from quchip.declarative.expr import materialize_array, materialize_expr
    from quchip.declarative.ops import LocalOps

    q = TiltedMode(levels=3, label="q")
    r = Resonator(freq=4.0, levels=2, label="r")
    saved = LocalOps(label="q", space=q.local_space(), device=q).level
    original = np.asarray(materialize_array(saved))

    class SavedCoupling(CouplingModel):
        def interaction(self, a, b, p):
            return 0.1 * saved * b.I

    q.mixing = 0.8
    frame_spec = "lab" if frame == "lab" else {"q": q.freq, "r": r.freq}
    chip = Chip([q, r], [SavedCoupling(q, r)], backend=backend, frame=frame_spec, approximation=Exact())
    resolved = chip.resolve()
    hq = np.asarray(materialize_array(q.unresolved_hamiltonian()))
    hr = np.asarray(materialize_array(r.unresolved_hamiltonian()))
    expected = np.kron(hq + 0.1 * original, np.eye(2)) + np.kron(np.eye(3), hr)
    time = 0.137
    if frame == "rotating":
        _, vectors = np.linalg.eigh(hq)
        levels = vectors @ np.diag(np.arange(3)) @ vectors.conj().T
        uq = (vectors * np.exp(2j * np.pi * q.freq * np.arange(3) * time)) @ vectors.conj().T
        ur = np.diag(np.exp(2j * np.pi * r.freq * np.arange(2) * time))
        rotation = np.kron(uq, ur)
        expected = rotation @ expected @ rotation.conj().T
        expected -= q.freq * np.kron(levels, np.eye(2)) + r.freq * np.kron(np.eye(3), np.diag(np.arange(2)))
    np.testing.assert_allclose(resolved.hamiltonian().matrix(backend=chip.backend, t=time), expected, atol=1e-12)
    np.testing.assert_allclose(materialize_array(saved), original, atol=1e-12)

    shifted = materialize_expr(saved, chip.backend, bindings={"q.mixing": 0.6}, local_bases=resolved.bases)
    direct = materialize_array(saved, bindings={"q.mixing": 0.6})
    np.testing.assert_allclose(chip.backend.to_array(shifted), direct, atol=1e-12)


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_custom_operator_name_does_not_prove_rotating_frame_invariance(backend):
    from quchip import Exact, Resonator
    from quchip.declarative import CouplingModel
    from quchip.devices.spaces import CustomSpace

    class CustomMode(DeviceModel):
        def local_space(self):
            return CustomSpace(2, {"n": np.diag([0, 1]), "I": np.asarray([[0, 1], [1, 0]])})

        def local_hamiltonian(self, op, p):
            return op.n

    class CustomCoupling(CouplingModel):
        def interaction(self, a, b, p):
            return 0.1 * a.I * b.level

    q = CustomMode(levels=2, label="q")
    r = Resonator(freq=4.0, levels=2, label="r")
    chip = Chip([q, r], [CustomCoupling(q, r)], backend=backend,
                frame={"q": 1.0, "r": 4.0}, approximation=Exact())
    time = 0.137
    phase = np.exp(2j * np.pi * time)
    expected = 0.1 * np.kron([[0, phase.conjugate()], [phase, 0]], np.diag([0, 1]))
    np.testing.assert_allclose(chip.hamiltonian().matrix(backend=chip.backend, t=time), expected, atol=1e-12)


@pytest.mark.parametrize("basis", ["native", "eigen"])
@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_local_energy_states_and_default_paulis_agree(basis, backend):
    """Ground, excited, coherent and leakage states obey the same Pauli signs."""
    if backend == "dynamiqs":
        pytest.importorskip("dynamiqs")
    q = TiltedMode(levels=3, label="q")
    chip = Chip([q], basis=basis, backend=backend)
    zero, one, two = (chip.bare_state(q=level) for level in range(3))
    states = [zero, one, (zero + one) / np.sqrt(2), (zero - one) / np.sqrt(2), (zero + 1j * one) / np.sqrt(2), two]
    expected = {"Z": [1, -1, 0, 0, 0, 0], "X": [0, 0, 1, -1, 0, 0], "Y": [0, 0, 0, 0, 1, 0]}
    for name, values in expected.items():
        operator = chip.observable(q, name)
        actual = [chip.backend.expect(operator, state) for state in states]
        np.testing.assert_allclose(actual, values, atol=1e-10)
    record = chip.resolve().bases["q"]
    authored_ground = np.asarray(record.vectors @ chip.backend.to_array(zero).reshape(-1))
    np.testing.assert_allclose(authored_ground, record.energy_vectors[:, 0], atol=1e-10)


@pytest.mark.parametrize("name", ["sigma_x", "sigma_y", "sigma_z", "sigma_plus", "sigma_minus"])
def test_default_paulis_follow_parameter_changes_without_stale_caches(name):
    """A changed local Hamiltonian rotates the computational projectors immediately."""
    q = TiltedMode(levels=3, label="q")
    operator = getattr(q, name)
    before = operator.full()
    operator.data = (123.0 * operator).data
    np.testing.assert_allclose(getattr(q, name).full(), before, atol=1e-12)
    q.mixing = 0.8
    after = getattr(q, name).full()
    assert np.linalg.norm(after - before) > 0.1
    fresh = TiltedMode(mixing=0.8, levels=3, label="fresh")
    np.testing.assert_allclose(after, getattr(fresh, name).full(), atol=1e-12)


@pytest.mark.parametrize("basis", ["native", "eigen"])
@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
@pytest.mark.parametrize("mixed", [False, True])
def test_population_uses_captured_local_energy_states(basis, backend, mixed):
    """Population is a local-energy occupation and stays fixed after source changes."""
    q = TiltedMode(levels=3, label="q")
    chip = Chip([q], basis=basis, backend=backend)
    target = chip.bare_state(q=1)
    state = (
        0.75 * chip.backend.state_to_dm(target) + 0.25 * chip.backend.state_to_dm(chip.bare_state(q=0))
        if mixed else target
    )
    if backend == "dynamiqs":
        dq = pytest.importorskip("dynamiqs")
        options = {"method": dq.method.Tsit5(atol=1e-10, rtol=1e-10)}
    else:
        options = {"atol": 1e-10, "rtol": 1e-10}
    result = QuantumSequence(chip).simulate(
        tlist=np.linspace(0.0, 0.2, 5), initial_state=state, partition=False, options=options,
    )
    q.mixing = 0.8
    expected = 0.75 if mixed else 1.0
    population, overlap = result.population(q, 1), result.overlap(target)
    np.testing.assert_allclose(population, expected, atol=1e-8)
    np.testing.assert_allclose(result.population(q, 0), 1.0 - expected, atol=1e-8)
    np.testing.assert_allclose(overlap, expected, atol=1e-8)
    import jax
    expected_type = jax.Array if backend == "dynamiqs" else np.ndarray
    assert isinstance(population, expected_type)
    assert isinstance(overlap, expected_type)


def test_named_operator_keeps_the_component_definition():
    """A component override is never replaced merely because its name is familiar."""
    class WeightedNumber(TiltedMode):
        def number_operator(self):
            return 2 * super().number_operator()

    q = WeightedNumber(levels=3, label="q")
    chip = Chip([q], basis="eigen")
    np.testing.assert_allclose(
        chip.observable(q, "n").full(), chip.observable(q, q.number_operator()).full(), atol=1e-12,
    )


def test_named_physical_operator_respects_custom_lookup_and_flux_definition():
    """Physical aliases use the component hook, including distinct phase and flux channels."""
    from quchip import DuffingTransmon

    class CustomReadout(DuffingTransmon):
        def local_operator(self, name):
            return self.number_operator() if name == "charge" else super().local_operator(name)

    q = CustomReadout(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
    chip = Chip([q])
    np.testing.assert_allclose(chip.observable(q, "charge").full(), q.number_operator().full(), atol=1e-12)
    np.testing.assert_allclose(
        chip.observable(q, "flux").full(), chip.observable(q, q.flux_coupling_operator()).full(), atol=1e-12,
    )


@pytest.mark.parametrize("basis", ["native", "eigen"])
def test_energy_population_is_independent_of_integration_frame(basis):
    """Frame subtraction and band decomposition use the same isolated energy generator."""
    q = TiltedMode(levels=3, label="q")
    for frame in ("lab", "rotating"):
        chip = Chip([q], basis=basis, frame=frame)
        result = QuantumSequence(chip).simulate(
            tlist=np.linspace(0.0, 1.0, 21), initial_state=chip.bare_state(q=1),
            e_ops=chip.e_ops(q="Z"), partition=False, )
        np.testing.assert_allclose(result.population(q, 1), 1.0, atol=1e-9)
        np.testing.assert_allclose(result.expect(q), -1.0, atol=1e-9)


@pytest.mark.parametrize("warm", [False, True])
def test_pauli_cache_keeps_jit_gradients_and_backend_selection(warm):
    """Cached energy operators retain gradients, native types, and independent copies."""
    import jax
    from quchip.backend import _backend_context, _coerce_backend

    q = TiltedMode(levels=3, label="q")
    if warm:
        _ = q.sigma_z
    backend = _coerce_backend("dynamiqs")
    with _backend_context(backend), jax.checking_leaks():
        def read():
            return backend.to_array(q.sigma_z)[0, 0].real
        assert jax.jit(read)() == pytest.approx(float(read()))

        def value(mixing):
            changed = q.copy()
            changed.mixing = mixing
            return backend.to_array(changed.sigma_z)[0, 0].real

        expected = -4 * q.freq * q.mixing / (q.freq**2 + 4 * q.mixing**2 + 0.16)**1.5
        assert jax.jit(jax.grad(value))(q.mixing) == pytest.approx(expected, rel=1e-10)
    assert q.sigma_z.full().shape == (3, 3)
