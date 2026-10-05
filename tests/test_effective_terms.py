"""Retained operators use the ordinary chip compilation and ownership paths."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quchip import RWA, Chip, DuffingTransmon, EffectiveTerms, CollapseChannel


def _devices():
    return [
        DuffingTransmon(freq=5 + i, anharmonicity=-0.2, levels=2, label=label)
        for i, label in enumerate(("a", "b", "c"))
    ]


def _terms(scale=0.1, *, with_channels=True):
    x = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    lowering = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    # Retain an actual three-body operator, including nonconserving bands.
    h = scale * jnp.kron(jnp.kron(x, x), x)
    jump = jnp.kron(lowering, jnp.eye(4)) + jnp.kron(jnp.eye(2), jnp.kron(lowering, jnp.eye(2)))
    return EffectiveTerms(
        ("a", "b", "c"),
        (2, 2, 2),
        h,
        channels=(CollapseChannel(jump, 0.03, "shared"),) if with_channels else (),
        label="retained",
    )


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_retained_terms_compile_copy_partition_and_roundtrip(backend):
    terms = _terms()
    chip = Chip(_devices(), backend=backend, effective_terms=[terms])
    expected = np.diag([0.0, 7.0, 6.0, 13.0, 5.0, 12.0, 11.0, 18.0]) + np.asarray(terms.hamiltonian)
    for candidate in (chip, chip.clone(), Chip.from_dict(json.loads(json.dumps(chip.to_dict())))):
        np.testing.assert_allclose(candidate.hamiltonian().matrix(backend=candidate.backend), expected, atol=1e-12)
        np.testing.assert_allclose(
            candidate.unresolved_hamiltonian().matrix(backend=candidate.backend), expected, atol=1e-12
        )
        jumps = candidate.backend._collapse_operators(candidate.resolve())
        assert len(jumps) == 1
        np.testing.assert_allclose(candidate.backend.to_array(jumps[0]), np.sqrt(0.03) * terms.channels[0].operator)
        assert len(candidate.partition()) == 1


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_retained_terms_transform_with_the_selected_frame(backend):
    chip = Chip(_devices(), backend=backend, effective_terms=[_terms(with_channels=False)])
    lab = np.asarray(chip.hamiltonian().matrix())
    frames = {"a": 1.0, "b": 2.0, "c": 3.0}
    generator = np.diag([0.0, 3.0, 2.0, 5.0, 1.0, 4.0, 3.0, 6.0])
    t = 0.17
    rotation = np.diag(np.exp(2j * np.pi * t * np.diag(generator)))
    expected = rotation @ lab @ rotation.conj().T - generator
    actual = chip.resolve(frame=frames).hamiltonian().matrix(t=t)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_retained_terms_are_captured_and_have_live_jax_leaves():
    matrix = np.diag([0.0, 0.1])
    term = EffectiveTerms(("a",), (2,), matrix, label="shift")
    chip = Chip([_devices()[0]], effective_terms=[term])
    captured = chip.resolve()
    matrix[1, 1] = 9.0
    np.testing.assert_allclose(chip.hamiltonian().matrix(), np.diag([0.0, 5.1]))
    chip["a"].freq = 6.0
    np.testing.assert_allclose(captured.hamiltonian().matrix(), np.diag([0.0, 5.1]))
    np.testing.assert_allclose(chip.hamiltonian().matrix(), np.diag([0.0, 6.1]))

    def value(scale):
        model = Chip(_devices(), backend="dynamiqs", effective_terms=[_terms(scale)])
        return jnp.real(model.resolve(frame="lab").hamiltonian().matrix(backend=model.backend)[0, 7])

    assert jax.jit(value)(0.17) == pytest.approx(0.17)
    assert jax.jit(jax.grad(value))(0.17) == pytest.approx(1.0)


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_collective_effective_channel_requires_a_compatible_frame(backend):
    chip = Chip(_devices(), backend=backend, effective_terms=[_terms()])
    lab = chip.resolve(frame="lab")
    common = chip.resolve(frame=5.0)
    lab_jump = chip.backend._collapse_operators(lab)[0]
    common_jump = chip.backend._collapse_operators(common)[0]
    np.testing.assert_allclose(chip.backend.to_array(common_jump), chip.backend.to_array(lab_jump))
    with pytest.raises(ValueError, match="Effective channel.*different phases"):
        chip.resolve(frame={"a": 5.0, "b": 6.0, "c": 7.0})


@pytest.mark.parametrize("matrix", [np.diag([0.0, np.nan]), np.array([[0.0, 1.0], [0.0, 0.0]])])
def test_effective_hamiltonian_rejects_nonphysical_concrete_values(matrix):
    with pytest.raises(ValueError, match="finite and Hermitian"):
        EffectiveTerms(("a",), (2,), matrix)


def test_effective_terms_validate_loaded_shapes_and_graph_membership():
    terms = EffectiveTerms(("a",), (2,), np.diag([0.0, 0.1]))
    with pytest.raises(ValueError, match="unknown devices"):
        Chip([_devices()[1]], effective_terms=[terms])
    data = terms.to_dict()
    data["dims"] = [3]
    with pytest.raises(ValueError, match="shape"):
        EffectiveTerms.from_dict(data)
    channel = CollapseChannel(np.eye(2), float("nan"), "bad")
    with pytest.raises(ValueError, match="must be finite"):
        EffectiveTerms(("a",), (2,), np.zeros((2, 2)), (channel,))


def test_effective_terms_notes_follow_the_derived_notes_and_survive_serialization():
    """Producer notes come last in physics_notes() and round-trip; payloads without notes still load."""
    source = _terms()
    terms = EffectiveTerms(source.labels, source.dims, source.hamiltonian, source.channels, "retained",
                           notes=("Second-order correction.",))
    chip = Chip(_devices(), effective_terms=[terms])
    loaded = Chip.from_dict(json.loads(json.dumps(chip.to_dict())))
    data = terms.to_dict()
    del data["notes"]

    assert terms.physics_notes()[-1] == "Second-order correction."
    assert loaded.effective_terms[0].notes == ("Second-order correction.",)
    assert "notes" not in _terms().to_dict()
    assert EffectiveTerms.from_dict(data).notes == ()
    with pytest.raises(ValueError, match="Invalid serialized EffectiveTerms fields"):
        EffectiveTerms.from_dict({**data, "comment": "unknown"})
    with pytest.raises(ValueError, match="nonempty strings"):
        EffectiveTerms(("a",), (2,), np.zeros((2, 2)), notes=("",))


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_common_frame_keeps_a_band_with_cancelling_level_weights_static(backend):
    """Level changes that cancel in a common frame give a static term, not a round-off carrier."""
    devices = [DuffingTransmon(freq=5.1 + 0.1 * i, anharmonicity=-0.2, levels=4, label=label)
               for i, label in enumerate(("a", "b", "c"))]
    dims = (4, 4, 4)
    # Column minus row level changes (3, -1, -2): summed one device at a time,
    # 3*5.2 - 5.2 - 2*5.2 leaves 1.8e-15 GHz in floating point.
    row, column = np.ravel_multi_index((0, 1, 2), dims), np.ravel_multi_index((3, 0, 0), dims)
    hamiltonian = np.zeros((64, 64))
    hamiltonian[row, column] = hamiltonian[column, row] = 0.01
    chip = Chip(devices, frame=5.2, approximation=RWA(), backend=backend,
                effective_terms=[EffectiveTerms(("a", "b", "c"), dims, hamiltonian, label="retained")])

    resolved = chip.resolve()

    assert not resolved.slh.H.dynamic_terms
    static = np.asarray(resolved.hamiltonian().matrix(backend=chip.backend))
    np.testing.assert_allclose(static[row, column], 0.01, atol=1e-15)
