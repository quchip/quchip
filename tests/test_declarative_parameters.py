from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

import quchip
from quchip import DeviceModel, Scalar, parameter
from quchip.declarative import Parameter, Setting, setting
from quchip.declarative.parameters import (
    UNBOUND,
    parameter_fields,
    setting_fields,
    validate_declared_fields,
)


class DeclaredFields:
    rate: float = parameter(
        default=None,
        nonnegative=True,
        unit="1/ns",
        noise=True,
    )
    basis: str | None = setting(default=None)


class ToyDevice(DeviceModel):
    freq: Scalar = parameter(default=UNBOUND, positive=True)
    detuning: Scalar = parameter(default=0.0)
    approximation = None

    def local_hamiltonian(self, op, p):
        return (p.freq + p.detuning) * op.n


def test_parameter_records_noise_ownership():
    field = parameter_fields(DeclaredFields)["rate"]

    assert isinstance(field, Parameter)
    assert field.noise is True


def test_setting_is_structural_not_a_parameter():
    assert setting_fields(DeclaredFields) == {
        "basis": Setting(default=None, serialize=True),
    }
    assert "basis" not in parameter_fields(DeclaredFields)


def test_setting_authoring_names_are_public():
    assert quchip.Setting is Setting
    assert quchip.setting is setting


def test_noise_parameter_must_be_serializable():
    class InvalidNoiseField:
        rate: float = parameter(default=None, noise=True, serialize=False)

    with pytest.raises(TypeError, match="Noise parameter 'rate' must be serializable"):
        validate_declared_fields(InvalidNoiseField)


def test_bare_parameter_is_required_at_runtime() -> None:
    class RequiredDevice(DeviceModel):
        value: Scalar = parameter()

        def local_hamiltonian(self, op, p):
            return p.value * op.n

    assert inspect.signature(RequiredDevice).parameters["value"].default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="value"):
        RequiredDevice()


def test_keyword_only_parameter_matches_runtime_signature() -> None:
    class KeywordDevice(DeviceModel):
        value: Scalar = parameter(default=1.0, kw_only=True)

        def local_hamiltonian(self, op, p):
            return p.value * op.n

    assert inspect.signature(KeywordDevice).parameters["value"].kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError, match="positional"):
        KeywordDevice(2.0)


def test_settings_cannot_opt_out_of_keyword_only_construction() -> None:
    with pytest.raises(ValueError, match="keyword-only"):
        setting(default=None, kw_only=False)


def test_parameter_fields_generate_constructor_and_attributes():
    """Declared parameter fields become constructor kwargs and attributes; unset fields take their declared defaults."""
    dev = ToyDevice(freq=5.0, levels=3, label="q")
    assert dev.freq == 5.0
    assert dev.detuning == 0.0
    assert dev.levels == 3
    assert dev.label == "q"


def test_unbound_construction_keeps_symbolic_hamiltonian_and_accepts_values_at_materialization():
    dev = ToyDevice(levels=3, label="q")
    assert dev.freq is UNBOUND
    hamiltonian = dev.unresolved_hamiltonian()
    assert hamiltonian.parameter_paths() == ("q.freq", "q.detuning")
    matrix = hamiltonian.matrix({"q.freq": 5.0})
    assert matrix.shape == (3, 3)


def test_positive_parameter_validation_uses_concrete_only_path():
    """A parameter declared positive=True rejects a concrete negative value at construction, raising ValueError."""
    with pytest.raises(ValueError, match="freq must be positive"):
        ToyDevice(freq=-1.0)


def test_traced_positive_parameter_does_not_concretize():
    """Positive-parameter validation skips concretization for traced values, staying differentiable via jax.grad."""
    def build(x):
        dev = ToyDevice(freq=x)
        return dev.freq * 2.0

    assert jax.grad(build)(jnp.asarray(5.0)) == 2.0


def test_parameter_pytree_leaves_include_declared_fields():
    """Flattening a DeviceModel pytree yields each declared parameter as its own leaf."""
    dev = ToyDevice(freq=jnp.asarray(5.0), detuning=jnp.asarray(0.1))
    leaves, _ = jtu.tree_flatten(dev)
    assert any(leaf is dev.freq for leaf in leaves)
    assert any(leaf is dev.detuning for leaf in leaves)


def test_parameter_pytree_roundtrip_preserves_base_state():
    """DeviceModel pytree round-trip preserves declared parameter values and static base state (label, levels)."""
    dev = ToyDevice(freq=jnp.asarray(5.0), detuning=jnp.asarray(0.1), levels=4, label="q")
    leaves, treedef = jtu.tree_flatten(dev)
    restored = jtu.tree_unflatten(treedef, leaves)
    assert restored.label == "q"
    assert restored.levels == 4
    assert float(restored.freq) == 5.0
    assert float(restored.detuning) == pytest.approx(0.1)


def test_parameter_pytree_roundtrip_preserves_noise_params():
    """DeviceModel pytree round-trip preserves T1 and T2 noise parameters."""
    dev = ToyDevice(freq=5.0, levels=3, T1=100.0, T2=80.0)
    leaves, treedef = jtu.tree_flatten(dev)
    restored = jtu.tree_unflatten(treedef, leaves)
    assert restored.T1 == 100.0
    assert restored.T2 == 80.0


def test_parameter_pytree_traceable_through_jit():
    """A DeviceModel instance passes through jax.jit as an argument, with its declared parameters traced correctly."""
    @jax.jit
    def freq_doubled(dev):
        return dev.freq * 2.0

    dev = ToyDevice(freq=jnp.asarray(5.0), levels=3, label="q")
    assert float(freq_doubled(dev)) == 10.0
