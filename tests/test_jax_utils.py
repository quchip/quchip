"""Tests for JAX utilities — tracer detection in pytrees."""

from __future__ import annotations

import numpy as np

from quchip.utils.jax_utils import contains_tracer

import pytest


pytestmark = pytest.mark.unit


def test_returns_false_for_pure_python_scalars():
    """A tuple of plain Python scalars has no tracer leaf."""
    assert contains_tracer((1.0, 2.0, 3)) is False


def test_returns_false_for_numpy_arrays():
    """A tuple of concrete NumPy arrays has no tracer leaf."""
    assert contains_tracer((np.array([1.0, 2.0]), np.array(3.0))) is False


def test_returns_false_for_nested_dict_of_concrete_values():
    """A nested dict/list pytree of concrete values has no tracer leaf."""
    tree = {"a": 1.0, "b": [np.array([2.0]), {"c": 3.0}]}
    assert contains_tracer(tree) is False


def test_returns_false_for_none():
    """``None`` has no tracer leaf."""
    assert contains_tracer(None) is False


def test_returns_true_when_jit_traces():
    """A value traced inside ``jax.jit`` is detected as a tracer."""
    import jax

    result = {"flag": False}

    def probe(x):
        result["flag"] = contains_tracer(x)
        return x

    jax.jit(probe)(1.0).block_until_ready()
    assert result["flag"] is True


def test_returns_false_outside_jit_with_concrete_jnp_array():
    """A concrete ``jax.numpy`` array outside any transform has no tracer leaf."""
    import jax.numpy as jnp
    assert contains_tracer(jnp.array([1.0, 2.0])) is False


def test_outside_trace_computes_a_concrete_value_on_the_callers_default_device():
    """Inside jax.jit, outside_trace returns a concrete value that it computes on the caller's default device."""
    import jax
    import jax.numpy as jnp

    from quchip.utils.jax_utils import outside_trace

    cpu = jax.devices("cpu")[0]
    devices = []

    def work():
        devices.append(jax.config.jax_default_device)
        return jnp.arange(3.0).sum()

    def scaled(x):
        with jax.default_device(cpu):
            value = outside_trace(work)
        assert not contains_tracer(value)
        return x * value

    assert jax.jit(scaled)(2.0) == 6.0
    assert devices == [cpu]
