"""Result-time queries select retained data without another solve."""

import numpy as np
import pytest

from quchip import Chip, DuffingTransmon, QuantumSequence


def _result(backend, states="none"):
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, T1=10.0, label="q")
    sequence = QuantumSequence(Chip([q], backend=backend, frame="rotating"))
    return sequence.simulate(tlist=[2.0, 3.0, 5.0], initial_state={"q": 1},
                             states=states, e_ops={"q": q.number_operator()})


def test_complex_and_multiple_observable_series_preserve_native_shapes():
    result = _result("qutip")
    xp = result._backend.array_module
    values = xp.asarray([[1+2j, 3+4j, 7+8j], [2-1j, 4-3j, 8-7j]])
    actual = result.observable_at([[2.5, 4.0]], values, method="interpolate")
    np.testing.assert_allclose(actual, [[[2+3j, 5+6j]], [[3-2j, 6-5j]]])
    assert type(actual).__module__.startswith("numpy")
    assert result.observable_at([], values).shape == (2, 0)


@pytest.mark.parametrize("method,query", [("nearest", 1.9), ("interpolate", 5.1),
                                          ("exact", np.nan), ("exact", [3.0, 6.0])])
def test_observable_queries_reject_invalid_coordinates(method, query):
    result = _result("qutip")
    with pytest.raises(ValueError, match="interval"):
        result.observable_at(query, result.expect("q"), method=method)


def test_observable_queries_require_a_matching_time_axis_and_method():
    result = _result("qutip")
    for values in (1.0, [1.0, 2.0], np.ones((3, 2))):
        with pytest.raises(ValueError, match="last axis"):
            result.observable_at(3.0, values)
    with pytest.raises(ValueError, match="method"):
        result.observable_at(3.0, result.expect("q"), method="cubic")
    with pytest.raises(ValueError, match="real"):
        result.observable_at(3.0 + 1j, result.expect("q"))


def test_observable_interpolation_preserves_jit_value_and_time_gradients():
    import jax
    import jax.numpy as jnp

    result = _result("dynamiqs")

    @jax.jit
    @jax.value_and_grad
    def lookup(time):
        return result.observable_at(time, jnp.asarray([2.0, 4.0, 10.0]), method="interpolate")

    value, derivative = lookup(jnp.asarray(4.0))
    assert value == pytest.approx(7.0)
    assert derivative == pytest.approx(3.0)
    gradient = jax.jit(jax.grad(lambda values: result.observable_at(4.0, values, method="interpolate")))(
        jnp.asarray([2.0, 4.0, 10.0]))
    np.testing.assert_allclose(gradient, [0.0, 0.5, 0.5])
    with pytest.raises(Exception, match="interval"):
        lookup(jnp.asarray(6.0))[0].block_until_ready()


def test_partitioned_observable_lookup_uses_the_shared_saved_grid():
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="a")
    q1 = DuffingTransmon(freq=6.0, anharmonicity=-0.2, levels=2, label="b")
    chip = Chip([q0, q1], frame="rotating")
    result = QuantumSequence(chip).simulate(tlist=[0.0, 1.0, 3.0],
        e_ops={"a": q0.number_operator()}, states="none")
    np.testing.assert_array_equal(result.observable_at([0.0, 3.0], result.expect("a")), [0.0, 0.0])


@pytest.mark.parametrize("states", ["all", "final"])
def test_native_state_lookup_retains_jit_validation(states):
    import jax
    import jax.numpy as jnp

    result = _result("dynamiqs", states)

    @jax.jit
    def lookup(time):
        return result._backend.to_array(result.state_at(time))

    np.testing.assert_allclose(lookup(jnp.asarray(5.0)), result._backend.to_array(result.final_state))
    if states == "all":
        np.testing.assert_allclose(lookup(jnp.asarray(3.0)), result._backend.to_array(result.states[1]))
    else:
        with pytest.raises(Exception, match="Full state history"):
            lookup(jnp.asarray(3.0)).block_until_ready()
    with pytest.raises(Exception, match="saved time"):
        lookup(jnp.asarray(4.0)).block_until_ready()


def test_partitioned_state_lookup_preserves_mixed_state_kind_and_order():
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, T1=10.0, label="a")
    q1 = DuffingTransmon(freq=6.0, anharmonicity=-0.2, levels=2, label="b")
    chip = Chip([q0, q1], frame="rotating")
    sequence = QuantumSequence(chip)
    options = dict(tlist=[0.0, 1.0, 3.0], initial_state={"a": 1, "b": 1})
    result = sequence.simulate(**options)
    joint = sequence.simulate(**options, partition=False)
    with pytest.warns(UserWarning, match="joint state"):
        selected = result.state_at(1.4, method="nearest")
    np.testing.assert_allclose(chip.backend.to_array(selected), chip.backend.to_array(joint.state_at(1.0)), atol=2e-6)
    with pytest.raises(ValueError, match="State lookup method"):
        result.state_at(1.0, method="interpolate")


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
@pytest.mark.parametrize("dtype,values,expected", [(np.uint8, [10, 0, 20], 5.0),
                                                  (np.int8, [120, -120, 0], 0.0)])
def test_interpolation_promotes_integer_values_before_arithmetic(backend, dtype, values, expected):
    result = _result(backend)
    assert result.observable_at(2.5, np.asarray(values, dtype=dtype), method="interpolate") == expected
