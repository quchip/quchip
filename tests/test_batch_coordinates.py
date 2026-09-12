"""A stacked time axis must represent the same saved coordinates at every point."""

import numpy as np
import pytest

from quchip import Chip, DuffingTransmon, QuantumSequence
from quchip.engine import solve_many


def _batch(backend, times):
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, T1=10.0, label="q")
    chip = Chip([q], backend=backend, frame="rotating")
    sequence = QuantumSequence(chip)
    return solve_many([sequence.build_problem(grid, initial_state={"q": 1},
                       e_ops={"q": q.number_operator()}) for grid in times],
                      progress=False)


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
@pytest.mark.parametrize("second", [[0.0, 0.5, 2.0], [0.0, 1.0, 2.0, 3.0]])
def test_batch_trace_stacking_rejects_different_time_coordinates(backend, second):
    batch = _batch(backend, [[0.0, 1.0, 2.0], second])
    for getter in (lambda: batch.expect("q"), lambda: batch.population("q", 1), lambda: batch.times,
                   lambda: batch.jump_rate("q.thermal_emission"),
                   lambda: batch.collapse_integral("q.thermal_emission")):
        with pytest.raises(ValueError, match="different time grids"):
            getter()
    np.testing.assert_array_equal(batch[1].times, second)
    for reduction in ("last", "max", "mean"):
        expected = [np.asarray(point.expect("q"))[-1] if reduction == "last"
                    else getattr(np, reduction)(np.asarray(point.expect("q"))) for point in batch]
        np.testing.assert_allclose(batch.expect("q", reduce=reduction), expected)


def test_jit_batch_time_alignment_is_checked_when_values_are_stacked():
    import jax
    import jax.numpy as jnp

    @jax.jit
    def traces(last):
        batch = _batch("dynamiqs", [jnp.asarray([0.0, 1.0, 2.0]), jnp.asarray([0.0, 1.0, last])])
        return batch.expect("q")

    assert traces(jnp.asarray(2.0)).shape == (2, 3)
    with pytest.raises(Exception, match="different time grids"):
        traces(jnp.asarray(3.0)).block_until_ready()


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_output_field_stacking_rejects_different_time_coordinates(backend):
    from quchip import PortNetwork, Resonator

    mode = Resonator(freq=0.2, levels=2, label="r")
    network = PortNetwork(label="line")
    network.port("coupler", target=mode, rate=0.01)
    plane = network.external_port("coupler")
    chip = Chip([mode], port_network=network, backend=backend, frame="rotating")
    sequence = QuantumSequence(chip)
    grids = [[0.0, 1.0, 3.0], [0.0, 2.0, 3.0]]
    batch = solve_many([sequence.build_problem(grid, initial_state={"r": 1}, e_ops={plane: plane.output})
                       for grid in grids], progress=False)
    with pytest.raises(ValueError, match="different time grids"):
        batch.output(plane)
    for point, times in zip(batch, grids):
        np.testing.assert_array_equal(point.output(plane).times, times)
        np.testing.assert_allclose(point.output(plane).photon_flux, 0.01 * np.exp(-0.01 * np.asarray(times)), atol=1e-8)
