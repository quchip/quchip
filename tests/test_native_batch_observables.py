"""Point-specific observables must not require temporary state histories."""

from dataclasses import replace

import numpy as np
import pytest

from quchip import Chip, DuffingTransmon, QuantumSequence
from quchip.engine import solve_batch
from quchip.engine.ir import SolveBatch


def _batch(states, noisy=True, scales=(1.0, 2.0)):
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, T1=10.0 if noisy else None, label="q")
    chip = Chip([q], backend="dynamiqs", frame="rotating")
    problem = QuantumSequence(chip).build_problem(np.linspace(0.0, 3.0, 41),
        initial_state={"q": 1}, e_ops={"q": q.number_operator()}, states=states)
    problems = tuple(replace(problem, e_ops=[scale * op for op in problem.e_ops]) for scale in scales)
    return SolveBatch(chip=chip, problems=problems)


@pytest.mark.parametrize("states", ["all", "final", "none"])
@pytest.mark.parametrize("noisy", [False, True])
def test_point_observables_obey_native_storage_choice(states, noisy):
    result = solve_batch(_batch(states, noisy), progress=False)
    for scale, point in zip((1.0, 2.0), result):
        expected = scale * np.exp(-np.asarray(point.times) / 10) if noisy else scale
        np.testing.assert_allclose(point.expect("q"), expected, atol=3e-6)
        assert bool(point.stats["options"]["save_states"]) == (states == "all")
        if states != "all":
            with pytest.raises(RuntimeError, match="Full state history"):
                _ = point.states
        if states != "none":
            assert point.final_state is not None


def test_point_observables_keep_native_jit_gradients_without_histories():
    import jax
    import jax.numpy as jnp

    @jax.jit
    @jax.value_and_grad
    def objective(scale):
        batch = _batch("none", scales=(scale, 2 * scale))
        result = solve_batch(batch, progress=False)
        return jnp.real(result.expect("q", reduce="last").sum())

    value, derivative = objective(jnp.asarray(0.7))
    assert value == pytest.approx(2.1 * np.exp(-0.3), abs=3e-6)
    assert derivative == pytest.approx(3 * np.exp(-0.3), abs=3e-6)


def test_point_observables_follow_dynamic_pulse_axes_and_gradients():
    import jax
    import jax.numpy as jnp
    from quchip import ChargeDrive, Square

    @jax.jit
    @jax.value_and_grad
    def objective(amplitude):
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
        chip = Chip([q], backend="dynamiqs", frame="rotating")
        drive = ChargeDrive(q)
        chip.wire(drive)
        sequence = QuantumSequence(chip)
        pulse = sequence.schedule(drive, envelope=Square(duration=0.01, amplitude=amplitude),
                                  freq=5.0, start_time=1.0)
        batch = sequence.build_batch(pulse.vary("duration", [0.01, 0.03]),
            tlist=[0.0, 2.0], states="none", e_ops={"q": q.number_operator()})
        batch = replace(batch, problems=tuple(replace(problem, e_ops=[scale * op for op in problem.e_ops])
                         for scale, problem in zip((1.0, 2.0), batch.problems)))
        result = solve_batch(batch, progress=False)
        return jnp.real(result.expect("q", reduce="last").sum())

    value, derivative = objective(jnp.asarray(1.0))
    widths, scales = np.asarray([0.01, 0.03]), np.asarray([1.0, 2.0])
    assert value == pytest.approx(np.sum(scales * np.sin(np.pi * widths) ** 2), abs=2e-6)
    assert derivative == pytest.approx(np.sum(scales * np.pi * widths * np.sin(2 * np.pi * widths)), abs=2e-6)
