"""Durable observable-builder and decomposition behavior."""

from __future__ import annotations

import numpy as np
import pytest

from quchip.backend import get_default_backend
from quchip.chip.chip import Chip
from quchip.devices.resonator import Resonator
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.engine.observables import decompose_eops


@pytest.fixture()
def chip_qr():
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=6.2, levels=5, label="r")
    return Chip([q, r]), q, r


def test_unknown_operator_is_rejected_during_resolution(chip_qr):
    chip, _, _ = chip_qr
    with pytest.raises(ValueError, match="Unknown operator"):
        decompose_eops(chip.e_ops(q="bogus"), chip, get_default_backend())


@pytest.mark.parametrize("backend", ["qutip", pytest.param("dynamiqs", marks=pytest.mark.validation)])
def test_zero_observables_preserve_authored_coordinates(backend):
    from quchip import QuantumSequence, solve_many

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
    r = Resonator(freq=6.0, levels=2, label="r")
    chip = Chip([q, r], backend=backend, frame="rotating")
    n = q.number_operator()
    sequence = QuantumSequence(chip)
    problems = [sequence.build_problem(
        [0.0, 0.1], initial_state={"q": 1, "r": 1}, states="none",
        e_ops={"q": [scale * n, n, 0 * n], "r": 0 * r.number_operator(),
               ("q", "r"): (n, scale * r.number_operator())},
    ) for scale in (0.0, 2.0)]
    result = solve_many(problems, progress=False)
    np.testing.assert_allclose(result.expect("q", index=0), [[0, 0], [2, 2]], atol=1e-8)
    np.testing.assert_allclose(result.expect("q", index=1), 1, atol=1e-8)
    np.testing.assert_allclose(result.expect("q", index=2), 0, atol=1e-8)
    np.testing.assert_allclose(result.expect("r"), 0, atol=1e-8)
    np.testing.assert_allclose(result.expect(("q", "r")), [[0, 0], [2, 2]], atol=1e-8)


@pytest.mark.validation
def test_observable_list_coordinates_and_gradient_at_zero():
    import jax
    import jax.numpy as jnp

    from quchip import QuantumSequence

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
    chip = Chip([q], backend="dynamiqs", frame="rotating")
    n = jnp.asarray(q.local_space().matrix("n"))

    @jax.jit
    @jax.value_and_grad
    def first_observable(scale):
        result = QuantumSequence(chip).simulate(
            [0.0, 0.1], initial_state={"q": 1}, states="none",
            e_ops={"q": [scale * n, n]}, )
        return jnp.real(result.expect("q", index=0)[-1])

    value, derivative = first_observable(jnp.asarray(0.0))
    assert value == pytest.approx(0.0, abs=1e-10)
    assert derivative == pytest.approx(1.0, abs=1e-10)
