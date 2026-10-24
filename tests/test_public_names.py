"""Checks for the bath occupation parameter."""

import jax
import pytest

from quchip import Chip, Resonator


def test_thermal_occupation_remains_differentiable_through_parameter_binding():
    """Bath occupation changes the downward rate with the analytical derivative 1/T1."""
    chip = Chip([Resonator(freq=6, T1=100, thermal_occupation=.1, label="r")])
    derivative = jax.jit(jax.grad(
        lambda n: chip.with_params({"r.thermal_occupation": n})["r"].intrinsic_decay_rate()
    ))(.2)
    assert derivative == pytest.approx(.01)
