from __future__ import annotations

import pathlib
import re

import jax
import jax.numpy as jnp

from quchip import EnvelopeShape, Scalar, qnp, parameter


class TraceableEnvelope(EnvelopeShape):
    duration: Scalar = parameter(positive=True)
    amplitude: Scalar = parameter(default=1.0)

    def value(self, t):
        return self.amplitude * qnp.sin(qnp.pi * t / self.duration)


def test_envelope_value_is_jax_differentiable():
    """A gradient flows through EnvelopeShape's amplitude parameter when built inside a jax.grad-traced function."""

    def objective(amplitude):
        env = TraceableEnvelope(duration=10.0, amplitude=amplitude)
        return env.value(jnp.asarray(5.0))

    assert jax.grad(objective)(jnp.asarray(2.0)) == 1.0


def test_declarative_package_has_no_numpy_imports():
    """No file under quchip/declarative imports NumPy or contains a bare np, keeping its array ops JAX-traceable."""
    root = pathlib.Path(__file__).resolve().parent.parent / "quchip" / "declarative"
    offenders = []
    pattern = re.compile(r"(?:^|[^q])\bnp\b")
    for path in root.rglob("*.py"):
        text = path.read_text()
        if "import numpy" in text or "from numpy" in text or pattern.search(text):
            offenders.append(str(path))
    assert offenders == []
