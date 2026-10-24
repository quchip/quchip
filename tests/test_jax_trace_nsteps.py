"""Regression tests locking JIT + grad through the full simulate / simulate_batch paths."""

from __future__ import annotations

from quchip.approximations import RWA

import pytest


@pytest.mark.optional_backend
def test_jit_and_grad_through_simulate_batch():
    """seq.simulate_batch() stays JIT-able because the operator fingerprint avoids concretizing operator values."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    from quchip import (
        ChargeDrive,
        Chip,
        DuffingTransmon,
        Gaussian,
        QuantumSequence,
        set_default_backend,
    )

    set_default_backend("dynamiqs")

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3)
    drive = ChargeDrive(target=q)
    chip = Chip(devices=[q], frame="rotating", approximation=RWA())
    chip.wire(drive)

    psi0 = jnp.zeros((3, 1), dtype=jnp.complex128).at[0, 0].set(1.0)
    tlist = jnp.linspace(0.0, 40.0, 60)

    @jax.jit
    def loss(amp):
        seq = QuantumSequence(chip)
        h = seq.schedule(drive, envelope=Gaussian(duration=40.0, amplitude=amp, sigmas=4.0), freq=5.0)
        h.vary("amplitude", jnp.array([amp * 0.9, amp, amp * 1.1]))
        batch = seq.simulate_batch(initial_state=psi0, tlist=tlist)
        return jnp.sum(batch.population(q, level=1, reduce="last"))

    loss_val = loss(jnp.array(0.04))
    grad_val = jax.grad(loss)(jnp.array(0.04))
    assert float(loss_val) > 0.0
    assert jnp.isfinite(grad_val), f"gradient must be finite, got {grad_val}"


@pytest.mark.validation
@pytest.mark.optional_backend
def test_jit_through_chip_analysis_methods():
    """chip.freq / chip.dressed_anharmonicity / chip.static_zz stay JIT-able as array-only kernels."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    from quchip import (
        Capacitive,
        Chip,
        DuffingTransmon,
        set_default_backend,
    )

    set_default_backend("dynamiqs")

    @jax.jit
    def loss(params):
        freq_q2, g = params
        q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.30, label="q1")
        q2 = DuffingTransmon(freq=freq_q2, anharmonicity=-0.30, label="q2")
        chip = Chip([q1, q2], [Capacitive("q1", "q2", g=g)])
        return chip.freq("q2") ** 2 + chip.static_zz("q1", "q2") ** 2 + chip.dressed_anharmonicity("q1") ** 2

    val, grad = jax.value_and_grad(loss)(jnp.array([5.1, 0.005]))
    assert jnp.isfinite(val) and jnp.isfinite(grad).all()


@pytest.mark.optional_backend
def test_dynamiqs_default_max_steps_skips_traced_tlist_span():
    """The solver-step heuristic must not concretize a JIT-traced pulse duration."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    from quchip.backend import default_solver_steps

    @jax.jit
    def traced_duration_pass_through(duration):
        tlist = jnp.linspace(0.0, duration, 11)
        assert default_solver_steps({"spectral_bound_ghz": 20.0}, tlist) is None
        return duration

    assert float(traced_duration_pass_through(jnp.array(12.0))) == 12.0
