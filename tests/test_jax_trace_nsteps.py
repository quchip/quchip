"""Regression tests locking JIT + grad through the full simulate / simulate_batch paths."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_labels():
    from quchip.utils.labeling import reset_label_counters
    reset_label_counters()
    yield
    reset_label_counters()


@pytest.mark.optional_backend
def test_jit_and_grad_through_sesolve():
    """@jax.jit + jax.grad must work through a full seq.simulate() loss."""
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
    chip = Chip(devices=[q], frame="rotating", rwa=True)
    chip.wire(drive)

    drive_freq = float(chip.freq(q))
    duration = 40.0
    tlist = jnp.linspace(0.0, duration, 100)

    def loss_fn(params):
        amp = params[0]
        envelope = Gaussian(duration=duration, amplitude=amp, sigmas=4.0)
        seq = QuantumSequence(chip)
        seq.schedule(drive, envelope=envelope, freq=drive_freq)
        result = seq.simulate(tlist=tlist, initial_state=chip.state({q: 0}), e_ops={q: q.number_operator()})
        final_n = jnp.real(result.expect_values(q)[-1])
        return (1.0 - final_n) ** 2

    params0 = jnp.array([0.04])

    # Must not raise TracerArrayConversionError.
    loss_jit = jax.jit(loss_fn)
    loss_val = loss_jit(params0)
    assert float(loss_val) >= 0.0, "loss must be non-negative"

    grad_val = jax.grad(loss_fn)(params0)
    assert jnp.isfinite(grad_val).all(), f"gradient must be finite, got {grad_val}"


@pytest.mark.optional_backend
def test_jit_and_grad_through_driven_mesolve_matches_finite_difference():
    """A driven dissipative two-level transmon stays differentiable through ``mesolve``."""
    pytest.importorskip("dynamiqs")
    import dynamiqs as dq
    import jax
    import jax.numpy as jnp
    import numpy as np

    from quchip import (
        ChargeDrive,
        Chip,
        DuffingTransmon,
        Gaussian,
        QuantumSequence,
        set_default_backend,
    )

    set_default_backend("dynamiqs")

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, T1=200.0, label="q")
    drive = ChargeDrive(target=q, label="drive")
    chip = Chip([q], frame="rotating", rwa=True)
    chip.wire(drive)
    tlist = jnp.linspace(0.0, 40.0, 61)
    method = dq.method.Tsit5(rtol=1e-9, atol=1e-11)

    def solve(amplitude):
        sequence = QuantumSequence(chip)
        sequence.schedule(
            drive,
            envelope=Gaussian(duration=40.0, amplitude=amplitude, sigmas=4.0),
            freq=5.0,
        )
        return sequence.simulate(
            tlist=tlist,
            initial_state=chip.bare_state({q: 0}),
            e_ops=chip.e_ops(q="n"),
            options={"store_states": False, "store_final_state": False, "method": method},
            check_truncation=False,
            partition=False,
        )

    def final_excited_population(amplitude):
        return jnp.real(solve(amplitude).expect_final(q))

    amplitude = jnp.array(0.02)
    probe = solve(amplitude)
    assert probe.solver == "mesolve"

    population, autodiff_gradient = jax.jit(jax.value_and_grad(final_excited_population))(amplitude)
    step = 1e-5
    finite_difference = (
        final_excited_population(amplitude + step) - final_excited_population(amplitude - step)
    ) / (2 * step)

    assert 0.0 <= float(population) <= 1.0
    assert np.isfinite(float(autodiff_gradient))
    assert not np.isclose(float(autodiff_gradient), 0.0, atol=1e-8)
    np.testing.assert_allclose(
        float(autodiff_gradient),
        float(finite_difference),
        rtol=2e-3,
        atol=2e-4,
    )


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
    chip = Chip(devices=[q], frame="rotating", rwa=True)
    chip.wire(drive)

    psi0 = jnp.zeros((3, 1), dtype=jnp.complex128).at[0, 0].set(1.0)
    tlist = jnp.linspace(0.0, 40.0, 60)

    @jax.jit
    def loss(amp):
        seq = QuantumSequence(chip)
        h = seq.schedule(drive, envelope=Gaussian(duration=40.0, amplitude=amp, sigmas=4.0),
                         freq=5.0)
        h.vary("amplitude", jnp.array([amp * 0.9, amp, amp * 1.1]))
        batch = seq.simulate_batch(initial_state=psi0, tlist=tlist)
        return jnp.sum(batch.population(q, level=1, reduce="last"))

    loss_val = loss(jnp.array(0.04))
    grad_val = jax.grad(loss)(jnp.array(0.04))
    assert float(loss_val) > 0.0
    assert jnp.isfinite(grad_val), f"gradient must be finite, got {grad_val}"


@pytest.mark.optional_backend
def test_jit_and_grad_through_overlap_population_wrappers():
    """``result.population`` and ``result.overlap`` fall back to backend-native arrays under JIT."""
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
    chip = Chip(devices=[q], frame="rotating", rwa=True)
    chip.wire(drive)

    psi0 = jnp.zeros((3, 1), dtype=jnp.complex128).at[0, 0].set(1.0)
    target_ket = jnp.zeros((3, 1), dtype=jnp.complex128).at[1, 0].set(1.0)
    tlist = jnp.linspace(0.0, 40.0, 60)

    @jax.jit
    def loss_population(amp):
        seq = QuantumSequence(chip)
        seq.schedule(drive, envelope=Gaussian(duration=40.0, amplitude=amp, sigmas=4.0),
                     freq=5.0)
        result = seq.simulate(initial_state=psi0, tlist=tlist)
        return result.population(q, level=1)[-1]

    @jax.jit
    def loss_overlap(amp):
        seq = QuantumSequence(chip)
        seq.schedule(drive, envelope=Gaussian(duration=40.0, amplitude=amp, sigmas=4.0),
                     freq=5.0)
        result = seq.simulate(initial_state=psi0, tlist=tlist)
        return result.overlap(target_ket)[-1]

    for fn in (loss_population, loss_overlap):
        val = fn(jnp.array(0.04))
        grad = jax.grad(fn)(jnp.array(0.04))
        assert jnp.isfinite(val) and jnp.isfinite(grad)


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
        chip = Chip([q1, q2], [Capacitive("q1", "q2", g=g)], rwa=True)
        return chip.freq("q2") ** 2 + chip.static_zz("q1", "q2") ** 2 \
            + chip.dressed_anharmonicity("q1") ** 2

    val = loss(jnp.array([5.1, 0.005]))
    grad = jax.grad(loss)(jnp.array([5.1, 0.005]))
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


@pytest.mark.optional_backend
def test_jit_through_mesolve_with_collapse_op():
    """``simulate``/``seq.simulate`` must JIT-trace through Lindblad mesolve."""
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

    class _LossyChargeDrive(ChargeDrive):
        def collapse_operators(self, device):
            return [0.05 * device.lowering_operator()]

    @jax.jit
    def loss(amp):
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        drv = _LossyChargeDrive(target=q, label="d")
        chip = Chip(devices=[q], frame="rotating", rwa=True)
        chip.wire(drv)
        seq = QuantumSequence(chip)
        seq.schedule(drv, envelope=Gaussian(duration=40.0, amplitude=amp, sigmas=4.0), freq=5.0)
        result = seq.simulate(
            initial_state=chip.bare_state({q: 0}),
            tlist=jnp.linspace(0.0, 40.0, 60),
        )
        return result.population(q, level=1)[-1]

    val = loss(jnp.array(0.04))
    grad = jax.grad(loss)(jnp.array(0.04))
    assert jnp.isfinite(val) and jnp.isfinite(grad)


@pytest.mark.optional_backend
def test_vmap_through_chip_freq_and_static_zz():
    """``vmap`` over chip parameters must compose with ``chip.freq`` / ``static_zz``."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    from quchip import Capacitive, Chip, DuffingTransmon, set_default_backend

    set_default_backend("dynamiqs")

    def freq_zz(freq_q2, g):
        q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.30, label="q1")
        q2 = DuffingTransmon(freq=freq_q2, anharmonicity=-0.30, label="q2")
        chip = Chip([q1, q2], [Capacitive(q1, q2, g=g)], rwa=True)
        return jnp.stack([chip.freq(q2), chip.static_zz(q1, q2)])

    batched = jax.jit(jax.vmap(freq_zz, in_axes=(0, 0)))(
        jnp.array([5.10, 5.15, 5.20]),
        jnp.array([0.005, 0.005, 0.005]),
    )
    assert batched.shape == (3, 2)
    assert jnp.isfinite(batched).all()


@pytest.mark.optional_backend
def test_jit_with_traced_carrier_freq_from_chip():
    """A JIT loss may use ``chip.freq(...)`` directly as the drive carrier."""
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

    @jax.jit
    def loss(qb_freq):
        q = DuffingTransmon(freq=qb_freq, anharmonicity=-0.30, levels=3, label="q")
        drv = ChargeDrive(target=q, label="d")
        chip = Chip(devices=[q], frame="rotating", rwa=True)
        chip.wire(drv)
        seq = QuantumSequence(chip)
        seq.schedule(
            drv,
            envelope=Gaussian(duration=25.0, amplitude=0.04, sigmas=4.0),
            freq=chip.freq(q),
        )
        result = seq.simulate(
            initial_state=chip.bare_state({q: 0}),
            tlist=jnp.linspace(0.0, 25.0, 60),
        )
        return result.population(q, level=1)[-1]

    val = loss(jnp.array(5.05))
    grad = jax.grad(loss)(jnp.array(5.05))
    assert jnp.isfinite(val) and jnp.isfinite(grad)
