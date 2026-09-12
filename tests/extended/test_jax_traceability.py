"""JAX traceability coverage for public array math."""

from __future__ import annotations

from quchip.approximations import RWA

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quchip import Capacitive, analyze_cr_susceptibility
from quchip.backend import _backend_context
from quchip.chip.chip import Chip
from quchip.control import ChargeDrive, ControlEquipment
from quchip.control.signal import AnalyticSignal
from quchip.control.envelopes import Gaussian
from quchip.control.sequence import QuantumSequence
from quchip.devices.resonator import Resonator
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.engine.ir import CanonicalOperator
from quchip.engine.frames import resolve_frame


def test_shift_phase_differentiates_with_respect_to_delay() -> None:
    """A concrete carrier frequency must not concretize a traced time shift."""
    from quchip.engine.ir import _shift_phase

    frequency = 1.7
    delay = jnp.asarray(0.4)
    gradient = jax.grad(lambda value: jnp.imag(_shift_phase(frequency, value)))(delay)
    expected = -frequency * jnp.cos(frequency * delay)

    np.testing.assert_allclose(gradient, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.optional_backend
def test_gaussian_ramsey_delay_gradient_matches_finite_difference() -> None:
    """A swept Ramsey delay remains differentiable through pulse scheduling."""
    pytest.importorskip("dynamiqs")

    q = DuffingTransmon(freq=5.005, anharmonicity=-0.30, levels=2, label="q")
    drive = ChargeDrive(target=q)
    chip = Chip(
        [q],
        frame={q: 5.0},
        approximation=RWA(),
        backend="dynamiqs",
        control_equipment=ControlEquipment(lines=[drive]),
    )
    sequence = QuantumSequence(chip)
    sequence.schedule(
        drive,
        envelope=Gaussian(duration=20.0, amplitude=0.02, sigmas=4),
        freq=5.0,
    )
    wait = sequence.delay(q, 80.0)
    sequence.schedule(
        drive,
        envelope=Gaussian(duration=20.0, amplitude=0.02, sigmas=4),
        freq=5.0,
    )
    initial_state = chip.state(q=0)
    tlist = jnp.linspace(0.0, 160.0, 161)

    def final_population(delay: jax.Array) -> jax.Array:
        result = sequence.simulate_batch(
            wait.vary("duration", jnp.asarray([delay]), name="delay"),
            tlist=tlist,
            initial_state=initial_state,
            progress=False,
            )
        return jnp.reshape(result.population("q", level=1, reduce="last"), (-1,))[0]

    delay = jnp.asarray(80.0)
    autodiff = jax.grad(final_population)(delay)
    step = 1e-2
    finite_difference = (
        final_population(delay + step) - final_population(delay - step)
    ) / (2.0 * step)

    assert abs(float(autodiff)) > 1e-5
    np.testing.assert_allclose(autodiff, finite_difference, rtol=1e-3, atol=1e-6)


class _JaxCollapseBackend:
    """Minimal backend stub for local collapse-operator traceability checks."""

    array_module = jnp

    @staticmethod
    def destroy(n: int) -> jax.Array:
        return jnp.diag(jnp.sqrt(jnp.arange(1, n, dtype=jnp.float32)), k=1).astype(jnp.complex64)

    @staticmethod
    def create(n: int) -> jax.Array:
        return _JaxCollapseBackend.dag(_JaxCollapseBackend.destroy(n))

    @staticmethod
    def dag(op: jax.Array) -> jax.Array:
        return jnp.conjugate(jnp.swapaxes(op, -1, -2))

    @staticmethod
    def number(n: int) -> jax.Array:
        return jnp.diag(jnp.arange(n, dtype=jnp.complex64))


def test_base_device_collapse_operators_accept_traced_noise_params() -> None:
    """Base T1/T2/thermal channels should not branch on traced noise values."""
    backend = _JaxCollapseBackend()

    @jax.jit
    def collapse_metric(T1: jax.Array, T2: jax.Array, n_bar: jax.Array) -> jax.Array:
        with _backend_context(backend):
            q = DuffingTransmon(
                freq=5.0,
                anharmonicity=-0.25,
                levels=3,
                T1=T1,
                T2=T2,
                thermal_occupation=n_bar,
            )
            c_ops = q.collapse_operators()
        return jnp.real(c_ops[0][0, 1] + c_ops[1][1, 0] + c_ops[2][1, 1])

    value = collapse_metric(
        jnp.asarray(10_000.0),
        jnp.asarray(8_000.0),
        jnp.asarray(0.05),
    )
    assert isinstance(value, jax.Array)
    assert jnp.isfinite(value)


def test_resonator_internal_quality_factor_collapse_accepts_traced_frequency() -> None:
    """Q-factor photon loss should preserve traced resonator frequency."""
    backend = _JaxCollapseBackend()

    @jax.jit
    def q_loss_coeff(freq: jax.Array) -> jax.Array:
        with _backend_context(backend):
            resonator = Resonator(freq=freq, internal_quality_factor=10_000.0, levels=3)
            c_ops = resonator.collapse_operators()
        return jnp.real(c_ops[0][0, 1])

    value = q_loss_coeff(jnp.asarray(6.0))
    assert isinstance(value, jax.Array)
    assert jnp.isfinite(value)


@pytest.mark.optional_backend
def test_drive_matrix_element_ratio_supports_jax_grad() -> None:
    """Dressed drive-element ratios preserve gradients through coupling strength."""
    pytest.importorskip("dynamiqs")
    from quchip import Capacitive

    def loss(g: jax.Array) -> jax.Array:
        q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q1")
        q2 = DuffingTransmon(freq=5.3, anharmonicity=-0.24, levels=3, label="q2")
        d1 = ChargeDrive(q1, label="d1")
        d2 = ChargeDrive(q2, label="d2")
        chip = Chip(
            [q1, q2],
            [Capacitive(q1, q2, g=g)],
            backend="dynamiqs",
            control_equipment=ControlEquipment([d1, d2]),
        )
        elements = chip.drive_matrix_elements(q1, drives=[d1, d2])
        return jnp.real(elements[d2] / elements[d1])

    value, grad = jax.value_and_grad(loss)(jnp.asarray(0.03))

    assert jnp.isfinite(value)
    assert jnp.isfinite(grad)
    assert abs(float(grad)) > 1e-6


@pytest.mark.optional_backend
def test_cr_susceptibility_supports_jax_grad() -> None:
    """Weak-drive ZX susceptibility preserves gradients through a bus frequency."""
    pytest.importorskip("dynamiqs")

    def loss(bus_freq: jax.Array) -> jax.Array:
        control = DuffingTransmon(freq=5.2, anharmonicity=-0.3, levels=3, label="c")
        target = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3, label="t")
        bus = Resonator(freq=bus_freq, levels=3, label="b")
        drive = ChargeDrive(control, label="d")
        chip = Chip(
            [control, target, bus],
            [
                Capacitive(control, bus, g=0.08),
                Capacitive(target, bus, g=0.08),
            ],
            backend="dynamiqs",
            control_equipment=ControlEquipment([drive]),
            approximation=RWA(),
        )
        result = analyze_cr_susceptibility(chip, control, target)
        return jnp.abs(result.ZX_per_amplitude) ** 2

    value, derivative = jax.value_and_grad(loss)(jnp.asarray(6.5))

    assert jnp.isfinite(value)
    assert jnp.isfinite(derivative)
    assert abs(float(derivative)) > 1e-8


@pytest.mark.optional_backend
def test_reference_freq_supports_jax_grad() -> None:
    """A device's ``reference_freq`` (readout/frame LO) must be differentiable."""
    # Detuning the reference surfaces idle precession Delta = omega - reference_freq in
    # transverse observables, so d<sigma_x>(T)/d(reference_freq) is finite and non-zero.
    pytest.importorskip("dynamiqs")
    from quchip.engine import simulate

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.30, levels=3, label="q")
    chip = Chip([q], frame="rotating", backend="dynamiqs")
    plus = (chip.bare_state({q: 0}) + chip.bare_state({q: 1})) / np.sqrt(2)
    t = jnp.linspace(0.0, 50.0, 26)

    # reference_freq sets the rotating frame, so detuning it precesses the state
    # relative to a fixed |+> target; the JAX-native overlap accessor keeps the
    # whole path differentiable without a backend-bound e_op.
    def loss(ref):
        q.reference_freq = ref
        r = simulate(chip, [], t, initial_state=plus, states="all")
        return jnp.real(r.overlap(plus)[-1])

    # Δ·T ≈ 0.25 cycle (Δ = 5 MHz over 50 ns): the steepest point of the overlap
    # fringe, so the gradient is O(1), a strong physical check of idle precession.
    value, grad = jax.value_and_grad(loss)(jnp.asarray(4.995))
    assert jnp.isfinite(value) and jnp.isfinite(grad)
    assert abs(float(grad)) > 1e-3, "reference_freq detuning must move the state's transverse overlap"


@pytest.mark.optional_backend
def test_chip_state_dressed_initial_state_traces_through_simulate() -> None:
    """``chip.state()`` (dressed) must work as an initial state under jit/grad."""
    # A dressed eigenstate of the static chip is stationary, so its bare-qubit population
    # is time-independent and stays near 1 for the weakly hybridized |1,0> label.
    pytest.importorskip("dynamiqs")
    from quchip import Capacitive, Resonator
    from quchip.engine import simulate

    def trace(freq):
        q = DuffingTransmon(freq=freq, anharmonicity=-0.3, levels=3, label="q")
        r = Resonator(freq=7.1, levels=3, label="r")
        chip = Chip(
            [q, r],
            [Capacitive(q, r, g=0.06)],
            frame="rotating",
            approximation=RWA(),
            backend="dynamiqs",
        )
        result = simulate(
            chip,
            [],
            jnp.linspace(0.0, 10.0, 8),
            initial_state=chip.state({q: 1, r: 0}),
            )
        return result.population(q, level=1)

    def loss(freq):
        return trace(freq)[-1]

    value, grad = jax.jit(jax.value_and_grad(loss))(jnp.asarray(5.02))
    assert jnp.isfinite(value) and jnp.isfinite(grad)
    assert float(value) > 0.9

    # Stationarity discriminates dressed from bare: the solve runs in the
    # rotating frame under RWA, so a lab-dressed eigenstate is constant only
    # up to RWA corrections (~4e-5 here), while a bare |1,0> would beat with
    # amplitude 4g²/Δ² ≈ 3e-3. 5e-4 sits between the two scales.
    pops = np.asarray(jax.jit(trace)(jnp.asarray(5.02)))
    np.testing.assert_allclose(pops, float(pops[0]), atol=5e-4)


@pytest.mark.optional_backend
def test_default_initial_state_omitted_traces_through_simulate() -> None:
    """Omitting ``initial_state`` under jit uses the traced dressed ground state, never a cached tracer."""
    # Exercises the _LazyDefaultState traced branch: chip.state() fires inside the trace and
    # its result must not be memoized, or a cached tracer would leak into the second call below.
    pytest.importorskip("dynamiqs")
    from quchip import Capacitive, Resonator
    from quchip.engine import simulate

    def loss(freq):
        q = DuffingTransmon(freq=freq, anharmonicity=-0.3, levels=3, label="q")
        r = Resonator(freq=7.1, levels=3, label="r")
        chip = Chip(
            [q, r],
            [Capacitive(q, r, g=0.06)],
            frame="rotating",
            approximation=RWA(),
            backend="dynamiqs",
        )
        result = simulate(chip, [], jnp.linspace(0.0, 10.0, 8))
        return result.population(q, level=0)[-1]

    fn = jax.jit(jax.value_and_grad(loss))
    value, grad = fn(jnp.asarray(5.02))
    assert jnp.isfinite(value) and jnp.isfinite(grad)
    assert float(value) > 0.99  # dressed ground state stays in the ground state

    value2, _ = fn(jnp.asarray(5.03))  # second trace/eval: no stale-tracer leak
    assert jnp.isfinite(value2)


@pytest.mark.optional_backend
def test_chip_state_traced_matches_eager_on_dynamiqs() -> None:
    """Traced kernel column selection must equal the eager dict-view state."""
    pytest.importorskip("dynamiqs")
    from quchip import Capacitive, Resonator

    def make(freq):
        q = DuffingTransmon(freq=freq, anharmonicity=-0.3, levels=3, label="q")
        r = Resonator(freq=7.1, levels=3, label="r")
        return Chip([q, r], [Capacitive(q, r, g=0.06)], backend="dynamiqs")

    eager = np.asarray(make(5.02).state(q=1).to_jax()).ravel()
    traced = np.asarray(jax.jit(lambda f: jnp.asarray(make(f).state(q=1).to_jax()))(jnp.asarray(5.02))).ravel()
    np.testing.assert_allclose(traced, eager, atol=1e-10)


@pytest.mark.optional_backend
def test_resolve_frame_accepts_traced_frame_dict() -> None:
    """Frame resolution should not concretize traced per-device frequencies."""
    pytest.importorskip("dynamiqs")
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q")
    chip = Chip([q], frame="rotating", backend="dynamiqs")

    @jax.jit
    def demod_freq(frame_freq):
        resolved = resolve_frame(chip, {"q": frame_freq})
        return resolved.demod_freqs["q"]

    value = demod_freq(jnp.asarray(5.0))
    assert isinstance(value, jax.Array)
    np.testing.assert_allclose(np.asarray(value), 0.0)


def test_crosstalk_matrix_grad_flows_end_to_end() -> None:
    """``jax.grad`` flows from a traced beta matrix through ``set_crosstalk_matrix`` to a leaked-signal loss."""
    from quchip.control import ControlEquipment, Crosstalk
    from quchip.engine.ir import Constant

    q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q1")
    q2 = DuffingTransmon(freq=5.5, anharmonicity=-0.2, levels=2, label="q2")
    d1 = ChargeDrive(target=q1, label="d1")
    d2 = ChargeDrive(target=q2, label="d2")
    equip = ControlEquipment(
        lines=[d1, d2],
        signal_chain=[Crosstalk(source=d1.label, victim=d2.label, beta=0.1)],
    )

    base_signals = {(d1.label, 0): AnalyticSignal(Constant(1.0 + 0.0j))}

    def loss(beta_flat: jax.Array) -> jax.Array:
        # Rebuild a 2x2 beta from a flat traced vector (diagonals fixed
        # at 1, off-diagonals swept).
        beta = jnp.asarray(
            [
                [jnp.asarray(1.0), beta_flat[0]],
                [beta_flat[1], jnp.asarray(1.0)],
            ]
        )
        equip.set_crosstalk_matrix(beta)

        built = equip.apply_signal_chain(base_signals)
        leaked = built[(d2.label, 0)]
        value = leaked.evaluate(0.0, xp=jnp)
        # Population-like scalar: |amplitude|^2.
        return jnp.real(value * jnp.conj(value))

    beta_flat = jnp.asarray([0.15, 0.02])
    grad = jax.grad(loss)(beta_flat)

    assert isinstance(grad, jax.Array)
    assert jnp.all(jnp.isfinite(grad))
    # ``beta_flat[1]`` corresponds to ``beta[1, 0]`` (source=d1, victim=d2);
    # that is the only entry that drives the leaked amplitude into the
    # victim channel in this setup, so its gradient must be non-zero.
    assert float(jnp.abs(grad[1])) > 1e-6


@pytest.mark.optional_backend
def test_dynamiqs_from_canonical_operator_accepts_traced_dia_offsets() -> None:
    """Dynamiqs DIA reconstruction should not concretize traced offsets."""
    pytest.importorskip("dynamiqs")
    from quchip.backend.dynamiqs import DynamiqsBackend

    backend = DynamiqsBackend()

    @jax.jit
    def build_with_offset(offset):
        canonical = CanonicalOperator.from_dia(
            jnp.asarray([[1.0 + 0.0j, 2.0 + 0.0j]], dtype=jnp.complex128),
            jnp.asarray([offset], dtype=jnp.int64),
            shape=(2, 2),
            dims=(2,),
            basis="fock",
            subsystem_labels=("q",),
        )
        return backend.to_array(backend.from_canonical_operator(canonical))

    rebuilt = build_with_offset(jnp.asarray(0))
    assert isinstance(rebuilt, jax.Array)
    np.testing.assert_allclose(np.asarray(rebuilt), np.array([[1.0 + 0.0j, 0.0], [0.0, 2.0 + 0.0j]]))


@pytest.mark.optional_backend
def test_dynamiqs_preserves_static_dia_structure_with_traced_values() -> None:
    """Differentiable DIA values lower sparsely when their offsets are static."""
    pytest.importorskip("dynamiqs")
    from quchip.backend.dynamiqs import DynamiqsBackend

    backend = DynamiqsBackend()
    seen: dict[str, str] = {}

    @jax.jit
    def lower(scale):
        canonical = CanonicalOperator.from_dia(
            scale * jnp.asarray([[0.0, 1.0]], dtype=jnp.complex128),
            np.asarray([1], dtype=int),
            shape=(2, 2),
            dims=(2,),
            basis="fock",
            subsystem_labels=("q",),
        )
        native = backend.from_canonical_operator(canonical)
        seen["type"] = type(native).__name__
        return backend.to_array(native)

    rebuilt = lower(jnp.asarray(3.0))
    assert seen["type"] == "SparseDIAQArray"
    np.testing.assert_allclose(np.asarray(rebuilt), [[0.0, 3.0], [0.0, 0.0]])
