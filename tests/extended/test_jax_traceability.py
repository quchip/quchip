"""JAX traceability coverage for public array math."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quchip import Capacitive, analyze_cr_susceptibility
from quchip.backend import _backend_context, SolverResult
from quchip.chip.chip import Chip, DressedResult
from quchip.control import ChargeDrive, ControlEquipment, DriveModulation
from quchip.control.signal import Crosstalk
from quchip.control.envelopes import Gaussian, Square
from quchip.control.sequence import QuantumSequence
from quchip.devices.resonator import Resonator
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.engine.bands import canonical_to_dense_array, decompose_bands, decompose_two_body_canonical_bands
from quchip.engine.ir import CanonicalOperator
from quchip.engine.stage1_frames import resolve_frame
from quchip.results.results import SimulationResult


def test_gaussian_waveform_accepts_jax_array_module() -> None:
    """Gaussian.waveform should accept ``jax.numpy`` explicitly."""
    envelope = Gaussian(duration=20.0, amplitude=0.8)
    t = jnp.linspace(0.0, 20.0, 64)
    waveform = envelope.waveform(t, xp=jnp)
    assert isinstance(waveform, jax.Array)


def test_drive_local_channels_with_jax_backend() -> None:
    """ChargeDrive.local_channels should return channels even when JAX is available."""
    from quchip.devices.transmon.duffing import DuffingTransmon

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
    drive = ChargeDrive(target=q)
    channels = drive.local_channels(q)
    assert len(channels) == 1
    assert channels[0].modulation is DriveModulation.SINGLE_TONE


def test_crosstalk_construction_and_apply() -> None:
    """Crosstalk signal transform stores parameters and apply() produces output."""
    source_device = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    victim_device = DuffingTransmon(freq=5.5, anharmonicity=-0.25, levels=3, label="q1")
    source_drive = ChargeDrive(target=source_device)
    victim_drive = ChargeDrive(target=victim_device)
    edge = Crosstalk(source=source_drive.label, victim=victim_drive.label, beta=0.1, theta=0.2, delay=0.5)
    assert edge.beta == 0.1
    assert edge.theta == 0.2
    assert edge.delay == 0.5
    assert edge.source == source_drive.label
    assert edge.victim == victim_drive.label

    # apply should inject a leaked signal for the victim key
    from quchip.engine.ir import Constant
    signals = {(source_drive.label, 0): Constant(1.0 + 0j)}
    result = edge.apply(signals)
    assert any(k[0] == victim_drive.label for k in result)


def test_crosstalk_apply_produces_polar_scale_node() -> None:
    """Crosstalk.apply() wraps leaked signals in PolarScale for JAX traceability."""
    from quchip.engine.ir import Constant, PolarScale
    edge = Crosstalk(source="src_drive", victim="vic_drive", beta=0.1, theta=0.2, delay=0.5)
    signals = {("src_drive", 0): Constant(1.0 + 0j)}
    result = edge.apply(signals)
    victim_signal = result[("vic_drive", 0)]
    assert isinstance(victim_signal, PolarScale)
    assert victim_signal.amplitude == 0.1
    assert victim_signal.theta == 0.2


def test_decomposition_helpers_preserve_jax_arrays() -> None:
    """Band decomposition helpers should keep the differentiable array type."""
    single = jnp.asarray(np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.complex128))
    single_bands = decompose_bands(single, 2)
    assert single_bands
    assert all(isinstance(band, jax.Array) for band in single_bands.values())

    two_body = CanonicalOperator.from_dense(
        jnp.asarray(np.eye(4, dtype=np.complex128)), dims=(2, 2), basis="fock", subsystem_labels=("a", "b")
    )
    two_body_bands = decompose_two_body_canonical_bands(two_body, [2, 2])
    assert two_body_bands
    assert all(isinstance(canonical_to_dense_array(band), jax.Array) for band in two_body_bands.values())


class _JaxCollapseBackend:
    """Minimal backend stub for local collapse-operator traceability checks."""

    array_module = jnp

    @staticmethod
    def destroy(n: int) -> jax.Array:
        return jnp.diag(jnp.sqrt(jnp.arange(1, n, dtype=jnp.float32)), k=1).astype(jnp.complex64)

    @staticmethod
    def dag(op: jax.Array) -> jax.Array:
        return jnp.conjugate(jnp.swapaxes(op, -1, -2))

    @staticmethod
    def number(n: int) -> jax.Array:
        return jnp.diag(jnp.arange(n, dtype=jnp.complex64))


def test_canonical_operator_preserves_jax_arrays() -> None:
    """CanonicalOperator.values should preserve JAX arrays."""
    from quchip.engine.ir import CanonicalOperator

    dense = jnp.asarray(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128))
    canonical = CanonicalOperator.from_dense(
        dense, dims=(2,), basis="fock", subsystem_labels=("q",)
    )

    assert canonical.layout == "dense"
    assert isinstance(canonical.values, jax.Array)


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
                thermal_population=n_bar,
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


def test_resonator_quality_factor_collapse_accepts_traced_frequency() -> None:
    """Q-factor photon loss should preserve traced resonator frequency."""
    backend = _JaxCollapseBackend()

    @jax.jit
    def q_loss_coeff(freq: jax.Array) -> jax.Array:
        with _backend_context(backend):
            resonator = Resonator(freq=freq, quality_factor=10_000.0, levels=3)
            c_ops = resonator.collapse_operators()
        return jnp.real(c_ops[0][0, 1])

    value = q_loss_coeff(jnp.asarray(6.0))
    assert isinstance(value, jax.Array)
    assert jnp.isfinite(value)


class _ResultJaxBackend:
    """Small JAX-friendly backend stub for result-surface traceability tests."""

    array_module = jnp

    @staticmethod
    def basis(dim: int, n: int) -> jax.Array:
        return jnp.eye(dim, dtype=jnp.complex64)[n]

    @staticmethod
    def tensor_states(*states: jax.Array) -> jax.Array:
        result = states[0]
        for state in states[1:]:
            result = jnp.kron(result, state)
        return result

    @staticmethod
    def dag(op: jax.Array) -> jax.Array:
        if op.ndim == 1:
            return jnp.conjugate(op)
        return jnp.conjugate(jnp.swapaxes(op, -1, -2))

    @staticmethod
    def matmul(a: jax.Array, b: jax.Array) -> jax.Array:
        if a.ndim == b.ndim == 1:
            return jnp.outer(a, b)
        return a @ b

    @staticmethod
    def expect(op: jax.Array, state: jax.Array) -> jax.Array:
        return jnp.vdot(state, op @ state)

    @staticmethod
    def ptrace(state: jax.Array, dev_idx: int, dims: list[int]) -> jax.Array:
        _ = dev_idx, dims
        return jnp.outer(state, jnp.conjugate(state))

    @staticmethod
    def state_to_dm(state: jax.Array) -> jax.Array:
        return jnp.outer(state, jnp.conjugate(state))

    @staticmethod
    def is_ket(state: jax.Array) -> bool:
        return state.ndim == 1

    @staticmethod
    def stack_states(states: list[jax.Array]) -> jax.Array:
        return jnp.stack([jnp.asarray(s, dtype=jnp.complex64) for s in states])

    @staticmethod
    def expect_over_time(op: jax.Array, stacked: jax.Array) -> jax.Array:
        # Toy backend: kets are 1-D, so the stack is (T, n).
        return jnp.einsum("ti,ij,tj->t", jnp.conjugate(stacked), op, stacked)

    @staticmethod
    def coerce_state(state: jax.Array, dims: tuple[int, ...] | None = None) -> jax.Array:
        # Matches the Backend ABC default: this stub's states are already
        # native, so coercion is a no-op.
        _ = dims
        return state


def test_chip_dressed_spectrum_supports_jax_grad() -> None:
    """The public dressed-spectrum accessor should preserve autodiff arrays."""
    chip = Chip([Resonator(freq=5.0, levels=2, label="r")])

    def loss(scale: jax.Array) -> jax.Array:
        chip._analysis._dressed_result = DressedResult(
            eigenvalues=jnp.asarray([scale, scale**2 + 1.0], dtype=jnp.float32),
            state_map={(0,): 0, (1,): 1},
            dressed_eigenvalues={(0,): 0.0, (1,): 1.0},
            assignment_overlaps={(0,): 1.0, (1,): 1.0},
            hybridized_labels=(),
            bare_labels=((0,), (1,)),
            bare_labels_by_dressed_index={0: (0,), 1: (1,)},
            eigenvector_matrix=jnp.eye(2, dtype=jnp.complex64),
        )
        chip._analysis._dressed_signature = chip._analysis._analysis_signature()
        return jnp.sum(chip.dressed_spectrum())

    grad = jax.grad(loss)(jnp.asarray(0.25))
    assert jnp.isfinite(grad)


def test_chip_freq_when_supports_jax_grad() -> None:
    """The public dressed-frequency helper should preserve autodiff arrays."""
    chip = Chip([Resonator(freq=5.0, levels=2, label="r"), Resonator(freq=6.0, levels=2, label="q")])

    def loss(scale: jax.Array) -> jax.Array:
        chip._analysis._dressed_result = DressedResult(
            eigenvalues=jnp.asarray([0.0, 1.0, 2.0, 3.0], dtype=jnp.float32),
            state_map={(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3},
            dressed_eigenvalues={
                (0, 0): jnp.asarray(0.0, dtype=jnp.float32),
                (0, 1): jnp.asarray(1.0 + scale, dtype=jnp.float32),
                (1, 0): jnp.asarray(2.0, dtype=jnp.float32),
                (1, 1): jnp.asarray(3.5 + scale, dtype=jnp.float32),
            },
            assignment_overlaps={(0, 0): 1.0, (0, 1): 1.0, (1, 0): 1.0, (1, 1): 1.0},
            hybridized_labels=(),
            bare_labels=((0, 0), (0, 1), (1, 0), (1, 1)),
            bare_labels_by_dressed_index={0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)},
            eigenvector_matrix=jnp.eye(4, dtype=jnp.complex64),
        )
        chip._analysis._dressed_signature = chip._analysis._analysis_signature()
        return chip.freq("q", when={"r": 1})

    grad = jax.grad(loss)(jnp.asarray(0.25))
    assert jnp.isfinite(grad)


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
                Capacitive(control, bus, g=0.08, rwa=True),
                Capacitive(target, bus, g=0.08, rwa=True),
            ],
            backend="dynamiqs",
            control_equipment=ControlEquipment([drive]),
            rwa=True,
        )
        result = analyze_cr_susceptibility(chip, control, target)
        return jnp.abs(result.ZX_per_amplitude) ** 2

    value, derivative = jax.value_and_grad(loss)(jnp.asarray(6.5))

    assert jnp.isfinite(value)
    assert jnp.isfinite(derivative)
    assert abs(float(derivative)) > 1e-8


def test_simulation_result_overlap_array_supports_jax_grad() -> None:
    """The public overlap-array helper should preserve autodiff arrays."""
    backend = _ResultJaxBackend()
    target = backend.basis(2, 1)
    times = jnp.linspace(0.0, 1.0, 4)

    def loss(theta: jax.Array) -> jax.Array:
        state = jnp.asarray(
            [jnp.cos(theta), jnp.sin(theta)],
            dtype=jnp.complex64,
        )
        result = SimulationResult(
            solver_result=SolverResult(
                times=times,
                states=[state],
                expect=None,
                final_state=state,
                stats=None,
                solver="sesolve",
            ),
            backend=backend,
            dims=[2],
            device_info=[("q", True)],
        )
        return jnp.sum(result.overlap_array(target))

    grad = jax.grad(loss)(jnp.asarray(0.3))
    assert jnp.isfinite(grad)


def test_problem_batch_manual_indexing_preserves_array_type() -> None:
    """ProblemBatch params bookkeeping should not coerce JAX payloads."""
    from quchip.control.batch import ProblemBatch

    params = np.empty((1,), dtype=object)
    params[(0,)] = {"x": jnp.asarray(1.0, dtype=jnp.float32)}
    class _Batch:
        batch_size = 1

        def element(self, idx):
            return object()

    batch = ProblemBatch(batch=_Batch(), params=params, shape=(1,), axes=(("x", [params[(0,)]["x"]]),))
    assert isinstance(batch.params_at(0)["x"], jax.Array)


@pytest.mark.optional_backend
def test_quantum_sequence_build_problem_accepts_traced_tlist() -> None:
    """Explicit JAX ``tlist`` should remain traceable through build_problem()."""
    pytest.importorskip("dynamiqs")
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q")
    drive = ChargeDrive(target=q)
    chip = Chip([q], frame="rotating", backend="dynamiqs",
                control_equipment=ControlEquipment(lines=[drive]))
    seq = QuantumSequence(chip)
    seq.schedule(drive, envelope=Square(duration=10.0, amplitude=0.01), freq=5.0)
    initial_state = chip.state(q=0)

    @jax.jit
    def build_with_tlist(tlist):
        problem = seq.build_problem(tlist=tlist, initial_state=initial_state)
        return problem.tlist

    traced_tlist = jnp.linspace(0.0, 10.0, 8)
    result_tlist = build_with_tlist(traced_tlist)

    assert isinstance(result_tlist, jax.Array)
    np.testing.assert_allclose(np.asarray(result_tlist), np.asarray(traced_tlist))


@pytest.mark.optional_backend
def test_simulate_jits_through_coupled_chip_dense_two_body_embed() -> None:
    """``jax.jit`` must span simulate() on a coupled rotating-frame chip through the dense embed_two_body path."""
    # Until 2026-06 that path concretized a reshape size via int(jnp.prod(...)), which broke
    # under jit since jnp constants are tracers inside a trace.
    pytest.importorskip("dynamiqs")
    from quchip import Capacitive, Resonator
    from quchip.engine import simulate

    def loss(freq):
        q = DuffingTransmon(freq=freq, anharmonicity=-0.3, levels=3, label="q")
        r = Resonator(freq=7.1, levels=3, label="r")
        chip = Chip([q, r], [Capacitive(q, r, g=0.06, rwa=True)],
                    frame="rotating", rwa=True, backend="dynamiqs")
        result = simulate(
            chip, [], jnp.linspace(0.0, 10.0, 8),
            initial_state=chip.bare_state({q: 1, r: 0}),
            check_truncation=False,
        )
        return result.population_array(q, level=1)[-1]

    value, grad = jax.jit(jax.value_and_grad(loss))(jnp.asarray(5.02))
    assert jnp.isfinite(value) and jnp.isfinite(grad)


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
        r = simulate(chip, [], t, initial_state=plus, options={"store_states": True})
        return jnp.real(r.overlap_array(plus)[-1])

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
        chip = Chip([q, r], [Capacitive(q, r, g=0.06, rwa=True)],
                    frame="rotating", rwa=True, backend="dynamiqs")
        result = simulate(
            chip, [], jnp.linspace(0.0, 10.0, 8),
            initial_state=chip.state({q: 1, r: 0}),
            check_truncation=False,
        )
        return result.population_array(q, level=1)

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
        chip = Chip([q, r], [Capacitive(q, r, g=0.06, rwa=True)],
                    frame="rotating", rwa=True, backend="dynamiqs")
        result = simulate(chip, [], jnp.linspace(0.0, 10.0, 8), check_truncation=False)
        return result.population_array(q, level=0)[-1]

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
        return Chip([q, r], [Capacitive(q, r, g=0.06, rwa=True)], backend="dynamiqs")

    eager = np.asarray(make(5.02).state(q=1).to_jax()).ravel()
    traced = np.asarray(
        jax.jit(lambda f: jnp.asarray(make(f).state(q=1).to_jax()))(jnp.asarray(5.02))
    ).ravel()
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
    from quchip.engine.ir import Constant, evaluate_signal_program

    q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q1")
    q2 = DuffingTransmon(freq=5.5, anharmonicity=-0.2, levels=2, label="q2")
    d1 = ChargeDrive(target=q1, label="d1")
    d2 = ChargeDrive(target=q2, label="d2")
    equip = ControlEquipment(
        lines=[d1, d2],
        signal_chain=[Crosstalk(source=d1.label, victim=d2.label, beta=0.1)],
    )

    base_signals = {(d1.label, 0): Constant(1.0 + 0.0j)}

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
        value = evaluate_signal_program(leaked, 0.0, xp=jnp)
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
