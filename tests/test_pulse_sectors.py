"""Exact pulse spaces preserve the complete generator and captured state semantics."""

import numpy as np
import pytest

from quchip import RWA, Capacitive, Chip, QuantumSequence, Resonator, check_truncation


def pair(*, loss=False, backend=None):
    a = Resonator(freq=0.25, levels=3, label="a", **({"T1": 20.0} if loss else {}))
    b = Resonator(freq=0.25, levels=3, label="b")
    return Chip([a, b], couplings=[Capacitive(a, b, g=0.02)], approximation=RWA(), frame="rotating", backend=backend)


def test_cascade_without_dissipation_keeps_its_exchange():
    """Without dissipation, the one-excitation space of a cascade excludes the vacuum that its exchange passes."""
    from quchip import PortNetwork

    r1, r2 = Resonator(freq=6.0, levels=3, label="r1"), Resonator(freq=6.0, levels=3, label="r2")
    network = PortNetwork(label="line")
    rate = 2 * np.pi * 0.01
    first, second = network.port("p1", target=r1, rate=rate), network.port("p2", target=r2, rate=rate)
    network.cascade(first, second)
    network.expose("out", input=first.input, output=second.output)
    sequence = QuantumSequence(Chip([r1, r2], port_network=network))
    times = np.linspace(0.0, 40.0, 81)
    problem = sequence.build_problem(times, initial_state={"r1": 1}, dissipation=False)
    assert problem.engine_result.excitation_space.size == 2
    result = sequence.simulate(tlist=times, initial_state={"r1": 1}, dissipation=False,
                               options={"atol": 1e-12, "rtol": 1e-10})
    # H_gen = (L_2^dag L_1 - h.c.) / 2i exchanges one photon at sqrt(k1 k2) / 2.
    np.testing.assert_allclose(result.population("r2", 1), np.sin(rate * times / 2) ** 2, atol=1e-6)


def test_exchange_uses_exact_space_and_exports_original_coordinates():
    """A one-excitation exchange evolves two amplitudes and retains tensor export semantics."""
    chip = pair()
    times = np.linspace(0, 10, 51)
    problem = QuantumSequence(chip).build_problem(times, initial_state={"a": 1}, dissipation=False, states="all")
    assert problem.engine_result.excitation_space.size == 2
    result = chip.solve(problem)
    # Analytic exchange. QuTiP diagonalizes this static generator, so only round-off remains.
    np.testing.assert_allclose(result.population("a", 1), np.cos(2 * np.pi * 0.02 * times) ** 2, atol=2e-6)
    assert result.final_state.shape == (9, 1)
    assert result.dims == [3, 3]
    np.testing.assert_allclose(result.reduced_state(times[-1], "a").tr(), 1, atol=1e-8)


def test_unsplit_simulation_assembles_its_compact_model_once(monkeypatch):
    """The partition probe and the joint solve share one compact assembly."""
    import quchip.engine.assembly as assembly

    chip, calls, build = pair(), [], assembly._build_static_h0

    def counted(*args, **kwargs):
        calls.append(1)
        return build(*args, **kwargs)

    monkeypatch.setattr(assembly, "_build_static_h0", counted)
    result = QuantumSequence(chip).simulate(tlist=[0, 1], initial_state={"a": 1}, dissipation=False)
    assert result.stats["excitation_reduction"]["dimension"] == 2
    assert len(calls) == 1


@pytest.mark.parametrize("backend", [None, pytest.param("dynamiqs", marks=pytest.mark.optional_backend)])
def test_decay_coherence_compact_replay_and_measurement(backend):
    """Loss adds vacuum without removing coherence, and captured states replay without lifting."""
    if backend:
        pytest.importorskip("dynamiqs")
    chip = pair(loss=True, backend=backend)
    initial = chip.compact_state((1, {}), (1j, {"a": 1}))
    seq = QuantumSequence(chip)
    times = np.linspace(0, 5, 21)
    problem = seq.build_problem(times, initial_state=initial, states="all")
    assert problem.engine_result.excitation_space.size == 3
    result = chip.solve(problem)
    assert abs(np.asarray(chip.backend.to_array(result.reduced_state(5, "a")))[0, 1]) > 0.1
    # The probabilities sum to the trace. QuTiP diagonalizes this static generator, and
    # dynamiqs integrates at rtol 1e-9, so both backends stay far inside 2e-6.
    np.testing.assert_allclose(np.sum(result.measure("a").probabilities), 1, atol=2e-6)
    resumed = seq.simulate(tlist=np.linspace(5, 10, 21), initial_state=result.compact_state(), states="final")
    reference = seq.simulate(tlist=np.linspace(0, 10, 41), initial_state=initial, states="final")
    # The replay adds one propagation segment with the same error sources.
    np.testing.assert_allclose(
        chip.backend.to_array(resumed.final_state), chip.backend.to_array(reference.final_state), atol=3e-6
    )
    assert check_truncation(result)


@pytest.mark.parametrize("loss", [False, True])
@pytest.mark.parametrize("frame", ["lab", "auto"])
def test_large_ring_never_embeds_full_operators(loss, frame, monkeypatch):
    """Twenty-four three-level devices evolve in 24 amplitudes with guarded exports."""
    from quchip.utils import memory as _memory

    # A fixed host memory keeps the export guard active on every platform.
    monkeypatch.setattr(_memory, "available_memory_bytes", lambda: 2**33)
    devices = [Resonator(freq=0.25, levels=3, label=f"r{i}", **({"T1": 20} if loss else {})) for i in range(24)]
    chip = Chip(
        devices,
        couplings=[Capacitive(a, devices[(i + 1) % 24], g=0.01) for i, a in enumerate(devices)],
        approximation=RWA(),
        frame=frame,
    )
    original_embed = chip.backend.embed

    def guarded_embed(operator, target, dims):
        assert np.prod(dims, dtype=object) < 1000, "Full tensor embedding reached the backend."
        return original_embed(operator, target, dims)

    monkeypatch.setattr(chip.backend, "embed", guarded_embed)
    problem = QuantumSequence(chip).build_problem(np.linspace(0, 2, 11), initial_state={"r0": 1}, dissipation=loss)
    assert problem.engine_result.dims == (25 if loss else 24,)
    result = QuantumSequence(chip).simulate(
        tlist=np.linspace(0, 2, 11),
        initial_state={"r0": np.array([0, 1, 0])},
        dissipation=loss,
    )
    from scipy.linalg import expm

    graph = 0.25 * np.eye(24)
    for i in range(24):
        graph[i, (i + 1) % 24] = graph[(i + 1) % 24, i] = 0.01
    expected = expm(-2j * np.pi * graph * 2)[:, 0]
    # The matrix exponential is exact, and QuTiP diagonalizes this static compact generator.
    for i in range(24):
        np.testing.assert_allclose(
            result.population(f"r{i}", 1)[-1], abs(expected[i]) ** 2 * (np.exp(-2 / 20) if loss else 1), atol=2e-6
        )
    saved = result.compact_state()
    indices = [saved.space.rows[tuple(int(j == i) for j in range(24))] for i in range(24)]
    values = np.asarray(saved.values)
    density = values @ values.conj().T if values.shape[-1] == 1 else values
    # The same exact reference and propagation as the populations.
    np.testing.assert_allclose(
        density[np.ix_(indices, indices)],
        np.outer(expected, expected.conj()) * (np.exp(-2 / 20) if loss else 1),
        atol=2e-6,
    )
    with pytest.raises(MemoryError):
        _ = result.final_state
    assert result.compact_state().values.shape == ((25, 25) if loss else (24, 1))


def test_transverse_drive_zero_amplitude_does_not_prove_conservation():
    """A microwave drive keeps its excitation-changing support even at zero amplitude."""
    from quchip import ChargeDrive, Square

    chip = pair()
    chip.wire(ChargeDrive(chip["a"], label="drive"))
    seq = QuantumSequence(chip)
    seq.schedule("drive", envelope=Square(duration=2, amplitude=0), freq=0.25)
    problem = seq.build_problem([0, 2], initial_state={"a": 1})
    assert problem.engine_result.excitation_space is None


@pytest.mark.unit
def test_jump_no_jump_product_prevents_false_vacuum_closure():
    """A jump preserving the vacuum can still move it through L dagger L."""
    from quchip import Bath
    from quchip.declarative import CollapseChannel

    class NonNormal(Bath):
        def dissipation(self, chip, bases=None):
            return (CollapseChannel(np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), 0.1, "mixed"),)

    q = Resonator(freq=0.25, levels=3, label="q")
    chip = Chip([q], baths=[NonNormal("collective_decay", rate=0.1)], frame="lab")
    problem = QuantumSequence(chip).build_problem([0, 1], initial_state={})
    assert problem.engine_result.excitation_space is None
    result = chip.solve(problem)
    assert abs(chip.backend.to_array(result.final_state)[0, 1]) > 0.01


@pytest.mark.unit
def test_thermal_channel_keeps_raising_support_at_zero_occupation():
    """An explicitly present absorption channel never disappears from the support proof."""
    q = Resonator(freq=0.25, levels=3, label="q", T1=20, thermal_occupation=0.0)
    problem = QuantumSequence(Chip([q])).build_problem([0, 1])
    assert problem.engine_result.excitation_space is None


@pytest.mark.unit
def test_truncated_basis_product_keeps_paths_through_discarded_levels():
    """A product on a truncated energy basis carries the changes of its complete matrix."""
    from quchip import EigenbasisDevice, Exact
    from quchip.declarative import CouplingModel, Scalar, parameter

    class ChargeSquaredNumber(CouplingModel):
        k: Scalar = parameter(unit="GHz")

        def interaction(self, a, b, p):
            return p.k * (a.charge @ a.charge) * b.n

    # The kept block of C has adjacent elements only. Level 3 gives C @ C a 0-1 element.
    C = np.array([[0, 1, 0, 0.5], [1, 0, 1, 0.7], [0, 1, 0, 0], [0.5, 0.7, 0, 0]])
    q = EigenbasisDevice([0.0, 0.2, 0.5, 0.9], charge_operator=C, levels=3, label="q")
    r = Resonator(freq=0.3, levels=3, label="r")
    chip = Chip([q, r], [ChargeSquaredNumber(q, r, k=0.05, label="qr")], approximation=Exact(), frame="lab")
    problem = QuantumSequence(chip).build_problem([0, 1], initial_state={"q": 1, "r": 1}, dissipation=False)
    assert problem.engine_result.excitation_space is None


@pytest.mark.parametrize("sign", [1, -1])
def test_collective_bright_dark_states(sign):
    """Collective decay keeps interference and the analytic bright/dark decay rates."""
    from quchip import Bath

    a = Resonator(freq=0.25, levels=2, label="a")
    b = Resonator(freq=0.25, levels=2, label="b")
    chip = Chip([a, b], baths=[Bath("collective_decay", rate=0.1)], frame=0.25)
    initial = chip.compact_state({"a": 1}, (sign, {"b": 1}))
    times = np.linspace(0, 5, 21)
    result = QuantumSequence(chip).simulate(tlist=times, initial_state=initial)
    assert result.stats["excitation_reduction"]["dimension"] == 3
    expected = np.exp(-0.2 * times) if sign == 1 else np.ones_like(times)
    # Analytic bright and dark decay. QuTiP diagonalizes this static generator, so only round-off remains.
    np.testing.assert_allclose(result.population("a", 1) + result.population("b", 1), expected, atol=2e-6)


def test_passive_cascade_matches_full_model(monkeypatch):
    """Cascade Hamiltonians, summed output jumps, and local channels agree with full tensor assembly."""
    from quchip import PortNetwork

    # Local relaxation and dephasing do not mix with the network channels.
    a = Resonator(freq=0.25, levels=3, label="a", T1=40.0, T2=30.0)
    b = Resonator(freq=0.25, levels=3, label="b")
    network = PortNetwork()
    pa, pb = network.port("pa", target=a, rate=0.04), network.port("pb", target=b, rate=0.09)
    network.cascade(pa, pb)
    network.expose("line", input=pa.input, output=pb.output)
    chip = Chip([a, b], port_network=network, frame=0.25)
    seq = QuantumSequence(chip)
    kwargs = dict(tlist=np.linspace(0, 10, 31), initial_state={"a": 1}, options={"atol": 1e-11, "rtol": 1e-10})
    actual = seq.simulate(**kwargs)
    assert actual.stats["excitation_reduction"]["totals"] == (0, 1)
    monkeypatch.setattr("quchip.engine.pulse_space.plan_pulse_space",
                        lambda *a, **k: (None, "test full reference", None))
    reference = seq.simulate(**kwargs)
    # Both routes integrate the same equations at rtol 1e-10, so 2e-7 bounds their difference.
    np.testing.assert_allclose(
        chip.backend.to_array(actual.final_state), chip.backend.to_array(reference.final_state), atol=2e-7
    )


@pytest.mark.validation
@pytest.mark.optional_backend
def test_jitted_parameter_gradient_matches_exchange_solution():
    """Differentiation through assembly and the native compact solver matches analytic exchange."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    chip = pair(backend="dynamiqs")
    edge = chip.couplings[0].label

    @jax.jit
    @jax.value_and_grad
    def objective(g):
        bound = chip.with_params({f"{edge}.g": g})
        result = QuantumSequence(bound).simulate(
            tlist=jnp.linspace(0, 3, 11), initial_state={"a": 1}, dissipation=False
        )
        return result.population("b", 1)[-1]

    g = 0.02
    value, gradient = objective(jnp.asarray(g))
    # Analytic exchange. The default Dopri8 rtol of 1e-9 keeps both values far inside these bounds.
    np.testing.assert_allclose(value, np.sin(2 * np.pi * g * 3) ** 2, atol=2e-6)
    np.testing.assert_allclose(gradient, 2 * np.pi * 3 * np.sin(4 * np.pi * g * 3), atol=2e-4)


@pytest.mark.validation
@pytest.mark.optional_backend
def test_traced_frequency_keeps_the_exact_space_and_detuned_exchange_gradient():
    """A traced device frequency keeps the two-amplitude space and the analytic detuned exchange."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    chip, g, t = pair(backend="dynamiqs"), 0.02, 10.0

    def population(frequency):
        bound = chip.with_params({"a.freq": frequency})
        result = QuantumSequence(bound).simulate(tlist=jnp.linspace(0, t, 21), initial_state={"a": 1},
                                                 dissipation=False)
        assert result.stats["excitation_reduction"]["dimension"] == 2
        return result.population("b", 1)[-1]

    def exchange(frequency):
        # One excitation shared by two detuned modes oscillates at the Rabi rate.
        rate = jnp.sqrt(g**2 + ((frequency - 0.25) / 2) ** 2)
        return g**2 / rate**2 * jnp.sin(2 * jnp.pi * rate * t) ** 2

    value, gradient = jax.jit(jax.value_and_grad(population))(0.26)
    # The default Dopri8 rtol of 1e-9 keeps both values far inside these bounds.
    np.testing.assert_allclose(value, exchange(0.26), atol=2e-6)
    np.testing.assert_allclose(gradient, jax.grad(exchange)(0.26), atol=2e-4)
    assert abs(float(gradient)) > 1.0


@pytest.mark.validation
@pytest.mark.optional_backend
@pytest.mark.parametrize("model", ["ladder", "energy basis"])
def test_zero_thermal_occupation_keeps_the_relaxation_gradient(model):
    """An explicit zero occupation keeps dP1/dT1 at its analytic value."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    from quchip import ChargeBasisTransmon

    def population(T1):
        noise = dict(label="q", T1=T1, thermal_occupation=0.0)
        if model == "ladder":
            device = Resonator(freq=0.25, levels=3, **noise)
        else:
            device = ChargeBasisTransmon(E_C=0.25, E_J=12.5, levels=3, coupling_channel="charge", **noise)
        chip = Chip([device], frame="rotating", backend="dynamiqs")
        result = QuantumSequence(chip).simulate(tlist=jnp.linspace(0, 4, 9), initial_state={"q": 1})
        return result.population("q", 1)[-1]

    gradient = jax.grad(population)(jnp.asarray(20.0))
    # Level 1 decays only to level 0, so P1 = exp(-t/T1) and dP1/dT1 = t exp(-t/T1) / T1**2.
    # The default Dopri8 rtol of 1e-9 keeps the gradient far inside 1e-6.
    np.testing.assert_allclose(gradient, 4 / 20**2 * np.exp(-4 / 20), rtol=1e-6)


@pytest.mark.parametrize("kind", ["intrinsic", "pump"])
def test_flux_controls_intrinsic_terms_and_frames_match_full_solver(kind, monkeypatch):
    """The shared compiler preserves delivered signals, intrinsic terms, and unequal-frame phases."""
    from quchip import (
        ControlEquipment,
        DuffingTransmon,
        Exact,
        FluxDrive,
        ParametricDrive,
        SquareWithGaussianEdges,
        TunableCapacitive,
    )
    from quchip.control.signal import Crosstalk, Delay, Gain
    from quchip.declarative import CouplingModel, CosineCoefficient, Scalar, TimeDependentTerm, parameter

    class ModulatedExchange(CouplingModel):
        g: Scalar = parameter(unit="GHz")

        def interaction(self, a, b, p):
            return p.g * (a.adag * b.a + a.a * b.adag)

        def time_terms(self, a, b, p):
            return (TimeDependentTerm(a.adag * b.a + a.a * b.adag, CosineCoefficient(0.005, 0.08)),)

    q = DuffingTransmon(freq=0.2, anharmonicity=-0.015, levels=3, label="q")
    r = DuffingTransmon(freq=0.24, anharmonicity=-0.012, levels=3, label="r")
    edge = (
        ModulatedExchange(q, r, g=0.012, label="edge")
        if kind == "intrinsic"
        else TunableCapacitive(q, r, g_0=0.012, label="edge")
    )
    chip = Chip([q, r], [edge], approximation=Exact() if kind == "intrinsic" else RWA(), frame={"q": 0.2, "r": 0.24})
    lines = [FluxDrive(q, label="zq"), FluxDrive(r, label="zr")]
    if kind == "pump":
        lines.append(ParametricDrive(edge, label="pump"))
    chip.connect(
        ControlEquipment(
            lines,
            signal_chain=[Gain("zq", 0.7), Delay("zq", 0.3), Crosstalk(source="zq", victim="zr", beta=0.2, delay=0.4)],
        )
    )
    seq = QuantumSequence(chip)
    seq.schedule("zq", envelope=SquareWithGaussianEdges(duration=4, amplitude=0.025), start_time=1)
    if kind == "pump":
        seq.pump(
            edge, envelope=SquareWithGaussianEdges(duration=3, amplitude=0.008), freq=0.04, phase=0.31, start_time=2
        )
    times = np.linspace(0.5, 7, 81)
    kwargs = dict(
        tlist=times,
        initial_state={"q": 1},
        dissipation=False,
        e_ops={"q": q.number_operator()},
        options={"atol": 1e-11, "rtol": 1e-10},
    )
    actual = seq.simulate(**kwargs)
    assert actual.stats["excitation_reduction"]["dimension"] == 2
    monkeypatch.setattr("quchip.engine.pulse_space.plan_pulse_space",
                        lambda *a, **k: (None, "full test reference", None))
    reference = seq.simulate(**kwargs)
    # Both solves use identical waveform lowering. Errors come only from integration at rtol 1e-10.
    np.testing.assert_allclose(actual.expect("q"), reference.expect("q"), atol=2e-7)
    np.testing.assert_allclose(
        chip.backend.to_array(actual.final_state), chip.backend.to_array(reference.final_state), atol=2e-7
    )


def test_native_energy_maps_preserve_local_queries_and_exports(monkeypatch):
    """A non-Fock native basis projects and lifts through captured isolated eigenvectors."""
    from quchip import ChargeBasisTransmon

    q = ChargeBasisTransmon(E_C=0.25, E_J=12, n_g=0.1, num_basis=7, basis="native", label="q")
    r = Resonator(freq=5.3, levels=3, label="r")
    chip = Chip([q, r], [Capacitive(q, r, g=0.03)], frame="lab", approximation=RWA())
    seq = QuantumSequence(chip)
    kwargs = dict(
        tlist=np.linspace(0, 0.2, 11),
        initial_state={"q": 1},
        dissipation=False,
        e_ops={"q": q.charge_coupling_operator()},
        options={"atol": 1e-11, "rtol": 1e-10},
    )
    actual = seq.simulate(**kwargs)
    assert actual.stats["excitation_reduction"]["dimension"] == 2
    monkeypatch.setattr("quchip.engine.pulse_space.plan_pulse_space",
                        lambda *a, **k: (None, "full test reference", None))
    reference = seq.simulate(**kwargs)
    # Both solves integrate at rtol 1e-10. Errors come only from integration.
    np.testing.assert_allclose(actual.population("q", 1), reference.population("q", 1), atol=2e-7)
    np.testing.assert_allclose(actual.expect("q"), reference.expect("q"), atol=2e-7)
    np.testing.assert_allclose(actual.measure("q").probabilities, reference.measure("q").probabilities, atol=2e-7)
    np.testing.assert_allclose(
        chip.backend.to_array(actual.reduced_state(0.2, "q")),
        chip.backend.to_array(reference.reduced_state(0.2, "q")),
        atol=2e-7,
    )
    np.testing.assert_allclose(
        chip.backend.to_array(actual.final_state), chip.backend.to_array(reference.final_state), atol=2e-7
    )


def test_pulse_space_operator_is_the_block_matrix_of_its_sectors():
    """A pulse-space operator equals the blocks of the same operator between its sectors."""
    from quchip import ChargeBasisTransmon
    from quchip.engine.assembly import _energy_operator, _resolve_system
    from quchip.engine.pulse_space import PulseSpace
    from quchip.engine.sectors import SectorModel

    q = ChargeBasisTransmon(E_C=0.25, E_J=12, n_g=0.0, num_basis=7, basis="native", label="q")
    r = Resonator(freq=5.3, levels=3, label="r")
    chip = Chip([q, r], [Capacitive(q, r, g=0.03)], approximation=RWA())
    resolution = _resolve_system(chip, chip.backend)
    totals = (0, 1, 2)
    space = PulseSpace.create(tuple(resolution.dims), ("q", "r"), totals, resolution.bases)
    model = SectorModel(chip, resolution=resolution)
    rng = np.random.default_rng(1)
    for support, dims in (((0,), [7]), ((0, 1), [7, 3])):
        size = int(np.prod(dims))
        local = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
        operator = chip.backend.from_array(local, dims=[dims, dims])
        pulse = np.asarray(chip.backend.to_array(space.embed(chip.backend, operator, support)))
        labels = tuple(chip.devices[index].label for index in support)
        energy = _energy_operator(operator, labels, resolution.bases, chip.backend)
        blocks = np.block([[model.block(energy, support, row, column) for column in totals] for row in totals])
        np.testing.assert_array_equal(pulse, blocks)


def test_counter_rotating_explicit_state_uses_parity_not_low_excitation_cutoff(monkeypatch):
    """Exact pair terms reach every odd sector, including three excitations from one."""
    from quchip import Exact

    chip = pair()
    chip = Chip(chip.devices, chip.couplings, approximation=Exact(), frame="lab")
    seq = QuantumSequence(chip)
    actual = seq.simulate(tlist=np.linspace(0, 3, 31), initial_state={"a": 1}, dissipation=False)
    assert actual.stats["excitation_reduction"]["totals"] == (1, 3)
    assert np.max(actual.population("b", 2)) > 1e-5
    monkeypatch.setattr("quchip.engine.pulse_space.plan_pulse_space",
                        lambda *a, **k: (None, "full test reference", None))
    reference = seq.simulate(tlist=np.linspace(0, 3, 31), initial_state={"a": 1}, dissipation=False)
    # QuTiP diagonalizes both static generators, so only round-off separates the routes.
    np.testing.assert_allclose(
        chip.backend.to_array(actual.final_state), chip.backend.to_array(reference.final_state), atol=2e-6
    )


@pytest.mark.validation
@pytest.mark.optional_backend
def test_compact_state_pytree_and_initial_state_batch():
    """Compact state leaves survive JIT and a homogeneous initial-state batch uses their union."""
    import jax
    import jax.numpy as jnp

    chip = pair(backend="dynamiqs")
    state = chip.compact_state({"a": 1})

    @jax.jit
    def evolve(initial):
        result = QuantumSequence(chip).simulate(tlist=jnp.linspace(0, 2, 11), initial_state=initial, dissipation=False)
        return result.compact_state()

    evolved = evolve(state)
    np.testing.assert_allclose(jnp.linalg.norm(evolved.values), 1, atol=2e-6)
    sequence = QuantumSequence(chip)
    batch = sequence.build_batch(sequence.vary("initial_state", [{}, {"a": 1}]), tlist=[0, 1], dissipation=False)
    assert all(p.engine_result.excitation_space.totals == (0, 1) for p in batch)
    result = chip.solve_many(batch, progress=False)
    np.testing.assert_allclose(result[0].population("a", 1), 0, atol=1e-10)
    assert result[1].population("a", 1)[-1] > 0.9


def test_per_point_batch_proves_one_space_from_all_initial_states(monkeypatch):
    """Duration and initial-state points share one exact space and match the product route."""
    from quchip import DuffingTransmon, FluxDrive, Square

    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.25, levels=3, label="q1")
    chip = Chip([q0, q1], couplings=[Capacitive(q0, q1, g=0.01)], approximation=RWA(), frame="rotating")
    chip.wire(FluxDrive(q0, label="z"))
    sequence = QuantumSequence(chip)
    pulse = sequence.schedule("z", envelope=Square(duration=2.0, amplitude=0.01))
    axis = sequence.zip(
        pulse.vary("duration", [2.0, 4.0]), sequence.vary("initial_state", [{"q0": 1}, {"q0": 1, "q1": 1}])
    )
    options = {"atol": 1e-11, "rtol": 1e-10}
    batch = sequence.build_batch(axis, dissipation=False, options=options)
    assert [problem.engine_result.excitation_space.totals for problem in batch] == [(1, 2), (1, 2)]
    actual = chip.solve_many(batch, progress=False)
    monkeypatch.setattr("quchip.engine.pulse_space.plan_pulse_space",
                        lambda *a, **k: (None, "full test reference", None))
    reference = chip.solve_many(sequence.build_batch(axis, dissipation=False, options=options), progress=False)
    for compact, full in zip(actual, reference):
        # Automatic grids can differ, but each pulse ends at the same time. Both routes integrate
        # the same equations at rtol 1e-10, so 2e-7 bounds their difference.
        assert compact.times[-1] == full.times[-1]
        for label in ("q0", "q1"):
            np.testing.assert_allclose(compact.population(label, 1)[-1], full.population(label, 1)[-1], atol=2e-7)


def test_compact_dressed_state_matches_full_assigned_state():
    """Compact dressed preparation gives the same physical state as the full labeled eigensystem."""
    chip = pair()
    chip.set_frame("lab")
    state = chip.compact_state({"a": 1}, dressed=True)
    result = QuantumSequence(chip).simulate(tlist=[0, 1], initial_state=state, dissipation=False)
    np.testing.assert_allclose(result.overlap(chip.state(a=1)), 1, atol=2e-7)


@pytest.mark.unit
def test_large_nonconserving_pulse_refuses_before_global_allocation(monkeypatch):
    """An ineligible large drive raises a memory error before full tensor materialization."""
    from quchip import ChargeDrive, Square
    from quchip.utils import memory as _memory

    # A fixed host memory keeps the assembly guard active on every platform.
    monkeypatch.setattr(_memory, "available_memory_bytes", lambda: 2**33)
    devices = [Resonator(freq=0.25, levels=3, label=f"r{i}") for i in range(24)]
    chip = Chip(devices)
    chip.wire(ChargeDrive(devices[0], label="xy"))
    sequence = QuantumSequence(chip)
    sequence.schedule("xy", envelope=Square(duration=1, amplitude=0), freq=0.25)
    with pytest.raises(MemoryError, match="Product-space operator assembly"):
        sequence.build_problem([0, 1], initial_state={})


@pytest.mark.unit
def test_full_pulse_route_counts_stored_operator_entries(monkeypatch):
    """A transverse pulse builds sparse product operators that dense estimates would refuse."""
    from quchip import ChargeDrive, Square
    from quchip.utils import memory as _memory

    devices = [Resonator(freq=0.25, levels=3, label=f"r{i}") for i in range(6)]
    chain = [Capacitive(a, b, g=0.01) for a, b in zip(devices, devices[1:])]
    chip = Chip(devices, couplings=chain, approximation=RWA())
    chip.wire(ChargeDrive(devices[0], label="xy"))
    sequence = QuantumSequence(chip)
    sequence.schedule("xy", envelope=Square(duration=1, amplitude=0.01), freq=0.25)
    # Six dense 729-state matrices need 51 MB. Each sparse operator and one state need less than 1 MB.
    monkeypatch.setattr(_memory, "available_memory_bytes", lambda: 10**7)
    problem = sequence.build_problem([0, 1], initial_state={})
    assert problem.engine_result.excitation_space is None
    assert problem.engine_result.dims == (3,) * 6


@pytest.mark.parametrize("backend, dimension",
                         [("qutip", 40), pytest.param("dynamiqs", 81, marks=pytest.mark.optional_backend)])
def test_low_memory_moves_only_dense_compact_operators_to_the_product_route(monkeypatch, backend, dimension):
    """QuTiP holds compact operators as CSR; dense compact operators that cannot fit solve on the product route."""
    from quchip import Exact
    from quchip.utils import memory as _memory

    if backend == "dynamiqs":
        dq = pytest.importorskip("dynamiqs")
        options = {"method": dq.method.Tsit5(atol=1e-11, rtol=1e-10)}
    else:
        options = {"atol": 1e-11, "rtol": 1e-10}
    devices = [Resonator(freq=0.25, levels=3, label=f"r{i}") for i in range(4)]
    chain = [Capacitive(a, b, g=0.01) for a, b in zip(devices, devices[1:])]
    chip = Chip(devices, couplings=chain, approximation=Exact(), backend=backend)
    sequence = QuantumSequence(chip)
    kwargs = dict(tlist=np.linspace(0, 3, 7), initial_state={"r0": 1}, dissipation=False, options=options)
    compact = sequence.simulate(**kwargs)
    assert compact.stats["excitation_reduction"]["dimension"] == 40
    # Dense operators on the 40 odd-parity states need 206 kB. Each sparse operator needs less than 20 kB.
    monkeypatch.setattr(_memory, "available_memory_bytes", lambda: 10**5)
    limited = sequence.simulate(**kwargs)
    assert limited.stats["excitation_reduction"]["dimension"] == dimension
    # Both routes integrate the same equations at rtol 1e-10, so 2e-7 bounds their difference.
    np.testing.assert_allclose(limited.population("r3", 1), compact.population("r3", 1), atol=2e-7)


def test_large_partitioned_pulse_keeps_local_results_and_guards_joint_export(monkeypatch):
    """Independent components remain exact without allocating their full joint vacuum."""
    from quchip.utils import memory as _memory

    # A fixed host memory keeps the export guard active on every platform.
    monkeypatch.setattr(_memory, "available_memory_bytes", lambda: 2**33)
    chip = Chip([Resonator(freq=0.25, levels=3, label=f"r{i}") for i in range(24)])
    result = QuantumSequence(chip).simulate(tlist=[0, 1])
    assert len(result.components) == 24
    np.testing.assert_allclose(result.population("r0", 0), 1, atol=1e-10)
    with pytest.warns(UserWarning, match="Materializing"), pytest.raises(MemoryError):
        _ = result.final_state


@pytest.mark.parametrize("full", [False, True])
def test_compact_replay_and_overlap_reject_changed_energy_bases(full, monkeypatch):
    """Saved compact coordinates cannot silently change meaning after local model edits."""
    from quchip import ChargeBasisTransmon

    def model(energy):
        q = ChargeBasisTransmon(E_C=0.25, E_J=energy, n_g=0.1, num_basis=7, basis="native", label="q")
        r = Resonator(freq=5.3, levels=3, label="r")
        return Chip([q, r], [Capacitive(q, r, g=0.03)], frame="lab", approximation=RWA())

    original, changed = model(12), model(13)
    state = original.compact_state({"q": 1})
    if full:
        monkeypatch.setattr("quchip.engine.pulse_space.plan_pulse_space",
                            lambda *a, **k: (None, "full test reference", None))
    result = QuantumSequence(original).simulate(tlist=[0, 0.01], initial_state=state, dissipation=False)
    np.testing.assert_allclose(result.overlap(state)[0], 1, atol=1e-10)
    with pytest.raises(ValueError, match="energy bases changed"):
        result.overlap(changed.compact_state({"q": 1}))
    with pytest.raises(ValueError, match="energy bases changed"):
        QuantumSequence(changed).build_problem([0, 0.01], initial_state=state, dissipation=False)
