"""Exact pulse spaces preserve the complete generator and captured state semantics."""

import numpy as np
import pytest

from quchip import RWA, Capacitive, Chip, QuantumSequence, Resonator


def pair(*, loss=False, backend=None):
    a = Resonator(freq=0.25, levels=3, label="a", **({"T1": 20.0} if loss else {}))
    b = Resonator(freq=0.25, levels=3, label="b")
    return Chip([a, b], couplings=[Capacitive(a, b, g=0.02)], approximation=RWA(), frame="rotating", backend=backend)


def test_exchange_uses_exact_space_and_exports_original_coordinates():
    """A one-excitation exchange evolves two amplitudes and retains tensor export semantics."""
    chip = pair()
    times = np.linspace(0, 10, 51)
    problem = QuantumSequence(chip).build_problem(times, initial_state={"a": 1}, dissipation=False, states="all")
    assert problem.engine_result.excitation_space.size == 2
    result = chip.solve(problem)
    np.testing.assert_allclose(result.population("a", 1), np.cos(2 * np.pi * 0.02 * times) ** 2, atol=2e-6)
    assert result.final_state.shape == (9, 1)
    assert result.dims == [3, 3]
    np.testing.assert_allclose(result.reduced_state(times[-1], "a").tr(), 1, atol=1e-8)


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
    np.testing.assert_allclose(np.sum(result.measure("a").probabilities), 1, atol=2e-6)
    resumed = seq.simulate(tlist=np.linspace(5, 10, 21), initial_state=result.compact_state(), states="final")
    reference = seq.simulate(tlist=np.linspace(0, 10, 41), initial_state=initial, states="final")
    np.testing.assert_allclose(
        chip.backend.to_array(resumed.final_state), chip.backend.to_array(reference.final_state), atol=3e-6
    )
    assert result.check_truncation()


@pytest.mark.parametrize("loss", [False, True])
@pytest.mark.parametrize("frame", ["lab", "auto"])
def test_large_ring_never_embeds_full_operators(loss, frame, monkeypatch):
    """Twenty-four three-level devices evolve in 24 amplitudes with guarded exports."""
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
    for i in range(24):
        np.testing.assert_allclose(
            result.population(f"r{i}", 1)[-1], abs(expected[i]) ** 2 * (np.exp(-2 / 20) if loss else 1), atol=2e-6
        )
    saved = result.compact_state()
    indices = [saved.space.rows[tuple(int(j == i) for j in range(24))] for i in range(24)]
    values = np.asarray(saved.values)
    density = values @ values.conj().T if values.shape[-1] == 1 else values
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
    np.testing.assert_allclose(result.population("a", 1) + result.population("b", 1), expected, atol=2e-6)


def test_passive_cascade_matches_full_model(monkeypatch):
    """Generated SLH Hamiltonians and summed output jumps agree with full tensor assembly."""
    from quchip import PortNetwork

    a, b = [Resonator(freq=0.25, levels=3, label=label) for label in ("a", "b")]
    network = PortNetwork()
    pa, pb = network.port("pa", target=a, rate=0.04), network.port("pb", target=b, rate=0.09)
    network.cascade(pa, pb)
    network.expose("line", input=pa.input, output=pb.output)
    chip = Chip([a, b], port_network=network, frame=0.25)
    seq = QuantumSequence(chip)
    times = np.linspace(0, 10, 31)
    actual = seq.simulate(tlist=times, initial_state={"a": 1})
    monkeypatch.setattr("quchip.engine.pulse_space.plan_pulse_space", lambda *a: (None, "test full reference"))
    reference = seq.simulate(tlist=times, initial_state={"a": 1})
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
    np.testing.assert_allclose(value, np.sin(2 * np.pi * g * 3) ** 2, atol=2e-6)
    np.testing.assert_allclose(gradient, 2 * np.pi * 3 * np.sin(4 * np.pi * g * 3), atol=2e-4)


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
    monkeypatch.setattr("quchip.engine.pulse_space.plan_pulse_space", lambda *a: (None, "full test reference"))
    reference = seq.simulate(**kwargs)
    # Both solves use identical waveform lowering; errors come only from integration.
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
    monkeypatch.setattr("quchip.engine.pulse_space.plan_pulse_space", lambda *a: (None, "full test reference"))
    reference = seq.simulate(**kwargs)
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


def test_counter_rotating_explicit_state_uses_parity_not_low_excitation_cutoff(monkeypatch):
    """Exact pair terms reach every odd sector, including three excitations from one."""
    from quchip import Exact

    chip = pair()
    chip = Chip(chip.devices, chip.couplings, approximation=Exact(), frame="lab")
    seq = QuantumSequence(chip)
    actual = seq.simulate(tlist=np.linspace(0, 3, 31), initial_state={"a": 1}, dissipation=False)
    assert actual.stats["excitation_reduction"]["totals"] == (1, 3)
    assert np.max(actual.population("b", 2)) > 1e-5
    monkeypatch.setattr("quchip.engine.pulse_space.plan_pulse_space", lambda *a: (None, "full test reference"))
    reference = seq.simulate(tlist=np.linspace(0, 3, 31), initial_state={"a": 1}, dissipation=False)
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


def test_compact_dressed_state_matches_full_assigned_state():
    """Compact dressed preparation gives the same physical state as the full labeled eigensystem."""
    chip = pair()
    chip.set_frame("lab")
    state = chip.compact_state({"a": 1}, dressed=True)
    result = QuantumSequence(chip).simulate(tlist=[0, 1], initial_state=state, dissipation=False)
    np.testing.assert_allclose(result.overlap(chip.state(a=1)), 1, atol=2e-7)


@pytest.mark.unit
def test_large_nonconserving_pulse_refuses_before_global_allocation():
    """An ineligible large drive raises a memory error before full tensor materialization."""
    from quchip import ChargeDrive, Square

    devices = [Resonator(freq=0.25, levels=3, label=f"r{i}") for i in range(24)]
    chip = Chip(devices)
    chip.wire(ChargeDrive(devices[0], label="xy"))
    sequence = QuantumSequence(chip)
    sequence.schedule("xy", envelope=Square(duration=1, amplitude=0), freq=0.25)
    with pytest.raises(MemoryError, match="Full pulse operator assembly"):
        sequence.build_problem([0, 1], initial_state={})


def test_large_partitioned_pulse_keeps_local_results_and_guards_joint_export():
    """Independent components remain exact without allocating their full joint vacuum."""
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
        monkeypatch.setattr("quchip.engine.pulse_space.plan_pulse_space", lambda *a: (None, "full test reference"))
    result = QuantumSequence(original).simulate(tlist=[0, 0.01], initial_state=state, dissipation=False)
    np.testing.assert_allclose(result.overlap(state)[0], 1, atol=1e-10)
    with pytest.raises(ValueError, match="energy bases changed"):
        result.overlap(changed.compact_state({"q": 1}))
    with pytest.raises(ValueError, match="energy bases changed"):
        QuantumSequence(changed).build_problem([0, 0.01], initial_state=state, dissipation=False)
