"""Dynamiqs backend behavior and solver coverage."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

pytestmark = pytest.mark.optional_backend

dynamiqs = pytest.importorskip("dynamiqs")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from quchip.backend.dynamiqs import DynamiqsBackend  # noqa: E402
from quchip.backend.protocol import Backend  # noqa: E402
from quchip.chip.chip import Chip  # noqa: E402
from quchip.devices.transmon.duffing import DuffingTransmon  # noqa: E402
from quchip.devices.resonator import Resonator  # noqa: E402
from quchip.chip.couplings import Capacitive  # noqa: E402
from quchip.control import ChargeDrive  # noqa: E402
from quchip.control.sequence import QuantumSequence  # noqa: E402
from quchip.control.envelopes import Square  # noqa: E402
from quchip.engine import simulate  # noqa: E402
from quchip.engine.ir import (  # noqa: E402
    CanonicalOperator,
    Carrier,
    Constant,
    DynamicTerm,
    HamiltonianDescription,
    ScalarModulation,
    Window,
)
from quchip.engine.stage1_frames import resolve_frame  # noqa: E402
from quchip.engine.stage2_assembly import build_hamiltonian_description  # noqa: E402
from quchip.engine.stage3_observables import decompose_eops  # noqa: E402


@pytest.fixture
def backend() -> DynamiqsBackend:
    """Return a fresh dynamiqs backend instance."""
    return DynamiqsBackend()


def test_dynamiqs_backend_enables_float64(backend: DynamiqsBackend) -> None:
    """Dynamiqs backend forces JAX float64 mode for physics accuracy."""
    assert jax.config.read("jax_enable_x64") is True
    assert backend.array_module is jnp


def test_protocol_methods_on_trivial_operators_and_states(backend: DynamiqsBackend) -> None:
    """Operator factories, algebra helpers, and state helpers match trivial analytics."""
    destroy = backend.to_array(backend.destroy(3))
    expected_destroy = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, np.sqrt(2.0)], [0.0, 0.0, 0.0]],
        dtype=complex,
    )
    npt.assert_allclose(np.asarray(destroy), expected_destroy, atol=1e-12)

    number = backend.number(3)
    npt.assert_allclose(np.asarray(backend.to_array(number)), np.diag([0.0, 1.0, 2.0]), atol=1e-12)
    npt.assert_allclose(np.asarray(backend.to_array(backend.identity(2))), np.eye(2), atol=1e-12)

    matrix = np.array([[1.0, 2.0j], [-2.0j, 3.0]], dtype=complex)
    op = backend.from_array(matrix)
    npt.assert_allclose(np.asarray(backend.to_array(op)), matrix, atol=1e-12)

    evals, estates = backend.eigenstates(number)
    npt.assert_allclose(np.asarray(evals), np.array([0.0, 1.0, 2.0]), atol=1e-12)
    assert len(estates) == 3
    npt.assert_allclose(abs(backend.overlap(estates[0], estates[0])), 1.0, atol=1e-12)
    npt.assert_allclose(abs(backend.overlap(estates[0], estates[1])), 0.0, atol=1e-12)

    psi = backend.basis(3, 1)
    rho = backend.state_to_dm(psi)
    assert backend.is_ket(psi) is True
    assert backend.is_ket(rho) is False
    npt.assert_allclose(backend.norm(psi), 1.0, atol=1e-12)
    npt.assert_allclose(backend.trace(rho), 1.0, atol=1e-12)


def test_sparse_canonical_roundtrip_preserves_dia_layout(backend: DynamiqsBackend) -> None:
    """Canonical-operator roundtrip on a sparse operator preserves the dia layout and values."""
    op = backend.destroy(4)
    canonical = backend.to_canonical_operator(op)
    rebuilt = backend.from_canonical_operator(canonical)

    assert canonical.layout == "dia"
    assert getattr(rebuilt, "layout", None) == dynamiqs.dia
    npt.assert_allclose(np.asarray(backend.to_array(rebuilt)), np.asarray(backend.to_array(op)), atol=1e-12)


def test_dense_canonical_roundtrip_stays_dense(backend: DynamiqsBackend) -> None:
    """Canonical-operator roundtrip on a dense operator preserves the dense layout and values."""
    op = backend.from_array(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=complex))
    canonical = backend.to_canonical_operator(op)
    rebuilt = backend.from_canonical_operator(canonical)

    assert canonical.layout == "dense"
    assert getattr(rebuilt, "layout", None) == dynamiqs.dense
    npt.assert_allclose(np.asarray(backend.to_array(rebuilt)), np.asarray(backend.to_array(op)), atol=1e-12)


def test_tensor_partial_trace_and_coherent_state_helpers(backend: DynamiqsBackend) -> None:
    """Tensor-state helpers and partial trace work on analytically trivial product states."""
    psi = backend.tensor_states(backend.basis(2, 1), backend.basis(3, 2))
    rho = backend.state_to_dm(psi)
    reduced = backend.ptrace(rho, 0, [2, 3])

    expected = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    npt.assert_allclose(np.asarray(backend.to_array(reduced)), expected, atol=1e-12)

    coherent = backend.coherent(8, 0.0)
    npt.assert_allclose(abs(backend.overlap(coherent, backend.basis(8, 0))), 1.0, atol=1e-12)


def test_sesolve_trivial_and_helper_accessors(backend: DynamiqsBackend) -> None:
    """Trivial sesolve preserves the ground state and exposes arrays/scalars cleanly."""
    H = backend.from_array(np.zeros((2, 2), dtype=complex))
    psi0 = backend.basis(2, 0)
    tlist = jnp.linspace(0.0, 1.0, 5)

    result = backend.sesolve(H, psi0, tlist, e_ops=[backend.number(2)])

    assert result.solver == "sesolve"
    assert result.states is not None
    assert len(result.states) == len(tlist)
    npt.assert_allclose(np.asarray(result.times), np.asarray(tlist), atol=1e-12)
    npt.assert_allclose(np.asarray(result.expect[0]), np.zeros(len(tlist)), atol=1e-12)
    for state in result.states:
        npt.assert_allclose(abs(backend.overlap(state, psi0)) ** 2, 1.0, atol=1e-12)


def test_mesolve_trivial_open_system_run(backend: DynamiqsBackend) -> None:
    """Trivial mesolve preserves the ground-state density matrix and keeps unit trace."""
    H = backend.from_array(np.zeros((2, 2), dtype=complex))
    rho0 = backend.state_to_dm(backend.basis(2, 0))
    tlist = jnp.linspace(0.0, 1.0, 4)
    c_ops = [np.sqrt(0.1) * backend.destroy(2)]

    result = backend.mesolve(H, rho0, tlist, c_ops=c_ops, e_ops=[backend.number(2)])

    assert result.solver == "mesolve"
    assert result.states is not None
    npt.assert_allclose(np.asarray(result.expect[0]), np.zeros(len(tlist)), atol=1e-12)
    npt.assert_allclose(backend.trace(result.final_state), 1.0, atol=1e-12)


def test_run_simulation_smoke_path_matches_runtime_expectations() -> None:
    """Dynamiqs backend works through simulate()/SimulationResult on a trivial chip."""
    backend = DynamiqsBackend()
    chip = Chip(
        devices=[DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")],
        backend=backend,
    )
    tlist = jnp.linspace(0.0, 1.0, 5)
    initial_state = chip.bare_state(q=0)

    result = simulate(chip, [], tlist, initial_state=initial_state)

    overlap = result.overlap_array(initial_state)
    npt.assert_allclose(np.asarray(overlap), np.ones(len(tlist)), atol=1e-9)
    assert result.states is not None
    assert len(result.states) == len(tlist)


def test_prepare_static_hamiltonian_preserves_sparse_layout() -> None:
    """prepare_hamiltonian on a static Hamiltonian keeps the rhs operator in dia layout."""
    backend = DynamiqsBackend()
    chip = Chip(
        devices=[DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")],
        backend=backend,
    )
    chip.dress()
    tlist = np.linspace(0.0, 10.0, 11)
    resolved = resolve_frame(chip, chip.frame)
    desc = build_hamiltonian_description(chip, [], resolved_frame=resolved)
    prepared = backend.prepare_hamiltonian(desc, tlist)

    assert getattr(prepared.rhs, "layout", None) == dynamiqs.dia


def test_prepare_driven_hamiltonian_remains_callable() -> None:
    """prepare_hamiltonian on a driven Hamiltonian returns a callable rhs producing an array-like sample."""
    backend = DynamiqsBackend()
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    drive = ChargeDrive(target=qubit)
    chip = Chip([qubit], backend=backend)
    from quchip.control.equipment import ControlEquipment
    chip.connect(ControlEquipment(lines=[drive]))
    chip.dress()
    tlist = np.linspace(0.0, 10.0, 11)
    resolved = resolve_frame(chip, chip.frame)

    from quchip.engine.ir import DriveOp

    drive_op = DriveOp(
        target_label="q",
        envelope=Square(duration=10.0, amplitude=0.01),
        freq=chip.freq(qubit),
        start_time=0.0,
        drive_label=drive.label,
    )

    desc = build_hamiltonian_description(chip, [drive_op], resolved_frame=resolved)
    assert desc.dynamic_terms
    assert all(isinstance(term.time_dependence, ScalarModulation) for term in desc.dynamic_terms)

    prepared = backend.prepare_hamiltonian(desc, tlist)
    assert callable(prepared.rhs)
    sample = prepared.rhs(0.0)
    assert hasattr(sample, "shape")


def test_prepare_scalar_modulation_hamiltonian_remains_callable() -> None:
    """prepare_hamiltonian on a scalar-modulation term yields a callable rhs producing an array-like sample."""
    backend = DynamiqsBackend()
    tlist = np.linspace(0.0, 10.0, 11)
    op = CanonicalOperator.from_dense(
        np.eye(2, dtype=complex),
        dims=(2,),
        basis="fock",
        subsystem_labels=("q",),
    )
    desc = HamiltonianDescription(
        static_terms=(),
        dynamic_terms=(
            DynamicTerm(
                operator=op,
                time_dependence=ScalarModulation(signal=Carrier(freq=0.5)),
                origin="drive",
            ),
        ),
        dims=(2,),
        metadata={},
    )
    prepared = backend.prepare_hamiltonian(desc, tlist)
    assert callable(prepared.rhs)
    sample = prepared.rhs(0.0)
    assert hasattr(sample, "shape")


def test_prepare_windowed_scalar_modulation_supports_traced_stop_time() -> None:
    """A windowed scalar modulation's traced stop time works under jax.jit and evaluates within the window."""
    backend = DynamiqsBackend()
    op = CanonicalOperator.from_dense(
        np.eye(2, dtype=complex),
        dims=(2,),
        basis="fock",
        subsystem_labels=("q",),
    )

    @jax.jit
    def sample_rhs(stop):
        desc = HamiltonianDescription(
            static_terms=(),
            dynamic_terms=(
                DynamicTerm(
                    operator=op,
                    time_dependence=ScalarModulation(
                        signal=Window(
                            child=Constant(1.0 + 0.0j),
                            start=0.0,
                            stop=stop,
                        )
                    ),
                    origin="drive",
                ),
            ),
            dims=(2,),
            metadata={},
        )
        prepared = backend.prepare_hamiltonian(desc, np.linspace(0.0, 1.0, 5))
        return backend.to_array(prepared.rhs(0.5))

    sample = np.asarray(sample_rhs(jnp.asarray(1.0)))
    npt.assert_allclose(sample, np.eye(2, dtype=complex), atol=1e-12)


def test_dict_eops_preserve_sparse_layout() -> None:
    """decompose_eops preserves the dia layout for dict-form expectation operators."""
    backend = DynamiqsBackend()
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3, label="q")
    resonator = Resonator(freq=6.8, levels=5, label="r", quality_factor=1e6)
    chip = Chip([qubit, resonator], [Capacitive(qubit, resonator, g=0.04, rwa=True)], backend=backend)

    e_ops_solver, _ = decompose_eops(chip.e_ops(r="a"), chip, backend)

    assert e_ops_solver is not None
    assert all(getattr(op, "layout", None) == dynamiqs.dia for op in e_ops_solver)


def test_chip_hamiltonian_emits_no_sparse_dense_warning() -> None:
    """chip.hamiltonian() does not trigger dynamiqs's sparse-to-dense layout conversion warning."""
    backend = DynamiqsBackend()
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3, label="q")
    resonator = Resonator(freq=6.8, levels=5, label="r", quality_factor=1e6)
    chip = Chip([qubit, resonator], [Capacitive(qubit, resonator, g=0.04, rwa=True)], backend=backend)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        chip.hamiltonian()

    assert not [w for w in captured if "sparse qarray has been converted to dense layout" in str(w.message)]


def test_rotating_frame_assembly_emits_no_sparse_dense_warning() -> None:
    """build_hamiltonian_description in the rotating frame does not trigger the sparse-to-dense warning."""
    backend = DynamiqsBackend()
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3, label="q")
    resonator = Resonator(freq=6.8, levels=5, label="r", quality_factor=1e6)
    chip = Chip(
        [qubit, resonator],
        [Capacitive(qubit, resonator, g=0.04, rwa=True)],
        frame="rotating",
        backend=backend,
    )
    chip.dress()
    resolved = resolve_frame(chip, chip.frame)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        build_hamiltonian_description(chip, [], resolved_frame=resolved)

    assert not [w for w in captured if "sparse qarray has been converted to dense layout" in str(w.message)]


def test_batched_sesolve_handles_heterogeneous_problems_sequentially(backend: DynamiqsBackend) -> None:
    """The inherited dict-form batched API solves each heterogeneous problem independently."""
    # The engine's native vmap path lives in solve_batch (typed SolveBatch, structurally
    # homogeneous by construction); batched_sesolve is the protocol's sequential default.
    H0 = backend.from_array(np.zeros((2, 2), dtype=complex))
    H1 = backend.from_array(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex))
    problems = [
        {"H": H0, "psi0": backend.basis(2, 0), "tlist": jnp.linspace(0.0, 1.0, 5),
         "e_ops": [backend.number(2)]},
        {"H": H1, "psi0": backend.basis(2, 1), "tlist": jnp.linspace(0.0, 2.0, 5),
         "e_ops": [backend.identity(2)]},
    ]

    batched = backend.batched_sesolve(problems, progress=False)
    separate = [backend.sesolve(**problem) for problem in problems]

    assert len(batched) == len(separate) == 2
    for batch_result, single_result in zip(batched, separate):
        npt.assert_allclose(
            np.asarray(backend.to_array(batch_result.final_state)),
            np.asarray(backend.to_array(single_result.final_state)),
            atol=1e-12,
        )
        npt.assert_allclose(np.asarray(batch_result.expect[0]), np.asarray(single_result.expect[0]), atol=1e-12)


def test_batched_sesolve_store_states_false_preserves_final_state(backend: DynamiqsBackend) -> None:
    """batched_sesolve with store_states=False still returns a final_state and a single-entry states list."""
    H = backend.from_array(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex))
    tlist = jnp.linspace(0.0, 1.0, 5)
    problems = [
        {"H": H, "psi0": backend.basis(2, 0), "tlist": tlist, "options": {"store_states": False}},
        {"H": H, "psi0": backend.basis(2, 1), "tlist": tlist, "options": {"store_states": False}},
    ]

    results = backend.batched_sesolve(problems, progress=False)

    assert len(results) == 2
    assert all(result.final_state is not None for result in results)
    assert all(result.states is not None and len(result.states) == 1 for result in results)


def test_batched_mesolve_matches_separate_solves(backend: DynamiqsBackend) -> None:
    """The inherited dict-form batched mesolve matches independent solves."""
    H = backend.from_array(np.zeros((2, 2), dtype=complex))
    tlist = jnp.linspace(0.0, 1.0, 4)
    problems = [
        {
            "H": H,
            "rho0": backend.state_to_dm(backend.basis(2, 0)),
            "tlist": tlist,
            "c_ops": [np.sqrt(0.1) * backend.destroy(2)],
            "e_ops": [backend.number(2)],
        },
        {
            "H": H,
            "rho0": backend.state_to_dm(backend.basis(2, 1)),
            "tlist": tlist,
            "c_ops": [np.sqrt(0.1) * backend.destroy(2)],
            "e_ops": [backend.number(2)],
        },
    ]

    batched = backend.batched_mesolve(problems, progress=False)
    separate = [backend.mesolve(**problem) for problem in problems]

    assert len(batched) == len(separate) == 2
    for batch_result, single_result in zip(batched, separate):
        npt.assert_allclose(
            np.asarray(backend.to_array(batch_result.final_state)),
            np.asarray(backend.to_array(single_result.final_state)),
            atol=1e-12,
        )
        npt.assert_allclose(np.asarray(batch_result.expect[0]), np.asarray(single_result.expect[0]), atol=1e-12)


def test_frequency_sweep_lowers_to_single_native_batched_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A carrier-frequency sweep sharing operator structure fans out as a single native-batched call."""
    # build_batch over pulse.vary("freq", ...) produces one SolveBatch whose
    # BatchedHamiltonianDescription carries a shared operator skeleton and per-element
    # ScalarModulation signals; the dynamiqs backend lowers it to one vmapped RHS via
    # prepare_batch and solves it in one call to solve_batch.
    from quchip.control.envelopes import Square as SquareEnv

    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.30, levels=3, label="q")
    drive = ChargeDrive(target=qubit, label="d")
    chip = Chip(devices=[qubit], frame="rotating", backend="dynamiqs", label="structural-test")
    chip.wire(drive)

    psi0 = chip.state(q=0)
    e_ops = chip.e_ops(q="n")
    tlist = np.linspace(0.0, 40.0, 80)

    seq = QuantumSequence(chip)
    pulse = seq.charge(qubit, envelope=SquareEnv(duration=40.0, amplitude=0.03), freq=5.0)
    freq_axis = pulse.vary("freq", [4.9, 5.0, 5.1], name="freq")

    backend = chip.backend
    original_solve_batch = backend.solve_batch
    original_prepare_batch = backend.prepare_batch
    solve_batch_sizes: list[int] = []
    prepare_batch_sizes: list[int] = []

    def counted_solve_batch(batch, *, progress=True):
        solve_batch_sizes.append(batch.batch_size)
        return original_solve_batch(batch, progress=progress)

    def counted_prepare_batch(description, tlist_):
        prepare_batch_sizes.append(description.batch_size)
        return original_prepare_batch(description, tlist_)

    monkeypatch.setattr(backend, "solve_batch", counted_solve_batch)
    monkeypatch.setattr(backend, "prepare_batch", counted_prepare_batch)

    problems = seq.build_batch(
        freq_axis,
        tlist=tlist,
        initial_state=psi0,
        e_ops=e_ops,
    )
    results = chip.solve_many(problems, progress=False)

    assert len(results) == 3
    assert solve_batch_sizes == [3], f"Expected single batch of 3, got {solve_batch_sizes}"
    # Dynamiqs prepares the whole batch once via prepare_batch (native vmap);
    # it never falls back to the per-element prepare_hamiltonian path.
    assert prepare_batch_sizes == [3]

    finals = [float(jnp.real(r.expect_final("q"))) for r in results]
    assert not np.allclose(finals[0], finals[1], atol=1e-3) or not np.allclose(finals[1], finals[2], atol=1e-3)


def test_dynamiqs_method_from_dict_accepts_nsteps_alias(backend: DynamiqsBackend) -> None:
    """_method_from_dict should accept 'nsteps' as an alias for 'max_steps'."""
    method = backend._method_from_dict({"nsteps": 1234})
    assert method is not None
    assert getattr(method, "max_steps", None) == 1234


def test_dynamiqs_options_from_dict_accepts_progress_bar_alias(backend: DynamiqsBackend) -> None:
    """_options_from_dict should accept 'progress_bar' as an alias for 'progress_meter'."""
    options = backend._options_from_dict({"progress_bar": True})
    assert getattr(options, "progress_meter", None) is True


def test_solve_batch_respects_quchip_option_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    """quchip option aliases (progress_bar, nsteps) reach dynamiqs options/method on the batched lane."""
    # Dynamiqs maps progress_bar -> progress_meter and nsteps -> max_steps via
    # _options_from_dict/_method_from_dict; intercept those calls to verify the aliases
    # actually arrive on the solve_batch lane, not just the per-problem lane.
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.30, levels=3, label="q")
    drive = ChargeDrive(target=qubit, label="d")
    chip = Chip(devices=[qubit], frame="rotating", backend="dynamiqs", label="solve-batch-options")
    chip.wire(drive)

    seq = QuantumSequence(chip)
    pulse = seq.charge(qubit, envelope=Square(duration=20.0, amplitude=0.03), freq=5.0)
    freq_axis = pulse.vary("freq", [4.95, 5.05], name="freq")

    problems = seq.build_batch(
        freq_axis,
        tlist=np.linspace(0.0, 20.0, 41),
        initial_state=chip.state(q=0),
        e_ops=chip.e_ops(q="n"),
        options={"progress_bar": False, "nsteps": 2048},
    )

    backend = chip.backend
    orig_options = backend._options_from_dict
    orig_method = backend._method_from_dict
    seen_options: list[dict] = []
    seen_method: list[dict] = []

    def capturing_options(raw: dict, *args: Any, **kwargs: Any) -> Any:
        seen_options.append(dict(raw))
        return orig_options(raw, *args, **kwargs)

    def capturing_method(raw: dict, *args: Any, **kwargs: Any) -> Any:
        seen_method.append(dict(raw))
        return orig_method(raw, *args, **kwargs)

    monkeypatch.setattr(backend, "_options_from_dict", capturing_options)
    monkeypatch.setattr(backend, "_method_from_dict", capturing_method)

    results = chip.solve_many(problems, progress=False)

    assert len(results) == 2
    # resolve_solver_options normalises progress_bar -> progress_meter and
    # nsteps -> max_steps before the dict reaches _options_from_dict /
    # _method_from_dict. Verify the normalised keys (with user values)
    # actually arrive at the dynamiqs option constructors on the batched
    # lane, not just the per-problem lane.
    assert any(raw.get("progress_meter") is False for raw in seen_options), (
        f"progress_bar alias did not reach _options_from_dict as progress_meter=False; "
        f"saw {seen_options!r}"
    )
    assert any(raw.get("max_steps") == 2048 for raw in seen_method), (
        f"nsteps alias did not reach _method_from_dict as max_steps=2048; "
        f"saw {seen_method!r}"
    )


# ---------------------------------------------------------------------------
# Cached jitted ``solve_problem`` (the dynamiqs single-solve artifact cache):
# parity vs the un-jitted protocol-default path, grad/vmap traceability, and
# adversarial cache-correctness (no structural collision, no stale-value reuse).
# ``benchmarks/repeated_solve_parity.py`` validated this manually; these are the
# make-test-lane regression guards for the optimization-loop hot path.
# ---------------------------------------------------------------------------
def _r5_build(open_system: bool):
    qubit = DuffingTransmon(
        freq=5.0, anharmonicity=-0.30, levels=3, label="q",
        T1=200.0 if open_system else None,
    )
    drive = ChargeDrive(target=qubit, label="d")
    chip = Chip(devices=[qubit], frame="rotating", backend="dynamiqs", label="r5")
    chip.wire(drive)
    return chip, qubit


def _r5_problem(chip, qubit, amp, tlist):
    seq = QuantumSequence(chip)
    seq.charge(qubit, envelope=Square(duration=40.0, amplitude=jnp.asarray(amp)), freq=5.0)
    return seq.build_problem(tlist=tlist, initial_state=chip.state(q=0), e_ops=chip.e_ops(q="n"))


def _r5_cached(problem):
    """The DynamiqsBackend override (cached jit path)."""
    return problem.chip.backend.solve_problem(problem)


def _r5_uncached(problem):
    """The un-jitted protocol-default reference (bypasses the override)."""
    return Backend.solve_problem(problem.chip.backend, problem)


@pytest.mark.parametrize("open_system", [False, True], ids=["sesolve", "mesolve"])
def test_cached_jit_solve_matches_uncached(open_system: bool) -> None:
    """The cached-jit solve_problem override matches the un-jitted protocol-default solve near machine precision."""
    chip, qubit = _r5_build(open_system)
    tlist = np.linspace(0.0, 40.0, 80)
    prob = _r5_problem(chip, qubit, 0.045, tlist)
    rc, ru = _r5_cached(prob), _r5_uncached(prob)
    exp_diff = float(np.max(np.abs(np.asarray(rc.expect[0]) - np.asarray(ru.expect[0]))))
    state_diff = float(
        np.max(np.abs(np.asarray(rc.final_state.to_jax()) - np.asarray(ru.final_state.to_jax())))
    )
    assert exp_diff < 1e-9
    assert state_diff < 1e-9


@pytest.mark.parametrize("open_system", [False, True], ids=["sesolve", "mesolve"])
def test_cached_jit_solve_grad_parity(open_system: bool) -> None:
    """Gradients through the cached-jit solve match the uncached path and a finite-difference reference."""
    def loss_factory(runner):
        chip, qubit = _r5_build(open_system)
        tlist = np.linspace(0.0, 40.0, 80)

        def loss(amp):
            r = runner(_r5_problem(chip, qubit, amp, tlist))
            return jnp.real(jnp.asarray(r.expect[0])[-1])

        return loss

    x = jnp.asarray(0.045)
    g_cached = float(jax.grad(loss_factory(_r5_cached))(x))
    loss_unc = loss_factory(_r5_uncached)
    g_uncached = float(jax.grad(loss_unc)(x))
    h = 1e-4
    fd = (float(loss_unc(x + h)) - float(loss_unc(x - h))) / (2 * h)
    # cached grad matches the uncached grad to ~machine precision; the
    # finite-difference reference is matched within its O(h^2) truncation error.
    assert abs(g_cached - g_uncached) < 1e-6
    assert abs(g_cached - fd) / max(abs(fd), 1e-12) < 1e-3


def test_cached_jit_solve_vmap_parity() -> None:
    """vmap over the cached-jit solve_problem matches sequential per-value calls exactly."""
    chip, qubit = _r5_build(False)
    tlist = np.linspace(0.0, 40.0, 80)

    def loss(amp):
        r = _r5_cached(_r5_problem(chip, qubit, amp, tlist))
        return jnp.real(jnp.asarray(r.expect[0])[-1])

    amps = jnp.asarray([0.03, 0.045, 0.06])
    vm = jax.vmap(loss)(amps)
    seq_vals = jnp.asarray([loss(a) for a in amps])
    assert float(jnp.max(jnp.abs(vm - seq_vals))) == 0.0


def test_cached_jit_solve_structure_change_is_a_cache_miss() -> None:
    """Structurally distinct problems (sesolve vs mesolve) get distinct cache entries, not shared artifacts."""
    backend = DynamiqsBackend()
    tlist = np.linspace(0.0, 40.0, 60)
    cache = backend._get_jit_solve_cache()
    cache.clear()

    probs = []
    for open_system in (False, True):
        qubit = DuffingTransmon(
            freq=5.0, anharmonicity=-0.30, levels=3, label="q",
            T1=200.0 if open_system else None,
        )
        drive = ChargeDrive(target=qubit, label="d")
        chip = Chip(devices=[qubit], frame="rotating", backend=backend, label="cm")
        chip.wire(drive)
        probs.append(_r5_problem(chip, qubit, 0.04, tlist))

    results = [backend.solve_problem(p) for p in probs]
    assert len(cache) == 2  # distinct structural signatures -> distinct artifacts
    for p, r in zip(probs, results):
        ru = Backend.solve_problem(backend, p)
        assert float(np.max(np.abs(np.asarray(r.expect[0]) - np.asarray(ru.expect[0])))) < 1e-9


def test_cached_jit_solve_no_stale_value_reuse() -> None:
    """Same structure with different traced values shares one compiled artifact but binds its own physics."""
    backend = DynamiqsBackend()
    tlist = np.linspace(0.0, 40.0, 60)
    cache = backend._get_jit_solve_cache()
    cache.clear()

    def problem_for(amp):
        qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.30, levels=3, label="q")
        drive = ChargeDrive(target=qubit, label="d")
        chip = Chip(devices=[qubit], frame="rotating", backend=backend, label="cv")
        chip.wire(drive)
        return _r5_problem(chip, qubit, amp, tlist)

    p_lo, p_hi = problem_for(0.02), problem_for(0.08)
    r_lo, r_hi = backend.solve_problem(p_lo), backend.solve_problem(p_hi)

    # identical structure -> a single shared compiled artifact
    assert len(cache) == 1
    # yet each matches its OWN uncached reference (the reused artifact binds the
    # fresh operators, not a stale closure)
    for p, r in ((p_lo, r_lo), (p_hi, r_hi)):
        ru = Backend.solve_problem(backend, p)
        assert float(np.max(np.abs(np.asarray(r.expect[0]) - np.asarray(ru.expect[0])))) < 1e-9
    # ... and the two genuinely differ (proves it did not silently return one)
    assert float(np.max(np.abs(np.asarray(r_lo.expect[0]) - np.asarray(r_hi.expect[0])))) > 1e-3
