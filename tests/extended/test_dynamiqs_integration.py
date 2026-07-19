"""Cross-backend Dynamiqs integration coverage for supported systems."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

pytestmark = pytest.mark.optional_backend

dynamiqs = pytest.importorskip("dynamiqs")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from quchip import (  # noqa: E402
    Capacitive,
    ChargeDrive,
    Chip,
    ControlEquipment,
    DuffingTransmon,
    Gaussian,
    QuantumSequence,
    Resonator,
)
from quchip.backend import reset_default_backend, set_default_backend  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_default_backend():
    """Keep backend selection explicit per scenario."""
    reset_default_backend()
    yield
    reset_default_backend()


def _set_backend(backend_name: str) -> None:
    set_default_backend(backend_name)


def _run_rabi_population(backend_name: str) -> np.ndarray:
    _set_backend(backend_name)
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3, label="q")
    drive = ChargeDrive(target=qubit, label="drive")
    chip = Chip(
        devices=[qubit],
        control_equipment=ControlEquipment(lines=[drive]),
        frame="lab",
        label=f"{backend_name}-rabi",
    )
    chip.dress()

    sequence = QuantumSequence(chip)
    sequence.charge(qubit, envelope=Gaussian(duration=60.0, amplitude=0.005, sigmas=4), freq=qubit.freq)
    result = sequence.simulate(
        tlist=np.linspace(0.0, 60.0, 121),
        initial_state=chip.state(q=0),
    )
    return np.asarray(result.population_array("q", 1))


def _run_dispersive_expectation(backend_name: str) -> np.ndarray:
    _set_backend(backend_name)
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3, label="q")
    resonator = Resonator(freq=6.8, levels=5, label="r", quality_factor=1e6)
    coupling = Capacitive(qubit, resonator, g=0.04, rwa=True)
    readout_drive = ChargeDrive(target=resonator, label="readout")
    chip = Chip(
        devices=[qubit, resonator],
        couplings=[coupling],
        control_equipment=ControlEquipment(lines=[readout_drive]),
        frame="rotating",
        label=f"{backend_name}-dispersive",
    )
    chip.dress()

    sequence = QuantumSequence(chip)
    sequence.schedule(
        readout_drive,
        envelope=Gaussian(duration=80.0, amplitude=0.01, sigmas=4),
        freq=chip.freq(resonator, when={qubit: 0}),
    )
    result = sequence.simulate(
        tlist=np.linspace(0.0, 80.0, 81),
        initial_state=chip.state(q=1, r=0),
        e_ops=chip.e_ops(r="a"),
    )
    return np.asarray(result._expect_data["r"].values, dtype=complex)


def _run_cr_expectation(backend_name: str) -> np.ndarray:
    _set_backend(backend_name)
    control = DuffingTransmon(freq=5.2, anharmonicity=-0.33, levels=3, label="q1")
    target = DuffingTransmon(freq=5.0, anharmonicity=-0.33, levels=3, label="q2")
    drive = ChargeDrive(target=control, label="cr")
    chip = Chip(
        devices=[control, target],
        couplings=[Capacitive(control, target, g=0.003)],
        control_equipment=ControlEquipment(lines=[drive]),
        frame="rotating",
        label=f"{backend_name}-cr",
    )
    chip.dress()

    sequence = QuantumSequence(chip)
    sequence.schedule(
        drive,
        envelope=Gaussian(duration=60.0, amplitude=0.03, sigmas=4),
        freq=chip.freq(target),
    )
    result = sequence.simulate(
        tlist=np.linspace(0.0, 60.0, 81),
        initial_state=chip.state(q1=1, q2=0),
        e_ops=chip.e_ops(q2="Z"),
    )
    return np.asarray(result._expect_data["q2"].values, dtype=complex)


def _run_mesolve_population(backend_name: str) -> tuple[np.ndarray, str]:
    _set_backend(backend_name)
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q", T1=120.0)
    chip = Chip(devices=[qubit], frame="rotating", label=f"{backend_name}-mesolve")
    chip.dress()

    result = QuantumSequence(chip).simulate(
        tlist=np.linspace(0.0, 60.0, 61),
        initial_state=chip.state(q=1),
    )
    return np.asarray(result.population_array("q", 1)), result.solver


def test_rabi_backend_agreement() -> None:
    """Rabi population traces from the qutip and dynamiqs backends agree within tolerance."""
    qutip = _run_rabi_population("qutip")
    dyn = _run_rabi_population("dynamiqs")
    npt.assert_allclose(dyn, qutip, atol=1e-3, rtol=1e-3)


def test_dispersive_backend_agreement() -> None:
    """Dispersive readout expectation traces from the qutip and dynamiqs backends agree."""
    qutip = _run_dispersive_expectation("qutip")
    dyn = _run_dispersive_expectation("dynamiqs")
    npt.assert_allclose(dyn, qutip, atol=1e-3, rtol=1e-3)


def test_cr_backend_agreement() -> None:
    """Cross-resonance expectation traces from the qutip and dynamiqs backends agree."""
    qutip = _run_cr_expectation("qutip")
    dyn = _run_cr_expectation("dynamiqs")
    npt.assert_allclose(dyn, qutip, atol=1e-3, rtol=1e-3)


def test_mesolve_backend_agreement() -> None:
    """Both backends resolve to the mesolve solver and agree on the T1-decay population trace."""
    qutip, qutip_solver = _run_mesolve_population("qutip")
    dyn, dyn_solver = _run_mesolve_population("dynamiqs")
    assert qutip_solver == "mesolve"
    assert dyn_solver == "mesolve"
    npt.assert_allclose(dyn, qutip, atol=1e-3, rtol=1e-3)


def _run_amplitude_trace(backend_name: str) -> np.ndarray:
    """Phase-sensitive ⟨1|ψ(t)⟩ under a resonant charge drive; the rotation drives it real-negative."""
    # amplitude_array must return the complex projection, not |<1|psi>|^2 (always
    # non-negative, would fail to reproduce the sign as the drive populates |1>).
    _set_backend(backend_name)
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3, label="q")
    drive = ChargeDrive(target=qubit, label="drive")
    chip = Chip(
        devices=[qubit],
        control_equipment=ControlEquipment(lines=[drive]),
        frame="rotating",
        label=f"{backend_name}-amp",
    )
    chip.dress()

    sequence = QuantumSequence(chip)
    sequence.charge(qubit, envelope=Gaussian(duration=60.0, amplitude=0.01, sigmas=4), freq=qubit.freq)
    result = sequence.simulate(tlist=np.linspace(0.0, 60.0, 121), initial_state=chip.state(q=0))
    return np.asarray(result.amplitude_array(chip.state(q=1)), dtype=complex)


def test_amplitude_array_phase_backend_agreement() -> None:
    """amplitude_array must be the phase-sensitive complex projection on both backends."""
    qutip = _run_amplitude_trace("qutip")
    dyn = _run_amplitude_trace("dynamiqs")
    # the trace genuinely carries phase: ⟨1|ψ⟩ goes clearly negative-real, which
    # no |·|² (always non-negative) accessor could reproduce
    assert np.min(qutip.real) < -0.1
    npt.assert_allclose(dyn, qutip, atol=1e-3, rtol=1e-3)


def test_public_loss_path_supports_jax_value_and_grad_on_dynamiqs() -> None:
    """A public infidelity loss built through the sequence API supports jax value_and_grad."""
    _set_backend("dynamiqs")
    tlist = jnp.linspace(0.0, 20.0, 41)

    def loss(amplitude):
        qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3, label="q")
        drive = ChargeDrive(target=qubit, label="drive")
        chip = Chip(
            devices=[qubit],
            control_equipment=ControlEquipment(lines=[drive]),
            frame="rotating",
            label="dynamiqs-grad-rabi",
        )
        chip.dress()

        sequence = QuantumSequence(chip)
        sequence.charge(
            qubit,
            envelope=Gaussian(duration=20.0, amplitude=amplitude, sigmas=4),
            freq=chip.freq(qubit),
        )
        result = sequence.simulate(
            tlist=tlist,
            initial_state=chip.state(q=0),
            options={"store_states": True, "store_final_state": True},
        )
        return 1.0 - result.overlap_array(chip.state(q=1))[-1]

    value, grad = jax.value_and_grad(loss)(jnp.asarray(0.02))
    assert jnp.isfinite(value)
    assert jnp.isfinite(grad)


def test_rotating_frame_coupled_sequence_supports_jax_grad_on_traced_chip_param() -> None:
    """A traced device frequency flowing through a rotating-frame coupled sequence yields a finite grad."""
    _set_backend("dynamiqs")
    tlist = jnp.linspace(0.0, 24.0, 61)

    def loss(control_freq):
        control = DuffingTransmon(freq=control_freq, anharmonicity=-0.3, levels=3, label="q1")
        target = DuffingTransmon(freq=4.82, anharmonicity=-0.31, levels=3, label="q2")
        drive = ChargeDrive(target=control, label="cr")
        chip = Chip(
            devices=[control, target],
            couplings=[Capacitive(control, target, g=0.0032, rwa=True)],
            control_equipment=ControlEquipment(lines=[drive]),
            frame="rotating",
            label="dynamiqs-rotating-grad-coupled",
        )

        sequence = QuantumSequence(chip)
        sequence.schedule(
            drive,
            envelope=Gaussian(duration=24.0, amplitude=0.015, sigmas=4),
            freq=target.drive_freq,
        )
        result = sequence.simulate(
            tlist=tlist,
            initial_state=chip.bare_state(q1=1, q2=0),
            options={"store_states": True, "store_final_state": True},
        )
        return 1.0 - result.overlap_array(chip.bare_state(q1=0, q2=1))[-1]

    value, grad = jax.value_and_grad(loss)(jnp.asarray(5.05))
    assert jnp.isfinite(value)
    assert jnp.isfinite(grad)


def test_unified_expect_method_on_dynamiqs() -> None:
    """The unified expect() method returns full trace on dynamiqs backend."""
    _set_backend("dynamiqs")
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=2, label="q")
    drive = ChargeDrive(target=qubit, label="drive")
    chip = Chip(
        devices=[qubit],
        control_equipment=ControlEquipment(lines=[drive]),
        frame="rotating",
        label="dynamiqs-unified-expect",
    )

    sequence = QuantumSequence(chip)
    sequence.schedule(
        drive,
        envelope=Gaussian(duration=20.0, amplitude=0.02, sigmas=4),
        freq=chip.freq("q"),
    )
    tlist = jnp.linspace(0.0, 20.0, 41)
    result = sequence.simulate(tlist=tlist, e_ops=chip.e_ops(q="Z"))

    values = result.expect("q")
    assert np.asarray(values).shape[0] == len(result.times)

    final = result.state()
    assert final is not None


def test_expect_final_supports_jax_grad_on_dynamiqs() -> None:
    """A loss built from ``result.expect_final`` supports jax value_and_grad on dynamiqs."""
    _set_backend("dynamiqs")
    tlist = jnp.linspace(0.0, 20.0, 41)

    def loss(amplitude):
        qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=2, label="q")
        drive = ChargeDrive(target=qubit, label="drive")
        chip = Chip(
            devices=[qubit],
            control_equipment=ControlEquipment(lines=[drive]),
            frame="rotating",
            label="dynamiqs-expect-final-grad",
        )

        sequence = QuantumSequence(chip)
        sequence.schedule(
            drive,
            envelope=Gaussian(duration=20.0, amplitude=amplitude, sigmas=4),
            freq=chip.freq("q"),
        )
        result = sequence.simulate(
            tlist=tlist,
            e_ops=chip.e_ops(q="Z"),
        )
        return 1.0 - jnp.real(result.expect_final("q"))

    value, grad = jax.value_and_grad(loss)(jnp.asarray(0.02))
    assert jnp.isfinite(value)
    assert jnp.isfinite(grad)


def test_per_call_dynamiqs_backend_with_qutip_built_state() -> None:
    """backend="dynamiqs" scopes one call; a QuTiP-native psi0 is coerced."""
    # Process default stays QuTiP — the state below is a qutip.Qobj.
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3, label="q")
    drive = ChargeDrive(target=qubit, label="drive")
    chip = Chip(
        devices=[qubit],
        control_equipment=ControlEquipment(lines=[drive]),
        frame="rotating",
        label="per-call-backend",
    )
    sequence = QuantumSequence(chip)
    sequence.schedule(
        drive,
        envelope=Gaussian(duration=40.0, amplitude=0.0125, sigmas=3),
        freq=chip.freq("q"),
    )
    tlist = np.linspace(0.0, 40.0, 81)
    psi0 = chip.state(q=0)  # QuTiP-native Qobj

    res_dq = sequence.simulate(tlist=tlist, initial_state=psi0, backend="dynamiqs")
    res_qt = sequence.simulate(tlist=tlist, initial_state=psi0)

    npt.assert_allclose(
        np.asarray(res_dq.population_array("q", level=1)),
        np.asarray(res_qt.population_array("q", level=1)),
        atol=1e-5,
    )
    # result accessors coerce foreign-native comparison states too:
    # a QuTiP-built target against the dynamiqs trajectory, and vice versa.
    npt.assert_allclose(
        np.asarray(res_dq.overlap_array(psi0)),
        np.asarray(res_qt.overlap_array(psi0)),
        atol=1e-5,
    )
    # the per-call override never leaks into the process default
    from quchip.backend.qutip import QuTiPBackend

    assert isinstance(chip.backend, QuTiPBackend)


def test_per_call_dynamiqs_backend_preserves_composite_qutip_state_dims() -> None:
    """A QuTiP-built composite ket keeps its subsystem dimensions under the override."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=2, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.3, levels=3, label="q1")
    chip = Chip([q0, q1], frame="rotating", label="composite-per-call-backend")
    sequence = QuantumSequence(chip)
    psi0 = chip.state(q0=0, q1=0)  # QuTiP Qobj with dims [[2, 3], [1]]

    result = sequence.simulate(
        tlist=np.linspace(0.0, 0.1, 3),
        initial_state=psi0,
        backend="dynamiqs",
        partition=False,
    )

    npt.assert_allclose(np.asarray(result.overlap_array(psi0)), 1.0, atol=1e-8)


def test_per_call_dynamiqs_batch_preserves_composite_qutip_state_dims() -> None:
    """Batched QuTiP-built composite kets keep their subsystem dimensions under the override."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=2, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.3, levels=3, label="q1")
    chip = Chip([q0, q1], frame="rotating", label="composite-batch-per-call-backend")
    sequence = QuantumSequence(chip)
    psi0 = chip.state(q0=0, q1=0)
    state_axis = sequence.vary("initial_state", [psi0, psi0], name="state")

    results = sequence.simulate_batch(
        state_axis,
        tlist=np.linspace(0.0, 0.1, 3),
        backend="dynamiqs",
        progress=False,
    )

    for result in results:
        npt.assert_allclose(np.asarray(result.overlap_array(psi0)), 1.0, atol=1e-8)


def test_per_call_dynamiqs_gradient_without_global_flip() -> None:
    """A jax.grad loss can pin dynamiqs per call while the default stays QuTiP."""
    tlist = jnp.linspace(0.0, 20.0, 41)

    def loss(amplitude):
        qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=2, label="q")
        drive = ChargeDrive(target=qubit, label="drive")
        chip = Chip(
            devices=[qubit],
            control_equipment=ControlEquipment(lines=[drive]),
            frame="rotating",
            label="per-call-grad",
        )
        sequence = QuantumSequence(chip)
        sequence.schedule(
            drive,
            envelope=Gaussian(duration=20.0, amplitude=amplitude, sigmas=4),
            freq=chip.freq("q"),
        )
        result = sequence.simulate(tlist=tlist, backend="dynamiqs")
        return 1.0 - jnp.real(result.population_array("q", level=1)[-1])

    value, grad = jax.value_and_grad(loss)(jnp.asarray(0.02))
    assert jnp.isfinite(value)
    assert jnp.isfinite(grad)


def test_concrete_build_ships_no_dead_zero_structure_to_dynamiqs() -> None:
    """A concretely-built coupled chip emits no zero dynamic terms and a pruned static term."""
    # Regression for the overhead-ladder warm-solve gap (benchmarks/overhead, 2026-07-04):
    # SparseDIA payloads carried structurally-zero counter-rotating band terms and the
    # coupling fold's exactly-cancelled diagonals stored as explicit zeros on H0, both
    # integrated by dynamiqs on every step.
    _set_backend("dynamiqs")
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.25, levels=3, label="q1")
    drive = ChargeDrive(target=q0, label="d0")
    chip = Chip(
        devices=[q0, q1],
        couplings=[Capacitive(q0, q1, g=0.01)],
        control_equipment=ControlEquipment(lines=[drive]),
        frame={q0: 5.0, q1: 5.1},
        rwa=True,
    )
    sequence = QuantumSequence(chip)
    sequence.schedule(drive, envelope=Gaussian(duration=20.0, amplitude=0.02, sigmas=3), freq=5.0)
    problem = sequence.build_problem(
        tlist=np.linspace(0.0, 20.0, 41),
        initial_state=chip.bare_state(q0=0, q1=0),
    )
    description = problem.hamiltonian

    # RWA exchange bands (±1, ∓1) on one coupling + two drive bands — no zero operators.
    assert len(description.dynamic_terms) == 4
    for term in description.dynamic_terms:
        values = np.asarray(term.operator.values)
        assert np.abs(values).max() > 0.0, f"structurally-zero {term.origin} term shipped to the solver"

    # The coupling fold cancels the lab-frame interaction out of H₀ exactly;
    # its offsets must be pruned so only the diagonal (offset 0) remains.
    static = description.static_terms[0].operator
    assert static.layout == "dia"
    assert tuple(int(x) for x in np.asarray(static.offsets)) == (0,)


def test_dropped_term_audit_survives_traced_coupling() -> None:
    """A jit-traced coupling g flows into the audit raw; the summary never concretizes it."""
    # DroppedTerm amplitude/frequency fields hold raw (possibly traced) GHz values by
    # contract, since chip parameters legitimately arrive as tracers on this backend.
    from quchip.engine.stage1_frames import resolve_frame
    from quchip.engine.stage2_assembly import build_hamiltonian_description

    seen: dict[str, str] = {}

    def build(g):
        q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
        q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.25, levels=3, label="q1")
        chip = Chip(
            [q0, q1],
            couplings=[Capacitive(q0, q1, g=g, label="cap01")],
            frame={q0: 5.0, q1: 5.1},
            rwa=True,
            backend="dynamiqs",
        )
        description = build_hamiltonian_description(
            chip, [], resolved_frame=resolve_frame(chip, chip.frame)
        )
        seen["summary"] = description.dropped_terms_summary()
        # band_weights is static structure, so selecting the populated a†b†
        # band is trace-safe; its amplitude (largest band element, g·√2·√2)
        # stays a raw tracer.
        record = next(dt for dt in description.dropped_terms if dt.band_weights == (1, 1))
        return record.amplitude * 2.0

    out = jax.jit(build)(jnp.asarray(0.01))
    assert "amp traced" in seen["summary"]
    assert "freq 10.1 GHz" in seen["summary"]  # pinned frame refs stay concrete
    npt.assert_allclose(np.asarray(out), 0.04)  # 2 × the band's 2g max element
