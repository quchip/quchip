"""Cross-backend Dynamiqs integration coverage for supported systems."""

from __future__ import annotations

from quchip.approximations import RWA

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


@pytest.mark.validation
def test_amplitude_array_phase_backend_agreement() -> None:
    """amplitude_array must be the phase-sensitive complex projection on both backends."""
    qutip = _run_amplitude_trace("qutip")
    dyn = _run_amplitude_trace("dynamiqs")
    # the trace genuinely carries phase: ⟨1|ψ⟩ goes clearly negative-real, which
    # no |·|² (always non-negative) accessor could reproduce
    assert np.min(qutip.real) < -0.1
    npt.assert_allclose(dyn, qutip, atol=1e-3, rtol=1e-3)


@pytest.mark.validation
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

    npt.assert_allclose(np.asarray(result.overlap(psi0)), 1.0, atol=1e-8)


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
        approximation=RWA(),
    )
    sequence = QuantumSequence(chip)
    sequence.schedule(drive, envelope=Gaussian(duration=20.0, amplitude=0.02, sigmas=3), freq=5.0)
    problem = sequence.build_problem(
        tlist=np.linspace(0.0, 20.0, 41),
        initial_state=chip.bare_state(q0=0, q1=0),
    )
    description = problem.engine_result

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
    from quchip.engine.frames import resolve_frame
    from quchip.engine.assembly import build_engine_result

    seen: dict[str, str] = {}

    def build(g):
        q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
        q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.25, levels=3, label="q1")
        chip = Chip(
            [q0, q1],
            couplings=[Capacitive(q0, q1, g=g, label="cap01")],
            frame={q0: 5.0, q1: 5.1},
            approximation=RWA(),
            backend="dynamiqs",
        )
        description = build_engine_result(chip, [], resolved_frame=resolve_frame(chip, chip.frame))
        seen["summary"] = description.dropped_terms_summary()
        # band_weights is static structure, so selecting the populated ab
        # band is trace-safe; its amplitude (the selected entry, g·√2·√2)
        # stays a raw tracer.
        record = next(dt for dt in description.dropped_terms if dt.band_weights == (1, 1))
        return record.amplitude * 2.0

    out = jax.jit(build)(jnp.asarray(0.01))
    assert "amp traced" in seen["summary"]
    assert "detuning traced" in seen["summary"]  # the maximizing entry depends on the traced coupling
    npt.assert_allclose(np.asarray(out), 0.04)  # 2 × the band's 2g max element
