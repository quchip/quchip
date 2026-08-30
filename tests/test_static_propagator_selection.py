"""Automatic constant-generator solver selection at the backend boundary."""

from __future__ import annotations

import numpy as np
import pytest

from quchip import ChargeDrive, Chip, DuffingTransmon, QuantumSequence, Square
from quchip.engine import build_problem


def _static_problem(
    backend: str,
    *,
    levels: int = 3,
    T1: float | None = None,
    points: int = 3,
    options: dict | None = None,
):
    qubit = DuffingTransmon(
        freq=5.0,
        anharmonicity=-0.25,
        levels=levels,
        label="q",
        T1=T1,
    )
    chip = Chip([qubit], frame="lab", backend=backend)
    problem = build_problem(
        chip,
        [],
        np.linspace(0.0, 1.0, points),
        options=options,
    )
    return chip.backend, problem


def _resolved_options(backend, problem) -> dict:
    prepared = backend.prepare_hamiltonian(problem.engine_result, problem.tlist)
    return backend._resolve_solve_config(problem, prepared)[3]


class TestQuTiPStaticPropagatorSelection:
    """QuTiP should diagonalize only eligible constant generators."""

    def test_small_static_closed_problem_selects_diagonalization(self) -> None:
        """A closed static Hilbert space through dimension 64 should select QuTiP's diagonal propagator."""
        backend, problem = _static_problem("qutip", levels=64)

        options = _resolved_options(backend, problem)

        assert options["method"] == "diag"
        assert "nsteps" not in options
        assert "max_step" not in options

    def test_small_static_open_problem_selects_liouvillian_diagonalization(self) -> None:
        """A static dissipative Hilbert space through dimension 12 should diagonalize its Liouvillian."""
        backend, problem = _static_problem("qutip", levels=12, T1=20.0)

        options = _resolved_options(backend, problem)

        assert options["method"] == "diag"
        assert "nsteps" not in options
        assert "max_step" not in options

    @pytest.mark.parametrize(
        ("levels", "T1"),
        [(65, None), (13, 20.0)],
        ids=["closed-above-limit", "open-above-limit"],
    )
    def test_large_static_problem_retains_adaptive_integrator(
        self,
        levels: int,
        T1: float | None,
    ) -> None:
        """Static generators above their dimension limit should retain QuTiP's adaptive default."""
        backend, problem = _static_problem("qutip", levels=levels, T1=T1)

        options = _resolved_options(backend, problem)

        assert "method" not in options
        assert "nsteps" in options

    def test_explicit_method_is_never_overridden(self) -> None:
        """An explicit QuTiP method should remain authoritative for an otherwise eligible problem."""
        backend, problem = _static_problem("qutip", options={"method": "adams"})

        options = _resolved_options(backend, problem)

        assert options["method"] == "adams"

    @pytest.mark.parametrize("step_option", [{"nsteps": 17}, {"max_step": 0.01}])
    def test_explicit_step_control_suppresses_automatic_method(self, step_option: dict) -> None:
        """Explicit QuTiP step controls should retain the adaptive integrator that consumes them."""
        backend, problem = _static_problem("qutip", options=step_option)

        options = _resolved_options(backend, problem)

        assert "method" not in options
        assert all(options[key] == value for key, value in step_option.items())

    def test_driven_problem_retains_adaptive_integrator(self) -> None:
        """Any explicit Hamiltonian time dependence should keep QuTiP's adaptive integrator."""
        qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        drive = ChargeDrive(qubit, label="xy")
        chip = Chip([qubit], frame="rotating", backend="qutip")
        chip.wire(drive)
        sequence = QuantumSequence(chip)
        sequence.schedule(
            drive,
            envelope=Square(duration=1.0, amplitude=0.02),
            freq=chip.freq(qubit),
        )
        problem = sequence.build_problem(tlist=np.linspace(0.0, 1.0, 3))

        options = _resolved_options(chip.backend, problem)

        assert problem.engine_result.dynamic_terms
        assert "method" not in options


@pytest.mark.optional_backend
def test_dynamiqs_static_problem_retains_adaptive_integrator() -> None:
    """quchip must not automatically select Dynamiqs' newer matrix-exponential method."""
    pytest.importorskip("dynamiqs")
    backend, problem = _static_problem("dynamiqs", levels=3, points=2)

    options = _resolved_options(backend, problem)

    assert "method" not in options
    assert "max_steps" in options
