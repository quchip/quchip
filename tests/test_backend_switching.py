"""Per-call backend switching, state coercion, and solver-option defaults.

The ``backend=`` argument of ``simulate``/``simulate_batch`` scopes one call
and takes precedence over the chip backend and process default. Initial states
native to another backend are coerced at the solve boundary. When the caller
omits ``nsteps``, QuTiP can infer an abort ceiling from spectral metadata.
"""

from __future__ import annotations

from quchip.approximations import RWA

import numpy as np
import pytest
from qutip import Qobj

from quchip import (
    Capacitive,
    ChargeBasisTransmon,
    ChargeDrive,
    Chip,
    DuffingTransmon,
    Gaussian,
    QuantumSequence,
    Resonator,
)
from quchip.backend import Backend, _coerce_backend, get_default_backend
from quchip.backend.qutip import QuTiPBackend


def _demo_chip(chip_backend=None):
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    chip = Chip(
        [q, r],
        couplings=[Capacitive(q, r, g=0.02)],
        frame="rotating",
        approximation=RWA(),
        backend=chip_backend,
    )
    drv = ChargeDrive(target=q, label="d")
    chip.wire(drv)
    return chip, drv, q, r


def test_named_backend_coercion_returns_shared_instance() -> None:
    """Coercing a backend name twice returns the identical shared instance, not a fresh one each time."""
    # Backend-side caches (dynamiqs jitted solves) live on the instance;
    # name-coercion must hand back one shared instance per name.
    assert _coerce_backend("qutip") is _coerce_backend("qutip")


def test_nsteps_default_is_at_least_200k(backend: QuTiPBackend) -> None:
    """The inferred nsteps ceiling is at least 200,000; an explicit value takes precedence."""
    tlist = np.linspace(0.0, 100.0, 11)
    resolved = backend.resolve_solver_options({}, metadata={"spectral_bound_ghz": 5.0}, tlist=tlist)
    assert resolved["nsteps"] >= 200_000

    # An explicit user choice is never overridden.
    resolved = backend.resolve_solver_options({"nsteps": 7}, metadata={"spectral_bound_ghz": 5.0}, tlist=tlist)
    assert resolved["nsteps"] == 7


def test_qutip_coerce_state_wraps_arrays_with_dims(backend: QuTiPBackend) -> None:
    """coerce_state wraps raw ket/density-matrix arrays into Qobj with the given dims, passing native states through."""
    ket = np.zeros(12, dtype=complex)
    ket[0] = 1.0
    coerced = backend.coerce_state(ket, dims=(3, 4))
    assert isinstance(coerced, Qobj)
    assert coerced.isket
    assert coerced.dims[0] == [3, 4]

    dm = np.eye(12, dtype=complex) / 12.0
    coerced_dm = backend.coerce_state(dm, dims=(3, 4))
    assert coerced_dm.dims == [[3, 4], [3, 4]]

    # Native states pass through untouched.
    assert backend.coerce_state(coerced, dims=(3, 4)) is coerced

    import importlib.util

    if importlib.util.find_spec("dynamiqs") is not None:
        from quchip.backend.dynamiqs import DynamiqsBackend

        dynamiqs = DynamiqsBackend()
        qarray = dynamiqs.from_array(np.diag([1.0, 2.0]))
        np.testing.assert_allclose(backend.to_array(backend.from_array(qarray)), np.diag([1.0, 2.0]))
        np.testing.assert_allclose(
            np.asarray(dynamiqs.to_array(dynamiqs.from_array(Qobj(np.diag([3.0, 4.0]))))),
            np.diag([3.0, 4.0]),
        )


def test_per_call_backend_scopes_exactly_one_call() -> None:
    """simulate's backend= argument scopes only that call; the chip reverts to the process default afterward."""
    chip, drv, q, r = _demo_chip()
    seq = QuantumSequence(chip)
    seq.schedule(drv, envelope=Gaussian(duration=20.0, sigmas=3, amplitude=0.01), freq=chip.freq(q))

    marker = QuTiPBackend()
    res = seq.simulate(
        tlist=np.linspace(0.0, 20.0, 21),
        e_ops={q: q.projector(1, 1)},
        initial_state=chip.state({q: 0, r: 0}),
        backend=marker,
    )
    assert np.all(np.isfinite(np.real(np.asarray(res.expect(q)))))

    # Outside the call, the chip resolves back to the process default.
    assert chip.backend is get_default_backend()
    assert chip.backend is not marker


def test_per_call_backend_outranks_chip_backend(monkeypatch) -> None:
    """A per-call backend outranks the chip-constructed backend once; the chip then reverts to its own choice."""
    chip_level = QuTiPBackend()
    per_call = QuTiPBackend()
    chip, drv, q, r = _demo_chip(chip_backend=chip_level)
    seq = QuantumSequence(chip)
    seq.schedule(drv, envelope=Gaussian(duration=20.0, sigmas=3, amplitude=0.01), freq=chip.freq(q))

    used: list[str] = []
    original = per_call.solve_problem
    monkeypatch.setattr(per_call, "solve_problem", lambda problem: (used.append("per_call"), original(problem))[1])

    seq.simulate(
        tlist=np.linspace(0.0, 20.0, 21),
        e_ops={q: q.projector(1, 1)},
        initial_state=chip.state({q: 0, r: 0}),
        backend=per_call,
    )
    assert used == ["per_call"]
    # The chip-level choice is restored afterward.
    assert chip.backend is chip_level


# Populations are read at t = 0, before any integration step, so only
# floating-point roundoff separates them from the prepared values.
_ROUNDOFF = 1e-12


def _transmon_chip(backend: str, levels: int = 7):
    """Return a projected charge-basis transmon, its chip on *backend*, and an empty sequence."""
    q = ChargeBasisTransmon(E_C=0.25, E_J=12.5, num_basis=7, levels=levels, basis="eigen", label="q")
    chip = Chip([q], backend=backend)
    return q, chip, QuantumSequence(chip)


def _initial_populations(result) -> np.ndarray:
    return np.array([float(np.real(np.asarray(result.population("q", level))[0])) for level in range(3)])


@pytest.mark.optional_backend
@pytest.mark.parametrize(
    ("chip_backend", "override", "levels"),
    [
        pytest.param("dynamiqs", "qutip", 7, id="reverse"),
    ],
)
def test_backend_override_keeps_chip_state(chip_backend: str, override: str, levels: int) -> None:
    """A chip.state() ket starts in its prepared level whichever backend solves it."""
    pytest.importorskip("dynamiqs")
    q, chip, sequence = _transmon_chip(chip_backend, levels)
    psi = chip.state({q: 1})
    for backend in (None, override):
        result = sequence.simulate(tlist=np.array([0.0, 1.0]), initial_state=psi, backend=backend)
        np.testing.assert_allclose(_initial_populations(result), [0.0, 1.0, 0.0], rtol=0, atol=_ROUNDOFF)


class _ProtocolBatchBackend(QuTiPBackend):
    """QuTiP solves dispatched through the protocol's default batch path."""

    prepare_batch = Backend.prepare_batch
    solve_batch = Backend.solve_batch


@pytest.mark.optional_backend
@pytest.mark.parametrize(
    ("chip_backend", "override"),
    [
        pytest.param("qutip", "dynamiqs", id="dynamiqs"),
        pytest.param("dynamiqs", _ProtocolBatchBackend(), id="protocol-batch"),
    ],
)
def test_backend_override_keeps_batched_chip_states(chip_backend: str, override: Backend | str) -> None:
    """Each batched chip.state() ket starts in its prepared level under a per-call backend override."""
    pytest.importorskip("dynamiqs")
    q, chip, sequence = _transmon_chip(chip_backend)
    states = sequence.vary("initial_state", [chip.state({q: level}) for level in range(3)])
    results = sequence.simulate_batch(states, tlist=np.array([0.0, 1.0]), backend=override, progress=False)
    for level, result in enumerate(results):
        np.testing.assert_allclose(_initial_populations(result), np.eye(3)[level], rtol=0, atol=_ROUNDOFF)


@pytest.mark.optional_backend
def test_backend_override_keeps_density_matrix_populations() -> None:
    """A density matrix built from chip states keeps its level populations under a per-call backend override."""
    pytest.importorskip("dynamiqs")
    q, chip, sequence = _transmon_chip("qutip")
    rho = 0.25 * chip.backend.state_to_dm(chip.state({q: 0})) + 0.75 * chip.backend.state_to_dm(chip.state({q: 1}))
    for backend in (None, "dynamiqs"):
        result = sequence.simulate(tlist=np.array([0.0, 1.0]), initial_state=rho, backend=backend)
        np.testing.assert_allclose(_initial_populations(result), [0.25, 0.75, 0.0], rtol=0, atol=_ROUNDOFF)
