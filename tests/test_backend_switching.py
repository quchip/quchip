"""Per-call backend switching, state coercion, and solver-option defaults.

The ``backend=`` argument of ``simulate``/``simulate_batch`` scopes one call:
it outranks the chip-constructed backend and the process default, and foreign
-native initial states are coerced at the solve boundary. The nsteps default
is a generous abort ceiling so user code never carries ``{"nsteps": ...}``.
"""

from __future__ import annotations

import numpy as np
from qutip import Qobj

from quchip import (
    Capacitive,
    ChargeDrive,
    Chip,
    DuffingTransmon,
    Gaussian,
    QuantumSequence,
    Resonator,
)
from quchip.backend import _coerce_backend, get_default_backend
from quchip.backend.qutip import QuTiPBackend


def _demo_chip(chip_backend=None):
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    chip = Chip(
        [q, r],
        couplings=[Capacitive(q, r, g=0.02)],
        frame="rotating",
        rwa=True,
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


def test_nsteps_default_is_a_generous_ceiling(backend: QuTiPBackend) -> None:
    """The default nsteps ceiling never aborts a solve, and an explicit user value always overrides it."""
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
    monkeypatch.setattr(
        per_call, "solve_problem", lambda problem: (used.append("per_call"), original(problem))[1]
    )

    seq.simulate(
        tlist=np.linspace(0.0, 20.0, 21),
        e_ops={q: q.projector(1, 1)},
        initial_state=chip.state({q: 0, r: 0}),
        backend=per_call,
    )
    assert used == ["per_call"]
    # The chip-level choice is restored afterward.
    assert chip.backend is chip_level
