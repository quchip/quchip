"""The memory checks know the available memory and refuse requests that cannot fit."""

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import splu

from quchip import VNA, Capacitive, Chip, DuffingTransmon, PortNetwork, Resonator, RWA
from quchip.utils import memory

pytestmark = pytest.mark.unit


def feedline(levels):
    """Two transmon-readout pairs on one cascaded feedline."""
    qs = [DuffingTransmon(freq=5.3 - 0.1 * i, anharmonicity=-0.26, levels=levels, label=f"q{i}", T1=3e4)
          for i in range(2)]
    rs = [Resonator(freq=6.5 + 0.1 * i, levels=levels, label=f"r{i}", internal_quality_factor=2e5) for i in range(2)]
    network = PortNetwork(label="feedline")
    ports = [network.port(r.label, target=r, rate=2 * np.pi * 0.002) for r in rs]
    network.cascade(*ports)
    network.expose("feed", input=ports[0], output=ports[-1])
    return Chip(qs + rs, [Capacitive(q, r, g=0.04) for q, r in zip(qs, rs)], port_network=network, frame=5.2,
                approximation=RWA())


def test_available_memory_is_known_and_an_impossible_request_raises():
    """psutil is a dependency, so the checks run on every platform."""
    available = memory.available_memory_bytes()

    assert isinstance(available, int) and available > 0
    assert not memory.fits_memory(1e18)
    with pytest.raises(MemoryError, match=r"A 1 EB allocation needs about 1000000000\.0 GB.*Use less\."):
        memory.require_memory(1e18, task="A 1 EB allocation", remedy="Use less.")


def test_a_cgroup_limit_below_system_memory_sets_the_available_memory(tmp_path, monkeypatch):
    """A container limit of 1 GB with 0.25 GB in use leaves 0.75 GB."""
    (tmp_path / "memory.max").write_text("1000000000\n")
    (tmp_path / "memory.current").write_text("250000000\n")
    monkeypatch.setattr(memory, "_CGROUP", tmp_path)
    assert memory.available_memory_bytes() == min(750_000_000, memory.psutil.virtual_memory().available)

    (tmp_path / "memory.max").write_text("max\n")
    assert memory.available_memory_bytes() > 0


def test_sparse_lu_estimate_stays_below_the_factors_that_superlu_stores():
    """The estimate is low but follows the fill, from a tridiagonal matrix to one that fills densely."""
    n = 400
    tridiagonal = sparse.diags([np.ones(n - 1), 4.0 * np.ones(n), np.ones(n - 1)], [-1, 0, 1])
    line = sparse.diags([-np.ones(19), 2.0 * np.ones(20), -np.ones(19)], [-1, 0, 1])
    grid = sparse.kronsum(line, line) + 0.1 * sparse.random(n, n, density=0.002, random_state=2)
    scattered = sparse.random(n, n, density=0.02, random_state=1) + 4.0 * sparse.eye(n)

    for matrix in (tridiagonal, grid, scattered):
        factors = splu(sparse.csc_matrix(matrix, dtype=complex))
        stored = 20 * (factors.L.nnz + factors.U.nnz - n)
        assert stored / 10 <= memory.sparse_lu_bytes(sparse.csr_matrix(matrix)) <= stored


def test_qutip_steady_state_refuses_lu_factors_that_cannot_fit(monkeypatch):
    """With 16 MB free, the generator of two driven feedline pairs fits, but its LU factors need about 34 MB."""
    chip = feedline(3)
    frequency = np.array([float(chip.freq("r0"))])
    monkeypatch.setattr(memory, "available_memory_bytes", lambda: 16 * 10**6)

    with pytest.raises(MemoryError, match=r"Factorizing the 6561 × 6561 stationary generator needs about"):
        VNA(chip).finite_power(frequency, 3e-2, options={})


def test_qutip_resolvent_refuses_lu_factors_that_cannot_fit(monkeypatch):
    """After the operating point, a probe sweep checks the factors of its shifted generator."""
    from quchip.backend.qutip import QuTiPBackend

    chip = feedline(3)
    frequency = np.array([float(chip.freq("r0"))])
    free = [10**12]
    monkeypatch.setattr(memory, "available_memory_bytes", lambda: free[0])
    solve = QuTiPBackend.steadystate

    def solve_then_leave_one_megabyte(self, *args, **kwargs):
        result = solve(self, *args, **kwargs)
        free[0] = 10**6
        return result

    monkeypatch.setattr(QuTiPBackend, "steadystate", solve_then_leave_one_megabyte)
    with pytest.raises(MemoryError, match=r"Factorizing the 6561 × 6561 stationary generator"):
        VNA(chip).sweep(frequency, options={"method": "direct"})


@pytest.mark.optional_backend
def test_dynamiqs_steady_state_counts_its_dense_factorization(monkeypatch):
    """The dense solve keeps six D²×D² arrays: the generator, its constrained copy, LU and SVD workspace."""
    pytest.importorskip("dynamiqs")
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q", T1=3e4)
    r = Resonator(freq=6.5, levels=3, label="r", internal_quality_factor=2e5)
    chip = Chip([q, r], [Capacitive(q, r, g=0.04)], frame=5.2, approximation=RWA(), backend="dynamiqs")
    generator = 16 * 81**2

    monkeypatch.setattr(memory, "available_memory_bytes", lambda: 5 * generator)
    with pytest.raises(MemoryError, match="The dense stationary solve at Hilbert dimension D = 9"):
        chip.steadystate()
    monkeypatch.setattr(memory, "available_memory_bytes", lambda: 7 * generator)
    assert chip.steadystate().state.shape == (9, 9)
