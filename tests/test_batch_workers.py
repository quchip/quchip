"""QuTiP batches use the worker count that the caller, the environment and the memory allow."""

import loky
import numpy as np
import psutil
import pytest

from quchip import ChargeDrive, Chip, DuffingTransmon, QuantumSequence, Square
from quchip.backend import qutip as qutip_backend
from quchip.backend.qutip import QuTiPBackend
from quchip.utils import memory


def sweep():
    """Eight drive amplitudes on one three-level transmon."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([q])
    xy = ChargeDrive(q, label="xy")
    chip.wire(xy)
    sequence = QuantumSequence(chip)
    pulse = sequence.schedule(xy, envelope=Square(duration=20.0, amplitude=0.01), freq=chip.freq(q))
    return sequence, pulse.vary("amplitude", np.linspace(0.005, 0.05, 8))


@pytest.fixture
def pool_sizes(monkeypatch):
    """Record the pool size of each parallel batch, and solve its points in this process."""
    sizes = []

    class InProcess:
        def map(self, task, items):
            return map(task, items)

    def executor(workers):
        sizes.append(workers)
        return InProcess()

    monkeypatch.setattr(QuTiPBackend, "_get_executor", staticmethod(executor))
    monkeypatch.setattr(QuTiPBackend, "_pool_workers", 0)
    return sizes


@pytest.mark.unit
def test_pool_size_follows_max_workers_and_loky_cpu_count(pool_sizes, monkeypatch):
    """LOKY_MAX_CPU_COUNT sets the default, max_workers overrides it, and one worker solves in this process."""
    sequence, amplitudes = sweep()
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "2")
    default = sequence.simulate_batch(amplitudes, duration=20.0, progress=False)
    three = sequence.simulate_batch(amplitudes, duration=20.0, progress=False, max_workers=3)
    one = sequence.simulate_batch(amplitudes, duration=20.0, progress=False, max_workers=1)

    assert pool_sizes == [2, 3]
    for result in (three, one):
        np.testing.assert_allclose(result.population("q", 1), default.population("q", 1), rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="max_workers must be a positive integer"):
        sequence.simulate_batch(amplitudes, duration=20.0, progress=False, max_workers=0)


@pytest.mark.unit
def test_pool_size_fits_the_available_memory(pool_sizes, monkeypatch):
    """With memory for three workers, the default runs and records three of eight, and a request for eight raises."""
    sequence, amplitudes = sweep()
    # LOKY_MAX_CPU_COUNT cannot raise the count above the CPUs of the machine, so the test sets the count.
    monkeypatch.setattr(loky, "cpu_count", lambda *args, **kwargs: 8)
    monkeypatch.setattr(memory, "available_memory_bytes", lambda: int(3.5 * qutip_backend._WORKER_BYTES))

    batch = sequence.simulate_batch(amplitudes, duration=20.0, progress=False)
    assert pool_sizes == [3]
    assert {result.stats["workers"] for result in batch} == {3}
    with pytest.raises(MemoryError, match="A batch on 8 worker processes needs about"):
        sequence.simulate_batch(amplitudes, duration=20.0, progress=False, max_workers=8)


@pytest.mark.e2e
def test_two_loky_workers_give_the_results_of_one_process():
    """A real pool of two workers solves the sweep, and the populations match an in-process solve."""
    sequence, amplitudes = sweep()
    pooled = sequence.simulate_batch(amplitudes, duration=20.0, progress=False, max_workers=2)
    workers = [process for process in psutil.Process().children(recursive=True)
               if "LokyProcess" in " ".join(process.cmdline())]
    serial = sequence.simulate_batch(amplitudes, duration=20.0, progress=False, max_workers=1)

    assert len(workers) == 2
    np.testing.assert_allclose(pooled.population("q", 1), serial.population("q", 1), rtol=0, atol=1e-12)


@pytest.mark.unit
def test_only_a_broken_pool_falls_back_to_one_process(monkeypatch):
    """A stopped worker reruns the batch in this process with a warning; other pool errors raise."""
    from concurrent.futures import BrokenExecutor

    sequence, amplitudes = sweep()
    errors = [BrokenExecutor("worker stopped"), RuntimeError("pickling failed")]

    class Failing:
        def map(self, task, items):
            raise errors.pop(0)

    monkeypatch.setattr(QuTiPBackend, "_get_executor", staticmethod(lambda workers: Failing()))
    monkeypatch.setattr(QuTiPBackend, "_shutdown_executor", staticmethod(lambda: None))
    with pytest.warns(UserWarning, match="reran in one process"):
        batch = sequence.simulate_batch(amplitudes, duration=20.0, progress=False, max_workers=2)
    assert {result.stats["workers"] for result in batch} == {1}
    with pytest.raises(RuntimeError, match="pickling failed"):
        sequence.simulate_batch(amplitudes, duration=20.0, progress=False, max_workers=2)
