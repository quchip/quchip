"""The memory checks know the available memory and refuse requests that cannot fit."""

import pytest

from quchip.utils import memory

pytestmark = pytest.mark.unit


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
