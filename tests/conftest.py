"""Shared fixtures for the quchip test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from quchip.backend import reset_default_backend
from quchip.backend.qutip import QuTiPBackend
from quchip.utils.labeling import reset_label_counters


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIZ_FILES = {
    "test_viz_chip_control.py",
    "test_viz_results_device.py",
}


@pytest.fixture(autouse=True)
def _clean_backend_state():
    """Reset module-level backend state after every test for isolation."""
    yield
    reset_default_backend()


@pytest.fixture(autouse=True)
def _reset_labels():
    """Reset auto-label counters before every test so tests are order-independent."""
    reset_label_counters()
    yield


@pytest.fixture
def backend() -> QuTiPBackend:
    """Return a fresh QuTiPBackend instance for each test."""
    return QuTiPBackend()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Assign test-lane markers from the folder layout."""
    for item in items:
        rel_path = Path(str(item.fspath)).resolve().relative_to(PROJECT_ROOT)
        filename = rel_path.name

        if "physics_sentinel" in rel_path.parts:
            item.add_marker(pytest.mark.physics_sentinel)
            continue

        if "extended" in rel_path.parts:
            item.add_marker(pytest.mark.extended)
            if filename in VIZ_FILES:
                item.add_marker(pytest.mark.viz)
            continue

        item.add_marker(pytest.mark.core)
