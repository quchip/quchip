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


@pytest.fixture
def scqubits_mhz():
    """Set scqubits' global energy unit to MHz for one test and restore the previous unit."""
    scqubits = pytest.importorskip("scqubits")
    previous = scqubits.get_units()
    scqubits.set_units("MHz")
    yield
    scqubits.set_units(previous)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Assign a daily suite and retain the independent location markers."""
    for item in items:
        suites = {name for name in ("unit", "e2e", "validation") if item.get_closest_marker(name)}
        if len(suites) > 1:
            raise pytest.UsageError(f"Conflicting test suites for {item.nodeid}: {sorted(suites)}")
        if not suites:
            item.add_marker(pytest.mark.e2e)

        rel_path = Path(str(item.fspath)).resolve().relative_to(PROJECT_ROOT)
        filename = rel_path.name

        if "physics_sentinel" in rel_path.parts:
            item.add_marker(pytest.mark.physics_sentinel)
            continue

        if "examples" in rel_path.parts:
            item.add_marker(pytest.mark.examples)
            continue

        if "extended" in rel_path.parts:
            item.add_marker(pytest.mark.extended)
            if filename in VIZ_FILES:
                item.add_marker(pytest.mark.viz)
            continue

        item.add_marker(pytest.mark.core)
