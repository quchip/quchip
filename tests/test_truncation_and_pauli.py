"""Tests for truncation warnings on SimulationResult."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import qutip as qt

from quchip.backend import get_default_backend, SolverResult
from quchip.results.results import SimulationResult

def _fake_result_with_top_pop(dev_dim: int, top_pop: float) -> SimulationResult:
    """Build a SimulationResult whose final state has population ``top_pop`` in |dev_dim-1>."""
    backend = get_default_backend()
    amps = np.zeros(dev_dim, dtype=complex)
    amps[0] = np.sqrt(max(0.0, 1.0 - top_pop))
    amps[-1] = np.sqrt(top_pop)
    ket = qt.Qobj(amps.reshape(-1, 1), dims=[[dev_dim], [1]])
    solver_result = SolverResult(
        times=np.array([0.0, 1.0]),
        states=None,
        expect=None,
        final_state=ket,
        stats={},
        solver="sesolve",
    )
    return SimulationResult(
        solver_result=solver_result,
        backend=backend,
        dims=[dev_dim],
        device_info=[("q0", True)],
        observable_traces=None,
    )


class TestTruncationWarnings:
    def test_warns_when_top_level_pop_exceeds_threshold(self) -> None:
        """check_truncation warns when the top-level population exceeds the threshold."""
        result = _fake_result_with_top_pop(dev_dim=4, top_pop=0.01)  # 1 %
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            observed = result.check_truncation(threshold=1e-3)
        assert any(issubclass(w.category, UserWarning) and "q0" in str(w.message) for w in caught)
        assert observed["q0"] == pytest.approx(0.01, abs=1e-10)

    def test_silent_when_top_level_pop_below_threshold(self) -> None:
        """check_truncation stays silent when the top-level population is below the threshold."""
        result = _fake_result_with_top_pop(dev_dim=4, top_pop=1e-6)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            observed = result.check_truncation(threshold=1e-3)
        assert not any(issubclass(w.category, UserWarning) for w in caught)
        assert observed["q0"] == pytest.approx(1e-6, abs=1e-12)

    def test_returns_empty_when_no_final_state(self) -> None:
        """check_truncation returns an empty mapping and warns nothing when there is no final state."""
        backend = get_default_backend()
        solver_result = SolverResult(
            times=np.array([0.0]),
            states=None,
            expect=None,
            final_state=None,
            stats={},
            solver="sesolve",
        )
        result = SimulationResult(
            solver_result=solver_result,
            backend=backend,
            dims=[3],
            device_info=[("q0", True)],
            observable_traces=None,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            observed = result.check_truncation()
        assert observed == {}
        assert not any(issubclass(w.category, UserWarning) for w in caught)
