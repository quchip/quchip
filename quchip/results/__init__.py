"""Backend-agnostic simulation output containers.

Whatever backend produced the solver output (QuTiP, dynamiqs/JAX, …), users
interact only with :class:`SimulationResult` and
:class:`SimulationBatchResult`. Named expectation-value traces are exposed
as :class:`ObservableTrace`, and raw backend output is wrapped into a
:class:`SimulationResult` via :func:`wrap_solver_result`. A partitioned
solve (see :mod:`quchip.engine.partitioned`) combines its per-component
results into a :class:`~quchip.results.partitioned.PartitionedSimulationResult`.
"""

from quchip.results.partitioned import PartitionedSimulationResult
from quchip.results.results import ObservableTrace, SimulationBatchResult, SimulationResult, wrap_solver_result
from quchip.results.steady_state import SteadyStateBatchResult, SteadyStateResult
from quchip.results.input_output import (
    OutputCorrelationResult,
    OutputSpectrumResult,
    SParameterResult,
)

__all__ = [
    "ObservableTrace",
    "SimulationBatchResult",
    "SimulationResult",
    "SteadyStateResult",
    "SteadyStateBatchResult",
    "SParameterResult",
    "OutputSpectrumResult",
    "OutputCorrelationResult",
    "PartitionedSimulationResult",
    "wrap_solver_result",
]
