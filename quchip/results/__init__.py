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

__all__ = [
    "ObservableTrace",
    "SimulationBatchResult",
    "SimulationResult",
    "PartitionedSimulationResult",
    "wrap_solver_result",
]
