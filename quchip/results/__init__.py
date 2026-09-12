"""Backend-agnostic simulation output containers.

Whatever backend produced the solver output (QuTiP, dynamiqs/JAX, …), users
interact only with :class:`SimulationResult` and
:class:`SimulationBatchResult`. Named expectation-value traces use
:class:`ObservableTrace`; external-plane amplitude, quadrature, and photon
flux use :class:`OutputFieldTrace`. Raw backend output is wrapped into a
:class:`SimulationResult` via :func:`wrap_solver_result`. A partitioned
solve (see :mod:`quchip.engine.partitioned`) combines its per-component
results into a :class:`~quchip.results.partitioned.PartitionedSimulationResult`.
"""

from quchip.results.partitioned import PartitionedSimulationResult
from quchip.results.results import (
    ObservableTrace,
    OutputFieldTrace,
    SimulationBatchResult,
    SimulationResult,
    wrap_solver_result,
)
from quchip.results.trajectories import TrajectoryResult
from quchip.results.steady_state import SteadyStateBatchResult, SteadyStateResult
from quchip.results.input_output import (
    OutputCorrelationResult,
    OutputSpectrumResult,
    MeanFieldResponseResult,
    SParameterResult,
)

from quchip.results.receiver import IQReceiver
from quchip.results.measurement import VNAMeasurement, VNAMeasurementSamples, VNAMeasurementStatistics

from quchip.results.terminal import IQReadout, StateMeasurement, StateSamples

__all__ = [
    "IQReadout",
    "StateMeasurement",
    "StateSamples",
    "IQReceiver",
    "VNAMeasurement",
    "VNAMeasurementSamples",
    "VNAMeasurementStatistics",
    "ObservableTrace",
    "OutputFieldTrace",
    "SimulationBatchResult",
    "SimulationResult",
    "TrajectoryResult",
    "SteadyStateResult",
    "SteadyStateBatchResult",
    "MeanFieldResponseResult",
    "SParameterResult",
    "OutputSpectrumResult",
    "OutputCorrelationResult",
    "PartitionedSimulationResult",
    "wrap_solver_result",
]
