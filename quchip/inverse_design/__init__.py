"""Inverse design from a numerical dressed-chip specification.

Component classes declare which input numbers are dressed targets and which
bare parameters normally move. :func:`fit_a_dress` compiles those declarations
without evaluating the desired chip, then solves the resulting static
observable problem with SciPy and an exact JAX Jacobian when available. The
fitted chip remains traceable and differentiable downstream.
"""

from quchip.inverse_design.fit import fit_a_dress
from quchip.inverse_design.types import FitADressResult, FitParameterReport, ObservableReport

__all__ = ["fit_a_dress", "FitADressResult", "FitParameterReport", "ObservableReport"]
