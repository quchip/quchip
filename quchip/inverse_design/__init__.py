"""Inverse-design utilities.

Fit bare chip parameters (device ``freq``/``anharmonicity``, a
coupling's scalar strength) so that dressed observables (frequencies,
anharmonicities, ``chi``, ``zz``, exchange) match user-supplied
targets. The fit is
classical (``scipy.optimize.least_squares``) but every downstream use
of the returned :class:`~quchip.chip.chip.Chip` remains fully
JAX-traceable and differentiable — see
:func:`quchip.inverse_design.fit.fit_a_dress` for details.
"""

from quchip.inverse_design.fit import fit_a_dress
from quchip.inverse_design.types import FitADressResult, ObservableReport

__all__ = ["fit_a_dress", "FitADressResult", "ObservableReport"]
