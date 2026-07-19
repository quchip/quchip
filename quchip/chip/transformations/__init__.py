"""Model reduction: ``eliminate()`` and its device/coupling target registry.

Public surface only; the extension seams (:func:`register_elimination_target`
for a new target kind, :func:`register_reduction_method` for a new device
reduction route) are documented on :mod:`quchip.chip.transformations.dispatch`
and :mod:`quchip.chip.transformations.methods` respectively. Importing this
package runs the shipped handlers' self-registration (below) — the same
side-effect-import pattern as :mod:`quchip.chip.retarget`'s rule registry.
"""

from __future__ import annotations

from quchip.chip.transformations.active_patch import ActivePatchResult, active_patch
from quchip.chip.transformations.dispatch import (
    EliminationTarget,
    eliminate,
    register_elimination_target,
)
from quchip.chip.transformations.methods import register_reduction_method
from quchip.chip.transformations.result import ChipTransform, EliminationResult

# Import handler modules for their registration side effects. Coupling first so it
# claims the coupling namespace before the device handler is consulted (the two
# namespaces are disjoint, so order is a readability choice, not a correctness one).
from quchip.chip.transformations import eliminate_coupling as _eliminate_coupling  # noqa: F401
from quchip.chip.transformations import eliminate_device as _eliminate_device      # noqa: F401

__all__ = [
    "ActivePatchResult",
    "active_patch",
    "ChipTransform",
    "EliminationResult",
    "eliminate",
    "EliminationTarget",
    "register_elimination_target",
    "register_reduction_method",
]
