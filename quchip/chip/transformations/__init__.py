"""Model reduction: ``eliminate()`` and its device, coupling and effective-term target registry.

Public surface only. The extension seams are
:func:`register_elimination_target` for a new target kind and
:func:`register_reduction_method` for a new device reduction route.
:func:`register_elimination_target` is documented on
:mod:`quchip.chip.transformations.dispatch`, and
:func:`register_reduction_method` on
:mod:`quchip.chip.transformations.methods`. Importing this package registers
the shipped handlers (below).
"""
# The shipped-handler registration uses the same side-effect-import pattern as
# the `quchip.chip.retarget` rule registry.

from __future__ import annotations

from quchip.chip.transformations.active_patch import ActivePatchResult, active_patch
from quchip.chip.transformations.dispatch import (
    EliminationTarget,
    eliminate,
    register_elimination_target,
)
from quchip.chip.transformations.methods import register_reduction_method
from quchip.chip.transformations.result import ChipTransform, EliminationResult, ReductionMap

# Import handler modules for their registration side effects. Chip keeps device,
# coupling and effective-term labels disjoint, so registration order is a
# readability choice, not a correctness one.
from quchip.chip.transformations import eliminate_coupling as _eliminate_coupling  # noqa: F401
from quchip.chip.transformations import eliminate_device as _eliminate_device      # noqa: F401
from quchip.chip.transformations import eliminate_effective as _eliminate_effective  # noqa: F401

__all__ = [
    "ActivePatchResult",
    "active_patch",
    "ChipTransform",
    "EliminationResult",
    "ReductionMap",
    "eliminate",
    "EliminationTarget",
    "register_elimination_target",
    "register_reduction_method",
]
