"""Model reduction: ``eliminate()`` and its device, coupling and effective-term targets.

Importing this package registers the shipped target handlers (below).
"""

from __future__ import annotations

from quchip.chip.transformations.active_patch import (
    ActivePatchResult,
    SpectatorEliminationResult,
    active_patch,
    eliminate_spectators,
)
from quchip.chip.transformations.dispatch import EliminationTarget, eliminate
from quchip.chip.sw import EliminationError
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
    "EliminationError",
    "EliminationResult",
    "ReductionMap",
    "eliminate",
    "eliminate_spectators",
    "SpectatorEliminationResult",
    "EliminationTarget",
]
