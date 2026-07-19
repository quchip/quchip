"""Chip topology — the composite quantum system and its two-body couplings.

A :class:`Chip` bundles devices, couplings, and (optionally) control
equipment into a single composite system that the engine can assemble
into a solver-ready problem. Dressed-state analysis lives in
:mod:`quchip.chip.analysis` and is attached to every chip as
``chip._analysis`` (exposed via the chip's public methods).
"""

from quchip.chip.analysis import DressedResult
from quchip.chip.baths import Bath
from quchip.chip.chip import Chip
from quchip.chip.couplings import Capacitive, Coupling, CrossKerr, TunableCapacitive
from quchip.chip.retarget import register_retarget_rule
from quchip.chip.transformations import (
    ActivePatchResult,
    ChipTransform,
    EliminationResult,
    active_patch,
    eliminate,
    register_elimination_target,
    register_reduction_method,
)

__all__ = [
    "Chip",
    "DressedResult",
    "Bath",
    "Capacitive",
    "Coupling",
    "CrossKerr",
    "TunableCapacitive",
    "ChipTransform",
    "EliminationResult",
    "eliminate",
    "ActivePatchResult",
    "active_patch",
    "register_retarget_rule",
    "register_elimination_target",
    "register_reduction_method",
]
