"""Chip topology: the composite quantum system and its two-body couplings.

A :class:`Chip` bundles devices, couplings, and (optionally) control equipment
into one composite system, which the engine assembles into a solver-ready
problem. Dressed-state analysis lives in :mod:`quchip.chip.analysis`.
"""
# Every chip holds the dressed-state analysis as `chip._analysis` and exposes it
# via the chip's public methods.

from quchip.chip.analysis import DressedResult, KerrMatrix
from quchip.chip.baths import Bath
from quchip.chip.chip import Chip
from quchip.chip.port_network import ComponentPort, NetworkPort, PortNetwork
from quchip.chip.ports import Port
from quchip.chip.couplings import Capacitive, Coupling, CrossKerr, TunableCapacitive
from quchip.chip.retarget import register_retarget_rule
from quchip.chip.transformations import (
    ActivePatchResult,
    ChipTransform,
    EliminationResult,
    ReductionMap,
    active_patch,
    eliminate,
    register_elimination_target,
    register_reduction_method,
)

__all__ = [
    "Chip",
    "DressedResult",
    "KerrMatrix",
    "Bath",
    "Port",
    "PortNetwork",
    "ComponentPort",
    "NetworkPort",
    "Capacitive",
    "Coupling",
    "CrossKerr",
    "TunableCapacitive",
    "ChipTransform",
    "EliminationResult",
    "ReductionMap",
    "eliminate",
    "ActivePatchResult",
    "active_patch",
    "register_retarget_rule",
    "register_elimination_target",
    "register_reduction_method",
]
