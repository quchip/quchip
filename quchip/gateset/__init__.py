"""Gate sets: calibrated native gates, their cached propagators, and circuits.

A :class:`GateSet` holds native gates on a chip. Each :class:`NativeGate` is a
``QuantumSequence`` fragment with named :class:`Knob` parameters and an ideal
unitary. :meth:`GateSet.calibrate` fits the knobs and the post-gate virtual-Z
phases against the ideal, and caches each gate's propagator. Circuits compile
to a pulse schedule (:meth:`GateSet.schedule`), to a propagator composed from
the cache (:meth:`GateSet.unitary`), or to a simulation
(:meth:`GateSet.simulate`).
"""

from quchip.gateset.builtins import cross_resonance, drag_sx
from quchip.gateset.calibration import GateCalibration, KnobResult
from quchip.gateset.gateset import CircuitRun, CircuitUnitary, GateSet
from quchip.gateset.native import Knob, NativeGate, Seed

__all__ = [
    "CircuitRun",
    "CircuitUnitary",
    "GateCalibration",
    "GateSet",
    "Knob",
    "KnobResult",
    "NativeGate",
    "Seed",
    "cross_resonance",
    "drag_sx",
]
