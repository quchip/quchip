"""Classical control surface: lines, signal chain, and envelopes."""

from quchip.control.equipment import ControlEquipment, CrosstalkMatrix
from quchip.control.signal import Crosstalk, Delay, Gain, SignalTransform
from quchip.control.signal_spec import DriveSignalSpec, DriveModulation
from quchip.control.drive import (
    BaseDrive,
    ChargeDrive,
    DriveChannel,
    FluxDrive,
    ParametricDrive,
    PhaseDrive,
)
from quchip.control.drives_two_photon import TwoPhotonDrive
from quchip.control.envelopes import (
    Gaussian,
    GaussianEdge,
    LinearRamp,
    Square,
    SquareWithGaussianEdges,
)

__all__ = [
    # Drive classes
    "BaseDrive",
    "SignalTransform",
    "Crosstalk",
    "Delay",
    "Gain",
    "DriveSignalSpec",
    "DriveModulation",
    "ChargeDrive",
    "DriveChannel",
    "FluxDrive",
    "ParametricDrive",
    "PhaseDrive",
    "TwoPhotonDrive",
    "ControlEquipment",
    "CrosstalkMatrix",
    # Pulse envelopes
    "Gaussian",
    "GaussianEdge",
    "LinearRamp",
    "Square",
    "SquareWithGaussianEdges",
]
