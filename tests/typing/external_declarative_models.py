"""Static contract for third-party declarative model constructors."""

from typing import Any

from quchip import CouplingDrive, DeviceDrive, Envelope, TimeCoefficient
from quchip.declarative import CouplingModel, DeviceModel, Scalar, parameter, qnp


class ExternalMode(DeviceModel):
    """Minimal third-party device."""

    freq: Scalar = parameter(positive=True)

    def local_hamiltonian(self, op: Any, p: Any) -> Any:
        return p.freq * op.n


class ExternalCoupling(CouplingModel):
    """Minimal third-party coupling."""

    strength: Scalar = parameter()

    def interaction(self, a: Any, b: Any, p: Any) -> Any:
        return p.strength * a.n * b.n


class ExternalDeviceDrive(DeviceDrive):
    """Minimal third-party device drive."""

    gain: Scalar = parameter()

    def hamiltonian(self, target: Any, signal: Any) -> Any:
        return self.gain * signal.i * target.number_operator()


class ExternalCouplingDrive(CouplingDrive):
    """Minimal third-party coupling drive."""

    gain: Scalar = parameter()

    def hamiltonian(self, target: Any, signal: Any) -> Any:
        return self.gain * signal.i * target.interaction_hamiltonian()


class ExternalEnvelope(Envelope):
    """Minimal third-party envelope."""

    duration: Scalar = parameter(positive=True)
    amplitude: Scalar = parameter()

    def value(self, local_time: Any) -> Any:
        return self.amplitude * qnp.asarray(local_time)


class ExternalCoefficient(TimeCoefficient):
    """Minimal third-party time coefficient."""

    amplitude: Scalar = parameter()

    def value(self, time: Any) -> Any:
        return self.amplitude * qnp.asarray(time)


mode_a = ExternalMode(freq=5.0, levels=3, label=None, T1=10_000.0)
mode_b = ExternalMode(freq=5.2, levels=3, label="b")
coupling = ExternalCoupling(mode_a, mode_b, strength=0.01, label="ab")
device_drive = ExternalDeviceDrive(mode_a, gain=0.5, label=None)
coupling_drive = ExternalCouplingDrive(coupling, gain=0.4, label="pump")
envelope = ExternalEnvelope(duration=10.0, amplitude=0.2)
coefficient = ExternalCoefficient(amplitude=0.3)
