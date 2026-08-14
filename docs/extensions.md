# Extend quchip

An extension declares local physics. quchip projects its operators into the
resolved basis, embeds them in the chip Hilbert space, applies frames and RWA,
and lowers the result to the selected backend. Extension code should not import
engine IR or branch on a backend.

## Choose the surface by ownership

| Physics | Author writes | Installed reference |
|---|---|---|
| Static device | parameters and `DeviceModel.local_hamiltonian()` | `DuffingTransmon` |
| Static coupling | endpoints, parameters, and `CouplingModel.interaction()` | `Capacitive` |
| Component-owned time dependence | `DeviceModel.time_terms()` or `CouplingModel.time_terms()` returning `TimeDependentTerm` | `FrequencyModulatedMode`, `ModulatedCapacitive` |
| Device control | `DeviceDrive.hamiltonian()` | `ChargeDrive`, `ChargePhaseDrive` |
| Coupling control | `CouplingDrive.hamiltonian()` | `ParametricDrive` |
| Classical hardware effect | `SignalTransform.apply()` | `CableLoss`, `Gain`, `Delay`, `Crosstalk` |
| Scheduled waveform | `Envelope.value()` | `CosineEnvelope`, `GaussianDRAG` |
| Unscheduled coefficient | `TimeCoefficient.value()` | `CosineCoefficient` |
| Device, drive, coupling, or bath loss | `dissipation()` returning `CollapseChannel` | `LossyKerrCavity`, `LossyChargeDrive`, `CollectiveDecayCoupling`, `Bath` |
| Custom local space | `LocalSpace.matrix()` with an explicit operator vocabulary | `SpinHalf` |
| Third-party conversion | `ModelMapping.import_model()` and `ModelMapping.export_model()` | scqubits mappings |

Specialized examples are importable from `quchip.extensions`. Their tests cover
analytical physics, serialization, projection, RWA, and JAX paths where each
property applies.

The reference files mirror the concepts they extend:

```text
quchip/extensions/
    devices.py
    couplings.py
    drives.py
    envelopes.py
    signals.py
    spaces.py
```

Loss examples live with their owning device, drive, or coupling.

## Static devices

Declare parameters and the Hamiltonian in the device's local operator
vocabulary. The constructor, validation, serialization, and JAX pytree are
synthesized from the fields.

```python
from typing import Any

from quchip import FockDevice, LocalOps, PhysicsExpr, Scalar, parameter


class Mode(FockDevice):
    _default_levels = 8

    freq: Scalar = parameter(positive=True, unit="GHz")

    def local_hamiltonian(self, op: LocalOps, p: Any) -> PhysicsExpr:
        return p.freq * op.n
```

`p.freq` is symbolic during authoring. The engine binds the instance value when
it resolves the model for a backend.

## Component-owned time dependence

Use `time_terms()` when a device or coupling changes with time without a
scheduled control pulse.

```python
from typing import Any

from quchip import CosineCoefficient, Scalar, TimeDependentTerm, parameter


class FrequencyModulatedMode(Mode):
    modulation_amplitude: Scalar = parameter(unit="GHz")
    modulation_frequency: Scalar = parameter(positive=True, unit="GHz")

    def time_terms(self, op, p: Any) -> tuple[TimeDependentTerm, ...]:
        return (
            TimeDependentTerm(
                operator=op.n,
                coefficient=CosineCoefficient(
                    amplitude=p.modulation_amplitude,
                    frequency=p.modulation_frequency,
                ),
            ),
        )
```

A coupling uses the same shape with both endpoint vocabularies:

```python
from quchip import CouplingModel


class ModulatedExchange(CouplingModel):
    static_strength: Scalar = parameter(unit="GHz")
    modulation_amplitude: Scalar = parameter(unit="GHz")
    modulation_frequency: Scalar = parameter(positive=True, unit="GHz")

    def interaction(self, a, b, p):
        return p.static_strength * (a.a * b.adag + a.adag * b.a)

    def time_terms(self, a, b, p):
        exchange = a.a * b.adag + a.adag * b.a
        return (
            TimeDependentTerm(
                operator=exchange,
                coefficient=CosineCoefficient(
                    amplitude=p.modulation_amplitude,
                    frequency=p.modulation_frequency,
                ),
            ),
        )
```

These terms are projected into the basis selected from the static Hamiltonian.
quchip does not construct an instantaneous moving basis.

## Drives and envelopes

A drive maps a delivered classical signal to local quantum physics. A drive
instance names the device or coupling connected to that physical line.

```python
from quchip import DeviceDrive


class ChargeLikeDrive(DeviceDrive):
    def hamiltonian(self, target, signal):
        return signal.i * target.charge_coupling_operator()


mode = Mode(freq=5.0, label="q")
line = ChargeLikeDrive(mode, label="xy")
```

`signal.i` and `signal.q` are the physical in-phase and quadrature components
after gain, delay, filtering, distortion, and crosstalk. A drive may couple
them to different observables or use nonlinear combinations:

```python
class IQDrive(DeviceDrive):
    def hamiltonian(self, target, signal):
        return (
            signal.i * target.x_control_operator()
            - signal.q * target.y_control_operator()
        )
```

Override `signal(pulse, target)` only when constructing the scheduled analytic
signal itself needs drive-specific physics. Carriers are optional for every
drive. Approximation choices belong to the chip and engine, not the drive.

Declare drive values with `parameter()` and structural choices with
`setting()`. Constructors and serialization are synthesized. Define a
scheduled waveform by subclassing `Envelope` and implementing `value(t)`;
define an unscheduled scalar coefficient by subclassing `TimeCoefficient` and
implementing `value(t)`. Both implementations should use `quchip.qnp` to remain
JAX-traceable.

```python
from quchip import Envelope, qnp


class CosineEnvelope(Envelope):
    duration: Scalar = parameter(positive=True, unit="ns")
    amplitude: Scalar = parameter(default=1.0)

    def value(self, local_time):
        return 0.5 * self.amplitude * (
            1.0 - qnp.cos(2.0 * qnp.pi * local_time / self.duration)
        )
```

Use a complex envelope for IQ control. `GaussianDRAG` implements
$E(t) = I(t) + i\beta\,dI/dt$, with signed $\beta$ in ns. An IQ envelope is one
complex control signal, not two drive lines.

An envelope owns local shape and relative I/Q only. Pass global phase to
`sequence.schedule(..., phase=...)`; timing and carrier belong to that
scheduled pulse as well.

Control equipment transforms complete analytic signals before
`hamiltonian()`. A delayed crosstalk copy therefore retains the source carrier,
both quadratures, and the carrier phase accumulated during the delay.

## Dissipation

Declare a configurable loss rate with `parameter(noise=True)`. Return the
unscaled jump operator and its rate separately:

```python
from quchip import CollapseChannel, parameter


class TwoPhotonLossMode(Mode):
    two_photon_loss_rate: Scalar = parameter(
        nonnegative=True,
        unit="1/ns",
        noise=True,
    )

    def dissipation(self, op, p):
        return super().dissipation(op, p) + (
            CollapseChannel(
                operator=op.a @ op.a,
                rate=p.two_photon_loss_rate,
                name="two_photon_loss",
            ),
        )
```

Rates use `1/ns`. The backend applies `sqrt(rate)` when constructing a
Lindblad operator. Devices, drives, and couplings author channels on their
local support. A `Bath` can author a channel across several targets.

## Custom spaces and interop

A custom local space defines its dimension and the matrices available by name.
Its `LocalSpace.matrix()` implementation must return a matrix with that fixed
dimension. See `quchip.extensions.SpinHalf` for a complete two-level example.

Subclass `ModelMapping` when a third-party object needs an explicit conversion.
Set `source` for import, `target` and `library` for export, and implement only
the directions the mapping supports. Importing `quchip.extensions` does not
load optional scqubits modules.

## Control equipment

Subclass `SignalTransform` for a classical hardware effect that acts on complete
analytic signals. Implement `apply(signals)` and return a new signal map. Declare
numeric fields in `_parameter_names` so sweeps can rebind them without mutating
the transform.

```python
from dataclasses import dataclass

from quchip.control import SignalTransform
from quchip.declarative import qnp


@dataclass(frozen=True)
class CableLoss(SignalTransform):
    line: str
    loss_db: float
    _parameter_names = ("loss_db",)

    def apply(self, signals):
        factor = qnp.power(10.0, -self.loss_db / 20.0)
        return {
            key: signal.scaled(factor) if key[0] == self.line else signal
            for key, signal in signals.items()
        }
```

Transforms receive the output of the preceding transform. `CrosstalkMatrix`
reads one shared input snapshot so its off-diagonal paths mix simultaneously.

## Checks for a new extension

Test only the properties the extension claims:

1. an analytical spectrum, operator identity, or decay law;
2. native and projected basis behavior where projection applies;
3. full and RWA behavior for excitation-changing terms;
4. units and amplitude normalization;
5. serialization;
6. a finite, nonzero JAX gradient through each differentiable parameter;
7. the complete `Chip` or `QuantumSequence` path;
8. both backends only when backend lowering changes.

If the engine must branch on an extension's concrete class, the model needs a
public physical capability instead.
