# Extending quchip

An extension declares local physics. quchip projects its operators into the
resolved basis and embeds them in the chip Hilbert space. It then applies
frames and RWA and converts the result for the selected backend. Extension
code returns quchip expressions and need not choose a numerical backend.

## Choose an extension type

| Physics | Author writes | Built in | On this page |
|---|---|---|---|
| Static device | parameters and `DeviceModel.local_hamiltonian()` | `DuffingTransmon` | `Mode` |
| Static coupling | endpoints, parameters, and `CouplingModel.interaction()` | `Capacitive` | |
| Component-owned time dependence | `DeviceModel.time_terms()` or `CouplingModel.time_terms()` returning `TimeDependentTerm` | | `FrequencyModulatedMode`, `ModulatedCapacitive` |
| Device control | `DeviceDrive.hamiltonian()` | `ChargeDrive` | `ChargeLikeDrive`, `ChargePhaseDrive` |
| Coupling control | `CouplingDrive.hamiltonian()` | `ParametricDrive` | |
| Classical hardware effect | `SignalTransform.apply()` | `Gain`, `Delay`, `Crosstalk` | `CableLoss` |
| Scheduled waveform | `Envelope.value()` | `CosineEnvelope`, `GaussianDRAG` | `HalfSine` |
| Unscheduled coefficient | `TimeCoefficient.value()` | `CosineCoefficient` | |
| Device, drive, or coupling loss | `dissipation()` returning `CollapseChannel` | device `T1` and `T2` | `LossyKerrCavity`, `LossyChargeDrive`, `CollectiveDecayCoupling` |
| Custom local space | `LocalSpace.matrix()` with an explicit operator vocabulary | | `SpinHalf` |
| Third-party conversion | `ModelMapping.import_model()` and `ModelMapping.export_model()` | scqubits mappings | |

The classes on this page are complete, but quchip does not ship them. Copy a
class into your own code to use it. The quchip tests run the code on this page
and check the physics of each class: analytical limits, serialization,
projection, RWA, and JAX paths where each property applies.

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
scheduled control pulse. The frequency of this mode is
$\omega_0 + \delta\omega\cos(2\pi\nu_m t + \phi_m)$, with frequencies in
GHz and $t$ in ns:

```python
from quchip import CosineCoefficient, TimeDependentTerm


class FrequencyModulatedMode(FockDevice):
    _default_levels = 10
    approximation = (
        "Single harmonic mode in a fixed Fock basis with an externally prescribed "
        "sinusoidal frequency coefficient."
    )

    frequency: Scalar = parameter(positive=True, unit="GHz", symbol=r"\omega_0")
    modulation_amplitude: Scalar = parameter(unit="GHz", symbol=r"\delta\omega")
    modulation_frequency: Scalar = parameter(positive=True, unit="GHz", symbol=r"\nu_m")
    modulation_phase: Scalar = parameter(default=0.0, unit="rad", symbol=r"\phi_m")

    @property
    def freq(self) -> Any:
        return self.frequency

    def local_hamiltonian(self, op: LocalOps, p: Any) -> PhysicsExpr:
        return p.frequency * op.n

    def time_terms(self, op: LocalOps, p: Any) -> tuple[TimeDependentTerm, ...]:
        return (
            TimeDependentTerm(
                operator=op.n,
                coefficient=CosineCoefficient(
                    amplitude=p.modulation_amplitude,
                    frequency=p.modulation_frequency,
                    phase=p.modulation_phase,
                ),
            ),
        )
```

`freq` is the reference frequency that frames and automatic carriers use.
`approximation` states the model's regime, and `chip.physics_notes()` reports
it.

A coupling uses the same shape with both endpoint vocabularies. This one
modulates a capacitive coupling between the endpoints' charge operators:

```python
from quchip import CouplingModel


class ModulatedCapacitive(CouplingModel):
    static_strength: Scalar = parameter(unit="GHz", symbol=r"g_0")
    modulation_amplitude: Scalar = parameter(unit="GHz", symbol=r"\delta g")
    modulation_frequency: Scalar = parameter(positive=True, unit="GHz", symbol=r"\nu_m")
    modulation_phase: Scalar = parameter(default=0.0, unit="rad", symbol=r"\phi_m")

    def interaction(self, a, b, p):
        return p.static_strength * a.charge * b.charge

    def time_terms(self, a, b, p):
        return (
            TimeDependentTerm(
                operator=a.charge * b.charge,
                coefficient=CosineCoefficient(
                    amplitude=p.modulation_amplitude,
                    frequency=p.modulation_frequency,
                    phase=p.modulation_phase,
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
    carrier_harmonic = 1

    def hamiltonian(self, target, signal):
        return signal.i * target.charge_coupling_operator()


mode = Mode(freq=5.0, label="q")
line = ChargeLikeDrive(mode, label="xy")
```

`signal.i` and `signal.q` are the physical in-phase and quadrature components
after gain, delay, filtering, distortion, and crosstalk. A drive can couple
them to different observables or use nonlinear combinations. This drive
couples $I$ to the charge $\hat Q$ and $Q$ to the phase $\hat P$, so that
$H_d = I\hat Q - Q\hat P$:

```python
from quchip import ChargeCoupled, PhaseCoupled, as_operator_expr


class ChargePhaseDrive(DeviceDrive):
    carrier_harmonic = 1

    def hamiltonian(self, target, signal):
        if not isinstance(target, ChargeCoupled) or not isinstance(target, PhaseCoupled):
            raise TypeError(
                f"ChargePhaseDrive requires {type(target).__name__} to define "
                "charge_coupling_operator() and phase_coupling_operator()."
            )
        label = target.label
        dims = (target.local_space().dimension,)
        charge = as_operator_expr(target.charge_coupling_operator(), labels=(label,), dims=dims,
                                  name=rf"\hat Q_{{{label}}}")
        phase = as_operator_expr(target.phase_coupling_operator(), labels=(label,), dims=dims,
                                 name=rf"\hat P_{{{label}}}")
        return signal.i * charge - signal.q * phase
```

On a Fock device this is $i(s\,a^\dagger - s^* a)$ for the delivered signal
$s = I + iQ$, so a positive carrier drives the resonant sideband at twice the
rate of `ChargeDrive` and RWA drops nothing. `PHYSICS.md` gives the sideband
weights on other devices.

Override `signal(pulse, target)` only when constructing the scheduled analytic
signal itself needs drive-specific physics. Approximation choices belong to the
chip and engine, not the drive.

The class attribute `carrier_harmonic` sets the carrier of a pulse scheduled
without `freq`. With `carrier_harmonic = k`, the carrier is
`k * chip.freq(target)`, and quchip warns with the chosen value. Set it only
when that carrier makes the drive's weight-`k` bands co-rotating, as both
drives above do. A drive without it keeps such a pulse at baseband.
`freq=None` gives a baseband pulse on every line.

Declare drive values with `parameter()` and structural choices with
`setting()`. Constructors and serialization are synthesized. Define a scheduled
waveform by subclassing `Envelope` and implementing `value(t)`. Define an
unscheduled scalar coefficient by subclassing `TimeCoefficient` and
implementing `value(t)`. Both implementations should use `quchip.qnp` to remain
JAX-traceable. This half-sine pulse rises from zero to its peak at mid-pulse:

```python
from quchip import Envelope, qnp


class HalfSine(Envelope):
    duration: Scalar = parameter(positive=True, unit="ns")
    amplitude: Scalar = parameter(default=1.0)

    def value(self, local_time):
        return self.amplitude * qnp.sin(qnp.pi * local_time / self.duration)
```

The default `area()` integrates `value()` numerically between the
`sampling_times()`. Override either one when the shape has a closed-form area
or a narrow feature.

Use a complex envelope for IQ control. `GaussianDRAG` implements
$E(t) = I(t) + i\beta\,dI/dt$, with signed $\beta$ in ns. An IQ envelope is one
complex control signal, not two drive lines.

An envelope owns local shape and relative I/Q only. Pass global phase to
`sequence.schedule(..., phase=...)`. Timing and carrier belong to that
scheduled pulse as well.

Control equipment transforms complete analytic signals before `hamiltonian()`.
A delayed crosstalk copy therefore keeps the source carrier, both quadratures,
and the carrier phase accumulated during the delay.

## Dissipation

Declare a configurable loss rate with `parameter(noise=True)`. Return the
unscaled jump operator and its rate separately. This Kerr cavity adds
two-photon loss to the channels it inherits:

```python
from quchip import CollapseChannel, KerrCavity


class LossyKerrCavity(KerrCavity):
    two_photon_loss_rate: Scalar = parameter(
        nonnegative=True,
        unit="1/ns",
        symbol=r"\kappa_2",
        noise=True,
        kw_only=True,
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
local support. A drive's channel acts on its target device:

```python
from quchip import ChargeDrive


class LossyChargeDrive(ChargeDrive):
    line_loss_rate: Scalar = parameter(nonnegative=True, unit="1/ns", noise=True, kw_only=True)

    def dissipation(self, device, op, p):
        return (CollapseChannel(op.a, p.line_loss_rate, "line_relaxation"),)
```

A coupling's channel can act on both of its devices. This coupling adds one
shared decay channel with the jump operator $a + b$, not two independent
relaxation channels:

```python
class CollectiveDecayCoupling(CouplingModel):
    exchange_strength: Scalar = parameter(unit="GHz", symbol="g")
    decay_rate: Scalar = parameter(
        nonnegative=True,
        unit="1/ns",
        symbol=r"\gamma_c",
        noise=True,
        kw_only=True,
    )

    def interaction(self, a, b, p):
        return p.exchange_strength * (a.a * b.adag + a.adag * b.a)

    def dissipation(self, a, b, p):
        return (CollapseChannel(a.a * b.I + a.I * b.a, p.decay_rate, "collective_decay"),)
```

## Custom spaces and interop

A custom local space defines its dimension and the matrices available by name.
Its `LocalSpace.matrix()` implementation must return a matrix with that fixed
dimension. This two-level spin has the Hamiltonian $H = -\omega\sigma_z/2$ on
a `CustomSpace` with its own operator vocabulary:

```python
from typing import Literal

import jax.numpy as jnp

from quchip import CustomSpace, DeviceModel, setting


def spin_operators() -> dict[str, Any]:
    sigma_x = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
    sigma_y = jnp.asarray([[0.0, -1j], [1j, 0.0]], dtype=jnp.complex128)
    sigma_z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)
    sigma_plus = jnp.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.complex128)
    sigma_minus = sigma_plus.conj().T
    return {
        "I": jnp.eye(2, dtype=jnp.complex128),
        "a": sigma_minus,
        "adag": sigma_plus,
        "n": sigma_plus @ sigma_minus,
        "charge": sigma_x,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "sigma_z": sigma_z,
        "sigma_plus": sigma_plus,
        "sigma_minus": sigma_minus,
    }


class SpinHalf(DeviceModel):
    _default_levels = 2
    computational = True
    approximation = "Exact two-level spin Hamiltonian in a fixed custom basis."

    basis: Literal["native", "eigen"] | None = setting(default=None, kw_only=True)
    freq: Scalar = parameter(positive=True, unit="GHz", symbol=r"\omega")

    def local_space(self) -> CustomSpace:
        return CustomSpace(2, spin_operators())

    def validate(self) -> None:
        if self.levels != 2:
            raise ValueError(f"SpinHalf has a fixed two-level local space; levels must be 2, got {self.levels}.")

    def local_hamiltonian(self, op, p):
        return -0.5 * p.freq * op.sigma_z

    def charge_coupling_operator(self):
        return LocalOps(label=self.label, space=self.local_space(), device=self).sigma_x

    def lowering_operator(self):
        return self.local_operator("sigma_minus")

    def raising_operator(self):
        return self.local_operator("sigma_plus")

    def number_operator(self):
        return self.raising_operator() @ self.lowering_operator()
```

`computational = True` marks the device as a qubit, and `basis` lets a chip
solve it in its native or energy basis.

Named ports and observables use the same vocabulary: an operator declared as
`op["dipole"]` is available through `device.local_operator("dipole")` and
`network.port(..., operator="dipole")`. The conventional names `"a"`, `"a_dag"`,
`"n"`, `"X"`, `"Y"`, `"Z"`, and `"I"` keep their device-hook meanings.

Default T1 and thermal channels use `lowering_operator()` and
`raising_operator()`. T2 uses `number_operator()`. Override those hooks to
select your model's transitions and dephasing operator, or declare the
complete channels in `dissipation()`. Matrix-element normalization sets how a
channel's rate relates to measured lifetimes. quchip does not infer selection
rules or dipole strengths from energy levels. The energy-level index used by
frames is separate from these physical operators.

For an intrinsically finite model, override `truncation_boundary()` to return
`None`. For a numerical cutoff, supply a `TruncationBoundary`. An undeclared
custom cutoff is reported unavailable.

Units remain GHz, ns and mK for every model: enter 1 MHz as `0.001` GHz and 1
microsecond as `1000` ns. Frames give one reference frequency per device.
Multitone terms can keep explicit time dependence. Local spaces are finite.

Subclass `ModelMapping` when a third-party object needs an explicit
conversion. Set `source` for import, `target` and `library` for export, and
implement only the directions the mapping supports.

## Control equipment

Subclass `SignalTransform` for a classical hardware effect that acts on complete
analytic signals. Implement `apply(signals)` and return a new signal map. Declare
numeric fields with `parameter()` and structural fields with `setting()`. Opt into
persistence with `serializable=True`, and import the extension before loading a
saved chip that uses it. Sweeps rebind parameters on independent transforms, whose
generated constructors use keyword arguments. Override `validate()` for
relationships between fields. Construction and binding call it after assigning all
candidate values, and numerical validation must keep tracing.

```python
from quchip.control import SignalTransform
from quchip.declarative import parameter, setting, qnp


class CableLoss(SignalTransform, serializable=True):
    linear_time_invariant = True

    line: str = setting()
    loss_db: float = parameter()

    def apply(self, signals):
        factor = qnp.power(10.0, -self.loss_db / 20.0)
        return {
            key: signal.scaled(factor) if key[0] == self.line else signal
            for key, signal in signals.items()
        }

    def referenced_lines(self):
        return (self.line,)
```

`referenced_lines()` lets wiring and partitioning validate the line label.
`linear_time_invariant = True` lets a fixed tone pass through the transform as a
phasor, as `chip.rotation_amplitude()` needs. Construct the example with
`CableLoss(line="drive", loss_db=3.0)`.

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

If the engine must branch on an extension's concrete class, the model
needs a public physical capability instead.
