# Resonance fluorescence and photon correlations

Drive a two-level emitter through a weak port and read the field that it
emits into a strong port. The output spectrum shows the Mollow triplet, and
the photon correlations show antibunching. Frequencies are in GHz, times in
ns, and decay rates in `1/ns`. Run the cells in order.

## Drive the emitter

A two-level transmon couples to two ports. The drive port is weak, so the
output port collects 95% of the emission. A coherent input of amplitude `β`,
in `sqrt(photons/ns)`, through a port with decay rate `γ` drives the emitter
at the Rabi frequency `Ω = 2 sqrt(γ) |β|` in rad/ns. Set `Ω` to four times
the total decay rate `Γ`, on resonance.

```python
import numpy as np
from quchip import Chip, DuffingTransmon, PortNetwork, VNA

drive_rate, output_rate = 0.0005, 0.01
decay = drive_rate + output_rate
rabi = 4 * decay
amplitude = rabi / (2 * np.sqrt(drive_rate))

q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q")
waveguide = PortNetwork(label="waveguide")
drive = waveguide.port("drive", target=q, rate=drive_rate)
output = waveguide.port("output", target=q, rate=output_rate)
emitter = Chip([q], port_network=waveguide, frame="rotating")

vna = VNA(emitter)
vna.pump(drive, freq=5.0, amplitude=amplitude)
```

`vna.pump()` sets a stationary tone. `output_spectrum()`, `g1()` and `g2()`
then solve the driven steady state and the field correlations at the chosen
reference plane.

## Read the output spectrum

The steady state of a resonantly driven two-level emitter has the excited
population `s/(2(1+s))`, with `s = 2Ω²/Γ²`. The emitted field has a coherent
part, a delta peak at the drive frequency, which holds the fraction
`1/(1+s)` of the photon flux. The rest is incoherent, and its spectrum is the
Mollow triplet: a central peak and two sidebands near `±Ω/2π`.

```python
offsets = np.linspace(-0.03, 0.03, 1201)
spectrum = vna.output_spectrum(output, frequencies=offsets)

saturation = 2 * rabi**2 / decay**2
excited = saturation / (2 * (1 + saturation))
np.testing.assert_allclose(spectrum.signal_photon_flux, output_rate * excited, rtol=1e-10)
np.testing.assert_allclose(spectrum.signal_coherent_flux / spectrum.signal_photon_flux,
                           1 / (1 + saturation), rtol=1e-10)

incoherent = np.trapezoid(spectrum.total_fluctuation_spectrum, offsets)
print(f"Photon flux: {float(spectrum.signal_photon_flux):.6f} per ns")
print(f"Coherent fraction: {float(spectrum.signal_coherent_flux / spectrum.signal_photon_flux):.6f}")
print(f"Spectrum area / incoherent flux: {float(incoherent / spectrum.signal_incoherent_flux):.4f}")
```

Output:

```text
Photon flux: 0.004848 per ns
Coherent fraction: 0.030303
Spectrum area / incoherent flux: 0.9997
```

`total_fluctuation_spectrum` is the normally ordered spectrum in quanta, so
its area is the incoherent photon flux. The ±30 MHz grid misses only the far
tails of the triplet. The coherent part has no width, so the result reports
it as `signal_coherent_flux` instead.

## Measure photon antibunching

The emitter holds at most one excitation, so it cannot emit two photons at
the same time, and `g2(0) = 0`. An emission leaves it in its ground state,
and `g2` then recovers with damped Rabi oscillations. For a resonant drive
the textbook result is

```{math}
g^{(2)}(\tau) = 1 - e^{-3\Gamma\tau/4}\left[\cos\mu\tau + \frac{3\Gamma}{4\mu}\sin\mu\tau\right],
\qquad \mu = \sqrt{\Omega^2 - \Gamma^2/16}.
```

```python
delays = np.linspace(0.0, 600.0, 241)
g2 = vna.g2(output, delays)
mu = np.sqrt(rabi**2 - decay**2 / 16)
textbook = 1 - np.exp(-3 * decay * delays / 4) * (np.cos(mu * delays)
                                                  + 3 * decay / (4 * mu) * np.sin(mu * delays))
np.testing.assert_allclose(g2.values, textbook, atol=1e-12)

g1 = vna.g1(output, [0.0, 2000.0])
print(f"g2(0): {abs(g2.values[0]):.6f}")
print(f"g1 after 2 us: {abs(g1.values[-1]):.6f}")
```

Output:

```text
g2(0): 0.000000
g1 after 2 us: 0.030317
```

`g1` measures the first-order coherence of the field. The incoherent part
loses its phase within a few `1/Γ`, so `g1` decays to the coherent fraction
of the flux, `1/(1+s)`.

<details>
<summary>Plot the spectrum and g2</summary>

```python
import shutil
import matplotlib.pyplot as plt

plt.style.use("../_static/quchip.mplstyle")
plt.rcParams["text.usetex"] = bool(shutil.which("latex"))
ink, red, blue = "#16181C", "#C92F33", "#246FA8"

fig, (spectrum_axis, g2_axis) = plt.subplots(1, 2, figsize=(7.6, 3.1), layout="constrained")
spectrum_axis.plot(1e3 * offsets, spectrum.total_fluctuation_spectrum, color=red)
spectrum_axis.set(xlabel="Offset from the drive (MHz)", ylabel="Spectrum (quanta)",
                  xlim=(-20, 20), ylim=(0, None))

g2_axis.plot(delays, textbook, color=ink, label="Textbook")
g2_axis.plot(delays[::6], g2.values[::6].real, "o", color=blue, ms=3.5, label="quchip")
g2_axis.axhline(1.0, color="#9AA0A8", lw=0.8, ls="--")
g2_axis.set(xlabel=r"Delay $\tau$ (ns)", ylabel=r"$g^{(2)}(\tau)$", xlim=(0, 600), ylim=(0, None))
g2_axis.legend(loc="lower right")

fig.savefig("resonance_fluorescence.svg")
plt.close(fig)
```

</details>

```{figure} ../images/resonance_fluorescence.svg
:alt: The Mollow triplet of the incoherent output spectrum beside the antibunched photon correlation g2, which starts at zero and recovers with damped Rabi oscillations.

Left: the incoherent output spectrum, with the central peak and the two Rabi
sidebands. Right: `g2` from quchip and from the textbook formula.
[PDF](../images/resonance_fluorescence.pdf)
```

## Split the field with a 90° hybrid

Many microwave experiments measure photon correlations with linear
amplifiers. A 90° hybrid splits the field into two lines, each with its own
amplifier. The two amplifiers add independent noise. This noise adds a
background to the cross-correlation of the two lines, but it does not
correlate them, so noise moments measured with the emitter off remove the
background. The hybrid sends half of the power
into each line, and vacuum enters its second input, so the cross-correlation
is still the `g2` of the emitter.

```python
setup = PortNetwork(label="hbt")
drive = setup.port("drive", target=q, rate=drive_rate)
output = setup.port("output", target=q, rate=output_rate)
hybrid = setup.hybrid90("hybrid")
setup.cascade(output, hybrid.input_terminal("left"))
line_a = setup.expose("a", input=output.input, output=hybrid.output_terminal("left"))
line_b = setup.expose("b", input=hybrid.input_terminal("right"), output=hybrid.output_terminal("right"))

split = VNA(Chip([q], port_network=setup, frame="rotating"))
split.pump(drive, freq=5.0, amplitude=amplitude)
cross = split.g2(line_a, delays, input=line_b)
np.testing.assert_allclose(cross.values, g2.values, atol=1e-12)
print(f"Flux per line / emitted flux: {float(cross.output_intensity / spectrum.signal_photon_flux):.6f}")
```

Output:

```text
Flux per line / emitted flux: 0.500000
```

`hybrid90()` is directional: it maps the two fields that enter it to the two
fields that leave it, with the scattering matrix `[[1, i], [i, 1]]/sqrt(2)`.
Wire its named terminals, `left` and `right`, with `cascade()` or
`connect()`, and expose each plane with `input=` and `output=`.
`beam_splitter(eta=...)` gives a directional splitter with any power split.

quchip evaluates `g1` and `g2` at reference planes before the amplifiers.
Through an amplifier, the normalized correlations need a detection bandwidth
for its broadband added noise, so the call raises. For the amplified lines,
`VNA.measure()` gives the IQ covariances and each noise source's
contribution, as in the [readout guide](steady-state-and-vna.md).
