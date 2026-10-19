"""Draw the overview figure of the README and the documentation front page.

Each panel repeats the model of one guide: the avoided crossing of the
statics guide, the short Gaussian pulse of the dynamics guide, the sampled
VNA trace of the fridge guide, and the fluxonium fit of the gradients guide.
The fit downloads the published measurements.

    python3 tools/overview_figure.py
    python3 tools/theme_figures.py guide_overview

The first command writes ``docs/images/guide_overview.svg`` and ``.png``. The
second derives the dark variants and copies the README pair to
``docs/_static/readme/``.
"""

from __future__ import annotations

import csv
import io
import shutil
from pathlib import Path
from urllib.request import urlopen

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.signal import butter

from quchip import (
    RWA,
    VNA,
    Capacitive,
    ChargeDrive,
    Chip,
    DuffingTransmon,
    Fluxonium,
    Gaussian,
    IQReceiver,
    PortNetwork,
    QuantumSequence,
    Resonator,
    SpectrumSweep,
    Sweep,
    qnp,
)

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "docs" / "images"
INK, ACCENT, BLUE, MUTED, FADED = "#16181C", "#C92F33", "#246FA8", "#50565A", "#9AA0A8"
PAPER_DATA = (
    "https://raw.githubusercontent.com/AndersenQubitLab/"
    "FPA-RO-experimental/57dc268dd048d1372db082c3ddd97a04871580bf/processed_data_fx8.csv"
)


def avoided_crossing():
    """Two transmons on a bus, with q2 swept through q1 (statics guide)."""
    q1 = DuffingTransmon(freq=5.30, anharmonicity=-0.65, levels=4, label="q1")
    q2 = DuffingTransmon(freq=5.58, anharmonicity=-0.65, levels=4, label="q2")
    bus = Resonator(freq=6.55, levels=4, label="bus")
    chip = Chip(
        [q1, q2, bus],
        couplings=[Capacitive(q1, bus, g=0.050, label="q1-bus"), Capacitive(q2, bus, g=0.050, label="q2-bus")],
        frame="rotating",
        approximation=RWA(),
    )
    q2_frequencies = np.linspace(5.285, 5.315, 241)
    sweep = SpectrumSweep(chip, [Sweep(q2_frequencies, name="q2.freq")], evals_count=4, overlap_threshold=0.0)
    result = sweep.run(progress=False)
    branches = result.eigenvalues[:, 1:3] - result.eigenvalues[:, :1]
    q1_branch = result.dressed_index(q1=1, q2=0, bus=0).astype(int) - 1
    return q2_frequencies, np.asarray(branches), np.asarray(q1_branch)


def short_pulse():
    """Populations during the short Gaussian pulse of the dynamics guide."""
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.30, levels=6, label="q")
    readout = Resonator(freq=6.8, levels=10, internal_quality_factor=6800, label="r")
    chip = Chip([qubit, readout], [Capacitive(qubit, readout, g=0.060, label="qr")],
                frame="rotating", approximation=RWA())
    line = ChargeDrive(qubit, label="qubit-charge")
    _ = chip.wire(line)
    f01 = float(chip.freq(qubit))
    duration = 3 / (np.pi * abs(float(chip.transition_frequency(qubit, 1, 2)) - f01))
    amplitude = chip.rotation_amplitude(qubit, line, Gaussian(duration=duration, sigmas=3.0))
    sequence = QuantumSequence(chip)
    pulse = Gaussian(duration=duration, sigmas=3.0, amplitude=amplitude)
    _ = sequence.schedule(line, envelope=pulse, freq=f01)
    times = np.linspace(0.0, 2.0 * duration, 301)
    result = sequence.simulate(tlist=times, initial_state=chip.state({qubit: 0, readout: 0}))
    populations = np.asarray([result.population(qubit, level) for level in range(3)]).real
    envelope = np.where(times <= duration, np.asarray(pulse.sample(times, real=True)), 0.0)
    return times, populations, envelope / np.max(envelope), duration


def vna_trace():
    """Three resonators on a bus behind the fridge wiring of the fridge guide."""
    mode_frequencies = np.array([6.4, 6.5, 6.6])
    internal_q = np.array([1800, 3200, 2400])
    external_q = np.array([450, 750, 550])
    bus = Resonator(freq=6.5, levels=3, internal_quality_factor=100_000, label="bus")
    resonators = [Resonator(freq=f, levels=3, internal_quality_factor=qi, label=f"r{i+1}")
                  for i, (f, qi) in enumerate(zip(mode_frequencies, internal_q))]
    bus_qe = 6.5
    bus_external_rate = 2 * np.pi * bus.freq / bus_qe
    bus_total_rate = bus_external_rate + 2 * np.pi * bus.freq / bus.internal_quality_factor
    external_rates = 2 * np.pi * mode_frequencies / external_q
    coupling_strengths = np.sqrt(external_rates * (
        (bus_total_rate / 2)**2 + (2 * np.pi * (mode_frequencies - bus.freq))**2
    ) / bus_external_rate) / (2 * np.pi)
    couplings = [Capacitive(bus, r, g=g, label=f"bus_r{i+1}")
                 for i, (r, g) in enumerate(zip(resonators, coupling_strengths))]

    filter_b, filter_a = butter(4, [4.0, 8.0], btype="bandpass", analog=True)

    def passband(frequency):
        return qnp.polyval(filter_b, 1j * frequency) / qnp.polyval(filter_a, 1j * frequency)

    fridge = PortNetwork(label="thermal_fridge")
    port = fridge.port("bus_coupler", target=bus, external_quality_factor=bus_qe)
    input_filter = fridge.filter("input_4_8GHz", transfer=passband)
    n_4k, n_cp, n_mxc = 12.329033161427, 0.046220887106121, 1.6829628323748e-07
    att_4k = fridge.attenuator("att_4K", loss_db=20, thermal_occupation=n_4k)
    att_cp = fridge.attenuator("att_CP", loss_db=20, thermal_occupation=n_cp)
    att_mxc = fridge.attenuator("att_MXC", loss_db=20, thermal_occupation=n_mxc)
    circ = fridge.circulator("circ")
    iso_1 = fridge.isolator("iso_1", thermal_occupation=n_mxc)
    iso_2 = fridge.isolator("iso_2", thermal_occupation=n_mxc)
    iso_loss = fridge.attenuator("iso_insertion", loss_db=1, thermal_occupation=n_mxc)
    output_filter = fridge.filter("output_4_8GHz", transfer=passband, thermal_occupation=n_mxc)
    coax = fridge.attenuator("output_coax", loss_db=2, thermal_occupation=n_4k)
    hemt = fridge.amplifier("HEMT_4K", gain_db=40, added_noise=8.0140842782029)
    room_amp = fridge.amplifier("amp_RT", gain_db=20, added_noise=925.22946424527)
    fridge.link(input_filter, att_4k, att_cp, att_mxc, circ.port(1))
    fridge.link(circ.port(2), port)
    fridge.link(circ.port(3), iso_1, iso_loss, iso_2, output_filter, coax, hemt, room_amp)
    drive = fridge.expose("drive", at=input_filter.port(1))
    readout = fridge.expose("readout", at=room_amp.port(2))
    chip = Chip([bus, *resonators], couplings, port_network=fridge, approximation=RWA(), frame="rotating")

    frequencies = np.linspace(6.34, 6.66, 401)
    vna = VNA(chip, ports=[drive, readout])
    steady_state = np.asarray(vna.sweep(frequencies).s(readout, drive))
    measurement = vna.measure(frequencies, amplitudes=20, input=drive, outputs=[readout])
    sampled = measurement.sample(1, receiver=IQReceiver(integration_time=10_000_000), seed=19)
    return frequencies, steady_state, np.asarray(sampled.ratio(readout)[0])


def fluxonium_fit():
    """Fit E_C, E_J and E_L to the measured fluxonium spectrum (gradients guide)."""
    with urlopen(PAPER_DATA) as response:  # noqa: S310 - fixed public research archive
        rows = list(csv.DictReader(io.StringIO(response.read().decode("utf-8"))))
    flux_all = np.asarray([float(row["phi_ext_qubit"]) for row in rows if row["phi_ext_qubit"]])
    f01_all = np.asarray([float(row["qubit_freq"]) for row in rows if row["qubit_freq"]])
    window = flux_all <= 0.85
    flux, f01 = flux_all[window], f01_all[window]
    training = np.arange(0, flux.size, 8)

    q = Fluxonium(E_C=0.72, E_J=4.4, E_L=0.68, phi_ext=0.5, levels=4, num_basis=160,
                  phi_max=5.0 * np.pi, basis="eigen", label="q")
    chip = Chip([q], [], basis="eigen", frame="lab", backend="dynamiqs")
    scale = jnp.asarray([1.0, 4.0, 1.0])

    def prediction(coordinates, points):
        energies = scale * jnp.exp(coordinates)
        return jax.vmap(lambda phi: chip.with_params(
            {"q.E_C": energies[0], "q.E_J": energies[1], "q.E_L": energies[2], "q.phi_ext": phi}
        ).freq("q"))(points)

    def loss(coordinates):
        residual = 1.0e3 * (prediction(coordinates, jnp.asarray(flux[training])) - jnp.asarray(f01[training]))
        return jnp.mean(2.0 * (jnp.sqrt(1.0 + (residual / 3.0)**2) - 1.0))

    loss_and_gradient = jax.jit(jax.value_and_grad(loss))

    def objective(coordinates):
        value, gradient = loss_and_gradient(jnp.asarray(coordinates))
        return float(value), np.asarray(gradient, dtype=float)

    fit = minimize(
        objective,
        np.log(np.asarray([0.72, 4.4, 0.68]) / np.asarray(scale)),
        method="L-BFGS-B",
        jac=True,
        bounds=[(np.log(0.3), np.log(1.5)), (np.log(2.0 / 4.0), np.log(6.0 / 4.0)), (np.log(0.3), np.log(1.5))],
        options={"maxiter": 160, "ftol": 1.0e-12, "gtol": 1.0e-9},
    )
    model_flux = np.linspace(0.5, 0.85, 351)
    model_f01 = np.asarray(prediction(jnp.asarray(fit.x), jnp.asarray(model_flux)))
    return flux, f01, training, model_flux, model_f01


def heading(axis, title, subtitle):
    axis.set_title(title, loc="left", fontsize=17, pad=30)
    axis.annotate(subtitle, (0, 1), xytext=(0, 9), xycoords="axes fraction", textcoords="offset points",
                  fontsize=11.5, color=MUTED, va="bottom")


def main() -> None:
    q2_frequencies, branches, q1_branch = avoided_crossing()
    pulse_times, populations, envelope, duration = short_pulse()
    frequencies, steady_state, sampled = vna_trace()
    flux, f01, training, model_flux, model_f01 = fluxonium_fit()

    plt.style.use(ROOT / "docs" / "_static" / "quchip.mplstyle")
    plt.rcParams["text.usetex"] = bool(shutil.which("latex"))
    plt.rcParams.update({"font.size": 12.5, "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
                         "legend.fontsize": 11.5})
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 7.6), layout="constrained")
    figure.get_layout_engine().set(w_pad=0.12, h_pad=0.12, wspace=0.06, hspace=0.08)
    spectra, pulses, chain, fit = axes.ravel()

    splitting = np.min(branches[:, 1] - branches[:, 0])
    spectra.plot(q2_frequencies, q2_frequencies, color=FADED, ls="--", lw=1.1)
    spectra.axhline(5.30, color=FADED, ls="--", lw=1.1)
    for index in (0, 1):
        q1_like = q1_branch == index
        spectra.plot(q2_frequencies, np.where(q1_like, branches[:, index], np.nan), color=ACCENT, lw=2.4)
        spectra.plot(q2_frequencies, np.where(q1_like, np.nan, branches[:, index]), color=INK, lw=2.4)
    spectra.set(xlabel=r"Bare $q_2$ frequency (GHz)", ylabel="Transition (GHz)",
                xlim=(q2_frequencies[0], q2_frequencies[-1]), xticks=[5.29, 5.30, 5.31], yticks=[5.29, 5.30, 5.31])
    heading(spectra, "Spectra", fr"Two transmons through an avoided crossing $\cdot$ $2J = {1e3 * splitting:.1f}$ MHz")

    for level, color in enumerate((INK, BLUE, ACCENT)):
        pulses.plot(pulse_times, populations[level], color=color, lw=2.4, label=fr"$|{level}\rangle$")
    pulses.plot(pulse_times, envelope, color=FADED, ls="--", lw=1.4, label="pulse")
    pulses.set(xlabel="Time (ns)", ylabel="Population", xlim=(0.0, pulse_times[-1]), ylim=(-0.02, 1.02),
               yticks=[0.0, 0.5, 1.0])
    pulses.legend(ncol=2, loc="upper right", handlelength=1.6, columnspacing=1.2)
    heading(pulses, "Pulses", fr"A {duration:.1f} ns Gaussian $\pi$ pulse leaks into $|2\rangle$")

    chain.plot(frequencies, 20 * np.log10(np.abs(sampled)), ".", color=ACCENT, ms=3.4, alpha=0.75)
    chain.plot(frequencies, 20 * np.log10(np.abs(steady_state)), color=INK, lw=1.6)
    chain.set(xlabel="Probe frequency (GHz)", ylabel=r"$|S_{21}|$ (dB)", xlim=(frequencies[0], frequencies[-1]),
              xticks=[6.4, 6.5, 6.6], yticks=[-10, -5, 0])
    heading(chain, "Readout chain", r"Sampled VNA trace through the fridge wiring $\cdot$ 10 ms")

    holdout = np.ones(flux.size, dtype=bool)
    holdout[training] = False
    fit.plot(flux[holdout], f01[holdout], "o", color=FADED, ms=4.2, alpha=0.8, label="measured")
    fit.plot(flux[training], f01[training], "x", color=INK, ms=6.5, mew=1.6, label="fit points")
    fit.plot(model_flux, model_f01, color=ACCENT, lw=2.4, label="quchip")
    fit.set(xlabel=r"External flux $\Phi_{\rm ext}/\Phi_0$", ylabel=r"$f_{01}$ (GHz)", xlim=(0.49, 0.86),
            xticks=[0.5, 0.6, 0.7, 0.8], yticks=[1, 2, 3, 4])
    fit.legend(loc="lower right")
    heading(fit, "Gradients", r"Fluxonium energies fitted to measured data with JAX gradients")

    figure.savefig(IMAGES / "guide_overview.svg")
    figure.savefig(IMAGES / "guide_overview.png")
    plt.close(figure)


if __name__ == "__main__":
    main()
