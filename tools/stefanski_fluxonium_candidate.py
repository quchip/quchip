"""Compare quchip with the experimental fluxonium spectrum of Stefanski et al."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from urllib.request import urlopen

import matplotlib
import numpy as np

from quchip import Chip, CouplingModel, Fluxonium, Resonator, Scalar, parameter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIGURE = ROOT / "docs" / "images" / "stefanski_fluxonium_candidate.png"
DATA_URL = (
    "https://raw.githubusercontent.com/AndersenQubitLab/"
    "FPA-RO-experimental/main/processed_data_fx8.csv"
)
READOUT_DATA_URL = (
    "https://raw.githubusercontent.com/AndersenQubitLab/"
    "FPA-RO-experimental/main/res_fit_results.csv"
)


class FluxoniumReadoutCoupling(CouplingModel):
    """Exchange interaction used in the authors' released readout model."""

    g: Scalar = parameter(unit="GHz")
    oscillator_length: Scalar = parameter(positive=True)

    def interaction(self, q, r, p):
        scale = np.sqrt(2.0)
        lowering = q.phi * (1.0 / (p.oscillator_length * scale)) + (
            1j * p.oscillator_length / scale
        ) * q.charge
        raising = q.phi * (1.0 / (p.oscillator_length * scale)) - (
            1j * p.oscillator_length / scale
        ) * q.charge
        return p.g * (lowering * r.adag + raising * r.a)


def load_published_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return measured flux, measured f01, and the authors' five fit parameters."""
    with urlopen(DATA_URL) as response:  # noqa: S310 - fixed public research archive
        rows = list(csv.DictReader(io.StringIO(response.read().decode("utf-8"))))

    flux = np.asarray(
        [float(row["phi_ext_qubit"]) for row in rows if row["phi_ext_qubit"]]
    )
    measured = np.asarray(
        [float(row["qubit_freq"]) for row in rows if row["qubit_freq"]]
    )
    fit_parameters = np.asarray(
        [float(rows[index]["energy_params"]) for index in range(5)]
    )
    return flux, measured, fit_parameters


def load_readout_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return measured flux, ground-state resonator frequency, and chi."""
    with urlopen(READOUT_DATA_URL) as response:  # noqa: S310 - fixed public research archive
        rows = list(csv.DictReader(io.StringIO(response.read().decode("utf-8"))))
    rows = [row for row in rows if row["phi_ext_disshift"]]
    return (
        np.asarray([float(row["phi_ext_disshift"]) for row in rows]),
        np.asarray([float(row["fr_q0"]) for row in rows]),
        np.asarray([float(row["chi"]) for row in rows]),
    )


def build_readout_chip(
    *, resonator_freq: float, coupling: float, E_J: float, E_C: float, E_L: float
) -> Chip:
    """Build the fluxonium-readout model released with the paper."""
    q = Fluxonium(
        E_C=E_C,
        E_J=E_J,
        E_L=E_L,
        phi_ext=0.5,
        levels=10,
        num_basis=300,
        phi_max=5.0 * np.pi,
        basis="eigen",
        label="q",
    )
    r = Resonator(freq=resonator_freq, levels=3, label="readout")
    edge = FluxoniumReadoutCoupling(
        q,
        r,
        g=coupling,
        oscillator_length=(8.0 * E_C / E_L) ** 0.25,
        label="q-readout",
    )
    return Chip([q, r], [edge], basis="eigen", frame="lab")


def main() -> None:
    flux, measured, fit_parameters = load_published_data()
    resonator_freq, coupling, E_J, E_C, E_L = fit_parameters

    # Start with the isolated device. The readout resonator is the natural next
    # model extension, but is not needed to expose the measured flux curve.
    predicted = np.asarray(
        [
            float(
                Fluxonium(
                    E_C=E_C,
                    E_J=E_J,
                    E_L=E_L,
                    phi_ext=phi,
                    levels=4,
                    num_basis=400,
                    phi_max=5.0 * np.pi,
                    basis="eigen",
                    label="q",
                ).freq
            )
            for phi in flux
        ]
    )
    residual_mhz = 1.0e3 * (predicted - measured)
    absolute_residual_mhz = np.abs(residual_mhz)

    readout_flux, measured_resonator, measured_chi_mhz = load_readout_data()
    readout_indices = np.flatnonzero(readout_flux <= 0.85)
    readout_chip = build_readout_chip(
        resonator_freq=resonator_freq,
        coupling=coupling,
        E_J=E_J,
        E_C=E_C,
        E_L=E_L,
    )
    predicted_resonator = []
    predicted_chi_mhz = []
    for phi in readout_flux[readout_indices]:
        point = readout_chip.with_params({"q.phi_ext": phi})
        predicted_resonator.append(float(point.freq("readout")))
        predicted_chi_mhz.append(
            500.0 * float(point.dispersive_shift("q", "readout"))
        )
    predicted_resonator = np.asarray(predicted_resonator)
    predicted_chi_mhz = np.asarray(predicted_chi_mhz)

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(8.2, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0, 1.5]},
        layout="constrained",
    )
    spectrum_axis, residual_axis, readout_axis = axes
    spectrum_axis.scatter(
        flux,
        measured,
        s=15,
        color="#262626",
        alpha=0.72,
        label="experiment",
        zorder=2,
    )
    spectrum_axis.plot(
        flux,
        predicted,
        color="#C9343A",
        linewidth=2.2,
        label="quchip, published fit parameters",
        zorder=3,
    )
    spectrum_axis.set_ylabel(r"$f_{01}$ (GHz)")
    spectrum_axis.legend(frameon=False, loc="lower right")
    spectrum_axis.grid(color="0.88", linewidth=0.7)

    residual_axis.axhline(0.0, color="0.45", linewidth=1.0)
    residual_axis.scatter(flux, residual_mhz, s=14, color="#246FA8", alpha=0.78)
    residual_axis.set(
        xlabel=r"External flux $\Phi_{\mathrm{ext}}/\Phi_0$",
        ylabel="model - data\n(MHz)",
        xlim=(float(flux.min()), float(flux.max())),
    )
    residual_axis.grid(color="0.88", linewidth=0.7)

    readout_axis.scatter(
        readout_flux,
        measured_chi_mhz,
        s=12,
        color="#262626",
        alpha=0.58,
        label="experiment",
    )
    readout_axis.plot(
        readout_flux[readout_indices],
        predicted_chi_mhz,
        color="#246FA8",
        linewidth=2.0,
        label="quchip readout model",
    )
    readout_axis.set(
        xlabel=r"External flux $\Phi_{\mathrm{ext}}/\Phi_0$",
        ylabel=r"$\chi$ (MHz)",
    )
    readout_axis.legend(frameon=False)
    readout_axis.grid(color="0.88", linewidth=0.7)

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)

    receipt = {
        "figure": str(FIGURE.relative_to(ROOT)),
        "source_data": DATA_URL,
        "experimental_points": int(measured.size),
        "authors_fit_parameters_ghz": {
            "resonator_freq": float(resonator_freq),
            "coupling": float(coupling),
            "E_J": float(E_J),
            "E_C": float(E_C),
            "E_L": float(E_L),
        },
        "rmse_mhz": float(np.sqrt(np.mean(residual_mhz**2))),
        "median_absolute_error_mhz": float(np.median(absolute_residual_mhz)),
        "p95_absolute_error_mhz": float(np.quantile(absolute_residual_mhz, 0.95)),
        "maximum_absolute_error_mhz": float(absolute_residual_mhz.max()),
        "points_above_10_mhz": int(np.count_nonzero(absolute_residual_mhz > 10.0)),
        "readout_model_points": int(readout_indices.size),
        "readout_frequency_rmse_mhz": float(
            1.0e3
            * np.sqrt(
                np.mean(
                    (
                        predicted_resonator
                        - measured_resonator[readout_indices]
                    )
                    ** 2
                )
            )
        ),
        "chi_rmse_mhz": float(
            np.sqrt(
                np.mean(
                    (predicted_chi_mhz - measured_chi_mhz[readout_indices]) ** 2
                )
            )
        ),
        "chi_median_absolute_error_mhz": float(
            np.median(
                np.abs(predicted_chi_mhz - measured_chi_mhz[readout_indices])
            )
        ),
    }
    print("RESULT stefanski_fluxonium=" + json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
