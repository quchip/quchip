"""Reproduce the static IST scaling in Hassani et al., Fig. 3(b,c)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import scqubits as scq

from quchip import Fluxonium

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIGURE = ROOT / "docs" / "images" / "hassani_ist_candidate.png"
E_J = 35.0
E_C = 0.15
E_L_VALUES = np.asarray([2.0, 1.5, 1.0, 0.75, 0.5, 0.25])
FLUX_VALUES = np.linspace(0.0, 0.5, 41)
LEVELS = 16
PHASE_GRID_POINTS = 700
PHASE_EXTENT = 3.0 * np.pi


def quchip_plasmon_transition(E_L: float, flux: float) -> float:
    """Return the brightest charge-coupled transition out of the ground state."""
    device = Fluxonium(
        E_C=E_C,
        E_J=E_J,
        E_L=E_L,
        phi_ext=flux,
        levels=LEVELS,
        num_basis=PHASE_GRID_POINTS,
        phi_max=PHASE_EXTENT,
        basis="eigen",
        label="ist",
    )
    energies = np.asarray(device.eigenenergies())
    vectors = np.asarray(device.eigenvectors())
    charge = np.asarray(device.charge_coupling_operator())
    strengths = np.abs(vectors[:, 0].conj() @ charge @ vectors) ** 2
    bright_index = int(np.argmax(strengths[1:]) + 1)
    return float(energies[bright_index])


def scqubits_plasmon_transition(E_L: float, flux: float) -> float:
    """Evaluate the paper's scqubits reference with the same branch rule."""
    device = scq.Fluxonium(
        EJ=E_J,
        EC=E_C,
        EL=E_L,
        flux=flux,
        cutoff=160,
        truncated_dim=LEVELS,
    )
    energies, vectors = device.eigensys(evals_count=LEVELS)
    strengths = np.abs(vectors[:, 0].conj() @ device.n_operator() @ vectors) ** 2
    bright_index = int(np.argmax(strengths[1:]) + 1)
    return float(energies[bright_index] - energies[0])


def main() -> None:
    quchip_curves = np.asarray(
        [
            [quchip_plasmon_transition(E_L, flux) for flux in FLUX_VALUES]
            for E_L in E_L_VALUES
        ]
    )
    reference_curves = np.asarray(
        [
            [scqubits_plasmon_transition(E_L, flux) for flux in FLUX_VALUES]
            for E_L in E_L_VALUES
        ]
    )

    ratios = E_J / E_L_VALUES
    dispersions_mhz = 1.0e3 * np.abs(quchip_curves[:, 0] - quchip_curves[:, -1])
    reference_dispersions_mhz = 1.0e3 * np.abs(
        reference_curves[:, 0] - reference_curves[:, -1]
    )
    high_ratio = ratios >= 35.0
    slope = float(np.polyfit(np.log(ratios[high_ratio]), np.log(dispersions_mhz[high_ratio]), 1)[0])
    max_reference_error_mhz = float(
        1.0e3 * np.max(np.abs(quchip_curves - reference_curves))
    )
    quchip_shifts_mhz = 1.0e3 * (quchip_curves - quchip_curves[:, :1])
    reference_shifts_mhz = 1.0e3 * (
        reference_curves - reference_curves[:, :1]
    )
    max_shift_error_mhz = float(
        np.max(np.abs(quchip_shifts_mhz - reference_shifts_mhz))
    )

    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(E_L_VALUES)))
    figure, (spectrum_axis, scaling_axis) = plt.subplots(
        1,
        2,
        figsize=(10.2, 4.4),
        layout="constrained",
    )
    for index, (E_L, color) in enumerate(zip(E_L_VALUES, colors)):
        spectrum_axis.plot(
            FLUX_VALUES,
            quchip_shifts_mhz[index],
            color=color,
            linewidth=2.0,
            label=rf"$E_L/h={E_L:g}$ GHz",
        )
        spectrum_axis.plot(
            FLUX_VALUES[::5],
            reference_shifts_mhz[index, ::5],
            linestyle="none",
            marker="o",
            markerfacecolor="white",
            markeredgecolor=color,
            markersize=3.8,
        )

    spectrum_axis.set(
        xlabel=r"External flux $\Phi_{\mathrm{ext}}/\Phi_0$",
        ylabel="Flux-induced plasmon shift (MHz)",
        xlim=(0.0, 0.5),
    )
    spectrum_axis.legend(frameon=False, fontsize=8, ncols=2)
    spectrum_axis.grid(color="0.85", linewidth=0.7)

    scaling_axis.loglog(
        ratios,
        dispersions_mhz,
        "o-",
        color="#C9343A",
        linewidth=2.0,
        label="quchip",
    )
    scaling_axis.loglog(
        ratios,
        reference_dispersions_mhz,
        "s",
        markerfacecolor="white",
        markeredgecolor="#111827",
        label="paper's scqubits model",
    )
    quadratic = dispersions_mhz[-1] * (ratios / ratios[-1]) ** -2
    scaling_axis.loglog(
        ratios,
        quadratic,
        linestyle="--",
        color="0.45",
        label=r"$(E_J/E_L)^{-2}$",
    )
    scaling_axis.set(
        xlabel=r"$E_J/E_L$",
        ylabel="Full-flux dispersion (MHz)",
    )
    scaling_axis.grid(which="both", color="0.85", linewidth=0.7)
    scaling_axis.legend(frameon=False, fontsize=8)

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)

    receipt = {
        "figure": str(FIGURE.relative_to(ROOT)),
        "phase_grid_points": PHASE_GRID_POINTS,
        "phase_extent_radians": PHASE_EXTENT,
        "flux_points": len(FLUX_VALUES),
        "E_J_ghz": E_J,
        "E_C_ghz": E_C,
        "E_L_ghz": E_L_VALUES.tolist(),
        "E_J_over_E_L": ratios.tolist(),
        "quchip_dispersion_mhz": dispersions_mhz.tolist(),
        "reference_dispersion_mhz": reference_dispersions_mhz.tolist(),
        "high_ratio_log_log_slope": slope,
        "max_pointwise_reference_error_mhz": max_reference_error_mhz,
        "max_flux_shift_reference_error_mhz": max_shift_error_mhz,
    }
    print("RESULT hassani_ist=" + json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
