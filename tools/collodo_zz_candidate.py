"""Render a candidate reproduction of Collodo et al. Fig. 2(a).

The quchip curve uses only public APIs and the capacitance/Josephson parameters
reported in the paper. It is intentionally kept separate from the statics
guide until we decide whether it should replace the existing paper example.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.constants import Planck, elementary_charge
from scipy.optimize import brentq

from quchip import (
    Capacitive,
    ChargeBasisTransmon,
    Chip,
    CouplingModel,
    Exact,
    FockDevice,
    LocalOps,
    PhysicsExpr,
    Scalar,
    parameter,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "_static" / "data"
FIGURE = ROOT / "docs" / "images" / "collodo_tunable_zz_candidate.png"
LEVELS = 6
CHARGE_BASIS_SIZE = 31
PAPER_FOCK_STATES = 8
COUPLER_EJ_MAX_GHZ = 37.3
COUPLER_JUNCTION_RATIO = 1.0 / 1.71
Q1_CROSSTALK = 0.04
Q2_CROSSTALK = 0.005


class SixthOrderTransmon(FockDevice):
    """Paper-local transmon: cosine expanded through phi**6."""

    _default_levels = PAPER_FOCK_STATES

    freq: Scalar = parameter(positive=True, unit="GHz")
    E_C: Scalar = parameter(positive=True, unit="GHz")
    E_J: Scalar = parameter(positive=True, unit="GHz")

    def local_hamiltonian(self, op: LocalOps, p: object) -> PhysicsExpr:
        phi_zpf = (2.0 * p.E_C / p.E_J) ** 0.25
        charge_zpf = (p.E_J / (32.0 * p.E_C)) ** 0.25
        phase = phi_zpf * (op.a + op.adag)
        charge = 1j * charge_zpf * (op.adag - op.a)
        phase2 = phase @ phase
        phase4 = phase2 @ phase2
        phase6 = phase4 @ phase2
        return (
            4.0 * p.E_C * (charge @ charge)
            + 0.5 * p.E_J * phase2
            - (p.E_J / 24.0) * phase4
            + (p.E_J / 720.0) * phase6
        )


class ChargeCharge(CouplingModel):
    """Full charge-charge term in the local oscillator coordinates."""

    g: Scalar = parameter(unit="GHz")

    def interaction(self, a: LocalOps, b: LocalOps, p: object) -> PhysicsExpr:
        charge_a = 1j * (a.adag - a.a)
        charge_b = 1j * (b.adag - b.a)
        return p.g * charge_a * charge_b


def load_data() -> tuple[np.ndarray, np.ndarray]:
    measured = np.genfromtxt(
        DATA / "collodo_figure2a_measurements.csv",
        delimiter=",",
        names=True,
        comments="#",
    )
    simulated = np.genfromtxt(
        DATA / "collodo_figure2a_simulation.csv",
        delimiter=",",
        names=True,
        comments="#",
    )
    return measured, simulated


def charging_energy_matrix() -> np.ndarray:
    """Return e^2 C^-1 / 2h in GHz from the supplemental circuit table."""
    capacitance_ff = np.asarray(
        [
            [84.66, -0.46, -6.4],
            [-0.46, 84.66, -6.4],
            [-6.4, -6.4, 73.2],
        ]
    )
    inverse_capacitance = np.linalg.inv(capacitance_ff * 1.0e-15)
    return elementary_charge**2 * inverse_capacitance / (2.0 * Planck * 1.0e9)


def transmon(E_C: float, E_J: float, label: str) -> ChargeBasisTransmon:
    return ChargeBasisTransmon(
        E_C=E_C,
        E_J=E_J,
        levels=LEVELS,
        num_basis=CHARGE_BASIS_SIZE,
        basis="eigen",
        label=label,
    )


def coupler_ej_for_isolated_frequency(E_C: float, frequency: float) -> float:
    """Invert the isolated exact-cosine transition frequency."""
    return float(
        brentq(
            lambda E_J: float(transmon(E_C, E_J, "probe").freq) - frequency,
            1.0,
            50.0,
        )
    )


def make_chip_from_ej(coupler_ej: float) -> Chip:
    E_C = charging_energy_matrix()
    q1 = transmon(E_C[0, 0], 15.3, "q1")
    q2 = transmon(E_C[1, 1], 17.49, "q2")
    coupler = transmon(E_C[2, 2], coupler_ej, "c")
    return Chip(
        [q1, q2, coupler],
        [
            Capacitive(q1, q2, g=8.0 * E_C[0, 1], label="q1-q2"),
            Capacitive(q1, coupler, g=8.0 * E_C[0, 2], label="q1-c"),
            Capacitive(q2, coupler, g=8.0 * E_C[1, 2], label="q2-c"),
        ],
        frame="lab",
        approximation=Exact(),
        backend="dynamiqs",
    )


def dressed_coupler_frequency(chip: Chip) -> float:
    """Return the coupler-like dressed transition, including strong mixing."""
    dressed = chip.dress(overlap_threshold=0.0, force=True)
    ground = dressed.state_map[(0, 0, 0)]
    excited = dressed.state_map[(0, 0, 1)]
    return float(dressed.eigenvalues[excited] - dressed.eigenvalues[ground])


def make_chip(coupler_frequency: float) -> Chip:
    """Build a chip whose dressed coupler-like transition is ``frequency``."""
    E_C = charging_energy_matrix()
    isolated_ej = coupler_ej_for_isolated_frequency(E_C[2, 2], coupler_frequency)
    lower = max(1.0, 0.65 * isolated_ej)
    upper = 1.35 * isolated_ej
    coupler_ej = brentq(
        lambda E_J: dressed_coupler_frequency(make_chip_from_ej(E_J))
        - coupler_frequency,
        lower,
        upper,
    )
    return make_chip_from_ej(float(coupler_ej))


def evaluate_zz(frequencies: np.ndarray) -> np.ndarray:
    return np.asarray([1.0e6 * float(make_chip(frequency).static_zz("q1", "q2")) for frequency in frequencies])


def sixth_order_matrix(E_C: float, E_J: float) -> np.ndarray:
    """Return one local sixth-order Hamiltonian in the paper's Fock basis."""
    lowering = np.diag(np.sqrt(np.arange(1, PAPER_FOCK_STATES)), 1).astype(complex)
    raising = lowering.conj().T
    phi_zpf = (2.0 * E_C / E_J) ** 0.25
    charge_zpf = (E_J / (32.0 * E_C)) ** 0.25
    phase = phi_zpf * (lowering + raising)
    charge = 1j * charge_zpf * (raising - lowering)
    phase2 = phase @ phase
    return (
        4.0 * E_C * (charge @ charge)
        + 0.5 * E_J * phase2
        - (E_J / 24.0) * (phase2 @ phase2)
        + (E_J / 720.0) * (phase2 @ phase2 @ phase2)
    )


def sixth_order_transition(E_C: float, E_J: float) -> float:
    energies = np.linalg.eigvalsh(sixth_order_matrix(E_C, E_J))
    return float(energies[1] - energies[0])


def sixth_order_device(E_C: float, E_J: float, label: str) -> SixthOrderTransmon:
    return SixthOrderTransmon(
        freq=sixth_order_transition(E_C, E_J),
        E_C=E_C,
        E_J=E_J,
        levels=PAPER_FOCK_STATES,
        label=label,
    )


def sixth_order_coupler_ej(E_C: float, frequency: float) -> float:
    return float(
        brentq(
            lambda E_J: sixth_order_transition(E_C, E_J) - frequency,
            1.0,
            50.0,
        )
    )


def coupler_flux_for_ej(E_J: float) -> float:
    """Return the 0-to-0.5 Phi0 branch of the published SQUID relation."""
    ratio = COUPLER_JUNCTION_RATIO
    cosine = (
        (E_J * (1.0 + ratio) / COUPLER_EJ_MAX_GHZ) ** 2
        - 1.0
        - ratio**2
    ) / (2.0 * ratio)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)) / (2.0 * np.pi))


def crosstalk_shifted_ej(E_J_idle: float, idle_flux: float, delta_flux: float) -> float:
    """Symmetric-SQUID continuation from a specified idle operating point."""
    scale = np.cos(np.pi * (idle_flux + delta_flux)) / np.cos(np.pi * idle_flux)
    return float(E_J_idle * abs(scale))


def make_sixth_order_chip(
    coupler_frequency: float,
    *,
    q1_idle_flux: float | None = None,
    q2_idle_flux: float | None = None,
) -> Chip:
    """Build the sixth-order, eight-state local-mode model in the supplement."""
    E_C = charging_energy_matrix()
    coupler_ej = sixth_order_coupler_ej(E_C[2, 2], coupler_frequency)
    idle_coupler_ej = sixth_order_coupler_ej(E_C[2, 2], 7.612)
    coupler_flux_delta = coupler_flux_for_ej(coupler_ej) - coupler_flux_for_ej(
        idle_coupler_ej
    )
    q1_ej = (
        15.3
        if q1_idle_flux is None
        else crosstalk_shifted_ej(
            15.3,
            q1_idle_flux,
            Q1_CROSSTALK * coupler_flux_delta,
        )
    )
    q2_ej = (
        17.49
        if q2_idle_flux is None
        else crosstalk_shifted_ej(
            17.49,
            q2_idle_flux,
            Q2_CROSSTALK * coupler_flux_delta,
        )
    )
    E_J = np.asarray([q1_ej, q2_ej, coupler_ej])
    devices = [
        sixth_order_device(E_C[index, index], E_J[index], label)
        for index, label in enumerate(("q1", "q2", "c"))
    ]
    charge_zpf = (E_J / (32.0 * np.diag(E_C))) ** 0.25

    def coupling(left: int, right: int, label: str) -> ChargeCharge:
        strength = 8.0 * E_C[left, right] * charge_zpf[left] * charge_zpf[right]
        return ChargeCharge(devices[left], devices[right], g=strength, label=label)

    return Chip(
        devices,
        [
            coupling(0, 1, "q1-q2"),
            coupling(0, 2, "q1-c"),
            coupling(1, 2, "q2-c"),
        ],
        frame="lab",
        approximation=Exact(),
        basis="eigen",
        backend="dynamiqs",
    )


def evaluate_sixth_order_zz(
    frequencies: np.ndarray,
    *,
    q1_idle_flux: float | None = None,
    q2_idle_flux: float | None = None,
) -> np.ndarray:
    return np.asarray(
        [
            1.0e6
            * float(
                make_sixth_order_chip(
                    frequency,
                    q1_idle_flux=q1_idle_flux,
                    q2_idle_flux=q2_idle_flux,
                ).static_zz("q1", "q2")
            )
            for frequency in frequencies
        ]
    )


def main() -> None:
    measured, simulated = load_data()

    measured_frequencies = measured["coupler_frequency_ghz"]
    dense_frequencies = np.linspace(
        float(np.min(measured_frequencies)),
        float(np.max(simulated["coupler_frequency_ghz"])),
        41,
    )
    evaluation_frequencies = np.unique(np.concatenate((dense_frequencies, measured_frequencies)))
    quchip_zz = evaluate_zz(evaluation_frequencies)
    quchip_at_measurements = np.interp(
        measured_frequencies,
        evaluation_frequencies,
        quchip_zz,
    )

    turning_point = int(np.argmin(simulated["coupler_frequency_ghz"]))
    paper_frequencies = simulated["coupler_frequency_ghz"][: turning_point + 1][::-1]
    paper_zz = simulated["static_zz_khz"][: turning_point + 1][::-1]
    paper_at_measurements = np.interp(
        measured_frequencies,
        paper_frequencies,
        paper_zz,
    )
    paper_window = paper_frequencies >= float(np.min(measured_frequencies) - 0.05)

    measured_zz = measured["static_zz_khz"]
    absolute_ratio = np.abs(quchip_at_measurements / measured_zz)
    idle_chip = make_chip(7.612)
    receipt = {
        "figure": str(FIGURE.relative_to(ROOT)),
        "measurement_count": int(len(measured)),
        "solver_dimension": int(np.prod(idle_chip.dims)),
        "authored_dimension": int(np.prod(idle_chip.authored_dims)),
        "levels_per_mode": LEVELS,
        "charge_basis_size": CHARGE_BASIS_SIZE,
        "idle_dressed_frequencies_ghz": {label: float(idle_chip.freq(label)) for label in ("q1", "q2", "c")},
        "quchip_idle_zz_khz": float(quchip_at_measurements[0]),
        "measured_idle_zz_khz": float(measured_zz[0]),
        "paper_model_idle_zz_khz": float(paper_at_measurements[0]),
        "quchip_strongest_zz_mhz": float(quchip_at_measurements[-1] / 1.0e3),
        "measured_strongest_zz_mhz": float(measured_zz[-1] / 1.0e3),
        "paper_model_strongest_zz_mhz": float(paper_at_measurements[-1] / 1.0e3),
        "quchip_vs_measurement_rmse_mhz": float(np.sqrt(np.mean((quchip_at_measurements - measured_zz) ** 2)) / 1.0e3),
        "quchip_vs_measurement_median_abs_khz": float(np.median(np.abs(quchip_at_measurements - measured_zz))),
        "quchip_vs_measurement_median_multiplicative_error": float(np.exp(np.median(np.abs(np.log(absolute_ratio))))),
        "paper_vs_measurement_rmse_mhz": float(np.sqrt(np.mean((paper_at_measurements - measured_zz) ** 2)) / 1.0e3),
    }

    figure, axis = plt.subplots(figsize=(7.4, 4.8), layout="constrained")
    axis.plot(
        paper_frequencies[paper_window],
        np.abs(paper_zz[paper_window]),
        color="#6B7280",
        linewidth=2.0,
        label="paper circuit model",
    )
    axis.plot(
        evaluation_frequencies,
        np.abs(quchip_zz),
        color="#C9343A",
        linewidth=2.2,
        label="quchip charge-basis circuit",
    )
    axis.errorbar(
        measured_frequencies,
        np.abs(measured_zz),
        yerr=measured["error_khz"],
        fmt="o",
        color="#111827",
        markerfacecolor="white",
        markersize=5.5,
        capsize=2.5,
        linewidth=1.0,
        label="measurement",
    )
    axis.axvline(7.612, color="#9CA3AF", linestyle="--", linewidth=1.0)
    axis.text(
        7.612,
        1.25,
        "idle",
        color="#6B7280",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    axis.set(
        xlabel="Coupler frequency (GHz)",
        ylabel=r"Static $|ZZ|$ (kHz)",
        yscale="log",
        ylim=(1.0, 1.0e5),
    )
    axis.grid(which="both", color="#D1D5DB", alpha=0.45, linewidth=0.7)
    axis.legend(frameon=False)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)

    print("RESULT collodo_candidate=" + json.dumps(receipt, sort_keys=True))
    for frequency, observed, paper_value, quchip_value in zip(
        measured_frequencies,
        measured_zz,
        paper_at_measurements,
        quchip_at_measurements,
    ):
        print(
            f"{frequency:5.3f} GHz  measurement={observed / 1e3:8.4f} MHz  "
            f"paper={paper_value / 1e3:8.4f} MHz  "
            f"quchip={quchip_value / 1e3:8.4f} MHz"
        )


if __name__ == "__main__":
    main()
