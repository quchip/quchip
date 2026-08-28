"""Prototype a differentiable fit to the Stefanski fluxonium data."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from urllib.request import urlopen

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from scipy.optimize import minimize

from quchip import Chip, CouplingModel, Fluxonium, Resonator, Scalar, parameter
from quchip.backend.dynamiqs import DynamiqsBackend

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIGURE = ROOT / "docs" / "images" / "stefanski_fluxonium_gradient_fit.png"
DATA_ROOT = (
    "https://raw.githubusercontent.com/AndersenQubitLab/"
    "FPA-RO-experimental/57dc268dd048d1372db082c3ddd97a04871580bf"
)


class FluxoniumReadoutCoupling(CouplingModel):
    """Exchange interaction used in the authors' released readout model."""

    g: Scalar = parameter(unit="GHz")
    oscillator_length: Scalar = parameter(positive=True)

    def interaction(self, q, r, p):
        scale = np.sqrt(2.0)
        lowering = q.phi / (p.oscillator_length * scale) + (
            1j * p.oscillator_length / scale
        ) * q.charge
        raising = q.phi / (p.oscillator_length * scale) - (
            1j * p.oscillator_length / scale
        ) * q.charge
        return p.g * (lowering * r.adag + raising * r.a)


def published_rows(filename: str) -> list[dict[str, str]]:
    """Read one CSV from the paper's pinned public repository."""
    with urlopen(f"{DATA_ROOT}/{filename}") as response:  # noqa: S310
        return list(csv.DictReader(io.StringIO(response.read().decode("utf-8"))))


def pseudo_huber(residual: jax.Array) -> jax.Array:
    """Smoothly limit the influence of the few large spectroscopy residuals."""
    return 2.0 * (jnp.sqrt(1.0 + residual**2) - 1.0)


def scipy_value_and_grad(compiled):
    """Adapt a compiled JAX value-and-gradient function for SciPy."""

    def evaluate(x):
        value, gradient = compiled(jnp.asarray(x))
        return float(value), np.asarray(gradient, dtype=float)

    return evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", type=int, default=0, choices=range(8))
    parser.add_argument("--stride", type=int, default=8, choices=(2, 4, 8))
    args = parser.parse_args()

    spectrum_rows = published_rows("processed_data_fx8.csv")
    readout_rows = [
        row
        for row in published_rows("res_fit_results.csv")
        if row["phi_ext_disshift"]
    ]

    fitted_values = np.asarray(
        [float(spectrum_rows[index]["energy_params"]) for index in range(5)]
    )
    published_fr, published_g, published_E_J, published_E_C, published_E_L = (
        fitted_values
    )
    published_energies = np.asarray([published_E_C, published_E_J, published_E_L])

    spectrum_flux_all = np.asarray(
        [float(row["phi_ext_qubit"]) for row in spectrum_rows if row["phi_ext_qubit"]]
    )
    spectrum_data_all = np.asarray(
        [float(row["qubit_freq"]) for row in spectrum_rows if row["qubit_freq"]]
    )
    spectrum_window = spectrum_flux_all <= 0.85
    spectrum_flux = spectrum_flux_all[spectrum_window]
    spectrum_data = spectrum_data_all[spectrum_window]

    # Fit on a sparse, deterministic subset and reserve all intervening points.
    if args.offset >= args.stride:
        parser.error("--offset must be smaller than --stride")
    spectrum_train_indices = np.arange(args.offset, spectrum_flux.size, args.stride)
    spectrum_holdout_mask = np.ones(spectrum_flux.size, dtype=bool)
    spectrum_holdout_mask[spectrum_train_indices] = False
    spectrum_train_flux = jnp.asarray(spectrum_flux[spectrum_train_indices])
    spectrum_train_data = jnp.asarray(spectrum_data[spectrum_train_indices])

    q = Fluxonium(
        E_C=0.72,
        E_J=4.4,
        E_L=0.68,
        phi_ext=0.5,
        levels=4,
        num_basis=160,
        phi_max=5.0 * np.pi,
        basis="eigen",
        label="q",
    )
    isolated_chip = Chip(
        [q],
        [],
        basis="eigen",
        frame="lab",
        backend=DynamiqsBackend(),
    )

    energy_scale = jnp.asarray([1.0, 4.0, 1.0])

    def energies_from_coordinates(coordinates):
        return energy_scale * jnp.exp(coordinates)

    def isolated_f01(energies, phi_ext):
        point = isolated_chip.with_params(
            {
                "q.E_C": energies[0],
                "q.E_J": energies[1],
                "q.E_L": energies[2],
                "q.phi_ext": phi_ext,
            }
        )
        return point.freq("q")

    def spectrum_prediction(coordinates, flux):
        energies = energies_from_coordinates(coordinates)
        return jax.vmap(lambda phi: isolated_f01(energies, phi))(flux)

    def spectrum_loss(coordinates):
        residual_mhz = 1.0e3 * (
            spectrum_prediction(coordinates, spectrum_train_flux)
            - spectrum_train_data
        )
        return jnp.mean(pseudo_huber(residual_mhz / 3.0))

    initial_energies = np.asarray([0.72, 4.4, 0.68])
    initial_coordinates = np.log(initial_energies / np.asarray(energy_scale))
    spectrum_history: list[float] = []
    compiled_spectrum_loss = jax.jit(jax.value_and_grad(spectrum_loss))
    spectrum_objective = scipy_value_and_grad(compiled_spectrum_loss)

    spectrum_fit = minimize(
        spectrum_objective,
        initial_coordinates,
        method="L-BFGS-B",
        jac=True,
        bounds=[
            (np.log(0.3), np.log(1.5)),
            (np.log(2.0 / 4.0), np.log(6.0 / 4.0)),
            (np.log(0.3), np.log(1.5)),
        ],
        callback=lambda x: spectrum_history.append(spectrum_objective(x)[0]),
        options={"maxiter": 160, "ftol": 1.0e-12, "gtol": 1.0e-9},
    )
    recovered_energies = np.asarray(
        energies_from_coordinates(jnp.asarray(spectrum_fit.x)), dtype=float
    )

    spectrum_model_flux = jnp.linspace(0.5, 0.85, 351)
    spectrum_model = np.asarray(
        spectrum_prediction(jnp.asarray(spectrum_fit.x), spectrum_model_flux)
    )
    spectrum_at_measurements = np.interp(
        spectrum_flux,
        np.asarray(spectrum_model_flux),
        spectrum_model,
    )
    spectrum_residual_mhz = 1.0e3 * (spectrum_at_measurements - spectrum_data)
    spectrum_holdout_residual_mhz = spectrum_residual_mhz[spectrum_holdout_mask]

    readout_flux_all = np.asarray(
        [float(row["phi_ext_disshift"]) for row in readout_rows]
    )
    readout_fr0_all = np.asarray([float(row["fr_q0"]) for row in readout_rows])
    readout_fr1_all = np.asarray([float(row["fr_q1"]) for row in readout_rows])
    readout_chi_all = np.asarray([float(row["chi"]) for row in readout_rows])
    readout_window = readout_flux_all <= 0.85
    readout_flux = readout_flux_all[readout_window]
    readout_fr0 = readout_fr0_all[readout_window]
    readout_fr1 = readout_fr1_all[readout_window]
    readout_chi = readout_chi_all[readout_window]

    readout_train_indices = np.arange(args.offset, readout_flux.size, args.stride)
    readout_holdout_mask = np.ones(readout_flux.size, dtype=bool)
    readout_holdout_mask[readout_train_indices] = False
    readout_train_flux = jnp.asarray(readout_flux[readout_train_indices])
    readout_train_data = jnp.asarray(
        np.column_stack(
            (
                readout_fr0[readout_train_indices],
                readout_fr1[readout_train_indices],
                readout_chi[readout_train_indices],
            )
        )
    )

    fitted_E_C, fitted_E_J, fitted_E_L = recovered_energies
    readout_q = Fluxonium(
        E_C=fitted_E_C,
        E_J=fitted_E_J,
        E_L=fitted_E_L,
        phi_ext=0.5,
        levels=10,
        num_basis=160,
        phi_max=5.0 * np.pi,
        basis="eigen",
        label="q",
    )
    readout = Resonator(freq=5.165, levels=3, label="readout")
    edge = FluxoniumReadoutCoupling(
        readout_q,
        readout,
        g=0.025,
        oscillator_length=(8.0 * fitted_E_C / fitted_E_L) ** 0.25,
        label="q-readout",
    )
    readout_chip = Chip(
        [readout_q, readout],
        [edge],
        basis="eigen",
        frame="lab",
        backend=DynamiqsBackend(),
    )

    readout_scale = jnp.asarray([5.175, 0.04])

    def readout_parameters(coordinates):
        return readout_scale * jnp.exp(coordinates)

    def readout_observables(parameters, phi_ext):
        fr, coupling = parameters
        point = readout_chip.with_params(
            {
                "q.phi_ext": phi_ext,
                "readout.freq": fr,
                "q-readout.g": coupling,
            }
        )
        fr0 = point.freq("readout")
        full_pull = point.dispersive_shift("q", "readout")
        return jnp.stack([fr0, fr0 + full_pull, 500.0 * full_pull])

    def readout_prediction(coordinates, flux):
        parameters = readout_parameters(coordinates)
        return jax.vmap(lambda phi: readout_observables(parameters, phi))(flux)

    def readout_loss(coordinates):
        prediction = readout_prediction(coordinates, readout_train_flux)
        residual = prediction - readout_train_data
        scaled = residual / jnp.asarray([0.001, 0.001, 0.25])
        return jnp.mean(pseudo_huber(scaled))

    initial_readout_parameters = np.asarray([5.165, 0.025])
    initial_readout_coordinates = np.log(
        initial_readout_parameters / np.asarray(readout_scale)
    )
    readout_history: list[float] = []
    compiled_readout_loss = jax.jit(jax.value_and_grad(readout_loss))
    readout_objective = scipy_value_and_grad(compiled_readout_loss)
    readout_fit = minimize(
        readout_objective,
        initial_readout_coordinates,
        method="L-BFGS-B",
        jac=True,
        bounds=[
            (np.log(5.15 / 5.175), np.log(5.20 / 5.175)),
            (np.log(0.005 / 0.04), np.log(0.08 / 0.04)),
        ],
        callback=lambda x: readout_history.append(readout_objective(x)[0]),
        options={"maxiter": 120, "ftol": 1.0e-12, "gtol": 1.0e-9},
    )
    recovered_readout = np.asarray(
        readout_parameters(jnp.asarray(readout_fit.x)), dtype=float
    )

    readout_model_flux = jnp.linspace(0.5, 0.85, 351)
    readout_model = np.asarray(
        readout_prediction(jnp.asarray(readout_fit.x), readout_model_flux)
    )
    readout_at_measurements = np.column_stack(
        [
            np.interp(readout_flux, np.asarray(readout_model_flux), readout_model[:, index])
            for index in range(3)
        ]
    )
    readout_residual = readout_at_measurements - np.column_stack(
        (readout_fr0, readout_fr1, readout_chi)
    )
    readout_holdout_residual = readout_residual[readout_holdout_mask]

    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), layout="constrained")
    spectrum_axis, readout_axis, spectrum_loss_axis, readout_loss_axis = axes.ravel()
    spectrum_axis.scatter(spectrum_flux, spectrum_data, s=11, color="0.25", alpha=0.55)
    spectrum_axis.plot(spectrum_model_flux, spectrum_model, color="#C92F33", linewidth=2.0)
    spectrum_axis.set(xlabel="External flux", ylabel=r"$f_{01}$ (GHz)")
    spectrum_axis.grid(color="0.88")

    readout_axis.scatter(readout_flux, readout_chi, s=11, color="0.25", alpha=0.55)
    readout_axis.plot(
        readout_model_flux,
        readout_model[:, 2],
        color="#246FA8",
        linewidth=2.0,
    )
    readout_axis.set(xlabel="External flux", ylabel=r"$\chi$ (MHz)")
    readout_axis.grid(color="0.88")

    spectrum_loss_axis.semilogy(spectrum_history, color="#C92F33")
    spectrum_loss_axis.set(xlabel="Optimizer iteration", ylabel="Spectrum loss")
    spectrum_loss_axis.grid(color="0.88")
    readout_loss_axis.semilogy(readout_history, color="#246FA8")
    readout_loss_axis.set(xlabel="Optimizer iteration", ylabel="Readout loss")
    readout_loss_axis.grid(color="0.88")
    figure_path = (
        FIGURE
        if args.offset == 0 and args.stride == 8
        else FIGURE.with_name(
            f"{FIGURE.stem}_stride_{args.stride}_offset_{args.offset}{FIGURE.suffix}"
        )
    )
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    receipt = {
        "spectrum_fit_success": bool(spectrum_fit.success),
        "spectrum_fit_message": str(spectrum_fit.message),
        "spectrum_train_points": int(spectrum_train_indices.size),
        "spectrum_holdout_points": int(spectrum_holdout_mask.sum()),
        "spectrum_iterations": int(spectrum_fit.nit),
        "published_E_C_E_J_E_L": published_energies.tolist(),
        "initial_E_C_E_J_E_L": initial_energies.tolist(),
        "recovered_E_C_E_J_E_L": recovered_energies.tolist(),
        "energy_relative_error": (
            (recovered_energies - published_energies) / published_energies
        ).tolist(),
        "spectrum_holdout_median_absolute_error_mhz": float(
            np.median(np.abs(spectrum_holdout_residual_mhz))
        ),
        "spectrum_holdout_rmse_mhz": float(
            np.sqrt(np.mean(spectrum_holdout_residual_mhz**2))
        ),
        "readout_fit_success": bool(readout_fit.success),
        "readout_fit_message": str(readout_fit.message),
        "readout_train_points": int(readout_train_indices.size),
        "readout_holdout_points": int(readout_holdout_mask.sum()),
        "readout_iterations": int(readout_fit.nit),
        "published_fr_g": [float(published_fr), float(published_g)],
        "initial_fr_g": initial_readout_parameters.tolist(),
        "recovered_fr_g": recovered_readout.tolist(),
        "readout_relative_error": (
            (recovered_readout - np.asarray([published_fr, published_g]))
            / np.asarray([published_fr, published_g])
        ).tolist(),
        "readout_holdout_frequency_rmse_mhz": float(
            1.0e3 * np.sqrt(np.mean(readout_holdout_residual[:, :2] ** 2))
        ),
        "readout_holdout_chi_rmse_mhz": float(
            np.sqrt(np.mean(readout_holdout_residual[:, 2] ** 2))
        ),
        "training_offset": args.offset,
        "training_stride": args.stride,
        "figure": str(figure_path.relative_to(ROOT)),
    }
    print("RESULT gradient_fit=" + json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
