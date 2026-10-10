"""Spectrum and eigenstate plots for each device."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from quchip.backend import get_default_backend
from quchip.declarative.expr import materialize_expr
from quchip.devices.base import BaseDevice
from quchip.engine.basis import resolve_device_basis
from quchip.viz._common import _basis_label, _draw_energy_ladder
from quchip.viz._style import _quchip_style, _resolve_single_axes


def plot_energy_levels(
    device: BaseDevice,
    *,
    ax: Any = None,
    color: str | None = None,
    linewidth: float = 2.0,
) -> Figure:
    """Plot the bare-Hamiltonian eigenenergies of a single device.

    Each eigenvalue of ``device.hamiltonian()`` shows as a horizontal bar with
    its index in the represented basis. The y-axis is energy in GHz. For a
    ``DuffingTransmon``, the gaps show the anharmonicity directly. For a
    ``Resonator``, the gaps are exactly equal.

    Parameters
    ----------
    device : BaseDevice
        Device whose bare spectrum is plotted.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. When ``None``, the function makes a new figure.
    color : str, optional
        Line colour for all levels. The default is the first ``tab10`` colour.
    linewidth : float
        Width of each level bar.

    Returns
    -------
    Figure
        The figure that holds the energy-ladder axes (``ax.figure`` when *ax*
        was given).

    Examples
    --------
    >>> import quchip as qc
    >>> qubit = qc.DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=4)
    >>> qubit.plot_energy_levels()  # doctest: +SKIP
    """
    backend = get_default_backend()
    energies = np.asarray(backend.eigenenergies(materialize_expr(device.hamiltonian(), backend)), dtype=float)
    level_color = color or plt.get_cmap("tab10")(0)
    entries = [(float(energy), _basis_label((level,))) for level, energy in enumerate(energies)]

    with _quchip_style():
        fig, axis = _resolve_single_axes(ax)
        _draw_energy_ladder(axis, entries, color=level_color, linewidth=linewidth)
        axis.set_ylabel("Energy (GHz)")
        axis.set_title(f"{device.label} energy levels", fontfamily="sans-serif")
        fig.tight_layout()
        return fig


def plot_wavefunction(
    device: BaseDevice,
    n: int,
    *,
    ax: Any = None,
    color: str | None = None,
) -> Figure:
    """Plot the represented-basis probability weights of eigenstate *n*.

    Show the eigenvector probabilities in the device's authored local
    coordinates. Circuit models can have more native basis coordinates than
    retained energy levels. The axis follows the actual length of the
    eigenvector.

    Parameters
    ----------
    device : BaseDevice
        Device whose eigenstates are diagonalised.
    n : int
        Eigenstate index (``0 <= n < device.levels``).
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. When ``None``, the function makes a new figure.
    color : str, optional
        Bar colour. The default is a ``tab10`` cycle per index.

    Returns
    -------
    Figure
        The figure that holds the bar-chart axes (``ax.figure`` when *ax* was
        given).

    Raises
    ------
    IndexError
        *n* is outside ``[0, device.levels)``.

    Examples
    --------
    >>> import quchip as qc
    >>> transmon = qc.DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=4)
    >>> transmon.plot_wavefunction(n=1)  # doctest: +SKIP
    """
    record = resolve_device_basis(device, basis="eigen", levels=device.levels)
    if n < 0 or n >= record.energy_vectors.shape[1]:
        raise IndexError(f"Eigenstate index {n} out of range for {device.levels} retained states")
    coefficients = np.asarray(record.energy_vectors[:, n])
    probabilities = np.abs(coefficients) ** 2
    x = np.arange(len(probabilities))
    cmap = plt.get_cmap("tab10")
    bar_colors: Any = [cmap(idx % cmap.N) for idx in x] if color is None else color

    with _quchip_style():
        fig, axis = _resolve_single_axes(ax)
        axis.bar(x, probabilities, color=bar_colors)
        ticks = x if len(x) <= 10 else np.linspace(0, len(x) - 1, 9, dtype=int)
        axis.set_xticks(ticks, [_basis_label((int(idx),)) for idx in ticks])
        axis.set_xlabel("Represented basis state")
        axis.set_ylabel("Probability")
        axis.set_ylim(0.0, 1.0)
        axis.set_title(f"{device.label} eigenstate n={n}", fontfamily="sans-serif")
        fig.tight_layout()
        return fig
