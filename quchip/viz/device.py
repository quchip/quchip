"""Per-device spectrum and eigenstate plots."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from quchip.backend import get_default_backend
from quchip.devices.base import BaseDevice
from quchip.viz._common import _basis_label, _draw_energy_ladder, _to_dense_array
from quchip.viz._style import _quchip_style, _resolve_single_axes


def plot_energy_levels(
    device: BaseDevice,
    *,
    ax: Any = None,
    color: str | None = None,
    linewidth: float = 2.0,
) -> Figure:
    """Plot a single device's bare-Hamiltonian eigenenergies.

    Each eigenvalue of ``device.hamiltonian()`` is drawn as a horizontal
    bar annotated with its index in the represented basis. The y-axis is
    energy in GHz. For a ``DuffingTransmon`` the gaps reveal the
    anharmonicity directly; for a ``Resonator`` they are exactly equal.

    Parameters
    ----------
    device : BaseDevice
        The device whose bare spectrum should be plotted.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw onto. When ``None`` a new figure is created.
    color : str, optional
        Line colour for every level. Defaults to the first ``tab10``
        colour.
    linewidth : float
        Width of each level bar.

    Returns
    -------
    Figure
        The figure holding the energy-ladder axes (``ax.figure`` when
        *ax* was given).

    Examples
    --------
    >>> import quchip as qc
    >>> qubit = qc.DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=4)
    >>> qubit.plot_energy_levels()  # doctest: +SKIP
    """
    backend = get_default_backend()
    energies = np.asarray(backend.eigenenergies(device.hamiltonian()), dtype=float)
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

    Shows ``|<k|psi_n>|^2`` for each bare basis state ``|k>``, ``k = 0
    ... levels - 1``, where ``|k>`` indexes whatever basis
    ``device.hamiltonian()`` is expressed in. Every stock device model
    shipped with ``quchip`` — including ``DuffingTransmon``,
    ``ChargeBasisTransmon``, ``Fluxonium``, and ``Resonator`` —
    returns a Hamiltonian that is already diagonal in its own retained
    eigenbasis, so eigenstate ``n`` is exactly ``|n>`` and the bar chart
    is a single delta bar for every one of them: this plot cannot show
    "mixing" for any built-in model. Mixing becomes visible only for a
    custom device whose ``hamiltonian()`` returns a matrix that is
    *not* diagonal in its represented basis.

    Parameters
    ----------
    device : BaseDevice
        The device whose eigenstates are diagonalised.
    n : int
        Eigenstate index (``0 <= n < device.levels``).
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw onto. When ``None`` a new figure is created.
    color : str, optional
        Bar colour. Defaults to a per-index ``tab10`` cycle.

    Returns
    -------
    Figure
        The figure holding the bar-chart axes (``ax.figure`` when *ax*
        was given).

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
    backend = get_default_backend()
    _energies, states = backend.eigenstates(device.hamiltonian())
    if n < 0 or n >= len(states):
        raise IndexError(f"Eigenstate index {n} out of range for {len(states)} states")

    coefficients = _to_dense_array(states[n], backend).reshape(-1)
    probabilities = np.abs(coefficients) ** 2
    x = np.arange(device.levels)
    cmap = plt.get_cmap("tab10")
    bar_colors: Any = [cmap(idx % cmap.N) for idx in x] if color is None else color

    with _quchip_style():
        fig, axis = _resolve_single_axes(ax)
        axis.bar(x, probabilities, color=bar_colors)
        axis.set_xticks(x, [_basis_label((int(idx),)) for idx in x])
        axis.set_xlabel("Represented basis state")
        axis.set_ylabel("Probability")
        axis.set_ylim(0.0, 1.0)
        axis.set_title(f"{device.label} eigenstate n={n}", fontfamily="sans-serif")
        fig.tight_layout()
        return fig
