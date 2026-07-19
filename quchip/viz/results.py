"""Simulation-result plots: populations, snapshots, expectations, Wigner.

These helpers consume a :class:`~quchip.results.results.SimulationResult`
(which is backend-agnostic) and emit Matplotlib figures. None of them
depend on a specific solver backend: partial-trace, ``expect``, and ket
construction all go through :class:`~quchip.backend.protocol.Backend`.

References
----------
- Nielsen & Chuang, *Quantum Computation and Quantum Information*,
  Cambridge University Press (2010) — density matrices, populations.
- Wigner, "On the Quantum Correction for Thermodynamic Equilibrium",
  *Phys. Rev.* **40**, 749 (1932) — original Wigner-function definition.
- Cahill & Glauber, "Density Operators and Quasiprobability Distributions",
  *Phys. Rev.* **177**, 1882 (1969) — Laguerre-polynomial expansion used
  by :func:`_wigner_from_density_matrix`.
- Leonhardt, *Essential Quantum Optics*, Cambridge University Press (2010)
  — modern continuous-variable treatment of the Wigner function.
"""

from __future__ import annotations

from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from quchip.devices.base import BaseDevice
from quchip.utils.labeling import resolve_label
from quchip.viz._common import (
    _basis_label,
    _normalize_time_index,
    _project_density_matrix,
    _reduce_result,
    _reduce_state,
    _to_dense_array,
)
from quchip.viz._style import _cyclic_colors, _quchip_style, _resolve_dual_axes, _resolve_single_axes

StateMode = Literal["population", "dm"]


def _state_colors(
    states: list[tuple[int, ...]],
    override: dict[tuple[int, ...], str] | None = None,
) -> dict[tuple[int, ...], Any]:
    """tab20 defaults with optional per-state overrides."""
    colors = _cyclic_colors(states, "tab20")
    if override:
        colors.update(override)
    return colors


def plot_populations(
    result: Any,
    *,
    trace_out: str | BaseDevice | list[str | BaseDevice] | None = None,
    computational: bool = False,
    ax: Any = None,
    linewidth: float = 2.5,
    legend: bool = True,
    colors: dict[tuple[int, ...], str] | None = None,
    threshold: float = 0.01,
) -> Figure:
    """Plot basis-state populations over time.

    The figure has time (ns) on the x-axis and population
    ``p_n(t) = Tr(|n><n| rho(t))`` on the y-axis, one line per
    represented-basis ket ``|n> = |n_1 n_2 ...>`` of the retained
    subsystems. States whose peak population stays below *threshold* are
    hidden; set *threshold* to ``0`` to show every state.

    Parameters
    ----------
    result : SimulationResult
        Output of :func:`quchip.engine.simulate`.
    trace_out : device, label, or list thereof, optional
        Subsystems to partial-trace over before computing populations.
        Accepts either device objects or their string labels (UX favourability).
        Requires ``options={"store_states": True}`` on the solver call.
    computational : bool
        When ``True``, restricts computational subsystems to their
        ``{|0>, |1>}`` subspace.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw onto. When ``None`` a new figure is created.
    linewidth : float
        Line width for every population trace.
    legend : bool
        Whether to draw a legend of the visible states.
    colors : dict, optional
        Per-state colour overrides, keyed by the same basis-state tuples
        used internally; unlisted states fall back to a ``tab20`` cycle.
    threshold : float
        Populations whose time-max falls below this value are omitted.

    Returns
    -------
    Figure
        The figure holding the population-trace axes (``ax.figure`` when
        *ax* was given).

    Raises
    ------
    RuntimeError
        *trace_out* is given but no states were stored (pass
        ``options={"store_states": True}`` to the solver).
    ValueError
        *trace_out* would remove every subsystem.
    """
    times, populations, _keep, _info = _reduce_result(result, trace_out, computational=computational)

    visible_states = [s for s in populations if np.max(populations[s]) >= threshold]
    if not visible_states:
        visible_states = list(populations)
    state_colors = _state_colors(visible_states, colors)

    with _quchip_style():
        fig, axis = _resolve_single_axes(ax)
        for state in visible_states:
            axis.plot(
                times, populations[state],
                label=_basis_label(state),
                linewidth=linewidth,
                color=state_colors[state],
            )

        axis.set_xlabel("Time (ns)")
        axis.set_ylabel("Population")
        axis.set_ylim(-0.02, 1.05)
        axis.set_title("State populations", fontfamily="sans-serif")
        if legend and visible_states:
            ncol = max(1, (len(visible_states) + 2) // 3)
            axis.legend(
                ncol=ncol, fontsize="small",
                loc="upper center", bbox_to_anchor=(0.5, 1.0),
                framealpha=0.8,
            )
        fig.tight_layout()
        return fig


def plot_state(
    result: Any,
    index: int,
    *,
    trace_out: str | BaseDevice | list[str | BaseDevice] | None = None,
    computational: bool = False,
    mode: StateMode = "population",
    ax: Any = None,
    cmap: str = "RdBu_r",
    color: str | None = None,
) -> Figure:
    """Plot a single stored state at time-index *index*.

    Two modes are supported:

    - ``"population"`` — a bar chart of diagonal elements ``p_n``.
    - ``"dm"`` — side-by-side heatmaps of ``Re(rho)`` and ``Im(rho)``;
      the colormap *cmap* is divergent, and *both* heatmaps share one
      symmetric normalization (``vmin=-m, vmax=+m`` for
      ``m = max(|Re(rho)|, |Im(rho)|)``) so their colours are directly
      comparable — an entry that looks equally saturated in both panels
      really is equal in magnitude.

    When both computational and non-computational subsystems are present,
    pass *trace_out* to focus on a target register and/or *computational*
    ``= True`` to restrict to the ``{|0>, |1>}`` subspace on computational
    devices (see Nielsen & Chuang, Ch. 2).

    Parameters
    ----------
    result : SimulationResult
        Output of :func:`quchip.engine.simulate`, with
        ``options={"store_states": True}``.
    index : int
        Stored-time index to plot. Supports Python-style negative
        indexing (``-1`` is the last stored time); must satisfy
        ``-N <= index < N`` for ``N = len(result.times)``.
    trace_out : device, label, or list thereof, optional
        Subsystems to partial-trace over before plotting.
    computational : bool
        When ``True``, restricts computational subsystems to their
        ``{|0>, |1>}`` subspace.
    mode : {"population", "dm"}
        Which representation to draw.
    ax : matplotlib.axes.Axes, optional
        For ``mode="population"``: a single axes (or ``None`` for a new
        figure). For ``mode="dm"``: an iterable of exactly two axes
        ``(real_ax, imag_ax)`` (or ``None`` for a new 1x2 figure).
    cmap : str
        Divergent colormap for ``mode="dm"`` heatmaps.
    color : str, optional
        Bar colour override for ``mode="population"``. Defaults to a
        per-state ``tab10`` cycle.

    Returns
    -------
    Figure
        The figure holding the plotted axes (``ax.figure`` when *ax*
        was given).

    Raises
    ------
    IndexError
        *index* is outside ``[-N, N)`` for ``N = len(result.times)``.
    ValueError
        *mode* is not ``"population"`` or ``"dm"``, or *trace_out*
        would remove every subsystem.
    RuntimeError
        No states were stored (pass ``options={"store_states": True}``
        to the solver).
    """
    index = _normalize_time_index(result, index)

    if mode == "population":
        times, populations, _keep, _info = _reduce_result(
            result, trace_out, computational=computational,
        )
        states = list(populations)
        values = [float(populations[state][index]) for state in states]
        cmap10 = plt.get_cmap("tab10")
        bar_colors: Any = (
            [cmap10(idx % cmap10.N) for idx in range(len(states))] if color is None else color
        )

        with _quchip_style():
            fig, axis = _resolve_single_axes(ax)
            x = np.arange(len(states))
            axis.bar(x, values, color=bar_colors)
            axis.set_xticks(x, [_basis_label(state) for state in states])
            axis.set_ylim(0.0, 1.0)
            axis.set_ylabel("Population")
            axis.set_title(
                f"State populations at t={float(times[index]):.3g} ns",
                fontfamily="sans-serif",
            )
            fig.tight_layout()
            return fig

    if mode != "dm":
        raise ValueError("mode must be 'population' or 'dm'")

    reduced_state, keep_indices, device_info = _reduce_state(result, index, trace_out)
    dims = [result.dims[idx] for idx in keep_indices]
    backend = result._backend
    dm = backend.as_density_matrix(reduced_state)
    dense_dm = _to_dense_array(dm, backend)
    projected_dm, plotted_states = _project_density_matrix(dense_dm, dims, device_info, computational)
    labels = [_basis_label(state) for state in plotted_states]

    real_part, imag_part = np.real(projected_dm), np.imag(projected_dm)
    m = max(float(np.max(np.abs(real_part))), float(np.max(np.abs(imag_part))), 1e-12)

    with _quchip_style():
        fig, (real_ax, imag_ax) = _resolve_dual_axes(ax)
        real_ax.imshow(real_part, cmap=cmap, interpolation="nearest", vmin=-m, vmax=m)
        imag_ax.imshow(imag_part, cmap=cmap, interpolation="nearest", vmin=-m, vmax=m)
        for axis, title in ((real_ax, "Re(rho)"), (imag_ax, "Im(rho)")):
            axis.set_title(title, fontfamily="sans-serif")
            axis.set_xticks(range(len(labels)), labels)
            axis.set_yticks(range(len(labels)), labels)
        fig.tight_layout()
        return fig


def _resolved_candidate_keys(entry: Any) -> list[Any]:
    """Return exact-match candidate keys for *entry*, raw form first.

    The raw *entry* is always a candidate — it matches already-correct
    keys and any exotic hashable ``resolve_label`` cannot handle. A
    ``resolve_label``-resolved form is added when it applies cleanly:
    element-wise for tuples (so a device-object correlator key like
    ``(q0, q1)`` resolves to ``("q0", "q1")``), or directly otherwise.
    ``resolve_label`` raises ``TypeError`` on values it cannot resolve
    (e.g. the trailing integer of a ``(key, index)`` selector) — that
    candidate is simply skipped rather than propagating.
    """
    candidates = [entry]
    try:
        candidates.append(
            tuple(resolve_label(part) for part in entry) if isinstance(entry, tuple) else resolve_label(entry)
        )
    except TypeError:
        pass
    return candidates


def _collect_expectation_traces(
    result: Any,
    keys: list[Any] | None,
) -> list[tuple[str, np.ndarray]]:
    """Return ``(label, values)`` traces to plot from dict-form ``e_ops``.

    When *keys* is ``None`` every recorded trace is returned. Otherwise
    each entry is matched against the registered
    :attr:`~quchip.results.results.SimulationResult.observable_traces`
    keys FIRST (see :func:`_resolved_candidate_keys`) — including tuple
    entries, so a legitimate correlator key like ``("q0", "q1")`` is
    never misread as a list-trace selector. Only when a two-element
    tuple does not itself resolve to a registered key is it interpreted
    as a ``(key, index)`` pair selecting one element of a list-valued
    observable.

    Raises
    ------
    KeyError
        An entry in *keys* is neither a registered
        ``observable_traces`` key (raw or resolved) nor a valid
        ``(key, index)`` list-trace selector.
    """
    traces_dict = result.observable_traces
    traces: list[tuple[str, np.ndarray]] = []

    def _append(label: str, trace: Any) -> None:
        if isinstance(trace, list):
            for idx, tr in enumerate(trace):
                traces.append((f"{label}[{idx}]", np.asarray(tr.values)))
        else:
            traces.append((label, np.asarray(trace.values)))

    if keys is None:
        for key, trace in traces_dict.items():
            _append(str(key), trace)
        return traces

    for entry in keys:
        for candidate in _resolved_candidate_keys(entry):
            if candidate in traces_dict:
                _append(str(entry), traces_dict[candidate])
                break
        else:
            if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], int):
                key, index = entry
                traces.append((f"{key}[{index}]", np.asarray(result.expect(key, index=index))))
            else:
                raise KeyError(
                    f"{entry!r} is not a registered observable_traces key and not a "
                    "valid (key, index) list-trace selector"
                )
    return traces


def plot_expectation(
    result: Any,
    *,
    keys: list[Any] | None = None,
    ax: Any = None,
    linewidth: float = 2.5,
    legend: bool = True,
    real: bool = True,
) -> Figure:
    """Plot dict-form expectation values over time.

    The x-axis is time (ns); the y-axis is
    :attr:`~quchip.results.results.SimulationResult.observable_traces`
    ``[key].values`` — the *post-processed* recorded trace for each
    observable ``O`` registered in the solver's ``e_ops`` dict. This is
    not unconditionally ``Tr(O rho(t))``: depending on how the
    observable was requested, ``.values`` may already include
    demodulation, phase correction, or band summation (see
    :class:`~quchip.results.results.ObservableTrace`; its ``.raw``
    field holds the pre-processing quantity instead). When *real* is
    ``True`` (the default) only ``Re`` of the trace is drawn; when
    ``False`` both the real part (solid) and imaginary part (dashed,
    lower alpha) are drawn in the same colour per key.

    Parameters
    ----------
    result : SimulationResult
        Output of :func:`quchip.engine.simulate`, with ``e_ops`` passed
        as a dict.
    keys : list, optional
        Each entry is either a bare key (``"cav"``) or a ``(key, index)``
        tuple selecting one element of a list-valued observable, matched
        against the registered ``observable_traces`` keys first — see
        :func:`_collect_expectation_traces`. String and device/drive
        keys are resolved with ``resolve_label`` so both are accepted
        interchangeably. Defaults to every registered trace.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw onto. When ``None`` a new figure is created.
    linewidth : float
        Line width for every trace.
    legend : bool
        Whether to draw a legend of the plotted keys.
    real : bool
        When ``True``, draw only the real part of each trace; when
        ``False``, draw both the real (solid) and imaginary (dashed)
        parts.

    Returns
    -------
    Figure
        The figure holding the expectation-trace axes (``ax.figure``
        when *ax* was given).

    Raises
    ------
    TypeError
        ``e_ops`` was not passed as a dict (``observable_traces`` is
        ``None``).
    KeyError
        An entry in *keys* does not resolve to a registered trace or a
        valid ``(key, index)`` selector.
    """
    if not isinstance(result.observable_traces, dict):
        raise TypeError("plot_expectation requires dict-form e_ops")

    traces = _collect_expectation_traces(result, keys)
    times = np.asarray(result.times, dtype=float)
    cmap = plt.get_cmap("tab10")

    with _quchip_style():
        fig, axis = _resolve_single_axes(ax)
        for idx, (label, values) in enumerate(traces):
            color = cmap(idx % cmap.N)
            axis.plot(
                times, np.real(values),
                label=label if real else f"Re({label})",
                linewidth=linewidth, color=color,
            )
            if not real:
                axis.plot(
                    times, np.imag(values), label=f"Im({label})",
                    linewidth=linewidth, color=color, linestyle="--", alpha=0.7,
                )

        axis.set_xlabel("Time (ns)")
        axis.set_ylabel("Expectation value")
        if legend and traces:
            axis.legend(fontsize="small", framealpha=0.8)
        axis.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig


def _wigner_from_density_matrix(rho: np.ndarray, xvec: np.ndarray, yvec: np.ndarray) -> np.ndarray:
    """Wigner function of *rho* via the Laguerre-polynomial Fock expansion.

    Uses the optics convention ``alpha = (x + i y) / sqrt(2)`` so that
    ``x = <X>`` and ``p = <P>`` with ``[X, P] = i``. The normalisation is
    the standard one: ``int W(x, p) dx dp = Tr(rho) = 1``.

    The series is the Cahill-Glauber representation
    (``Phys. Rev. 177, 1882, 1969``),

    .. math::

        W(\\alpha) = \\frac{2}{\\pi} \\sum_{m, n} \\rho_{mn}
          \\langle n | D(\\alpha) (-1)^{\\hat N} D^{\\dagger}(\\alpha) | m \\rangle,

    evaluated with associated Laguerre polynomials (see Leonhardt,
    *Essential Quantum Optics*, Ch. 3).
    """
    from scipy.special import gammaln, genlaguerre

    dim = rho.shape[0]
    X, Y = np.meshgrid(xvec, yvec)
    A = (X + 1j * Y) / np.sqrt(2)
    B = 4.0 * np.abs(A) ** 2

    W = np.zeros(A.shape, dtype=float)
    for m in range(dim):
        if np.abs(rho[m, m]) > 0.0:
            W += np.real(rho[m, m] * (-1) ** m * genlaguerre(m, 0)(B))
        for n in range(m + 1, dim):
            if np.abs(rho[m, n]) > 0.0:
                W += 2.0 * np.real(
                    rho[m, n]
                    * (-1) ** m
                    * (2.0 * A) ** (n - m)
                    * np.exp(0.5 * (gammaln(m + 1) - gammaln(n + 1)))
                    * genlaguerre(m, n - m)(B)
                )

    return W * np.exp(-B / 2.0) / np.pi


def plot_wigner(
    result: Any,
    index: int = -1,
    *,
    trace_out: str | BaseDevice | list[str | BaseDevice] | None = None,
    xvec: np.ndarray | None = None,
    yvec: np.ndarray | None = None,
    ax: Any = None,
    cmap: str = "RdBu_r",
    colorbar: bool = True,
) -> Figure:
    """Plot the Wigner quasi-probability distribution of a stored state.

    Axes are the phase-space quadratures ``x`` (position-like) and
    ``p`` (momentum-like); the colourmap is divergent and symmetric about
    zero so negative regions — the hallmark of non-classical states —
    stand out directly.

    When *xvec* is not supplied the plot window is auto-sized from the
    mean photon number ``<n> = Tr(rho n_hat)`` of the reduced state
    (computed directly from ``diag(rho)`` to avoid an O(d^2) matmul for
    a diagonal-only observable), extending to at least ``+/-3``.

    Exactly one subsystem must remain after *trace_out* — a Wigner
    function is a single-mode phase-space picture, and its basis indices
    are interpreted directly as photon numbers ``n = 0, 1, 2, ...``.
    This is checked; what is *not*, and cannot be, checked from result
    metadata alone is the remaining precondition: the retained
    subsystem's represented basis must actually *be* a photon-number
    ladder (true for a bosonic mode such as ``Resonator``, false for a
    device whose represented basis is not Fock, e.g. a charge- or
    flux-basis qubit) — passing such a device silently produces a
    Wigner-shaped plot with no such physical meaning.

    Parameters
    ----------
    result : SimulationResult
        Output of :func:`quchip.engine.simulate`, with
        ``options={"store_states": True}``.
    index : int
        Stored-time index to plot. Supports Python-style negative
        indexing (``-1``, the default, is the last stored time); must
        satisfy ``-N <= index < N`` for ``N = len(result.times)``.
    trace_out : device, label, or list thereof, optional
        Subsystems to partial-trace over before plotting. Required
        whenever more than one subsystem is stored — see Raises.
    xvec, yvec : ndarray, optional
        Phase-space grids for the ``x``/``p`` quadratures. Defaults to
        an auto-sized, evenly spaced grid (see above); *yvec* defaults
        to *xvec* when only *xvec* is given.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw onto. When ``None`` a new figure is created.
    cmap : str
        Divergent colormap, symmetric about zero.
    colorbar : bool
        Whether to attach a colorbar.

    Returns
    -------
    Figure
        The figure holding the Wigner-function axes (``ax.figure`` when
        *ax* was given).

    Raises
    ------
    IndexError
        *index* is outside ``[-N, N)`` for ``N = len(result.times)``.
    ValueError
        More or fewer than one subsystem remains after *trace_out*; the
        message lists the retained device labels and a *trace_out*
        value that isolates a single one of them.
    RuntimeError
        No states were stored (pass ``options={"store_states": True}``
        to the solver).

    References
    ----------
    - Wigner, *Phys. Rev.* **40**, 749 (1932).
    - Cahill & Glauber, *Phys. Rev.* **177**, 1882 (1969).
    - Leonhardt, *Essential Quantum Optics* (2010), Ch. 3.
    """
    index = _normalize_time_index(result, index)
    reduced_state, keep_indices, device_info = _reduce_state(result, index, trace_out)
    if len(keep_indices) != 1:
        retained_labels = [label for label, _computational in device_info]
        raise ValueError(
            "plot_wigner requires exactly one retained subsystem (its basis indices are "
            f"interpreted as photon numbers); {len(retained_labels)} are retained: "
            f"{retained_labels}. Pass trace_out naming all but one of these devices, e.g. "
            f"trace_out={retained_labels[:-1]!r} to keep only {retained_labels[-1]!r}."
        )
    backend = result._backend

    dm = backend.as_density_matrix(reduced_state)
    rho = np.asarray(backend.to_array(dm), dtype=complex)
    dim = rho.shape[0]

    if xvec is None:
        n_mean = float(np.real(np.sum(np.diag(rho) * np.arange(dim))))
        extent = max(np.sqrt(max(n_mean, 0.0)) * 2.5, 3.0)
        xvec = np.linspace(-extent, extent, 200)
    if yvec is None:
        yvec = xvec

    W = _wigner_from_density_matrix(rho, xvec, yvec)
    wmax = np.max(np.abs(W))

    with _quchip_style():
        fig, axis = _resolve_single_axes(ax)
        im = axis.contourf(xvec, yvec, W, levels=100, cmap=cmap, vmin=-wmax, vmax=wmax)
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(r"$p$")
        axis.set_aspect("equal")
        t = float(result.times[index])
        axis.set_title(f"Wigner function at t = {t:.1f} ns", fontfamily="sans-serif")
        if colorbar:
            fig.colorbar(im, ax=axis, label=r"$W(\alpha)$")
        fig.tight_layout()
        return fig
