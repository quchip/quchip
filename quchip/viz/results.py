"""Plots of simulation results: populations, snapshots, expectations, Wigner.

These helpers turn a backend-agnostic
:class:`~quchip.results.results.SimulationResult` into Matplotlib figures,
independent of the solver backend.

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
# Partial-trace, `expect`, and ket construction all go through `Backend`.

from __future__ import annotations

from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from quchip.devices.base import BaseDevice
from quchip.results.input_output import SParameterResult
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

    The x-axis is time (ns) and the y-axis is the population
    ``p_n(t) = Tr(|n><n| rho(t))``. One line is drawn for each
    represented-basis ket ``|n> = |n_1 n_2 ...>`` of the retained subsystems.
    States whose peak population stays below *threshold* are hidden. Set
    *threshold* to ``0`` to show all states.

    Parameters
    ----------
    result : SimulationResult
        Output of :func:`quchip.engine.simulate`.
    trace_out : device, label, or list thereof, optional
        Subsystems to partial-trace over before computing populations. Accepts
        device objects or their string labels. The solver call must use
        ``states="all"``.
    computational : bool
        When ``True``, restrict computational subsystems to their
        ``{|0>, |1>}`` subspace.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. When ``None``, a new figure is created.
    linewidth : float
        Line width for all population traces.
    legend : bool
        Draw a legend of the visible states.
    colors : dict, optional
        Per-state colour overrides, keyed by the internal basis-state tuples.
        States not in the dict use a ``tab20`` cycle.
    threshold : float
        Populations whose maximum over time is below this value are omitted.

    Returns
    -------
    Figure
        The figure that holds the population-trace axes (``ax.figure`` when
        *ax* was given).

    Raises
    ------
    RuntimeError
        *trace_out* is given but no states were stored (pass
        ``states="all"`` to the solver).
    ValueError
        *trace_out* would remove all subsystems.
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

    - ``"population"``: a bar chart of diagonal elements ``p_n``.
    - ``"dm"``: side-by-side heatmaps of ``Re(rho)`` and ``Im(rho)`` with the
      divergent colormap *cmap*. *Both* heatmaps share one symmetric
      normalization (``vmin=-m, vmax=+m`` for
      ``m = max(|Re(rho)|, |Im(rho)|)``), so their colours compare directly.
      Equal saturation in the two panels means equal magnitude.

    If computational and non-computational subsystems are both present, pass
    *trace_out* to focus on a target register. Alternatively, set
    *computational* ``= True`` to restrict computational devices to the
    ``{|0>, |1>}`` subspace (see Nielsen & Chuang, Ch. 2).

    Parameters
    ----------
    result : SimulationResult
        Output of :func:`quchip.engine.simulate`, with
        ``states="all"``.
    index : int
        Stored-time index to plot. Python-style negative indexes are supported
        (``-1`` is the last stored time). The index must satisfy
        ``-N <= index < N`` for ``N = len(result.times)``.
    trace_out : device, label, or list thereof, optional
        Subsystems to partial-trace over before plotting.
    computational : bool
        When ``True``, restrict computational subsystems to their
        ``{|0>, |1>}`` subspace.
    mode : {"population", "dm"}
        The representation to draw.
    ax : matplotlib.axes.Axes, optional
        For ``mode="population"``: a single axes (or ``None`` for a new
        figure). For ``mode="dm"``: an iterable of exactly two axes
        ``(real_ax, imag_ax)`` (or ``None`` for a new 1x2 figure).
    cmap : str
        Divergent colormap for ``mode="dm"`` heatmaps.
    color : str, optional
        Bar colour override for ``mode="population"``. The default is a
        ``tab10`` cycle per state.

    Returns
    -------
    Figure
        The figure that holds the plotted axes (``ax.figure`` when *ax* was
        given).

    Raises
    ------
    IndexError
        *index* is outside ``[-N, N)`` for ``N = len(result.times)``.
    ValueError
        *mode* is not ``"population"`` or ``"dm"``, or *trace_out* would remove
        all subsystems.
    RuntimeError
        No states were stored (pass ``states="all"``
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
    (e.g. the trailing integer of a ``(key, index)`` selector); that
    candidate is skipped.
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

    The x-axis is time (ns). The y-axis is
    :attr:`~quchip.results.results.SimulationResult.observable_traces`
    ``[key].values``, the *post-processed* recorded trace for each observable
    ``O`` registered in the solver's ``e_ops`` dict, which is not always
    ``Tr(O rho(t))``.

    ``.values`` depends on how you requested the observable and can already
    include demodulation, phase correction, or band summation (see
    :class:`~quchip.results.results.ObservableTrace`). Its ``.raw`` field holds
    the unprocessed quantity. When *real* is ``True`` (the default), only
    ``Re`` of the trace is drawn. When *real* is ``False``, the real part
    (solid) and the imaginary part (dashed, lower alpha) are drawn in the same
    colour per key.

    Parameters
    ----------
    result : SimulationResult
        Output of :func:`quchip.engine.simulate`, with ``e_ops`` passed
        as a dict.
    keys : list, optional
        Each entry is a bare key (``"cav"``) or a ``(key, index)`` tuple that
        selects one element of a list-valued observable. Entries are matched
        against the registered ``observable_traces`` keys first (see
        :func:`_collect_expectation_traces`). ``resolve_label`` resolves string
        keys and device/drive keys, so the two forms are interchangeable. The
        default is all registered traces.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. When ``None``, a new figure is created.
    linewidth : float
        Line width for all traces.
    legend : bool
        Draw a legend of the plotted keys.
    real : bool
        When ``True``, draw only the real part of each trace. When ``False``,
        draw the real (solid) and imaginary (dashed) parts.

    Returns
    -------
    Figure
        The figure that holds the expectation-trace axes (``ax.figure`` when
        *ax* was given).

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
    """Evaluate W(x, p) with [X, P] = i and integral Tr(rho).

    QuTiP's Clenshaw expansion uses alpha = (x + i*p)/sqrt(2).
    See Leonhardt, Essential Quantum Optics, chapter 3.
    """
    from qutip import Qobj, wigner

    return wigner(Qobj(rho), xvec, yvec, g=np.sqrt(2), method="clenshaw")


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

    The axes are the phase-space quadratures ``x`` (position-like) and ``p``
    (momentum-like). The colourmap is divergent and symmetric about zero, so
    negative regions, the hallmark of non-classical states, are easy to see.

    Without *xvec*, the plot window is sized from the reduced state's mean
    photon number ``<n> = Tr(rho n_hat)`` and extends to at least ``+/-3``. The
    mean photon number is computed directly from ``diag(rho)``.

    Exactly one subsystem must remain after *trace_out*. A Wigner function is a
    single-mode phase-space picture, and its basis indices are read directly as
    photon numbers ``n = 0, 1, 2, ...``. The function checks this condition.

    One more precondition cannot be seen from the result metadata, so the
    function does *not* check it: the retained subsystem's represented basis
    must *be* a photon-number ladder. This holds for a bosonic mode such as
    ``Resonator``, but not for a device whose represented basis is not Fock,
    for example a charge-basis or flux-basis qubit. If you pass such a device,
    the function silently makes a Wigner-shaped plot without physical meaning.

    Parameters
    ----------
    result : SimulationResult
        Output of :func:`quchip.engine.simulate`, with
        ``states="all"``.
    index : int
        Stored-time index to plot. Python-style negative indexes are supported
        (``-1``, the default, is the last stored time). The index must satisfy
        ``-N <= index < N`` for ``N = len(result.times)``.
    trace_out : device, label, or list thereof, optional
        Subsystems to partial-trace over before plotting. Required when more
        than one subsystem is stored (see Raises).
    xvec, yvec : ndarray, optional
        Phase-space grids for the ``x``/``p`` quadratures. The default is an
        automatically sized, equally spaced grid (see above). If only *xvec* is
        given, *yvec* defaults to *xvec*.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. When ``None``, a new figure is created.
    cmap : str
        Divergent colormap, symmetric about zero.
    colorbar : bool
        Attach a colorbar.

    Returns
    -------
    Figure
        The figure that holds the Wigner-function axes (``ax.figure`` when *ax*
        was given).

    Raises
    ------
    IndexError
        *index* is outside ``[-N, N)`` for ``N = len(result.times)``.
    ValueError
        More or fewer than one subsystem remains after *trace_out*. The message
        lists the retained device labels and a *trace_out* value that isolates
        a single one of them.
    RuntimeError
        No states were stored (pass ``states="all"``
        to the solver).

    References
    ----------
    - Wigner, *Phys. Rev.* **40**, 749 (1932).
    - Cahill & Glauber, *Phys. Rev.* **177**, 1882 (1969).
    - Leonhardt, *Essential Quantum Optics* (2010), Ch. 3.
    """
    # Computing the mean photon number from `diag(rho)` avoids an O(d^2) matmul
    # for a diagonal observable.
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


def plot_sparameters(
    result: Any,
    pairs: list[tuple[Any, Any]] | None = None,
    *,
    select: dict[str, int] | None = None,
    kind: Literal["db_phase", "magnitude", "iq"] = "db_phase",
    axes: Any = None,
) -> Figure:
    """Plot entries from a small-signal scattering result.

    Parameters
    ----------
    result : SParameterResult
        Result that :meth:`VNA.sweep` returns. The function rejects other result types.
    pairs : list of (output, input), optional
        Matrix entries to draw. If omitted with exactly two planes, all four
        entries are drawn. Otherwise, the first input column is drawn, up to six
        entries.
    select : dict of str to int, optional
        Indices on sweep axes other than frequency. Omitted axes use index 0,
        and the plot title shows those defaults. Frequency stays the x axis, and
        you cannot select it.
    kind : {"db_phase", "magnitude", "iq"}
        ``"db_phase"`` stacks ``20 log10|S|`` in dB above unwrapped phase in
        degrees. ``"magnitude"`` draws ``|S|``. ``"iq"`` draws ``Im S`` against
        ``Re S``.
    axes : matplotlib.axes.Axes or iterable of matplotlib.axes.Axes, optional
        Existing axes to draw on: two for ``"db_phase"`` and one for the other
        kinds. When ``None``, create the necessary axes.

    Returns
    -------
    Figure
        The figure that contains the plots.

    Raises
    ------
    TypeError
        *result* is not an SParameterResult.
    ValueError
        *kind* is not ``"db_phase"``, ``"magnitude"``, or ``"iq"``, *select*
        names ``"frequency"``, or no sweep axis remains for the x axis.
    KeyError
        *select* contains an unknown sweep-axis name.
    """
    if not isinstance(result, SParameterResult):
        raise TypeError(f"plot_sparameters requires an SParameterResult; got {type(result).__name__}.")
    if kind not in {"db_phase", "magnitude", "iq"}:
        raise ValueError(f"Unknown plot kind {kind!r}; use 'db_phase', 'magnitude', or 'iq'.")
    ports = result.ports
    if pairs is None:
        pairs = (
            [(out, inp) for inp in ports for out in ports]
            if len(ports) == 2
            else [(out, ports[0]) for out in ports][:6]
        )
    names = result.axis_names
    select = select or {}
    unknown = set(select) - set(names)
    if unknown:
        raise KeyError(f"Unknown axes {sorted(unknown)}; available: {list(names)}")
    if "frequency" in select:
        raise ValueError("select indexes non-frequency axes; frequency is always the x axis.")
    free = [name for name in names if name not in select]
    if not free:
        raise ValueError("plot_sparameters needs one unselected sweep axis; the result has none.")
    sweep_axis = "frequency" if "frequency" in free else free[0]
    index = tuple(slice(None) if name == sweep_axis else select.get(name, 0) for name in names)
    x_values = np.asarray(dict(result.axes)[sweep_axis], dtype=float)
    defaulted = [name for name in names if name != sweep_axis and name not in select]
    title = f"defaults: {', '.join(f'{name}[0]' for name in defaulted)}" if defaulted else ""

    colors = _cyclic_colors(range(len(pairs)), "tab10")
    with _quchip_style():
        drawn: tuple[Any, ...]
        if kind == "db_phase":
            if axes is None:
                fig, drawn = plt.subplots(2, 1, sharex=True, figsize=(6.0, 6.0))
            else:
                drawn = tuple(axes)
                fig = drawn[0].figure
        else:
            fig, single = _resolve_single_axes(axes)
            drawn = (single,)
        for idx, (output, input_) in enumerate(pairs):
            values = np.asarray(result.s(output, input_))[index]
            label = f"S({resolve_label(output)}, {resolve_label(input_)})"
            color = colors[idx]
            if kind == "db_phase":
                with np.errstate(divide="ignore"):
                    drawn[0].plot(x_values, 20.0 * np.log10(np.abs(values)), label=label, color=color)
                drawn[1].plot(x_values, np.degrees(np.unwrap(np.angle(values))), label=label, color=color)
            elif kind == "magnitude":
                drawn[0].plot(x_values, np.abs(values), label=label, color=color)
            else:
                drawn[0].plot(np.real(values), np.imag(values), label=label, color=color)
        x_label = "Frequency (GHz)" if sweep_axis == "frequency" else sweep_axis
        if kind == "db_phase":
            drawn[0].set_ylabel("|S| (dB)")
            drawn[1].set_ylabel("Phase (deg)")
            drawn[1].set_xlabel(x_label)
        elif kind == "magnitude":
            drawn[0].set_ylabel("|S|")
            drawn[0].set_xlabel(x_label)
        else:
            drawn[0].set_xlabel("Re S")
            drawn[0].set_ylabel("Im S")
            drawn[0].set_aspect("equal", adjustable="datalim")
        drawn[0].legend()
        if title:
            drawn[0].set_title(title, fontsize=9)
    return fig
