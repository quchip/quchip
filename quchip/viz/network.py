"""Static schematic of a chip's PortNetwork: Markov core, reference sections, planes."""

from __future__ import annotations

from collections import deque
from typing import Any

from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from quchip.chip.port_network import PortNetwork
from quchip.utils.jax_utils import maybe_concrete_scalar
from quchip.utils.labeling import resolve_label
from quchip.viz._style import _quchip_style, _resolve_single_axes

_CORE_COLOR = "#dbe9f6"
_REFERENCE_COLOR = "#f6e7c1"
_PLANE_COLOR = "#e2f0d9"
_PLANE_EDGE = "#2f7d32"
_HIDDEN_COLOR = "#7f7f7f"
_COLUMN_WIDTH = 3.0
_ROW_HEIGHT = 2.8


def _value_text(value: Any, precision: int = 3) -> str:
    """Format one tracked value; traced values render as ``<traced>``."""
    concrete = maybe_concrete_scalar(value)
    return "<traced>" if concrete is None else f"{concrete:.{precision}g}"


def _network_of(source: Any) -> PortNetwork:
    network = source if isinstance(source, PortNetwork) else getattr(source, "port_network", None)
    if network is None:
        raise ValueError("plot_port_network requires a PortNetwork or a chip with one attached.")
    return network


def _annotation(component: Any) -> list[str]:
    """Return the value lines drawn under a component label."""
    kind = component._kind
    parameters = component._parameters
    ports = [port for port in component._local_ports if port is not None]
    if ports:
        port = ports[0]
        targets = ", ".join(resolve_label(target) for target in port._targets)
        if port.rate is not None:
            return [f"port -> {targets}", f"rate {_value_text(port.rate)} /ns"]
        return [f"port -> {targets}", f"Q_ext {_value_text(port.external_quality_factor)}"]
    if kind == "circulator":
        return ["->".join((*component.sides, component.sides[0]))]
    if kind == "isolator":
        return ["1 -> 2", "2 -> load"]
    if kind == "filter":
        return [f"filter({', '.join(parameters)})"]
    if kind == "delay":
        return [f"{_value_text(parameters['duration'])} ns"]
    if kind == "amplifier":
        return [f"G {_value_text(parameters['gain'])}", f"n_add {_value_text(parameters['added_noise'])}"]
    if kind in {"attenuator", "beam_splitter"}:
        return [f"eta {_value_text(parameters['eta'])}"]
    if kind == "phase_shift":
        return [f"phase {_value_text(parameters['phase'])} rad"]
    return [kind]


def _columns(network: PortNetwork) -> dict[str, int]:
    """Assign each component a column by undirected graph distance from the chip ports."""
    neighbours: dict[str, set[str]] = {component.label: set() for component in network.components}
    for input_key, output_key in network._connections.items():
        neighbours[input_key[0]].add(output_key[0])
        neighbours[output_key[0]].add(input_key[0])
    columns: dict[str, int] = {}
    queue: deque[str] = deque()
    for component in network.components:
        if any(port is not None for port in component._local_ports):
            columns[component.label] = 0
            queue.append(component.label)
    while queue:
        label = queue.popleft()
        for other in sorted(neighbours[label]):
            if other not in columns:
                columns[other] = columns[label] + 1
                queue.append(other)
    for component in network.components:
        columns.setdefault(component.label, 0)
    return columns


def _box(axis: Any, x: float, y: float, width: float, height: float, **style: Any) -> None:
    axis.add_patch(
        FancyBboxPatch((x - width / 2, y - height / 2), width, height, boxstyle="round,pad=0.05", **style)
    )


def plot_port_network(
    source: Any,
    *,
    show_hidden: bool = True,
    ax: Any = None,
) -> Figure:
    """Draw a port network as a left-to-right schematic.

    Chip ports and Markovian components show in the shaded ``Markov core``. Reference sections
    (delays, filters, and amplifiers) and external planes show to its right. Cables made with
    ``link`` use double-headed edges, and directional connections use single arrows. A plane with
    separate ``input=`` and ``output=`` terminals has one arrow to each terminal. When *show_hidden*
    is true, hidden vacuum and load channels show as dashed stubs. Annotations show component
    values, including rates, ``eta``, durations, gain, and added noise. Traced values show as
    ``<traced>``.

    Parameters
    ----------
    source : Chip or PortNetwork
        A chip with an attached port network, or the network itself.
    show_hidden : bool
        Draw hidden vacuum and load stubs.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. When ``None``, create a figure and axes.

    Returns
    -------
    Figure
        The figure that contains the schematic (``ax.figure`` when *ax* is given).

    Raises
    ------
    ValueError
        *source* is not a PortNetwork and has no attached port network.
    """
    network = _network_of(source)
    columns = _columns(network)
    rows: dict[int, list[str]] = {}
    for label, column in columns.items():
        rows.setdefault(column, []).append(label)
    positions: dict[str, tuple[float, float]] = {}
    for column, labels in rows.items():
        for index, label in enumerate(labels):
            positions[label] = (_COLUMN_WIDTH * column, -_ROW_HEIGHT * (index - (len(labels) - 1) / 2.0))
    plane_column = max(columns.values(), default=0) + 1
    exposures = network.external_ports
    plane_positions: dict[str, tuple[float, float]] = {}
    for exposure in exposures:
        y = (positions[exposure._input_key[0]][1] + positions[exposure._output_key[0]][1]) / 2.0
        while any(abs(y - taken[1]) < 0.7 for taken in plane_positions.values()):
            y -= 0.8
        plane_positions[exposure.label] = (_COLUMN_WIDTH * plane_column, y)
    tallest = max(len(labels) for labels in rows.values())

    with _quchip_style():
        fig, axis = _resolve_single_axes(ax, figsize=(2.4 * (plane_column + 1.5), 1.2 + 2.2 * tallest))
        axis.set_axis_off()
        axis.set_aspect("equal")

        core = [label for label in columns if not network._is_reference(label)]
        xs = [positions[label][0] for label in core]
        ys = [positions[label][1] for label in core]
        _box(
            axis,
            (min(xs) + max(xs)) / 2,
            (min(ys) + max(ys)) / 2,
            max(xs) - min(xs) + 2.2,
            max(ys) - min(ys) + 2.4,
            facecolor=_CORE_COLOR,
            edgecolor="none",
            zorder=0,
        )
        axis.text(min(xs) - 1.1, max(ys) + 1.3, "Markov core", fontsize=9, color="#355c7d")

        for component in network.components:
            x, y = positions[component.label]
            reference = network._is_reference(component.label)
            fill = _REFERENCE_COLOR if reference else "white"
            _box(axis, x, y, 1.6, 1.4, facecolor=fill, edgecolor="#404040", zorder=2)
            axis.text(x, y + 0.4, component.label, ha="center", va="center", fontsize=9, weight="bold", zorder=3)
            notes = "\n".join(_annotation(component))
            axis.text(x, y + 0.08, notes, ha="center", va="top", fontsize=7, zorder=3)
            if not show_hidden:
                continue
            count = len(component._hidden_pairs)
            for stub, (_, _, hidden_label) in enumerate(component._hidden_pairs):
                stub_x = x + 1.0 * (stub - (count - 1) / 2.0)
                stub_style: dict[str, Any] = {"linestyle": "--", "color": _HIDDEN_COLOR, "linewidth": 1, "zorder": 1}
                axis.plot([stub_x, stub_x], [y - 0.7, y - 1.2], **stub_style)
                stub_name = hidden_label.split(".")[-1]
                axis.text(stub_x, y - 1.3, stub_name, ha="center", va="top", fontsize=6, color=_HIDDEN_COLOR)

        drawn: set[frozenset[tuple[str, str]]] = set()
        for input_key, output_key in network._connections.items():
            pair = frozenset((input_key, output_key))
            if pair in drawn:
                continue
            drawn.add(pair)
            reciprocal = network._connections.get(output_key) == input_key
            axis.annotate(
                "",
                xy=positions[input_key[0]],
                xytext=positions[output_key[0]],
                arrowprops={
                    "arrowstyle": "<->" if reciprocal else "->",
                    "color": "#303030",
                    "lw": 1.2,
                    "shrinkA": 30,
                    "shrinkB": 30,
                    "mutation_scale": 14,
                },
                zorder=1,
            )

        for exposure in exposures:
            x, y = plane_positions[exposure.label]
            plane_edge = (x - 0.5, y)
            source_x, source_y = positions[exposure._input_key[0]]
            sink_x, sink_y = positions[exposure._output_key[0]]
            source_edge, sink_edge = (source_x + 0.8, source_y), (sink_x + 0.8, sink_y)
            symmetric = exposure._input_key == exposure._output_key
            legs = (
                [(plane_edge, source_edge, "<->")]
                if symmetric
                else [(source_edge, plane_edge, "->"), (plane_edge, sink_edge, "->")]
            )
            for head, tail, style in legs:
                arrow: dict[str, Any] = {"arrowstyle": style, "color": _PLANE_EDGE, "lw": 1.2, "mutation_scale": 14}
                if not symmetric:
                    arrow["connectionstyle"] = "arc3,rad=0.25"
                axis.annotate("", xy=head, xytext=tail, arrowprops=arrow, zorder=1)
            _box(axis, x, y, 1.0, 0.6, facecolor=_PLANE_COLOR, edgecolor=_PLANE_EDGE, zorder=2)
            axis.text(x, y, exposure.label, ha="center", va="center", fontsize=8, zorder=3)

        axis.relim()
        axis.autoscale_view()
        axis.margins(0.15)
    return fig
