"""Chip topology (pyvis) and dressed-spectrum (matplotlib) plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from quchip.chip.chip import Chip
from quchip.utils.jax_utils import maybe_concrete_scalar
from quchip.viz._common import _basis_label, _draw_energy_ladder
from quchip.viz._style import _GRAPH_PALETTE, _cyclic_colors, _quchip_style, _resolve_single_axes

_NODE_SHAPES = {"device": "dot", "drive": "diamond", "coupling": "square"}
_NODE_SIZES = {"device": 25, "drive": 15, "coupling": 12}
_EDGE_COLORS = {"coupling": "#4d4d4d", "wiring": "#ff7f0e", "crosstalk": "#b22222"}
_EDGE_LENGTHS = {"coupling": 150, "wiring": 80, "crosstalk": 300}


def _device_node_id(label: str) -> str:
    """Namespace a device label as a topology node id (``"device:<label>"``)."""
    return f"device:{label}"


def _coupling_node_id(label: str) -> str:
    """Namespace a coupling label as a topology node id (``"coupling:<label>"``)."""
    return f"coupling:{label}"


def _drive_node_id(label: str) -> str:
    """Namespace a drive label as a topology node id (``"drive:<label>"``)."""
    return f"drive:{label}"


def _device_label(device: Any) -> str:
    """Return the multi-line node label for *device*: its label and current ``freq``."""
    freq = maybe_concrete_scalar(getattr(device, "freq", None))
    freq_text = "n/a" if freq is None else f"{freq:.3f} GHz"
    return f"{device.label}\n{freq_text}"


def _control_label(drive: Any) -> str:
    """Return the multi-line node label for *drive* (edge pumps always name their pumped coupling)."""
    class_name = type(drive).__name__
    if drive.target_kind == "edge":
        return f"{drive.label}\n({class_name})\n→ {drive.target_label}"
    if drive.label:
        return f"{drive.label}\n({class_name})"
    return f"{class_name}\n→ {drive.device_label}"


def _collect_topology(
    chip: Chip, *, exclude: set[str] | None = None
) -> tuple[dict[str, dict[str, Any]], list[tuple[str, str, dict[str, Any]]]]:
    """Collect node/edge dicts for devices, couplings, drives, and crosstalks.

    Every coupling is its own junction node splitting the device-device
    edge in two (``device -- junction -- device``), so an edge-pump
    control (:class:`~quchip.control.drive.ParametricDrive`) has a node
    to attach to — it modulates the coupling element itself, not either
    endpoint device. Node ids are namespaced by kind
    (``"device:q0"``, ``"coupling:tc"``, ``"drive:pump"``) so a device,
    coupling, and drive that happen to share a user-chosen label can
    never collide or silently overwrite one another in *nodes*.

    When couplings are excluded from the render (``"coupling" in
    exclude``), any edge-pump drives that would otherwise attach to a
    now-absent junction node are omitted too, rather than left dangling.

    Returns ``(nodes, edges)`` where *nodes* maps namespaced node id to
    attributes (insertion-ordered: devices, then couplings, then
    drives) and *edges* is a list of ``(start, end, attributes)``
    triples of namespaced node ids.
    """
    _exclude = exclude or set()
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[tuple[str, str, dict[str, Any]]] = []
    coupling_labels: set[str] = set()

    for device in chip.devices:
        nodes[_device_node_id(device.label)] = {
            "kind": "device",
            "label": _device_label(device),
            "class_name": type(device).__name__,
            "computational": device.computational,
        }

    if "coupling" not in _exclude:
        for coupling in chip.couplings:
            coupling_labels.add(coupling.label)
            strength = maybe_concrete_scalar(coupling.coupling_strength)
            strength_text = "n/a" if strength is None else f"{strength:.3g} GHz"
            junction_id = _coupling_node_id(coupling.label)
            edge_label = f"{type(coupling).__name__}\ng={strength_text}"
            nodes[junction_id] = {
                "kind": "coupling",
                "label": f"{coupling.label}\n{edge_label}",
                "class_name": type(coupling).__name__,
                "computational": False,
            }
            edges.append((
                _device_node_id(coupling.device_a_label), junction_id, {"kind": "coupling", "label": edge_label},
            ))
            edges.append((
                junction_id, _device_node_id(coupling.device_b_label), {"kind": "coupling", "label": edge_label},
            ))

    if "drive" not in _exclude and chip.control_equipment is not None:
        for drive in chip.control_equipment.lines:
            is_edge_pump = drive.target_kind == "edge"
            if is_edge_pump and drive.target_label not in coupling_labels:
                # No junction node to attach to — either couplings were
                # excluded from this render, or the pump's coupling isn't
                # on this chip. Either way, omit the dangling control.
                continue
            drive_id = _drive_node_id(drive.label)
            nodes[drive_id] = {
                "kind": "drive",
                "label": _control_label(drive),
                "class_name": type(drive).__name__,
                "computational": False,
            }
            target_id = (
                # target_label is non-None here: edge-pump implies a
                # ParametricDrive, whose target_label override returns str,
                # never None.
                _coupling_node_id(drive.target_label)  # type: ignore[arg-type]
                if is_edge_pump
                # device_label is non-None here: Chip.connect() rejects unconnected
                # drives before they reach control_equipment.lines.
                else _device_node_id(drive.device_label)  # type: ignore[arg-type]
            )
            edges.append((
                drive_id,
                target_id,
                {"kind": "wiring", "label": f"{type(drive).__name__} → {drive.target_label}"},
            ))

        if "crosstalk" not in _exclude:
            for xt in chip.crosstalks:
                # Only connect channels the drive pass actually rendered: a
                # crosstalk edge must never resurrect a drive that was
                # deliberately omitted (an edge pump whose coupling is
                # excluded from this render).
                source_id, victim_id = _drive_node_id(xt.source), _drive_node_id(xt.victim)
                if source_id not in nodes or victim_id not in nodes:
                    continue
                beta = maybe_concrete_scalar(abs(xt.beta))
                beta_text = "n/a" if beta is None else f"{beta:.3g}"
                edges.append((
                    source_id, victim_id,
                    {"kind": "crosstalk", "label": f"|xi|={beta_text}"},
                ))

    return nodes, edges


def plot_graph(
    chip: Chip,
    path: str = "chip_topology.html",
    *,
    full: bool = True,
    exclude: set[str] | None = None,
    layout: str = "force_atlas",
    height: str = "800px",
    width: str = "100%",
) -> str:
    """Write an interactive HTML chip topology and return its path.

    The rendered graph is a standalone, offline-capable HTML file: devices
    are drawn as dots and drives as diamonds, with couplings, control
    wiring, and crosstalk distinguished by edge colour and dash pattern.
    Node colours are auto-assigned per *class* (``DuffingTransmon``,
    ``Resonator``, ``ChargeDrive``, ...) so new device/drive kinds are
    visually distinct without user intervention.

    Layouts:

    - ``"force_atlas"`` (default) — ``force_atlas_2based`` physics with
      larger/heavier computational devices so the register sits centrally
      and auxiliary couplers/drives orbit.
    - ``"hierarchical"`` — pyvis ``hrepulsion`` on explicit levels
      (computational devices → other devices/couplings → drives), useful
      for chips with many auxiliary drives or cross-resonance lines.

    *exclude* takes priority over *full*; passing ``full=False`` with no
    *exclude* hides drives and crosstalk (chip-only view). Every coupling
    is rendered as its own junction node splitting the device-device edge
    (so an edge-pump control has a node to attach to); excluding
    couplings also removes any edge-pump controls that would otherwise
    dangle (see :func:`_collect_topology`).

    Parameters
    ----------
    chip : Chip
        The chip to visualise.
    path : str
        Destination ``.html`` path. Returned for convenience.
    full : bool
        If ``False``, hide drives and crosstalk unless *exclude* is given.
    exclude : set of str, optional
        Any of ``{"coupling", "drive", "crosstalk"}`` to hide entirely.
    layout : str
        ``"force_atlas"`` or ``"hierarchical"``.
    height, width : str
        CSS dimensions forwarded to ``pyvis.network.Network``.

    Returns
    -------
    str
        *path*, unchanged — returned for convenience so the call can be
        chained into whatever opens/serves the file.

    Raises
    ------
    ImportError
        ``pyvis`` is not installed.
    ValueError
        *layout* is not one of ``"force_atlas"`` or ``"hierarchical"``.
    """
    try:
        from pyvis.network import Network
    except ImportError as exc:
        raise ImportError("pyvis is required for graph visualization: pip install 'quchip[viz]'") from exc

    valid_layouts = {"force_atlas", "hierarchical"}
    if layout not in valid_layouts:
        raise ValueError(f"layout must be one of {sorted(valid_layouts)}, got {layout!r}")

    if exclude is None and not full:
        exclude = {"drive", "crosstalk"}

    nodes, edges = _collect_topology(chip, exclude=exclude)

    class_names = list(dict.fromkeys(data["class_name"] for data in nodes.values()))
    class_colors = _cyclic_colors(class_names, _GRAPH_PALETTE)

    net = Network(height=height, width=width, directed=False, notebook=False, cdn_resources="in_line")
    if layout == "force_atlas":
        net.force_atlas_2based(
            gravity=-50, central_gravity=0.005, spring_length=150,
            spring_strength=0.04, damping=0.5, overlap=1,
        )
    else:  # layout == "hierarchical" (the only other option; validated above)
        net.hrepulsion(
            node_distance=200, central_gravity=0.1, spring_length=200,
            spring_strength=0.05, damping=0.15,
        )
    net.neighborhood_highlight = True
    net.select_menu = False
    net.set_edge_smooth("continuous")
    net.toggle_hide_edges_on_drag(True)
    net.toggle_stabilization(True)
    net.set_options('{"physics": {"stabilization": {"iterations": 200}}}')

    use_levels = layout == "hierarchical"

    for node, data in nodes.items():
        kind = data["kind"]
        is_computational = data.get("computational", False)
        node_kwargs: dict[str, Any] = {
            "label": data["label"],
            "title": data["label"],
            "color": class_colors.get(data["class_name"], "#999999"),
            "shape": _NODE_SHAPES.get(kind, "dot"),
        }

        if use_levels:
            node_kwargs["size"] = _NODE_SIZES.get(kind, 20)
            if kind == "drive":
                node_kwargs["level"] = 2
            elif is_computational:
                node_kwargs["level"] = 0
            else:
                node_kwargs["level"] = 1
        elif is_computational:
            node_kwargs["size"], node_kwargs["mass"] = 30, 3
        elif kind == "device":
            node_kwargs["size"], node_kwargs["mass"] = 20, 2
        elif kind == "coupling":
            node_kwargs["size"], node_kwargs["mass"] = 10, 1
        else:
            node_kwargs["size"], node_kwargs["mass"] = 12, 1

        net.add_node(node, **node_kwargs)

    for start, end, data in edges:
        kind = data["kind"]
        edge_kwargs: dict[str, Any] = {
            "title": data["label"],
            "color": _EDGE_COLORS.get(kind, "#4d4d4d"),
            "dashes": kind == "crosstalk",
            "width": 2.5 if kind == "coupling" else 1.5,
        }
        if not use_levels:
            edge_kwargs["length"] = _EDGE_LENGTHS.get(kind, 150)
        net.add_edge(start, end, **edge_kwargs)

    net.write_html(path)

    # Center tooltip text on multi-line labels.
    out = Path(path)
    out.write_text(out.read_text().replace(
        "<head>", "<head>\n<style>.vis-tooltip { text-align: center; }</style>", 1,
    ))
    return path


def plot_energy_levels(
    chip: Chip,
    *,
    ax: Any = None,
    max_states: int | None = None,
) -> Figure:
    """Plot dressed chip eigenenergies relative to the ground state.

    Each level is rendered as a horizontal bar annotated with its dressed
    represented-basis tuple label ``|n_1 n_2 ...>`` — the per-device
    bare-basis assignment selected by the chip's dressing analysis (see
    ``chip.analysis``). The y-axis is energy in GHz; the ground state is
    shifted to zero.

    Parameters
    ----------
    chip : Chip
        The chip whose dressed spectrum should be plotted.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw onto. When ``None`` a new figure is created.
    max_states : int, optional
        How many of the lowest-lying levels to show. Defaults to
        ``min(12, total_levels)``.

    Returns
    -------
    Figure
        The figure holding the energy-ladder axes (``ax.figure`` when
        *ax* was given).

    Examples
    --------
    >>> import quchip as qc
    >>> from quchip.viz.chip import plot_energy_levels
    >>> qb0 = qc.DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3)
    >>> qb1 = qc.DuffingTransmon(freq=5.2, anharmonicity=-0.3, levels=3)
    >>> chip = qc.Chip([qb0, qb1], [qc.Capacitive(qb0, qb1, g=0.005)])
    >>> fig = plot_energy_levels(chip, max_states=6)
    """
    dressed = chip._ensure_dressed()
    levels = sorted(dressed.dressed_eigenvalues.items(), key=lambda item: item[1])
    if max_states is None:
        max_states = min(12, len(levels))
    displayed = levels[:max_states]
    ground_energy = 0.0 if not displayed else float(displayed[0][1])
    line_color = plt.get_cmap("tab10")(0)
    entries = [(float(energy) - ground_energy, _basis_label(state)) for state, energy in displayed]

    with _quchip_style():
        fig, axis = _resolve_single_axes(ax)
        _draw_energy_ladder(axis, entries, color=line_color, linewidth=2.0)
        axis.set_ylabel("Energy relative to ground (GHz)")
        axis.set_title("Dressed energy levels", fontfamily="sans-serif")
        fig.tight_layout()
        return fig
