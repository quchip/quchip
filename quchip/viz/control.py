"""Pulse-sequence timeline plot."""

from __future__ import annotations

from typing import Any

from matplotlib.figure import Figure

from quchip.viz._style import _cyclic_colors, _quchip_style, _resolve_single_axes


def _lane_label(lane: tuple[str, str]) -> str:
    """Format a ``(target_label, drive_label)`` lane key as ``"target:drive"``."""
    target_label, line = lane
    return f"{target_label}:{line}"


def _lane_color_map(
    lanes: list[tuple[str, str]],
    *,
    color_by: str,
) -> dict[tuple[str, str], Any]:
    """Assign tab10 colors grouped by ``"device"`` or ``"line"``."""
    if color_by not in {"device", "line"}:
        raise ValueError("color_by must be 'device' or 'line'")

    group_idx = 0 if color_by == "device" else 1
    palette_keys = list(dict.fromkeys(lane[group_idx] for lane in lanes))
    colors = _cyclic_colors(palette_keys, "tab10")
    return {lane: colors[lane[group_idx]] for lane in lanes}


def plot_sequence(
    sequence: Any,
    *,
    ax: Any = None,
    lane_order: list[tuple[str, str]] | None = None,
    color_by: str = "device",
) -> Figure:
    """Plot a :class:`~quchip.control.sequence.QuantumSequence` as a lane chart.

    Each row ("lane") is a ``(target_label, drive_label)`` channel; each
    bar is one scheduled envelope, drawn from its ``start_time`` to
    ``start_time + envelope.duration`` on the x-axis (ns), annotated with
    the envelope class name and carrier frequency in GHz. Idle channels
    that were created (``channel_cursors`` advanced) but never scheduled
    show up as empty lanes.

    Parameters
    ----------
    sequence : QuantumSequence
        The schedule to visualise.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw onto. When ``None`` a new figure is created.
    lane_order : list of (target_label, drive_label), optional
        Explicit ordering (top-to-bottom). Defaults to sorted lane keys.
    color_by : {"device", "line"}
        Group bars by target device or by drive line for colour mapping.

    Returns
    -------
    Figure
        The figure holding the lane-chart axes (``ax.figure`` when *ax*
        was given).

    Raises
    ------
    ValueError
        *color_by* is not ``"device"`` or ``"line"``.
    """
    scheduled_ops = sequence.scheduled_ops
    lane_set = {(op.target_label, op.drive_label) for op in scheduled_ops}
    lane_set.update(lane for lane, cursor in sequence.channel_cursors.items() if cursor > 0.0)
    lanes = lane_order or sorted(lane_set)
    lane_colors = _lane_color_map(lanes, color_by=color_by)
    lane_to_y = {lane: idx for idx, lane in enumerate(lanes)}

    grouped: dict[tuple[str, str], list[Any]] = {lane: [] for lane in lanes}
    for op in scheduled_ops:
        lane = (op.target_label, op.drive_label)
        if lane in grouped:
            grouped[lane].append(op)

    with _quchip_style():
        fig, axis = _resolve_single_axes(ax, figsize=(8.0, 4.5))
        for lane in lanes:
            lane_ops = sorted(grouped[lane], key=lambda op: op.start_time)
            y = lane_to_y[lane]
            for idx, op in enumerate(lane_ops):
                bar = axis.barh(
                    y, op.envelope.duration, left=op.start_time, height=0.7,
                    color=lane_colors[lane], edgecolor="black", linewidth=0.8, alpha=0.85,
                )[0]
                bar.set_label(f"{_lane_label(lane)}:{idx}")
                freq_label = "" if op.freq is None else f" @ {op.freq:.3f} GHz"
                axis.text(
                    op.start_time + op.envelope.duration / 2.0, y,
                    f"{type(op.envelope).__name__}{freq_label}",
                    ha="center", va="center", fontsize=8,
                )

        axis.set_yticks(range(len(lanes)), [_lane_label(lane) for lane in lanes])
        axis.set_xlabel("Time (ns)")
        axis.set_ylabel("Channel")
        axis.set_title("Pulse sequence", fontfamily="sans-serif")
        axis.set_ylim(-0.75, len(lanes) - 0.25 if lanes else 0.75)
        if sequence.total_duration > 0.0:
            axis.set_xlim(0.0, sequence.total_duration * 1.05)
        fig.tight_layout()
        return fig
