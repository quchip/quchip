"""Shared Matplotlib styling and axes-resolution helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

_RC_OVERRIDES: dict[str, object] = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "figure.figsize": (6.0, 4.0),
    "axes.grid": True,
    "grid.color": "#808080",
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "mathtext.fontset": "dejavuserif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
}


@contextmanager
def _quchip_style() -> Iterator[None]:
    """Apply shared rc-param overrides; revert on exit."""
    with mpl.rc_context(rc=cast(Any, _RC_OVERRIDES)):
        yield


# Graph node palette: tab10's first eight colours with green and orange
# deliberately swapped (positions 1 and 2), so the second node class reads
# green rather than orange. Hand-tuned for the topology graph; kept explicit
# rather than derived from "tab10" so that ordering choice is not lost.
_GRAPH_PALETTE = [
    "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def _cyclic_colors(keys: Iterable[Any], palette: str | Sequence[Any]) -> dict[Any, Any]:
    """Map each key to a colour by cycling a palette in iteration order.

    *palette* is either a matplotlib colormap name (e.g. ``"tab10"``) or an
    explicit sequence of colours. Keys wrap with ``idx % len`` so an arbitrary
    number of keys always resolves to a colour. Callers that need deduplicated
    keys (e.g. ``dict.fromkeys``) should dedup before calling.
    """
    if isinstance(palette, str):
        cmap = plt.get_cmap(palette)
        colors: Sequence[Any] = [cmap(i) for i in range(cmap.N)]
    else:
        colors = palette
    return {key: colors[idx % len(colors)] for idx, key in enumerate(keys)}


def _resolve_single_axes(
    ax: Axes | None,
    *,
    figsize: tuple[float, float] = (6.0, 4.0),
) -> tuple[Figure, Axes]:
    """Return ``(fig, ax)``, creating a new figure when *ax* is ``None``."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    return cast(Figure, ax.figure), ax


def _resolve_dual_axes(
    ax: Iterable[Axes] | None,
    *,
    figsize: tuple[float, float] = (10.0, 4.0),
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Return ``(fig, (ax1, ax2))``, creating a 1x2 figure when *ax* is ``None``."""
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        return fig, (axes[0], axes[1])

    try:
        axes = tuple(ax)
    except TypeError as exc:
        raise ValueError("ax must be None or an iterable of exactly two Axes") from exc
    if len(axes) != 2:
        raise ValueError("ax must be None or an iterable of exactly two Axes")
    return axes[0].figure, (axes[0], axes[1])
