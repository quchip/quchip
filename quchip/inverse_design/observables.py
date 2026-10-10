"""Target compilation for :func:`quchip.fit_a_dress`.

Targets come from component declarations and explicit constraints, without
diagonalizing the desired model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quchip.utils.labeling import resolve_label


_DRESSED_KIND_ALIASES = {
    "g": "coupling_strength",
    "coupling_strength": "coupling_strength",
    "chi": "cross_kerr",
    "cross_kerr": "cross_kerr",
    "static_zz": "cross_kerr",
    "zz": "cross_kerr",
    "exchange": "exchange_rate",
    "exchange_rate": "exchange_rate",
    "freq": "freq",
    "anharmonicity": "anharmonicity",
}


@dataclass(frozen=True)
class TargetSpec:
    """A single optimization target.

    Attributes
    ----------
    kind
        Observable kind. One of ``"freq"``, ``"anharmonicity"``,
        ``"cross_kerr"``, ``"exchange_rate"``, ``"coupling_strength"``.
    label
        Device label, ``(label_a, label_b)`` tuple, or coupling label
        that locates this target on the chip.
    target
        Desired value in GHz.
    source
        Origin: ``"component default"`` or ``"explicit"``.
    """

    kind: str
    label: Any
    target: float
    source: str = "explicit"


def _normalize_dressed_kind(kind: object) -> str:
    """Return the canonical desired-chip observable name for *kind*."""
    name = str(kind)
    try:
        return _DRESSED_KIND_ALIASES[name]
    except KeyError as exc:
        available = sorted(set(_DRESSED_KIND_ALIASES.values()))
        raise ValueError(f"Unknown dressed constraint {name!r}. Available canonical names: {available}.") from exc


def _resolve_target_locator(locator: Any) -> Any:
    """Normalize a component or pair locator without evaluating a chip."""
    if isinstance(locator, tuple):
        if len(locator) != 2:
            raise ValueError(f"Pair constraints require exactly two device objects or labels, got {locator!r}.")
        return tuple(resolve_label(part) for part in locator)
    return resolve_label(locator)


def build_dressed_target_specs(
    chip,
    constraints: dict | None = None,
) -> tuple[TargetSpec, ...]:
    """Compile a desired chip's declared numbers into dressed constraints.

    Compilation never calls a dressed-analysis method on ``chip``. Devices and couplings supply
    their component-owned defaults. Explicit constraints extend those defaults, replace the same
    ``(kind, locator)`` entry, or remove it with ``None``.

    Parameters
    ----------
    chip
        Numerical desired-chip specification.
    constraints
        Optional ``{component_or_pair: {observable: value_or_none}}`` mapping.
        Pair locators do not have to agree with a direct coupling edge.
    """
    keyed: dict[tuple[str, Any], TargetSpec] = {}

    for device in chip.devices:
        for kind, value in device.default_dressed_targets().items():
            canonical = _normalize_dressed_kind(kind)
            spec = TargetSpec(canonical, device.label, float(value), "component default")
            keyed[(spec.kind, spec.label)] = spec

    for coupling in chip.couplings:
        kind, value = coupling.default_dressed_target()
        canonical = _normalize_dressed_kind(kind)
        spec = TargetSpec(canonical, coupling.label, float(value), "component default")
        keyed[(spec.kind, spec.label)] = spec

    if constraints is None:
        return tuple(keyed.values())
    if not isinstance(constraints, dict):
        raise TypeError(f"constraints must be None or a dict, got {type(constraints).__name__}")

    for raw_locator, metrics in constraints.items():
        locator = _resolve_target_locator(raw_locator)
        if not isinstance(metrics, dict):
            raise TypeError(
                "constraints values must be dicts mapping observable names to "
                f"numeric values or None, got {type(metrics).__name__} for {locator!r}."
            )
        for raw_kind, value in metrics.items():
            kind = _normalize_dressed_kind(raw_kind)
            key = (kind, locator)
            if value is None:
                keyed.pop(key, None)
            else:
                keyed[key] = TargetSpec(kind, locator, float(value), "explicit")

    return tuple(keyed.values())
