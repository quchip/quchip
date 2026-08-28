"""Target compilation for :func:`quchip.fit_a_dress`.

The desired-chip path reads component-declared numbers without dressing the
input chip. The deprecated compatibility path remains separate so its
seed-and-target semantics are not silently reinterpreted during migration.
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
        Observable kind — one of ``"freq"``, ``"anharmonicity"``,
        ``"chi"``, ``"zz"``, ``"exchange"``, ``"g"``.
    label
        Device label, ``(label_a, label_b)`` tuple, or coupling label
        that locates this target on the chip.
    target
        Desired value in GHz.
    source
        Where the target came from: ``"component default"``, ``"explicit"``,
        or ``"legacy"`` for the deprecated compatibility contract.
    """

    kind: str
    label: Any
    target: float
    source: str = "legacy"


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

    Unlike :func:`build_target_specs`, this desired-chip path never calls a
    dressed-analysis method on ``chip``. Devices and couplings provide their
    component-owned defaults; explicit constraints extend those defaults,
    replace the same ``(kind, locator)`` entry, or remove it with ``None``.

    Parameters
    ----------
    chip
        Numerical desired-chip specification.
    constraints
        Optional ``{component_or_pair: {observable: value_or_none}}`` mapping.
        Pair locators need not correspond to a direct coupling edge.
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


def infer_coupling_mode(coupling, override: str | None = None) -> str:
    """Infer whether a coupling's strength is targeted as ``"g"``, ``"chi"``, or ``"zz"``.

    When both devices are computational, the user almost always cares
    about the static ``zz``; a qubit-resonator pair is instead best
    anchored by the dispersive ``chi``; everything else is fit through
    the bare coupling strength. An explicit ``override`` short circuits
    the heuristic — including to ``"chi"`` on a pair that would not
    otherwise infer it; :func:`build_target_specs` validates that a
    ``"chi"`` target always has exactly one computational endpoint,
    whichever path selected it.

    Parameters
    ----------
    coupling
        The coupling whose mode is being inferred.
    override
        Explicit mode (``"g"``, ``"chi"``, or ``"zz"``), or ``None`` to
        use the computational-endpoint heuristic.

    Returns
    -------
    str
        ``"g"``, ``"chi"``, or ``"zz"``.
    """
    if override is not None:
        return override
    a_comp = getattr(coupling.device_a, "computational", False)
    b_comp = getattr(coupling.device_b, "computational", False)
    if a_comp and b_comp:
        return "zz"
    if a_comp or b_comp:
        return "chi"
    return "g"


def _validate_chi_target(chip, spec: TargetSpec) -> None:
    """Ensure a ``"chi"`` :class:`TargetSpec` resolves to a coupling with exactly one computational endpoint.

    Applied to every ``"chi"`` spec regardless of origin — an auto
    device-implied target, a ``coupling_targets`` entry, or an explicit
    ``observable_targets`` entry — since ``_chi`` (the observable this
    spec anchors) is only physically meaningful between one qubit
    (computational) endpoint and one readout (non-computational)
    endpoint.
    """
    coupling = next((c for c in chip.couplings if c.label == spec.label), None)
    if coupling is None:
        raise ValueError(
            f"'chi' target label {spec.label!r} does not match any coupling on the chip; "
            "'chi' (dispersive shift) targets must be keyed by a coupling label."
        )
    a_comp = getattr(coupling.device_a, "computational", False)
    b_comp = getattr(coupling.device_b, "computational", False)
    if a_comp == b_comp:
        raise ValueError(
            f"Coupling {coupling.label!r} is targeted as 'chi', which requires exactly one "
            "computational endpoint (chi is the dispersive shift of a readout mode "
            f"conditioned on a qubit state); got device_a={coupling.device_a_label!r} "
            f"computational={a_comp}, device_b={coupling.device_b_label!r} computational={b_comp}."
        )


def _coerce_explicit_target_specs(observable_targets: dict | None) -> tuple[TargetSpec, ...]:
    """Normalize ``observable_targets`` into a tuple of :class:`TargetSpec`."""
    if not observable_targets:
        return ()

    specs: list[TargetSpec] = []
    for key, metrics in observable_targets.items():
        if not isinstance(metrics, dict):
            raise TypeError(
                "observable_targets values must be dicts mapping observable kinds to numeric targets, "
                f"got {type(metrics).__name__}"
            )
        resolved_key = tuple(resolve_label(part) for part in key) if isinstance(key, tuple) else resolve_label(key)
        for kind, target in metrics.items():
            specs.append(TargetSpec(str(kind), resolved_key, float(target)))
    return tuple(specs)


def build_target_specs(chip, coupling_targets: dict, observable_targets: dict | None = None) -> tuple[TargetSpec, ...]:
    """Merge device defaults, coupling targets, and explicit observables.

    The returned ordering is:

    1. For every device, the *dressed* 0→1 transition frequency
       (``chip.freq(device)``) is anchored. Computational devices also
       anchor on the dressed anharmonicity
       (``chip.dressed_anharmonicity(device)``). Anchoring the dressed
       observable rather than the bare device attribute makes the
       defaults model-agnostic — Duffing, charge-basis transmon, and
       fluxonium all expose a dressed 0→1 spacing, even when the bare
       parametrization is wholly different (no ``freq`` or
       ``anharmonicity`` attribute).
    2. One coupling target per entry in ``coupling_targets``, with
       mode resolved via :func:`infer_coupling_mode` (override if a
       string mode is supplied, heuristic otherwise).
    3. Every explicit entry from ``observable_targets``.

    An explicit ``(kind, label)`` in ``observable_targets`` suppresses
    the same-``(kind, label)`` default from steps 1 and 2, so the user
    never ends up with duplicate anchors for the same quantity.

    Returns
    -------
    tuple[TargetSpec, ...]
        Merged specs in the order documented above.

    Raises
    ------
    ValueError
        A coupling target mode is not one of ``"chi"``, ``"zz"``, ``"g"``;
        or any ``"chi"`` target in the merged result — from an auto
        device-implied default, a ``coupling_targets`` entry, or an
        explicit ``observable_targets`` entry — does not resolve to a
        coupling with exactly one computational endpoint (see
        :func:`_validate_chi_target`).
    """
    explicit_specs = _coerce_explicit_target_specs(observable_targets)
    explicit_keys = {(spec.kind, spec.label) for spec in explicit_specs}
    normalized_coupling_targets = {resolve_label(key): value for key, value in coupling_targets.items()}
    specs: list[TargetSpec] = []
    for device in chip.devices:
        spec = TargetSpec("freq", device.label, float(chip.freq(device)))
        if (spec.kind, spec.label) not in explicit_keys:
            specs.append(spec)
        if device.computational and device.levels >= 3:
            spec = TargetSpec(
                "anharmonicity",
                device.label,
                float(chip.dressed_anharmonicity(device)),
            )
            if (spec.kind, spec.label) not in explicit_keys:
                specs.append(spec)
    for coupling in chip.couplings:
        if coupling.label not in normalized_coupling_targets:
            continue
        mode = infer_coupling_mode(coupling, normalized_coupling_targets[coupling.label])
        if mode not in ("chi", "zz", "g"):
            raise ValueError(f"Unsupported coupling target mode {mode!r} for {coupling.label!r}")
        spec = TargetSpec(mode, coupling.label, float(coupling.coupling_strength))
        if (spec.kind, spec.label) not in explicit_keys:
            specs.append(spec)
    specs.extend(explicit_specs)
    for spec in specs:
        if spec.kind == "chi":
            _validate_chi_target(chip, spec)
    return tuple(specs)
