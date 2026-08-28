"""Fit bare chip parameters to a numerical dressed specification.

Devices and couplings declare how their input numbers map to dressed targets.
The desired chip is read structurally, not diagonalized to discover those
targets. Candidate chips are then evaluated in a bounded nonlinear
least-squares solve. The deprecated explicit-target path remains available
through 0.2.x for migration.

References
----------

Dispersive regime (``chi``, ``zz``, and dressed frequencies):

- Koch et al., *Phys. Rev. A* **76**, 042319 (2007),
  "Charge-insensitive qubit design derived from the Cooper pair box"
  — the DuffingTransmon approximation and its dispersive shifts.
- Gambetta et al., *Phys. Rev. A* **74**, 042318 (2006),
  "Qubit-photon interactions in a cavity" — the
  ``chi = g^2 / Delta`` qubit-resonator dispersive shift at leading
  order, the dispersive-regime intuition behind the ``chi`` seed search
  (:func:`_estimate_bare_g`). The ``zz`` seed makes no leading-order
  claim of its own: the seed search only requires the target to be
  *bracketed* by the observable at the endpoints of
  ``seed_strength_bounds`` (checked, not assumed) — it does not require
  the observable to be monotone in between; :func:`scipy.optimize.brentq`
  finds a consistent root regardless of the observable's direction
  (increasing or decreasing) within the bracket.

JAX traceability
----------------

Every bare parameter here (``device.freq``, ``device.anharmonicity``, a
coupling's ``coupling_strength``) is a sweepable, differentiable
quantity. A chip using a JAX-native backend supplies a traced
dressed-observable residual and exact Jacobian; SciPy consumes their concrete
values only at the bounded trust-region boundary. The optimizer itself is not
JAX-traceable, while the output :class:`~quchip.chip.chip.Chip` remains fully
traceable for every downstream operation.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq, least_squares

from quchip.analysis.effective_hamiltonian import effective_hamiltonian_between_states
from quchip.backend import _backend_context
from quchip.chip import Chip
from quchip.inverse_design.observables import (
    TargetSpec,
    build_dressed_target_specs,
    build_target_specs,
)
from quchip.inverse_design.subsystems import (
    build_local_subsystem,
    choose_evaluator,
    device_labels_for_local_eval,
)
from quchip.inverse_design.types import FitADressResult, FitParameterReport, ObservableReport
from quchip.utils.labeling import resolve_label


def _pair_labels(label: Any, kind: str) -> tuple[str, str]:
    """Validate a 2-tuple label for pair observables (``zz``, ``exchange``)."""
    if not isinstance(label, tuple) or len(label) != 2:
        raise ValueError(f"{kind} targets require a 2-tuple of device labels, got {label!r}")
    return str(label[0]), str(label[1])


def _static_exchange_rate(chip: Chip, label: Any) -> Any:
    """Off-diagonal element of the single-excitation effective Hamiltonian.

    Returns the static exchange rate between two devices in GHz, via the
    public :func:`~quchip.analysis.effective_hamiltonian.effective_hamiltonian_between_states`
    seam — no direct access to chip/analysis internals here.
    """
    a, b = _pair_labels(label, "exchange")
    n = len(chip.devices)
    state_a = [0] * n
    state_b = [0] * n
    state_a[chip.device_index(a)] = 1
    state_b[chip.device_index(b)] = 1
    h_eff = effective_hamiltonian_between_states(chip, tuple(state_a), tuple(state_b))
    return jnp.real(h_eff[0, 1])


def _chi(chip: Chip, coupling) -> float:
    """Dispersive shift ``chi = (omega_r|1> - omega_r|0>) / 2`` for one coupling.

    The single definition of the dispersive shift used everywhere in the
    fit. Devices are resolved from ``chip`` by the coupling's labels, so
    the shift is evaluated on whatever (sub)chip the caller passes — the
    isolated two-device sub-chip during the seed root solve, or the
    one-hop local neighborhood (or the full chip) during residual
    evaluation. The computational device is the qubit ``q``; the other
    is the readout mode ``r``.
    """
    a_dev = chip[coupling.device_a_label]
    b_dev = chip[coupling.device_b_label]
    q = a_dev if a_dev.computational else b_dev
    r = b_dev if not b_dev.computational else a_dev
    return (chip.freq(r, when={q: 1}) - chip.freq(r, when={q: 0})) / 2.0


def _estimate_bare_g(
    chip: Chip,
    coupling,
    spec: TargetSpec,
    seed_strength_bounds: tuple[float, float] = (1e-6, 0.25),
) -> float:
    """Root-solve a 2-device sub-chip to find a bare coupling strength that matches the target ``chi`` or ``zz``.

    A good seed matters because the outer least-squares problem is
    non-convex in the coupling strength near the dispersive regime. The
    only requirement here is that the target observable be *bracketed*
    by the endpoints of ``seed_strength_bounds`` — checked below, not
    merely assumed. The observable need not be monotone in between:
    :func:`scipy.optimize.brentq` finds a strength consistent with the
    target regardless of whether the observable increases or decreases
    across the bracket. The search runs on a 2-device sub-chip — no
    neighbors, no crosstalk — for a strength that reproduces the target
    observable, handing that value to the full fit.

    The sub-chip is built via the coupling's own structural copy/rebind
    path (:meth:`~quchip.chip.coupling_base.BaseCoupling.copy`), which
    preserves constructor-only subclass state without coupling-type reconstruction, so this works for
    any coupling, not only ``g``-attribute ones — and
    :meth:`~quchip.chip.coupling_base.BaseCoupling.set_coupling_strength`
    writes each trial magnitude. It also carries over the parent chip's
    backend, isolated from chip-context-dependent conveniences such as
    ``device.drive_freq`` / ``device.dressed_freq``.

    Parameters
    ----------
    seed_strength_bounds
        ``(lo, hi)`` magnitude bounds for the root solve.

    Raises
    ------
    ValueError
        The target observable is not bracketed by the endpoint values —
        seeding never returns a saturated endpoint silently.
    """
    signed_cross_kerr = spec.kind == "cross_kerr"
    target_val = float(spec.target) if signed_cross_kerr else abs(float(spec.target))
    if target_val == 0.0:
        return 0.0
    dev_a = coupling.device_a.copy()
    dev_b = coupling.device_b.copy()
    device_map = {dev_a.label: dev_a, dev_b.label: dev_b}
    sub_coupling = coupling.copy(device_map)
    sub = Chip(
        [dev_a, dev_b],
        [sub_coupling],
        frame=chip.frame,
        approximation=chip.approximation,
        basis=chip.basis,
        backend=chip.backend,
    )

    def obs_at_strength(strength: float) -> float:
        # Only reached for chi/zz seeds (see _pack_initial_params), so the
        # final branch is the zz case — there is no g fallback.
        sub_coupling.set_coupling_strength(strength)
        if spec.kind == "chi":
            return abs(_chi(sub, sub_coupling))
        if spec.kind == "cross_kerr":
            return float(sub.static_zz(sub_coupling.device_a, sub_coupling.device_b))
        return abs(float(sub.static_zz(sub_coupling.device_a, sub_coupling.device_b)))

    lo, hi = seed_strength_bounds
    obs_lo, obs_hi = obs_at_strength(lo), obs_at_strength(hi)
    if not min(obs_lo, obs_hi) <= target_val <= max(obs_lo, obs_hi):
        raise ValueError(
            f"Cannot seed a bare coupling strength for {coupling.label!r}'s target "
            f"{spec.kind}={target_val!r}: not bracketed by the observable at "
            f"seed_strength_bounds={seed_strength_bounds!r} "
            f"(observable(lo)={obs_lo!r}, observable(hi)={obs_hi!r})."
        )
    return brentq(lambda strength: obs_at_strength(strength) - target_val, lo, hi)


def _selected_tunable_names(device, device_selection: Mapping[str, tuple[str, ...]] | None) -> tuple[str, ...]:
    """Declared tunable-parameter names free for *device* (all of them, when *device_selection* is ``None``)."""
    if device_selection is None:
        return tuple(device.tunable_params())
    return tuple(device_selection.get(device.label, ()))


def _coupling_is_selected(coupling, coupling_selection: Mapping[str, tuple[str, ...]] | None) -> bool:
    """Whether *coupling*'s scalar strength is free (always, when *coupling_selection* is ``None``)."""
    if coupling_selection is None:
        return True
    return coupling.coupling_strength_name in coupling_selection.get(coupling.label, ())


def _selected_parameter_names(
    chip: Chip,
    device_selection: Mapping[str, tuple[str, ...]] | None,
    coupling_selection: Mapping[str, tuple[str, ...]] | None,
) -> list[str]:
    """Names of every free parameter, in chip order and each component's declared parameter order.

    ``device_selection``/``coupling_selection`` of ``None`` means every
    declared device tunable and every coupling strength is free (the
    ``fit_parameters=None`` default). A mapping restricts each component to
    its listed names; a component absent from the mapping is fully frozen.
    This is a structural computation only — no seed root-solving — so a
    caller that just needs the free-parameter *count* (the identifiability
    check in :func:`fit_a_dress`) does not pay for :func:`_estimate_bare_g`.
    """
    names: list[str] = []
    for device in chip.devices:
        selected = set(_selected_tunable_names(device, device_selection))
        for param_name in device.tunable_params():
            if param_name in selected:
                names.append(f"{device.label}.{param_name}")
    for coupling in chip.couplings:
        if _coupling_is_selected(coupling, coupling_selection):
            names.append(f"{coupling.label}.{coupling.coupling_strength_name}")
    return names


def _resolve_fit_parameters(
    chip: Chip,
    fit_parameters: Mapping,
    *,
    argument_name: str = "fit_parameters",
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """Resolve a ``fit_parameters`` mapping into per-device and per-coupling free-parameter allowlists.

    ``fit_parameters`` is the *complete* free-parameter selection: a
    component (device or coupling, given as the object or its label) absent
    from the mapping is fully frozen, and an empty name-collection value
    explicitly freezes a listed component. Device parameter names validate
    against :meth:`~quchip.devices.base.BaseDevice.tunable_params` (which
    walks the device's declared ``tunable_param_names`` — the generic seam
    a user-authored :class:`~quchip.devices.base.BaseDevice` subclass
    already gets for free); coupling names validate against exactly
    ``(coupling.coupling_strength_name,)``.

    Returns
    -------
    tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]
        ``(device_selection, coupling_selection)``, each ``{label: names}``
        for the components actually listed in ``fit_parameters``.

    Raises
    ------
    ValueError
        A key does not match any device or coupling label on ``chip``; a
        name is not among the resolved component's declared tunables; two
        keys resolve to the same label; or a value is a bare string rather
        than a collection of names.
    """
    devices_by_label = {device.label: device for device in chip.devices}
    couplings_by_label = {coupling.label: coupling for coupling in chip.couplings}
    known_labels = sorted(set(devices_by_label) | set(couplings_by_label))

    device_selection: dict[str, tuple[str, ...]] = {}
    coupling_selection: dict[str, tuple[str, ...]] = {}
    seen_labels: set[str] = set()

    for key, value in fit_parameters.items():
        label = resolve_label(key)
        if label in seen_labels:
            raise ValueError(f"{argument_name} has duplicate entries resolving to component {label!r}.")
        seen_labels.add(label)

        if isinstance(value, str):
            raise ValueError(
                f"{argument_name}[{label!r}] must be a collection of parameter names (e.g. ('{value}',)), "
                f"got the bare string {value!r}."
            )
        try:
            names = tuple(str(name) for name in value)
        except TypeError as exc:
            raise ValueError(
                f"{argument_name}[{label!r}] must be an iterable of parameter names, got {value!r}."
            ) from exc

        if label in devices_by_label:
            device = devices_by_label[label]
            available = tuple(device.tunable_params())
            unknown = [name for name in names if name not in available]
            if unknown:
                raise ValueError(
                    f"{argument_name}[{label!r}] names {unknown} are not tunable parameters of "
                    f"{type(device).__name__} {label!r}. Available: {list(available)}."
                )
            device_selection[label] = names
        elif label in couplings_by_label:
            coupling = couplings_by_label[label]
            available = (coupling.coupling_strength_name,)
            unknown = [name for name in names if name not in available]
            if unknown:
                raise ValueError(
                    f"{argument_name}[{label!r}] names {unknown} are not the declared coupling-strength "
                    f"parameter of {type(coupling).__name__} {label!r}. Available: {list(available)}."
                )
            coupling_selection[label] = names
        else:
            raise ValueError(
                f"{argument_name} key {label!r} does not match any device or coupling on the chip. "
                f"Available labels: {known_labels}."
            )

    return device_selection, coupling_selection


def _pack_initial_params(
    chip: Chip,
    target_specs: tuple[TargetSpec, ...],
    seed_strength_bounds: tuple[float, float] = (1e-6, 0.25),
    device_selection: Mapping[str, tuple[str, ...]] | None = None,
    coupling_selection: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Flatten the chip's free bare parameters into a (names, values) pair.

    Every parameter that a device declares via
    :meth:`~quchip.devices.base.BaseDevice.tunable_params` is packed by
    default (``device_selection is None``) — the optimizer is free to move
    any of them, regardless of which targets are active. This keeps
    :func:`fit_a_dress` agnostic to the specific device model: a
    :class:`DuffingTransmon` exposes ``freq``/``anharmonicity``, a
    :class:`Fluxonium` exposes ``E_C``/``E_J``/``E_L``/``phi_ext``, and so
    on. ``device_selection``/``coupling_selection`` (see
    :func:`_resolve_fit_parameters`) restrict packing to a per-component
    allowlist; a component absent from a non-``None`` selection is skipped
    entirely — its bare value stays whatever the seed chip declares. Each
    packed coupling is named ``f"{coupling.label}.{coupling.coupling_strength_name}"``
    — ``g`` for :class:`~quchip.chip.couplings.Capacitive`, ``g_0`` for
    :class:`~quchip.chip.couplings.TunableCapacitive`, ``chi`` for
    :class:`~quchip.chip.couplings.CrossKerr`, or whatever a custom
    coupling declares. A packed coupling whose target is ``chi`` or ``zz``
    gets a physically motivated seed via :func:`_estimate_bare_g`; all
    others start at their current
    :attr:`~quchip.chip.coupling_base.BaseCoupling.coupling_strength`.

    INVARIANT: ``names`` is always a subset of the complete key set
    :func:`_rebuild_candidate` can dispatch — a name absent from the
    packed vector simply leaves that parameter at its cloned value.

    Parameters
    ----------
    seed_strength_bounds
        Forwarded to :func:`_estimate_bare_g` for ``chi``/``zz`` seeds.
    """
    names = _selected_parameter_names(chip, device_selection, coupling_selection)
    name_set = set(names)
    values: list[float] = []
    for device in chip.devices:
        for param_name, param_value in device.tunable_params().items():
            if f"{device.label}.{param_name}" in name_set:
                values.append(float(param_value))
    coupling_specs = {
        spec.label: spec
        for spec in target_specs
        if isinstance(spec.label, str) and spec.kind in ("chi", "zz", "cross_kerr")
    }
    for coupling in chip.couplings:
        key = f"{coupling.label}.{coupling.coupling_strength_name}"
        if key not in name_set:
            continue
        spec = coupling_specs.get(coupling.label)
        if spec and spec.kind in ("chi", "zz", "cross_kerr"):
            if spec.kind == "cross_kerr" and not coupling.reduces_to_crosskerr:
                values.append(float(spec.target))
            else:
                values.append(_estimate_bare_g(chip, coupling, spec, seed_strength_bounds))
        else:
            values.append(float(coupling.coupling_strength))
    return names, np.asarray(values, dtype=float)


def _rebuild_candidate(chip: Chip, names: list[str], values: Any) -> Chip:
    """Rebind the selected fit parameters on an isolated chip copy."""
    return chip.with_params(dict(zip(names, values, strict=True)))


def _working_chip(candidate: Chip, label: Any, evaluator: str) -> Chip:
    """Pick the full chip or a one-hop local neighborhood for this target."""
    if evaluator != "local":
        return candidate
    return build_local_subsystem(candidate, device_labels_for_local_eval(candidate, label))


def _coupling_by_label(chip: Chip, label: Any):
    """Return the coupling on *chip* with the given label."""
    return next(c for c in chip.couplings if c.label == label)


def _evaluate_spec(candidate: Chip, spec: TargetSpec, evaluator: str) -> Any:
    """Compute the observable this ``spec`` anchors, on the appropriate working chip."""
    match spec.kind:
        case "freq":
            return _working_chip(candidate, spec.label, evaluator).freq(spec.label)
        case "anharmonicity":
            return _working_chip(candidate, spec.label, evaluator).dressed_anharmonicity(spec.label)
        case "chi":
            coupling = _coupling_by_label(candidate, spec.label)
            working = _working_chip(candidate, (coupling.device_a_label, coupling.device_b_label), evaluator)
            return _chi(working, coupling)
        case "zz":
            if isinstance(spec.label, tuple):
                working = _working_chip(candidate, spec.label, evaluator)
                a, b = _pair_labels(spec.label, "zz")
                return working.static_zz(a, b)
            coupling = _coupling_by_label(candidate, spec.label)
            working = _working_chip(candidate, (coupling.device_a_label, coupling.device_b_label), evaluator)
            return working.static_zz(coupling.device_a_label, coupling.device_b_label)
        case "cross_kerr":
            if isinstance(spec.label, tuple):
                working = _working_chip(candidate, spec.label, evaluator)
                a, b = _pair_labels(spec.label, "cross_kerr")
                return working.static_zz(a, b)
            coupling = _coupling_by_label(candidate, spec.label)
            working = _working_chip(
                candidate,
                (coupling.device_a_label, coupling.device_b_label),
                evaluator,
            )
            return working.static_zz(coupling.device_a_label, coupling.device_b_label)
        case "exchange":
            return _static_exchange_rate(_working_chip(candidate, spec.label, evaluator), spec.label)
        case "exchange_rate":
            if not isinstance(spec.label, tuple):
                coupling = _coupling_by_label(candidate, spec.label)
                label = (coupling.device_a_label, coupling.device_b_label)
            else:
                label = spec.label
            return _static_exchange_rate(_working_chip(candidate, label, evaluator), label)
        case "g":
            return _coupling_by_label(candidate, spec.label).coupling_strength
        case "coupling_strength":
            return _coupling_by_label(candidate, spec.label).coupling_strength
        case _:
            raise ValueError(f"Unknown spec kind {spec.kind!r}")


def _auto_bounds(chip: Chip, names: list[str], x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-parameter box bounds that keep the TRF solver physical.

    A coupling's scalar strength (``g``, ``g_0``, ``chi``, …) is symmetric
    around zero — the *sign* of a capacitive-type coupling carries physical
    meaning and must not be frozen. Every device-side parameter delegates
    to :meth:`~quchip.devices.base.BaseDevice.tunable_param_bounds` so
    each device declares the valid range for its own bare parameters
    (``freq``, ``anharmonicity``, ``E_C``/``E_J``/``E_L``/``phi_ext``,
    ``n_g``, …) without any global registry.
    """
    devices_by_label = {device.label: device for device in chip.devices}
    couplings_by_label = {coupling.label: coupling for coupling in chip.couplings}
    lower: list[float] = []
    upper: list[float] = []
    for name, value in zip(names, x0, strict=True):
        owner_label, _, param_name = name.rpartition(".")
        device = devices_by_label.get(owner_label)
        if device is not None:
            lo, hi = device.tunable_param_bounds(param_name, float(value))
            lower.append(float(lo))
            upper.append(float(hi))
            continue
        coupling = couplings_by_label.get(owner_label)
        if coupling is not None and param_name == coupling.coupling_strength_name:
            mag = max(0.25, 2.0 * abs(value)) if value != 0.0 else 0.25
            lower.append(-mag)
            upper.append(mag)
            continue
        raise ValueError(f"Unsupported latent parameter {name!r}")
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def _parameter_seed_metadata(
    chip: Chip,
    names: list[str],
    target_specs: tuple[TargetSpec, ...],
    *,
    legacy_mode: bool,
    user_starts: set[str],
) -> dict[str, tuple[str, str | None]]:
    """Describe where each optimizer start and coupling-sign branch came from."""
    couplings = {coupling.label: coupling for coupling in chip.couplings}
    coupling_specs = {
        spec.label: spec
        for spec in target_specs
        if isinstance(spec.label, str) and spec.kind in {"chi", "zz", "cross_kerr"}
    }
    metadata: dict[str, tuple[str, str | None]] = {}
    for name in names:
        owner, _, _ = name.rpartition(".")
        coupling = couplings.get(owner)
        if name in user_starts:
            metadata[name] = ("user start", "user supplied" if coupling is not None else None)
            continue
        if coupling is None:
            metadata[name] = (
                "seed chip" if legacy_mode else "component declaration",
                None,
            )
            continue
        spec = coupling_specs.get(owner)
        if spec is not None and spec.kind == "cross_kerr" and not coupling.reduces_to_crosskerr:
            metadata[name] = ("target value", "target sign")
        elif spec is not None:
            metadata[name] = ("isolated-pair root solve", "positive convention")
        else:
            metadata[name] = (
                "seed chip" if legacy_mode else "component declaration",
                "declared sign",
            )
    return metadata


def _identifiability_receipt(
    jacobian: Any,
    names: list[str],
    parameter_scale: np.ndarray,
) -> dict[str, Any]:
    """Analyze the final residual Jacobian in the solver's scaled coordinates."""
    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    scaled = matrix * np.asarray(parameter_scale, dtype=float)[None, :]
    _, singular_values, right_vectors = np.linalg.svd(scaled, full_matrices=True)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = float(np.finfo(float).eps * max(scaled.shape, default=0) * largest)
    rank = int(np.count_nonzero(singular_values > tolerance))
    n_parameters = len(names)
    rank_deficient = rank < n_parameters
    if rank_deficient:
        condition_number = float("inf")
    elif singular_values.size:
        condition_number = float(singular_values[0] / singular_values[n_parameters - 1])
    else:
        condition_number = float("inf")

    weak_directions: list[dict[str, Any]] = []
    for index in range(rank, n_parameters):
        vector = right_vectors[index]
        magnitude = float(np.max(np.abs(vector)))
        relative = vector / magnitude if magnitude else vector
        weak_directions.append(
            {
                "singular_value": (float(singular_values[index]) if index < singular_values.size else 0.0),
                "relative_weights": {
                    name: float(weight)
                    for name, weight in zip(names, relative, strict=True)
                    if abs(float(weight)) >= 1e-8
                },
            }
        )

    return {
        "jacobian_rank": rank,
        "jacobian_shape": tuple(int(size) for size in scaled.shape),
        "jacobian_rank_tolerance": tolerance,
        "jacobian_singular_values": tuple(float(value) for value in singular_values),
        "jacobian_condition_number": condition_number,
        "rank_deficient": rank_deficient,
        "weak_parameter_directions": tuple(weak_directions),
        "jacobian_parameter_scale": dict(zip(names, (float(value) for value in parameter_scale), strict=True)),
    }


def _coerce_mapping(value: Mapping | None, name: str) -> dict:
    """Return *value* as a plain dict, treating ``None`` as empty and rejecting non-``Mapping`` input."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"{name} must be None or a Mapping, got {type(value).__name__}")


def _jax_residual_functions(
    chip: Chip,
    names: list[str],
    target_specs: tuple[TargetSpec, ...],
    evaluator: str,
    targets: np.ndarray,
    scales: np.ndarray,
):
    """Return JAX residual functions for a JAX-backed chip, else ``None``."""
    backend = chip.backend
    if backend.array_module is not jnp:
        return None
    targets_jax = jnp.asarray(targets)
    scales_jax = jnp.asarray(scales)

    def residual(values):
        with _backend_context(backend):
            candidate = _rebuild_candidate(chip, names, values)
            observed = jnp.stack([jnp.asarray(_evaluate_spec(candidate, spec, evaluator)) for spec in target_specs])
        return (observed - targets_jax) / scales_jax

    return jax.jit(residual), jax.jit(jax.jacrev(residual))


def fit_a_dress(
    chip: Chip,
    *,
    constraints: Mapping | None = None,
    vary: Mapping | None = None,
    start: Mapping | None = None,
    coupling_targets: Mapping | None = None,
    observable_targets: Mapping | None = None,
    fit_parameters: Mapping | None = None,
    max_hilbert_dim: int = 10_000,
    seed_strength_bounds: tuple[float, float] = (1e-6, 0.25),
    max_nfev: int = 1000,
) -> FitADressResult:
    """Find bare parameters for a numerically specified dressed chip.

    ``fit_a_dress(desired)`` reads component-owned target declarations without
    evaluating ``desired``. Common spectral devices interpret their declared
    frequencies and anharmonicities as dressed targets. Capacitive and
    cross-Kerr couplings interpret their declared scalar as the full
    ``cross_kerr = E11 - E10 - E01 + E00`` target.  The returned
    :attr:`~quchip.inverse_design.types.FitADressResult.chip` is a fitted clone;
    ``desired`` is never mutated.

    ``constraints`` adds or replaces numerical observables, ``vary`` replaces
    the component-owned free-parameter selection, and ``start`` replaces
    selected starting values. The deprecated ``coupling_targets``,
    ``observable_targets``, and ``fit_parameters`` keywords keep their
    seed-chip semantics through 0.2.x. They cannot be mixed with the
    desired-chip keywords and will be removed in 0.3.0.

    Parameters
    ----------
    chip
        Desired dressed-chip specification. Component declarations supply
        numerical targets; no dressed analysis is run on this object.
    constraints
        Additional ``{component_or_pair: {observable: value_or_none}}``
        constraints. Supported canonical observables are ``"freq"``,
        ``"anharmonicity"``, ``"cross_kerr"``, ``"exchange_rate"``, and
        ``"coupling_strength"``.  ``"zz"``/``"static_zz"``, ``"exchange"``,
        and ``"g"`` are accepted aliases.  An explicit value replaces the
        same component default; ``None`` removes it. Device pairs need not be
        direct coupling edges.
    vary
        Complete desired-chip allowlist of bare parameters that may move:
        ``{component_or_label: name_collection}``. When omitted, each
        component's conservative inverse-design policy is used. Components
        absent from an explicit mapping are frozen.
    start
        Optional ``{"<component>.<parameter>": value}`` replacements for the
        selected optimizer starting values. A key not selected by ``vary``
        raises instead of being silently ignored.
    coupling_targets
        Deprecated since 0.2.1; use ``constraints``. Maps a coupling (or its
        label) to a target mode:
        ``"chi"``, ``"zz"``, or ``"g"``. For listed couplings, the
        coupling's current strength is interpreted as the target value
        for that mode. With ``fit_parameters=None``, couplings not listed
        here are still free — they are optimized, just without a dedicated
        anchor; a ``fit_parameters`` mapping can freeze them regardless (a
        coupling target does not itself make a coupling free). A ``"chi"``
        target requires the coupling to have exactly one computational
        endpoint; both-computational or neither-computational raises
        :class:`ValueError` at construction.
    observable_targets
        Deprecated since 0.2.1; put numerical dressed values on ``chip`` and
        add extra observables with ``constraints``. Keys are devices/labels or
        ``(device_a, device_b)`` tuples; values are ``{kind: value}``
        dicts. Supported kinds: ``"freq"``, ``"anharmonicity"``
        (device), ``"exchange"``, ``"zz"`` (pair). Device-level
        targets override the auto-targeted defaults for the same
        ``(kind, label)``.
    fit_parameters
        Deprecated since 0.2.1; use ``vary``. ``None`` in the compatibility
        path selects every declared device tunable
        (:meth:`~quchip.devices.base.BaseDevice.tunable_params`) and every
        coupling's scalar strength is free.
        A mapping is the *complete* free-parameter allowlist:
        ``{component_or_label: name_collection}``, where a device's
        ``name_collection`` is a subset of its declared tunable names and
        a coupling's is a subset of
        ``(coupling.coupling_strength_name,)``. A component (device or
        coupling, given as the object or its label) absent from the
        mapping is fully frozen — it does **not** default to free. An
        empty ``name_collection`` explicitly freezes a listed component.
        A bare string value (e.g. ``"E_J"`` instead of ``("E_J",)``) is
        rejected, since a string is itself a collection of characters.
        Selected parameters are packed in chip order and each component's
        own declared parameter order, not mapping or tuple order.
        :attr:`~quchip.inverse_design.types.FitADressResult.initial_params`
        / :attr:`~quchip.inverse_design.types.FitADressResult.final_params`
        contain only the selected (free) parameters.
    max_hilbert_dim
        Above this total Hilbert-space size the fit switches from
        dressing the whole chip to dressing one-hop subsystems per
        target (see :mod:`quchip.inverse_design.subsystems`).
    seed_strength_bounds
        ``(lo, hi)`` magnitude bounds for the bare-coupling-strength
        seed root solve (:func:`_estimate_bare_g`) used for
        ``cross_kerr``/``chi``/``zz`` coupling targets. The target observable must be bracketed by the
        values at these two endpoints, or seeding raises
        :class:`ValueError` rather than silently returning a saturated
        endpoint.
    max_nfev
        Maximum number of residual evaluations for the SciPy
        Trust-Region Reflective solver.

    Returns
    -------
    FitADressResult
        Fitted chip clone, loss, residual history, per-target
        :class:`ObservableReport` tuples, packed parameters, and
        solver metadata.

    Raises
    ------
    ValueError
        Desired-chip and deprecated compatibility keyword families are mixed;
        a ``"chi"`` coupling target does not have exactly one
        computational endpoint, or a ``chi``/``zz`` seed's target
        observable is not bracketed within ``seed_strength_bounds``; an
        automatic desired-chip plan is underdetermined by count or rank; a
        ``fit_parameters`` key does not resolve to a device or coupling
        label on ``chip``, names a parameter the resolved component does
        not declare, resolves the same label twice, or is a bare string
        rather than a name collection; or ``fit_parameters`` selects zero
        free parameters overall.

    Warns
    -----
    UserWarning
        A fit using explicit ``vary`` or deprecated compatibility keywords is
        underdetermined by target count or final scaled-Jacobian rank. Explicit
        plans are returned with diagnostics because the caller has taken
        ownership of the ambiguity; automatic desired-chip plans raise instead.

    Notes
    -----
    Residuals are normalized by ``max(|target|, 1e-9)`` so every anchor
    contributes on equal *relative*-error footing. A coupling's scalar
    strength bounds are symmetric around zero — the sign of a
    capacitive-type coupling is physical and must not be constrained.
    The solver's convergence tolerances (``ftol``/``xtol``/``gtol`` =
    ``1e-11``) and its ``x_scale`` floor (``1e-3``, applied per parameter
    as ``max(abs(x0), 1e-3)``) are fixed fitter policy, not exposed as
    options.

    **Identifiability.** The free-parameter-vs-residual count is necessary but
    not sufficient. The fitter therefore computes the final SVD rank and
    condition number from normalized residuals in the same scaled parameter
    coordinates used by the solver. A custom
    :class:`~quchip.declarative.models.DeviceModel`
    whose ``tunable_param_names`` is discovered (the derived default, not an
    explicit declaration) is not automatically fit-ready: an unbounded
    parameter still needs a :meth:`~quchip.devices.base.BaseDevice.tunable_param_bounds`
    rule before the optimizer can search it.

    **JAX traceability.** When the chip uses a JAX-native backend, the
    complete parameter-to-residual map and its exact Jacobian are
    JAX-traceable; SciPy receives their concrete values for bounded
    trust-region control. The optimizer itself is not differentiated.
    Other backends retain SciPy's numerical Jacobian. The returned chip
    remains fully traceable and differentiable in either case.
    """
    legacy_mode = any(value is not None for value in (coupling_targets, observable_targets, fit_parameters))
    if legacy_mode and any(value is not None for value in (constraints, vary, start)):
        raise ValueError(
            "Use either the desired-chip API (constraints/vary/start) or the deprecated compatibility API "
            "(coupling_targets/observable_targets/fit_parameters), not both."
        )
    if legacy_mode:
        warnings.warn(
            "coupling_targets=, observable_targets=, and fit_parameters= are deprecated since "
            "quchip 0.2.1 and will be removed in 0.3.0. Build a numerical desired chip and use "
            "constraints=, vary=, and start= instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    if legacy_mode:
        ct = _coerce_mapping(coupling_targets, "coupling_targets")
        ot_mapping = (
            _coerce_mapping(observable_targets, "observable_targets") if observable_targets is not None else None
        )
        target_specs = build_target_specs(chip, ct, ot_mapping)
    else:
        constraint_mapping = _coerce_mapping(constraints, "constraints") if constraints is not None else None
        target_specs = build_dressed_target_specs(chip, constraint_mapping)

    manual_selection = fit_parameters if legacy_mode else vary
    if manual_selection is None and legacy_mode:
        device_selection: dict[str, tuple[str, ...]] | None = None
        coupling_selection: dict[str, tuple[str, ...]] | None = None
    elif manual_selection is None:
        targeted_labels = {spec.label for spec in target_specs if isinstance(spec.label, str)}
        device_selection = {
            device.label: device.default_fit_parameters()
            for device in chip.devices
            if device.label in targeted_labels and device.default_fit_parameters()
        }
        coupling_selection = {
            coupling.label: (coupling.coupling_strength_name,)
            for coupling in chip.couplings
            if coupling.label in targeted_labels
            and any(spec.label == coupling.label and spec.kind != "coupling_strength" for spec in target_specs)
        }
    else:
        selection_name = "fit_parameters" if legacy_mode else "vary"
        fp_mapping = _coerce_mapping(manual_selection, selection_name)
        device_selection, coupling_selection = _resolve_fit_parameters(
            chip,
            fp_mapping,
            argument_name=selection_name,
        )

    n_free_parameters = len(_selected_parameter_names(chip, device_selection, coupling_selection))
    n_target_residuals = len(target_specs)
    automatic_desired_plan = not legacy_mode and vary is None
    selection_name = "fit_parameters" if legacy_mode else "vary"
    if n_free_parameters == 0:
        raise ValueError(
            f"{selection_name} selects zero free parameters; fit_a_dress has nothing to optimize. "
            "List at least one device parameter or coupling strength with a non-empty selection."
        )
    underdetermined_by_count = n_free_parameters > n_target_residuals
    if underdetermined_by_count:
        message = (
            f"fit_a_dress has {n_free_parameters} free parameters but only {n_target_residuals} target "
            f"residuals; the fit is underdetermined by count. Select fewer parameters with {selection_name}. "
            "Target count alone does not guarantee identifiability."
        )
        if automatic_desired_plan:
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=2)

    names, x0 = _pack_initial_params(chip, target_specs, seed_strength_bounds, device_selection, coupling_selection)
    user_starts: set[str] = set()
    if start is not None:
        start_mapping = _coerce_mapping(start, "start")
        user_starts = set(start_mapping)
        unknown = sorted(set(start_mapping) - set(names))
        if unknown:
            raise ValueError(
                f"start contains parameters that are not selected by vary: {unknown}. Selected parameters: {names}."
            )
        indices = {name: index for index, name in enumerate(names)}
        x0 = x0.copy()
        for name, value in start_mapping.items():
            x0[indices[name]] = float(value)
    targets_arr = np.asarray([spec.target for spec in target_specs], dtype=float)
    scales = np.maximum(np.abs(targets_arr), 1e-9)
    lower, upper = _auto_bounds(chip, names, x0)
    parameter_scale = np.maximum(np.abs(x0), 1e-3)
    seed_metadata = _parameter_seed_metadata(
        chip,
        names,
        target_specs,
        legacy_mode=legacy_mode,
        user_starts=user_starts,
    )
    evaluator = choose_evaluator(chip, max_hilbert_dim)

    def concrete_residuals(x: np.ndarray) -> np.ndarray:
        candidate = _rebuild_candidate(chip, names, x)
        values = np.asarray([_evaluate_spec(candidate, spec, evaluator) for spec in target_specs], dtype=float)
        return (values - targets_arr) / scales

    jacobian_mode = "finite-difference"
    residuals = concrete_residuals
    jacobian: Any = "2-point"
    try:
        jax_functions = _jax_residual_functions(
            chip,
            names,
            target_specs,
            evaluator,
            targets_arr,
            scales,
        )
    except ImportError:
        jax_functions = None

    if jax_functions is not None:
        residual_jax, jacobian_jax = jax_functions

        def residuals(x: np.ndarray) -> np.ndarray:
            return np.asarray(residual_jax(jnp.asarray(x)), dtype=float)

        def jacobian(x: np.ndarray) -> np.ndarray:
            return np.asarray(jacobian_jax(jnp.asarray(x)), dtype=float)

        jacobian_mode = "jax"

    untracked_residuals = residuals
    r0 = untracked_residuals(x0)
    history: list[float] = [float(r0 @ r0)]
    last_history_point = np.asarray(x0, dtype=float).copy()

    def tracked_residuals(x: np.ndarray) -> np.ndarray:
        """Evaluate residuals and retain one loss per distinct parameter vector."""
        nonlocal last_history_point
        values = untracked_residuals(x)
        point = np.asarray(x, dtype=float)
        if not np.array_equal(point, last_history_point):
            history.append(float(values @ values))
            last_history_point = point.copy()
        return values

    result = least_squares(
        tracked_residuals,
        x0=x0,
        jac=jacobian,
        bounds=(lower, upper),
        method="trf",
        x_scale=parameter_scale,
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
        max_nfev=max_nfev,
    )
    final_loss = float(2.0 * result.cost)
    if not np.array_equal(last_history_point, np.asarray(result.x, dtype=float)):
        history.append(final_loss)
    else:
        history[-1] = final_loss

    fitted_chip = _rebuild_candidate(chip, names, result.x)
    identifiability = _identifiability_receipt(result.jac, names, parameter_scale)
    if identifiability["rank_deficient"]:
        weak_names = sorted(
            {
                name
                for direction in identifiability["weak_parameter_directions"]
                for name in direction["relative_weights"]
            }
        )
        message = (
            f"fit_a_dress final Jacobian rank {identifiability['jacobian_rank']} for "
            f"{n_free_parameters} free parameters; the fitted bare chip is not locally identifiable. "
            f"Weak parameter directions involve {weak_names}."
        )
        if automatic_desired_plan:
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=2)

    # Reconstruct observed values at start/end by inverting the residual scaling.
    initial_vec = r0 * scales + targets_arr
    final_vec = untracked_residuals(result.x) * scales + targets_arr
    initial_reports = tuple(
        ObservableReport(
            spec.kind,
            spec.label,
            spec.target,
            float(initial),
            float(initial),
            evaluator,
            spec.source,
        )
        for spec, initial in zip(target_specs, initial_vec, strict=True)
    )
    final_reports = tuple(
        ObservableReport(
            spec.kind,
            spec.label,
            spec.target,
            float(initial),
            float(final),
            evaluator,
            spec.source,
        )
        for spec, initial, final in zip(target_specs, initial_vec, final_vec, strict=True)
    )
    parameter_reports = tuple(
        FitParameterReport(
            name=name,
            initial=float(initial),
            final=float(final),
            lower_bound=float(lo),
            upper_bound=float(hi),
            seed_source=seed_metadata[name][0],
            sign_choice=seed_metadata[name][1],
        )
        for name, initial, final, lo, hi in zip(
            names,
            x0,
            result.x,
            lower,
            upper,
            strict=True,
        )
    )

    return FitADressResult(
        chip=fitted_chip,
        loss=final_loss,
        history=np.asarray(history, dtype=float),
        initial_targets=initial_reports,
        final_targets=final_reports,
        initial_params=dict(zip(names, x0, strict=True)),
        final_params=dict(zip(names, result.x, strict=True)),
        parameter_reports=parameter_reports,
        solver_info={
            "method": "trf",
            "status": int(result.status),
            "message": result.message,
            "nfev": int(result.nfev),
            "jacobian": jacobian_mode,
            "n_free_parameters": n_free_parameters,
            "n_target_residuals": n_target_residuals,
            "underdetermined_by_count": underdetermined_by_count,
            "history_axis": "distinct residual evaluation",
            "n_recorded_evaluations": len(history),
            "input_contract": "legacy-seed" if legacy_mode else "desired-chip",
            **identifiability,
        },
    )
