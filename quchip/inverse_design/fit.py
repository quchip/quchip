"""Dressed-observable fitting of bare chip parameters.

Calibration data arrives in the dressed frame: a spectroscopy peak is
a dressed eigenvalue, a Ramsey shift is a dispersive ``chi``, a
parked-qubit detuning is a static ``zz``. The chip model, on the
other hand, is parameterized by *bare* quantities — device ``freq``
and ``anharmonicity``, a coupling's scalar strength (``g``, ``g_0``,
``chi``, depending on the coupling type — see
:attr:`~quchip.chip.coupling_base.BaseCoupling.coupling_strength`).
:func:`fit_a_dress` closes that gap: given targets on dressed
observables, it finds bare parameters whose dressed spectrum
reproduces them.

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
from quchip.inverse_design.observables import TargetSpec, build_target_specs
from quchip.inverse_design.subsystems import (
    build_local_subsystem,
    choose_evaluator,
    device_labels_for_local_eval,
)
from quchip.inverse_design.types import FitADressResult, ObservableReport
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
    preserves the coupling's RWA override and any constructor-only
    subclass state — no coupling-type reconstruction, so this works for
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
    target_val = abs(float(spec.target))
    if target_val == 0.0:
        return 0.01
    dev_a = coupling.device_a.copy()
    dev_b = coupling.device_b.copy()
    device_map = {dev_a.label: dev_a, dev_b.label: dev_b}
    sub_coupling = coupling.copy(device_map)
    sub = Chip(
        [dev_a, dev_b],
        [sub_coupling],
        frame=chip.frame,
        rwa=chip.rwa,
        backend=chip._backend,
    )

    def obs_at_strength(strength: float) -> float:
        # Only reached for chi/zz seeds (see _pack_initial_params), so the
        # final branch is the zz case — there is no g fallback.
        sub_coupling.set_coupling_strength(strength)
        if spec.kind == "chi":
            return abs(_chi(sub, sub_coupling))
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


def _selected_tunable_names(
    device, device_selection: Mapping[str, tuple[str, ...]] | None
) -> tuple[str, ...]:
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
    chip: Chip, fit_parameters: Mapping
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
            raise ValueError(f"fit_parameters has duplicate entries resolving to component {label!r}.")
        seen_labels.add(label)

        if isinstance(value, str):
            raise ValueError(
                f"fit_parameters[{label!r}] must be a collection of parameter names (e.g. ('{value}',)), "
                f"got the bare string {value!r}."
            )
        try:
            names = tuple(str(name) for name in value)
        except TypeError as exc:
            raise ValueError(
                f"fit_parameters[{label!r}] must be an iterable of parameter names, got {value!r}."
            ) from exc

        if label in devices_by_label:
            device = devices_by_label[label]
            available = tuple(device.tunable_params())
            unknown = [name for name in names if name not in available]
            if unknown:
                raise ValueError(
                    f"fit_parameters[{label!r}] names {unknown} are not tunable parameters of "
                    f"{type(device).__name__} {label!r}. Available: {list(available)}."
                )
            device_selection[label] = names
        elif label in couplings_by_label:
            coupling = couplings_by_label[label]
            available = (coupling.coupling_strength_name,)
            unknown = [name for name in names if name not in available]
            if unknown:
                raise ValueError(
                    f"fit_parameters[{label!r}] names {unknown} are not the declared coupling-strength "
                    f"parameter of {type(coupling).__name__} {label!r}. Available: {list(available)}."
                )
            coupling_selection[label] = names
        else:
            raise ValueError(
                f"fit_parameters key {label!r} does not match any device or coupling on the chip. "
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
        spec.label: spec for spec in target_specs if isinstance(spec.label, str) and spec.kind in ("chi", "zz")
    }
    for coupling in chip.couplings:
        key = f"{coupling.label}.{coupling.coupling_strength_name}"
        if key not in name_set:
            continue
        spec = coupling_specs.get(coupling.label)
        if spec and spec.kind in ("chi", "zz"):
            values.append(_estimate_bare_g(chip, coupling, spec, seed_strength_bounds))
        else:
            values.append(float(coupling.coupling_strength))
    return names, np.asarray(values, dtype=float)


def _rebuild_candidate(chip: Chip, names: list[str], values: Any) -> Chip:
    """Clone the seed and overwrite bare parameters from the packed vector.

    ``names`` may be a strict subset of the chip's full parameter set (see
    :func:`_pack_initial_params` and ``fit_parameters`` selection) — a
    device parameter or coupling strength absent from ``names`` is simply
    left at its cloned (seed) value, i.e. frozen. Each device parameter
    present is dispatched through
    :meth:`~quchip.devices.base.BaseDevice.set_tunable_param`, which is the
    single seam every concrete device customizes — no
    ``freq``/``anharmonicity`` hardcoding here. Each coupling parameter
    present is dispatched through
    :meth:`~quchip.chip.coupling_base.BaseCoupling.set_coupling_strength`,
    the matching seam for a coupling's own scalar strength — no ``.g``
    hardcoding here either.
    """
    candidate = chip.clone()
    value_map = dict(zip(names, values, strict=True))
    for device in candidate.devices:
        for param_name in device.tunable_params():
            key = f"{device.label}.{param_name}"
            if key in value_map:
                device.set_tunable_param(param_name, value_map[key])
    for coupling in candidate.couplings:
        key = f"{coupling.label}.{coupling.coupling_strength_name}"
        if key in value_map:
            coupling.set_coupling_strength(value_map[key])
    # Sanity: any leftover keys would silently no-op above. Catch wiring drift.
    expected = {
        f"{device.label}.{name}"
        for device in candidate.devices
        for name in device.tunable_params()
    } | {f"{coupling.label}.{coupling.coupling_strength_name}" for coupling in candidate.couplings}
    unexpected = set(value_map) - expected
    if unexpected:
        raise RuntimeError(
            f"_rebuild_candidate received parameters with no destination: {sorted(unexpected)}"
        )
    return candidate


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
        case "exchange":
            return _static_exchange_rate(_working_chip(candidate, spec.label, evaluator), spec.label)
        case "g":
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
            observed = jnp.stack([
                jnp.asarray(_evaluate_spec(candidate, spec, evaluator))
                for spec in target_specs
            ])
        return (observed - targets_jax) / scales_jax

    return jax.jit(residual), jax.jit(jax.jacrev(residual))


def fit_a_dress(
    chip: Chip,
    *,
    coupling_targets: Mapping | None = None,
    observable_targets: Mapping | None = None,
    fit_parameters: Mapping | None = None,
    max_hilbert_dim: int = 10_000,
    seed_strength_bounds: tuple[float, float] = (1e-6, 0.25),
    max_nfev: int = 1000,
) -> FitADressResult:
    """Fit bare chip parameters so dressed observables match targets.

    With ``fit_parameters=None`` (the default), every device ``freq`` /
    ``anharmonicity`` and every coupling's scalar strength
    (:attr:`~quchip.chip.coupling_base.BaseCoupling.coupling_strength`) is a
    free variable in a bounded non-linear least-squares problem (scipy
    Trust-Region Reflective). A ``fit_parameters`` mapping instead is the
    *complete* free-parameter allowlist — see ``fit_parameters`` below — so
    any device or coupling it does not list is frozen instead of free. The
    fitted chip is returned as a clone — the input chip is never mutated.

    Parameters
    ----------
    chip
        Seed chip. Seeds only set the optimizer's starting point.
    coupling_targets
        Mapping from coupling (or its label) to a target mode:
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
        Mapping of target observables. Keys are devices/labels or
        ``(device_a, device_b)`` tuples; values are ``{kind: value}``
        dicts. Supported kinds: ``"freq"``, ``"anharmonicity"``
        (device), ``"exchange"``, ``"zz"`` (pair). Device-level
        targets override the auto-targeted defaults for the same
        ``(kind, label)``.
    fit_parameters
        ``None`` (default): every declared device tunable
        (:meth:`~quchip.devices.base.BaseDevice.tunable_params`) and every
        coupling's scalar strength is free — the pre-existing behavior.
        A mapping is instead the *complete* free-parameter allowlist:
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
        seed root solve (:func:`_estimate_bare_g`) used for ``chi``/``zz``
        coupling targets. The target observable must be bracketed by the
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
        A ``"chi"`` coupling target does not have exactly one
        computational endpoint, or a ``chi``/``zz`` seed's target
        observable is not bracketed within ``seed_strength_bounds``; a
        ``fit_parameters`` key does not resolve to a device or coupling
        label on ``chip``, names a parameter the resolved component does
        not declare, resolves the same label twice, or is a bare string
        rather than a name collection; or ``fit_parameters`` selects zero
        free parameters overall.

    Warns
    -----
    UserWarning
        The number of free parameters exceeds the number of target
        residuals (underdetermined by count). This is a necessary, not
        sufficient, identifiability condition: it does not analyze the
        Jacobian's rank, so a count-sufficient fit can still be
        practically underdetermined.

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

    **Two identifiability hazards the count check above does not catch.**
    The free-parameter-vs-residual count is necessary but not sufficient: it
    cannot detect a flat Jacobian direction — a free parameter no target
    observable actually responds to — which stays underdetermined regardless
    of the count. And a custom :class:`~quchip.declarative.models.DeviceModel`
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
    ct = _coerce_mapping(coupling_targets, "coupling_targets")
    ot_mapping = _coerce_mapping(observable_targets, "observable_targets") if observable_targets is not None else None

    target_specs = build_target_specs(chip, ct, ot_mapping)

    if fit_parameters is None:
        device_selection: dict[str, tuple[str, ...]] | None = None
        coupling_selection: dict[str, tuple[str, ...]] | None = None
    else:
        fp_mapping = _coerce_mapping(fit_parameters, "fit_parameters")
        device_selection, coupling_selection = _resolve_fit_parameters(chip, fp_mapping)

    n_free_parameters = len(_selected_parameter_names(chip, device_selection, coupling_selection))
    n_target_residuals = len(target_specs)
    if n_free_parameters == 0:
        raise ValueError(
            "fit_parameters selects zero free parameters; fit_a_dress has nothing to optimize. "
            "List at least one device parameter or coupling strength with a non-empty selection."
        )
    underdetermined_by_count = n_free_parameters > n_target_residuals
    if underdetermined_by_count:
        warnings.warn(
            f"fit_a_dress has {n_free_parameters} free parameters but only {n_target_residuals} target "
            "residuals; the fit is underdetermined by count. Select fewer parameters with fit_parameters. "
            "Target count alone does not guarantee identifiability.",
            UserWarning,
            stacklevel=2,
        )

    names, x0 = _pack_initial_params(chip, target_specs, seed_strength_bounds, device_selection, coupling_selection)
    targets_arr = np.asarray([spec.target for spec in target_specs], dtype=float)
    scales = np.maximum(np.abs(targets_arr), 1e-9)
    lower, upper = _auto_bounds(chip, names, x0)
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

    r0 = residuals(x0)
    history: list[float] = [float(r0 @ r0)]

    result = least_squares(
        residuals,
        x0=x0,
        jac=jacobian,
        bounds=(lower, upper),
        method="trf",
        x_scale=np.maximum(np.abs(x0), 1e-3),
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
        max_nfev=max_nfev,
    )
    history.append(float(2.0 * result.cost))

    fitted_chip = _rebuild_candidate(chip, names, result.x)

    # Reconstruct observed values at start/end by inverting the residual scaling.
    initial_vec = r0 * scales + targets_arr
    final_vec = residuals(result.x) * scales + targets_arr
    initial_reports = tuple(
        ObservableReport(spec.kind, spec.label, spec.target, float(initial), float(initial), evaluator)
        for spec, initial in zip(target_specs, initial_vec, strict=True)
    )
    final_reports = tuple(
        ObservableReport(spec.kind, spec.label, spec.target, float(initial), float(final), evaluator)
        for spec, initial, final in zip(target_specs, initial_vec, final_vec, strict=True)
    )

    return FitADressResult(
        chip=fitted_chip,
        loss=float(2.0 * result.cost),
        history=np.asarray(history, dtype=float),
        initial_targets=initial_reports,
        final_targets=final_reports,
        initial_params=dict(zip(names, x0, strict=True)),
        final_params=dict(zip(names, result.x, strict=True)),
        solver_info={
            "method": "trf",
            "status": int(result.status),
            "message": result.message,
            "nfev": int(result.nfev),
            "jacobian": jacobian_mode,
            "n_free_parameters": n_free_parameters,
            "n_target_residuals": n_target_residuals,
            "underdetermined_by_count": underdetermined_by_count,
        },
    )
