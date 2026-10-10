"""Bare states, dressed eigenstates, and normalized superpositions for chips.

String shorthand such as ``"eg1"`` uses the device order and level symbols that
:func:`set_state_order` sets. Dressed-state selection uses the assigned
eigenvector column and stays JAX-traceable on a fixed assignment.
"""

from __future__ import annotations

import warnings
from math import prod
from typing import TYPE_CHECKING, Any, Mapping

from quchip.backend.protocol import State
from quchip.utils.constants import TWO_PI
from quchip.utils.jax_utils import array_namespace, maybe_concrete_scalar
from quchip.utils.labeling import merge_labeled_values, resolve_label

if TYPE_CHECKING:
    from quchip.chip.chip import Chip
    from quchip.devices.base import BaseDevice


# Default letter → energy-level map for ``chip.bare_state("eg1")`` shorthand.
# Covers the ``g``/``e``/``f``/``h`` bra-ket convention common in
# superconducting-qubit papers. Users may pass their own via
# :func:`set_state_order` (``levels=...``).
_DEFAULT_LEVEL_SYMBOLS: dict[str, int] = {"g": 0, "e": 1, "f": 2, "h": 3}


def set_state_order(
    chip: "Chip",
    *devices: "str | BaseDevice",
    levels: Mapping[str, int] | None = None,
) -> None:
    """Declare the device order that parses string-state shorthands.

    After this call, :meth:`Chip.bare_state`, :meth:`Chip.state`, and
    :meth:`Chip.superposition` accept single-string specifications. In these
    strings, each character is one level per device, in *devices* order. Level
    symbols default to ``g=0, e=1, f=2, h=3``. The digits ``0..9`` are always
    accepted as energy-level indices.

    Every chip device must be named exactly once.

    Examples
    --------
    >>> from quchip import DuffingTransmon, Resonator, Chip
    >>> qb = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="qb")
    >>> tc = DuffingTransmon(freq=5.5, anharmonicity=-0.20, levels=3, label="tc")
    >>> cr = Resonator(freq=7.0, levels=4, label="cr")
    >>> chip = Chip([qb, tc, cr])
    >>> chip.set_state_order(qb, tc, cr)
    >>> _ = chip.bare_state("eg1")  # {qb: 1, tc: 0, cr: 1}
    """
    order = tuple(resolve_label(d) for d in devices)
    available = list(chip._device_map.keys())
    unknown = [lbl for lbl in order if lbl not in chip._device_map]
    if unknown:
        raise ValueError(f"Unknown device(s) in state order: {unknown}. Available: {available}")
    if len(set(order)) != len(order):
        raise ValueError(f"Duplicate device in state order: {order}")
    missing = sorted(set(available) - set(order))
    if missing:
        raise ValueError(
            f"set_state_order must name every device; missing {missing}. "
            f"Available: {available}"
        )
    chip._state_order = order
    if levels is not None:
        chip._level_symbols = dict(levels)


def copy_state_configuration(source: "Chip", target: "Chip") -> None:
    """Keep the relative string-state order and symbols on surviving devices."""
    if source._state_order is not None:
        target.set_state_order(
            *(label for label in source._state_order if label in target.device_map),
            levels=source._level_symbols,
        )


def parse_state_string(chip: "Chip", s: str) -> dict[str, int]:
    """Parse ``chip.bare_state("eg1")`` style strings into ``{label: index}``."""
    if chip._state_order is None:
        raise ValueError(
            "String-state shorthand requires chip.set_state_order(...) to "
            "declare device order first."
        )
    if len(s) != len(chip._state_order):
        raise ValueError(
            f"State string {s!r} has {len(s)} chars but {len(chip._state_order)} "
            f"devices are declared in state order {chip._state_order}."
        )
    out: dict[str, int] = {}
    for label, ch in zip(chip._state_order, s):
        if ch.isdigit():
            out[label] = int(ch)
        elif ch in chip._level_symbols:
            out[label] = chip._level_symbols[ch]
        else:
            known = sorted(chip._level_symbols)
            raise ValueError(
                f"Unknown level symbol {ch!r} in state {s!r}. "
                f"Known symbols: {known}; digits 0-9 always accepted."
            )
    return out


def normalize_device_state_mapping(
    chip: "Chip",
    device_states: Mapping[str | "BaseDevice", Any] | str | None,
    keyword_states: dict[str, Any],
) -> dict[str, Any]:
    """Normalize a mapping or string state specification to device-label keys.

    Parse strings with the chip's declared state order. Reject unsupported
    input types and duplicate device specifications across the mapping and
    keywords.
    """
    mapping: Mapping[Any, Any] | None
    mapping = parse_state_string(chip, device_states) if isinstance(device_states, str) else device_states

    if mapping is not None and not isinstance(mapping, Mapping):
        raise TypeError(
            "device_states must be a mapping keyed by device label or "
            f"BaseDevice, got {type(mapping).__name__}"
        )

    return merge_labeled_values(mapping, keyword_states)


def superposition(
    chip: "Chip",
    *components: Mapping[str | "BaseDevice", int] | str | tuple[Any, Any],
) -> State:
    """Normalized bare-basis superposition of tensor-product states.

    Each component is a bare-state spec or an ``(amplitude, spec)`` tuple for
    weighted mixing. A bare-state spec is a dict keyed by device or label, or a
    string after you call :func:`set_state_order`. The weights are uniform by
    default, and the result is normalized to unit norm.

    Unlike :meth:`~quchip.Chip.state`, this function stays in the bare product
    basis and does no dressed diagonalization, so the probe basis is explicit.

    Examples
    --------
    >>> import numpy as np
    >>> from quchip import DuffingTransmon, Resonator, Chip
    >>> qb = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="qb")
    >>> cr = Resonator(freq=7.0, levels=4, label="cr")
    >>> chip = Chip([qb, cr])
    >>> _ = chip.superposition({qb: 0}, {qb: 1})  # equal |00> + |10>
    >>> _ = chip.superposition(                   # weighted mix
    ...     (np.sqrt(0.3), {qb: 1, cr: 0}),
    ...     (np.sqrt(0.7), {qb: 1, cr: 1}),
    ... )
    """
    if not components:
        raise ValueError("superposition requires at least one component")

    amps: list[Any] = []
    kets: list[State] = []
    bases = chip.resolve().bases
    for component in components:
        if (
            isinstance(component, tuple)
            and len(component) == 2
            and not isinstance(component[0], (dict, str, Mapping))
        ):
            amp, spec = component
        else:
            amp, spec = 1.0, component
        resolved = normalize_device_state_mapping(chip, spec, {})
        kets.append(_bare_state_from_bases(chip, resolved, bases))
        amps.append(amp)

    psi = amps[0] * kets[0]
    for amp, ket in zip(amps[1:], kets[1:]):
        psi = psi + amp * ket
    backend = chip.backend
    norm = backend.norm(psi)
    # Backend.norm may return a traced 0-d array (dynamiqs). The zero-norm
    # short-circuit runs directly when concretely readable. Otherwise the
    # divisor itself must be guarded (mirrors Bath._bose's safe-denominator
    # pattern): xp.where evaluates both branches, so dividing by the
    # (possibly zero) traced norm directly would still produce 0/0 -> NaN
    # in the unselected branch. Replacing the zero norm with 1.0 before the
    # division makes an all-zero traced amplitude set return the
    # already-zero (unnormalized) state instead of NaN.
    concrete_norm = maybe_concrete_scalar(norm)
    if concrete_norm is not None:
        return psi if concrete_norm <= 0 else psi / norm
    xp = backend.array_module
    safe_norm = xp.where(norm == 0, 1.0, norm)
    return psi / safe_norm


def bare_state(
    chip: "Chip",
    device_states: Mapping[str | "BaseDevice", int | State] | str | None = None,
    /,
    **device_state_kwargs: int | State,
) -> State:
    """Product state from per-device energy levels or authored local kets.

    You can specify each device as an energy-level index (``int``) or as a ket vector
    in that device's authored local space. Unmentioned devices default to the ground
    state (level 0). Unlike :meth:`~quchip.Chip.state`, this function does **not**
    diagonalize the coupled system.

    Accepts a string shorthand (for example ``"eg1"``) after you call
    :func:`set_state_order`.
    """
    resolved = normalize_device_state_mapping(chip, device_states, device_state_kwargs)
    return _bare_state_from_bases(chip, resolved, chip.resolve().bases)


def _bare_state_from_bases(
    chip: "Chip",
    resolved: Mapping[str, Any],
    bases: Mapping[str, Any],
) -> State:
    """Build a solver ket from local level indices or authored local arrays."""
    backend = chip.backend
    available = list(chip._device_map.keys())
    prepared = dict(resolved)

    for label in resolved:
        if label not in chip._device_map:
            raise ValueError(f"Unknown device label '{label}'. Available labels: {available}")

    for label, val in resolved.items():
        if isinstance(val, bool):
            raise ValueError(f"Level index for '{label}' must be an integer, got {type(val).__name__}: {val!r}")
        basis = bases[label]
        if isinstance(val, int):
            if val < 0:
                raise ValueError(f"Level index for '{label}' must be >= 0, got {val}")
            if val >= basis.resolved_dim:
                raise ValueError(
                    f"Level index {val} for '{label}' exceeds the resolved "
                    f"dimension {basis.resolved_dim}."
                )
        else:
            from quchip.declarative.expr import (
                as_state_expr,
                materialize_expr,
            )

            device = chip[label]
            expression = as_state_expr(
                val,
                labels=(device.label,),
                dims=(basis.native_dim,),
                name=rf"\lvert\psi_{{{device.label}}}\rangle",
                owner=device,
                scope=device.label,
            )
            val = materialize_expr(expression, backend)
            prepared[label] = val
            shape = getattr(val, "shape", None)
            is_array_ket = shape is not None and (
                len(shape) == 1 or (len(shape) == 2 and shape[1] == 1)
            )
            if not is_array_ket and not backend.is_ket(val):
                raise ValueError(f"State for '{label}' must be a ket vector, got a non-ket state")
            if shape is not None and shape[0] != basis.native_dim:
                raise ValueError(
                    f"Authored state dimension for '{label}' is {val.shape[0]}, "
                    f"expected {basis.native_dim}."
                )

    kets: list[State] = []
    for dev in chip.devices:
        val = prepared.get(dev.label)
        basis = bases[dev.label]
        if val is None:
            level = 0
        elif isinstance(val, int):
            level = val
        else:
            authored = backend.to_array(val).reshape(basis.native_dim, -1)
            projected = basis.vectors.conj().T @ authored
            authored_norm = backend.array_module.linalg.norm(authored)
            projected_norm = backend.array_module.linalg.norm(projected)
            lost = maybe_concrete_scalar(authored_norm**2 - projected_norm**2)
            if lost is not None and lost > 1e-10:
                warnings.warn(
                    f"Projection discarded {lost:.3g} of the state norm on {dev.label!r}.",
                    stacklevel=2,
                )
            kets.append(
                backend.from_array(
                    projected,
                    dims=[[basis.resolved_dim], [1]],
                )
            )
            continue

        vector = basis.energy_state(level).reshape(basis.resolved_dim, 1)
        kets.append(backend.from_array(vector, dims=[[basis.resolved_dim], [1]]))

    if len(kets) == 1:
        return kets[0]
    return backend.tensor_states(*kets)


def default_initial_state(chip: "Chip", engine_result: Any, start_time: Any) -> State:
    """Return the default initial state for a solve.

    Use the eigenstate assigned to the all-ground label, taken from the
    undriven static lab-frame Hamiltonian that ``engine_result.approximation``
    keeps. Make its overlap with the bare product real and nonnegative, then
    express it in the solve frame at ``start_time``. If the bare product is
    already an eigenstate, return the bare product directly.
    """
    from quchip.engine.assembly import _may_raise_ground

    approximation = engine_result.approximation
    moves = engine_result.slh.has_network_hamiltonian or _may_raise_ground(chip, approximation, engine_result.bases)
    ground = chip.analysis._ground_ket(approximation) if moves else None
    if ground is None:
        return _bare_state_from_bases(chip, {}, engine_result.bases)
    ket = _to_solve_frame(ground, chip, engine_result, start_time)
    dims = list(engine_result.dims)
    return chip.backend.from_array(ket.reshape(-1, 1), dims=[dims, [1] * len(dims)])


def _to_solve_frame(ket: Any, chip: "Chip", engine_result: Any, start_time: Any) -> Any:
    """Express a lab-frame ket in the solve's frame at ``start_time``.

    The solver evolves ``U(t)† psi`` with ``U(t) = exp(-i 2π t Σ_i f_i n_i)``
    in absolute time, where ``n_i`` is each device's energy-level index.
    """
    xp = chip.backend.array_module
    frequencies = engine_result.resolved_frame.frequencies
    tensor = ket.reshape(tuple(engine_result.dims))
    for axis, device in enumerate(chip.devices):
        angle = TWO_PI * start_time * frequencies.get(device.label, 0.0)
        if maybe_concrete_scalar(angle) == 0.0:
            continue
        record = engine_result.bases[device.label]
        phases = xp.exp(1j * xp.asarray(angle) * xp.arange(record.resolved_dim))
        transform = record.energy_to_solver()
        if transform is None:
            local = xp.diag(phases)
        else:
            transform = xp.asarray(transform)
            local = (transform * phases) @ xp.conj(transform).T
        tensor = xp.moveaxis(xp.tensordot(local, tensor, axes=([1], [axis])), 0, axis)
    return tensor.reshape(-1)


def materialize_state_spec(
    chip: "Chip",
    state_spec: Any,
    bases: Mapping[str, Any],
) -> State:
    """Materialize one authored state specification in the engine's solver basis.

    An omitted state is :func:`default_initial_state`, which needs the solve's engine result.
    """
    if isinstance(state_spec, (Mapping, str)):
        resolved = normalize_device_state_mapping(chip, state_spec, {})
        return _bare_state_from_bases(chip, resolved, bases)

    from quchip.declarative.expr import as_state_expr, materialize_expr

    backend = chip.backend
    ordered = [bases[device.label] for device in chip.devices]
    authored_dims = tuple(record.native_dim for record in ordered)
    resolved_dims = tuple(record.resolved_dim for record in ordered)
    authored_dim = prod(authored_dims)
    resolved_dim = prod(resolved_dims)
    # QuTiP ``Qobj`` (``full``) and dynamiqs ``QArray`` (``to_jax``) states are in
    # solver coordinates whichever backend solves; plain arrays and symbolic
    # specifications are authored.
    resolved_native = (
        backend.is_native_state(state_spec) or hasattr(state_spec, "full") or hasattr(state_spec, "to_jax")
    )

    shape = getattr(state_spec, "shape", None)
    if shape is None:
        state_spec = as_state_expr(
            state_spec,
            labels=tuple(device.label for device in chip.devices),
            dims=authored_dims,
            name=r"\lvert\psi\rangle",
            owner=chip,
            scope="chip",
        )
    state = materialize_expr(state_spec, backend)
    shape = getattr(state, "shape", None)
    if shape is None:
        raise TypeError("Initial state must be a semantic specification, callable, or ket.")
    if resolved_native:
        if shape[0] != resolved_dim:
            raise ValueError(
                f"A solver-native initial state must have dimension {resolved_dim}, got {shape}."
            )
        return state
    if shape[0] != authored_dim or len(shape) not in (1, 2) or (len(shape) == 2 and shape[1] != 1):
        raise ValueError(
            "Initial-state dimension must match the authored or resolved chip space; "
            f"got {shape}, authored {authored_dim}, resolved {resolved_dim}."
        )

    authored = backend.to_array(state).reshape(authored_dim, 1)
    xp = array_namespace(ordered[0].vectors)
    transform = ordered[0].vectors
    for record in ordered[1:]:
        transform = xp.kron(transform, record.vectors)
    projected = transform.conj().T @ authored
    authored_norm = xp.linalg.norm(authored)
    projected_norm = xp.linalg.norm(projected)
    lost = maybe_concrete_scalar(authored_norm**2 - projected_norm**2)
    if lost is not None and lost > 1e-10:
        warnings.warn(
            f"Projection discarded {lost:.3g} of the initial-state norm.",
            stacklevel=2,
        )
    return backend.from_array(projected, dims=[list(resolved_dims), [1]])
