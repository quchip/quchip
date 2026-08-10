"""State factories for :class:`~quchip.chip.chip.Chip`.

These helpers build bare tensor-product kets, dressed eigenstates, and
normalized superpositions, plus the ``chip.set_state_order(...)`` +
string-shorthand machinery where ``"eg1"``-style specs name one
level per device. The chip forwards its public state surface
(:meth:`Chip.state`, :meth:`Chip.bare_state`, :meth:`Chip.superposition`,
:meth:`Chip.set_state_order`) here; users normally call the chip methods,
not these functions directly.

Module-level functions (taking ``chip`` as the first argument) mirror
:mod:`quchip.chip.serialization`. The only per-chip state involved — the
declared device order and level symbols — lives on the chip itself
(``chip._state_order`` / ``chip._level_symbols``), set by
:func:`set_state_order`.

All spec inputs route through :func:`normalize_device_state_mapping`,
which is the single place the ``str`` shorthand is parsed into a
``{label: value}`` dict, so the public factories never repeat the
``isinstance(..., str)`` guard.

The dressed :func:`state` path stays JAX-traceable: it forwards to
:meth:`ChipAnalysis.state`, which selects the assigned eigenvector
column through the :func:`~quchip.chip.dressing.label_eigensystem`
array kernel.
"""

from __future__ import annotations

import warnings
from math import prod
from typing import TYPE_CHECKING, Any, Mapping

from quchip.backend.protocol import State
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
    """Declare the device order used to parse string-state shorthands.

    After this is called, :meth:`Chip.bare_state`, :meth:`Chip.state`, and
    :meth:`Chip.superposition` accept single-string specifications where
    each character is one level per device in *devices* order.
    Level symbols default to ``g=0, e=1, f=2, h=3``; digits ``0..9``
    are always accepted as energy-level indices.

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
    """Merge a mapping (or string shorthand) and kwargs into ``{label: value}``.

    A ``str`` *device_states* is the single place the ``"eg1"`` shorthand
    is parsed (via :func:`parse_state_string`), so every public state
    factory routes through here instead of repeating the guard. After the
    string shorthand and the mapping type-check, the resolve-and-dedup step
    is delegated to :func:`~quchip.utils.labeling.merge_labeled_values` — the
    same primitive the bare-tuple builder uses, so a duplicate device
    specification means the same thing here and in spectroscopy sweeps.
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

    Each component is either a bare-state spec (dict keyed by device or
    label, or a string when :func:`set_state_order` has been called) or
    an ``(amplitude, spec)`` tuple for weighted mixing. Uniform weights
    by default; results are normalized to unit norm.

    Unlike :func:`state`, this stays in the bare product basis — no
    dressed diagonalization — so the probe basis is explicit.

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
    bases = chip.engine_result().bases
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


def state(
    chip: "Chip",
    device_states: Mapping[str | "BaseDevice", int] | str | None = None,
    /,
    **device_state_kwargs: int,
) -> State:
    """Dressed eigenstate assigned from the given product-state level labels.

    Accepts a string shorthand (e.g. ``"eg1"``) when
    :func:`set_state_order` has been called.

    Safe inside ``jax.jit``/``grad``/``vmap``: under tracing the
    assigned eigenvector column is selected through the
    :func:`~quchip.chip.dressing.label_eigensystem` array kernel, so
    dressed initial states are differentiable end-to-end. The global
    phase is gauge-dependent (``eigh`` column convention) —
    populations and ``|overlap|`` figures of merit are unaffected.
    """
    return chip._analysis.state(device_states, **device_state_kwargs)


def bare_state(
    chip: "Chip",
    device_states: Mapping[str | "BaseDevice", int | State] | str | None = None,
    /,
    **device_state_kwargs: int | State,
) -> State:
    """Product state from per-device energy levels or authored local kets.

    Each device may be specified as either an energy-level index (``int``)
    or a ket vector in that device's authored local space. Devices not
    mentioned default to the ground state (level 0). Unlike :func:`state`
    this does **not** diagonalize the coupled system.

    Accepts a string shorthand (e.g. ``"eg1"``) when
    :func:`set_state_order` has been called.
    """
    resolved = normalize_device_state_mapping(chip, device_states, device_state_kwargs)
    return _bare_state_from_bases(chip, resolved, chip.engine_result().bases)


def _bare_state_from_bases(
    chip: "Chip",
    resolved: Mapping[str, Any],
    bases: Mapping[str, Any],
) -> State:
    """Build a solver ket from semantic levels or authored local arrays."""
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

        if basis.kind == "eigen":
            kets.append(backend.basis(basis.resolved_dim, level))
        else:
            vector = basis.energy_vectors[:, level].reshape(basis.native_dim, 1)
            kets.append(backend.from_array(vector, dims=[[basis.native_dim], [1]]))

    if len(kets) == 1:
        return kets[0]
    return backend.tensor_states(*kets)


def materialize_state_spec(
    chip: "Chip",
    state_spec: Any,
    bases: Mapping[str, Any],
) -> State:
    """Materialize one authored state specification in the engine's solver basis."""
    if state_spec is None or isinstance(state_spec, (Mapping, str)):
        resolved = normalize_device_state_mapping(chip, state_spec, {})
        return _bare_state_from_bases(chip, resolved, bases)

    from quchip.declarative.expr import as_state_expr, materialize_expr

    backend = chip.backend
    ordered = [bases[device.label] for device in chip.devices]
    authored_dims = tuple(record.native_dim for record in ordered)
    resolved_dims = tuple(record.resolved_dim for record in ordered)
    authored_dim = prod(authored_dims)
    resolved_dim = prod(resolved_dims)
    resolved_native = backend.is_native_state(state_spec)

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
