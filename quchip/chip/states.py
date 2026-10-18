"""Bare states, dressed eigenstates, and normalized superpositions for chips.

String shorthand such as ``"eg1"`` uses the device order and level symbols that
:func:`set_state_order` sets. Dressed-state selection uses the assigned
eigenvector column and stays JAX-traceable on a fixed assignment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import jax

from quchip.backend.protocol import State
from quchip.utils.jax_utils import maybe_concrete_scalar
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
    """Normalized superposition of the dressed states of level labels.

    Each component is a level label or an ``(amplitude, label)`` pair. A label
    is a dict keyed by device or label, or a string after you call
    :func:`set_state_order`, and names the dressed state that
    :meth:`~quchip.Chip.state` returns. The weights are uniform by default,
    and the result is normalized to unit norm. :func:`bare_state` takes the
    same components for a superposition of bare products.

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
    return _combine(chip, components, chip.state)


def _combine(chip: "Chip", components: tuple[Any, ...], prepare: Any) -> State:
    """Return the normalized sum of the states that ``prepare`` gives for each component."""
    if not components:
        raise ValueError("A superposition requires at least one component.")
    amps: list[Any] = []
    kets: list[State] = []
    for component in components:
        if (
            isinstance(component, tuple)
            and len(component) == 2
            and not isinstance(component[0], (dict, str, Mapping))
        ):
            amp, spec = component
        else:
            amp, spec = 1.0, component
        kets.append(prepare(spec))
        amps.append(amp)

    psi = amps[0] * kets[0]
    for amp, ket in zip(amps[1:], kets[1:]):
        psi = psi + amp * ket
    backend = chip.backend
    norm = backend.norm(psi)
    # Backend.norm may return a traced 0-d array (dynamiqs). A concrete zero
    # norm raises. Otherwise the divisor itself must be guarded (mirrors
    # Bath._bose's safe-denominator pattern): xp.where evaluates both branches,
    # so dividing by the (possibly zero) traced norm directly would still
    # produce 0/0 -> NaN in the unselected branch. Replacing the zero norm
    # with 1.0 before the division makes an all-zero traced amplitude set
    # return the already-zero (unnormalized) state instead of NaN.
    concrete_norm = maybe_concrete_scalar(norm)
    if concrete_norm is not None:
        if concrete_norm <= 0:
            raise ValueError("The superposition has zero norm.")
        return psi / norm
    xp = backend.array_module
    safe_norm = xp.where(norm == 0, 1.0, norm)
    return psi / safe_norm


def bare_state(
    chip: "Chip",
    *components: Mapping[str | "BaseDevice", int | State] | str | tuple[Any, Any],
    **device_state_kwargs: int | State,
) -> State:
    """Product state, or normalized superposition of product states, from energy levels or local kets.

    You can specify each device as an energy-level index (``int``) or as a ket vector
    in that device's authored local space. Unmentioned devices default to the ground
    state (level 0). Unlike :meth:`~quchip.Chip.state`, this function does **not**
    diagonalize the coupled system.

    One mapping or string, keywords, or both give one product state. Several
    components, or ``(amplitude, specification)`` pairs, give the normalized
    superposition of their products, as :func:`superposition` combines
    dressed states. Accepts a string shorthand (for example ``"eg1"``) after
    you call :func:`set_state_order`.

    Inside ``jax.jit``, a state that depends on no traced value stays a
    constant, so a solve can read its excitation support.
    """
    from quchip.engine.states import bare_state_from_bases

    bases = chip.resolve().bases

    def product(spec: Any, keywords: dict[str, Any] | None = None) -> State:
        resolved = normalize_device_state_mapping(chip, spec, keywords or {})
        return bare_state_from_bases(chip, resolved, bases)

    weighted = any(
        isinstance(component, tuple) and len(component) == 2 and not isinstance(component[0], (dict, str, Mapping))
        for component in components
    )
    if device_state_kwargs and (len(components) > 1 or weighted):
        raise TypeError("Keyword levels name one product state. Write each component of a superposition as a mapping.")
    # jit stages every array operation, so a fixed ket would become a tracer of unknown support.
    with jax.ensure_compile_time_eval():
        if len(components) <= 1 and not weighted:
            return product(components[0] if components else None, device_state_kwargs)
        return _combine(chip, components, product)
