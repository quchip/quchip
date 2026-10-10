"""Reference-plane transforms applied outside the Markovian core."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from quchip.utils.constants import TWO_PI
from quchip.utils.jax_utils import contains_tracer, maybe_concrete_scalar, select_array_module


@dataclass(frozen=True)
class ReferenceDelay:
    """One reference section with propagation duration ``duration`` ns."""

    label: str
    duration: Any

    def __post_init__(self) -> None:
        concrete = maybe_concrete_scalar(self.duration)
        if concrete is not None and concrete < 0:
            raise ValueError(f"Delay {self.label!r} duration must be non-negative, got {self.duration}.")

    @property
    def tracked_values(self) -> tuple[Any, ...]:
        """Return the values that can carry JAX tracers."""
        return (self.duration,)


@dataclass(frozen=True)
class ReferenceFilter:
    """One passive filter reference section and its tracked parameters.

    ``transfer(frequency, **parameters)`` accepts a scalar or array frequency in
    GHz and returns a complex amplitude transfer.
    """

    label: str
    transfer: Callable[..., Any]
    parameters: Mapping[str, Any]
    loss_occupation: Any = None

    @property
    def tracked_values(self) -> tuple[Any, ...]:
        """Return the values that can carry JAX tracers."""
        return (*self.parameters.values(), self.loss_occupation)

    def __call__(self, frequency: Any) -> Any:
        """Evaluate the complex transfer at ``frequency`` in GHz.

        ``frequency`` can be scalar or array-valued. Reject a concrete result whose
        magnitude is more than one. Do not concretize traced results for this
        check.
        """
        value = self.transfer(frequency, **self.parameters)
        if not contains_tracer(value) and np.any(np.abs(np.asarray(value)) > 1.0 + 1e-12):
            raise ValueError(
                f"Filter {self.label!r} must be passive, but its evaluated transfer has |H| > 1. "
                "Gain requires an amplifier section."
            )
        return value


@dataclass(frozen=True)
class ReferenceAmplifier:
    """A phase-preserving output-line amplifier.

    ``gain`` is the power gain ``G``. ``added_noise`` is the input-referred
    symmetrized noise ``n_add`` in quanta, with the quantum floor
    ``(1 - 1/G) / 2``.
    """

    label: str
    gain: Any
    added_noise: Any

    def __post_init__(self) -> None:
        gain = maybe_concrete_scalar(self.gain)
        noise = maybe_concrete_scalar(self.added_noise)
        if gain is not None and gain < 1.0:
            raise ValueError(f"Amplifier {self.label!r} power gain must be at least 1; got {self.gain}.")
        if gain is not None and noise is not None:
            floor = (1.0 - 1.0 / gain) / 2.0
            if noise < floor - 1e-12:
                raise ValueError(
                    f"Amplifier {self.label!r} added_noise={self.added_noise} is below the "
                    f"phase-preserving quantum limit {floor} input-referred symmetrized quanta "
                    f"for gain={self.gain}."
                )

    @property
    def tracked_values(self) -> tuple[Any, ...]:
        """Return the values that can carry JAX tracers."""
        return (self.gain, self.added_noise)

    @property
    def added_noise_density(self) -> Any:
        """Return the output-referred normally ordered added noise density.

        For power gain ``G`` and input-referred noise ``n_add``, the density is
        ``G n_add + (G - 1) / 2``.
        """
        return self.gain * self.added_noise + (self.gain - 1.0) / 2.0


@dataclass(frozen=True)
class ReferenceLoss:
    """One directed traversal of a matched passive output-line component."""

    label: str
    eta: Any
    occupation: Any = None

    @property
    def tracked_values(self) -> tuple[Any, ...]:
        """Return numerical source and transmission parameters."""
        return (self.eta, self.occupation)


ReferenceElement = ReferenceDelay | ReferenceFilter | ReferenceAmplifier | ReferenceLoss


@dataclass(frozen=True)
class ReferencePlane:
    """Inbound and outbound reference runs for one exposure, in propagation order."""

    inbound: tuple[ReferenceElement, ...] = ()
    outbound: tuple[ReferenceElement, ...] = ()


@dataclass(frozen=True)
class FieldChannel:
    """Field metadata shared by mode-space and operator-space acquisition."""

    key: str
    reference: ReferencePlane
    input_occupation: Any = None


def _array_module(elements: tuple[ReferenceElement, ...], frequency: Any, xp: Any) -> Any:
    """Return ``xp``, or the module matching traced content in ``elements`` and ``frequency``."""
    if xp is not None:
        return xp
    values: list[Any] = [frequency]
    for element in elements:
        values.extend(element.tracked_values)
    return select_array_module(contains_tracer(tuple(values)))


def cw_transfer(elements: tuple[ReferenceElement, ...], frequency: Any, xp: Any = None) -> Any:
    """Return the exact continuous-wave transfer of one reference leg.

    At ``frequency`` in GHz, each delay contributes
    ``exp(+i 2π frequency duration)`` and each filter contributes
    ``H(frequency)``. Delay durations are in ns.
    """
    xp = _array_module(elements, frequency, xp)
    transfer = xp.asarray(1.0 + 0.0j)
    for element in elements:
        if isinstance(element, ReferenceDelay):
            transfer = transfer * xp.exp(
                1j * TWO_PI * xp.asarray(frequency) * xp.asarray(element.duration)
            )
        elif isinstance(element, ReferenceFilter):
            transfer = transfer * xp.asarray(element(frequency))
        elif isinstance(element, ReferenceLoss):
            transfer = transfer * xp.sqrt(xp.asarray(element.eta))
        else:
            transfer = transfer * xp.sqrt(xp.asarray(element.gain))
    return transfer


def carrier_transfer(elements: tuple[ReferenceElement, ...], carrier: Any, xp: Any = None) -> Any:
    """Return the narrowband filter transfer of one reference leg.

    Evaluate every filter at ``carrier`` in GHz. Ignore delays because transient
    propagation applies them as time shifts.
    """
    xp = _array_module(elements, carrier, xp)
    flat = tuple(element for element in elements if not isinstance(element, ReferenceDelay))
    return cw_transfer(flat, carrier, xp)


def noise_contributions(elements: tuple[ReferenceElement, ...], frequency: Any, xp: Any = None) -> dict[str, Any]:
    """Return the chain's output-referred normally ordered added noise density.

    At ``frequency`` in GHz, walk ``elements`` in propagation order. A filter
    applies ``N <- |H(f)|^2 N``. An amplifier applies
    ``N <- G N + G n_add + (G - 1) / 2``.
    """
    xp = _array_module(elements, frequency, xp)
    contributions: dict[str, Any] = {}
    for element in elements:
        if isinstance(element, ReferenceDelay):
            continue
        if isinstance(element, ReferenceAmplifier):
            gain, added = element.gain, element.added_noise_density
        else:
            gain = (xp.abs(xp.asarray(element(frequency))) ** 2 if isinstance(element, ReferenceFilter)
                    else element.eta)
            occupation = element.loss_occupation if isinstance(element, ReferenceFilter) else element.occupation
            added = (1 - gain) * (0.0 if occupation is None else occupation)
        contributions = {label: gain * value for label, value in contributions.items()}
        contributions[element.label] = xp.zeros_like(xp.asarray(frequency, dtype=float)) + added
    return contributions


def noise_density(elements: tuple[ReferenceElement, ...], frequency: Any, xp: Any = None) -> Any:
    """Return total normally ordered added noise from the source-wise propagation."""
    xp = _array_module(elements, frequency, xp)
    return sum(noise_contributions(elements, frequency, xp).values(),
               xp.zeros_like(xp.asarray(frequency, dtype=float)))


def noise_colors(elements: tuple[ReferenceElement, ...]) -> dict[str, bool]:
    """Mark sources whose emission passes through a frequency-dependent filter."""
    colored = False
    sources = {}
    for element in reversed(elements):
        colored = colored or isinstance(element, ReferenceFilter)
        if not isinstance(element, ReferenceDelay):
            sources[element.label] = colored
    return sources


def has_colored_noise(elements: tuple[ReferenceElement, ...]) -> bool:
    """Return whether a potentially occupied source emits colored noise."""
    colors = noise_colors(elements)
    for element in elements:
        if isinstance(element, ReferenceDelay) or not colors[element.label]:
            continue
        if isinstance(element, ReferenceLoss) and maybe_concrete_scalar(element.eta) == 1:
            continue
        occupation = source_occupation(element)
        if occupation is not None and maybe_concrete_scalar(occupation) != 0:
            return True
    return False


def source_occupation(element: ReferenceElement) -> Any:
    """Return a declared source occupation, or None for implicit vacuum."""
    if isinstance(element, ReferenceDelay):
        return None
    if isinstance(element, ReferenceAmplifier):
        return element.added_noise_density
    return element.loss_occupation if isinstance(element, ReferenceFilter) else element.occupation


def has_filter(elements: tuple[ReferenceElement, ...]) -> bool:
    """Return whether a reference leg contains a filter section."""
    return any(isinstance(element, ReferenceFilter) for element in elements)


def has_amplifier(elements: tuple[ReferenceElement, ...]) -> bool:
    """Return ``True`` when ``elements`` contains an amplifier."""
    return any(isinstance(element, ReferenceAmplifier) for element in elements)


def time_shift(elements: tuple[ReferenceElement, ...]) -> Any:
    """Return the summed delay duration of one reference leg, in ns."""
    return sum((element.duration for element in elements if isinstance(element, ReferenceDelay)), 0.0)
