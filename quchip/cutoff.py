"""Explicit retained-space policies for chips.

An approximation (:class:`~quchip.approximations.RWA` or
:class:`~quchip.approximations.Exact`) selects Hamiltonian terms. A space
policy selects the product energy labels that a calculation keeps. quchip
never sets or changes a chip's space policy. Without a policy,
:meth:`~quchip.VNA.finite_power` and :meth:`~quchip.VNA.measure` can truncate
their own stationary solve of a large excitation-conserving chip, and they
warn which truncation they kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Sequence


@dataclass(frozen=True)
class ExcitationCutoff:
    """Keep the product energy labels whose levels sum to at most ``max_total``.

    A label assigns one isolated local energy level ``n_i`` to each device, and
    its total excitation is ``N = sum_i n_i``. The retained space holds every
    label with ``N <= max_total``. The cutoff counts levels, not energy, so it
    is not an energy window.

    With projector ``P`` on the retained space, the model keeps ``P H(t) P``,
    including every matrix element between retained sectors, and the projected
    jumps ``P L P``. Counter-rotating terms stay when the chip's approximation
    keeps them. A cutoff with ``max_total >= sum_i (d_i - 1)`` keeps the full
    product space and leaves every calculation unchanged.

    Parameters
    ----------
    max_total : int
        Largest retained total excitation, at least 1.

    Examples
    --------
    >>> from quchip import Capacitive, Chip, DuffingTransmon, Exact, ExcitationCutoff, Resonator
    >>> q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    >>> r = Resonator(freq=7.0, levels=6, label="r")
    >>> chip = Chip([q, r], [Capacitive(q, r, g=0.1)], approximation=Exact())
    >>> limited = chip.with_space(ExcitationCutoff(max_total=3))
    >>> f_q = limited.freq(q)  # dressed 0 -> 1 frequency of the truncated model, GHz
    """

    max_total: int

    def __post_init__(self) -> None:
        value = self.max_total
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"max_total must be a positive integer, got {value!r}.")
        object.__setattr__(self, "max_total", int(value))

    def binds(self, dims: Sequence[int]) -> bool:
        """Return whether the cutoff removes product labels of a space with local ``dims``.

        Parameters
        ----------
        dims : sequence of int
            Retained level count of each device.
        """
        return self.max_total < sum(int(dimension) - 1 for dimension in dims)

    def to_dict(self) -> dict[str, Any]:
        """Return the stable serialized policy."""
        return {"type": "ExcitationCutoff", "max_total": self.max_total}

    @staticmethod
    def from_dict(data: Any) -> "ExcitationCutoff":
        """Restore a policy from :meth:`to_dict` output.

        Parameters
        ----------
        data : mapping
            Serialized ``ExcitationCutoff`` record.
        """
        if not isinstance(data, dict) or data.get("type") != "ExcitationCutoff":
            raise ValueError(f"Unknown space policy {data!r}.")
        if set(data) != {"type", "max_total"}:
            raise TypeError(f"Unsupported serialized space fields: {sorted(set(data) - {'type'})}")
        return ExcitationCutoff(data["max_total"])


def require_space(value: Any) -> ExcitationCutoff | None:
    """Validate a public space policy.

    Parameters
    ----------
    value : ExcitationCutoff or None
        Policy to validate. ``None`` keeps the full product space.
    """
    if value is not None and not isinstance(value, ExcitationCutoff):
        raise TypeError(f"space must be an ExcitationCutoff or None, got {type(value).__name__}.")
    return value


def active_cutoff(chip: Any) -> ExcitationCutoff | None:
    """Return the chip's cutoff when it removes product labels, else ``None``.

    Parameters
    ----------
    chip : Chip
        Chip whose policy and retained local dimensions are read.
    """
    policy = chip.space
    if policy is None or not policy.binds(chip.dims):
        return None
    return policy


def reject_cutoff(chip: Any, workflow: str) -> None:
    """Raise when a workflow without cutoff support receives an active cutoff.

    Parameters
    ----------
    chip : Chip
        Chip whose policy is checked.
    workflow : str
        Name of the calling workflow, used in the message.
    """
    policy = active_cutoff(chip)
    if policy is not None:
        raise ValueError(
            f"{workflow} does not support an excitation cutoff, and this chip keeps only "
            f"N <= {policy.max_total}. Use chip.with_space(None) to run it on the full product space."
        )


__all__ = ["ExcitationCutoff"]
