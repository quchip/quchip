"""Acyclic reference transformations downstream of the Markovian boundary."""

from dataclasses import dataclass
from typing import Any

from quchip.engine.reference import (
    ReferenceAmplifier, ReferenceDelay, ReferenceFilter, ReferenceLoss, cw_transfer, noise_density,
)


@dataclass(frozen=True)
class OutputStep:
    """One directed output terminal in a frozen downstream calculation."""

    key: tuple[str, str]
    base: Any
    feeders: tuple[tuple[Any, tuple[str, str] | None, Any], ...] = ()
    reference: Any = None


@dataclass(frozen=True)
class OutputNetwork:
    """Resolved field mixing and independent sources outside quantum dynamics.

    Steps are topologically ordered. Base maps reconstruct an unaffected
    terminal from the complete unitary boundary. Only downstream steps can
    change, and no reference step can feed a quantum coupling or a feedback
    loop.
    """

    steps: tuple[OutputStep, ...]
    outputs: tuple[tuple[str, str], ...]
    boundary: Any
    size: int

    @property
    def colored(self) -> bool:
        """Return whether internal propagation makes the spectra frequency-dependent."""
        return any(isinstance(step.reference, (ReferenceFilter, ReferenceDelay)) for step in self.steps)

    def evaluate(self, frequency: Any, xp: Any) -> tuple[Any, dict[str, Any]]:
        """Return the output field matrix and source-wise normal cross-spectra."""
        fields: dict[tuple[str, str], Any] = {}
        noises: dict[tuple[str, str], dict[str, Any]] = {}
        densities = {}
        for step in self.steps:
            if not step.feeders:
                fields[step.key] = xp.asarray(step.base)
                noises[step.key] = {}
                continue
            transfer = 1.0 if step.reference is None else cw_transfer((step.reference,), frequency, xp)
            field = xp.zeros(self.size, dtype=complex)
            noise: dict[str, Any] = {}
            for coefficient, upstream, base in step.feeders:
                field = field + coefficient * (xp.asarray(base) if upstream is None else fields[upstream])
                if upstream is not None:
                    for source, value in noises[upstream].items():
                        noise[source] = noise.get(source, 0.0) + coefficient * value
            fields[step.key] = transfer * field
            noises[step.key] = {source: transfer * value for source, value in noise.items()}
            if isinstance(step.reference, (ReferenceAmplifier, ReferenceFilter, ReferenceLoss)):
                source = '.'.join(step.key)
                densities[source] = noise_density((step.reference,), frequency, xp)
                noises[step.key][source] = xp.asarray(1.0)
        boundary = xp.asarray(self.boundary)
        transform = boundary @ xp.stack([fields[key] for key in self.outputs])
        contributions = {}
        for source, density in densities.items():
            column = boundary @ xp.asarray([noises[key].get(source, 0.0) for key in self.outputs])
            contributions[source] = density * column[:, None] * xp.conj(column[None, :])
        return transform, contributions


def output_mixing(slh: Any, frequency: Any, xp: Any) -> Any:
    """Return the full-channel output map, with identity on intrinsic hidden baths."""
    count = len(slh.channels)
    if slh.output_network is None:
        return xp.eye(count, dtype=complex)
    block, _ = slh.output_network.evaluate(frequency, xp)
    return pad_mixing(block, count, xp)


def pad_mixing(block: Any, count: int, xp: Any) -> Any:
    """Extend a network field map by identity on intrinsic hidden channels."""
    remaining = count - len(block)
    return xp.block([[block, xp.zeros((len(block), remaining))],
                     [xp.zeros((remaining, len(block))), xp.eye(remaining)]])
