"""Physical Markovian input-output channels attached to a chip."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from quchip.declarative.dissipation import CollapseChannel, normalize_dissipation
from quchip.utils.jax_utils import maybe_concrete_scalar
from quchip.utils.labeling import auto_label, resolve_label

if TYPE_CHECKING:
    from quchip.chip.chip import Chip


class Port:
    """One accessible Markovian channel with a dimensionless coupling operator."""

    _type_prefix = "port"
    _parameter_names = ("rate", "external_quality_factor", "phase")

    def __init__(
        self,
        target: Any | Sequence[Any],
        *,
        rate: Any = None,
        external_quality_factor: Any = None,
        operator: Any = None,
        phase: Any = 0.0,
        label: str | None = None,
    ) -> None:
        if isinstance(target, Sequence) and not isinstance(target, (str, bytes)):
            targets = tuple(target)
        else:
            targets = (target,)
        if not targets:
            raise ValueError("Port requires at least one target device.")
        if (rate is None) == (external_quality_factor is None):
            raise ValueError("Port requires exactly one of rate or external_quality_factor.")
        if external_quality_factor is not None and len(targets) != 1:
            raise ValueError("external_quality_factor is defined only for a single target device.")
        if operator is None and len(targets) != 1:
            raise ValueError("A multi-device Port requires an explicit coupling operator.")

        self._targets = targets
        self.rate = rate
        self.external_quality_factor = external_quality_factor
        self.operator = operator
        self.phase = phase
        self.label = label if label is not None else auto_label(self._type_prefix)

    @staticmethod
    def _validate_positive(name: str, value: Any) -> None:
        concrete = maybe_concrete_scalar(value)
        if concrete is not None and concrete <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("rate", "external_quality_factor"):
            self._validate_positive(name, value)
            other_name = "external_quality_factor" if name == "rate" else "rate"
            if hasattr(self, other_name):
                other = getattr(self, other_name)
                if (value is None) == (other is None):
                    raise ValueError("Port requires exactly one of rate or external_quality_factor.")
        super().__setattr__(name, value)

    def resolve_targets(self, chip: "Chip") -> tuple[str, ...]:
        """Return target labels after checking that they belong to *chip*."""
        labels = tuple(resolve_label(target) for target in self._targets)
        unknown = [label for label in labels if label not in chip.device_map]
        if unknown:
            raise ValueError(f"Port {self.label!r} targets unknown device(s) {unknown}.")
        if len(set(labels)) != len(labels):
            raise ValueError(f"Port {self.label!r} repeats a target device.")
        return labels

    def rate_value(self, chip: "Chip") -> Any:
        """Return the external coupling rate in ``1/ns``."""
        if self.rate is not None:
            return self.rate
        label = self.resolve_targets(chip)[0]
        target = chip[label]
        if not hasattr(target, "freq"):
            raise TypeError("external_quality_factor requires a target with a freq parameter.")
        return 2.0 * np.pi * target.freq / self.external_quality_factor

    def _authored_operator(self, chip: "Chip") -> Any:
        labels = self.resolve_targets(chip)
        if self.operator is None:
            return chip[labels[0]].lowering_operator()
        if isinstance(self.operator, str):
            if len(labels) != 1:
                raise ValueError("A named Port operator requires exactly one target.")
            return chip[labels[0]].local_operator(self.operator)
        return self.operator

    def _collapse_channels_with_paths(
        self,
        chip: "Chip",
    ) -> tuple[tuple[CollapseChannel, tuple[str, ...]], ...]:
        labels = self.resolve_targets(chip)
        normalized = normalize_dissipation(
            (CollapseChannel(self._authored_operator(chip), self.rate_value(chip), "external_coupling"),),
            labels=labels,
            dims=tuple(chip[label].local_space().dimension for label in labels),
            owner=self,
            scope=f"port.{self.label}",
        )
        rate_paths: tuple[str, ...]
        if self.rate is not None:
            rate_paths = (f"port.{self.label}.rate",)
        else:
            rate_paths = (
                f"{labels[0]}.freq",
                f"port.{self.label}.external_quality_factor",
            )
        return tuple((channel, tuple(dict.fromkeys((*paths, *rate_paths)))) for channel, paths in normalized)

    def parameter_values(self) -> dict[str, Any]:
        """Return active sweepable port values."""
        return {
            name: value
            for name in self._parameter_names
            if (value := getattr(self, name)) is not None
        }

    def set_parameter_value(self, name: str, value: Any) -> None:
        """Set one port parameter on an isolated chip copy."""
        if name not in self._parameter_names:
            raise KeyError(name)
        setattr(self, name, value)

    def copy(self) -> "Port":
        """Return an independent port retaining label-based targets."""
        operator = self.operator.copy() if isinstance(self.operator, np.ndarray) else self.operator
        return Port(
            tuple(resolve_label(target) for target in self._targets),
            rate=self.rate,
            external_quality_factor=self.external_quality_factor,
            operator=operator,
            phase=self.phase,
            label=self.label,
        )

    def physics_notes(self) -> list[str]:
        """Return the input-output convention owned by this port."""
        return [
            "This accessible Markovian channel uses "
            "L = exp(i phase) sqrt(rate) A and b_out = b_in - L; "
            "the same L sets damping, coherent input coupling, and reported output."
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize port targets and scalar coupling data."""
        operator: Any = self.operator
        if operator is not None and not isinstance(operator, str):
            if hasattr(operator, "matrix"):
                operator = operator.matrix()
            elif hasattr(operator, "to_jax"):
                operator = operator.to_jax()
            elif hasattr(operator, "full"):
                operator = operator.full()
            array = np.asarray(operator, dtype=complex)
            operator = {
                "kind": "dense",
                "real": array.real.tolist(),
                "imag": array.imag.tolist(),
            }
        elif isinstance(operator, str):
            operator = {"kind": "named", "name": operator}
        return {
            "targets": [resolve_label(target) for target in self._targets],
            "rate": self.rate,
            "external_quality_factor": self.external_quality_factor,
            "operator": operator,
            "phase": self.phase,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Port":
        """Reconstruct a port from label-based serialized data."""
        targets = data["targets"]
        target: Any = targets[0] if len(targets) == 1 else targets
        operator = data.get("operator")
        if isinstance(operator, dict):
            if operator.get("kind") == "named":
                operator = operator["name"]
            elif operator.get("kind") == "dense":
                operator = np.asarray(operator["real"]) + 1j * np.asarray(operator["imag"])
            else:
                raise TypeError(f"Unknown serialized Port operator kind: {operator.get('kind')!r}")
        return cls(
            target,
            rate=data.get("rate"),
            external_quality_factor=data.get("external_quality_factor"),
            operator=operator,
            phase=data.get("phase", 0.0),
            label=data.get("label"),
        )
