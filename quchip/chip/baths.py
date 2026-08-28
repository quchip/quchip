"""Chip-level baths — shared / collective Lindblad dissipation.

A :class:`Bath` is **not** a device: it owns no Hilbert-space factor and no
Hamiltonian term. It owns only collapse operators that couple a *set* of
devices to a common environment.
This is the layer for physics that lives *around* devices: a single chip
temperature (every device thermalizes at it) or correlated/collective
dissipation (collective decay, correlated dephasing) that per-device noise —
independent by construction — cannot express.

Rates are in 1/ns (the Lindblad convention; no 2π scaling — that boundary is
Hamiltonian-only). The thermal Bose factor uses ``k_B`` in GHz/mK, so
``n̄ = 1 / expm1(freq / (k_B * T))`` with ``freq`` in GHz and ``T`` in mK.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from quchip.declarative.dissipation import CollapseChannel, normalize_dissipation
from quchip.declarative.expr import ParameterNamespace, PhysicsExpr
from quchip.declarative.ops import LocalOps
from quchip.declarative.parameters import Parameter
from quchip.devices.spaces import FockSpace
from quchip.utils.constants import k_B
from quchip.utils.jax_utils import maybe_concrete_scalar
from quchip.utils.labeling import auto_label, resolve_label

if TYPE_CHECKING:
    from quchip.chip.chip import Chip

_BATH_MODELS = ("thermal", "collective_decay", "correlated_dephasing")


def _bose_occupation(temperature: Any, frequency: Any) -> Any:
    """Traceable Bose occupation with the physical zero-temperature limit."""
    is_zero = temperature == 0
    safe_denominator = jnp.where(is_zero, 1.0, k_B * temperature)
    finite = 1.0 / jnp.expm1(frequency / safe_denominator)
    return jnp.where(is_zero, 0.0, finite)


class Bath:
    """A shared environment coupling a set of devices to a common bath.

    Attach at construction (``Chip(..., baths=[...])``) or at any time
    after via :meth:`~quchip.chip.chip.Chip.add_bath` — the next
    simulate/solve collects the bath's collapse operators automatically.

    Parameters
    ----------
    recipe : str
        Built-in collapse-channel model. One of ``"thermal"``,
        ``"collective_decay"``, or ``"correlated_dephasing"``. The argument
        name is retained for API and serialization compatibility.
    targets : list[BaseDevice | str] | None
        Devices the bath couples to (objects or labels). ``None`` (default)
        means *every* device in the chip — natural for a global thermal bath.
    temperature : float | None
        Bath temperature in mK (required for ``"thermal"``). May be a JAX
        tracer for sweeps / gradients.
    rate : float | None
        Bath–device coupling rate γ in 1/ns. For ``"thermal"`` it is the
        environmental coupling rate (explicit — never silently borrowed from a
        device ``T1``, so it cannot double-count device-level noise). For the
        collective models it is the overall jump rate. ``None`` defaults to
        ``1.0`` (user controls the absolute scale elsewhere).
    correlated : bool
        ``"thermal"`` only: ``False`` (default) emits independent per-device
        channels sharing one temperature. ``True`` is unsupported and raises
        :class:`NotImplementedError`. The collective models always emit a single correlated
        operator regardless of this flag.
    label : str | None
        Auto-generated ``"bath_{n}"`` when omitted.

    Examples
    --------
    >>> from quchip import DuffingTransmon, Chip, Bath
    >>> q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    >>> chip = Chip([q])
    >>> _ = chip.add_bath(Bath("thermal", temperature=20.0))  # global 20 mK bath
    >>> _ = chip.add_bath(Bath("collective_decay", targets=[q], rate=0.01))
    """

    _type_prefix = "bath"
    _parameter_names = ("temperature", "rate")

    def __init__(
        self,
        recipe: str,
        targets: list[Any] | None = None,
        *,
        temperature: Any = None,
        rate: Any = None,
        correlated: bool = False,
        label: str | None = None,
    ) -> None:
        if recipe not in _BATH_MODELS:
            raise ValueError(
                f"Unknown bath model {recipe!r}. Expected one of {_BATH_MODELS}."
            )
        if recipe == "thermal" and temperature is None:
            raise ValueError("The 'thermal' bath model requires a temperature (mK).")
        if recipe == "thermal" and correlated:
            raise NotImplementedError(
                "Collective thermal baths are unsupported; use correlated=False "
                "(independent channels sharing one temperature)."
            )
        self.recipe = recipe
        self._targets = targets
        self.temperature = temperature
        self.rate = rate
        self.label = label if label is not None else auto_label(self._type_prefix)

    def __setattr__(self, name: str, value: Any) -> None:
        """Reject a concrete negative ``temperature`` or ``rate`` (construction and later writes).

        Mirrors the concrete-only validation
        :class:`~quchip.devices.base.BaseDevice` and
        :class:`~quchip.declarative.models.CouplingModel` run on their own
        fields (checks apply to concrete scalars only; a traced value passes
        unchecked). Without this, a negative Bose occupation or a
        invalid Lindblad rate in :meth:`collapse_channels` is reachable
        from a raw ``bath.temperature = -5`` or ``bath.rate = -1``.
        """
        if name in ("temperature", "rate"):
            concrete = maybe_concrete_scalar(value)
            if concrete is not None and concrete < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        super().__setattr__(name, value)

    def resolve_targets(self, chip: "Chip") -> list[str]:
        """Return the ordered target device labels (defaults to all devices)."""
        if self._targets is None:
            return [d.label for d in chip.devices]
        return [resolve_label(t) for t in self._targets]

    @property
    def separable(self) -> bool:
        """Whether this bath factorizes into independent per-target channels.

        ``True`` for models that emit one collapse operator per target
        (``"thermal"`` with independent channels); ``False`` for models that
        emit a single jump operator summed over targets (``"collective_decay"``,
        ``"correlated_dephasing"``). Partitioning treats a non-separable bath's
        target set as one inseparable block.
        """
        return self.recipe == "thermal"

    def __repr__(self) -> str:
        """Return a compact bath-model and target summary."""
        targets = "all" if self._targets is None else [resolve_label(t) for t in self._targets]
        return (
            f"Bath(label={self.label!r}, model={self.recipe!r}, "
            f"temperature={self.temperature}, rate={self.rate}, targets={targets})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the bath; ``targets`` are stored as label strings."""
        targets = None if self._targets is None else [resolve_label(t) for t in self._targets]
        return {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "recipe": self.recipe,
            "targets": targets,
            "temperature": self.temperature,
            "rate": self.rate,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Bath":
        """Reconstruct a bath from serialized state (targets as label strings)."""
        return cls(
            d["recipe"],
            targets=d.get("targets"),
            temperature=d.get("temperature"),
            rate=d.get("rate"),
            label=d.get("label"),
        )

    def physics_notes(self) -> list[str]:
        """Return human-readable declarations of this bath's model and scope.

        Mirrors :meth:`~quchip.chip.coupling_base.BaseCoupling.physics_notes`:
        one entry naming the model and its targets, plus a model-specific
        assumption a user of this bath should be aware of.
        """
        targets = "all devices" if self._targets is None else ", ".join(resolve_label(t) for t in self._targets)
        notes = [f"Bath model: '{self.recipe}'; targets: {targets}."]
        if self.recipe == "thermal":
            notes.append(
                "Independent per-target thermal channels sharing one bath temperature; "
                "correlated=True is unsupported."
            )
        elif self.recipe == "collective_decay":
            notes.append(
                "Single shared rank-one collective channel L = sum_i a_i with a separate rate "
                "(equal-phase, equal-weight; not general super/subradiant decay)."
            )
        else:
            notes.append(
                "Single shared common-mode dephasing channel L = sum_i n_i with a separate rate "
                "(maximally correlated; not a general target-dependent correlation structure)."
            )
        return notes

    def copy(self) -> "Bath":
        """Independent copy of this bath (targets normalize to label strings).

        Used by ``Chip.clone`` and ``eliminate`` so a transformed chip never
        shares live ``Bath`` objects with its source — mutating one chip's
        bath must not silently change another chip's physics. Parameter
        values (temperature, rate) are carried by reference, so traced
        values stay traced.
        """
        return Bath.from_dict(self.to_dict())

    def parameter_values(self) -> dict[str, Any]:
        """Return active bath values by local field name."""
        return {
            name: value
            for name in self._parameter_names
            if (value := getattr(self, name)) is not None
        }

    def set_parameter_value(self, name: str, value: Any) -> None:
        """Apply one bath value on an isolated bath copy."""
        if name not in self._parameter_names:
            raise KeyError(name)
        setattr(self, name, value)

    def _resolved_bases(self, chip: "Chip", bases: Mapping[str, Any] | None) -> Mapping[str, Any]:
        """Return supplied engine bases or resolve them for direct inspection."""
        if bases is not None:
            return bases
        from quchip.engine.basis import resolve_device_basis

        return {
            device.label: resolve_device_basis(
                device,
                basis=chip.resolve_basis(device),
                levels=(
                    device.resolved_dimension(chip.basis)
                    if chip.resolve_basis(device) == "eigen"
                    else None
                ),
            )
            for device in chip.devices
        }

    @staticmethod
    def _semantic_operator(record: Any, kind: str, xp: Any) -> Any:
        """Express an energy-ordered lowering or number operator in authored coordinates."""
        dimension = record.resolved_dim
        if kind == "lowering":
            semantic = xp.diag(xp.sqrt(xp.arange(1, dimension)), 1).astype(complex)
        else:
            semantic = xp.diag(xp.arange(dimension)).astype(complex)
        vectors = xp.asarray(record.energy_vectors)
        return vectors @ semantic @ vectors.conj().T

    def _operator_expr(
        self,
        device: Any,
        record: Any,
        kind: str,
        xp: Any,
    ) -> PhysicsExpr:
        """Author a bath operator in the device's declared local coordinates."""
        space = device.local_space()
        if isinstance(space, FockSpace):
            op = LocalOps(label=device.label, space=space, device=device)
            if kind == "lowering":
                return op.a
            if kind == "raising":
                return op.adag
            return op.n

        semantic_kind = "lowering" if kind == "raising" else kind
        matrix = self._semantic_operator(record, semantic_kind, xp)
        if kind == "raising":
            matrix = matrix.conj().T
        return PhysicsExpr.from_matrix(
            matrix,
            labels=(device.label,),
            dims=(record.native_dim,),
            name=rf"\hat L_{{{self.label},{device.label}}}",
        )

    def dissipation(
        self,
        chip: "Chip",
        bases: Mapping[str, Any] | None = None,
    ) -> tuple[CollapseChannel, ...]:
        """Return authored full-chip collapse channels.

        ``"thermal"`` emits independent per-target relaxation/absorption
        pairs sharing one bath temperature (:meth:`_bose`). The two
        collective models instead each emit a single jump operator summed
        over the resolved targets:

        - ``"collective_decay"``: ``L = sum_i a_i`` at rate ``gamma`` — an
          equal-phase, equal-weight rank-one collective channel, *not*
          general collective (super/subradiant) decay, which requires
          per-pair phase and weight factors set by the target geometry
          (Lehmberg, *Phys. Rev. A* **2**, 883 (1970), for the general
          collective-radiative-decay construction).
        - ``"correlated_dephasing"``: ``L = sum_i n_i`` at rate ``gamma`` —
          maximally correlated common-mode dephasing (every target shares
          the identical dephasing fluctuation), *not* general correlated
          dephasing with a target-dependent correlation structure (Breuer &
          Petruccione, *The Theory of Open Quantum Systems*, Oxford, 2002,
          Ch. 3, for the general Lindblad construction).

        Contributions remain backend-neutral; the engine projects and lowers
        them with the same basis records used for Hamiltonian terms.
        """
        xp = jnp
        labels = self.resolve_targets(chip)
        records = self._resolved_bases(chip, bases)
        terms: list[CollapseChannel] = []
        fields = {name: Parameter() for name in self._parameter_names}
        p = ParameterNamespace(f"bath.{self.label}", fields)
        gamma = 1.0 if self.rate is None else p.rate
        chip_labels = tuple(device.label for device in chip.devices)

        if self.recipe == "thermal":
            for lbl in labels:
                dev = chip[lbl]
                n_bar = PhysicsExpr.from_function(
                    _bose_occupation,
                    p.temperature,
                    PhysicsExpr.literal(dev.freq),  # type: ignore[attr-defined]  # BaseDevice contract
                    labels=(),
                    dims=(),
                    name="n_bar",
                )
                lowering = self._operator_expr(dev, records[lbl], "lowering", xp)
                raising = self._operator_expr(dev, records[lbl], "raising", xp)
                terms.append(
                    CollapseChannel(
                        lowering.embed(chip_labels, chip.authored_dims),
                        gamma * (n_bar + 1.0),
                        f"thermal_emission:{lbl}",
                    )
                )
                terms.append(
                    CollapseChannel(
                        raising.embed(chip_labels, chip.authored_dims),
                        gamma * n_bar,
                        f"thermal_absorption:{lbl}",
                    )
                )
            return tuple(terms)

        # Collective models: a single summed jump operator over the targets.
        summed: PhysicsExpr | None = None
        for lbl in labels:
            device = chip[lbl]
            record = records[lbl]
            kind = "lowering" if self.recipe == "collective_decay" else "number"
            local = self._operator_expr(device, record, kind, xp)
            embedded = local.embed(chip_labels, chip.authored_dims)
            summed = embedded if summed is None else summed + embedded
        if summed is not None:
            terms.append(CollapseChannel(summed, gamma, self.recipe))
        return tuple(terms)

    def _collapse_channels_with_paths(
        self,
        chip: "Chip",
        bases: Mapping[str, Any] | None = None,
    ) -> tuple[tuple[CollapseChannel, tuple[str, ...]], ...]:
        fields = {name: Parameter() for name in self._parameter_names}
        return normalize_dissipation(
            self.dissipation(chip, bases),
            labels=tuple(device.label for device in chip.devices),
            dims=chip.authored_dims,
            owner=self,
            scope=f"bath.{self.label}",
            allowed=fields,
            bindings={
                f"bath.{self.label}.{name}": value
                for name in fields
                if (value := getattr(self, name)) is not None
            },
        )

    def collapse_channels(
        self,
        chip: "Chip",
        bases: Mapping[str, Any] | None = None,
    ) -> tuple[CollapseChannel, ...]:
        """Return normalized full-chip bath channels."""
        return tuple(
            channel
            for channel, _paths in self._collapse_channels_with_paths(chip, bases)
        )
