"""Chip-level baths for shared or collective Lindblad dissipation.

A :class:`Bath` is **not** a device. It owns no Hilbert-space factor and no
Hamiltonian term, only collapse operators that couple a *set* of devices to a
common environment. This layer is for physics *around* devices, e.g. a single
chip temperature at which every device thermalizes. It also covers correlated
or collective dissipation (collective decay, correlated dephasing), which
per-device noise, independent by construction, cannot express.

Rates are in 1/ns (the Lindblad convention), with no 2π scaling, because that
boundary applies only to the Hamiltonian. The thermal Bose factor uses ``k_B``
in GHz/mK, so ``n̄ = 1 / expm1(freq / (k_B * T))`` with ``freq`` in GHz and
``T`` in mK.
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
from quchip.utils.values import copy_value

if TYPE_CHECKING:
    from quchip.chip.chip import Chip

_BATH_MODELS = ("thermal", "collective_decay", "correlated_dephasing")


def _bose_occupation(temperature: Any, frequency: Any) -> Any:
    """Traceable Bose occupation with the physical zero-temperature limit."""
    is_zero = temperature == 0
    safe_denominator = jnp.where(is_zero, 1.0, k_B * temperature)
    finite = 1.0 / jnp.expm1(frequency / safe_denominator)
    return jnp.where(is_zero, 0.0, finite)


def _adjoint(operator: Any) -> Any:
    return operator.dag() if hasattr(operator, "dag") else jnp.asarray(operator).conj().T


class Bath:
    """A shared environment that couples a set of devices to a common bath.

    Attach the bath at construction (``Chip(..., baths=[...])``) or later with
    :meth:`~quchip.chip.chip.Chip.add_bath`. The next simulate or solve call
    collects the bath's collapse operators automatically.

    Parameters
    ----------
    recipe : str
        Built-in collapse-channel model: ``"thermal"``, ``"collective_decay"``,
        or ``"correlated_dephasing"``. The argument name stays the same for API
        and serialization compatibility.
    targets : list[BaseDevice | str] | None
        Devices the bath couples to (objects or labels). ``None`` (default)
        means *every* device in the chip, the natural choice for a global
        thermal bath.
    temperature : float | None
        Temperature in mK, required for ``"thermal"``. It can be a JAX tracer
        for sweeps or gradients.
    rate : float | None
        Bath–device coupling rate γ in 1/ns. For ``"thermal"``, it is the
        environmental coupling rate. It is explicit and never silently borrowed
        from a device ``T1``, so it cannot double-count device-level noise. For
        the collective models, it is the overall jump rate. ``None`` defaults
        to ``1.0`` (the user controls the absolute scale elsewhere).
    correlated : bool
        ``"thermal"`` only. ``False`` (default) emits independent per-device channels that
        share one temperature. ``True`` is not supported and raises
        :class:`NotImplementedError`. The collective models always emit a single correlated
        operator, independent of this flag.
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
        self._retained: dict[str, tuple[Any, Any, Any]] | None = None
        self.recipe = recipe
        self._targets = targets
        self.temperature = temperature
        self.rate = rate
        self.label = label if label is not None else auto_label(self._type_prefix)

    def __setattr__(self, name: str, value: Any) -> None:
        """Reject a concrete negative ``temperature`` or ``rate`` (construction and later writes).

        Mirrors the concrete-only validation that
        :class:`~quchip.devices.base.BaseDevice` and
        :class:`~quchip.declarative.models.CouplingModel` run on their own
        fields, so a traced value passes unchecked.
        """
        # Without this validation, a raw `bath.temperature = -5` or
        # `bath.rate = -1` can cause a negative Bose occupation or an invalid
        # Lindblad rate in `collapse_channels`.
        if name in ("temperature", "rate"):
            concrete = maybe_concrete_scalar(value)
            if concrete is not None and concrete < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        super().__setattr__(name, value)

    def resolve_targets(self, chip: "Chip") -> list[str]:
        """Return ordered target labels, defaulting to all devices.

        Parameters
        ----------
        chip : Chip
            Chip that supplies labels when targets were omitted.
        """
        if self._targets is None or self._retained is not None:
            return [d.label for d in chip.devices]
        return [resolve_label(t) for t in self._targets]

    @property
    def separable(self) -> bool:
        """Whether this bath factorizes into independent per-target channels.

        ``True`` for models that emit one collapse operator per target
        (``"thermal"`` with independent channels). ``False`` for models that
        emit a single jump operator summed over targets (``"collective_decay"``,
        ``"correlated_dephasing"``). Partitioning treats a non-separable bath's
        target set as one inseparable block.
        """
        return self.recipe == "thermal" and self._retained is None

    def __repr__(self) -> str:
        """Return a compact bath-model and target summary."""
        targets = "all" if self._targets is None else [resolve_label(t) for t in self._targets]
        return (
            f"Bath(label={self.label!r}, model={self.recipe!r}, "
            f"temperature={self.temperature}, rate={self.rate}, targets={targets})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the bath, with ``targets`` stored as label strings."""
        targets = None if self._targets is None else [resolve_label(t) for t in self._targets]
        return {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "retained": None if self._retained is None else {
                label: [float(freq), *[dict(real=jnp.asarray(op).real.tolist(), imag=jnp.asarray(op).imag.tolist())
                                       for op in (lowering, number)]]
                for label, (freq, lowering, number) in self._retained.items()
            },
            "recipe": self.recipe,
            "targets": targets,
            "temperature": self.temperature,
            "rate": self.rate,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Bath":
        """Reconstruct a bath from serialized state.

        Parameters
        ----------
        d : dict[str, Any]
            Payload produced by :meth:`to_dict`. Targets are label strings.
        """
        bath = cls(
            d["recipe"],
            targets=d.get("targets"),
            temperature=d.get("temperature"),
            rate=d.get("rate"),
            label=d.get("label"),
        )
        if d.get("retained") is not None:
            bath._retained = {}
            for label, entry in d["retained"].items():
                if not isinstance(label, str) or not label or len(entry) != 3:
                    raise ValueError("A retained bath target needs a label, frequency and two operators.")
                frequency, *operators = entry
                matrices = []
                for operator in operators:
                    real, imag = jnp.asarray(operator["real"]), jnp.asarray(operator["imag"])
                    if real.ndim != 2 or real.shape != imag.shape or real.shape[0] != real.shape[1]:
                        raise ValueError("Retained bath operators need square, matching real/imag matrices.")
                    if not bool(jnp.all(jnp.isfinite(real) & jnp.isfinite(imag))):
                        raise ValueError("Retained bath operators must be finite.")
                    matrices.append(real + 1j * imag)
                if not bool(jnp.isfinite(frequency)) or frequency < 0:
                    raise ValueError("Retained bath frequencies must be finite and nonnegative.")
                bath._retained[label] = (frequency, matrices[0], matrices[1])
        return bath

    def physics_notes(self) -> list[str]:
        """Return human-readable declarations of this bath's model and scope.

        Mirrors :meth:`~quchip.chip.coupling_base.BaseCoupling.physics_notes`.
        Returns one entry naming the model and its targets, plus a
        model-specific assumption that users must know.
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

        ``Chip.clone`` and ``eliminate`` use this method, so a transformed chip
        never shares live ``Bath`` objects with its source. Parameter values
        (temperature, rate) are carried by reference, so traced values stay
        traced.
        """
        # Changing one chip's bath must not silently change another chip's
        # physics.
        result = Bath(self.recipe, None if self._targets is None else [resolve_label(t) for t in self._targets],
                      temperature=copy_value(self.temperature), rate=copy_value(self.rate), label=self.label)
        result._retained = copy_value(self._retained, readonly=True)
        return result

    def parameter_values(self) -> dict[str, Any]:
        """Return active bath values by local field name."""
        return {
            name: value
            for name in self._parameter_names
            if (value := getattr(self, name)) is not None
        }

    def set_parameter_value(self, name: str, value: Any) -> None:
        """Apply one bath value on an isolated bath copy.

        Parameters
        ----------
        name : {"temperature", "rate"}
            Local parameter name.
        value : Any
            Temperature in mK or rate in 1/ns.
        """
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

        ``"thermal"`` emits independent per-target relaxation/absorption pairs
        that share one bath temperature. The two collective models each emit a
        single jump operator summed over the resolved targets:

        - ``"collective_decay"``: ``L = sum_i a_i`` at rate ``gamma``, an
          equal-phase, equal-weight rank-one collective channel. It is *not*
          general collective (super/subradiant) decay, which requires per-pair
          phase and weight factors set by the target geometry (Lehmberg, *Phys.
          Rev. A* **2**, 883 (1970), for the general collective-radiative-decay
          construction).
        - ``"correlated_dephasing"``: ``L = sum_i n_i`` at rate ``gamma``, i.e.
          maximally correlated common-mode dephasing, where every target shares
          the identical dephasing fluctuation. It is *not* general correlated
          dephasing with a target-dependent correlation structure (Breuer &
          Petruccione, *The Theory of Open Quantum Systems*, Oxford, 2002, Ch.
          3, for the general Lindblad construction).

        Contributions stay backend-neutral.

        Parameters
        ----------
        chip : Chip
            Chip whose targets and tensor-product order define the operators.
        bases : mapping or None, default=None
            Captured basis records keyed by device label. ``None`` resolves them.
        """
        # The thermal Bose occupation is computed in `_bose`.
        #
        # The engine projects and lowers them with the same basis records it
        # uses for Hamiltonian terms.
        fields = {name: Parameter() for name in self._parameter_names}
        p = ParameterNamespace(f"bath.{self.label}", fields)
        gamma = 1.0 if self.rate is None else p.rate
        terms: list[CollapseChannel] = []
        summed: PhysicsExpr | None = None
        for label, frequency, lowering, number in self._target_operators(chip, bases):
            if self.recipe == "thermal":
                n_bar = PhysicsExpr.from_function(
                    _bose_occupation, p.temperature, PhysicsExpr.literal(frequency),
                    labels=(), dims=(), name="n_bar",
                )
                terms.extend((
                    CollapseChannel(lowering, gamma * (n_bar + 1), f"thermal_emission:{label}"),
                    CollapseChannel(PhysicsExpr.from_function(
                        _adjoint, lowering, labels=lowering.labels, dims=chip.authored_dims, name="raising",
                    ), gamma * n_bar, f"thermal_absorption:{label}"),
                ))
            else:
                operator = lowering if self.recipe == "collective_decay" else number
                summed = operator if summed is None else summed + operator
        if summed is not None:
            terms.append(CollapseChannel(summed, gamma, self.recipe))
        return tuple(terms)

    def _target_operators(self, chip: "Chip", bases: Mapping[str, Any] | None = None) -> Any:
        """Yield each bath target before summing collective amplitudes or applying rates."""
        labels, dims = tuple(d.label for d in chip.devices), chip.authored_dims
        if self._retained is not None:
            for label, (frequency, lowering, number) in self._retained.items():
                if label in chip.device_map:
                    frequency = getattr(chip[label], "freq", frequency)
                yield (label, frequency, *[
                    PhysicsExpr.from_matrix(op, labels=labels, dims=dims, name=f"{self.label}:{label}")
                    for op in (lowering, number)
                ])
            return
        records = self._resolved_bases(chip, bases)
        for label in self.resolve_targets(chip):
            device = chip[label]
            yield (label, getattr(device, "freq", 0.0), *[
                self._operator_expr(device, records[label], kind, jnp).embed(labels, dims)
                for kind in ("lowering", "number")
            ])

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
        """Return normalized full-chip bath channels.

        Parameters
        ----------
        chip : Chip
            Chip whose targets define the channels.
        bases : mapping or None, default=None
            Captured basis records, or ``None`` to resolve them.
        """
        return tuple(
            channel
            for channel, _paths in self._collapse_channels_with_paths(chip, bases)
        )
