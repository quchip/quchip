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

from typing import TYPE_CHECKING, Any

from quchip.backend.protocol import Operator
from quchip.utils.constants import k_B
from quchip.utils.jax_utils import maybe_concrete_scalar
from quchip.utils.labeling import auto_label, resolve_label

if TYPE_CHECKING:
    from quchip.chip.chip import Chip

_RECIPES = ("thermal", "collective_decay", "correlated_dephasing")


class Bath:
    """A shared environment coupling a set of devices to a common bath.

    Attach at construction (``Chip(..., baths=[...])``) or at any time
    after via :meth:`~quchip.chip.chip.Chip.add_bath` — the next
    simulate/solve collects the bath's collapse operators automatically.

    Parameters
    ----------
    recipe : str
        One of ``"thermal"``, ``"collective_decay"``, ``"correlated_dephasing"``.
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
        collective recipes it is the overall jump rate. ``None`` defaults to
        ``1.0`` (user controls the absolute scale elsewhere).
    correlated : bool
        ``"thermal"`` only: ``False`` (default) emits independent per-device
        channels sharing one temperature. ``True`` is reserved for a genuinely
        collective thermal jump operator and currently raises
        :class:`NotImplementedError` — it is a documented future refinement, not
        a silent no-op. The collective recipes always emit a single correlated
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
        if recipe not in _RECIPES:
            raise ValueError(f"Unknown bath recipe {recipe!r}. Expected one of {_RECIPES}.")
        if recipe == "thermal" and temperature is None:
            raise ValueError("The 'thermal' recipe requires a temperature (mK).")
        if recipe == "thermal" and correlated:
            raise NotImplementedError(
                "Collective thermal baths are not yet implemented; use correlated=False "
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
        ``sqrt(negative)`` NaN in :meth:`collapse_operators` is reachable
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

        ``True`` for recipes that emit one collapse operator per target
        (``"thermal"`` with independent channels); ``False`` for recipes that
        emit a single jump operator summed over targets (``"collective_decay"``,
        ``"correlated_dephasing"``). Partitioning treats a non-separable bath's
        target set as one inseparable block.
        """
        return self.recipe == "thermal"

    def __repr__(self) -> str:
        """Return a compact recipe / target summary."""
        targets = "all" if self._targets is None else [resolve_label(t) for t in self._targets]
        return (
            f"Bath(label={self.label!r}, recipe={self.recipe!r}, "
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
        """Return human-readable declarations of this bath's recipe and scope.

        Mirrors :meth:`~quchip.chip.coupling_base.BaseCoupling.physics_notes`:
        one entry naming the recipe and its targets, plus a recipe-specific
        assumption a user of this bath should be aware of.
        """
        targets = "all devices" if self._targets is None else ", ".join(resolve_label(t) for t in self._targets)
        notes = [f"Recipe: '{self.recipe}'; targets: {targets}."]
        if self.recipe == "thermal":
            notes.append(
                "Independent per-target thermal channels sharing one bath temperature "
                "(correlated=True — a genuinely collective thermal jump operator — is reserved "
                "and currently raises NotImplementedError)."
            )
        elif self.recipe == "collective_decay":
            notes.append(
                "Single shared rank-one collective channel L = sqrt(rate) * sum_i a_i "
                "(equal-phase, equal-weight; not general super/subradiant decay)."
            )
        else:
            notes.append(
                "Single shared common-mode dephasing channel L = sqrt(rate) * sum_i n_i "
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

    def _bose(self, freq: Any, xp: Any) -> Any:
        """Thermal occupation n̄(freq, T); JAX-safe (no branch on traced T), T=0-safe.

        ``k_B`` is ``k_B/h`` in GHz/mK, so ``freq/(k_B*T)`` is dimensionless.
        The physical ``T -> 0`` limit is ``n̄ -> 0`` (a zero-temperature bath
        carries no thermal photons), but ``T`` exactly ``0`` cannot reach that
        limit through a live division: ``k_B*T`` would be the divisor, and
        for a concrete zero-valued (plain Python or NumPy) temperature that
        raises ``ZeroDivisionError``/emits ``inf`` before ``expm1`` ever
        combines it back down to a finite value. The denominator is
        therefore replaced with a nonzero placeholder *before* the division
        runs — guarding the division's input, not just selecting between two
        already-computed branch outputs, since ``xp.where`` evaluates both
        branches and a zero divisor in the unselected branch would still
        raise or poison gradients with ``NaN`` — and the exact ``T=0``
        result (``0.0``) is selected explicitly afterward. No Python branch
        on the traced temperature either way.

        Returns
        -------
        Any
            The thermal occupation n̄, a possibly-traced scalar.
        """
        is_zero = self.temperature == 0
        safe_denominator = xp.where(is_zero, 1.0, k_B * self.temperature)
        n_bar_finite = 1.0 / xp.expm1(freq / safe_denominator)
        return xp.where(is_zero, 0.0, n_bar_finite)

    def collapse_operators(self, chip: "Chip") -> list[Operator]:
        """Fully-embedded collapse operators for this bath.

        ``"thermal"`` emits independent per-target relaxation/absorption
        pairs sharing one bath temperature (:meth:`_bose`). The two
        collective recipes instead each emit a single jump operator summed
        over the resolved targets:

        - ``"collective_decay"``: ``L = sqrt(gamma) * sum_i a_i`` — an
          equal-phase, equal-weight rank-one collective channel, *not*
          general collective (super/subradiant) decay, which requires
          per-pair phase and weight factors set by the target geometry
          (Lehmberg, *Phys. Rev. A* **2**, 883 (1970), for the general
          collective-radiative-decay construction).
        - ``"correlated_dephasing"``: ``L = sqrt(gamma) * sum_i n_i`` —
          maximally correlated common-mode dephasing (every target shares
          the identical dephasing fluctuation), *not* general correlated
          dephasing with a target-dependent correlation structure (Breuer &
          Petruccione, *The Theory of Open Quantum Systems*, Oxford, 2002,
          Ch. 3, for the general Lindblad construction).

        Always called from inside ``with _backend_context(chip.backend):`` (see
        :func:`quchip.engine.stage4_problem._collect_c_ops`), so this method must
        not open its own backend context.
        """
        backend = chip.backend
        xp = backend.array_module
        labels = self.resolve_targets(chip)
        ops: list[Operator] = []
        gamma = 1.0 if self.rate is None else self.rate

        if self.recipe == "thermal":
            for lbl in labels:
                idx = chip.device_index(lbl)
                dev = chip[lbl]
                n_bar = self._bose(dev.freq, xp)  # type: ignore[attr-defined]  # BaseDevice contract: all concrete devices expose freq
                relax = xp.sqrt(gamma * (n_bar + 1.0)) * dev.lowering_operator()
                absorb = xp.sqrt(gamma * n_bar) * dev.raising_operator()
                ops.append(backend.embed(relax, idx, chip.dims))
                ops.append(backend.embed(absorb, idx, chip.dims))
            return ops

        # Collective recipes: a single summed jump operator over the targets.
        summed: Operator | None = None
        for lbl in labels:
            idx = chip.device_index(lbl)
            dev = chip[lbl]
            local = dev.lowering_operator() if self.recipe == "collective_decay" else dev.number_operator()
            embedded = backend.embed(local, idx, chip.dims)
            summed = embedded if summed is None else summed + embedded
        if summed is not None:
            ops.append(xp.sqrt(gamma) * summed)
        return ops
