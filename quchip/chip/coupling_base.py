"""Coupling base class and registry.

Split out of :mod:`quchip.chip.couplings` so the declarative API in
:mod:`quchip.declarative.models` can subclass :class:`BaseCoupling`
without re-entering the import chain through ``Capacitive``
(:class:`Capacitive` now itself depends on :class:`CouplingModel`).
The concrete coupling classes — :class:`Capacitive`,
:class:`TunableCapacitive`, :class:`Coupling` — still live in
:mod:`quchip.chip.couplings`.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from quchip.backend.protocol import Operator
from quchip.devices.base import BaseDevice
from quchip.utils.labeling import auto_label, resolve_label
from quchip.utils.registry import Registrable
from quchip.utils.state_versioning import StateVersioned

if TYPE_CHECKING:
    from quchip.chip.chip import Chip
    from quchip.engine.ir import DroppedTerm, ScalarModulation


class BaseCoupling(StateVersioned, Registrable, ABC, registry_root=True):
    """Abstract base for a two-body coupling between :class:`BaseDevice`s.

    Subclasses own their local interaction Hamiltonian ``H_int`` acting on
    ``H_a ⊗ H_b``. The engine embeds this local form into the full chip
    tensor space; couplings never touch the engine directly.

    Subclasses auto-register via the shared
    :class:`~quchip.utils.registry.Registrable` mixin — no manual
    registration step is needed. Ensure the module defining the subclass is
    imported so the registration runs.

    Parameters
    ----------
    device_a, device_b : BaseDevice or str
        The two coupled devices, given as device objects or their label
        strings. Label-string references are *late-bound*: the coupling
        remembers the string, and :class:`Chip`
        resolves it to the matching device instance at construction time.
        Before that resolution the coupling cannot produce an interaction
        Hamiltonian.
    label : str, optional
        Human-readable label. Auto-generated from ``_type_prefix`` when
        omitted (e.g. ``"cap_0"`` for :class:`Capacitive`).

    Notes
    -----
    All coupling parameters (``g``, any tunable envelope, etc.) must be
    JAX-traceable so sweeps and gradient-based optimization work without
    forcing concretization.
    """

    _type_prefix: str = "coupling"

    # Whether an existing edge of this coupling type is a valid fold target for
    # elimination's mediated-exchange consolidation
    # (:mod:`quchip.chip.transformations.eliminate_device`). A scalar
    # ``coupling_strength`` alone does not prove an edge is exchange-compatible
    # — folding a mediated exchange into e.g. a dispersive ``g · n̂_a n̂_b``
    # coupling would silently change its physics. True only for couplings
    # whose full interaction is the dipole-dipole ``(a + a†)(b + b†)`` form
    # (:class:`~quchip.chip.couplings.Capacitive`,
    # :class:`~quchip.chip.couplings.TunableCapacitive`); an edge that does not
    # declare this is left unchanged and the mediated exchange is added as its
    # own parallel edge instead.
    folds_exchange: bool = False

    # Whether this coupling's exchange physics reduces to a dispersive
    # CrossKerr shift under coupling-target elimination
    # (:mod:`quchip.chip.transformations.eliminate_coupling`). The reduction
    # reads the chip's exact dressed spectrum and assumes an exchange-like
    # interaction — a coupling whose physics is not of that form (e.g. a
    # longitudinal or user-supplied :class:`~quchip.chip.couplings.Coupling`
    # interaction) would silently mis-model as a CrossKerr, so only couplings
    # declaring this capability are eligible. True for
    # :class:`~quchip.chip.couplings.Capacitive` and
    # :class:`~quchip.chip.couplings.TunableCapacitive`.
    reduces_to_crosskerr: bool = False

    # The endpoint device references are structural — a rebinding during
    # ``copy()`` is not a physics change — so they must not bump state_version.
    # Mutation tracking, the seed, ``state_version`` and ``_finish_init`` are
    # owned by the shared StateVersioned mixin; tracking switches on
    # automatically once construction finishes (no manual ``_finish_init``).
    _untracked_names = frozenset({"device_a", "device_b"})

    def __init__(
        self,
        device_a: BaseDevice | str,
        device_b: BaseDevice | str,
        *,
        label: str | None = None,
    ) -> None:
        """Store endpoint references and assign a stable coupling label."""
        for name, dev in (("device_a", device_a), ("device_b", device_b)):
            if not isinstance(dev, (BaseDevice, str)):
                raise TypeError(
                    f"{name} must be a BaseDevice or label string, got {type(dev).__name__}."
                )
        self.device_a = device_a
        self.device_b = device_b
        self._rwa: bool | None = None
        self.label = label if label is not None else auto_label(type(self)._type_prefix)

    def copy(self, device_map: dict[str, BaseDevice]) -> "BaseCoupling":
        """Shallow copy rebound to the device instances in *device_map*."""
        cloned = copy.copy(self)
        object.__setattr__(cloned, "device_a", device_map[self.device_a_label])
        object.__setattr__(cloned, "device_b", device_map[self.device_b_label])
        return cloned

    @property
    def device_a_label(self) -> str:
        """Label of the first coupled device (works pre- and post-binding)."""
        return resolve_label(self.device_a)

    @property
    def device_b_label(self) -> str:
        """Label of the second coupled device (works pre- and post-binding)."""
        return resolve_label(self.device_b)

    @property
    def is_resolved(self) -> bool:
        """Whether both device references are bound to :class:`BaseDevice` instances."""
        return isinstance(self.device_a, BaseDevice) and isinstance(self.device_b, BaseDevice)

    def __repr__(self) -> str:
        """Return a minimal endpoint summary (default for full-control subclasses)."""
        return f"{type(self).__name__}('{self.device_a_label}' <-> '{self.device_b_label}', label={self.label!r})"

    def _resolve_devices(self, device_map: dict[str, BaseDevice]) -> None:
        """Bind pending label-string references to device instances."""
        for attr in ("device_a", "device_b"):
            current = getattr(self, attr)
            if isinstance(current, str):
                resolved = device_map.get(current)
                if resolved is None:
                    raise ValueError(
                        f"Coupling {self!r} references device {current!r} "
                        f"which is not in the device list. "
                        f"Available labels: {list(device_map.keys())}"
                    )
                object.__setattr__(self, attr, resolved)

    @property
    @abstractmethod
    def coupling_strength(self) -> float:
        """Scalar coupling strength in GHz."""
        ...

    @property
    def coupling_strength_name(self) -> str:
        """Display name of the scalar :attr:`coupling_strength` parameter.

        Default ``"g"`` (the conventional coupling-strength symbol, and the
        actual attribute name on :class:`~quchip.chip.couplings.Coupling`).
        :class:`~quchip.declarative.models.CouplingModel` overrides this to
        the name of its first declared parameter field; a subclass with a
        different primary-scalar convention overrides it directly.
        """
        return "g"

    def set_coupling_strength(self, value: Any) -> None:
        """Write *value* into the scalar named by :attr:`coupling_strength_name`.

        The mutation counterpart of :attr:`coupling_strength`: callers that
        need to move a coupling's primary scalar (optimizers, sweeps) go
        through this seam instead of assuming an attribute name (``g``
        holds only for :class:`~quchip.chip.couplings.Capacitive` /
        :class:`~quchip.chip.couplings.Coupling`; :class:`~quchip.chip.couplings.TunableCapacitive`
        uses ``g_0``, :class:`~quchip.chip.couplings.CrossKerr` uses ``chi``).
        Default implementation covers the common case where the writable
        attribute name matches :attr:`coupling_strength_name` exactly; a
        subclass whose writable attribute differs from its display name
        overrides this.
        """
        setattr(self, self.coupling_strength_name, value)

    @property
    def rwa(self) -> bool | None:
        """Per-coupling RWA override; ``None`` inherits the chip's default."""
        return self._rwa

    @rwa.setter
    def rwa(self, value: bool | None) -> None:
        """Set the per-coupling RWA override (``None`` inherits the chip default)."""
        self._rwa = value

    def rwa_keeps_band(self, delta_a: int, delta_b: int) -> bool:
        """Whether the RWA retains the ``(Δa, Δb)`` excitation-change band.

        The structural RWA policy for this coupling: when the chip
        resolves RWA (:meth:`Chip.resolve_rwa`), only bands accepted
        here survive — in :meth:`Chip.hamiltonian`'s mask and in stage
        2's band filter alike. Offsets follow the engine convention
        ``Δ = col − row`` (``+1`` = lowering). Overrides must stay
        symmetric under joint sign flip so the retained operator is
        Hermitian, and must depend only on the integer offsets — never
        on frequency values, which may be traced.

        Default: total-excitation-conserving (``Δa + Δb == 0``), the
        beam-splitter selection of the textbook coupling RWA.
        """
        return delta_a + delta_b == 0

    @abstractmethod
    def interaction_hamiltonian(self) -> Operator:
        """Return the full ``H_int`` on the local ``H_a ⊗ H_b`` subspace.

        Always the complete (non-RWA) interaction: the RWA is a chip
        policy the engine applies structurally, keeping only the bands
        :meth:`rwa_keeps_band` accepts when :meth:`Chip.resolve_rwa`
        resolves ``True`` for this coupling. Couplings author one form.
        """
        ...

    def physics_notes(self) -> list[str]:
        """Return human-readable declarations of this coupling's approximations."""
        return [
            f"Coupled devices: '{self.device_a_label}' ↔ '{self.device_b_label}'",
            f"RWA policy: {self.rwa if self.rwa is not None else 'inherits chip default'}",
        ]

    def parametric_operator(self, chip: Any) -> Any | None:
        """Backend operator a scheduled edge pump multiplies, or ``None`` when this coupling is not modulable.

        The base coupling is static-only; declarative couplings opt in by
        implementing :meth:`CouplingModel.parametric_interaction`.
        """
        _ = chip
        return None

    def dropped_terms(self) -> list["DroppedTerm"]:
        """Return advisory records for terms this coupling's model itself elides.

        RWA band drops are reported generically by stage 2; this hook is
        for *other* approximations a coupling applies inside
        :meth:`interaction_hamiltonian`. Default: nothing is dropped.
        """
        return []

    def dynamic_interaction_terms(self, chip: "Chip") -> list[tuple[Operator, "ScalarModulation"]]:
        """Time-dependent interaction terms (default: none)."""
        return []

    def collapse_operators(self, chip: "Chip") -> list[Operator]:
        """Lindblad collapse operators on the local subspace (default: none)."""
        _ = chip
        return []

    def to_dict(self) -> dict[str, Any]:
        """Serialize structural fields into a JSON-safe dictionary."""
        data = super().to_dict()
        data["device_a_label"] = self.device_a_label
        data["device_b_label"] = self.device_b_label
        data["label"] = self.label
        return data
