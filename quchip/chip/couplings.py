"""Coupling models for two-body interactions between devices.

Couplers own their local Hamiltonians. A coupling
defines the interaction Hamiltonian on the two-device subspace
``H_int ∈ L(H_a ⊗ H_b)``; the engine embeds it into the full chip space
at assembly time. Coupling strengths are in GHz.

References
----------
Blais, A., Huang, R.-S., Wallraff, A., Girvin, S. M., & Schoelkopf, R. J.
    Cavity quantum electrodynamics for superconducting electrical circuits:
    An architecture for quantum computation. PRA 69, 062320 (2004), Eq. 11.
Krantz, P., Kjaergaard, M., Yan, F., Orlando, T. P., Gustavsson, S., &
    Oliver, W. D. A quantum engineer's guide to superconducting qubits.
    Applied Physics Reviews 6, 021318 (2019), §V.
Blais, A., Grimsmo, A. L., Girvin, S. M., & Wallraff, A. Circuit quantum
    electrodynamics. Rev. Mod. Phys. 93, 025005 (2021).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, ClassVar, cast

from quchip.backend.protocol import Operator
from quchip.chip.coupling_base import BaseCoupling as _BaseCoupling
from quchip.declarative.expr import PhysicsExpr
from quchip.declarative.models import CouplingModel
from quchip.declarative.ops import EndpointOps
from quchip.declarative.parameters import UNBOUND, Scalar, parameter
from quchip.devices.base import BaseDevice

if TYPE_CHECKING:
    from quchip.backend.protocol import Backend


__all__ = ["Capacitive", "Coupling", "CrossKerr", "TunableCapacitive"]


def _get_backend() -> "Backend":
    """Resolve the active default backend at call time (lazy import)."""
    from quchip.backend import get_default_backend

    return get_default_backend()


def _full(s: Any, a: EndpointOps, b: EndpointOps) -> PhysicsExpr:
    """Full charge-charge capacitive interaction."""
    return s * a.charge * b.charge


class Capacitive(CouplingModel):
    """Capacitive (charge-charge) coupling between two devices.

    The capacitive interaction between two electromagnetic modes takes the
    canonical dipole-dipole form in the raising/lowering basis:

    - Full form:   ``H_int = g · (a + a†)(b + b†)``
    - Band-RWA form: ``H_int = g · (a†b + a b†)`` (derived, not authored)

    The coupling authors only the full form; :class:`~quchip.RWA`
    retaining the ``Δa + Δb == 0`` bands of it, which is exactly
    ``g · (a†b + a b†)``. The RWA drops the counter-rotating terms ``a b``
    and ``a† b†``, valid when ``ω_a + ω_b ≫ g`` — the sum-frequency
    condition that makes those terms fast-rotating and hence negligible.
    This is distinct from the dispersive condition ``|ω_a − ω_b| ≫ g``,
    which instead governs whether the *retained* exchange term
    ``g · (a†b + a b†)`` can be treated perturbatively (see
    :class:`TunableCapacitive` / :func:`~quchip.chip.transformations.eliminate`
    for the dispersive reduction). Approximation is selected on the chip or
    for one solve, never on the coupling.

    Parameters
    ----------
    device_a, device_b : BaseDevice or str
        The two coupled devices, given as objects or label strings.
        Label-string references are late-bound via :class:`Chip`.
    g : float
        Coupling strength in GHz. May be a traced JAX scalar for
        sweeps / autodiff.
    label : str, optional
        Human-readable label; defaults to ``"cap_{n}"``.

    References
    ----------
    Blais et al., PRA 69, 062320 (2004), Eq. 11.
    Krantz et al., Appl. Phys. Rev. 6, 021318 (2019), §V.B.
    Blais et al., Rev. Mod. Phys. 93, 025005 (2021), §II.B.

    Examples
    --------
    >>> from quchip import DuffingTransmon, Resonator, Capacitive
    >>> q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    >>> r = Resonator(freq=7.0, levels=5, label="r")
    >>> coupling = Capacitive(q, r, g=0.05)
    """

    _type_prefix: ClassVar[str] = "cap"
    folds_exchange: ClassVar[bool] = True
    reduces_to_crosskerr: ClassVar[bool] = True
    default_fit_observable: ClassVar[str] = "cross_kerr"

    g: Scalar = parameter(default=UNBOUND, unit="GHz", symbol="g")

    def default_dressed_target(self) -> tuple[str, Any]:
        """Use exchange rate for an edge between non-computational modes."""
        if (
            isinstance(self.device_a, BaseDevice)
            and isinstance(self.device_b, BaseDevice)
            and not self.device_a.computational
            and not self.device_b.computational
        ):
            return "exchange_rate", self.coupling_strength
        return super().default_dressed_target()

    def interaction(self, a: EndpointOps, b: EndpointOps, p: Any) -> PhysicsExpr:
        """Return the full capacitive interaction ``g * (a + a†)(b + b†)``."""
        return _full(p.g, a, b)

    def physics_notes(self) -> list[str]:
        """Return the declared capacitive interaction."""
        notes = super().physics_notes()
        notes.append("Interaction form: g · (a + a†)(b + b†)")
        return notes

    def __repr__(self) -> str:
        """Return a compact endpoint/coupling summary."""
        return f"Capacitive('{self.device_a_label}' <-> '{self.device_b_label}', g={self.g})"

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        device_a: BaseDevice,
        device_b: BaseDevice,
    ) -> "Capacitive":
        """Reconstruct a capacitive coupling from serialized state."""
        return cls(
            device_a=device_a,
            device_b=device_b,
            g=d["g"],
            label=cast(str, d.get("label")),
        )


class TunableCapacitive(CouplingModel):
    r"""Capacitive coupling with a scheduled parametric pump.

    Effective two-body interaction with a static mean
    coupling strength:

    .. math::
        H_{\text{int}} \;=\; g_0\,\hat Q_a\hat Q_b

    where :math:`\hat Q` is each endpoint's physical charge-like coupling
    operator (the position quadrature for a Fock model). The engine applies
    any requested RWA after local-basis materialization. :math:`g_0` is the
    static coupling strength in GHz; it may be
    a JAX tracer and flows through :func:`jax.grad` without
    concretization.

    Time-dependence is not a construction-time parameter: a
    :class:`~quchip.control.drive.ParametricDrive` wired onto this
    coupling schedules a pump δ(t) via
    :meth:`~quchip.control.sequence.QuantumSequence.pump`, multiplying
    the same operator structure the static term uses
    (:meth:`parametric_interaction`).

    Parameters
    ----------
    device_a, device_b : BaseDevice
        The two coupled devices.
    g_0 : float
        Static (mean) coupling strength in GHz. May be a JAX tracer.
    label : str, optional
        Human-readable label; defaults to ``"tunable_cap_{n}"``.

    Notes
    -----
    The pump multiplies :meth:`parametric_interaction`; frame and RWA logic
    stay in the engine. A pump
    tone at the qubits' difference frequency ``|ω_a − ω_b|`` activates the
    parametric beam-splitter / iSWAP exchange, while a tone at the sum
    frequency ``ω_a + ω_b`` instead activates two-mode-squeezing
    (``a†b†``) terms. Either is expressed via the drive's ``freq``
    argument, not a coupling-side carrier.

    References
    ----------
    McKay, Filipp, Mezzacapo, Magesan, Chow & Gambetta,
    *Universal Gate for Fixed-Frequency Qubits via a Tunable Bus*,
    Phys. Rev. Applied **6**, 064007 (2016) — parametric two-qubit
    gates via coupler flux modulation.

    Krantz et al., *A quantum engineer's guide to superconducting
    qubits*, Appl. Phys. Rev. **6**, 021318 (2019), §V.D — tunable
    couplers.

    Examples
    --------
    >>> from quchip import Chip, ControlEquipment, DuffingTransmon, ParametricDrive, TunableCapacitive
    >>> q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    >>> q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.22, levels=3, label="q1")
    >>> tc = TunableCapacitive(q0, q1, g_0=0.0, label="tc")
    >>> pump = ParametricDrive(tc, label="pump")
    >>> chip = Chip([q0, q1], couplings=[tc], control_equipment=ControlEquipment([pump]))
    >>> # seq.pump(tc, envelope=..., freq=...) schedules δg(t); see QuantumSequence.pump.
    """

    _type_prefix: ClassVar[str] = "tunable_cap"
    is_effective: ClassVar[bool] = True
    folds_exchange: ClassVar[bool] = True
    reduces_to_crosskerr: ClassVar[bool] = True
    default_fit_observable: ClassVar[str] = "cross_kerr"

    g_0: Scalar = parameter(default=UNBOUND, unit="GHz", symbol="g_0")

    def interaction(self, a: EndpointOps, b: EndpointOps, p: Any) -> PhysicsExpr:
        """Static contribution ``g_0 · (a + a†)(b + b†)``."""
        return _full(p.g_0, a, b)

    def parametric_interaction(self, a: EndpointOps, b: EndpointOps, p: Any) -> PhysicsExpr:
        """Modulable charge-charge structure a scheduled pump multiplies."""
        _ = p
        return a.charge * b.charge

    def physics_notes(self) -> list[str]:
        """Return the tunable-coupling and effective-model assumptions."""
        notes = super().physics_notes()
        notes.append("Interaction form: g_0 · (a + a†)(b + b†)")
        notes.append(
            "Effective parametric model: the physical coupler mode is eliminated; coupler "
            "leakage and mediated shifts beyond exchange are not represented — use a physical "
            "bus device plus eliminate() when they matter"
        )
        return notes

    def __repr__(self) -> str:
        """Return a compact endpoint/coupling summary."""
        return (
            f"TunableCapacitive('{self.device_a_label}' <-> '{self.device_b_label}', "
            f"g_0={self.g_0})"
        )


class CrossKerr(CouplingModel):
    """Cross-Kerr (dispersive) coupling ``H_int = χ · n̂_a n̂_b``.

    The effective diagonal interaction left when an exchange coupling is
    reduced in the dispersive regime — the natural coupling for effective
    readout chips (qubit + resonator + ``CrossKerr`` probed by an ordinary
    charge line) and static-ZZ modelling. Diagonal in both endpoints, so the
    RWA and full forms coincide and the term is frame-trivial.

    Declared approximation: this is a *uniform-pull* model —
    one χ per edge, the same shift per endpoint excitation; per-level χ
    differences, dispersive breakdown at the critical photon number, and
    Purcell decay are not represented (fold Purcell into endpoint ``T1``
    via ``eliminate()`` when it matters).

    Parameters
    ----------
    device_a, device_b : BaseDevice or str
        The two coupled devices, as objects or label strings.
    chi : float
        Cross-Kerr shift in GHz per excitation pair, sign included
        (convention: full pull ``E₁₁ − E₁₀ − E₀₁ + E₀₀``). May be a JAX
        tracer.
    label : str, optional
        Defaults to ``"crosskerr_{n}"``.
    """

    _type_prefix: ClassVar[str] = "crosskerr"
    is_effective: ClassVar[bool] = True
    default_fit_observable: ClassVar[str] = "cross_kerr"

    chi: Scalar = parameter(default=UNBOUND, unit="GHz", symbol=r"\chi")

    def interaction(self, a: EndpointOps, b: EndpointOps, p: Any) -> PhysicsExpr:
        """Full form ``χ · n̂_a n̂_b`` (diagonal; identical under RWA)."""
        return p.chi * (a.level * b.level)

    def parametric_interaction(self, a: EndpointOps, b: EndpointOps, p: Any) -> PhysicsExpr:
        """Modulable structure ``n̂_a n̂_b`` — δχ(t) pumps ride this."""
        _ = p
        return a.level * b.level

    def physics_notes(self) -> list[str]:
        """Return the declared dispersive-approximation provenance."""
        notes = super().physics_notes()
        notes.append(
            "Effective dispersive model: uniform pull χ per excitation pair; per-level χ "
            "differences, n_crit breakdown, and Purcell are not represented"
        )
        return notes

    def __repr__(self) -> str:
        """Return a compact endpoint/χ summary."""
        return f"CrossKerr('{self.device_a_label}' <-> '{self.device_b_label}', chi={self.chi})"


class Coupling(_BaseCoupling):
    """Generic two-body coupling with a user-supplied interaction.

    Use this escape hatch when no concrete coupling class models the
    desired physics (inductive, longitudinal, cross-Kerr test forms,
    synthetic spin-spin couplings, photonics-style beam-splitters, …).
    The user supplies the operator structure; this class only provides
    the ``g`` scaling, RWA pass-through, and bookkeeping.

    Two mutually exclusive modes:

    **Product form** — ``H_int = g · op_a(device_a) ⊗ op_b(device_b)``::

        Coupling(q, r, g=0.02,
            op_a=lambda d: d.number_operator(),
            op_b=lambda d: d.number_operator())

    **Callable form** — ``H_int = g · interaction(device_a, device_b, backend)``::

        Coupling(q, r, g=0.02,
            interaction=lambda a, b, bk: (
                bk.tensor(bk.dag(a.lowering_operator()), b.lowering_operator())
                + bk.tensor(a.lowering_operator(), bk.dag(b.lowering_operator()))
            ))

    The selected chip approximation is applied to the complete user-supplied
    operator after authored physics is assembled.
    """

    _type_prefix: ClassVar[str] = "coupling"

    def __init__(
        self,
        device_a: BaseDevice | str,
        device_b: BaseDevice | str,
        g: float,
        *,
        op_a: Callable[[BaseDevice], Operator] | None = None,
        op_b: Callable[[BaseDevice], Operator] | None = None,
        interaction: Callable[[BaseDevice, BaseDevice, "Backend"], Operator] | None = None,
        label: str | None = None,
    ) -> None:
        """Initialize a user-defined product-form or callable interaction."""
        super().__init__(device_a, device_b, label=label)

        has_product = op_a is not None or op_b is not None
        has_interaction = interaction is not None
        if has_product and has_interaction:
            raise ValueError("Provide either (op_a, op_b) or interaction, not both.")
        if not (has_product or has_interaction):
            raise ValueError("Provide either (op_a, op_b) or interaction.")
        if has_product and (op_a is None or op_b is None):
            raise ValueError("Both op_a and op_b are required for product-form coupling.")

        self.g = g
        self._op_a = op_a
        self._op_b = op_b
        self._interaction = interaction

    @property
    def coupling_strength(self) -> float:
        """Scalar prefactor ``g`` supplied by the user."""
        return self.g

    def interaction_hamiltonian(self) -> Operator:
        """User-defined interaction on ``H_a ⊗ H_b``, scaled by ``g``."""
        backend = _get_backend()
        # device_a/device_b are typed BaseDevice | str for late label binding, but
        # Chip._resolve_devices() resolves both to concrete BaseDevice instances
        # before interaction_hamiltonian() is ever invoked (see is_resolved).
        if self._interaction is not None:
            return self.g * self._interaction(cast(BaseDevice, self.device_a), cast(BaseDevice, self.device_b), backend)
        # op_a / op_b both guaranteed non-None by __init__ validation.
        return self.g * backend.tensor(
            self._op_a(cast(BaseDevice, self.device_a)),  # type: ignore[misc]
            self._op_b(cast(BaseDevice, self.device_b)),  # type: ignore[misc]
        )

    def __repr__(self) -> str:
        """Return a compact endpoint/mode summary."""
        mode = "interaction" if self._interaction is not None else "product"
        return f"Coupling('{self.device_a_label}' <-> '{self.device_b_label}', g={self.g}, mode={mode})"

    def physics_notes(self) -> list[str]:
        """Return notes describing the user-supplied interaction mode."""
        notes = super().physics_notes()
        mode = "user-supplied interaction" if self._interaction is not None else "product form op_a ⊗ op_b"
        notes.append(f"Interaction form: {mode}")
        return notes

    def to_dict(self) -> dict[str, Any]:
        """Reject serialization because callables cannot be made persistent."""
        raise NotImplementedError(
            "Generic Coupling carries user-defined callables and cannot be serialized. "
            "Use a concrete coupling subclass for persistent storage."
        )

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        device_a: "BaseDevice",
        device_b: "BaseDevice",
    ) -> "Coupling":
        """Reject deserialization because callables cannot be reconstructed."""
        raise NotImplementedError(
            "Generic Coupling carries user-defined callables and cannot be deserialized. "
            "Use a concrete coupling subclass for persistent storage."
        )
