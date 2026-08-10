"""Fluxonium authored on a finite phase grid."""

from __future__ import annotations

from math import pi
from typing import Any, ClassVar, Literal

from quchip.declarative.expr import PhysicsExpr
from quchip.declarative.models import DeviceModel
from quchip.declarative.ops import LocalOps
from quchip.declarative.parameters import Scalar, parameter
from quchip.devices.spaces import PhaseGridSpace


class Fluxonium(DeviceModel):
    """Fluxonium with its Hamiltonian authored in a finite phase-grid basis."""

    _type_prefix = "fluxonium"
    tunable_param_names = ("E_C", "E_J", "E_L", "phi_ext")
    approximation = (
        "Finite phase-grid model with a second-order charge kinetic operator; "
        "accuracy is governed by num_basis and phi_max."
    )
    computational = True
    requires_projection_levels: ClassVar[bool] = True
    structural_setting_names = (
        "num_basis",
        "phi_max",
        "basis",
        "projection_levels",
        "collapse_model",
        "coupling_channel",
        "collapse_rate_threshold",
    )

    E_C: Scalar = parameter(positive=True, unit="GHz", symbol="E_C")
    E_J: Scalar = parameter(positive=True, unit="GHz", symbol="E_J")
    E_L: Scalar = parameter(positive=True, unit="GHz", symbol="E_L")
    phi_ext: Scalar = parameter(default=0.0, symbol=r"\varphi_{\mathrm{ext}}")

    def __init__(
        self,
        E_C: Scalar,
        E_J: Scalar,
        E_L: Scalar,
        phi_ext: Scalar = 0.0,
        levels: int | None = None,
        label: str | None = None,
        *,
        num_basis: int = 400,
        phi_max: float = 5.0 * pi,
        basis: Literal["native", "eigen"] | None = None,
        collapse_model: Literal["fermi_golden", "ladder"] = "fermi_golden",
        coupling_channel: Literal["charge", "flux"] | None = None,
        collapse_rate_threshold: float = 1e-8,
        **noise: Any,
    ) -> None:
        if num_basis < 3:
            raise ValueError(f"num_basis must be >= 3, got {num_basis}")
        if phi_max <= 0:
            raise ValueError(f"phi_max must be positive, got {phi_max}")
        self._validate_basis_request(
            basis=basis,
            levels=levels,
            native_dimension=num_basis,
        )
        if collapse_model not in ("fermi_golden", "ladder"):
            raise ValueError(
                f"collapse_model must be 'fermi_golden' or 'ladder', got {collapse_model!r}"
            )
        if coupling_channel not in (None, "charge", "flux"):
            raise ValueError("coupling_channel must be 'charge', 'flux', or None.")
        if collapse_rate_threshold < 0:
            raise ValueError("collapse_rate_threshold must be non-negative.")

        self.num_basis = num_basis
        self.phi_max = phi_max
        self.basis = basis
        self.projection_levels = levels
        self.collapse_model = collapse_model
        self.coupling_channel = coupling_channel
        self.collapse_rate_threshold = collapse_rate_threshold
        super().__init__(
            levels=num_basis,
            label=label,
            E_C=E_C,
            E_J=E_J,
            E_L=E_L,
            phi_ext=phi_ext,
            **noise,
        )

    def local_space(self) -> PhaseGridSpace:
        """Return the authored finite phase-grid space."""
        return PhaseGridSpace(points=self.num_basis, extent=self.phi_max)

    def local_hamiltonian(self, op: LocalOps, p: Any) -> PhysicsExpr:
        """Return the native fluxonium Hamiltonian in ordinary GHz."""
        shifted_phase = op.phi + (2.0 * pi) * p.phi_ext * op.I
        return (
            4.0 * p.E_C * op.n2
            + 0.5 * p.E_L * (shifted_phase @ shifted_phase)
            - p.E_J * op.cos_phi
        )

    def _basis_record(self) -> Any:
        from quchip.engine.basis import resolve_device_basis

        return resolve_device_basis(
            self,
            basis="eigen",
            levels=self.projection_levels or min(10, self.num_basis),
        )

    @property
    def freq(self) -> Any:
        """Return the isolated 0-to-1 transition in GHz."""
        energies = self._basis_record().energies
        return energies[1] - energies[0]

    def eigenenergies(self) -> Any:
        """Return isolated energies shifted to zero at the ground state."""
        energies = self._basis_record().energies
        return energies - energies[0]

    def eigenvectors(self) -> Any:
        """Return the isolated energy-ordered phase-grid eigenvectors."""
        return self._basis_record().energy_vectors

    def charge_coupling_operator(self) -> Any:
        """Return the authored charge operator."""
        return self.local_space().matrix("n")

    def phase_coupling_operator(self) -> Any:
        """Return the authored phase operator."""
        return self.local_space().matrix("phi")

    def flux_coupling_operator(self) -> Any:
        """Return the authored flux-line phase operator."""
        return self.phase_coupling_operator()

    def _validate_param_write(self, name: str, value: Any) -> None:
        super()._validate_param_write(name, value)
        if name == "phi_max" and value <= 0:
            raise ValueError(f"phi_max must be positive, got {value}")
        if name == "coupling_channel" and value not in (None, "charge", "flux"):
            raise ValueError("coupling_channel must be 'charge', 'flux', or None.")
        if name == "num_basis":
            if value < 3:
                raise ValueError(f"num_basis must be >= 3, got {value}")
            if self.projection_levels is not None and self.projection_levels > value:
                raise ValueError(
                    f"num_basis cannot be smaller than levels ({self.projection_levels})."
                )

    def _truncation_note(self) -> str:
        policy = self.basis or "inherits chip (native when standalone)"
        return (
            f"Authored phase grid: {self.num_basis} points over "
            f"[-{self.phi_max:.3g}, {self.phi_max:.3g}); solver basis: {policy}; "
            f"retained levels: {self.projection_levels}"
        )

    def physics_notes(self) -> list[str]:
        notes = super().physics_notes()
        notes.append("Charge kinetic term uses a second-order centered finite difference")
        return notes

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["num_basis"] = self.num_basis
        data["phi_max"] = self.phi_max
        data["basis"] = self.basis
        data["levels"] = self.projection_levels
        data["collapse_model"] = self.collapse_model
        data["coupling_channel"] = self.coupling_channel
        data["collapse_rate_threshold"] = self.collapse_rate_threshold
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Fluxonium":
        return cls(
            E_C=data["E_C"],
            E_J=data["E_J"],
            E_L=data["E_L"],
            phi_ext=data.get("phi_ext", 0.0),
            levels=data.get("levels"),
            label=data.get("label"),
            num_basis=data.get("num_basis", 400),
            phi_max=data.get("phi_max", 5.0 * pi),
            basis=data.get("basis"),
            collapse_model=data.get("collapse_model", "fermi_golden"),
            coupling_channel=data.get("coupling_channel"),
            collapse_rate_threshold=data.get("collapse_rate_threshold", 1e-8),
            **cls._noise_kwargs_from_dict(data),
        )._restore_reference_freq(data)
