"""Transmon authored in a finite integer-charge basis."""

from __future__ import annotations

import warnings
from operator import index
from typing import Any, ClassVar, Literal

from quchip.declarative.expr import PhysicsExpr
from quchip.declarative.models import DeviceModel
from quchip.declarative.ops import LocalOps
from quchip.declarative.parameters import UNBOUND, Scalar, parameter
from quchip.declarative.dissipation import CollapseChannel
from quchip.devices.base import _energy_basis_dissipation
from quchip.devices.spaces import ChargeSpace
from quchip.utils.jax_utils import maybe_concrete_scalar


class ChargeBasisTransmon(DeviceModel):
    r"""Transmon with its Hamiltonian authored in the integer-charge basis.

    The native Hamiltonian is :math:`4E_C(n-n_g)^2-E_J\cos\varphi` in
    ordinary GHz.

    Parameters
    ----------
    E_C, E_J : float
        Positive charging and Josephson energies in GHz.
    n_g : float, default 0.0
        Offset charge in Cooper-pair units.
    levels : int or None, default None
        Number of projected eigenstates. Required when ``basis="eigen"``;
        ``None`` uses the native dimension for the native basis.
    label : str or None, default None
        Device label.
    num_basis : int, keyword-only, default 61
        Odd integer-charge basis size, at least 3.
    basis : {None, "native", "eigen"}, keyword-only
        Requested solver basis. ``None`` inherits the chip policy and uses
        native basis for standalone resolution; ``native`` keeps the charge
        basis, while ``eigen`` projects onto ``levels`` energy states.
    collapse_model : {"fermi_golden", "ladder"}, default "fermi_golden"
        Relaxation construction; Fermi-golden-rule relaxation needs
        ``coupling_channel="charge"`` when ``T1`` is set.
    coupling_channel : {None, "charge"}, default None
        Physical operator for matrix-element relaxation. Only ``"charge"``
        is supported and selects the authored charge operator :math:`n`.
    collapse_rate_threshold : float, default 1e-8
        Non-negative dimensionless cutoff on squared matrix-element ratios
        relative to the selected ``0 -> 1`` transition.
    T1, T2 : float or None
        Relaxation and dephasing times in ns; ``None`` disables each channel.
    thermal_occupation : float or None
        Mean thermal occupation (dimensionless); ``None`` disables thermal
        absorption.
    noise : keyword arguments
        The concrete constructor accepts inherited noise fields as keyword
        arguments.

    References
    ----------
    See Koch et al., *Phys. Rev. A* 76, 042319 (2007),
    https://doi.org/10.1103/PhysRevA.76.042319.
    """

    _type_prefix = "charge_basis_transmon"
    tunable_param_names = ("E_C", "E_J", "n_g")
    approximation = (
        "Exact diagonalization in a finite integer-charge basis; "
        "accuracy is governed by num_basis."
    )
    computational = True
    requires_projection_levels: ClassVar[bool] = True
    structural_setting_names = (
        "num_basis",
        "basis",
        "projection_levels",
        "collapse_model",
        "coupling_channel",
        "collapse_rate_threshold",
    )
    E_C: Scalar = parameter(default=UNBOUND, positive=True, unit="GHz", symbol="E_C")
    E_J: Scalar = parameter(default=UNBOUND, positive=True, unit="GHz", symbol="E_J")
    n_g: Scalar = parameter(default=0.0, symbol="n_g")

    def dissipation(self, op: Any, p: Any) -> tuple[CollapseChannel, ...]:
        del op
        return _energy_basis_dissipation(self, p)

    def __init__(
        self,
        E_C: Scalar,
        E_J: Scalar,
        n_g: Scalar = 0.0,
        levels: int | None = None,
        label: str | None = None,
        *,
        num_basis: int = 61,
        basis: Literal["native", "eigen"] | None = None,
        collapse_model: Literal["fermi_golden", "ladder"] = "fermi_golden",
        coupling_channel: Literal["charge"] | None = None,
        collapse_rate_threshold: float = 1e-8,
        **noise: Any,
    ) -> None:
        if index(num_basis) < 3 or num_basis % 2 == 0:
            raise ValueError(f"num_basis must be an odd integer >= 3, got {num_basis}")
        self._validate_basis_request(
            basis=basis,
            levels=levels,
            native_dimension=num_basis,
        )
        if collapse_model not in ("fermi_golden", "ladder"):
            raise ValueError(
                f"collapse_model must be 'fermi_golden' or 'ladder', got {collapse_model!r}"
            )
        if coupling_channel not in (None, "charge"):
            raise ValueError("ChargeBasisTransmon only supports coupling_channel='charge'.")
        if collapse_model == "fermi_golden" and noise.get("T1") is not None and coupling_channel is None:
            raise ValueError("coupling_channel is required when T1 uses matrix-element relaxation.")
        if collapse_rate_threshold < 0:
            raise ValueError("collapse_rate_threshold must be non-negative.")

        self.num_basis = num_basis
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
            n_g=n_g,
            **noise,
        )

    def local_space(self) -> ChargeSpace:
        """Return the authored integer-charge space."""
        return ChargeSpace(self.num_basis)

    def local_hamiltonian(self, op: LocalOps, p: Any) -> PhysicsExpr:
        """Return the charge-basis transmon Hamiltonian.

        Parameters
        ----------
        op : LocalOps
            Charge-space operator namespace.
        p : ParameterNamespace
            Symbolic ``E_C``, ``E_J`` and ``n_g`` values.
        """
        shifted_charge = op.n - p.n_g * op.I
        return 4.0 * p.E_C * (shifted_charge @ shifted_charge) - p.E_J * op.cos_phi

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
        """Return the isolated energy-ordered eigenvectors in the charge basis."""
        return self._basis_record().energy_vectors

    def charge_coupling_operator(self) -> Any:
        """Return the authored charge operator."""
        return self.local_space().matrix("n")

    def phase_coupling_operator(self) -> Any:
        """Return sin(phi) in the authored charge basis."""
        return self.local_space().matrix("sin_phi")

    def tunable_param_bounds(self, name: str, value: float) -> tuple[float, float]:
        """Return physical bounds for a tunable parameter.

        Parameters
        ----------
        name : str
            Tunable parameter name.
        value : float
            Concrete optimizer seed.
        """
        if name == "n_g":
            return (-0.5, 0.5)
        return super().tunable_param_bounds(name, value)

    def _validate_param_write(self, name: str, value: Any) -> None:
        super()._validate_param_write(name, value)
        if name == "coupling_channel" and value not in (None, "charge"):
            raise ValueError("coupling_channel must be 'charge' or None.")
        if name == "num_basis":
            if index(value) < 3 or value % 2 == 0:
                raise ValueError(f"num_basis must be an odd integer >= 3, got {value}")
            if self.projection_levels is not None and self.projection_levels > value:
                raise ValueError(
                    f"num_basis cannot be smaller than levels ({self.projection_levels})."
                )

    def _truncation_note(self) -> str:
        policy = self.basis or "inherits chip (native when standalone)"
        return (
            f"Authored integer-charge basis: {self.num_basis} states; "
            f"solver basis: {policy}; retained levels: {self.projection_levels}"
        )

    def physics_notes(self) -> list[str]:
        notes = super().physics_notes()
        cutoff = (self.num_basis - 1) // 2
        notes.append(f"Integer charge basis: n in [-{cutoff}, +{cutoff}]")
        notes.append("The phase channel is sin(phi); phi itself is not single-valued in this basis")
        return notes

    @classmethod
    def from_frequency(
        cls,
        freq: float,
        anharmonicity: float,
        n_g: float = 0.0,
        levels: int | None = None,
        label: str | None = None,
        *,
        num_basis: int = 61,
        basis: Literal["native", "eigen"] | None = None,
        **kwargs: Any,
    ) -> "ChargeBasisTransmon":
        """Construct from the leading transmon-regime inversion.

        Parameters
        ----------
        freq, anharmonicity : float
            Calibrated transition and anharmonicity in GHz.
        n_g : float, default 0.0
            Offset charge in Cooper-pair units.
        levels : int or None, default None
            Projected eigenstate count.
        label : str or None, default None
            Device label.
        num_basis : int, keyword-only, default 61
            Odd charge-basis dimension.
        basis : {None, "native", "eigen"}, keyword-only
            Basis policy.
        **kwargs : Any
            Noise and other constructor keyword arguments.
        """
        E_C = -anharmonicity
        E_J = (freq + E_C) ** 2 / (8.0 * E_C)
        ratio = maybe_concrete_scalar(E_J / E_C)
        if ratio is not None and ratio < 20.0:
            warnings.warn(
                f"from_frequency gives E_J/E_C approximately {ratio:.1f}, below transmon "
                "regime; use explicit E_C and E_J when this approximation is unsuitable.",
                stacklevel=2,
            )
        return cls(
            E_C=E_C,
            E_J=E_J,
            n_g=n_g,
            levels=levels,
            label=label,
            num_basis=num_basis,
            basis=basis,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["num_basis"] = self.num_basis
        data["basis"] = self.basis
        data["levels"] = self.projection_levels
        data["collapse_model"] = self.collapse_model
        data["coupling_channel"] = self.coupling_channel
        data["collapse_rate_threshold"] = self.collapse_rate_threshold
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChargeBasisTransmon":
        """Reconstruct from serialized constructor data.

        Parameters
        ----------
        data : mapping
            Record produced by :meth:`to_dict`.
        """
        return cls(
            E_C=data["E_C"],
            E_J=data["E_J"],
            n_g=data.get("n_g", 0.0),
            levels=data.get("levels"),
            label=data.get("label"),
            num_basis=data.get("num_basis", 61),
            basis=data.get("basis"),
            collapse_model=data.get("collapse_model", "fermi_golden"),
            coupling_channel=data.get("coupling_channel"),
            collapse_rate_threshold=data.get("collapse_rate_threshold", 1e-8),
            **cls._noise_kwargs_from_dict(data),
        )._restore_reference_freq(data)
