r"""A frozen energy-basis device for third-party models without a native quchip model."""

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from quchip.declarative.expr import PhysicsExpr
from quchip.declarative.dissipation import CollapseChannel
from quchip.devices.base import (
    BaseDevice,
    _energy_basis_dissipation,
)


def _operator_matrix(name: str, value: Any, dimension: int) -> Any:
    matrix = jnp.asarray(value, dtype=jnp.complex128)
    if matrix.shape != (dimension, dimension):
        raise ValueError(
            f"{name} must have shape {(dimension, dimension)}, got {matrix.shape}."
        )
    return matrix


def _operator_to_json(value: Any | None) -> list[list[list[float]]] | None:
    if value is None:
        return None
    matrix = np.asarray(value)
    return [matrix.real.tolist(), matrix.imag.tolist()]


def _operator_from_json(value: list[list[list[float]]] | None) -> np.ndarray | None:
    if value is None:
        return None
    real, imag = value
    return np.asarray(real) + 1j * np.asarray(imag)


class EigenbasisDevice(BaseDevice):
    """Represent a frozen source spectrum and its energy-basis operators.

    Source circuit parameters cannot be differentiated through this snapshot.

    Parameters
    ----------
    energies : array_like, shape (n,)
        Energy-ordered levels in GHz, with ``n >= 2``. The first energy
        is subtracted from every level.
    charge_operator, phase_operator : array_like, shape (n, n), or None
        Operators in the same source eigenbasis. ``None`` leaves that drive
        channel unavailable; supplied matrix elements retain source conventions.
    levels : int or None, default None
        Number of retained energy levels. ``None`` retains all supplied levels.
    label : str or None, default None
        Device label; omission generates ``eigenbasis_<index>``.
    source_type : str or None, default None
        Provenance label, such as ``"scqubits.ZeroPi"``.
    collapse_model : {"fermi_golden", "ladder"}, default "fermi_golden"
        Relaxation from transition matrix elements or an ideal ladder operator.
    coupling_channel : {"charge", "flux"} or None, default None
        Operator for matrix-element relaxation; ``"flux"`` uses
        ``phase_operator``. Required when ``T1`` is set with ``"fermi_golden"``.
    collapse_rate_threshold : float, default 1e-8
        Nonnegative cutoff on normalized downward-transition strengths.
    **noise : scalar or None
        ``T1`` and ``T2`` in ns, and dimensionless ``thermal_occupation``.
        ``None`` leaves the corresponding channel absent. See
        :class:`~quchip.devices.base.BaseDevice` for noise constraints.

    References
    ----------
    Blais et al., Rev. Mod. Phys. 93, 025005 (2021),
    https://doi.org/10.1103/RevModPhys.93.025005, for energy-basis
    operators and coupling to dissipative environments.
    """

    _type_prefix = "eigenbasis"
    tunable_param_names = ()
    def __init__(
        self,
        energies: Any,
        *,
        charge_operator: Any | None = None,
        phase_operator: Any | None = None,
        levels: int | None = None,
        label: str | None = None,
        source_type: str | None = None,
        collapse_model: Literal["fermi_golden", "ladder"] = "fermi_golden",
        coupling_channel: Literal["charge", "flux"] | None = None,
        collapse_rate_threshold: float = 1e-8,
        **noise: Any,
    ) -> None:
        values = jnp.asarray(energies, dtype=jnp.float64)
        if values.ndim != 1 or values.shape[0] < 2:
            raise ValueError("energies must be a one-dimensional array with at least two values.")
        dimension = int(values.shape[0])
        retained = dimension if levels is None else levels
        self._validate_basis_request(
            basis="eigen",
            levels=retained,
            native_dimension=dimension,
        )
        if collapse_model not in ("fermi_golden", "ladder"):
            raise ValueError("collapse_model must be 'fermi_golden' or 'ladder'.")
        if coupling_channel not in (None, "charge", "flux"):
            raise ValueError("coupling_channel must be 'charge', 'flux', or None.")
        if collapse_model == "fermi_golden" and noise.get("T1") is not None and coupling_channel is None:
            raise ValueError("coupling_channel is required when T1 uses matrix-element relaxation.")
        if collapse_rate_threshold < 0:
            raise ValueError("collapse_rate_threshold must be non-negative.")

        self._energies = values - values[0]
        self._charge_operator = (
            None
            if charge_operator is None
            else _operator_matrix("charge_operator", charge_operator, dimension)
        )
        self._phase_operator = (
            None
            if phase_operator is None
            else _operator_matrix("phase_operator", phase_operator, dimension)
        )
        self.basis = "eigen"
        self.projection_levels = retained
        self.source_type = source_type
        self.collapse_model = collapse_model
        self.coupling_channel = coupling_channel
        self.collapse_rate_threshold = collapse_rate_threshold
        super().__init__(levels=dimension, label=label, **noise)

    def dissipation(self, op: Any, p: Any) -> tuple[CollapseChannel, ...]:
        """Return relaxation and dephasing channels for the stored spectrum.

        Parameters
        ----------
        op : LocalOps
            Local operator namespace; unused because stored matrices define the basis.
        p : parameter namespace
            Bound noise parameters for this calculation.

        Returns
        -------
        tuple of CollapseChannel
            Channels selected by ``collapse_model`` and the active noise parameters.

        See Also
        --------
        EigenbasisDevice : Dissipation choices and physics reference.
        """
        del op
        return _energy_basis_dissipation(self, p)

    def unresolved_hamiltonian(self) -> PhysicsExpr:
        """Return the frozen source spectrum as the authored Hamiltonian."""
        return PhysicsExpr.from_matrix(
            jnp.diag(self._energies.astype(jnp.complex128)),
            labels=(self.label,),
            dims=(self.levels,),
            name=rf"\hat H_{{{self.label}}}",
        )

    @property
    def freq(self) -> Any:
        """Return the stored zero-to-one transition in GHz."""
        return self._energies[1]

    def eigenenergies(self) -> Any:
        """Return the stored ground-shifted energy table."""
        return self._energies

    def eigenvectors(self) -> Any:
        """Return the identity map because the authored basis is energy ordered."""
        return jnp.eye(self.levels, dtype=jnp.complex128)

    def charge_coupling_operator(self) -> Any:
        """Return the supplied charge-like operator in the authored basis."""
        if self._charge_operator is None:
            raise ValueError("This imported model did not supply charge_operator.")
        return self._charge_operator

    def phase_coupling_operator(self) -> Any:
        """Return the supplied phase-like operator in the authored basis."""
        if self._phase_operator is None:
            raise ValueError("This imported model did not supply phase_operator.")
        return self._phase_operator

    def physics_notes(self) -> list[str]:
        notes = super().physics_notes()
        source = f" from {self.source_type}" if self.source_type else ""
        notes.append(
            f"Frozen energy-basis snapshot{source}; source-model parameters are not differentiable."
        )
        return notes

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update(
            energies=np.asarray(self._energies).tolist(),
            charge_operator=_operator_to_json(self._charge_operator),
            phase_operator=_operator_to_json(self._phase_operator),
            levels=self.projection_levels,
            source_type=self.source_type,
            collapse_model=self.collapse_model,
            coupling_channel=self.coupling_channel,
            collapse_rate_threshold=self.collapse_rate_threshold,
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EigenbasisDevice":
        """Restore a frozen device declaration.

        Parameters
        ----------
        data : dict
            Output of :meth:`to_dict`, including energies and optional operator
            matrices encoded as separate real and imaginary nested lists.

        Returns
        -------
        EigenbasisDevice
            Restored device with noise and reference-frequency metadata.
        """
        return cls(
            data["energies"],
            charge_operator=_operator_from_json(data.get("charge_operator")),
            phase_operator=_operator_from_json(data.get("phase_operator")),
            levels=data.get("levels"),
            label=data.get("label"),
            source_type=data.get("source_type"),
            collapse_model=data.get("collapse_model", "fermi_golden"),
            coupling_channel=data.get("coupling_channel"),
            collapse_rate_threshold=data.get("collapse_rate_threshold", 1e-8),
            **cls._noise_kwargs_from_dict(data),
        )._restore_reference_freq(data)
