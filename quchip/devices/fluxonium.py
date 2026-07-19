r"""Fluxonium — phase-basis fluxonium qubit model.

Hamiltonian (native basis):

.. math::

   H = 4 E_C \hat n^2 + \tfrac{1}{2} E_L (\hat\varphi + 2\pi \varphi_\mathrm{ext})^2
       - E_J \cos\hat\varphi

where :math:`\hat\varphi` is diagonal on a plane-wave phase grid and
:math:`\hat n = -i \partial_\varphi` via 2nd-order central finite
difference. Non-periodic phase basis because :math:`E_L` breaks the
compact-phase symmetry.

Convergence
-----------
The kinetic term uses 2nd-order central finite differences on the phase
grid, so the discretization error scales as ``1/num_basis**2``. As a
measured example (not a general guarantee): at ``E_J=4``, ``E_C=1``,
``E_L=0.9`` at the flux sweet spot (``phi_ext=0.5``), the default
``num_basis=400`` gives an eigenvalue error of about ``1.3e-2`` GHz
relative to a converged reference. Double ``num_basis`` until the
observable of interest stabilizes.

References
----------
* Manucharyan, Koch, Glazman & Devoret, *Fluxonium: single Cooper-pair
  circuit free of charge offsets*, Science **326**, 113 (2009).
* Nguyen et al., *Blueprint for a High-Performance Fluxonium Quantum
  Processor*, PRX Quantum **3**, 037001 (2022).
* Smith, Kou, Vool, Koch, Glazman & Devoret, *Superconducting circuit
  protected by two-Cooper-pair tunneling*, npj QI **6**, 8 (2020) —
  relaxation matrix-element conventions.

Example
-------
>>> from quchip.devices.fluxonium import Fluxonium
>>> q = Fluxonium(E_C=1.0, E_J=4.0, E_L=1.0, phi_ext=0.5, levels=5)
"""

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from quchip.backend.protocol import Operator
from quchip.devices.circuit import CircuitDevice, _check_positive_energy


class Fluxonium(CircuitDevice):
    r"""Fluxonium qubit on a non-periodic plane-wave phase basis.

    Parameters
    ----------
    E_C : float
        Charging energy in GHz. Positive. JAX-traceable.
    E_J : float
        Josephson energy in GHz. Positive. JAX-traceable.
    E_L : float
        Inductive energy in GHz. Positive. JAX-traceable.
    phi_ext : float, default ``0.0``
        External flux in units of :math:`\Phi_0` (``0.5`` = half-flux
        sweet spot). JAX-traceable.
    levels : int, default ``10``
        Truncated eigenbasis size.
    label : str or None
    num_basis : int, default ``400``
        Phase-basis grid points.
    phi_max : float, default ``5 * pi``
        Phase grid half-range; grid is ``[-phi_max, +phi_max)``.
    collapse_model, coupling_channel, collapse_rate_threshold : see
        :class:`CircuitDevice`. ``coupling_channel`` is required when
        ``collapse_model='fermi_golden'`` (the default) with ``T1`` set
        — pick ``'flux'`` at or near the flux sweet spot
        (``phi_ext = 0.5``), where relaxation is flux-noise-dominated,
        and ``'charge'`` for charge-operator-limited T1 regimes.
    **noise_kwargs
        Forwarded to :class:`BaseDevice`.

    Notes
    -----
    The T1 collapse-operator model depends on ``coupling_channel``:
    ``'charge'`` uses :math:`\hat n` matrix elements (Breuer-Petruccione
    §3.4, Smith 2020 §III.B); ``'flux'`` uses :math:`\hat\varphi` matrix
    elements (proportional to :math:`\partial H/\partial\varphi_\mathrm{ext}`
    since only the :math:`\hat\varphi` term is operator-valued there).
    Inherited pure dephasing uses level-index scaling — physically
    incomplete for fluxonium away from sweet spot, where
    flux-noise-weighted dephasing is the physical channel. Sweet-spot
    accurate dephasing is a follow-up PR.
    """

    _type_prefix: str = "fluxonium"
    tunable_param_names = ("E_C", "E_J", "E_L", "phi_ext")
    _ENERGY_PARAM_NAMES = ("E_C", "E_J", "E_L")
    approximation = (
        "Exact diagonalization on a finite phase grid; 2nd-order central finite "
        "differences for the kinetic term; accuracy governed by num_basis."
    )

    def __init__(
        self,
        E_C: float,
        E_J: float,
        E_L: float,
        phi_ext: float = 0.0,
        levels: int = 10,
        label: str | None = None,
        *,
        num_basis: int = 400,
        phi_max: float | None = None,
        collapse_model: Literal["fermi_golden", "ladder"] = "fermi_golden",
        coupling_channel: Literal["charge", "flux"] | None = None,
        collapse_rate_threshold: float = 1e-8,
        **noise_kwargs: float | None,
    ) -> None:
        _check_positive_energy("E_C", E_C)
        _check_positive_energy("E_J", E_J)
        _check_positive_energy("E_L", E_L)
        if num_basis < 3:
            raise ValueError(f"num_basis must be >= 3, got {num_basis}")
        if levels > num_basis:
            raise ValueError(f"levels ({levels}) cannot exceed num_basis ({num_basis})")
        phi_max_value = 5.0 * np.pi if phi_max is None else phi_max
        _check_positive_energy("phi_max", phi_max_value)

        super().__init__(
            levels=levels,
            label=label,
            collapse_model=collapse_model,
            coupling_channel=coupling_channel,
            collapse_rate_threshold=collapse_rate_threshold,
            **noise_kwargs,
        )
        self.E_C = E_C
        self.E_J = E_J
        self.E_L = E_L
        self.phi_ext = phi_ext
        self.num_basis = num_basis
        self.phi_max = phi_max_value

    def _validate_param_write(self, name: str, value: Any) -> None:
        """Extend :meth:`CircuitDevice._validate_param_write` with the ``phi_max`` positivity check."""
        super()._validate_param_write(name, value)
        if name == "phi_max":
            _check_positive_energy("phi_max", value)

    # ------------------------------------------------------------------
    # Native-basis construction (phase basis)
    # ------------------------------------------------------------------

    def _build_native_hamiltonian(self) -> Any:
        phi_grid = jnp.linspace(-self.phi_max, self.phi_max, self.num_basis, endpoint=False)
        d_phi = 2.0 * self.phi_max / self.num_basis

        # Kinetic: 4 E_C · n̂² where n̂ = -i ∂/∂φ, 2nd-order central FD.
        laplacian = (
            jnp.eye(self.num_basis, k=1)
            - 2.0 * jnp.eye(self.num_basis)
            + jnp.eye(self.num_basis, k=-1)
        ) / d_phi**2
        K = -4.0 * self.E_C * laplacian

        # Potential: diagonal in phase basis.
        V_diag = (
            0.5 * self.E_L * (phi_grid + 2.0 * jnp.pi * self.phi_ext) ** 2
            - self.E_J * jnp.cos(phi_grid)
        )

        return K.astype(jnp.complex128) + jnp.diag(V_diag.astype(jnp.complex128))

    def _native_charge_operator(self) -> Any:
        r"""n̂ = -i ∂/∂φ via 2nd-order central finite difference."""
        d_phi = 2.0 * self.phi_max / self.num_basis
        plus = jnp.eye(self.num_basis, k=1).astype(jnp.complex128)
        minus = jnp.eye(self.num_basis, k=-1).astype(jnp.complex128)
        return -1j * (plus - minus) / (2.0 * d_phi)

    def _native_phase_operator(self) -> Any:
        phi_grid = jnp.linspace(-self.phi_max, self.phi_max, self.num_basis, endpoint=False)
        return jnp.diag(phi_grid.astype(jnp.complex128))

    # ------------------------------------------------------------------
    # Protocol: FluxCoupled (flux couples through φ̂ on a fluxonium)
    # ------------------------------------------------------------------

    def flux_coupling_operator(self) -> Operator:
        r"""Return the flux-line coupling operator :math:`V^\dagger \hat\varphi V`."""
        return self.project_operator(self._native_phase_operator())

    def physics_notes(self) -> list[str]:
        """Return declared phase-basis discretization assumptions."""
        notes = super().physics_notes()
        notes.append(
            f"Phase basis: plane waves on φ ∈ [-{self.phi_max:.2f}, {self.phi_max:.2f}) "
            f"with {self.num_basis} grid points (non-periodic — E_L breaks compact-phase symmetry)"
        )
        return notes

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @property
    def computational(self) -> bool:
        """Fluxonium is a computational qubit."""
        return True

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Extend :meth:`CircuitDevice.to_dict` with the fluxonium circuit parameters."""
        data = super().to_dict()
        data["E_C"] = float(self.E_C)
        data["E_J"] = float(self.E_J)
        data["E_L"] = float(self.E_L)
        data["phi_ext"] = float(self.phi_ext)
        data["num_basis"] = int(self.num_basis)
        data["phi_max"] = float(self.phi_max)
        return data

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Fluxonium":
        return cls(
            E_C=float(d["E_C"]),
            E_J=float(d["E_J"]),
            E_L=float(d["E_L"]),
            phi_ext=float(d.get("phi_ext", 0.0)),
            levels=int(d.get("levels", 10)),
            label=d.get("label"),
            num_basis=int(d.get("num_basis", 400)),
            phi_max=float(d.get("phi_max", 5 * np.pi)),
            collapse_model=d.get("collapse_model", "fermi_golden"),
            coupling_channel=d.get("coupling_channel"),
            collapse_rate_threshold=float(d.get("collapse_rate_threshold", 1e-8)),
            **cls._noise_kwargs_from_dict(d),
        )._restore_reference_freq(d)
